using CSV, DataFrames, Statistics, StatsBase, Distributions, LinearAlgebra
using Optim, ForwardDiff, SpecialFunctions, Printf, Plots, Random

gr()
default(size=(900, 560), dpi=150, legend=:topleft, grid=true, framestyle=:box)

const FIGS = joinpath(@__DIR__, "figures")
mkpath(FIGS)

function datapath(k)
    f = "problem$(k).csv"
    for d in (joinpath(@__DIR__, "data"), @__DIR__, dirname(@__DIR__))
        isfile(joinpath(d, f)) && return joinpath(d, f)
    end
    error("$f not found. Put the assignment CSV files in this folder or in a data subfolder.")
end

readproblem(k) = CSV.read(datapath(k), DataFrame)

function fourmoments(x)
    n = length(x)
    m = mean(x)
    v = sum((x .- m) .^ 2) / (n - 1)
    s = sqrt(v)
    b1 = sum((x .- m) .^ 3) / n / (sum((x .- m) .^ 2) / n)^1.5
    b2 = sum((x .- m) .^ 4) / n / (sum((x .- m) .^ 2) / n)^2 - 3
    g1 = b1 * sqrt(n * (n - 1)) / (n - 2)
    g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * b2 + 6)
    (n=n, mean=m, var=v, sd=s, skew_biased=b1, skew=g1, exkurt_biased=b2, exkurt=g2)
end

aicc(ll, k, n) = 2k - 2ll + (2k^2 + 2k) / (n - k - 1)
bic(ll, k, n) = k * log(n) - 2ll

function olsfit(X, y)
    n, p = size(X)
    XtXi = inv(X'X)
    b = XtXi * (X'y)
    e = y - X * b
    s2 = dot(e, e) / (n - p)
    se = sqrt.(s2 * diag(XtXi))
    sst = sum((y .- mean(y)) .^ 2)
    sse = dot(e, e)
    r2 = 1 - sse / sst
    (beta=b, se=se, resid=e, s2=s2, s=sqrt(s2), r2=r2,
     adjr2=1 - (1 - r2) * (n - 1) / (n - p), tstat=b ./ se,
     smallest_eig=minimum(eigvals(X'X)))
end

normal_ll(e, sigma) = -length(e) / 2 * log(2pi * sigma^2) - sum(abs2, e) / (2 * sigma^2)

function t_ll(e, sigma, nu)
    c = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(pi * nu) - log(sigma)
    length(e) * c - (nu + 1) / 2 * sum(log1p.((e ./ sigma) .^ 2 ./ nu))
end

function durbinwatson(e)
    sum(diff(e) .^ 2) / sum(abs2, e)
end

function pacf_to_ar(psi)
    p = length(psi)
    phi = collect(psi[1:1])
    for k in 2:p
        prev = copy(phi)
        phi = [prev[j] - psi[k] * prev[k-j] for j in 1:k-1]
        push!(phi, psi[k])
    end
    phi
end

function ar_autocov(phi, s2, maxlag)
    p = length(phi)
    T = promote_type(eltype(phi), typeof(s2))
    A = zeros(T, p + 1, p + 1)
    for k in 0:p
        A[k+1, k+1] += one(T)
        for i in 1:p
            A[k+1, abs(k - i)+1] -= phi[i]
        end
    end
    b = zeros(T, p + 1)
    b[1] = s2
    g = A \ b
    gam = zeros(T, maxlag + 1)
    gam[1:min(p + 1, maxlag + 1)] = g[1:min(p + 1, maxlag + 1)]
    for k in (p+1):maxlag
        gam[k+1] = sum(phi[i] * gam[k-i+1] for i in 1:p)
    end
    gam
end

function ma_autocov(theta, s2, maxlag)
    q = length(theta)
    T = promote_type(eltype(theta), typeof(s2))
    th = vcat(one(T), collect(theta))
    gam = zeros(T, maxlag + 1)
    for k in 0:min(q, maxlag)
        gam[k+1] = s2 * sum(th[i+1] * th[i+k+1] for i in 0:(q-k))
    end
    gam
end

function gaussian_ll(r, gam)
    n = length(r)
    G = [gam[abs(i - j)+1] for i in 1:n, j in 1:n]
    C = cholesky(Symmetric(G); check=false)
    issuccess(C) || return -Inf
    z = C.L \ r
    -0.5 * (n * log(2pi) + 2 * sum(log, diag(C.U)) + dot(z, z))
end

function fit_arma(x, p, q; restarts=6, seed=20260905)
    n = length(x)
    npar = 2 + p + q
    function negll(v)
        mu = v[1]
        s = exp(v[2])
        phi = p > 0 ? pacf_to_ar(tanh.(v[3:2+p])) : Float64[]
        theta = q > 0 ? -pacf_to_ar(tanh.(v[3+p:2+p+q])) : Float64[]
        gam = p > 0 ? ar_autocov(phi, s^2, n - 1) : ma_autocov(theta, s^2, n - 1)
        ll = gaussian_ll(x .- mu, gam)
        isfinite(ll) ? -ll : 1e10
    end
    rng = MersenneTwister(seed)
    best = nothing
    for r in 1:restarts
        v0 = vcat(mean(x), log(std(x)), r == 1 ? zeros(p + q) : 0.6 .* randn(rng, p + q))
        res = optimize(negll, v0, NelderMead(),
                       Optim.Options(iterations=20000, g_tol=1e-10))
        res = optimize(negll, Optim.minimizer(res), NelderMead(),
                       Optim.Options(iterations=20000, g_tol=1e-10))
        if best === nothing || Optim.minimum(res) < Optim.minimum(best)
            best = res
        end
    end
    v = Optim.minimizer(best)
    ll = -Optim.minimum(best)
    (mu=v[1], sigma=exp(v[2]),
     phi=p > 0 ? pacf_to_ar(tanh.(v[3:2+p])) : Float64[],
     theta=q > 0 ? -pacf_to_ar(tanh.(v[3+p:2+p+q])) : Float64[],
     ll=ll, k=npar, aicc=aicc(ll, npar, n), aic=2npar - 2ll, bic=bic(ll, npar, n), n=n)
end
