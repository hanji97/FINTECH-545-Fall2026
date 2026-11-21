# Binomial trees. Week 06, section 6 (CRR tree), Week 07, sections 1.3 (American
# exercise) and 2.3 (non-recombining tree for discrete dividends).
#
# Node (i, j) is time step j with i up moves, price S u^i d^(j-i). From (i, j)
# the tree moves up to (i+1, j+1) with probability p and down to (i, j+1).

"""
    bt_american(call, S, X, T, r, b, σ, N; american=true)

Recombining CRR tree with cost of carry `b`:

    u = e^{σ√Δt},  d = 1/u,  p = (e^{bΔt} - d)/(u - d),  Δt = T/N

Backward induction; with `american=true` each node takes the larger of the
continuation value and the exercise payoff (Week 07, 1.3). With
`american=false` it prices the European option on the same grid, which is what
the early exercise premium is measured against.
"""
function bt_american(call::Bool, S, X, T, r, b, σ, N::Int; american::Bool=true)
    dt = T / N
    u = exp(σ * sqrt(dt))
    d = 1 / u
    p = (exp(b * dt) - d) / (u - d)
    df = exp(-r * dt)
    z = call ? 1.0 : -1.0

    # u^k for k = -N..N, so a node price is S * upow[2i - j + N + 1]
    upow = [u^k for k in -N:N]
    price(i, j) = S * upow[2i - j + N + 1]

    v = [max(0.0, z * (price(i, N) - X)) for i in 0:N]
    for j in (N-1):-1:0
        for i in 0:j
            # ascending i: v[i+2] still holds the step j+1 value when it is read
            cont = df * (p * v[i+2] + (1 - p) * v[i+1])
            v[i+1] = american ? max(cont, z * (price(i, j) - X)) : cont
        end
    end
    return v[1]
end

"""
    bt_american(call, S, X, T, r, divAmts, divTimes, σ, N; american=true)

Non-recombining tree for discrete cash dividends (Week 07, 2.3). `divTimes`
are integer step indices on the 0..N grid, so dividend k is paid at
`divTimes[k] * T/N`. The recursion is the one in the notes:

 1. With no dividend left inside the tree, the value is the recombining tree
    with b = r.
 2. Otherwise the nodes are built up to the first dividend step. At each node
    - hold:     a fresh tree from S - D with the remaining dividends;
    - exercise: the payoff on the cum-dividend price S (the call holder who
                exercises gets the stock and the dividend with it).
 3. Backward induction from there to the start.

The dividend replaces the carry, so the tree grows at r. With D dividends at
fixed fractions of T the cost grows like N^(D+2), N⁴ for the two dividends in
case 12.3; N = 500 runs in a few seconds.
"""
function bt_american(call::Bool, S, X, T, r, divAmts::Vector{Float64}, divTimes::Vector{Int},
                     σ, N::Int; american::Bool=true)
    if isempty(divAmts) || divTimes[1] > N
        return bt_american(call, S, X, T, r, r, σ, N; american=american)
    end

    dt = T / N
    u = exp(σ * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    df = exp(-r * dt)
    z = call ? 1.0 : -1.0
    k = divTimes[1]

    v = Vector{Float64}(undef, k + 1)
    for i in 0:k
        cum = S * u^i * d^(k - i)                           # price just before the dividend
        hold = bt_american(call, cum - divAmts[1], X, T - k * dt, r,
                           divAmts[2:end], divTimes[2:end] .- k, σ, N - k; american=american)
        v[i+1] = american ? max(hold, z * (cum - X)) : hold
    end
    for j in (k-1):-1:0
        for i in 0:j
            cont = df * (p * v[i+2] + (1 - p) * v[i+1])
            v[i+1] = american ? max(cont, z * (S * u^i * d^(j - i) - X)) : cont
        end
    end
    return v[1]
end

"""
    central_diff(f, x, h)

First and second central differences (Week 07, 3.3):

    f'(x)  ≈ (f(x+h) - f(x-h)) / 2h
    f''(x) ≈ (f(x+h) + f(x-h) - 2f(x)) / h²
"""
function central_diff(f, x, h)
    up, mid, dn = f(x + h), f(x), f(x - h)
    return (up - dn) / (2h), (up + dn - 2mid) / h^2
end

"""
    fd_step(x)

Central difference step ∛ε·max(|x|, 1). The central difference has
truncation error O(h²) and rounding error O(ε/h) from the subtraction in the
numerator (Week 07, 3.3); h ~ ∛ε balances the two.
"""
fd_step(x) = cbrt(eps(Float64)) * max(abs(x), 1.0)

"""
    fd_gradient(f, x)

Central difference gradient of a scalar function of a vector, one input at a
time with step `fd_step(xᵢ)`.
"""
function fd_gradient(f, x::AbstractVector)
    g = Vector{Float64}(undef, length(x))
    for i in eachindex(x)
        h = fd_step(x[i])
        up = collect(float.(x)); up[i] += h
        dn = collect(float.(x)); dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

"""
    bt_american_greeks(call, S, X, T, r, b, σ, N; dS=1.5)

Value and numerical sensitivities of the American option on the CRR tree
(Week 07, 3.3). The tree has no formula to differentiate, so

    grad    central differences of V(S, X, T, r, b, σ) in each input
    gamma   second central difference in S with step dS

The tree value is piecewise linear in S between nodes, so gamma needs a step
that spans several nodes rather than the tiny step used for the gradient.

Returns (value, grad, gamma), grad ordered (S, X, T, r, b, σ). grad[3] is
∂V/∂T, the negative of theta as defined in Week 07, 3.1, and grad[4] moves r
with the carry b held fixed.
"""
function bt_american_greeks(call::Bool, S, X, T, r, b, σ, N::Int; dS=1.5)
    V(p) = bt_american(call, p[1], p[2], p[3], p[4], p[5], p[6], N)
    p = [S, X, T, r, b, σ]
    v = V(p)
    grad = fd_gradient(V, p)
    _, gamma = central_diff(s -> V([s, X, T, r, b, σ]), S, dS)
    return v, grad, gamma
end
