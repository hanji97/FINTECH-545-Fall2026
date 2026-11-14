# Binomial trees. Week 06 §6 (CRR tree), Week 07 §1.3 (American exercise) and
# §2.3 (non-recombining tree for discrete dividends).
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

 1. No dividend left inside the tree: use the recombining tree with b = r.
 2. Otherwise build the nodes up to the first dividend step. At each node
    - hold:     start a fresh tree from S - D with the remaining dividends;
    - exercise: the payoff on the cum-dividend price S (the call holder who
                exercises gets the stock and the dividend with it).
 3. Roll back to the start.

The dividend replaces the carry, so the tree grows at r. Cost grows like N³
with two dividends; N = 500 runs in a few seconds.
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
