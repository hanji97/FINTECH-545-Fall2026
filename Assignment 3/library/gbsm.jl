# Generalized Black Scholes Merton, its Greeks, and implied volatility.
# Week 06 §4.3 and §7, Week 07 §3.1. The call signature matches the class
# library so the two can be swapped, but the body is written from the formulas
# in the notes.
#
#   b = rf       Black Scholes 1973, no dividend
#   b = rf - q   Merton 1973, continuous dividend yield q
#   b = 0        Black 1976, futures
#
# theta is -∂V/∂T per year (a long option has negative theta); vega and rho
# are per 1.00 of vol and of rate.

struct GBSM
    value
    delta
    gamma
    vega
    theta
    rho
end

"""
    gbsm(call, underlying, strike, ttm, rf, b, ivol; includeGreeks=false)

    d₁ = (ln(S/X) + (b + σ²/2)T) / (σ√T),   d₂ = d₁ - σ√T
    call = S e^{(b-r)T} Φ(d₁) - X e^{-rT} Φ(d₂)
    put  = X e^{-rT} Φ(-d₂) - S e^{(b-r)T} Φ(-d₁)

The put is the call with every sign flipped, so both are written with
ω = +1 (call) or -1 (put):

    V = ω [S e^{(b-r)T} Φ(ω d₁) - X e^{-rT} Φ(ω d₂)]

and the Greeks in Week 07, Table 1 follow from the same ω. Delta is always
returned; gamma, vega, theta and rho are filled in when `includeGreeks=true`
and are 0 otherwise.
"""
function gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol; includeGreeks=false)
    ω = call ? 1.0 : -1.0
    sqT = sqrt(ttm)
    d1 = (log(underlying / strike) + (b + 0.5 * ivol^2) * ttm) / (ivol * sqT)
    d2 = d1 - ivol * sqT
    Sfwd = underlying * exp((b - rf) * ttm)      # S e^{(b-r)T}, i.e. S e^{-qT}
    Xpv = strike * exp(-rf * ttm)                # X e^{-rT}
    Φ(x) = cdf(Normal(), x)

    value = ω * (Sfwd * Φ(ω * d1) - Xpv * Φ(ω * d2))
    delta = ω * exp((b - rf) * ttm) * Φ(ω * d1)
    includeGreeks || return GBSM(value, delta, 0.0, 0.0, 0.0, 0.0)

    φd1 = pdf(Normal(), d1)
    gamma = exp((b - rf) * ttm) * φd1 / (underlying * ivol * sqT)    # same for call and put
    vega = Sfwd * φd1 * sqT                                           # same for call and put
    # theta = -∂V/∂T: time decay of the volatility term, the carry on the
    # stock leg and the interest on the strike leg
    theta = -Sfwd * φd1 * ivol / (2sqT) - ω * (b - rf) * Sfwd * Φ(ω * d1) - ω * rf * Xpv * Φ(ω * d2)
    rho = ω * ttm * Xpv * Φ(ω * d2)
    return GBSM(value, delta, gamma, vega, theta, rho)
end

"""
    price_bounds(call, underlying, strike, ttm, rf, b)

GBSM value as σ → 0 and as σ → ∞. A quote outside these has no implied
volatility. The lower bound is the discounted forward intrinsic value,
`max(0, S e^{(b-r)T} - X e^{-rT})` for a call, which sits above the plain
intrinsic value `S - X` because of the interest on the strike.
"""
function price_bounds(call::Bool, underlying, strike, ttm, rf, b)
    fwd = underlying * exp((b - rf) * ttm)       # prepaid forward, S e^{-qT}
    pvx = strike * exp(-rf * ttm)
    return call ? (max(0.0, fwd - pvx), fwd) : (max(0.0, pvx - fwd), pvx)
end

"""
    implied_vol(call, price, underlying, strike, ttm, rf, b)

Invert GBSM for σ with a root finder (Week 06, 7). The value is strictly
increasing in σ, so there is exactly one root whenever the price is strictly
inside `price_bounds`. The search starts on [1e-6, 5] and doubles the upper end
until the bracket changes sign (up to σ = 100). Returns `(σ, reason)`; σ is
NaN, with the reason, when there is no root.
"""
function implied_vol(call::Bool, price, underlying, strike, ttm, rf, b; lo=1e-6, hi=5.0)
    lower, upper = price_bounds(call, underlying, strike, ttm, rf, b)
    price < lower && return (NaN, @sprintf("price %.2f is below the lower bound %.4f", price, lower))
    price == lower && return (NaN, "price equals the zero-volatility bound")
    price >= upper && return (NaN, @sprintf("price %.2f is at or above the upper bound %.4f", price, upper))
    f(σ) = gbsm(call, underlying, strike, ttm, rf, b, σ).value - price
    f(lo) > 0 && return (NaN, "price is indistinguishable from the lower bound")
    while f(hi) < 0
        hi *= 2
        hi > 100 && return (NaN, "no root below 10,000% volatility")
    end
    return (find_zero(f, (lo, hi), Bisection(); xatol=1e-14), "")
end
