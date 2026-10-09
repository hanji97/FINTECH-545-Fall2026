"""Distribution fitting by maximum likelihood and information criteria (Weeks 1-2).

The Student t is parameterised by (nu, mu, s): s is a SCALE, not the standard
deviation; sd = s * sqrt(nu / (nu - 2)) (Week 1, Table 4).
"""

import numpy as np
from scipy import optimize, stats


def fit_normal(x):
    """MLE normal fit. Returns dict(mu, sigma, ll, k).

    The MLE sigma uses the 1/n denominator (Week 2, Section 4.2).
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sigma = np.sqrt(np.mean((x - mu) ** 2))
    ll = float(np.sum(stats.norm.logpdf(x, mu, sigma)))
    return {"mu": float(mu), "sigma": float(sigma), "ll": ll, "k": 2}


def fit_t(x, fix_mu=None):
    """MLE Student t fit. Returns dict(nu, mu, s, ll, k).

    Optimises over (log(nu - 2), mu, log s) so nu > 2 (finite variance) and
    s > 0 hold automatically. If ``fix_mu`` is given the location is held
    there and k drops by one.
    """
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)

    def unpack(p):
        nu = 2.0 + np.exp(p[0])
        if fix_mu is None:
            mu, s = p[1], np.exp(p[2])
        else:
            mu, s = fix_mu, np.exp(p[1])
        return nu, mu, s

    def nll(p):
        nu, mu, s = unpack(p)
        return -np.sum(stats.t.logpdf(x, nu, loc=mu, scale=s))

    best = None
    for nu0 in (3.0, 5.0, 10.0, 30.0):         # several starts: likelihood can be flat in nu
        s0 = sd * np.sqrt((nu0 - 2) / nu0)
        p0 = [np.log(nu0 - 2)] + ([np.median(x)] if fix_mu is None else []) + [np.log(s0)]
        res = optimize.minimize(nll, p0, method="Nelder-Mead",
                                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000})
        if best is None or res.fun < best.fun:
            best = res
    nu, mu, s = unpack(best.x)
    return {"nu": float(nu), "mu": float(mu), "s": float(s),
            "ll": float(-best.fun), "k": 3 if fix_mu is None else 2}


def aic(ll, k):
    return 2 * k - 2 * ll


def aicc(ll, k, n):
    """AICc = AIC + (2k^2 + 2k) / (n - k - 1)  (Week 2, Section 5.3)."""
    return aic(ll, k) + (2 * k * k + 2 * k) / (n - k - 1)


def bic(ll, k, n):
    return k * np.log(n) - 2 * ll
