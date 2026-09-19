"""scipy's NIG maximum likelihood fit, used to check the Julia one.

    python nig_scipy_reference.py <input.csv> [column]

Reads one column of numbers, fits a Normal Inverse Gaussian, writes the four
parameters to stdout as JSON.

scipy carries the NIG as (a, b, loc, scale) with a = alpha*delta and
b = beta*delta, so converting back to the course convention:

    delta = scale;  alpha = a / delta;  beta = b / delta;  mu = loc

The log likelihood is reported too, since that is what decides which of two
optimisers got closer to the maximum.
"""

import json
import sys

import numpy as np
import scipy
from scipy import stats


def fit(x):
    a, b, loc, scale = stats.norminvgauss.fit(x)
    delta = scale
    return {
        "mu": float(loc),
        "alpha": float(a / delta),
        "beta": float(b / delta),
        "delta": float(delta),
        "loglikelihood": float(
            np.sum(stats.norminvgauss.logpdf(x, a, b, loc=loc, scale=scale))
        ),
        "scipy_version": scipy.__version__,
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    path = argv[1]
    column = argv[2] if len(argv) > 2 else None

    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    if column is None:
        column = names[0]
    elif column not in names:
        print(f"column {column!r} not in {names}", file=sys.stderr)
        return 2

    x = np.asarray(data[column], dtype=float)
    json.dump(fit(x), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
