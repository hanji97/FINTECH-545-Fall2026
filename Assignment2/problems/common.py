"""Shared helpers for the problem scripts: paths, seeds, output writers."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # headless rendering
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

SEED = 545                      # one fixed seed so every simulated number reproduces
N_SIM = 100_000

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def save_json(name, obj):
    """Write results to output/<name>.json (the report reads these)."""
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=float)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
