"""Reproduce every number, figure and the PDF report.

    python run_all.py            # problems 1-5, then the report
    python run_all.py --no-pdf   # problems only (no pandoc / LaTeX needed)
"""

import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "problems"))      # so the scripts can import common.py

for k in range(1, 6):
    print(f"\n========== Problem {k} ==========")
    runpy.run_path(str(ROOT / "problems" / f"problem{k}.py"), run_name="__main__")

if "--no-pdf" not in sys.argv:
    print("\n========== Report ==========")
    subprocess.run([sys.executable, str(ROOT / "report" / "build_report.py")], check=True)
