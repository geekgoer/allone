"""Summarize v6 pilot or full capacity results."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERIC = (
    ROOT
    / "experiments"
    / "falcon_paper_fidelity_acado_v5_phase_consistent"
    / "summarize_paper40k_results.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    runs_root = HERE / ("runs_pilot" if args.mode == "pilot" else "runs_full")
    completed = subprocess.run(
        [sys.executable, str(GENERIC), "--runs-root", str(runs_root)],
        cwd=ROOT,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
