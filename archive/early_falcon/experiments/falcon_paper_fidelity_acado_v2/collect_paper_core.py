"""Collect the corrected isolated FALCON 40k core with ACADO/qpOASES."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.collector import collect  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "paper_core_config.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    collect(
        args.config,
        output_root=args.output_root,
        resume=args.resume,
        smoke_test=args.smoke_test,
        auto_launch=args.auto_launch,
        render=args.render,
    )


if __name__ == "__main__":
    main()
