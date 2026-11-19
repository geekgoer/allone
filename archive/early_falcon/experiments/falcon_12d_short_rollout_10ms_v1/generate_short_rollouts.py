"""Generate resumable five-step 10 m/s ACADO short-rollout teacher data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.short_rollout_dataset import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("short_rollout_120k_config.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    generate(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
