"""Generate resumable independent recovery queries (not paper Table-II data)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.recovery_dataset import generate  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "recovery_extension_config.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    generate(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
