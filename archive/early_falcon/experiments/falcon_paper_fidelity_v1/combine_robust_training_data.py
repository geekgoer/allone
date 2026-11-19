"""Combine paper-core splits with the separately labelled recovery extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.combined_dataset import combine  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-splits", type=Path, default=here / "dnn_dataset_paper_core_40k")
    parser.add_argument(
        "--recovery-dataset",
        type=Path,
        default=here / "recovery_extension_60k" / "final" / "falcon_recovery_extension_60k.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "dnn_dataset_robust_100k")
    parser.add_argument("--seed", type=int, default=12025)
    args = parser.parse_args()
    print(json.dumps(combine(args.paper_splits, args.recovery_dataset, args.output_dir, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
