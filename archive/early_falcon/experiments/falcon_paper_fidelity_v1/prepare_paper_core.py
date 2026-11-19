"""Prepare the immutable 40k paper core as whole-episode 80/10/10 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.dataset import prepare_episode_split  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=here / "paper_core_40k" / "final" / "falcon_paper_core_40k.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "dnn_dataset_paper_core_40k")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--allow-noncore", action="store_true")
    args = parser.parse_args()
    result = prepare_episode_split(
        args.dataset,
        args.output_dir,
        seed=args.seed,
        require_paper_core=not args.allow_noncore,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
