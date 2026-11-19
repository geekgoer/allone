"""Build the v6 paper-facing 40k row-wise 80/10/10 split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "experiments" / "falcon_paper_fidelity_acado_v5_phase_consistent"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from prepare_paper40k_dataset import prepare  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "paper_core_40k_falcon12_v6_diverse"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core",
        type=Path,
        default=DEFAULT_ROOT / "final" / "falcon_paper_core_40k_falcon12_v6_diverse.npz",
    )
    parser.add_argument(
        "--environments", type=Path, default=DEFAULT_ROOT / "environments.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "dnn_dataset_falcon12_v6_diverse40k"
    )
    parser.add_argument("--seed", type=int, default=62026)
    args = parser.parse_args()
    args.teacher_policy = (
        "global_continuous_measured_start_diverse_acado_qpoases_v6_paper_core_only"
    )
    prepare(args)


if __name__ == "__main__":
    main()
