"""Build a deterministic row-wise 80/10/10 split from the FALCON 40k core.

This paper-facing capacity dataset excludes the phase-recovery extension,
which remains available as a separate robustness evaluation dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_dataset import (
    DEFAULT_CORE,
    DEFAULT_ENVIRONMENTS,
    _atomic_npz,
    _core_gate_input,
    _gate_lookup,
    _load,
    _write_json,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "dnn_dataset_falcon12_v5_paper40k_rowsplit"


def prepare(args: argparse.Namespace) -> dict[str, object]:
    core_path = args.core.resolve()
    core = _load(core_path)
    if core["X"].shape != (40000, 12) or core["U"].shape != (40000, 4):
        raise ValueError(
            f"paper core must contain X=(40000,12), U=(40000,4); "
            f"got X={core['X'].shape}, U={core['U'].shape}"
        )
    if not np.isfinite(core["U"]).all():
        raise ValueError("paper-core controls contain NaN or Inf")

    paper_x = _core_gate_input(core, _gate_lookup(args.environments.resolve()))
    rows = len(paper_x)
    arrays = {
        "X": paper_x.astype(np.float32),
        "U": core["U"].astype(np.float32),
        "sample_id": np.arange(rows, dtype=np.int64),
        "episode_id": core["episode_id"].astype(np.int64),
        "initial_state_id": core["initial_state_id"].astype(np.int64),
        "environment_id": core["environment_id"].astype(np.int64),
        "step_id": core["step_id"].astype(np.int64),
        "target_gate_id": core["target_gate_id"].astype(np.int64),
        "sample_family": np.full(rows, -1, dtype=np.int8),
        "data_source": np.zeros(rows, dtype=np.int8),
        "reference_speed": core["reference_speed"].astype(np.float32),
    }

    shuffled = np.random.default_rng(args.seed).permutation(rows)
    split_indices = {
        "train": shuffled[:32000],
        "validation": shuffled[32000:36000],
        "test": shuffled[36000:40000],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_summary: dict[str, object] = {}
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, indices in split_indices.items():
        selected = {key: value[indices] for key, value in arrays.items()}
        split_arrays[name] = selected
        _atomic_npz(output / f"{name}.npz", selected)
        split_summary[name] = {
            "samples": int(len(indices)),
            "episodes_represented": int(np.unique(selected["episode_id"]).size),
            "initial_states_represented": int(np.unique(selected["initial_state_id"]).size),
            "environments_represented": int(np.unique(selected["environment_id"]).size),
        }

    from prepare_dnn_dataset import normalization_from_train

    normalization = normalization_from_train(
        split_arrays["train"]["X"], split_arrays["train"]["U"]
    )
    _write_json(
        output / "normalization.json",
        {key: value.tolist() for key, value in normalization.items()},
    )
    manifest: dict[str, object] = {
        "format_version": 1,
        "purpose": "FALCON Table-II-style 40k capacity comparison",
        "paper_core_source": str(core_path),
        "teacher_policy": str(args.teacher_policy),
        "input_contract": "falcon_paper_gate_relative_euler_12d_v1",
        "input_dimensions": 12,
        "input_features": [
            "gate_position_error_x", "gate_position_error_y", "gate_position_error_z",
            "gate_euler_error_roll", "gate_euler_error_pitch", "gate_euler_error_yaw",
            "velocity_x", "velocity_y", "velocity_z",
            "acceleration_x", "acceleration_y", "acceleration_z",
        ],
        "attitude_representation": "wrapped XYZ Euler-angle difference in radians",
        "output_dimensions": 4,
        "output_features": [
            "collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z"
        ],
        "paper_core_samples": 40000,
        "robustness_extension_samples": 0,
        "total_samples": 40000,
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_unit": "individual_labeled_pair",
        "split_seed": int(args.seed),
        "temporal_neighbor_leakage_possible": True,
        "reference_speed_values": [5.0],
        "splits": split_summary,
        "disclosure": (
            "This paper-facing dataset uses only the published-size 40k continuous core. "
            "The paper specifies an 80/10/10 split but not its unit; this artifact uses "
            "a deterministic row-wise interpretation. Episode-disjoint and recovery "
            "results must be reported separately as stricter robustness evaluations."
        ),
    }
    _write_json(output / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--environments", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=62025)
    parser.add_argument(
        "--teacher-policy",
        default="global_continuous_acado_qpoases_v5_paper_core_only",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
