"""Create a group-safe, within-environment randomized 80/10/10 300-D split.

Every split receives data from every environment and every source kind.  This
is intentionally different from an environment-disjoint generalization test:
the objective here is a fair supervised approximation test with matched
control-state distributions.  Five samples belonging to a short rollout keep
the same split assignment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.mpc_complete_300d import INPUT_CONTRACT, INPUT_DIMENSIONS, INPUT_FEATURES, OUTPUT_CONTRACT


SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _allocate_groups(groups: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    """Exact 80/10/10 allocation of whole query groups."""
    values = np.asarray(groups, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    train_count = int(round(len(values) * RATIOS["train"]))
    validation_count = int(round(len(values) * RATIOS["validation"]))
    test_count = len(values) - train_count - validation_count
    if min(train_count, validation_count, test_count) < 0:
        raise RuntimeError("invalid group allocation")
    return {
        "train": values[:train_count],
        "validation": values[train_count:train_count + validation_count],
        "test": values[train_count + validation_count:],
    }


def prepare(source_path: Path, output: Path, *, seed: int) -> None:
    with np.load(source_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if data["X"].shape != (140000, 300) or data["U"].shape != (140000, 4):
        raise ValueError("expected the completed 140000 x 300 MPC-complete dataset")
    if not np.isfinite(data["X"]).all() or not np.isfinite(data["U"]).all():
        raise ValueError("source contains NaN or Inf")
    assignment = np.full(len(data["X"]), -1, dtype=np.int8)
    group_sets: dict[str, set[int]] = {name: set() for name in SPLITS}
    group_sizes: dict[tuple[int, int], int] = {}
    for environment in range(40):
        for source_kind in (0, 1, 2):
            indices = np.flatnonzero((data["environment_id"] == environment) & (data["source_kind"] == source_kind))
            groups = np.unique(data["query_group_id"][indices])
            expected_groups = {0: 1000, 1: 1500, 2: 200}[source_kind]
            if len(groups) != expected_groups:
                raise RuntimeError(f"environment={environment} source={source_kind} has {len(groups)} groups, expected {expected_groups}")
            allocation = _allocate_groups(groups, seed=seed + environment * 101 + source_kind * 100003)
            for split_index, name in enumerate(SPLITS):
                selected = allocation[name]
                mask = np.isin(data["query_group_id"], selected)
                picked = indices[mask[indices]]
                # Index through this source/environment's rows, so a same-ID
                # group could never be selected from another partition.
                assignment[picked] = split_index
                group_sets[name].update(int(item) for item in selected)
            group_sizes[(environment, source_kind)] = int(len(indices))
    if np.any(assignment < 0):
        raise RuntimeError("some rows were not assigned")
    if any(group_sets[left].intersection(group_sets[right]) for left in SPLITS for right in SPLITS if left < right):
        raise RuntimeError("a query group leaked across splits")

    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_dataset": str(source_path),
        "split_protocol": "within_each_environment_and_source_group_randomized_80_10_10",
        "not_environment_disjoint": True,
        "seed": seed,
        "input_dimensions": INPUT_DIMENSIONS,
        "input_features": list(INPUT_FEATURES),
        "input_contract": INPUT_CONTRACT,
        "output_contract": OUTPUT_CONTRACT,
        "teacher_policy": "reset_converged_acado_rti_min4_max16_delta1e-3_v1",
        "reference_speed_values": [10.0],
        "splits": {},
    }
    for split_index, name in enumerate(SPLITS):
        indices = np.flatnonzero(assignment == split_index)
        arrays = {key: value[indices] for key, value in data.items()}
        _atomic_npz(output / f"{name}.npz", arrays)
        env_counts = {str(env): int(np.count_nonzero(arrays["environment_id"] == env)) for env in range(40)}
        source_counts = {str(kind): int(np.count_nonzero(arrays["source_kind"] == kind)) for kind in (0, 1, 2)}
        bucket_counts = {str(kind): int(np.count_nonzero(arrays["control_bucket"] == kind)) for kind in (0, 1, 2)}
        expected_per_env = {"train": 2800, "validation": 350, "test": 350}[name]
        if set(env_counts.values()) != {expected_per_env}:
            raise RuntimeError(f"{name} does not contain the exact expected count from every environment")
        report["splits"][name] = {
            "samples": int(len(indices)), "environment_counts": env_counts, "source_counts": source_counts,
            "source_fraction": {key: value / len(indices) for key, value in source_counts.items()},
            "control_bucket_counts": bucket_counts,
            "control_bucket_fraction": {key: value / len(indices) for key, value in bucket_counts.items()},
            "query_groups": len(group_sets[name]),
        }

    with np.load(output / "train.npz", allow_pickle=False) as train:
        x = np.asarray(train["X"], dtype=np.float64)
        u = np.asarray(train["U"], dtype=np.float64)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    u_mean, u_std = u.mean(axis=0), u.std(axis=0)
    x_std = np.maximum(x_std, 1e-6)
    u_std = np.maximum(u_std, 1e-6)
    normalization = {
        "input_contract": "falcon_mpc_complete_solver_window_300d_v1", "output_contract": "falcon_acado_first_control_4d_v1",
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "u_mean": u_mean.tolist(), "u_std": u_std.tolist(),
        "fit_split": "train", "fit_samples": int(len(x)),
    }
    _atomic_json(output / "normalization.json", normalization)
    report["normalization"] = {"fit_split": "train", "fit_samples": int(len(x)), "constant_x_dimensions": int(np.count_nonzero(x.std(axis=0) < 1e-12)), "constant_u_dimensions": int(np.count_nonzero(u.std(axis=0) < 1e-12))}
    report["status"] = "valid"
    _atomic_json(output / "split_manifest.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(Path(__file__).with_name("teacher_dataset_300d_140k_v1") / "final" / "falcon_mpc_complete_300d_140k_v1.npz"))
    parser.add_argument("--output", default=str(Path(__file__).with_name("dnn_dataset_300d_140k_v1_within_environment")))
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    prepare(Path(args.source).resolve(), Path(args.output).resolve(), seed=args.seed)


if __name__ == "__main__":
    main()
