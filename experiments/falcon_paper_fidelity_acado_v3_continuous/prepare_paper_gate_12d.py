"""Build the published FALCON 12-D interface from the immutable ACADO data.

The v3 collector stores enough raw state and environment metadata to derive
the paper input without re-running AirSim or the MPC teacher.  MPC labels are
copied bit-for-bit; only X is replaced by

    [p_gate - p, euler_gate - euler_drone, velocity, acceleration].

FALCON states an 80/10/10 random split but does not publish the split unit or
seed.  The default is a deterministic row split, which is the usual literal
interpretation.  ``--split-unit episode`` is available as a leakage-resistant
alternative, but must be reported as an implementation choice.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    HERE
    / "paper_core_40k_acado_v3_continuous"
    / "final"
    / "falcon_paper_core_40k_acado_v3_continuous.npz"
)
DEFAULT_ENVIRONMENTS = HERE / "paper_core_40k_acado_v3_continuous" / "environments.json"
DEFAULT_OUTPUT = HERE / "dnn_dataset_paper_gate_12d_euler"

INPUT_CONTRACT = "falcon_paper_gate_relative_euler_12d_v1"
INPUT_FEATURES = (
    "gate_position_error_x",
    "gate_position_error_y",
    "gate_position_error_z",
    "gate_euler_error_roll",
    "gate_euler_error_pitch",
    "gate_euler_error_yaw",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
)
OUTPUT_FEATURES = (
    "collective_thrust_acceleration",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
)


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _quaternion_to_euler(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized [w,x,y,z] quaternions to roll/pitch/yaw."""

    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternion array must have shape (N, 4)")
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("quaternion array contains a zero quaternion")
    q = q / norms
    w, x, y, z = q.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.column_stack((roll, pitch, yaw))


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _normalization(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
    x_mean = x.mean(axis=0, dtype=np.float64)
    x_std = x.std(axis=0, dtype=np.float64)
    u_mean = u.mean(axis=0, dtype=np.float64)
    u_std = u.std(axis=0, dtype=np.float64)
    x_std = np.maximum(x_std, 1e-8)
    u_std = np.maximum(u_std, 1e-8)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "u_mean": u_mean.astype(np.float32),
        "u_std": u_std.astype(np.float32),
    }


def _split_indices(
    episode_id: np.ndarray,
    *,
    seed: int,
    split_unit: str,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if split_unit == "row":
        shuffled = rng.permutation(len(episode_id))
        train_count = int(np.floor(0.8 * len(shuffled)))
        validation_count = int(np.floor(0.1 * len(shuffled)))
        return {
            "train": np.sort(shuffled[:train_count]),
            "validation": np.sort(shuffled[train_count : train_count + validation_count]),
            "test": np.sort(shuffled[train_count + validation_count :]),
        }
    episodes = np.unique(episode_id.astype(np.int64))
    shuffled_episodes = rng.permutation(episodes)
    train_count = int(np.floor(0.8 * len(shuffled_episodes)))
    validation_count = int(np.floor(0.1 * len(shuffled_episodes)))
    selected = {
        "train": shuffled_episodes[:train_count],
        "validation": shuffled_episodes[train_count : train_count + validation_count],
        "test": shuffled_episodes[train_count + validation_count :],
    }
    return {
        name: np.flatnonzero(np.isin(episode_id, values))
        for name, values in selected.items()
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    environments_path = args.environments.resolve()
    output = args.output_dir.resolve()
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "X",
        "U",
        "episode_id",
        "initial_state_id",
        "environment_id",
        "target_gate_id",
        "position",
        "quaternion",
        "velocity",
        "acceleration",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"source dataset is missing arrays: {missing}")
    rows = len(arrays["X"])
    episode_ids = np.unique(arrays["episode_id"].astype(np.int64))
    if rows != 40000 or episode_ids.size != 400:
        raise ValueError("FALCON paper core must contain 40,000 rows in 400 episodes")
    if arrays["U"].shape != (rows, 4):
        raise ValueError("source U must have shape (40000, 4)")
    for name, width in (
        ("position", 3),
        ("quaternion", 4),
        ("velocity", 3),
        ("acceleration", 3),
    ):
        if arrays[name].shape != (rows, width):
            raise ValueError(f"source {name} has unexpected shape {arrays[name].shape}")

    environments = json.loads(environments_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for environment in environments:
        environment_id = int(environment["environment_id"])
        for gate in environment["gates"]:
            lookup[(environment_id, int(gate["index"]))] = (
                np.asarray(gate["center"], dtype=np.float64),
                np.asarray(gate["orientation"], dtype=np.float64),
            )

    gate_positions = np.empty((rows, 3), dtype=np.float64)
    gate_quaternions = np.empty((rows, 4), dtype=np.float64)
    for index, key in enumerate(
        zip(
            arrays["environment_id"].astype(np.int64),
            arrays["target_gate_id"].astype(np.int64),
        )
    ):
        try:
            gate_positions[index], gate_quaternions[index] = lookup[key]
        except KeyError as exc:
            raise ValueError(f"no gate metadata for environment/gate {key}") from exc

    gate_euler = _quaternion_to_euler(gate_quaternions)
    drone_euler = _quaternion_to_euler(arrays["quaternion"])
    paper_x = np.concatenate(
        (
            gate_positions - arrays["position"],
            _wrap_angle(gate_euler - drone_euler),
            arrays["velocity"],
            arrays["acceleration"],
        ),
        axis=1,
    ).astype(np.float32)
    if paper_x.shape != (rows, 12) or not np.isfinite(paper_x).all():
        raise RuntimeError("derived paper input is not a finite (40000, 12) array")
    if not np.isfinite(arrays["U"]).all():
        raise ValueError("source control labels contain non-finite values")

    converted = dict(arrays)
    converted["X_source_trajectory_relative"] = np.asarray(arrays["X"], dtype=np.float32)
    converted["X"] = paper_x
    converted["gate_position"] = gate_positions.astype(np.float32)
    converted["gate_orientation"] = gate_quaternions.astype(np.float32)
    indices = _split_indices(
        arrays["episode_id"], seed=args.seed, split_unit=args.split_unit
    )
    membership = np.zeros(rows, dtype=np.int8)
    for selected in indices.values():
        membership[selected] += 1
    if not np.all(membership == 1):
        raise RuntimeError("split indices are not disjoint and exhaustive")

    output.mkdir(parents=True, exist_ok=True)
    aligned = {
        key: value
        for key, value in converted.items()
        if np.asarray(value).ndim > 0 and len(value) == rows
    }
    split_summary: dict[str, Any] = {}
    for name, selected in indices.items():
        _atomic_npz(output / f"{name}.npz", {key: value[selected] for key, value in aligned.items()})
        split_summary[name] = {
            "samples": int(len(selected)),
            "episodes_represented": int(np.unique(arrays["episode_id"][selected]).size),
            "initial_state_ids": sorted(
                np.unique(arrays["initial_state_id"][selected]).astype(int).tolist()
            ),
            "environment_ids": sorted(
                np.unique(arrays["environment_id"][selected]).astype(int).tolist()
            ),
        }

    train = indices["train"]
    normalization = _normalization(paper_x[train], arrays["U"][train])
    _write_json(
        output / "normalization.json",
        {key: value.tolist() for key, value in normalization.items()},
    )
    manifest = {
        "format_version": 1,
        "purpose": "FALCON Table-II-style six-model training",
        "source": str(source),
        "environments": str(environments_path),
        "teacher_policy": "global_continuous_quintic_plus_official_rpg_mpc_acado_qpoases",
        "source_input_contract": "falcon_track_p_trajectory_relative_12d_v1",
        "input_contract": INPUT_CONTRACT,
        "input_dimensions": 12,
        "input_features": list(INPUT_FEATURES),
        "attitude_representation": "wrapped XYZ Euler-angle difference in radians",
        "output_dimensions": 4,
        "output_features": list(OUTPUT_FEATURES),
        "label_preservation": "U copied unchanged from source ACADO/qpOASES dataset",
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_unit": args.split_unit,
        "split_seed": int(args.seed),
        "paper_ambiguity": (
            "FALCON publishes 80/10/10 but not the random seed or whether rows or "
            "whole trajectories are split"
        ),
        "reference_speed_values": sorted(
            float(value) for value in np.unique(arrays.get("reference_speed", np.empty(0)))
        ),
        "samples": rows,
        "episodes": int(episode_ids.size),
        "splits": split_summary,
        "x_min": paper_x.min(axis=0).astype(float).tolist(),
        "x_max": paper_x.max(axis=0).astype(float).tolist(),
        "x_std": paper_x.std(axis=0).astype(float).tolist(),
    }
    _write_json(output / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--environments", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--split-unit", choices=("row", "episode"), default="row")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
