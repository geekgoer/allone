"""Build fixed episode-disjoint FALCON-12 splits from core plus recovery data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState  # noqa: E402
from falcon_paper_fidelity.contracts import (  # noqa: E402
    OUTPUT_FEATURES,
    PAPER_GATE_EULER_INPUT_CONTRACT,
    PAPER_GATE_EULER_INPUT_FEATURES,
    build_paper_gate_euler_input,
)
from prepare_dnn_dataset import normalization_from_train  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_CORE = HERE / "paper_core_40k_falcon12_v5_stable" / "final" / "falcon_paper_core_40k_falcon12_v5_stable.npz"
DEFAULT_ENVIRONMENTS = HERE / "paper_core_40k_falcon12_v5_stable" / "environments.json"
DEFAULT_RECOVERY_ROOT = HERE / "phase_recovery_80k_falcon12_v5_stable_v2"
DEFAULT_OUTPUT = HERE / "dnn_dataset_falcon12_v5_stable"


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _gate_lookup(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    environments = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(environment["environment_id"]), int(gate["index"])): (
            np.asarray(gate["center"], dtype=np.float64),
            np.asarray(gate["orientation"], dtype=np.float64),
        )
        for environment in environments
        for gate in environment["gates"]
    }


def _core_gate_input(core: Mapping[str, np.ndarray], lookup: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rows = len(core["U"])
    result = np.empty((rows, 12), dtype=np.float32)
    for index in range(rows):
        key = (int(core["environment_id"][index]), int(core["target_gate_id"][index]))
        gate_position, gate_orientation = lookup[key]
        state = RawDroneState(
            position=core["position"][index],
            quaternion=core["quaternion"][index],
            velocity=core["velocity"][index],
            acceleration=core["acceleration"][index],
            angular_velocity=np.zeros(3),
        )
        result[index] = build_paper_gate_euler_input(
            state, gate_position, gate_orientation
        )
    return result


def _episode_split(episodes: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    unique = np.unique(episodes.astype(np.int64))
    if unique.size != 400:
        raise ValueError(f"expected 400 core episodes, found {unique.size}")
    shuffled = np.random.default_rng(seed).permutation(unique)
    return {
        "train": np.sort(shuffled[:320]),
        "validation": np.sort(shuffled[320:360]),
        "test": np.sort(shuffled[360:400]),
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    core_path = args.core.resolve()
    environments_path = args.environments.resolve()
    recovery_root = args.recovery_root.resolve()
    recovery_manifest_path = recovery_root / "manifest.json"
    recovery_manifest = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    if recovery_manifest.get("status") != "complete":
        raise ValueError("phase-recovery dataset is not complete")
    recovery_path = recovery_root / str(recovery_manifest["final_dataset"])
    core = _load(core_path)
    recovery = _load(recovery_path)
    if core["X"].shape != (40000, 12) or core["U"].shape != (40000, 4):
        raise ValueError("core must contain the published 40,000 pairs")
    if recovery["X"].ndim != 2 or recovery["X"].shape[1] != 12 or recovery["U"].shape[1] != 4:
        raise ValueError("recovery data must contain X=(N,12), U=(N,4)")
    if not np.isfinite(core["U"]).all() or not np.isfinite(recovery["U"]).all():
        raise ValueError("teacher controls contain NaN or Inf")
    core_x = _core_gate_input(core, _gate_lookup(environments_path))
    episode_sets = _episode_split(core["episode_id"], args.seed)

    core_rows = len(core_x)
    recovery_rows = len(recovery["X"])
    core_arrays = {
        "X": core_x,
        "U": core["U"].astype(np.float32),
        "sample_id": np.arange(core_rows, dtype=np.int64),
        "episode_id": core["episode_id"].astype(np.int64),
        "initial_state_id": core["initial_state_id"].astype(np.int64),
        "environment_id": core["environment_id"].astype(np.int64),
        "step_id": core["step_id"].astype(np.int64),
        "target_gate_id": core["target_gate_id"].astype(np.int64),
        "sample_family": np.full(core_rows, -1, dtype=np.int8),
        "data_source": np.zeros(core_rows, dtype=np.int8),
        "reference_speed": core["reference_speed"].astype(np.float32),
    }
    recovery_arrays = {
        "X": recovery["X"].astype(np.float32),
        "U": recovery["U"].astype(np.float32),
        "sample_id": core_rows + recovery["sample_id"].astype(np.int64),
        "episode_id": recovery["base_episode_id"].astype(np.int64),
        "initial_state_id": recovery["initial_state_id"].astype(np.int64),
        "environment_id": recovery["environment_id"].astype(np.int64),
        "step_id": recovery["base_step_id"].astype(np.int64),
        "target_gate_id": recovery["target_gate_id"].astype(np.int64),
        "sample_family": recovery["sample_family"].astype(np.int8),
        "data_source": np.ones(recovery_rows, dtype=np.int8),
        "reference_speed": np.full(recovery_rows, 5.0, dtype=np.float32),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, object] = {}
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    for offset, (name, selected_episodes) in enumerate(episode_sets.items()):
        core_mask = np.isin(core_arrays["episode_id"], selected_episodes)
        recovery_mask = np.isin(recovery_arrays["episode_id"], selected_episodes)
        combined = {
            key: np.concatenate((core_arrays[key][core_mask], recovery_arrays[key][recovery_mask]))
            for key in core_arrays
        }
        order = np.random.default_rng(args.seed + 1009 * (offset + 1)).permutation(len(combined["X"]))
        combined = {key: value[order] for key, value in combined.items()}
        split_arrays[name] = combined
        _atomic_npz(output / f"{name}.npz", combined)
        split_summaries[name] = {
            "samples": int(len(order)),
            "episodes": int(len(selected_episodes)),
            "episode_ids": selected_episodes.astype(int).tolist(),
            "paper_core_samples": int(np.count_nonzero(combined["data_source"] == 0)),
            "phase_recovery_samples": int(np.count_nonzero(combined["data_source"] == 1)),
        }
    normalization = normalization_from_train(
        split_arrays["train"]["X"], split_arrays["train"]["U"]
    )
    _write_json(
        output / "normalization.json",
        {key: value.tolist() for key, value in normalization.items()},
    )
    manifest: dict[str, object] = {
        "format_version": 1,
        "purpose": "FALCON 12-D closed-loop-capable capacity comparison",
        "paper_core_source": str(core_path),
        "phase_recovery_source": str(recovery_path),
        "teacher_policy": "global_continuous_nominal_start_plus_phase_consistent_acado_qpoases_v5",
        "input_contract": PAPER_GATE_EULER_INPUT_CONTRACT,
        "input_dimensions": 12,
        "input_features": list(PAPER_GATE_EULER_INPUT_FEATURES),
        "attitude_representation": "wrapped XYZ Euler-angle difference in radians",
        "output_dimensions": 4,
        "output_features": list(OUTPUT_FEATURES),
        "paper_core_samples": core_rows,
        "robustness_extension_samples": recovery_rows,
        "total_samples": core_rows + recovery_rows,
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_unit": "whole_source_episode_shared_by_core_and_recovery",
        "split_seed": int(args.seed),
        "temporal_neighbor_leakage": False,
        "reference_speed_values": [5.0],
        "data_source_values": {"0": "published-style continuous core", "1": "phase-consistent recovery extension"},
        "splits": split_summaries,
        "disclosure": (
            "The 40k core follows the published count/protocol. The additional recovery "
            "rows are an engineering robustness extension and are reported separately."
        ),
    }
    _write_json(output / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--environments", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=62025)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
