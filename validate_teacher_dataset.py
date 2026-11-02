"""Validate a completed FALCON MPC teacher dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_store import is_valid_episode, load_episode, validate_merged_dataset


def validate_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json was not found in {dataset_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"dataset manifest is not completed: {manifest.get('status')!r}")
    episode_count = int(manifest["episode_count"])
    steps = int(manifest["steps_per_episode"])
    states = int(manifest["initial_state_count"])
    environments = int(manifest["environment_count"])
    for episode_id in range(episode_count):
        key = f"episode_{episode_id:04d}"
        entry = manifest["episodes"][key]
        if entry.get("status") != "completed":
            raise ValueError(f"{key} is not completed")
        episode_path = dataset_root / Path(entry["file"])
        if not is_valid_episode(
            episode_path,
            expected_steps=steps,
            expected_episode_id=episode_id,
        ):
            raise ValueError(f"{key} is missing or invalid: {episode_path}")
    final_name = manifest.get("final_dataset")
    final_path = dataset_root / Path(final_name) if final_name else dataset_root / "final" / "teacher_dataset.npz"
    stats = validate_merged_dataset(
        final_path,
        expected_samples=episode_count * steps,
        expected_episodes=episode_count,
        expected_steps=steps,
        expected_initial_states=states,
        expected_environments=environments,
    )
    arrays = load_episode(final_path)
    config_path = dataset_root / "collection_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    max_thrust = float(config.get("mpc", {}).get("max_thrust_accel", 18.0))
    thrust = arrays["U"][:, 0]
    thrust_saturated_fraction = float(
        ((thrust <= 1e-9) | (thrust >= max_thrust - 1e-9)).mean()
    )
    warnings: list[str] = []
    target_gate_ids = sorted({int(value) for value in arrays["target_gate_id"]})
    required_gate_ids = config.get("dataset", {}).get("required_gate_ids")
    if required_gate_ids is None and config.get("dataset", {}).get("require_all_gates"):
        required_gate_ids = list(range(1, int(config.get("environment", {}).get("gate_count", 0)) + 1))
    if required_gate_ids:
        required_gate_ids = sorted({int(value) for value in required_gate_ids})
        missing_gate_ids = sorted(set(required_gate_ids) - set(target_gate_ids))
        if missing_gate_ids:
            raise ValueError(
                f"dataset is missing required target gate ids: {missing_gate_ids}; "
                f"observed={target_gate_ids}"
            )
        for episode_id in range(episode_count):
            key = f"episode_{episode_id:04d}"
            entry = manifest["episodes"][key]
            episode_arrays = load_episode(dataset_root / Path(entry["file"]))
            episode_gate_ids = {int(value) for value in episode_arrays["target_gate_id"]}
            missing_episode_gate_ids = sorted(set(required_gate_ids) - episode_gate_ids)
            if missing_episode_gate_ids:
                raise ValueError(
                    f"{key} is missing required target gate ids: {missing_episode_gate_ids}; "
                    f"observed={sorted(episode_gate_ids)}"
                )
    if stats["zero_input_dimensions"]:
        warnings.append(f"input dimensions always zero: {stats['zero_input_dimensions']}")
    if stats["controls_identical"]:
        warnings.append("all U rows are identical")
    if thrust_saturated_fraction > 0.5:
        warnings.append(
            f"thrust is at a configured bound for {thrust_saturated_fraction:.1%} of samples"
        )
    return {
        "status": "valid",
        "root": str(dataset_root),
        "final_dataset": str(final_path),
        "episodes": episode_count,
        "samples_per_episode": steps,
        "samples": stats["samples"],
        "X_shape": stats["X_shape"],
        "U_shape": stats["U_shape"],
        "initial_states": states,
        "environments": environments,
        "deadline_misses": stats["deadline_misses"],
        "average_mpc_solve_time": stats["average_mpc_solve_time"],
        "X_min": stats["X_min"],
        "X_max": stats["X_max"],
        "X_mean": stats["X_mean"],
        "X_std": stats["X_std"],
        "U_min": stats["U_min"],
        "U_max": stats["U_max"],
        "U_mean": stats["U_mean"],
        "U_std": stats["U_std"],
        "X_min_per_dim": stats["X_min_per_dim"],
        "X_max_per_dim": stats["X_max_per_dim"],
        "U_min_per_dim": stats["U_min_per_dim"],
        "U_max_per_dim": stats["U_max_per_dim"],
        "zero_input_dimensions": stats["zero_input_dimensions"],
        "constant_input_dimensions": stats["constant_input_dimensions"],
        "controls_identical": stats["controls_identical"],
        "target_gate_ids": target_gate_ids,
        "required_gate_ids": required_gate_ids or [],
        "thrust_saturated_fraction": thrust_saturated_fraction,
        "reference_speed_values": stats.get("reference_speed_values", []),
        "data_source_counts": stats.get("data_source_counts", {}),
        "path_context_shape": stats.get("path_context_shape"),
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = validate_dataset(args.dataset_root)
    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(
            f"VALID: episodes={result['episodes']}, samples={result['samples']}, "
            f"X.shape={tuple(result['X_shape'])}, U.shape={tuple(result['U_shape'])}, "
            f"deadline_misses={result['deadline_misses']}"
        )
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
