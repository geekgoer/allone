"""Create MPC-labelled recovery samples from a completed nominal dataset.

This is an offline augmentation stage.  It does not modify the nominal
dataset and does not connect to AirSim.  Each perturbation changes the
controller state, then recomputes the label with the same MPC and the route's
future reference window.  Reusing the old label after changing X would make
the samples inconsistent, so labels are never copied for synthetic rows.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from dataset_store import (
    EpisodeSamples,
    atomic_write_json,
    is_valid_episode,
    load_episode,
    merge_episodes,
    save_episode_atomic,
    validate_merged_dataset,
)
from gate_route import ContinuousGateRoute
from mpc_controller import MPCController
from training_environment_generator import load_training_environments


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "teacher_dataset_robust_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "teacher_dataset_robust_v1_augmented"


def _episode_key(episode_id: int) -> str:
    return f"episode_{episode_id:04d}"


def _episode_path(root: Path, episode_id: int) -> Path:
    return root / "episodes" / f"{_episode_key(episode_id)}.npz"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_paths(source_root: Path) -> list[Path]:
    manifest = _load_json(source_root / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(
            f"source dataset must be completed before augmentation: {manifest.get('status')!r}"
        )
    paths = []
    for episode_id in range(int(manifest["episode_count"])):
        entry = manifest["episodes"][_episode_key(episode_id)]
        path = source_root / Path(entry["file"])
        if not is_valid_episode(
            path,
            expected_steps=int(manifest["steps_per_episode"]),
            expected_episode_id=episode_id,
        ):
            raise ValueError(f"source episode is missing or invalid: {path}")
        paths.append(path)
    return paths


def _perturbation_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # Keep the first augmentation set compact enough for ordinary Windows
    # storage.  DAgger will add larger, rollout-derived deviations later.
    for value in (0.5,):
        for sign in (-1.0, 1.0):
            specs.append(
                {
                    "name": f"position_y_{sign:+.1f}_{value:.2f}",
                    "x_delta": np.array([0.0, -sign * value, 0.0] + [0.0] * 9),
                    "physical_offset": np.array([0.0, sign * value, 0.0]),
                }
            )
    for value in (0.5,):
        for sign in (-1.0, 1.0):
            specs.append(
                {
                    "name": f"position_z_{sign:+.1f}_{value:.2f}",
                    "x_delta": np.array([0.0, 0.0, -sign * value] + [0.0] * 9),
                    "physical_offset": np.array([0.0, 0.0, sign * value]),
                }
            )
    for axis, index in (("y", 7), ("z", 8)):
        for value in (0.5,):
            for sign in (-1.0, 1.0):
                x_delta = np.zeros(12, dtype=np.float64)
                x_delta[index] = sign * value
                specs.append(
                    {
                        "name": f"velocity_{axis}_{sign:+.1f}_{value:.1f}",
                        "x_delta": x_delta,
                        "physical_offset": np.zeros(3, dtype=np.float64),
                    }
                )
    for axis, index in (("roll", 3), ("pitch", 4)):
        for value in (0.1,):
            for sign in (-1.0, 1.0):
                x_delta = np.zeros(12, dtype=np.float64)
                x_delta[index] = sign * value
                specs.append(
                    {
                        "name": f"attitude_{axis}_{sign:+.1f}_{value:.2f}",
                        "x_delta": x_delta,
                        "physical_offset": np.zeros(3, dtype=np.float64),
                    }
                )
    return specs


def _episode_from_arrays(arrays: dict[str, np.ndarray], *, episode_id: int) -> EpisodeSamples:
    return EpisodeSamples(
        X=np.asarray(arrays["X"]),
        U=np.asarray(arrays["U"]),
        episode_id=episode_id,
        initial_state_id=int(np.asarray(arrays["initial_state_id"])[0]),
        environment_id=int(np.asarray(arrays["environment_id"])[0]),
        step_id=np.asarray(arrays["step_id"]),
        target_gate_id=np.asarray(arrays["target_gate_id"]),
        timestamp=np.asarray(arrays["timestamp"]),
        position=np.asarray(arrays["position"]),
        quaternion=np.asarray(arrays["quaternion"]),
        velocity=np.asarray(arrays["velocity"]),
        acceleration=np.asarray(arrays["acceleration"]),
        reference_position=np.asarray(arrays["reference_position"]),
        reference_orientation=np.asarray(arrays["reference_orientation"]),
        mpc_solve_time=np.asarray(arrays["mpc_solve_time"]),
        actual_loop_period=np.asarray(arrays["actual_loop_period"]),
        deadline_miss=np.asarray(arrays["deadline_miss"]),
        path_context=np.asarray(arrays["path_context"]) if "path_context" in arrays else None,
        reference_speed=np.asarray(arrays["reference_speed"]) if "reference_speed" in arrays else None,
        data_source=np.asarray(arrays["data_source"]) if "data_source" in arrays else None,
        perturbation=np.asarray(arrays["perturbation"]) if "perturbation" in arrays else None,
        student_control=np.asarray(arrays["student_control"]) if "student_control" in arrays else None,
        shield_active=np.asarray(arrays["shield_active"]) if "shield_active" in arrays else None,
    )


def _make_perturbed_arrays(
    source: dict[str, np.ndarray],
    *,
    environment: Any,
    gate_config: dict[str, Any],
    mpc_config: dict[str, Any],
    dt: float,
    spec: dict[str, Any],
    data_source_code: int,
) -> dict[str, np.ndarray]:
    x = np.asarray(source["X"], dtype=np.float64)
    steps = x.shape[0]
    x_delta = np.asarray(spec["x_delta"], dtype=np.float64)
    physical_offset = np.asarray(spec["physical_offset"], dtype=np.float64)
    if x_delta.shape != (12,) or physical_offset.shape != (3,):
        raise ValueError("invalid perturbation specification")
    perturbed_x = x + x_delta[None, :]
    route = ContinuousGateRoute(
        environment.gates,
        np.asarray(source["position"])[0],
        approach_distance=float(gate_config["approach_distance"]),
        exit_distance=float(gate_config["exit_distance"]),
        reference_speed=float(environment.reference_speed),
        start_time=0.0,
        gate_width=float(gate_config["width"]),
        gate_height=float(gate_config["height"]),
        gate_thickness=float(gate_config["thickness"]),
        clearance=float(gate_config.get("clearance", 0.25)),
        completion_radius=float(gate_config["threshold"]),
    )
    controller = MPCController(
        dt=dt,
        horizon=int(mpc_config["horizon"]),
        max_horizontal_accel=float(mpc_config.get("max_horizontal_accel", 4.0)),
        max_thrust_accel=float(mpc_config.get("max_thrust_accel", 18.0)),
        max_body_rate=float(mpc_config.get("max_body_rate", 3.0)),
    )
    controls: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    source_positions = np.asarray(source["position"], dtype=np.float64)
    for step_id in range(steps):
        now = step_id * dt
        window = route.reference_window(now, dt=dt, horizon=controller.horizon)
        controls.append(controller.solve(perturbed_x[step_id], window))
        if "path_context" in source:
            contexts.append(np.asarray(source["path_context"])[step_id] - physical_offset)
    result = {key: np.asarray(value).copy() for key, value in source.items()}
    result["X"] = perturbed_x
    result["U"] = np.asarray(controls, dtype=np.float64)
    result["position"] = source_positions + physical_offset[None, :]
    if contexts:
        result["path_context"] = np.asarray(contexts, dtype=np.float64)
    result["data_source"] = np.full(steps, data_source_code, dtype=np.int8)
    result["perturbation"] = np.tile(physical_offset, (steps, 1))
    return result


def _new_manifest(
    *,
    source_root: Path,
    output_root: Path,
    source_manifest: dict[str, Any],
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    base_count = int(source_manifest["episode_count"])
    total = base_count * (1 + len(specs))
    episodes: dict[str, dict[str, Any]] = {}
    for episode_id in range(total):
        if episode_id < base_count:
            source_id = episode_id
            variant = "nominal"
            source_code = 0
        else:
            relative = episode_id - base_count
            variant_id, source_id = divmod(relative, base_count)
            variant = specs[variant_id]["name"]
            source_code = 1
        episodes[_episode_key(episode_id)] = {
            "episode_id": episode_id,
            "source_episode_id": source_id,
            "variant": variant,
            "data_source_code": source_code,
            "status": "pending",
            "attempts": 0,
            "sample_count": 0,
            "file": str(Path("episodes") / f"{_episode_key(episode_id)}.npz"),
            "last_error": None,
        }
    return {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "base_episode_count": base_count,
        "perturbation_count": len(specs),
        "episode_count": total,
        "steps_per_episode": int(source_manifest["steps_per_episode"]),
        "initial_state_count": int(source_manifest["initial_state_count"]),
        "environment_count": int(source_manifest["environment_count"]),
        "variants": [spec["name"] for spec in specs],
        "episodes": episodes,
    }


def _update_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, manifest)


def augment(
    source_root: str | Path = DEFAULT_SOURCE,
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    resume: bool = False,
    limit: int | None = None,
) -> Path:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    source_manifest = _load_json(source / "manifest.json")
    source_paths = _source_paths(source)
    source_config = _load_json(source / "collection_config.json")
    environments = load_training_environments(source / "environments.json")
    specs = _perturbation_specs()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume to continue")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        expected_variants = [spec["name"] for spec in specs]
        if manifest.get("variants") != expected_variants:
            raise ValueError("resume output was created with a different perturbation specification")
    else:
        manifest = _new_manifest(
            source_root=source,
            output_root=output,
            source_manifest=source_manifest,
            specs=specs,
        )
        (output / "initial_states.json").write_text(
            (source / "initial_states.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (output / "environments.json").write_text(
            (source / "environments.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        config = deepcopy(source_config)
        config["dataset"]["root"] = output.name
        config["dataset"]["output_name"] = "teacher_dataset_robust_v1_augmented.npz"
        config["dataset"]["strategy"] = "nominal_plus_state_perturbations"
        config["dataset"]["data_source_code"] = 0
        (output / "collection_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _update_manifest(manifest_path, manifest)

    config = _load_json(output / "collection_config.json")
    steps = int(manifest["steps_per_episode"])
    base_count = int(manifest["base_episode_count"])
    base_config = source_config
    max_items = int(manifest["episode_count"]) if limit is None else min(int(limit), int(manifest["episode_count"]))
    for episode_id in range(max_items):
        key = _episode_key(episode_id)
        entry = manifest["episodes"][key]
        path = _episode_path(output, episode_id)
        if entry.get("status") == "completed" and is_valid_episode(
            path, expected_steps=steps, expected_episode_id=episode_id
        ):
            continue
        source_id = int(entry["source_episode_id"])
        try:
            source_arrays = load_episode(source_paths[source_id])
            if episode_id < base_count:
                arrays = {key: np.asarray(value).copy() for key, value in source_arrays.items()}
                arrays["episode_id"] = np.full(steps, episode_id, dtype=np.int64)
                arrays["data_source"] = np.zeros(steps, dtype=np.int8)
                arrays["perturbation"] = np.zeros((steps, 3), dtype=np.float64)
            else:
                variant_id = (episode_id - base_count) // base_count
                arrays = _make_perturbed_arrays(
                    source_arrays,
                    environment=environments[int(source_arrays["environment_id"][0])],
                    gate_config=config["gate"],
                    mpc_config=config["mpc"],
                    dt=float(config["timing"]["dt"]),
                    spec=specs[variant_id],
                    data_source_code=1,
                )
                arrays["episode_id"] = np.full(steps, episode_id, dtype=np.int64)
            save_episode_atomic(path, _episode_from_arrays(arrays, episode_id=episode_id))
            entry["status"] = "completed"
            entry["sample_count"] = steps
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["last_error"] = None
        except Exception as exc:
            entry["status"] = "failed"
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["last_error"] = f"{type(exc).__name__}: {exc}"
            _update_manifest(manifest_path, manifest)
            raise
        _update_manifest(manifest_path, manifest)
        completed = sum(item.get("status") == "completed" for item in manifest["episodes"].values())
        print(
            f"Episode: {episode_id + 1:04d} / {int(manifest['episode_count']):04d} | "
            f"Completed: {completed} | Samples: {completed * steps} / "
            f"{int(manifest['episode_count']) * steps}",
            flush=True,
        )

    failed = [
        episode_id
        for episode_id in range(int(manifest["episode_count"]))
        if manifest["episodes"][_episode_key(episode_id)].get("status") != "completed"
    ]
    if not failed:
        paths = [_episode_path(output, episode_id) for episode_id in range(int(manifest["episode_count"]))]
        final_path = output / "final" / "teacher_dataset_robust_v1_augmented.npz"
        stats = merge_episodes(
            paths,
            final_path,
            expected_steps=steps,
            expected_episode_count=int(manifest["episode_count"]),
            metadata_csv_path=output / "final" / "teacher_metadata.csv",
        )
        validate_merged_dataset(
            final_path,
            expected_samples=int(manifest["episode_count"]) * steps,
            expected_episodes=int(manifest["episode_count"]),
            expected_steps=steps,
            expected_initial_states=int(manifest["initial_state_count"]),
            expected_environments=int(manifest["environment_count"]),
        )
        manifest["status"] = "completed"
        manifest["final_dataset"] = str(Path("final") / final_path.name)
        manifest["statistics"] = stats
        _update_manifest(manifest_path, manifest)
        print(f"Augmentation completed: X.shape={tuple(stats['X_shape'])}, U.shape={tuple(stats['U_shape'])}")
    else:
        manifest["status"] = "incomplete"
        manifest["failed_episode_ids"] = failed
        _update_manifest(manifest_path, manifest)
        print(f"Augmentation incomplete; rerun with --resume. Remaining episodes: {failed}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N output episodes")
    args = parser.parse_args()
    augment(args.source_root, args.output_root, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
