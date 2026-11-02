"""Resumable, independent-state QP imitation dataset; never connects to AirSim."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from airsim_interface import RawDroneState
from state_adapter import normalize_quaternion, quaternion_multiply
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from .convex_teacher import ConvexTeacher
from .decision_dataset import ROOT, FAMILIES, PLANNER_CONFIG, atomic_json, atomic_npz, build_features, sha256
from .decision_dataset_96k import SPLITS, route_for_group


EXPERIMENT = ROOT / "controller_architecture_study/results/decision_convex_96k_v1"
CONTRACT = "decision_convex_local_relative_pv_300d_v1"
VERSION = "independent_convex_multiroute_96k_v1"
SEED = 20261003


def specification(teacher, smoke=False):
    paths = [Path(__file__), Path(__file__).with_name("convex_teacher.py"),
             Path(__file__).with_name("decision_dataset.py"), Path(__file__).with_name("decision_dataset_96k.py"),
             ROOT/"falcon_paper_fidelity/mpc_complete_300d.py", ROOT/"falcon_paper_fidelity/motion_primitive.py",
             ROOT/"state_adapter.py", ROOT/"airsim_interface.py", ROOT/"environment.py"]
    value = {"version": VERSION, "contract": CONTRACT, "seed": SEED, "teacher": teacher.signature(),
        "environment_ids": [0, 48, 54] if smoke else list(range(60)),
        "routes_per_environment": 1 if smoke else 20, "samples_per_route": 8 if smoke else 80,
        "environment_splits": SPLITS, "families": list(FAMILIES),
        "sampling": "balanced_full_phase_interior_and_boundary_v1",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths}}
    # JSON-compatible values also make a saved manifest compare exactly on resume.
    return json.loads(json.dumps(value))


def sample_problem(trajectory, planner, rng, family):
    if family == 0:
        phase = 0. if rng.random() < .25 else rng.uniform(0., min(2., trajectory.segments[0].end_time))
    elif family == 2:
        phase = np.clip(rng.choice(trajectory.boundary_times)+rng.uniform(-.8, .8), 0., trajectory.duration)
    elif family == 3:
        phase = rng.uniform(trajectory.segments[-1].start_time, trajectory.duration)
    else:
        phase = rng.uniform(0., trajectory.duration)
    refs = planner.reference_window_for_trajectory(trajectory, elapsed=float(phase), dt=.1, horizon=21)
    p, v, q = refs[0].position.copy(), refs[0].velocity.copy(), refs[0].orientation.copy()
    # Dense interiors matter: most samples aren't pinned to extreme box corners.
    scale = 1. if rng.random() < .2 else rng.uniform(.05, 1.)
    if family in (0, 1, 2, 3):
        p += rng.uniform(-.08, .08, 3)
        v += rng.uniform(-.15, .15, 3)
    if family in (4, 7):
        p += scale*rng.uniform(-1., 1., 3)*[.6, 1., .6]
    if family in (5, 7):
        v += scale*rng.uniform(-1.5, 1.5, 3)
    if family in (6, 7):
        rotation = scale*rng.uniform(-.25, .25, 3)
        angle = np.linalg.norm(rotation)
        delta = np.r_[np.cos(angle/2), rotation*np.sin(angle/2)/max(angle, 1e-12)]
        q = normalize_quaternion(quaternion_multiply(q, delta))
    if family == 0:
        v = v*rng.uniform(0., 1.)
        if phase == 0.:
            v[:] = 0.
    v *= min(1., 8./max(np.linalg.norm(v), 1e-12))
    return RawDroneState(p, q, v, np.zeros(3), np.zeros(3)), refs, float(phase)


def preflight(output):
    teacher, planner = ConvexTeacher(), QuinticMotionPrimitivePlanner(PLANNER_CONFIG)
    results = []
    for env in range(4):
        trajectory, _ = route_for_group(env, env % 4, 20, planner)
        for index in range(16):
            rng = np.random.default_rng([SEED, env, index])
            state, refs, phase = sample_problem(trajectory, planner, rng, index % 8)
            base, d = teacher.query(state, refs)
            repeated, _ = teacher.query(state, refs)
            nearby = replace(state, position=state.position+[.01, -.01, .01])
            neighbor, _ = teacher.query(nearby, refs)
            after_neighbor, _ = teacher.query(state, refs)
            results.append({"environment_id": env, "route_index": env % 4, "family_id": index % 8, "phase_seconds": phase,
                "repeat_delta": float(max(abs(base-repeated).max(), abs(base-after_neighbor).max())),
                "neighbor_delta_per_output": abs(base-neighbor).tolist(), **{key: value for key, value in d.items()
                    if key != "first_acceleration"}})
    repeat = max(row["repeat_delta"] for row in results)
    nearby = np.max([row["neighbor_delta_per_output"] for row in results], axis=0)
    passed = repeat <= 2e-5 and np.all(nearby < [.1, .1, .1, .1])
    report = {"status": "PASS" if passed else "FAIL", "config": specification(teacher),
        "queries_checked": len(results), "max_repeat_delta": repeat,
        "max_neighbor_delta_per_output": nearby.tolist(),
        "max_constraint_violation": max(row["constraint_violation"] for row in results),
        "max_stationarity_scaled": max(row["stationarity_scaled"] for row in results),
        "mean_solve_ms": float(np.mean([row["solve_ms"] for row in results])), "rows": results,
        "airsim_flight_verified": False, "test_read": False,
        "note": "Finite numerical probes, not a global smoothness or closed-loop certificate."}
    atomic_json(output/"preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "config")}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("teacher preflight failed; do not generate production labels")
    return report


def generate(output, *, smoke=False, resume=False):
    teacher, planner = ConvexTeacher(), QuinticMotionPrimitivePlanner(PLANNER_CONFIG)
    config = specification(teacher, smoke)
    if not smoke:
        numerical = json.loads((output/"preflight.json").read_text(encoding="utf-8"))
        flight = json.loads((output/"teacher_acceptance.json").read_text(encoding="utf-8"))
        if (numerical["status"] != "PASS" or numerical["config"] != config or flight["status"] != "PASS"
                or flight["teacher_config"] != config or flight["controller"] != "convex_QP_teacher"):
            raise ValueError("current numerical preflight AND teacher AirSim flight must PASS before collection")
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or manifest["config"] != config:
            raise ValueError("use --resume with unchanged code/settings; otherwise use a new output directory")
    else:
        permitted = {"preflight.json", "teacher_acceptance.json", "flight"}
        if output.exists() and any(path.name not in permitted for path in output.iterdir()):
            raise ValueError("refusing to overwrite an unrecognized dataset directory")
        manifest = {"config": config, "chunks": {}, "complete": False}
        atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    samples, routes = config["samples_per_route"], config["routes_per_environment"]
    total = len(config["environment_ids"])*routes*samples
    for env in config["environment_ids"]:
        for route in range(routes):
            name = f"env{env:02d}_route{route:02d}"
            path, route_path = output/"chunks"/f"{name}.npz", output/"routes"/f"{name}.json"
            if name in manifest["chunks"]:
                meta = manifest["chunks"][name]
                if sha256(path) != meta["sha256"] or sha256(route_path) != meta["route_sha256"]:
                    raise ValueError(f"completed chunk or route changed: {name}")
                continue
            trajectory, layout = route_for_group(env, route, 20, planner)
            rows = []
            for index in range(samples):
                rng = np.random.default_rng([SEED, env, route, index])
                state, refs, phase = sample_problem(trajectory, planner, rng, index % 8)
                try:
                    control, diagnostic = teacher.query(state, refs)
                    repeat = teacher.query(state, refs)[0] if index == 0 else control
                    delta = float(abs(control-repeat).max())
                    if delta > 2e-5:
                        raise RuntimeError("label repeat failed")
                except Exception as exc:
                    atomic_json(output/"failure.json", {"environment": env, "route": route, "index": index,
                                "error": str(exc), "note": "No resampling or unconverged label acceptance."})
                    raise
                rows.append({"X": build_features(state, refs), "U": control, "row_id": (env*routes+route)*samples+index,
                    "environment_id": env, "route_id": layout["route_id"], "family_id": index % 8,
                    "geometry_family": layout["geometry_family"], "heading_yaw": layout["heading_yaw"],
                    "phase_seconds": phase, "actual_speed": float(np.linalg.norm(state.velocity)),
                    "reference_speed": float(np.linalg.norm(refs[0].velocity)),
                    "position_error": float(np.linalg.norm(state.position-refs[0].position)),
                    "repeat_delta": delta, "repeat_checked": index == 0, **diagnostic})
            atomic_json(route_path, layout)
            atomic_npz(path, {key: np.asarray([row[key] for row in rows]) for key in rows[0]})
            manifest["chunks"][name] = {"sha256": sha256(path), "route_sha256": sha256(route_path), "samples": samples}
            atomic_json(manifest_path, manifest)
            count = sum(chunk["samples"] for chunk in manifest["chunks"].values())
            print(f"accepted={count}/{total} env={env} route={route} elapsed_this_run={time.perf_counter()-started:.1f}s", flush=True)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"Collection complete: {output}; run prepare next.", flush=True)


def prepare(output):
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    c = manifest["config"]
    if not manifest["complete"] or c["contract"] != CONTRACT or c["version"] != VERSION:
        raise ValueError("incomplete or incompatible dataset")
    names = [f"env{env:02d}_route{route:02d}" for env in c["environment_ids"] for route in range(c["routes_per_environment"])]
    if set(manifest["chunks"]) != set(names):
        raise ValueError("missing or extra chunks")
    blocks = []
    for name in names:
        meta, path = manifest["chunks"][name], output/"chunks"/f"{name}.npz"
        if sha256(path) != meta["sha256"] or sha256(output/"routes"/f"{name}.json") != meta["route_sha256"]:
            raise ValueError(f"changed chunk/route: {name}")
        with np.load(path, allow_pickle=False) as archive:
            blocks.append({key: archive[key] for key in archive.files})
    data = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    count = len(names)*c["samples_per_route"]
    expected_ids = np.concatenate([np.arange((env*c["routes_per_environment"]+route)*c["samples_per_route"],
                                           (env*c["routes_per_environment"]+route+1)*c["samples_per_route"])
                                 for env in c["environment_ids"] for route in range(c["routes_per_environment"])])
    expected_env = np.repeat(c["environment_ids"], c["routes_per_environment"]*c["samples_per_route"])
    expected_routes = np.repeat([env*20+route for env in c["environment_ids"]
                                for route in range(c["routes_per_environment"])], c["samples_per_route"])
    if (data["X"].shape != (count, 300) or data["U"].shape != (count, 4)
            or not np.array_equal(data["row_id"], expected_ids) or any(not np.isfinite(v).all() for v in data.values())
            or not np.array_equal(data["environment_id"], expected_env) or not np.array_equal(data["route_id"], expected_routes)
            or np.any(data["family_id"] != data["row_id"] % 8) or np.any(data["X"][:, :3] != 0.)
            or np.max(data["constraint_violation"]) > 2e-4 or np.max(data["stationarity_scaled"]) > 2e-4
            or np.max(data["repeat_delta"]) > 2e-5 or np.any(data["U"] < [2.-1e-4, -3.-1e-4, -3.-1e-4, -2.-1e-4])
            or np.any(data["U"] > [20.+1e-4, 3.+1e-4, 3.+1e-4, 2.+1e-4])):
        raise ValueError("invalid sample mapping, features, controls or numerical certificates")
    dataset = output/"dnn_dataset"
    split = {"version": VERSION, "contract": CONTRACT, "source_manifest_sha256": sha256(output/"manifest.json"),
             "environment_ids": SPLITS, "counts": {}}
    if (dataset/"split_manifest.json").exists():
        saved = json.loads((dataset/"split_manifest.json").read_text(encoding="utf-8"))
        if saved["source_manifest_sha256"] != split["source_manifest_sha256"]:
            raise ValueError("prepared source changed; use a new output directory")
    for name, ids in SPLITS.items():
        mask = np.isin(data["environment_id"], ids)
        split["counts"][name] = int(mask.sum())
        if not mask.any():
            raise ValueError(f"empty split: {name}")
        atomic_npz(dataset/f"{name}.npz", {key: value[mask] for key, value in data.items()})
    train = np.isin(data["environment_id"], SPLITS["train"])
    validation = np.isin(data["environment_id"], SPLITS["validation"])
    norm = {"input_contract": CONTRACT, "fit_split": "train", "fit_samples": int(train.sum())}
    for name, key in (("x", "X"), ("u", "U")):
        values = data[key][train].astype(float)
        norm[name+"_mean"], norm[name+"_std"] = values.mean(0).tolist(), np.maximum(values.std(0), 1e-6).tolist()
    atomic_json(dataset/"normalization.json", norm)
    split["artifact_sha256"] = {name: sha256(dataset/name) for name in ("train.npz", "validation.npz", "test.npz", "normalization.json")}
    atomic_json(dataset/"split_manifest.json", split)
    audit = {"status": "PASS", "contract": CONTRACT, "samples": count, "split_counts": split["counts"],
        "max_constraint_violation": float(data["constraint_violation"].max()),
        "max_stationarity_scaled": float(data["stationarity_scaled"].max()),
        "max_repeat_delta": float(data["repeat_delta"].max()), "max_slack": float(data["max_slack"].max()),
        "rest_start_samples": int(np.sum((data["family_id"] == 0) & (data["actual_speed"] == 0))),
        "family_counts_by_split": {name: {family: int(np.sum(np.isin(data["environment_id"], ids) & (data["family_id"] == i)))
              for i, family in enumerate(FAMILIES)} for name, ids in SPLITS.items()},
        "actual_speed_range": [float(data["actual_speed"].min()), float(data["actual_speed"].max())],
        "label_std_train": norm["u_std"], "label_std_validation": data["U"][validation].std(0).tolist(),
        "active_constraint_fractions_by_family": {family: {
            "acceleration_bound": float(data["acceleration_bound_active"][data["family_id"] == i].mean()),
            "tilt_bound": float(data["tilt_bound_active"][data["family_id"] == i].mean()),
            "soft_corridor_slack": float(np.mean(data["max_slack"][data["family_id"] == i] > 1e-4))}
            for i, family in enumerate(FAMILIES)},
        "test_used_for_model_selection": False,
        "note": "New teacher/new normalization: not directly comparable to old normalized MSE. No monotonic-capacity guarantee."}
    atomic_json(output/"data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "generate", "prepare"))
    parser.add_argument("--output", type=Path, default=EXPERIMENT)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.smoke_test:
        output = output.with_name(output.name+"_smoke")
    if args.stage == "preflight":
        preflight(output)
    elif args.stage == "generate":
        generate(output, smoke=args.smoke_test, resume=args.resume)
    else:
        prepare(output)


if __name__ == "__main__":
    main()
