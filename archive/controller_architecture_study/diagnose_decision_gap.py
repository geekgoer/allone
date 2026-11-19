"""CPU-only diagnostics of frozen Train/Validation data and best checkpoints.

No model updates, split changes, Test reads, or AirSim connections.
"""

import argparse
import json

import numpy as np

from .decision_dataset import FAMILIES, RefinedTeacher, atomic_json, build_features, sha256
from .decision_dataset_96k import EXPERIMENT, sample_problem
from .eval_offline import load_checkpoint, load_normalization


def read_split(name):
    with np.load(EXPERIMENT / "dnn_dataset" / f"{name}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def summarize(data):
    lower, upper = np.array([2., -3., -3., -2.]), np.array([20., 3., 3., 2.])
    return {"samples": len(data["X"]), "routes": len(np.unique(data["route_id"])),
        "families": {name: {"samples": int((data["family_id"] == index).sum()),
            "label_mean": data["U"][data["family_id"] == index].mean(axis=0).tolist(),
            "label_std": data["U"][data["family_id"] == index].std(axis=0).tolist(),
            "saturated_fraction": float(np.any((data["U"][data["family_id"] == index] <= lower + .01)
                | (data["U"][data["family_id"] == index] >= upper - .01), axis=1).mean())}
            for index, name in enumerate(FAMILIES)},
        "geometry_counts": {str(int(kind)): int((data["geometry_family"] == kind).sum())
                            for kind in np.unique(data["geometry_family"])},
        "max_kkt": float(data["kkt"].max()), "max_action_delta": float(data["action_delta"].max())}


def nearest(torch, query, candidates, *, excluded=None):
    distances = torch.cdist(torch.from_numpy(query), torch.from_numpy(candidates))
    if excluded is not None:
        distances[torch.from_numpy(excluded)] = float("inf")
    values, indices = distances.min(dim=1)
    return values.numpy() / np.sqrt(query.shape[1]), indices.numpy()


def fresh_state_probe(torch, train, norm):
    from .fly_decision_dnn import StoredRoutePlanner

    teacher = RefinedTeacher()
    source = json.loads((EXPERIMENT / "manifest.json").read_text())
    if sha256(teacher.controller.library_path) != source["config"]["dll_sha256"]:
        raise ValueError("teacher DLL changed")
    rng = np.random.default_rng(20261017)
    features, targets, route_ids, rejections = [], [], [], 0
    for env in (0, 12, 24, 36):
        for kind in range(4):
            route_index = kind + 4 * (env // 12)
            name = f"env{env:02d}_route{route_index:02d}"
            path = EXPERIMENT / "routes" / f"{name}.json"
            if sha256(path) != source["chunks"][name]["route_sha256"]:
                raise ValueError("saved route changed")
            layout = json.loads(path.read_text())
            planner = StoredRoutePlanner(layout)
            route_ids.append(layout["route_id"])
            for _ in range(8):
                for attempt in range(8):
                    state, references, _ = sample_problem(planner.trajectory, planner, rng, 7)
                    try:
                        target, _ = teacher.query(state, references)
                    except RuntimeError:
                        rejections += 1
                        if attempt == 7:
                            raise
                        continue
                    features.append(build_features(state, references))
                    targets.append(target)
                    break
    fresh_x = (np.asarray(features) - norm["x_mean"]) / norm["x_std"]
    fresh_u = (np.asarray(targets) - norm["u_mean"]) / norm["u_std"]
    selected = np.isin(train["route_id"], route_ids) & (train["family_id"] == 7)
    result = {"fresh_queries": len(features), "training_routes": route_ids,
        "existing_training_rows": int(selected.sum()), "rejected_queries": rejections,
        "models": {}, "note": "Independent combined-recovery states on existing Train routes; diagnostic only."}
    for name in ("plain_mlp_h4_w64_seed0", "plain_mlp_h4_w128_seed0", "plain_mlp_h4_w1024_seed0"):
        model = load_checkpoint(EXPERIMENT / "runs" / name / "best_model.pt", device="cpu")
        with torch.inference_mode():
            predicted = model(torch.from_numpy(fresh_x)).numpy()
            stored_x = (train["X"][selected] - norm["x_mean"]) / norm["x_std"]
            stored_u = (train["U"][selected] - norm["u_mean"]) / norm["u_std"]
            stored_prediction = model(torch.from_numpy(stored_x)).numpy()
        errors = np.mean((predicted - fresh_u)**2, axis=1)
        result["models"][name] = {"existing_train_points_mse": float(np.mean((stored_prediction - stored_u)**2)),
            "fresh_states_known_route_mse": float(errors.mean()),
            "fresh_states_median_mse": float(np.median(errors)),
            "fresh_states_p95_mse": float(np.percentile(errors, 95))}
    return result


def diagnose(probes, fresh=False):
    import torch

    torch.set_num_threads(2)
    train, val = read_split("train"), read_split("validation")
    norm = load_normalization(EXPERIMENT / "dnn_dataset")
    result = {"test_read": False, "models_updated": False, "airsim_connected": False,
              "train": summarize(train), "validation": summarize(val)}
    manifest = json.loads((EXPERIMENT / "dnn_dataset/split_manifest.json").read_text())
    result["integrity"] = {
        "train_validation_routes_disjoint": not bool(set(train["route_id"]) & set(val["route_id"])),
        "normalization_train_mean_max_delta": float(np.max(np.abs(
            train["X"].mean(axis=0, dtype=np.float64) - norm["x_mean"]))),
        "constant_input_count": int(np.sum(train["X"].std(axis=0) < 1e-8)),
        "source_manifest_matches_split": sha256(EXPERIMENT / "manifest.json") == manifest["source_manifest_sha256"]}
    x_train = (train["X"] - norm["x_mean"]) / norm["x_std"]
    x_val = (val["X"] - norm["x_mean"]) / norm["x_std"]
    u_val = (val["U"] - norm["u_mean"]) / norm["u_std"]
    summary = json.loads((EXPERIMENT / "runs/comparison_capacity.json").read_text())
    result["cached_models"] = summary["models"]
    result["validation_details"] = {}
    query_indices = None
    for name in ("plain_mlp_h4_w64_seed0", "plain_mlp_h4_w128_seed0", "plain_mlp_h4_w256_seed0",
                 "plain_mlp_h4_w1024_seed0", "plain_mlp_h8_w512_seed0"):
        model = load_checkpoint(EXPERIMENT / "runs" / name / "best_model.pt", device="cpu")
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(x_val), 512):
                predictions.append(model(torch.from_numpy(x_val[start:start + 512])).numpy())
        predicted = np.concatenate(predictions)
        errors = np.mean((predicted - u_val)**2, axis=1)
        clipped = np.clip(predicted * norm["u_std"] + norm["u_mean"],
                          [2., -3., -3., -2.], [20., 3., 3., 2.])
        saturation = np.any((val["U"] <= np.array([2., -3., -3., -2.]) + .01)
                            | (val["U"] >= np.array([20., 3., 3., 2.]) - .01), axis=1)
        ranked = np.sort(errors)[::-1]
        by_route = [(int(route), float(errors[val["route_id"] == route].mean()))
                    for route in np.unique(val["route_id"])]
        details = {"normalized_mse": float(errors.mean()),
            "clipped_normalized_mse": float(np.mean(((clipped - val["U"]) / norm["u_std"])**2)),
            "families": {family: {"mse": float(errors[val["family_id"] == index].mean()),
                "total_error_share": float(errors[val["family_id"] == index].sum() / errors.sum())}
                for index, family in enumerate(FAMILIES)},
            "geometry_mse": {str(int(kind)): float(errors[val["geometry_family"] == kind].mean())
                             for kind in np.unique(val["geometry_family"])},
            "environment_mse": {str(int(env)): float(errors[val["environment_id"] == env].mean())
                                for env in np.unique(val["environment_id"])},
            "top_1_percent_error_share": float(ranked[:len(errors)//100].sum() / errors.sum()),
            "top_5_percent_error_share": float(ranked[:len(errors)//20].sum() / errors.sum()),
            "saturated_mse": float(errors[saturation].mean()),
            "unsaturated_mse": float(errors[~saturation].mean()),
            "worst_routes": sorted(by_route, key=lambda item: -item[1])[:10]}
        result["validation_details"][name] = details
        if name == "plain_mlp_h4_w1024_seed0":
            combined = np.flatnonzero(val["family_id"] == 7)
            query_indices = combined[np.argsort(errors[combined])[-probes:]] if probes else []
        print(f"CPU diagnostic {name}: validation={errors.mean():.6g}, "
              f"combined error share={details['families']['combined_recovery']['total_error_share']:.1%}", flush=True)
        del model
    result["nearest_neighbor_diagnostic"] = {}
    rng = np.random.default_rng(312)
    for family, name in enumerate(FAMILIES):
        ti, vi = np.flatnonzero(train["family_id"] == family), np.flatnonzero(val["family_id"] == family)
        tq, vq = rng.choice(ti, 128, replace=False), rng.choice(vi, 128, replace=False)
        val_dist, neighbors = nearest(torch, x_val[vq], x_train[ti])
        train_dist, train_neighbors = nearest(torch, x_train[tq], x_train[ti],
            excluded=train["route_id"][tq, None] == train["route_id"][ti][None, :])
        val_delta = (val["U"][vq] - train["U"][ti[neighbors]]) / norm["u_std"]
        train_delta = (train["U"][tq] - train["U"][ti[train_neighbors]]) / norm["u_std"]
        result["nearest_neighbor_diagnostic"][name] = {
            "queries_per_split": 128, "metric": "full standardized 300D RMS distance, same state family",
            "train_cross_route_distance_median": float(np.median(train_dist)),
            "validation_to_train_distance_median": float(np.median(val_dist)),
            "train_cross_route_label_mse": float(np.mean(train_delta**2)),
            "validation_nearest_train_label_mse": float(np.mean(val_delta**2))}
    if probes:
        from airsim_interface import RawDroneState
        from state_adapter import normalize_quaternion
        from .fly_decision_dnn import StoredRoutePlanner

        stored = json.loads((EXPERIMENT / "manifest.json").read_text())
        teacher = RefinedTeacher()
        if sha256(teacher.controller.library_path) != stored["config"]["dll_sha256"]:
            raise ValueError("teacher DLL changed; refuse a misleading label probe")
        rows = []
        for index in query_indices:
            route_index = int(val["route_id"][index] % 20)
            route_name = f"env{val['environment_id'][index]:02d}_route{route_index:02d}"
            route_path = EXPERIMENT / "routes" / f"{route_name}.json"
            if sha256(route_path) != stored["chunks"][route_name]["route_sha256"]:
                raise ValueError("saved route changed")
            planner = StoredRoutePlanner(json.loads(route_path.read_text()))
            refs = planner.reference_window_for_trajectory(planner.trajectory,
                elapsed=float(val["phase_seconds"][index]), dt=.1, horizon=21)
            feature = val["X"][index]
            signs = np.array([1., -1., -1.])
            state = RawDroneState(refs[0].position - feature[10:13] * signs,
                normalize_quaternion(feature[3:7] * [1., 1., -1., -1.]),
                feature[7:10] * signs, np.zeros(3), np.zeros(3))
            try:
                output, diagnostic = teacher.query(state, refs)
                repeat, _ = teacher.query(state, refs)
                row = {"row_id": int(val["row_id"][index]), "family": "combined_recovery",
                    "feature_reconstruction_max_delta": float(np.max(np.abs(build_features(state, refs) - feature))),
                    "stored_label": val["U"][index].tolist(), "queried_label": output.tolist(),
                    "stored_label_max_delta": float(np.max(np.abs(output - val["U"][index]))),
                    "repeat_max_delta": float(np.max(np.abs(output - repeat))), "diagnostics": diagnostic}
                row["small_position_perturbation"] = []
                for epsilon in (.001, .01):
                    perturbed = RawDroneState(state.position + np.array([epsilon, -epsilon, epsilon]),
                        state.quaternion, state.velocity, state.acceleration, state.angular_velocity)
                    try:
                        control, _ = teacher.query(perturbed, refs)
                        row["small_position_perturbation"].append({"per_axis_m": epsilon,
                            "control_max_delta": float(np.max(np.abs(control - output)))})
                    except RuntimeError as exc:
                        row["small_position_perturbation"].append({"per_axis_m": epsilon, "error": str(exc)})
            except RuntimeError as exc:
                row = {"row_id": int(val["row_id"][index]), "error": str(exc)}
            rows.append(row)
        result["label_probes"] = {"selection": "largest DNN's worst combined-recovery Validation rows",
                                  "rows": rows, "note": "Selected diagnostics, not a proof about all labels."}
    if fresh:
        result["fresh_state_probe"] = fresh_state_probe(torch, train, norm)
    destination = EXPERIMENT / "diagnosis" / "train_validation_gap.json"
    atomic_json(destination, result)
    print(f"Diagnostic report: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-probes", type=int, default=0)
    parser.add_argument("--fresh-state-probe", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.teacher_probes <= 32:
        parser.error("teacher-probes must be between 0 and 32")
    diagnose(args.teacher_probes, args.fresh_state_probe)
