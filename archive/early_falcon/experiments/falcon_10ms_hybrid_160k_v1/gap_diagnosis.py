"""CPU audit and frozen ablations for cold-RTI data. Never opens test.npz."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dnn_model_utils import build_model
from prepare_dnn_dataset import _atomic_npz, normalization_from_train

SOURCE = HERE / "dnn_dataset_v4_cold_rti_labels_v1"
OUTPUT = HERE / "diagnostics" / "cold_rti_gap_v1"
ARMS = ("environment", "frame_iid", "episode_within_environment",
        "environment_low_kkt", "environment_matched_random")


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def take(data, indices):
    return {key: value[indices] for key, value in data.items()}


def control_bucket(u):
    """Recompute from CURRENT labels, not pre-relabel bucket metadata."""
    lower = np.array([2., -3., -3., -2.])
    upper = np.array([20., 3., 3., 2.])
    margin = np.minimum(u - lower, upper - u)
    saturated = np.any(margin <= 1e-3, axis=1)
    near = np.any(margin <= .15 * (upper - lower), axis=1)
    return np.where(saturated, 2, np.where(near, 1, 0))


def strata(data, include_environment=False):
    speed_bin = np.digitize(np.linalg.norm(data["velocity"], axis=1), [3.5, 8., 12.])
    columns = [data["source_kind"], control_bucket(data["U"]), speed_bin]
    if include_environment:
        columns.insert(0, data["environment_id"])
    return np.unique(np.column_stack(columns), axis=0, return_inverse=True)[1]


def stratified_rows(labels, count, rng):
    """Exact sample count using largest-remainder allocation, no error tuning."""
    ids, sizes = np.unique(labels, return_counts=True)
    raw = sizes * (count / len(labels))
    quotas = np.floor(raw).astype(int)
    order = np.argsort(-(raw - quotas), kind="stable")
    quotas[order[:count - int(quotas.sum())]] += 1
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == key), int(n), replace=False)
        for key, n in zip(ids, quotas)
    ]))


def overlap(data, a, b):
    return {key: int(len(np.intersect1d(data[key][a], data[key][b])))
            for key in ("environment_id", "episode_id", "query_group_id")}


def describe(data):
    u = data["U"].astype(np.float64)
    return {
        "samples": len(u),
        "environments": np.unique(data["environment_id"]).tolist(),
        "episodes": len(np.unique(data["episode_id"])),
        "source_counts": {str(int(k)): int(np.sum(data["source_kind"] == k))
                          for k in np.unique(data["source_kind"])},
        "low_speed_count": int(np.sum(np.linalg.norm(data["velocity"], axis=1) <= 3.5)),
        "current_label_saturation_fraction": float(np.mean(control_bucket(u) == 2)),
        "legacy_bucket_mismatch_fraction": float(np.mean(control_bucket(u) != data["control_bucket"])),
        "kkt_quantiles_p0_p50_p90_p95_p99_p100":
            np.quantile(data["cold_rti_kkt"], [0, .5, .9, .95, .99, 1]).tolist(),
        "u_mean": u.mean(0).tolist(), "u_std": u.std(0).tolist(),
    }


def make_arms(data, train_count, seed=20260919):
    n = len(data["X"])
    rng = np.random.default_rng(seed)
    tr, va = np.arange(train_count), np.arange(train_count, n)
    iid_va = stratified_rows(strata(data, True), len(va), rng)
    # Random whole episodes in EVERY development environment. All perturbations
    # derived from an episode and every five-step rollout stay with that episode.
    envs = np.unique(data["environment_id"])
    episode_env = {}
    for episode in np.unique(data["episode_id"]):
        values = np.unique(data["environment_id"][data["episode_id"] == episode])
        if len(values) != 1:
            raise ValueError("episode_id is not globally unique across environments")
        episode_env[int(episode)] = int(values[0])
    target_episodes = int(round(len(episode_env) * len(va) / n))
    if not len(envs) <= target_episodes < 2 * len(envs):
        raise ValueError("Expected approximately 10 episodes per environment")
    extra = set(rng.choice(envs, target_episodes - len(envs), replace=False).tolist())
    held_episodes = []
    for env in envs:
        candidates = [ep for ep, e in episode_env.items() if e == env]
        count = 1 + int(env in extra)
        if len(candidates) <= count:
            raise ValueError("Insufficient episodes for within-environment holdout")
        held_episodes.extend(rng.choice(candidates, count, replace=False).tolist())
    grouped_va = np.flatnonzero(np.isin(data["episode_id"], held_episodes))
    threshold = float(np.quantile(data["cold_rti_kkt"][tr], .95))
    low = tr[data["cold_rti_kkt"][tr] <= threshold]
    # Match retained counts per source/current-control-bucket/speed stratum.
    labels = strata(data)
    matched = np.sort(np.concatenate([
        rng.choice(tr[labels[tr] == key], int(np.sum(labels[low] == key)), replace=False)
        for key in np.unique(labels[tr])
    ]))
    arms = {
        "environment": (tr, va),
        "frame_iid": (np.setdiff1d(np.arange(n), iid_va), iid_va),
        "episode_within_environment": (np.setdiff1d(np.arange(n), grouped_va), grouped_va),
        "environment_low_kkt": (low, va),
        "environment_matched_random": (matched, va),
    }
    for name, (a, b) in arms.items():
        if len(np.intersect1d(a, b)) or not len(a) or not len(b):
            raise ValueError(f"Invalid row split: {name}")
        if name != "frame_iid" and overlap(data, a, b)["query_group_id"]:
            raise ValueError(f"Rollout leakage: {name}")
        if name == "episode_within_environment" and overlap(data, a, b)["episode_id"]:
            raise ValueError("Episode leakage")
    return arms, threshold


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "design.json").exists() or (output / "pool.npz").exists():
        raise FileExistsError("Prepared output exists; use audit/train or a NEW --output directory")
    files = ("train.npz", "validation.npz", "normalization.json", "split_manifest.json")
    hashes = {name: digest(source / name) for name in files}
    train, val = (read_npz(source / f"{s}.npz") for s in ("train", "validation"))
    if train.keys() != val.keys():
        raise ValueError("Train/validation metadata differ")
    data = {k: np.concatenate([train[k], val[k]]) for k in train}
    for k in ("X", "U", "cold_rti_kkt", "velocity"):
        if not np.isfinite(data[k]).all():
            raise ValueError(f"Non-finite {k}")
    arms, threshold = make_arms(data, len(train["X"]))
    meta = json.loads((source / "split_manifest.json").read_text(encoding="utf-8"))
    norm = normalization_from_train(train["X"], train["U"])
    saved = json.loads((source / "normalization.json").read_text(encoding="utf-8"))
    for key in norm:
        np.testing.assert_allclose(norm[key], saved[key], rtol=1e-6, atol=1e-7)
    design = {
        "version": 1, "seed": 20260919, "source": str(source.resolve()),
        "source_sha256": hashes, "test_opened": False,
        "pool": "original train + validation only; original test is excluded",
        "kkt_threshold": threshold, "kkt_threshold_rule": "original TRAIN p95, not a convergence tolerance",
        "teacher_policy": meta["teacher_policy"], "input_contract": meta["input_contract"],
        "common_normalization": {k: v.tolist() for k, v in norm.items()},
        "cautions": [
            "frame_iid is deliberately optimistic: adjacent and related samples may leak.",
            "KKT filtering is a sensitivity experiment, NOT proof that removed labels are wrong.",
            "All environment arms use exactly the same FULL validation rows.",
            "Different arms fit normalization on their own training rows only.",
            "Compare common-scale MSE and physical per-output MSE, not only own-normalized MSE.",
        ], "arms": {},
    }
    _atomic_npz(output / "pool.npz", data)
    for name, (a, b) in arms.items():
        _atomic_npz(output / f"{name}_indices.npz", {"train": a, "validation": b})
        design["arms"][name] = {
            "train": describe(take(data, a)), "validation": describe(take(data, b)),
            "overlap": overlap(data, a, b),
        }
    if hashes != {name: digest(source / name) for name in files}:
        raise RuntimeError("Source changed during preparation")
    design["prepared_sha256"] = {p.name: digest(p) for p in
                                [output / "pool.npz", *[output / f"{a}_indices.npz" for a in ARMS]]}
    write_json(output / "design.json", design)
    print(f"Prepared {len(data['X'])} development rows; original test untouched: {output}", flush=True)


def error_summary(pred_physical, data, norm, kkt_threshold):
    err = (pred_physical.astype(np.float64) - data["U"]) / np.asarray(norm["u_std"])
    squared = err ** 2
    row = squared.mean(1)
    def part(mask):
        if not np.any(mask):
            return {"samples": 0}
        return {"samples": int(np.sum(mask)), "common_normalized_mse": float(row[mask].mean()),
                "common_normalized_mse_per_output": squared[mask].mean(0).tolist(),
                "physical_mse_per_output": (squared[mask] * np.asarray(norm["u_std"]) ** 2).mean(0).tolist(),
                "total_sse_fraction": float(row[mask].sum() / max(row.sum(), 1e-30))}
    result = part(np.ones(len(row), dtype=bool))
    result["source"] = {str(int(k)): part(data["source_kind"] == k) for k in np.unique(data["source_kind"])}
    result["kkt_le_train_p95"] = part(data["cold_rti_kkt"] <= kkt_threshold)
    result["kkt_gt_train_p95"] = part(data["cold_rti_kkt"] > kkt_threshold)
    result["low_speed"] = part(np.linalg.norm(data["velocity"], axis=1) <= 3.5)
    result["top_1pct_sse_fraction"] = float(np.sort(row)[-max(1, len(row) // 100):].sum() / max(row.sum(), 1e-30))
    return result


def predict_physical(torch, model, data, norm, device="cpu"):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["X"]), 512):
            x = ((data["X"][start:start+512].astype(np.float32) - norm["x_mean"]) / norm["x_std"]).astype(np.float32)
            outputs.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    return np.concatenate(outputs) * norm["u_std"] + norm["u_mean"]


def nearest_report(train, query, norm, limit, seed):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    idx = stratified_rows(query["source_kind"], min(limit, len(query["X"])), rng)
    tx = (train["X"] - norm["x_mean"]) / norm["x_std"]
    qx = (query["X"][idx] - norm["x_mean"]) / norm["x_std"]
    tree = cKDTree(tx)
    distances, neighbors = tree.query(qx, k=min(64, len(tx)), workers=2)
    other = train["episode_id"][neighbors] != query["episode_id"][idx, None]
    found = other.any(1)
    positions = other.argmax(1)
    selected = neighbors[np.arange(len(idx)), positions][found]
    distance = distances[np.arange(len(idx)), positions][found]
    qindices = idx[found]
    differences = ((train["U"][selected] - query["U"][qindices]) / norm["u_std"]) ** 2
    if not len(selected):
        return {"queries": len(idx), "matched_other_episode": 0}
    nearest_tenth = distance <= np.quantile(distance, .1)
    worst = np.argsort(differences.mean(1) / np.maximum(distance, 1e-8))[-8:][::-1]
    return {
        "queries": len(idx), "matched_other_episode": int(found.sum()),
        "unmatched_within_64_neighbors": int((~found).sum()),
        "distance_p10_p50_p90_p99": np.quantile(distance, [.1, .5, .9, .99]).tolist(),
        "neighbor_label_difference_mse": float(differences.mean()),
        "nearest_10pct_label_difference_mse": float(differences[nearest_tenth].mean()),
        "per_output_label_difference_mse": differences.mean(0).tolist(),
        "examples": [{"query_local_row": int(qindices[j]), "train_local_row": int(selected[j]),
                      "distance": float(distance[j]), "label_difference_mse": float(differences[j].mean())}
                     for j in worst],
        "interpretation": "Nearby, NOT identical X; this is NOT an irreducible-error bound or proof of conflicting labels.",
    }


def audit(output, checkpoints, sample_count):
    import torch
    torch.set_num_threads(2)
    design = json.loads((output / "design.json").read_text(encoding="utf-8"))
    data = read_npz(output / "pool.npz")
    indices = read_npz(output / "environment_indices.npz")
    splits = {s: take(data, i) for s, i in indices.items()}
    norm = {k: np.asarray(v, dtype=np.float64) for k, v in design["common_normalization"].items()}
    report = {"test_opened": False, "sample_count_per_split": sample_count, "models": {},
              "data": {s: describe(d) for s, d in splits.items()}}
    for s, d in splits.items():
        report.setdefault("nearest_other_episode", {})[s] = nearest_report(splits["train"], d, norm, 1500, 90)
    kkt_edges = np.unique(np.quantile(splits["train"]["cold_rti_kkt"], [0, .5, .9, .95, .99, 1]))
    for checkpoint in checkpoints:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ck_norm = {k: np.asarray(v, dtype=np.float32) for k, v in ck["normalization"].items()}
        for k in norm:
            np.testing.assert_allclose(ck_norm[k], norm[k], atol=1e-6, rtol=1e-5)
        if ck.get("teacher_policy") != design["teacher_policy"]:
            raise ValueError("Checkpoint teacher policy differs")
        model = build_model(torch, depth=ck["depth"], width=ck["width"], input_dim=12).eval()
        model.load_state_dict(ck["model_state"])
        result = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
                  "epoch": ck["epoch"], "sampling": "same source-stratified rows for every model; seed 77"}
        for s, d in splits.items():
            idx = stratified_rows(d["source_kind"], min(sample_count, len(d["X"])), np.random.default_rng(77))
            sampled = take(d, idx)
            pred = predict_physical(torch, model, sampled, ck_norm)
            result[s] = error_summary(pred, sampled, norm, design["kkt_threshold"])
            result[s]["kkt_bins"] = {}
            bins = np.searchsorted(kkt_edges[1:-1], sampled["cold_rti_kkt"], side="right")
            for bucket in np.unique(bins):
                mask = bins == bucket
                result[s]["kkt_bins"][str(int(bucket))] = error_summary(pred[mask], take(sampled, mask), norm, design["kkt_threshold"])
            result[s]["kkt_bin_edges_from_train"] = kkt_edges.tolist()
            print(checkpoint.parent.name, s, result[s]["common_normalized_mse"], flush=True)
        report["models"][checkpoint.parent.name] = result
        write_json(output / "audit.json", report)
    if not checkpoints:
        write_json(output / "audit.json", report)
    print(f"Audit: {output / 'audit.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.action == "prepare":
        prepare(args.source.resolve(), args.output.resolve())
    else:
        audit(args.output.resolve(), args.checkpoint, args.samples)


if __name__ == "__main__":
    main()
