"""Export existing independent 237D models for Ubuntu/DPU experiments; never train."""

from collections import Counter
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .pool237_portable import predict_raw
from .verify_pool237_bundle import digest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "controller_architecture_study/results"
DATA = RESULTS / "decision_convex_recovery_v1/datasets/features_recovery"
DEST = ROOT / "ubuntu_transfer/falcon_independent_237d_dpu_pool_v1_20261016"
CONTRACT = "decision_convex_local_relative_pv_237d_v1"
EXPECTED = {
    "train.npz": "974e26269053bc9bda664c125332b69623818b8c0b9189470275fb32d37eb1f8",
    "validation.npz": "62da58b5ded2cfbf79e2f00d7ac526fad18b8004a936f2fe8073cb1ed3919a18",
    "normalization.json": "fe9c44ee80aef1cf97138536a66a4ee3a126f459f2ef963370db354ada8f19ac",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy(source, name):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = digest(source)
    shutil.copy2(source, target)
    if digest(target) != expected or digest(source) != expected:
        raise ValueError("source changed during copy: " + str(source))
    return name


def candidates():
    rows = []
    for group, parent in (("main18", RESULTS / "decision_convex_recovery_scenic_falcon18_v1/runs"),
                          ("residual600", RESULTS / "decision_convex_residual4_v1/runs")):
        rows.extend((group + "_" + p.name, group, p) for p in sorted(parent.iterdir()) if (p / "best_model.pt").is_file())
    p = RESULTS / "decision_convex_residual33_long5000_v1/residual_mlp_h33_w1024_seed0"
    rows.append(("residual5000_" + p.name, "residual5000", p))
    for width in (128, 256, 512, 1024):
        p = RESULTS / ("decision_convex_recovery_v1/runs/features_recovery/plain_mlp_h4_w%d_seed0" % width)
        rows.append(("recovery200_" + p.name, "recovery200", p))
    if Counter(group for _, group, _ in rows) != {"main18": 18, "residual600": 4, "residual5000": 1, "recovery200": 4}:
        raise ValueError("unexpected candidate catalog; review before exporting")
    return rows


def collect_flights(entries):
    by_hash = {entry["source_checkpoint_sha256"]: entry for entry in entries}
    suite_path = RESULTS / "dynamic_complexity_237d_v1/joint_model_period_v1/summary.json"
    suite = read(suite_path)
    if suite["status"] != "COMPLETE" or suite["provenance"]["input_contract"] != CONTRACT:
        raise ValueError("unexpected historical flight suite")
    copy(suite_path, "evidence/joint_model_period_summary.json")
    for trial in suite["trials"]:
        path = Path(trial["report"])
        report = read(path)
        alias = trial["model"]
        proven = report["provenance"]["models"][alias]
        entry = by_hash.get(proven["checkpoint_sha256"])
        if entry is None:
            raise ValueError("flight checkpoint no longer matches this catalog: " + alias)
        for name, expected in trial["artifact_sha256"].items():
            if digest(path.parent / name) != expected:
                raise ValueError("historical flight artifact changed: " + str(path.parent / name))
        if not (trial["eligible"] and report["mode"] == "dnn" and report["model_selection"] == alias
                and report["flight_passed"] and report["realtime_passed"]
                and report["acceptance_passed"] and not report["errors"]
                and all(report["flight"][v]["crossings"] == 5 for v in ("Drone1", "Drone2", "Drone3"))):
            raise ValueError("historical candidate was not an accepted full DNN flight: " + trial["case"])
        evidence = copy(path, "evidence/hashed_flights/" + trial["case"] + "/report.json")
        entry["flight_status"] = "HASH_MATCHED_THREE_DRONE_PASS"
        entry["flight_evidence"].append({"report": evidence, "period_ms": trial["period_ms"],
            "checkpoint_hash_matched": True, "nominal_speed_mps": report["nominal_speed_mps"],
            "environment_id": report["environment_id"], "route_index": report["route_index"],
            "valid_crossings": 15, "tracking_mse_m2": trial["tracking_mse_m2"]})
        if trial["period_ms"] == 50:
            for name in ("requests.jsonl", "flight_trace.json"):
                copy(path.parent / name, "evidence/hashed_flights/" + trial["case"] + "/" + name)
    for entry in entries:
        run = Path(entry["source_checkpoint"]).parent
        for path in sorted((run / "flight").glob("*.json")):
            report = read(path)
            if (report.get("status") == "PASS" and report.get("controller") == "convex_237D_DNN_only"
                    and report.get("checkpoint") == entry["source_checkpoint"]):
                target = copy(path, "evidence/path_only_flights/" + entry["id"] + "/" + path.name)
                if entry["flight_status"] == "NOT_FOUND":
                    entry["flight_status"] = "PATH_ONLY_HISTORICAL_PASS_NO_CHECKPOINT_HASH"
                entry["flight_evidence"].append({"report": target, "checkpoint_hash_matched": False,
                    "disturbance": report.get("disturbance", "none"), "valid_crossings": report.get("valid_crossings"),
                    "environment_id": report["environment_id"], "route_index": report["route_index"]})


def main():
    import torch
    from .models.factory import build_model

    archive = DEST.with_suffix(".zip")
    if DEST.exists() or archive.exists() or archive.with_suffix(".zip.sha256").exists():
        raise FileExistsError("transfer version exists; do not overwrite the frozen bundle")
    torch.set_num_threads(1)
    rows = candidates()
    for name, expected in EXPECTED.items():
        if digest(DATA / name) != expected:
            raise ValueError("dataset changed: " + name)
    norm = read(DATA / "normalization.json")
    if norm["input_contract"] != CONTRACT:
        raise ValueError("wrong input contract")
    rng = np.random.default_rng(237016)
    with np.load(DATA / "train.npz", allow_pickle=False) as saved:
        family = saved["family_id"]
        indices = np.concatenate([rng.choice(np.flatnonzero(family == f), 512, replace=False) for f in range(8)])
        shared = DEST / "shared"
        shared.mkdir(parents=True)
        np.savez_compressed(shared / "calibration_train_4096.npz", X=saved["X"][indices], U=saved["U"][indices],
                            family_id=family[indices], source_row_indices=indices)
    with np.load(DATA / "validation.npz", allow_pickle=False) as saved:
        family = saved["family_id"]
        golden_indices = np.concatenate([np.flatnonzero(family == f)[np.linspace(0, np.count_nonzero(family == f)-1, 4, dtype=int)] for f in range(8)])
        raw = saved["X"][golden_indices].astype(np.float32)
        teacher = saved["U"][golden_indices].astype(np.float32)
    copy(DATA / "normalization.json", "shared/normalization.json")
    copy(DATA / "validation.npz", "shared/validation.npz")
    copy(DATA / "split_manifest.json", "shared/dataset_split_manifest.json")
    write(shared / "calibration_manifest.json", {"source": "TRAIN_ONLY", "source_sha256": EXPECTED["train.npz"],
          "samples": 4096, "samples_per_family": 512, "seed": 237016, "row_indices": indices.tolist(),
          "golden_source": "validation", "golden_row_indices": golden_indices.tolist(), "test_read": False})
    inputs = (raw - np.asarray(norm["x_mean"], dtype=np.float32)) / np.asarray(norm["x_std"], dtype=np.float32)
    entries = []
    for i, (identifier, group, run) in enumerate(rows, 1):
        print("Export %d/%d %s" % (i, len(rows), identifier), flush=True)
        config, completion = read(run / "config.json"), read(run / "completion.json")
        if config.get("input_dim") != 237 or config.get("input_contract") != CONTRACT:
            raise ValueError("not a compatible 237D controller: " + str(run))
        if config["dataset_file_sha256"]["normalization.json"] != EXPECTED["normalization.json"]:
            raise ValueError("normalization mismatch")
        checkpoint = run / "best_model.pt"
        original_hash = digest(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(torch, model_type=payload["model_type"], input_dim=237, output_dim=4,
                            hidden_layers=payload["hidden_layers"], hidden_width=payload["hidden_width"])
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        arrays = {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
        if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("expected finite FP32 model weights")
        directory = "models/" + identifier
        target = DEST / directory
        target.mkdir(parents=True)
        compact = {key: payload[key] for key in ("model_state", "model_type", "hidden_layers", "hidden_width", "input_dim", "epoch")}
        compact.update(source_checkpoint_sha256=original_hash, input_contract=CONTRACT, optimizer_state_included=False)
        torch.save(compact, target / "best_model.pt")
        np.savez_compressed(target / "weights_fp32.npz", **arrays)
        linear = [{"weight_key": name + ".weight", "bias_key": name + ".bias",
                   "in_features": module.in_features, "out_features": module.out_features}
                  for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
        spec = {"family": "independent_dense_controller_not_supernet", "model_type": payload["model_type"],
                "input_dim": 237, "output_dim": 4, "input_contract": CONTRACT,
                "hidden_layers": payload["hidden_layers"], "hidden_width": payload["hidden_width"],
                "linear_layers": linear, "residual_scale": 1 / math.sqrt(max(payload["hidden_layers"]-1, 1)),
                "residual_definition": "ReLU(x + residual_scale * Linear(x)); alpha NOT folded",
                "normalization_inside_model": False, "normalization_file": "../../shared/normalization.json",
                "input_to_network": "normalized_float32", "network_output": "normalized_float32_linear_4d",
                "physical_output": "normalized_output * u_std + u_mean; no AirSim sign/throttle conversion",
                "source_checkpoint_sha256": original_hash, "source_epoch": payload["epoch"],
                "numpy_golden_tolerance_physical": 1e-5, "DPU_compiled": False}
        with torch.inference_mode():
            output = model(torch.from_numpy(inputs)).numpy()
        physical = output * np.asarray(norm["u_std"], dtype=np.float32) + np.asarray(norm["u_mean"], dtype=np.float32)
        error = float(np.max(np.abs(predict_raw(arrays, spec, raw, norm) - physical)))
        if not np.isfinite(error) or error > spec["numpy_golden_tolerance_physical"]:
            raise ValueError("NumPy export discrepancy: %s %.9g" % (identifier, error))
        np.savez_compressed(target / "golden_vectors.npz", X_raw=raw, X_normalized=inputs,
                            U_fp32_normalized=output, U_fp32_physical=physical,
                            U_teacher_physical=teacher, validation_row_indices=golden_indices)
        reloaded = torch.load(target / "best_model.pt", map_location="cpu", weights_only=True)
        if any(not torch.equal(value, reloaded["model_state"][key]) for key, value in model.state_dict().items()):
            raise ValueError("compact checkpoint changed weight tensors")
        if digest(checkpoint) != original_hash:
            raise ValueError("source checkpoint changed during export")
        spec["numpy_pytorch_max_abs_physical_error"] = error
        spec["deployment_checkpoint_sha256"] = digest(target / "best_model.pt")
        write(target / "model_spec.json", spec)
        for name in ("config.json", "completion.json", "complexity.json", "offline_metrics.json"):
            if (run / name).exists():
                copy(run / name, directory + "/provenance/" + name)
        entries.append({"id": identifier, "cohort": group, "directory": directory,
            "model_type": payload["model_type"], "hidden_layers": payload["hidden_layers"],
            "hidden_width": payload["hidden_width"], "parameter_count": sum(value.size for value in arrays.values()),
            "mac_count": sum(value.size for key, value in arrays.items() if key.endswith("weight")),
            "validation_normalized_mse": completion["best_validation_normalized_mse"],
            "epochs_completed": completion["epochs_completed"], "best_epoch": completion["best_epoch"],
            "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": original_hash,
            "deployment_checkpoint_sha256": spec["deployment_checkpoint_sha256"],
            "suggested_kernel_name": "p237_" + original_hash[:12], "DPU_compiled": False,
            "flight_status": "NOT_FOUND", "flight_evidence": []})
        del arrays, compact, reloaded, model, payload
        gc.collect()
    collect_flights(entries)
    registry = {"family": "independent_dense_controllers_not_supernet", "input_dim": 237, "output_dim": 4,
                "input_contract": CONTRACT, "normalization_sha256": EXPECTED["normalization.json"],
                "models": entries, "test_read": False, "new_airsim_flight_started": False}
    write(DEST / "model_registry.json", registry)
    audit = {"statuses": dict(Counter(entry["flight_status"] for entry in entries)),
             "note": "No evidence found does not prove no flight ever occurred; path-only evidence is not a checkpoint hash proof.",
             "models": [{key: entry[key] for key in ("id", "source_checkpoint_sha256", "flight_status", "flight_evidence")} for entry in entries]}
    write(DEST / "evidence/flight_audit.json", audit)
    for name in ("pool237_portable.py", "verify_pool237_bundle.py"):
        copy(ROOT / "controller_architecture_study" / name, "reference/" + name)
    copy(ROOT / "controller_architecture_study/test_pool237_missing.ps1", "windows/test_pool237_missing.ps1")
    for name in ("convex_features.py", "fast_convex_preprocess.py", "convex_teacher.py"):
        copy(ROOT / "controller_architecture_study" / name, "source_snapshot/" + name)
    for name in ("plain_mlp.py", "residual_mlp.py", "factory.py"):
        copy(ROOT / "controller_architecture_study/models" / name, "source_snapshot/models/" + name)
    copy(ROOT / "control_adapter.py", "source_snapshot/control_adapter.py")
    copy(ROOT / "controller_architecture_study/docs/UBUNTU_237D_DPU_MODEL_POOL_EXPERIMENT_HANDOFF_20261016.md", "README_UBUNTU_237D_POOL.md")
    copy(ROOT / "controller_architecture_study/docs/UBUNTU_INCREMENTAL_SUPERNET_M3_HANDOFF_20261015.md", "background/SUPERNET_M3_HANDOFF_REFERENCE_ONLY.md")
    records = {p.relative_to(DEST).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(DEST.rglob("*")) if p.is_file()}
    write(DEST / "bundle_manifest.json", {"bundle_id": DEST.name, "created_utc": datetime.now(timezone.utc).isoformat(),
          "family": registry["family"], "models": len(entries), "files": records,
          "source_dataset_sha256": EXPECTED, "optimizer_states_removed_from_derived_checkpoints": True,
          "source_checkpoints_modified": False, "test_read": False, "DPU_compiled": False})
    print("Creating ZIP; existing source checkpoints and the M3 bundle were not modified.", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as packed:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                compression = zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED
                packed.write(path, path.relative_to(DEST.parent).as_posix(), compress_type=compression)
    with zipfile.ZipFile(archive) as packed:
        if packed.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
    archive_hash = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(archive_hash + "  " + archive.name + "\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "archive_bytes": archive.stat().st_size, "sha256": archive_hash,
          "models": len(entries), "flight_audit": audit["statuses"], "test_read": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
