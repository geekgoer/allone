"""Create a new immutable-by-convention M3 transfer snapshot; never overwrite sources."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

from verify_frozen_bundle import digest


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "controller_architecture_study/results/incremental_supernet_237d_m2_lsq_v3"
BUNDLE = ROOT / "ubuntu_transfer/falcon_incremental_supernet_m3_v1_20261015"
CHECKPOINT = "e6100b0612b8be6ac0a78786632d96b653f7b30fd6cc4a3eb563dd4ecc323bdc"


def main():
    archive = BUNDLE.with_suffix(".zip")
    receipt = archive.with_suffix(".zip.sha256")
    if any(path.exists() for path in (BUNDLE, archive, receipt)):
        raise FileExistsError("freeze target already exists; preserve it and choose a new version explicitly")
    export = RUN / "export"
    report = json.loads((export / "export_report.json").read_text(encoding="utf-8"))
    spec = json.loads((export / "quantization_spec.json").read_text(encoding="utf-8"))
    if (digest(RUN / "best_model.pt") != CHECKPOINT or report["checkpoint_sha256"] != CHECKPOINT
            or spec["package_version"] != CHECKPOINT):
        raise ValueError("checkpoint/report/spec no longer match the approved candidate")
    if (report["status"] != "NUMERICS_PASS_QUALITY_NOT_MET" or not report["torch_integer_bit_exact"]
            or not report["maximum_quality_goal_met"] or not report["grid_quality_goal_met"]
            or report["switch_non_degradation_goal_met"] or report["test_read"]):
        raise ValueError("candidate status changed; review before freezing")
    for name, expected in report["artifact_sha256"].items():
        if digest(export / name) != expected:
            raise ValueError("export artifact changed: " + name)
    for name, expected in spec["package_file_sha256"].items():
        if digest(export / name) != expected:
            raise ValueError("quantization artifact changed: " + name)

    files = {"deployment/" + name: export / name for name in (*report["artifact_sha256"], "export_report.json")}
    for name in ("config.json", "completion.json", "initial_metrics.json", "history.json"):
        files["provenance/" + name] = RUN / name
    files["provenance/qat_best_model.pt"] = RUN / "best_model.pt"
    fp32 = ROOT / "controller_architecture_study/results/incremental_supernet_237d_v4_grid5000/export"
    for name in ("model_spec.json", "export_report.json"):
        files["provenance/fp32_" + name] = fp32 / name
    code = ROOT / "controller_architecture_study/incremental_supernet"
    for name in ("integer_reference.py", "verify_frozen_bundle.py"):
        files["reference/" + name] = code / name
    for path in sorted(code.glob("*.py")):
        files["source_snapshot/incremental_supernet/" + path.name] = path
    for name in ("convex_features.py", "fast_convex_preprocess.py", "convex_teacher.py"):
        files["source_snapshot/" + name] = ROOT / "controller_architecture_study" / name
    files["source_snapshot/control_adapter.py"] = ROOT / "control_adapter.py"
    docs = ROOT / "controller_architecture_study/docs"
    handoff = docs / "UBUNTU_INCREMENTAL_SUPERNET_M3_HANDOFF_20261015.md"
    files["README_UBUNTU_HANDOFF.md"] = handoff
    for name in ("INCREMENTAL_SUPERNET_IMPLEMENTATION_PLAN_20261012.md",
                 "INCREMENTAL_SUPERNET_ARCHITECTURE_20261012.md",
                 "DYNAMIC_CONTROL_SCHEDULING_FRAMEWORK_20261005.md",
                 "SCHEDULING_RESEARCH_LOG_20261005.md",
                 "M2_QUANTIZATION_DIAGNOSIS_RESEARCH_20261014.md"):
        files["background/" + name] = docs / name
    files["background/M2_LSQ_V3.md"] = code / "M2_LSQ_V3.md"
    files["background/FALCON_extracted_text.txt"] = ROOT / "tmp_falcon_text.txt"
    data = ROOT / "controller_architecture_study/results/decision_convex_recovery_v1/datasets/features_recovery"
    files["provenance/dataset_split_manifest.json"] = data / "split_manifest.json"

    # Hash before and after copying: freeze must not mix files changed by another process.
    records = {}
    for name, source in sorted(files.items()):
        expected = digest(source)
        target = BUNDLE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != expected or digest(source) != expected:
            raise ValueError("source changed during freeze: " + str(source))
        records[name] = {"bytes": target.stat().st_size, "sha256": expected,
                         "source_relative_path": source.relative_to(ROOT).as_posix()}
    manifest = {"bundle_id": BUNDLE.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_sha256": CHECKPOINT, "source_run": RUN.relative_to(ROOT).as_posix(),
                "approval_scope": "M3_FUNCTIONAL_PROTOTYPE_CANDIDATE_WITH_DOCUMENTED_QUALITY_EXCEPTION",
                "model_quality_status": report["status"], "all_quality_goals_met": False,
                "FPGA_verified": False, "flight_oscillation_reduction_verified": False,
                "test_read": False, "dataset_arrays_included": False,
                "files": records}
    (BUNDLE / "freeze_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                packed.write(path, path.relative_to(BUNDLE.parent).as_posix())
    with zipfile.ZipFile(archive) as packed:
        if packed.testzip() is not None:
            raise ValueError("archive CRC check failed")
    archive_hash = digest(archive)
    receipt.write_text(archive_hash + "  " + archive.name + "\n", encoding="ascii")
    print(json.dumps({"bundle": str(BUNDLE), "archive": str(archive), "sha256": archive_hash,
                      "archive_bytes": archive.stat().st_size, "files": len(records),
                      "source_files_modified": False, "quality_status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
