from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODEL_NAMES = (
    "depth10_width512",
    "depth10_width1024",
    "depth22_width512",
    "depth22_width1024",
    "depth34_width512",
    "depth34_width1024",
)

MODEL_FILES = (
    "best_model.pt",
    "controller_scripted.pt",
    "history.csv",
    "metrics.json",
    "model_spec.json",
    "normalization.json",
    "per_gate_metrics.csv",
    "per_output_metrics.csv",
)

EXPORT_FILES = (
    "weights_fp32.npz",
    "golden_vectors.npz",
    "model_spec.json",
    "normalization.json",
    "export_manifest.json",
)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(require(source), destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and package the six v6 Diverse-40k FALCON 12-D models for Ubuntu."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Package destination root (default: <project>/ubuntu_transfer).",
    )
    parser.add_argument("--golden-count", type=int, default=1000)
    parser.add_argument(
        "--reuse-exports",
        action="store_true",
        help="Reuse complete export directories instead of regenerating golden vectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(__file__).resolve().parent
    project_root = experiment_root.parents[1]
    run_root = experiment_root / "runs_full" / "seed_0"
    dataset_root = experiment_root / "dnn_dataset_falcon12_v6_diverse40k"
    output_root = (args.output_root or (project_root / "ubuntu_transfer")).resolve()
    export_program = project_root / "export_dnn_artifacts.py"
    handoff = experiment_root / "WINDOWS_UBUNTU_FALCON12_V6_DIVERSE40K_HANDOFF.md"

    if args.golden_count <= 0:
        raise ValueError("--golden-count must be positive")

    dataset_files = (
        "train.npz",
        "validation.npz",
        "test.npz",
        "normalization.json",
        "split_manifest.json",
    )
    for name in dataset_files:
        require(dataset_root / name)
    require(export_program)
    require(handoff)

    model_records: list[dict[str, object]] = []
    for model_id, model_name in enumerate(MODEL_NAMES, start=1):
        model_dir = run_root / model_name
        for name in MODEL_FILES:
            require(model_dir / name)

        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        spec = json.loads((model_dir / "model_spec.json").read_text(encoding="utf-8"))
        if not metrics.get("training_complete"):
            raise RuntimeError(f"training_complete is false: {model_name}")
        if spec.get("input_shape") != [1, 12] or spec.get("output_shape") != [1, 4]:
            raise RuntimeError(f"unexpected model shape: {model_name}")
        if spec.get("input_contract") != "falcon_paper_gate_relative_euler_12d_v1":
            raise RuntimeError(f"unexpected input contract: {model_name}")

        export_dir = model_dir / "export"
        export_complete = all((export_dir / name).exists() for name in EXPORT_FILES)
        if not args.reuse_exports or not export_complete:
            command = [
                sys.executable,
                str(export_program),
                "--model",
                str(model_dir / "best_model.pt"),
                "--data-dir",
                str(dataset_root),
                "--output-dir",
                str(export_dir),
                "--golden-count",
                str(args.golden_count),
            ]
            print(f"EXPORT {model_name}", flush=True)
            subprocess.run(command, cwd=project_root, check=True)
        for name in EXPORT_FILES:
            require(export_dir / name)

        model_records.append(
            {
                "model_id": model_id,
                "name": model_name,
                "depth": metrics["depth"],
                "width": metrics["width"],
                "parameters": metrics["parameters"],
                "best_epoch": metrics["best_epoch"],
                "epochs_completed": metrics["epochs_completed"],
                "termination_reason": metrics["termination_reason"],
                "test_normalized_mse": metrics["test_normalized"]["mse"],
                "test_physical_mse": metrics["test"]["mse"],
                "input_contract": spec["input_contract"],
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"falcon12_v6_diverse40k_six_dnn_seed0_{timestamp}"
    staging = output_root / package_name
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    copy_file(handoff, staging / handoff.name)
    for name in dataset_files:
        copy_file(dataset_root / name, staging / "dataset" / name)

    metadata_files = (
        "README.md",
        "paper_core_config.json",
        "preflight_audit.json",
        "core_diversity_audit.json",
    )
    for name in metadata_files:
        copy_file(experiment_root / name, staging / "metadata" / name)
    copy_file(run_root.parent / "capacity_summary.json", staging / "metadata" / "capacity_summary.json")

    core_root = experiment_root / "paper_core_40k_falcon12_v6_diverse"
    for name in (
        "collection_config.json",
        "environments.json",
        "initial_states.json",
        "manifest.json",
        "summary.txt",
    ):
        copy_file(core_root / name, staging / "metadata" / "paper_core" / name)

    for model_name in MODEL_NAMES:
        source_dir = run_root / model_name
        destination_dir = staging / "models" / model_name
        for name in MODEL_FILES:
            copy_file(source_dir / name, destination_dir / name)
        for name in EXPORT_FILES:
            copy_file(source_dir / "export" / name, destination_dir / "export" / name)

    source_files = (
        project_root / "dnn_model_utils.py",
        project_root / "export_dnn_artifacts.py",
        project_root / "state_adapter.py",
        project_root / "train_dnn_controller.py",
        project_root / "run_single_drone_dnn_gates.py",
        project_root / "falcon_paper_fidelity" / "contracts.py",
        experiment_root / "collect_paper_core.py",
        experiment_root / "prepare_dataset.py",
        experiment_root / "train_six_models.py",
        experiment_root / "summarize_results.py",
        experiment_root / "test_six_models_air_sim.ps1",
    )
    for source in source_files:
        relative = source.relative_to(project_root)
        copy_file(source, staging / "source" / relative)

    flight_root = experiment_root / "flight_tests_v6"
    if flight_root.exists():
        for report in sorted(flight_root.glob("*.json")):
            copy_file(report, staging / "flight_tests_v6" / report.name)

    package_manifest = {
        "package_name": package_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "FALCON 12-D v6 Diverse-40k six-DNN experiment",
        "input_contract": "falcon_paper_gate_relative_euler_12d_v1",
        "input_dimensions": 12,
        "output_dimensions": 4,
        "seed": 0,
        "dataset_samples": 40000,
        "split_samples": {"train": 32000, "validation": 4000, "test": 4000},
        "nominal_speed_mps": 5.0,
        "primary_metric": "test_normalized_mse",
        "strict_capacity_monotonicity": False,
        "models": model_records,
        "warning": (
            "This package uses wrapped Euler gate-attitude errors. It is not input-compatible "
            "with the previous rotation-vector FLCN/v2 model package even though both are 12-D."
        ),
    }
    (staging / "package_manifest.json").write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for file_path in sorted(path for path in staging.rglob("*") if path.is_file()):
        relative = file_path.relative_to(staging).as_posix()
        checksum_lines.append(f"{sha256(file_path)}  {relative}")
    (staging / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    archive = Path(shutil.make_archive(str(staging), "zip", root_dir=staging.parent, base_dir=staging.name))
    print(json.dumps({
        "status": "PASS",
        "package_directory": str(staging),
        "zip": str(archive),
        "zip_sha256": sha256(archive),
        "models": len(model_records),
        "golden_count_per_model": args.golden_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
