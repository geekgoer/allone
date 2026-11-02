"""Export and package the six validated historical 5 m/s FALCON 12-D models."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


MODEL_SOURCES = (
    (1, "depth10_width512", "runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth10_width512"),
    (2, "depth10_width1024", "runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth10_width1024"),
    (3, "depth22_width512", "runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth22_width512"),
    (4, "depth22_width1024", "runs_server/paper_track_p_12d_5ms_v1_server_depth22_width1024"),
    (5, "depth34_width512", "runs_server/paper_track_p_12d_5ms_v1_server_depth34_width512"),
    (6, "depth34_width1024", "runs_server/paper_track_p_12d_5ms_v1_server_depth34_width1024"),
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

DATASET_FILES = (
    "train.npz",
    "validation.npz",
    "test.npz",
    "normalization.json",
    "split_manifest.json",
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


def normalization_arrays(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        # Training, TorchScript inference, export_dnn_artifacts.py and the DPU
        # contract all consume float32 normalization values.  The dataset JSON
        # retains more decimal digits than the per-run JSON, so comparing their
        # parsed float64 values would reject harmless float32 round-off.
        key: np.asarray(payload[key], dtype=np.float32)
        for key in ("x_mean", "x_std", "u_mean", "u_std")
    }


def verify_same_normalization(reference: dict[str, np.ndarray], candidate_path: Path) -> None:
    candidate = normalization_arrays(candidate_path)
    expected_shapes = {"x_mean": (12,), "x_std": (12,), "u_mean": (4,), "u_std": (4,)}
    for key, expected_shape in expected_shapes.items():
        if reference[key].shape != expected_shape or candidate[key].shape != expected_shape:
            raise RuntimeError(f"invalid normalization shape for {key}: {candidate_path}")
        if not np.array_equal(reference[key], candidate[key]):
            max_difference = float(np.max(np.abs(reference[key] - candidate[key])))
            raise RuntimeError(
                f"float32 normalization mismatch for {key}: {candidate_path}; "
                f"max_abs={max_difference}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Reuse complete per-model export directories instead of regenerating them.",
    )
    parser.add_argument(
        "--without-pytorch",
        action="store_true",
        help="Omit best_model.pt and controller_scripted.pt to make a smaller deployment-only archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.golden_count <= 0:
        raise ValueError("--golden-count must be positive")

    experiment_root = Path(__file__).resolve().parent
    project_root = experiment_root.parents[1]
    dataset_root = experiment_root / "dnn_dataset"
    output_root = (args.output_root or (project_root / "ubuntu_transfer")).resolve()
    export_program = project_root / "export_dnn_artifacts.py"
    handoff = experiment_root / "HISTORICAL_5MS_SIX_DNN_UBUNTU_HANDOFF.md"

    require(export_program)
    require(handoff)
    for name in DATASET_FILES:
        require(dataset_root / name)
    reference_normalization = normalization_arrays(dataset_root / "normalization.json")

    model_records: list[dict[str, object]] = []
    source_directories: dict[str, Path] = {}
    for model_id, model_name, relative_source in MODEL_SOURCES:
        model_dir = experiment_root / relative_source
        source_directories[model_name] = model_dir
        for name in MODEL_FILES:
            require(model_dir / name)

        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        spec = json.loads((model_dir / "model_spec.json").read_text(encoding="utf-8"))
        if int(metrics.get("input_dimensions", -1)) != 12:
            raise RuntimeError(f"unexpected input dimension: {model_name}")
        if spec.get("input_shape") != [1, 12] or spec.get("output_shape") != [1, 4]:
            raise RuntimeError(f"unexpected model shape: {model_name}")
        verify_same_normalization(reference_normalization, model_dir / "normalization.json")

        export_dir = model_dir / "export_ubuntu_historical_5ms"
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
            print(f"EXPORT model_id={model_id} {model_name}", flush=True)
            subprocess.run(command, cwd=project_root, check=True)
        for name in EXPORT_FILES:
            require(export_dir / name)

        export_manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
        if int(export_manifest.get("input_dimensions", -1)) != 12:
            raise RuntimeError(f"export input dimension mismatch: {model_name}")
        if int(export_manifest.get("golden_count", -1)) != args.golden_count:
            raise RuntimeError(
                f"golden count mismatch for {model_name}; rerun without --reuse-exports"
            )

        model_records.append(
            {
                "model_id": model_id,
                "name": model_name,
                "depth": int(metrics["depth"]),
                "width": int(metrics["width"]),
                "parameters": int(metrics["parameters"]),
                "best_epoch": int(metrics["best_epoch"]),
                "historical_test_physical_mse": float(metrics["test"]["mse"]),
                "historical_test_rmse": float(metrics["test"]["rmse"]),
                "source_directory": relative_source,
                "numpy_pytorch_max_abs_error": float(
                    export_manifest["numpy_pytorch_max_abs_error"]
                ),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"falcon_historical_5ms_12d_six_{timestamp}"
    staging = output_root / package_name
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    copy_file(handoff, staging / handoff.name)
    for name in DATASET_FILES:
        copy_file(dataset_root / name, staging / "dataset" / name)

    metadata_sources = (
        (experiment_root / "README.md", "experiment_README.md"),
        (experiment_root / "teacher_config.json", "teacher_config.json"),
        (experiment_root / "teacher_dataset" / "manifest.json", "teacher_manifest.json"),
        (
            experiment_root / "mse_reaudit_20260916",
            "mse_reaudit_20260916",
        ),
        (
            experiment_root / "physical_benchmark_v1" / "benchmark_config.json",
            "physical_benchmark_v1/benchmark_config.json",
        ),
        (
            experiment_root
            / "physical_benchmark_v1"
            / "results"
            / "common_state_full"
            / "common_state_summary.json",
            "physical_benchmark_v1/common_state_summary.json",
        ),
        (
            experiment_root
            / "physical_benchmark_v1"
            / "results"
            / "common_state_full"
            / "common_state_summary.csv",
            "physical_benchmark_v1/common_state_summary.csv",
        ),
        (
            experiment_root
            / "physical_benchmark_v1"
            / "results"
            / "closed_loop_full"
            / "closed_loop_summary.json",
            "physical_benchmark_v1/closed_loop_summary.json",
        ),
        (
            experiment_root
            / "physical_benchmark_v1"
            / "results"
            / "closed_loop_full"
            / "closed_loop_summary.csv",
            "physical_benchmark_v1/closed_loop_summary.csv",
        ),
    )
    for source, relative_destination in metadata_sources:
        if source.is_dir():
            shutil.copytree(require(source), staging / "metadata" / relative_destination)
        else:
            copy_file(source, staging / "metadata" / relative_destination)

    files_to_copy = MODEL_FILES
    if args.without_pytorch:
        files_to_copy = tuple(
            name for name in MODEL_FILES if name not in {"best_model.pt", "controller_scripted.pt"}
        )
    for model_name, source_dir in source_directories.items():
        destination_dir = staging / "models" / model_name
        for name in files_to_copy:
            copy_file(source_dir / name, destination_dir / name)
        for name in EXPORT_FILES:
            copy_file(
                source_dir / "export_ubuntu_historical_5ms" / name,
                destination_dir / "export" / name,
            )

    source_files = (
        project_root / "dnn_model_utils.py",
        project_root / "export_dnn_artifacts.py",
        project_root / "state_adapter.py",
        project_root / "track_p_12d_contract.py",
        project_root / "falcon_paper_fidelity" / "contracts.py",
    )
    for source in source_files:
        copy_file(source, staging / "source" / source.relative_to(project_root))

    package_manifest = {
        "package_name": package_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "historical 5 m/s Track-P 12-D six-model deployment set",
        "input_contract": "falcon_track_p_gate_relative_12d_v1",
        "legacy_contract_alias": "[p_gate-p, r_gate-r, velocity, acceleration] in AirSim NED",
        "attitude_representation": "shortest rotation vector",
        "input_dimensions": 12,
        "output_dimensions": 4,
        "nominal_speed_mps": 5.0,
        "control_period_seconds": 0.05,
        "dataset_samples": 480000,
        "split_samples": {"train": 384000, "validation": 48000, "test": 48000},
        "golden_count_per_model": args.golden_count,
        "pytorch_files_included": not args.without_pytorch,
        "model_id_mapping": {str(record["model_id"]): record["name"] for record in model_records},
        "models": model_records,
        "closed_loop_acceptance": {
            "episodes": 36,
            "successes": 36,
            "gate_crossings": 180,
            "deadline_misses": 0,
        },
        "warnings": [
            "The six historical test MSE values are not monotonic with network capacity.",
            "Training provenance is mixed: the first three local runs used resume/warm-start history, while the three server runs were fresh.",
            "The teacher is the project linear MPC, not the later ACADO/qpOASES teacher and not an exact unpublished PAMPC implementation.",
            "Do not reuse INT8, ELF, normalization, or golden files from the previous v4/v6 12-D packages.",
        ],
    }
    (staging / "package_manifest.json").write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for file_path in sorted(path for path in staging.rglob("*") if path.is_file()):
        checksum_lines.append(f"{sha256(file_path)}  {file_path.relative_to(staging).as_posix()}")
    (staging / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    archive = Path(
        shutil.make_archive(str(staging), "zip", root_dir=staging.parent, base_dir=staging.name)
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "package_directory": str(staging),
                "zip": str(archive),
                "zip_sha256": sha256(archive),
                "models": len(model_records),
                "golden_count_per_model": args.golden_count,
                "pytorch_files_included": not args.without_pytorch,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
