"""Package the current six 5 m/s FALCON 12-D deployment models."""

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
    (1, "depth10_width512", "runs_retrain_depth10_width512_100_seed123"),
    (2, "depth10_width1024", "runs_progressive_200_to_700_batch512_seed42/depth10_width1024"),
    (3, "depth22_width512", "runs_progressive_200_to_700_batch512_seed42/depth22_width512"),
    (4, "depth22_width1024", "runs_progressive_200_to_700_batch512_seed42/depth22_width1024"),
    (5, "depth34_width512", "runs_progressive_200_to_700_batch512_seed42/depth34_width512"),
    (6, "depth34_width1024", "runs_resume_depth34_width1024_lr5e-5_300_seed42"),
)

MODEL_FILES = (
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
    if not path.is_file():
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
    return {key: np.asarray(payload[key], dtype=np.float32) for key in ("x_mean", "x_std", "u_mean", "u_std")}


def verify_normalization(reference: dict[str, np.ndarray], path: Path) -> None:
    candidate = normalization_arrays(path)
    for key, expected in reference.items():
        if candidate[key].shape != expected.shape or not np.array_equal(candidate[key], expected):
            raise RuntimeError(f"normalization mismatch: {path} ({key})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--golden-count", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.golden_count <= 0:
        raise ValueError("--golden-count must be positive")

    experiment_root = Path(__file__).resolve().parent
    project_root = experiment_root.parents[1]
    dataset_root = experiment_root / "dnn_dataset"
    export_program = project_root / "export_dnn_artifacts.py"
    output_root = (args.output_root or (experiment_root / "ubuntu_transfer")).resolve()
    require(export_program)
    reference_normalization = normalization_arrays(require(dataset_root / "normalization.json"))
    for name in DATASET_FILES:
        require(dataset_root / name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"falcon_current_5ms_12d_six_{timestamp}"
    staging = output_root / package_name
    staging.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, object]] = []
    for model_id, model_name, relative_source in MODEL_SOURCES:
        source = experiment_root / relative_source
        for name in MODEL_FILES:
            require(source / name)
        metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        spec = json.loads((source / "model_spec.json").read_text(encoding="utf-8"))
        if int(metrics.get("input_dimensions", -1)) != 12 or spec.get("input_shape") != [1, 12] or spec.get("output_shape") != [1, 4]:
            raise RuntimeError(f"invalid model contract: {model_name}")
        verify_normalization(reference_normalization, source / "normalization.json")

        destination = staging / "models" / model_name
        for name in MODEL_FILES:
            copy_file(source / name, destination / name)
        export_dir = destination / "export"
        command = [
            sys.executable,
            str(export_program),
            "--model",
            str(source / "best_model.pt"),
            "--data-dir",
            str(dataset_root),
            "--output-dir",
            str(export_dir),
            "--golden-count",
            str(args.golden_count),
        ]
        print(f"EXPORT {model_id}: {model_name}", flush=True)
        subprocess.run(command, cwd=project_root, check=True)
        for name in EXPORT_FILES:
            require(export_dir / name)

        export_manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
        if int(export_manifest.get("input_dimensions", -1)) != 12:
            raise RuntimeError(f"invalid export contract: {model_name}")
        records.append(
            {
                "model_id": model_id,
                "name": model_name,
                "source_directory": relative_source,
                "depth": int(metrics["depth"]),
                "width": int(metrics["width"]),
                "parameters": int(metrics["parameters"]),
                "seed": int(metrics.get("seed", -1)),
                "epochs_completed": int(metrics["epochs_completed"]),
                "best_epoch": int(metrics["best_epoch"]),
                "validation_normalized_mse": float(metrics["validation_normalized"]["mse"]),
                "test_normalized_mse": float(metrics["test_normalized"]["mse"]),
                "validation_physical_mse": float(metrics["validation"]["mse"]),
                "test_physical_mse": float(metrics["test"]["mse"]),
                "numpy_pytorch_max_abs_error": float(export_manifest["numpy_pytorch_max_abs_error"]),
            }
        )

    for name in DATASET_FILES:
        copy_file(dataset_root / name, staging / "dataset" / name)
    for source_name in ("dnn_model_utils.py", "export_dnn_artifacts.py"):
        copy_file(project_root / source_name, staging / "source" / source_name)

    package_manifest = {
        "package_name": package_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "Current six 5 m/s FALCON Track-P 12-D DNN deployment package",
        "input_contract": "falcon_track_p_gate_relative_12d_v1",
        "input_definition": "[p_gate-p, shortest_rotation_vector(q_gate*inverse(q_vehicle)), velocity, acceleration] in AirSim NED",
        "output_definition": "[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]",
        "input_dimensions": 12,
        "output_dimensions": 4,
        "nominal_speed_mps": 5.0,
        "control_period_seconds": 0.05,
        "dataset_samples": 480000,
        "split_samples": {"train": 384000, "validation": 48000, "test": 48000},
        "golden_count_per_model": args.golden_count,
        "old_first_and_last_models_excluded": True,
        "models": records,
    }
    (staging / "package_manifest.json").write_text(json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Current FALCON 5 m/s 12-D six-model Ubuntu handoff

This package contains the current six models selected for redeployment. The old first model and old last model are deliberately excluded.

## Model selection

| ID | Model | Source | Best epoch | Validation normalized MSE | Test normalized MSE | Validation physical MSE | Test physical MSE |
|---:|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['model_id']} | {r['name']} | `{r['source_directory']}` | {r['best_epoch']} | {r['validation_normalized_mse']:.8e} | {r['test_normalized_mse']:.8e} | {r['validation_physical_mse']:.8e} | {r['test_physical_mse']:.8e} |"
        for r in records
    ) + """

## Input and output contract

The network input is exactly 12 values in AirSim NED coordinates:

```text
X[0:3]  = p_gate - p_vehicle
X[3:6]  = shortest rotation vector of q_gate * inverse(q_vehicle)
X[6:9]  = current linear velocity
X[9:12] = current linear acceleration
```

The output order is:

```text
[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]
```

Inference must use the included train-only normalization:

```text
x_norm = (x_physical - x_mean) / x_std
u_norm = DNN(x_norm)
u_physical = u_norm * u_std + u_mean
```

Do not use the old package's normalization, INT8 calibration, ELF, or golden vectors. Rebuild all deployment artifacts from this package.

## Package contents

- `models/<name>/controller_scripted.pt`: current TorchScript FP32 model.
- `models/<name>/export/weights_fp32.npz`: framework-neutral FP32 weights for rebuilding and quantization.
- `models/<name>/export/golden_vectors.npz`: FP32 parity vectors.
- `models/<name>/export/normalization.json`: model normalization.
- `dataset/train.npz`: calibration source; use only this split for INT8 calibration.
- `dataset/validation.npz` and `dataset/test.npz`: offline verification only.
- `SHA256SUMS.txt`: transfer integrity check.

## Ubuntu deployment sequence

1. Run `sha256sum -c SHA256SUMS.txt` after unpacking.
2. Rebuild each FP32 network from `export/weights_fp32.npz` in layer order. Hidden layers use ReLU; the final layer has no activation.
3. Verify each model against `export/golden_vectors.npz` before quantization.
4. Calibrate INT8 using `dataset/train.npz` only.
5. Compile the actual target DPU architecture and run board-side golden-vector verification for all six models.

The reported normalized MSE is the standardized-output MSE used by this training run. The physical MSE in the table is the directly interpretable raw-output metric. Neither should be silently presented as an exact numerical reproduction of the paper's Table II unless the paper's metric definition and data protocol are matched exactly.
"""
    (staging / "README_UBUNTU.md").write_text(readme, encoding="utf-8")

    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(staging).as_posix()}"
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    ]
    (staging / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    archive = Path(shutil.make_archive(str(staging), "zip", root_dir=staging.parent, base_dir=staging.name))
    print(json.dumps({"status": "PASS", "package_directory": str(staging), "zip": str(archive), "zip_sha256": sha256(archive), "models": len(records)}, indent=2))


if __name__ == "__main__":
    main()
