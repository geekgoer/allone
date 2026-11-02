"""Package the completed six-model 12-D cold-RTI run for Ubuntu/Vitis-AI.

The package is built from the progressive 200/300/.../700 epoch run, not
from the 300-D experiment and not from any older 12-D run.  Existing package
paths are never overwritten; use a new --output-root for a new package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATASET_NAME = "dnn_dataset_v4_cold_rti_labels_v1"
RUN_NAME = "runs_cold_rti_progressive_200_to_700_batch512_seed0"
PACKAGE_NAME = "falcon_cold_rti_12d_six_models_seed0"

MODELS = (
    ("depth10_width512", 10, 512, 200),
    ("depth22_width512", 22, 512, 300),
    ("depth10_width1024", 10, 1024, 400),
    ("depth34_width512", 34, 512, 500),
    ("depth22_width1024", 22, 1024, 600),
    ("depth34_width1024", 34, 1024, 700),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_source(dataset: Path, runs: Path) -> dict:
    manifest = _load_json(dataset / "split_manifest.json")
    if int(manifest.get("input_dimensions", -1)) != 12:
        raise ValueError("source dataset is not 12-D")
    if manifest.get("input_contract") != "falcon_track_p_trajectory_relative_12d_v1":
        raise ValueError("unexpected 12-D input contract")
    if manifest.get("teacher_policy") != "fully_initialized_acado_then_exactly_one_rti_step_v1":
        raise ValueError("source dataset is not the cold-RTI label set")

    required_data = (
        "train.npz",
        "validation.npz",
        "test.npz",
        "normalization.json",
        "split_manifest.json",
    )
    missing = [name for name in required_data if not (dataset / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset is incomplete: {missing}")

    for name, depth, width, expected_epochs in MODELS:
        run = runs / "seed_0" / name
        metrics_path = run / "metrics.json"
        metrics = _load_json(metrics_path)
        if metrics.get("training_complete") is not True:
            raise ValueError(f"{name}: training_complete is not true")
        if int(metrics.get("input_dimensions", -1)) != 12:
            raise ValueError(f"{name}: model is not 12-D")
        if int(metrics.get("depth", -1)) != depth or int(metrics.get("width", -1)) != width:
            raise ValueError(f"{name}: architecture metadata mismatch")
        if int(metrics.get("epochs_completed", -1)) != expected_epochs:
            raise ValueError(
                f"{name}: expected {expected_epochs} epochs, "
                f"got {metrics.get('epochs_completed')}"
            )
        for required in ("best_model.pt", "controller_scripted.pt", "model_spec.json", "normalization.json", "metrics.json"):
            if not (run / required).is_file():
                raise FileNotFoundError(f"{name}: missing {required}")
    return manifest


def _handoff_text() -> str:
    model_lines = "\n".join(
        f"- `{name}`: depth={depth}, width={width}, fixed epochs={epochs}"
        for name, depth, width, epochs in MODELS
    )
    return f"""# FALCON 12-D cold-RTI 六模型 Ubuntu/Vitis-AI 交接

## 1. 交接范围

本包来自 Windows 端：

```text
experiments/falcon_10ms_hybrid_160k_v1/
  dnn_dataset_v4_cold_rti_labels_v1/
  runs_cold_rti_progressive_200_to_700_batch512_seed0/
```

这是 12-D、4-D 输出的六个 FC/ReLU DNN。不要与 300-D MPC-complete
实验或旧的 25-D 模型混用。

{model_lines}

## 2. 输入输出契约

输入是归一化前的 12-D 物理量：

```text
0..2   reference_position_error_x/y/z
3..5   reference_attitude_error_x/y/z
6..8   reference_velocity_error_x/y/z
9..11  reference_acceleration_x/y/z
```

坐标系是 AirSim NED。输出顺序固定为：

```text
0 collective_thrust_acceleration
1 body_rate_x
2 body_rate_y
3 body_rate_z
```

推理公式：

```text
x_norm = (x_physical - x_mean) / x_std
y_norm = DNN(x_norm)
u_physical = y_norm * u_std + u_mean
```

`controller_scripted.pt` 不包含归一化。每个模型目录中的
`export/normalization.json` 是同一份 12/4 维参数。

## 3. Ubuntu 端安装和校验

在解压后的包根目录执行：

```bash
sha256sum -c SHA256SUMS.txt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ubuntu.txt
python verify_package.py
```

`verify_package.py` 会检查六个模型的输入输出维度、层数、黄金向量和
NumPy/PyTorch 导出一致性。它不运行 AirSim，也不运行 MPC。

## 4. Vitis-AI 量化顺序

对每个 `models/<model_name>/export/` 单独处理：

1. 读取 `model_spec.json`；
2. 使用 `weights_fp32.npz` 构建对应的 FC/ReLU FP32 图；
3. 用 `golden_vectors.npz` 验证 FP32 输出；
4. 使用 `calibration/train.npz` 中的 12-D `X` 构造校准输入；
5. 量化到 INT8；
6. 对该模型单独保存量化器、编译产物和日志；
7. 用同一个 `golden_vectors.npz` 比较 FP32 与 INT8 输出；
8. 再进行目标板 DPU 编译和板端 1000 条黄金向量测试。

不要用测试集做量化校准，也不要让六个模型共享同一个量化输出目录。

推荐的板端验收阈值由 Ubuntu/Vitis-AI 工程现有规范决定；至少应记录：

```text
模型名
输入维度
输出维度
FP32 与 INT8 的逐输出 MAE/RMSE
最大绝对误差
DPU-only latency
server total latency
```

## 5. 重要限制

- 这批模型的训练标签是 cold-RTI 教师标签；
- 六个模型只有 seed=0；
- 训练轮数是递增的 200/300/400/500/600/700，不是统一训练轮数；
- `best_model.pt` 是 Windows 端验证集最优 checkpoint；
- `latest_checkpoint.pt` 仅用于继续训练，不放入最小部署包；
- 不能把当前 DNN 输出直接称为与 MPC 数值完全相同；
- 上板前必须完成 FP32、INT8 和板端黄金向量验证。
"""


def _verify_script() -> str:
    return r'''"""Verify the six-model Ubuntu handoff package without AirSim."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = (
    ("depth10_width512", 10, 512),
    ("depth22_width512", 22, 512),
    ("depth10_width1024", 10, 1024),
    ("depth34_width512", 34, 512),
    ("depth22_width1024", 22, 1024),
    ("depth34_width1024", 34, 1024),
)


def main() -> None:
    import torch

    dataset = ROOT / "calibration"
    with np.load(dataset / "train.npz", allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        u = np.asarray(archive["U"], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 12 or u.shape != (len(x), 4):
        raise RuntimeError(f"invalid calibration arrays: X={x.shape}, U={u.shape}")

    for name, expected_depth, expected_width in MODELS:
        model_dir = ROOT / "models" / name
        spec = json.loads((model_dir / "export" / "model_spec.json").read_text())
        if spec["input_shape"] != [1, 12] or spec["output_shape"] != [1, 4]:
            raise RuntimeError(f"{name}: invalid IO shape")
        if spec["depth"] != expected_depth or spec["width"] != expected_width:
            raise RuntimeError(f"{name}: architecture mismatch")

        with np.load(model_dir / "export" / "golden_vectors.npz", allow_pickle=False) as golden:
            x_norm = np.asarray(golden["X_normalized"], dtype=np.float32)
            expected = np.asarray(golden["U_pytorch"], dtype=np.float32)
        scripted = torch.jit.load(str(model_dir / "controller_scripted.pt"), map_location="cpu").eval()
        with torch.no_grad():
            actual_norm = scripted(torch.from_numpy(x_norm)).numpy()
        normalization = json.loads((model_dir / "export" / "normalization.json").read_text())
        u_std = np.asarray(normalization["u_std"], dtype=np.float32)
        u_mean = np.asarray(normalization["u_mean"], dtype=np.float32)
        actual = actual_norm * u_std + u_mean
        max_error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"{name}: golden parity failed, max_abs_error={max_error}")
        print(f"{name}: input=12 output=4 golden_max_abs_error={max_error:.3e}")
    print(f"PASS: six models verified; calibration_samples={len(x)}")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "ubuntu_transfer",
    )
    args = parser.parse_args()

    dataset = HERE / DATASET_NAME
    runs = HERE / RUN_NAME
    manifest = _validate_source(dataset, runs)

    output_root = args.output_root.resolve()
    package = output_root / PACKAGE_NAME
    archive = output_root / f"{PACKAGE_NAME}.zip"
    if package.exists() or archive.exists():
        raise FileExistsError(
            f"package already exists: {package} or {archive}; "
            "choose a new --output-root"
        )
    package.mkdir(parents=True)

    # Keep all files needed for calibration and independent evaluation, but
    # omit the large optimizer checkpoints and the raw collection shards.
    for name in ("train.npz", "validation.npz", "test.npz", "normalization.json", "split_manifest.json", "label_report.json", "provenance.json", "preflight.json"):
        _copy(dataset / name, package / "calibration" / name)

    for name, depth, width, expected_epochs in MODELS:
        source = runs / "seed_0" / name
        destination = package / "models" / name
        for file_name in (
            "best_model.pt",
            "controller_scripted.pt",
            "model_spec.json",
            "normalization.json",
            "metrics.json",
            "history.csv",
            "per_output_metrics.csv",
            "per_gate_metrics.csv",
        ):
            _copy(source / file_name, destination / file_name)

        export_dir = destination / "export"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "export_dnn_artifacts.py"),
            "--model",
            str(source / "best_model.pt"),
            "--data-dir",
            str(dataset),
            "--output-dir",
            str(export_dir),
            "--golden-count",
            "1000",
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(f"export failed for {name} with exit code {result.returncode}")

    (package / "requirements-ubuntu.txt").write_text(
        "numpy>=1.26,<3\n"
        "torch>=2.1\n",
        encoding="utf-8",
    )
    (package / "README_UBUNTU.md").write_text(_handoff_text(), encoding="utf-8")
    (package / "verify_package.py").write_text(_verify_script(), encoding="utf-8")
    (package / "package_manifest.json").write_text(
        json.dumps(
            {
                "package": PACKAGE_NAME,
                "source_dataset": str(dataset.resolve()),
                "source_runs": str(runs.resolve()),
                "input_dimensions": 12,
                "output_dimensions": 4,
                "teacher_policy": manifest["teacher_policy"],
                "models": [
                    {"name": name, "depth": depth, "width": width, "epochs": epochs}
                    for name, depth, width, epochs in MODELS
                ],
                "golden_count_per_model": 1000,
                "omitted": ["latest_checkpoint.pt", "raw collection shards", "AirSim"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    files = sorted(path for path in package.rglob("*") if path.is_file())
    checksum_path = package / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(package).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    files = sorted(path for path in package.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in files:
            bundle.write(path, Path(PACKAGE_NAME) / path.relative_to(package))

    print(f"Package directory: {package}")
    print(f"Upload archive: {archive}")
    print(f"Archive bytes: {archive.stat().st_size}")


if __name__ == "__main__":
    main()
