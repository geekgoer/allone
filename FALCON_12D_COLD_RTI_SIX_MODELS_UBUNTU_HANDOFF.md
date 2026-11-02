# FALCON 12-D cold-RTI 六模型 Ubuntu 交接

本交接包对应 Windows 端最近完成的六个 12D DNN：

```text
experiments/falcon_10ms_hybrid_160k_v1/
  runs_cold_rti_progressive_200_to_700_batch512_seed0/
```

模型分别训练 200、300、400、500、600、700 轮，seed=0，batch size=512，学习率 `1e-4`。它们不是 300D 模型，也不是旧 25D 模型。

## Windows 打包

在项目根目录运行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $trainPy .\experiments\falcon_10ms_hybrid_160k_v1\package_cold_rti_six_models_ubuntu.py `
  --output-root .\experiments\falcon_10ms_hybrid_160k_v1\ubuntu_transfer
```

打包前脚本会检查：

- 六个模型是否全部 `training_complete=true`；
- 输入是否为 12D、输出是否为 4D；
- 六个模型是否分别完成 200/300/400/500/600/700 轮；
- 数据集是否为 cold-RTI 12D 数据集；
- 每个模型是否能导出 FP32 权重和 1000 条黄金向量。

输出：

```text
F:\FalconReproduction\airsim\_control\experiments\falcon_10ms_hybrid_160k_v1\ubuntu_transfer\falcon_cold_rti_12d_six_models_seed0.zip
```

将这个 ZIP 上传到 Ubuntu。

## Ubuntu 解压和验证

```bash
unzip falcon_cold_rti_12d_six_models_seed0.zip
cd falcon_cold_rti_12d_six_models_seed0

sha256sum -c SHA256SUMS.txt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ubuntu.txt
python verify_package.py
```

验证通过后，再进入 Vitis-AI 量化和 DPU 编译。

## 每个模型的量化输入

每个模型目录都包含：

```text
best_model.pt
controller_scripted.pt
model_spec.json
normalization.json
metrics.json
history.csv
export/weights_fp32.npz
export/model_spec.json
export/normalization.json
export/golden_vectors.npz
export/export_manifest.json
```

校准数据位于：

```text
calibration/train.npz
```

校准只使用 `train.npz`，不要用 `test.npz` 选择量化参数。六个模型必须使用各自独立的量化输出目录。

## 输入输出契约

12D 输入顺序：

```text
reference_position_error_x/y/z
reference_attitude_error_x/y/z
reference_velocity_error_x/y/z
reference_acceleration_x/y/z
```

4D 输出顺序：

```text
collective_thrust_acceleration
body_rate_x
body_rate_y
body_rate_z
```

归一化：

```text
x_norm = (x_physical - x_mean) / x_std
y_norm = DNN(x_norm)
u_physical = y_norm * u_std + u_mean
```

`controller_scripted.pt` 不包含归一化。Ubuntu、Caffe、DPU 服务和 Windows 测试端必须使用同一份 `normalization.json`。

## 上板顺序

1. 用 `weights_fp32.npz` 构建 FP32 FC/ReLU 网络；
2. 用 `golden_vectors.npz` 验证 NumPy/PyTorch/FP32 网络；
3. 使用 `calibration/train.npz` 生成 INT8 校准输入；
4. 六个模型分别量化；
5. 六个模型分别编译到板卡 DPU architecture；
6. 用黄金向量比较 FP32 与 INT8 输出；
7. 板端测量 DPU-only、服务总时间和通信时间；
8. 所有模型通过后再进入 FPGA shadow/closed-loop。

不要把 25D 模型的归一化、黄金向量、输入层或 UDP 请求格式混到这批 12D 模型中。
