# FALCON 12-D cold-RTI 六模型 Ubuntu/Vitis-AI 交接

## 1. 交接范围

本包来自 Windows 端：

```text
experiments/falcon_10ms_hybrid_160k_v1/
  dnn_dataset_v4_cold_rti_labels_v1/
  runs_cold_rti_progressive_200_to_700_batch512_seed0/
```

这是 12-D、4-D 输出的六个 FC/ReLU DNN。不要与 300-D MPC-complete
实验或旧的 25-D 模型混用。

- `depth10_width512`: depth=10, width=512, fixed epochs=200
- `depth22_width512`: depth=22, width=512, fixed epochs=300
- `depth10_width1024`: depth=10, width=1024, fixed epochs=400
- `depth34_width512`: depth=34, width=512, fixed epochs=500
- `depth22_width1024`: depth=22, width=1024, fixed epochs=600
- `depth34_width1024`: depth=34, width=1024, fixed epochs=700

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
