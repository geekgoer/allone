# FALCON 12-D v6 Diverse-40k 六模型：Windows → Ubuntu/Vitis-AI 交接

## 1. 交接对象与当前结论

本文件交接 Windows 端已经训练完成的 **v6 Diverse-40k、12输入、4输出、六个 Linear+ReLU DNN**。六个目录均已确认存在以下产物：

```text
best_model.pt
controller_scripted.pt
normalization.json
model_spec.json
metrics.json
history.csv
per_output_metrics.csv
per_gate_metrics.csv
test_predictions.npz
```

六个 `metrics.json` 均满足 `training_complete=true`。但必须如实保留一个重要结论：本批六模型的测试误差比较接近，并且**没有**呈现“参数量越大、MSE越低”的严格单调趋势。因此，这批模型可以继续用于闭环飞行、量化误差、板端时延和调度可行性实验，但不能把当前结果表述为已经复现 FALCON Table II 的单调能力排序。

上一批已交给 Ubuntu、并已部署为 FLCN/v2 六模型的 v4 Independent-120k 模型使用的是另一个输入契约和另一套数据。v6 不是旧板端模型的直接替换文件；Ubuntu 必须重新做 FP32 重建、量化、编译和黄金验证。

## 2. 数据与教师控制器

权威实验目录：

```text
experiments/falcon_paper_fidelity_acado_v6_diverse40k/
```

数据协议：

- 10个初始状态；
- 40个环境；
- 400个 episode；
- 每个 episode 100步；
- 采样周期 `0.05 s`，即20 Hz；
- 总计40,000条 `(X,U)`；
- 固定规划器目标速度 `5.0 m/s`；
- 全局连续 C2 轨迹；
- 初始轨迹速度取实测初始速度，再连续收敛到5 m/s；
- teacher 为 `official_rpg_mpc_acado_qpoases`；
- ACADO/qpOASES 预测时域20步，预测间隔0.1 s；
- 确定性逐行80/10/10划分：32,000/4,000/4,000。

数据文件：

```text
dnn_dataset_falcon12_v6_diverse40k/train.npz
dnn_dataset_falcon12_v6_diverse40k/validation.npz
dnn_dataset_falcon12_v6_diverse40k/test.npz
dnn_dataset_falcon12_v6_diverse40k/normalization.json
dnn_dataset_falcon12_v6_diverse40k/split_manifest.json
```

`test.npz` 没有参与 early stopping、checkpoint 选择或量化校准。

## 3. v6 的12维输入契约

契约名称：

```text
falcon_paper_gate_relative_euler_12d_v1
```

输入严格按以下顺序排列：

```text
X[0:3]  = p_gate - p_vehicle
X[3:6]  = wrapped_euler_gate_minus_vehicle (roll, pitch, yaw)
X[6:9]  = current velocity (NED)
X[9:12] = current acceleration (NED)
```

完整特征名：

```text
gate_position_error_x
gate_position_error_y
gate_position_error_z
gate_euler_error_roll
gate_euler_error_pitch
gate_euler_error_yaw
velocity_x
velocity_y
velocity_z
acceleration_x
acceleration_y
acceleration_z
```

坐标系为 AirSim NED。姿态三维量是经过 `[-pi,pi]` 包裹的欧拉角差，不是四元数逐项相减，也不是上一版 v4 使用的最短旋转向量。

### 与上一版 v4/FLCN-v2 模型的关键差异

上一版 v4 的契约是：

```text
falcon_track_p_gate_relative_12d_v1
```

其中 `X[3:6]` 为相对四元数的最短旋转向量。v6 的 `X[3:6]` 为 wrapped Euler 差。两者虽然都是12维，数值语义不同，绝不能复用旧归一化、旧黄金向量或旧 INT8 模型。

## 4. 输出与推理公式

输出顺序固定为：

```text
U[0] = collective_thrust_acceleration
U[1] = body_rate_x
U[2] = body_rate_y
U[3] = body_rate_z
```

`controller_scripted.pt` 只包含网络本体。CPU、Caffe、DPU和UDP服务都必须执行：

```text
x_normalized = (x_physical - x_mean) / x_std
u_normalized = DNN(x_normalized)
u_physical   = u_normalized * u_std + u_mean
```

归一化长度必须为12/12/4/4。

## 5. 六模型的最终离线指标

`normalized MSE` 是训练与选模的主要指标：先用训练集统计量归一化四维控制标签，再对全部样本和四个输出求均方误差。

`physical MSE` 定义为：

```text
mean((U_pred_physical - U_true_physical)^2)
```

同样对全部测试样本和四个物理输出求平均。由于四个输出的单位和尺度不同，它适合本批模型之间比较，但不能在没有确认论文定义时直接当成论文 Table II 的 MSE。

| Model ID | 网络 | 参数量 | 完成方式 | 最佳epoch | Test normalized MSE | Test physical MSE |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 10×512 | 2,109,956 | 20,000 epochs | 19,798 | 0.0122859767 | 0.0009145516 |
| 2 | 10×1024 | 8,414,212 | 手动冻结最佳checkpoint | 14,863 | **0.0119507527** | **0.0008923745** |
| 3 | 22×512 | 5,261,828 | 1,000 epochs早停 | 102 | 0.0137794642 | 0.0013958449 |
| 4 | 22×1024 | 21,009,412 | 1,000 epochs早停 | 253 | 0.0125338975 | 0.0010645584 |
| 5 | 34×512 | 8,413,700 | 1,000 epochs早停 | 217 | 0.0137625160 | 0.0012355912 |
| 6 | 34×1024 | 33,604,612 | 1,000 epochs早停 | 144 | 0.0133377174 | 0.0012203164 |

汇总文件：

```text
runs_full/capacity_summary.json
```

其中：

```text
strict_all_comparisons_pass = false
smallest_to_largest_relative_gain = -0.08560497
```

也就是说，最大模型的 normalized MSE 比最小模型约高8.56%，而不是更低。宽度从512增到1024时，三个深度下均有小幅改善；深度增加则没有稳定改善。

## 6. Windows 闭环验收状态

六个模型的离线训练产物完整，但生成本交接文档时，v6 六模型还没有全部完成 AirSim 闭环实飞验收。Windows 已提供统一脚本：

```text
test_six_models_air_sim.ps1
```

它会根据 `model_spec.json` 自动使用 v6 的 Euler 姿态输入；也可用 `-Variant v4` 重跑上一版旋转向量模型作为对照。实飞失败不等于 UDP/DPU 出错，而首先应检查 DNN 输入语义、闭环分布漂移和训练标签是否能由当前12维输入唯一决定。

在 Ubuntu 开始长时间量化与编译前，建议先完成至少以下 Windows 检查：

1. v6 10×512 通过5门或记录明确失败模式；
2. v6 六模型逐一运行并保存 `flight_tests_v6/*.json`；
3. 不把离线 MSE 当成闭环成功率；
4. 若六模型均无法前进或均在相同阶段发散，先暂停板端替换并修正训练/输入契约。

## 7. Windows 权威模型目录

```text
experiments/falcon_paper_fidelity_acado_v6_diverse40k/
  runs_full/seed_0/
    depth10_width512/
    depth10_width1024/
    depth22_width512/
    depth22_width1024/
    depth34_width512/
    depth34_width1024/
```

每个目录中部署时优先使用：

```text
best_model.pt
controller_scripted.pt
normalization.json
model_spec.json
metrics.json
export/weights_fp32.npz
export/golden_vectors.npz
export/normalization.json
export/model_spec.json
export/export_manifest.json
```

`latest_checkpoint.pt` 只用于恢复训练，不放入部署包。

## 8. Ubuntu/Vitis-AI 后续顺序

对六个模型分别执行：

1. 断言输入维数12、输出维数4、输入契约为 `falcon_paper_gate_relative_euler_12d_v1`；
2. 用 `weights_fp32.npz` 重建 FP32 FC/ReLU 网络；
3. 用各自 `golden_vectors.npz` 验证 NumPy/Caffe FP32；
4. 仅用 `train.npz` 构建 calibration subset；
5. Vitis-AI 1.2.1 INT8量化；
6. INT8与FP32黄金向量比较；
7. 针对AXU3EGB真实DPU架构分别编译；
8. 板端六模型各跑1000条黄金向量；
9. 测量六模型 DPU、服务总时延和 UDP RTT；
10. Windows 更新为 Euler 12-D输入后，先 shadow，再单机FPGA闭环，最后三机静态组合。

现有 FLCN/v2 的12-float UDP帧格式和 model ID 1–6 可以保留，但下列内容必须换成 v6：

- 六个 INT8 网络；
- 六套模型归一化/黄金向量；
- Windows 输入构造中的姿态三维量；
- 板端模型包与验收报告。

## 9. SVM/调度实验注意事项

如果后续 SVM 使用 `c_i = 1 / beta_i` 作为能力值，必须先统一 `beta_i` 定义。建议同时保留：

```text
beta_normalized = test_normalized_mse
beta_physical   = test_physical_mse
```

不要为了制造单调趋势而挑选非最优 checkpoint。当前 v6 的能力排序和参数量不单调，应如实输入调度实验，或把“严格复现 Table II 的能力序列”和“v6 实测能力序列”作为两组实验分别报告。

## 10. 一句话交接结论

Windows 已生成完整的 v6 Diverse-40k 六个12-D FP32 DNN，但六模型 MSE 接近且不随容量单调改善；Ubuntu 可以据此重做六模型 Vitis-AI/DPU部署实验，但必须使用新的 Euler 12-D契约，并在正式替换旧FLCN/v2模型前等待 Windows 闭环实飞结果。
