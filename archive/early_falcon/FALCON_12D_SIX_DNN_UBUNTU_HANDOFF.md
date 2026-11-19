# FALCON 12-D 六模型：Windows → Ubuntu/Vitis-AI 交接

## 1. 本文档的地位

本文档交接的是 Windows 端已经完成训练的 **FALCON Track-P 风格 12 输入、4 输出、六种 FC/ReLU DNN**。

它修正并替代旧文档 `WINDOWS_WSL_TRACK_E_MULTI_DNN_HANDOFF.md` 中与以下内容有关的说明：

- 输入维数；
- 输入特征语义；
- 模型目录；
- 数据集目录；
- 归一化参数；
- 黄金向量；
- Caffe/Vitis-AI 输入层；
- FPGA UDP 请求载荷。

旧文档描述的是独立的 **Track-E 25-D** 工程线，不得把旧的25维权重、25维归一化、25维黄金向量或固定25个 `float32` 的 UDP 请求用于本批12维模型。旧Track-E结果可以保留作为扩展实验，但不应再称为论文12-D DNN复现结果。

## 2. 当前完成状态

Windows 已经在同一个数据划分、相同优化配置和 `seed=0` 下训练完成以下六个模型：

| FALCON候选 | 网络 | 参数量 | 最佳epoch | Test normalized MSE | Test physical MSE |
|---|---:|---:|---:|---:|---:|
| 1 | 34×1024 | 33,604,612 | 182 | 0.0107254805 | 0.0112059508 |
| 2 | 22×1024 | 21,009,412 | 337 | 0.0110553298 | 0.0124496203 |
| 3 | 10×1024 | 8,414,212 | 496 | 0.0144131072 | 0.0191935897 |
| 4 | 34×512 | 8,413,700 | 242 | 0.0144379297 | 0.0162441023 |
| 5 | 22×512 | 5,261,828 | 370 | 0.0149971303 | 0.0184547380 |
| 6 | 10×512 | 2,109,956 | 489 | 0.0169985741 | 0.0231817644 |

以训练和选模采用的主要指标 `test_normalized_mse` 排序，误差随参数量增加严格下降。最小模型到最大模型下降约 **36.9%**。34×512与10×1024的参数量几乎相同，因此两者 normalized MSE 也几乎相同，这是合理结果。

汇总证据：

```text
experiments/falcon_paper_fidelity_acado_v4_independent/
  runs_balanced_120k_seed0/capacity_summary.json
```

当前结果只包含一个训练种子。六个架构均已完成，可以进入论文复现中的六候选模型部署和能力/时延测量；但不得把它表述为旧交接文档要求的“6模型×3 seeds”统计实验。如果最终论文需要均值和标准差，仍需补做 `seed=1,2`。

## 3. 12维输入契约

契约名称：

```text
falcon_track_p_gate_relative_12d_v1
```

单个输入向量为：

```text
X = [p_gate - p_vehicle,
     rotation_vector(q_gate * inverse(q_vehicle)),
     velocity,
     acceleration]
```

精确顺序如下：

| 索引 | 特征 | 含义 |
|---:|---|---|
| 0 | gate_position_error_x | 门/局部目标位置减无人机位置，x |
| 1 | gate_position_error_y | 门/局部目标位置减无人机位置，y |
| 2 | gate_position_error_z | 门/局部目标位置减无人机位置，z |
| 3 | gate_attitude_error_x | 相对四元数最短旋转向量，x |
| 4 | gate_attitude_error_y | 相对四元数最短旋转向量，y |
| 5 | gate_attitude_error_z | 相对四元数最短旋转向量，z |
| 6 | velocity_x | 当前速度，x |
| 7 | velocity_y | 当前速度，y |
| 8 | velocity_z | 当前速度，z |
| 9 | acceleration_x | 当前加速度，x |
| 10 | acceleration_y | 当前加速度，y |
| 11 | acceleration_z | 当前加速度，z |

坐标系为 AirSim NED：`x=North, y=East, z=Down`。默认控制器坐标转换是显式恒等映射，不得在 Ubuntu 或板端额外翻转 z 轴。

姿态误差不是欧拉角逐项相减，也不是四元数逐项相减，而是：

```text
q_error = q_gate * inverse(q_vehicle)
attitude_error = shortest_rotation_vector(q_error)
```

与旧25维输入相比，本模型**没有**以下13维：四个未来路径点相对位置（12维）和参考速度（1维）。这是为了尽可能保持 FALCON 原论文公开的12维接口。

## 4. 四维输出契约

输出顺序固定为：

```text
U[0] = collective_thrust_acceleration
U[1] = body_rate_x
U[2] = body_rate_y
U[3] = body_rate_z
```

不得重新排列输出通道。输出层无激活函数。

## 5. 网络结构

六个模型全部为 FP32 全连接网络：

```text
12 → Linear/ReLU × (depth-1) → Linear(4)
```

- `depth` 表示 Linear 层总数，而不是隐藏层数量；
- 隐藏层数量为 `depth-1`；
- 每个隐藏 Linear 后使用 ReLU；
- 最后一层为4维 Linear，无 ReLU；
- 隐藏层使用 Kaiming 初始化，输出层使用 Xavier 初始化；
- 控制周期元数据为 `0.05 s`，即20 Hz；
- 本数据集的局部规划器名义/终端参考速度为 `5.0 m/s`。

## 6. 归一化与实际推理公式

`controller_scripted.pt` 只包含 FC/ReLU 网络本体，**不内置归一化或反归一化**。CPU、Caffe、DPU和UDP服务必须执行同一公式：

```text
x_normalized = (x_physical - x_mean) / x_std
u_normalized = DNN(x_normalized)
u_physical   = u_normalized * u_std + u_mean
```

本批模型的归一化参数长度必须满足：

```text
x_mean.shape = [12]
x_std.shape  = [12]
u_mean.shape = [4]
u_std.shape  = [4]
```

六模型使用同一数据集归一化。权威文件为：

```text
experiments/falcon_paper_fidelity_acado_v4_independent/
  dnn_dataset_balanced_120k/normalization.json
```

不得复用Track-E的25维 `normalization.json`。

## 7. 教师数据与划分

本次使用的数据集不是旧的五门连续飞行数据，而是独立状态查询数据：

```text
experiments/falcon_paper_fidelity_acado_v4_independent/
  teacher_state_queries_balanced_120k/
  dnn_dataset_balanced_120k/
```

主要属性：

- 总样本数：120,000；
- 输入形状：`[120000, 12]`；
- 标签形状：`[120000, 4]`；
- teacher：`official_rpg_mpc_acado_qpoases`；
- 每条独立状态都重新调用 ACADO/qpOASES MPC 生成标签；
- 六类状态各20,000条：正常跟踪、横向恢复、高度恢复、姿态恢复、高速恢复、组合边界恢复；
- 控制区间分布：interior 72,000，near-limit 30,000，saturated 18,000；
- 以 `query_group_id` 为单位做确定性80/10/10划分；
- train/validation/test 分别为96,000/12,000/12,000；
- 三个集合无 group 泄漏；
- 测试集没有参与 early stopping；
- 主要训练和选模指标为 normalized MSE。

这属于保持12维接口的鲁棒性扩展，不应声称它就是论文未公开细节下完全相同的40k连续轨迹数据。

## 8. Windows 模型目录

Windows 权威根目录：

```text
F:/FalconReproduction/airsim_control/
  experiments/falcon_paper_fidelity_acado_v4_independent/
    runs_balanced_120k_seed0/seed_0/
```

其下六个目录为：

```text
depth10_width512/
depth10_width1024/
depth22_width512/
depth22_width1024/
depth34_width512/
depth34_width1024/
```

每个已训练目录现有：

```text
best_model.pt
controller_scripted.pt
latest_checkpoint.pt
model_spec.json
normalization.json
metrics.json
history.csv
per_output_metrics.csv
per_gate_metrics.csv
test_predictions.npz
```

其中：

- `best_model.pt`：验证集 normalized MSE 最优的标准 PyTorch checkpoint；
- `controller_scripted.pt`：不含归一化的 TorchScript 网络；
- `latest_checkpoint.pt`：用于恢复训练，不是部署首选；
- `metrics.json`：验证/测试指标和训练完成状态；
- `model_spec.json`：输入维度、特征顺序、网络结构和坐标系；
- `normalization.json`：12输入/4输出归一化参数。

## 9. 交给 Ubuntu 前的框架无关导出

目前六个训练目录已经完整，但若 Ubuntu/Vitis-AI 流程需要 `weights_fp32.npz` 和 `golden_vectors.npz`，应先在 Windows 项目根目录逐模型执行以下导出。该步骤不重新训练模型。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

$dataDir = '.\experiments\falcon_paper_fidelity_acado_v4_independent\dnn_dataset_balanced_120k'
$runRoot = '.\experiments\falcon_paper_fidelity_acado_v4_independent\runs_balanced_120k_seed0\seed_0'
$models = @(
  'depth10_width512',
  'depth10_width1024',
  'depth22_width512',
  'depth22_width1024',
  'depth34_width512',
  'depth34_width1024'
)

foreach ($model in $models) {
  & $trainPy .\export_dnn_artifacts.py --model "$runRoot\$model\best_model.pt" --data-dir $dataDir --output-dir "$runRoot\$model\export" --golden-count 1000
  if ($LASTEXITCODE -ne 0) { throw "Export failed: $model" }
}
```

每个 `export/` 应包含：

```text
weights_fp32.npz
model_spec.json
normalization.json
golden_vectors.npz
export_manifest.json
```

导出程序输出的 `numpy_pytorch_max_abs_error` 应接近浮点舍入误差，建议要求 `<=1e-5`。

推荐向 Ubuntu 复制：

```text
FALCON_12D_SIX_DNN_UBUNTU_HANDOFF.md
experiments/falcon_paper_fidelity_acado_v4_independent/
  dnn_dataset_balanced_120k/train.npz
  dnn_dataset_balanced_120k/validation.npz
  dnn_dataset_balanced_120k/test.npz
  dnn_dataset_balanced_120k/normalization.json
  dnn_dataset_balanced_120k/split_manifest.json
  teacher_state_queries_balanced_120k/manifest.json
  runs_balanced_120k_seed0/capacity_summary.json
  runs_balanced_120k_seed0/seed_0/<六个模型目录>/
```

其中 `train.npz` 用于从训练集构建 INT8 calibration subset；`validation.npz` 和
`test.npz` 用于 Ubuntu 端独立复核，不得用测试集做量化校准或模型选择。

若只传部署最小包，每个模型至少传：

```text
best_model.pt
controller_scripted.pt
metrics.json
model_spec.json
normalization.json
export/weights_fp32.npz
export/model_spec.json
export/normalization.json
export/golden_vectors.npz
export/export_manifest.json
```

## 10. Ubuntu/Vitis-AI 后续执行顺序

对六个模型分别执行：

1. 读取 `model_spec.json`，断言输入为12、输出为4；
2. 使用 `weights_fp32.npz` 重建 FP32 FC/ReLU 网络；
3. 用该模型自己的 `golden_vectors.npz` 验证 PyTorch/NumPy一致性；
4. 构建输入为 `1×12` 的 Caffe/1×1 Conv 等价网络；
5. 验证 Caffe FP32 与黄金输出；
6. 使用12维校准输入执行 Vitis-AI 1.2.1 INT8量化；
7. 比较 INT8、Caffe FP32、PyTorch FP32黄金输出；
8. 针对AXU3EGB的真实DPU architecture分别编译六个模型；
9. 板端逐模型运行1000条黄金向量；
10. 测量每个模型的DPU执行时间、服务总时间和网络往返时间；
11. 只有六个模型均通过正确性检查并得到真实时延后，才构建FALCON的能力/时延数据并训练SVM/SVR或调度策略。

量化、编译和板端验证必须按模型分别保存报告，不能让六个模型共享或覆盖同一个量化输出目录。

## 11. UDP协议必须同步从25维改为12维

旧Windows客户端 `fpga_udp_client.py` 中存在以下固定定义：

```text
REQUEST_PAYLOAD_FORMAT = "!25f"
x_normalized.shape == (25,)
```

因此，旧 `FLCN/v1` 客户端和板端服务不能原样用于这批12维模型。Ubuntu端必须与Windows端共同完成以下修改：

- 请求载荷由25个 `float32` 改成12个 `float32`；
- 板端输入张量元素数改成12；
- 正规化参数改成12维版本；
- 黄金客户端改成发送12维向量；
- shadow/closed-loop桥接程序构造 `falcon_track_p_gate_relative_12d_v1`；
- 响应仍为4个 `float32`，顺序保持 `[F, wx, wy, wz]`；
- 协议应增加明确的输入维数检查，推荐升级协议版本或使用不同端口，避免25维与12维客户端静默混用。

在修改协议后，先完成“Windows 12-D黄金客户端 → UDP → AXU3EGB DPU → 4-D返回”的1000条一致性验证，再进入AirSim影子模式和闭环控制。

## 12. Ubuntu Agent 的验收检查清单

开始量化前：

- [ ] 确认处理的是本文件所列12-D模型，不是Track-E 25-D模型；
- [ ] 六个 `metrics.json` 均包含 `training_complete: true`；
- [ ] 六个 `model_spec.json` 均为 `input_shape: [1,12]`、`output_shape: [1,4]`；
- [ ] 归一化长度为12/12/4/4；
- [ ] 六模型架构分别为10/22/34层和512/1024宽度；
- [ ] FP32 NumPy重建通过1000条黄金向量；
- [ ] Caffe输入层和校准数据均为12维；
- [ ] 旧25维UDP客户端/服务没有被误用。

进入SVM/SVR或调度实验前：

- [ ] 六个INT8模型全部完成编译；
- [ ] 六模型板端黄金验证全部通过；
- [ ] 每个模型都获得真实DPU延迟，而不是按参数量估算；
- [ ] 能力指标使用统一测试集和统一MSE定义；
- [ ] 报告中说明当前是单seed结果；
- [ ] 报告中说明120k独立状态数据是保持论文12-D接口的鲁棒性扩展。

## 13. 不应做的事情

- 不要把25维权重裁掉13维后当作12维模型；
- 不要把12维输入补零成25维后继续使用旧DPU模型；
- 不要复用旧Track-E的归一化或黄金向量；
- 不要改变12维特征顺序；
- 不要给姿态误差使用欧拉角或四元数逐项相减；
- 不要跳过输出反归一化；
- 不要用测试集选择checkpoint或调参；
- 不要把单seed结果写成三seed均值；
- 不要在真实板端时延得到之前训练最终的论文级调度模型。

## 14. 一句话交接结论

Windows 已完成六种 FALCON 候选架构的 **12-D FP32 DNN** 训练，并在统一测试集上得到随模型容量总体严格改善的 normalized MSE；Ubuntu 后续应基于这六个12维模型重新执行 FP32重建、INT8量化、DPU编译、板端黄金验证和时延测量，同时废止旧Track-E 25维模型在本条论文复现主线中的接口假设。
