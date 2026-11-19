# 2026-09-24 起训练的 SCENIC Table III 12 个 DNN 说明

## 1. 实验批次

- **实验批次标记**：2026 年 9 月 24 日 19:21 开始（时间来自本次实验记录）。
- **实验目的**：按照 SCENIC 论文 Table III 的网络规模，比较不同深度和宽度的 DNN 对当前 MPC 控制器输出的拟合能力，以及模型参数量和 MAC 数量。
- **结果目录**：`controller_architecture_study/results/scenic_table3_v1`
- **比较结果目录**：`controller_architecture_study/results/scenic_table3_v1/comparison`

本批次共包含 12 个网络。其中 8 个是本批次新训练的模型，4 个模型已经在前面的粗粒度架构实验中训练过，因此直接复用其权重和评估结果，没有重复训练。

## 2. 数据集

### 2.1 数据集位置与规模

训练使用：

```text
F:\FalconReproduction\airsim_control\experiments\paper_track_p_12d_5ms_v1\dnn_dataset
```

| 文件 | 样本数 | 用途 |
|---|---:|---|
| `train.npz` | 384,000 | 模型参数训练 |
| `validation.npz` | 48,000 | 选择最佳 checkpoint 和早停 |
| `test.npz` | 48,000 | 训练结束后的最终评估 |
| `normalization.json` | — | 输入和输出归一化参数 |
| `split_manifest.json` | — | 数据来源、环境划分和输入契约 |

总数据量为 **480,000 条样本**。

### 2.2 数据生成方式

这批数据不是让无人机连续飞完五个门后，把整段飞行轨迹直接拿来训练。每条样本都是独立的状态查询：

1. 生成一个门和无人机之间的相对状态；
2. 随机设置位置误差、姿态误差、速度和加速度；
3. 对这个独立状态调用一次 MPC；
4. 保存该状态和对应的 MPC 控制输出。

因此相邻样本不要求连续，也不代表同一条飞行轨迹的相邻时刻。训练目标是：

```text
DNN(X_state) ≈ MPC(X_state)
```

而不是学习固定的“五门飞行流程”。

### 2.3 数据划分

数据按照 `environment_id` 划分，而不是把同一环境中的样本随机打散到三个集合：

| 数据集 | 环境 ID | 环境数 | episode 数 | 样本数 | 比例 |
|---|---|---:|---:|---:|---:|
| train | 0–47 | 48 | 48 | 384,000 | 80% |
| validation | 48–53 | 6 | 6 | 48,000 | 10% |
| test | 54–59 | 6 | 6 | 48,000 | 10% |

### 2.4 输入与输出

输入是 AirSim NED 坐标系下的 12 维状态：

```text
[
  gate_position_error_x, gate_position_error_y, gate_position_error_z,
  gate_attitude_error_x, gate_attitude_error_y, gate_attitude_error_z,
  velocity_x, velocity_y, velocity_z,
  acceleration_x, acceleration_y, acceleration_z
]
```

对应含义为：

```text
X[0:3]   = 门中心位置 - 无人机当前位置
X[3:6]   = 门姿态误差
X[6:9]   = 无人机当前线速度
X[9:12]  = 无人机当前线加速度
```

输出是 MPC 的 4 维控制量：

```text
[
  collective_thrust_acceleration,
  body_rate_x,
  body_rate_y,
  body_rate_z
]
```

### 2.5 状态覆盖范围

教师数据包含 6 类独立状态：

| 状态族 | 样本数 | 主要覆盖内容 |
|---|---:|---|
| `nominal_tracking` | 119,365 | 正常门前跟踪 |
| `lateral_recovery` | 86,561 | 横向位置和速度恢复 |
| `vertical_recovery` | 82,266 | 高度位置和速度恢复 |
| `attitude_recovery` | 62,464 | 姿态误差恢复 |
| `near_gate_fast` | 57,655 | 接近门且速度较高 |
| `combined_recovery` | 71,689 | 位置、姿态、速度、加速度组合异常 |

主要范围：门前距离约 `0.05–16 m`，横向误差约 `-3–3 m`，垂直误差约 `-2.4–2.4 m`，速度分量包含负速度到高速状态，加速度分量约 `-5.5–5.5 m/s²`。边界状态额外提高采样概率，边缘采样概率为 `0.3`。

## 3. MPC 教师

这批数据使用工程中的 **快速线性 Track-P MPC 教师**，不是后续 Ubuntu 流程中的 ACADO/PAMPC 精确实现。因此准确表述是：

> 固定 5 m/s 参考速度下，12D Track-P 线性 MPC 控制器的 DNN 模仿学习实验。

### 3.1 MPC 参数

| 参数 | 值 |
|---|---:|
| 控制周期 `dt` | `0.05 s` |
| 控制频率 | `20 Hz` |
| MPC horizon | `20` |
| 参考速度 | `5.0 m/s` |
| 最大水平加速度 | `6.0 m/s²` |
| 最大推力加速度 | `18.0 m/s²` |
| 最大机体角速度 | `3.0 rad/s` |

当前 12D 输入没有显式目标速度维度，所以参考速度固定为 `5.0 m/s`，保证同一个 12D 状态对应唯一的教师标签。如果在同一状态下混入多个目标速度，MPC 可能给出不同标签，映射就不再是单值函数。

### 3.2 教师数据配置

配置文件：`experiments/paper_track_p_12d_5ms_v1/teacher_config.json`

关键设置：

```text
dt = 0.05 s
environment_count = 60
samples_per_environment = 8000
reference_speed = 5.0 m/s
teacher_policy = track_p_independent_gate_state_mpc_queries_v1
```

总数据量：`60 × 8000 = 480000` 条。

## 4. 统一训练协议

8 个本批次新训练模型统一使用：

| 项目 | 配置 |
|---|---|
| 网络类型 | 普通全连接 MLP |
| 激活函数 | ReLU |
| 残差连接 | 无 |
| 输入维度 | 12 |
| 输出维度 | 4 |
| 优化器 | Adam，`foreach=False` |
| 初始学习率 | `1e-4` |
| batch size | `512` |
| 最大 epoch | `500` |
| 最小 epoch | `100` |
| early stopping patience | `50` |
| 梯度裁剪 | `gradient_clip_norm=1.0` |
| 随机种子 | `seed=0` |
| 选择指标 | validation normalized MSE |
| 训练方式 | 串行训练，逐模型保存 checkpoint |

输入和输出均使用训练集统计量标准化：

```text
x_norm = (x - x_mean) / x_std
u_norm = (u - u_mean) / u_std
```

损失为四维归一化控制输出上的 MSE：

```text
mean((u_DNN_norm - u_MPC_norm)^2)
```

验证集用于选择最佳 checkpoint 和早停；测试集只在训练结束后评估，不参与训练决策。

## 5. 12 个模型与实际训练结果

工程中的 `hidden_layers` 只统计隐藏层数量。SCENIC 论文的 `L_i` 按总 FC 层数理解，因此：

```text
hidden_layers = paper_L - 1
```

| 编号 | 论文 `L_i` | 隐藏层×宽度 | 来源 | 完成 epoch | 最佳 epoch | Test normalized MSE | 参数量 | MACs |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 13 | 12×1024 | 本批次新训练 | 357 | 307 | 1.51325e-04 | 11,563,012 | 11,550,720 |
| 2 | 11 | 10×1024 | 本批次新训练 | 500 | 479 | 1.03389e-04 | 9,463,812 | 9,453,568 |
| 3 | 11 | 10×512 | 之前训练后复用 | 367 | 317 | 1.10966e-04 | 2,372,612 | 2,367,488 |
| 4 | 9 | 8×1024 | 本批次新训练 | 500 | 461 | 1.00925e-04 | 7,364,612 | 7,356,416 |
| 5 | 9 | 8×512 | 本批次新训练 | 500 | 471 | 5.52588e-05 | 1,847,300 | 1,843,200 |
| 6 | 9 | 8×256 | 本批次新训练 | 500 | 492 | 1.20729e-05 | 464,900 | 462,848 |
| 7 | 7 | 6×1024 | 本批次新训练 | 500 | 486 | 9.95193e-05 | 5,265,412 | 5,259,264 |
| 8 | 7 | 6×256 | 之前训练后复用 | 496 | 446 | 1.03384e-05 | 333,316 | 331,776 |
| 9 | 7 | 6×128 | 之前训练后复用 | 425 | 375 | 1.29574e-05 | 84,740 | 83,968 |
| 10 | 5 | 4×1024 | 本批次新训练 | 500 | 498 | 8.21736e-05 | 3,166,212 | 3,162,112 |
| 11 | 5 | 4×128 | 之前训练后复用 | 500 | 498 | 9.46265e-06 | 51,716 | 51,200 |
| 12 | 3 | 2×1024 | 本批次新训练 | 500 | 492 | 1.19552e-05 | 1,067,012 | 1,064,960 |

完整机器可读结果：

```text
controller_architecture_study/results/scenic_table3_v1/comparison/scenic_table3_comparison.csv
```

## 6. 结果解读

当前测试集上总体最好的模型是：

```text
plain_mlp_h4_w128_seed0
test normalized MSE = 9.46265e-06
parameters          = 51,716
MACs                = 51,200
```

本批次新训练模型中表现最好的模型是：

```text
plain_mlp_h2_w1024_seed0
test normalized MSE = 1.19552e-05
parameters          = 1,067,012
MACs                = 1,064,960
```

结果说明：

1. 当前任务中，网络越大并不必然带来更低的 MPC 拟合误差；
2. 较小模型已经能够拟合当前快速线性 MPC 的主要规律；
3. 大模型在当前固定训练预算下没有取得最优结果；
4. 达到 500 epoch 上限的模型可能仍受优化预算限制，不能简单解释为模型容量不足；
5. 每个架构目前只有一个 seed，不能据此宣称全局最优架构；
6. 最终候选应同时考虑离线 MSE、参数量、MACs、闭环成功率和硬件端到端延迟。

## 7. 与论文数值的关系

配置文件保留了 SCENIC Table III 的论文测试损失，用于结构趋势参考。但论文和本工程在数据集、仿真器、MPC 实现、归一化方式、训练 seed 和训练预算方面可能不同，因此论文损失与当前 MSE **不能直接做绝对数值比较**。

当前实验的有效比较范围是：相同 12D 数据、相同 MPC 教师、相同训练/验证/测试划分和相同训练协议下的 12 个网络之间的比较。

## 8. 产物

每个模型目录通常包含：

```text
config.json
best_model.pt
last_model.pt
train_log.csv
validation_curve.svg
offline_metrics.json
complexity.json
```

- `best_model.pt`：验证集指标最好的权重；
- `last_model.pt`：最后一轮状态，可继续训练；
- `train_log.csv`：逐 epoch 训练和验证记录；
- `offline_metrics.json`：validation/test 离线指标；
- `complexity.json`：参数量、MACs 和 checkpoint 大小；
- `validation_curve.svg`：验证损失曲线。

## 9. 实验边界

本批次只验证 DNN 对 MPC 标签的离线拟合能力，没有直接证明：

- DNN 在所有 AirSim 初始状态下都能安全闭环；
- DNN 一定比当前 CPU MPC 更快；
- 量化、剪枝、DPU 编译后仍保持相同误差；
- FPGA 端网络延迟一定满足完整 FALCON 调度周期；
- 离线 MSE 最小的模型就是系统级最优模型。

后续用于 FPGA/DPU 或动态切换的候选选择，至少还应检查：

```text
离线 test MSE
模型参数量和 MACs
控制周期端到端延迟
AirSim 闭环成功率
FPGA/DPU 量化后误差
动态模型切换时的稳定性
```
