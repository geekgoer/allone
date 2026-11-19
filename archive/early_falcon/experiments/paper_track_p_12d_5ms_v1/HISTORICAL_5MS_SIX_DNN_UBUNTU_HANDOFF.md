# FALCON 历史 5 m/s 六模型：Windows → Ubuntu/Vitis AI 1.2.1 交接

## 1. 交接结论

本包交接六个已经在 Windows AirSim 中完成统一闭环复测的 12 输入、4 输出、Linear+ReLU DNN。六个模型在六个统一场景中全部成功：

- `36/36` 个闭环 episode 成功；
- `180/180` 次门平面穿越有效；
- 没有实际碰撞；
- 没有 50 ms 控制周期 deadline miss。

因此，这批模型可以用于下一阶段的 FP32 重建、Vitis AI 1.2.1 INT8 量化、针对 AXU3EGB 实际 DPU 架构编译 ELF、板端黄金向量验证和时延测量。

必须同时保留一个事实：这六个模型的离线 MSE **不随模型容量单调下降**，而且训练来源并非完全同构。因此它们适合继续完成“六模型可部署性与异构调度流程”，但不能把当前 MSE 表格写成 FALCON Table II 的精确复现。

## 2. 模型 ID 与权威结果

建议继续沿用既有 FLCN/v2 模型 ID：

| model_id | 模型 | 参数量 | 最佳 epoch | 历史 Test physical MSE | 统一 AirSim 状态 MSE | 闭环 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 10×512 | 2,109,956 | 1083 | `5.4548e-5` | `2.0122e-5` | 6/6 |
| 2 | 10×1024 | 8,414,212 | 1092 | `4.9361e-5` | `4.4351e-5` | 6/6 |
| 3 | 22×512 | 5,261,828 | 1706 | `8.2553e-5` | `4.3444e-5` | 6/6 |
| 4 | 22×1024 | 21,009,412 | 1721 | `1.4764e-3` | `1.8732e-3` | 6/6 |
| 5 | 34×512 | 8,413,700 | 3430 | `1.7346e-3` | `1.0154e-3` | 6/6 |
| 6 | 34×1024 | 33,604,612 | 664 | `2.0173e-3` | `1.8177e-3` | 6/6 |

“历史 Test physical MSE”是在共同的 48,000 条测试集上重新审计得到；“统一 AirSim 状态 MSE”是在六个新 AirSim 教师 episode 的 1,322 个共同状态上比较 DNN 与教师输出得到。两个数属于不同测试分布，不能相互替换。

对应证据位于：

```text
metadata/mse_reaudit_20260916/
metadata/physical_benchmark_v1/common_state_summary.json
metadata/physical_benchmark_v1/closed_loop_summary.json
```

## 3. 12 维输入契约

规范契约名：

```text
falcon_track_p_gate_relative_12d_v1
```

旧 checkpoint 中的描述性别名：

```text
[p_gate-p, r_gate-r, velocity, acceleration] in AirSim NED
```

两者在本包中表示同一契约。精确输入为：

```text
X[0:3]  = p_gate - p_vehicle
X[3:6]  = shortest_rotation_vector(q_gate * inverse(q_vehicle))
X[6:9]  = current linear velocity
X[9:12] = current linear acceleration
```

特征顺序：

| 索引 | 特征 |
|---:|---|
| 0..2 | gate position error x/y/z |
| 3..5 | gate attitude shortest-rotation-vector error x/y/z |
| 6..8 | velocity x/y/z |
| 9..11 | acceleration x/y/z |

坐标系为 AirSim NED，`x=North, y=East, z=Down`。姿态误差不是欧拉角逐项相减，也不是四元数分量相减。Ubuntu 重建、黄金测试和 Windows/FPGA 闭环必须保持完全相同的 12 维顺序。

参考速度固定为 `5.0 m/s`，没有作为额外输入。控制周期元数据为 `0.05 s`（20 Hz）。

## 4. 输出契约

四维输出顺序固定为：

```text
U[0] = collective_thrust_acceleration
U[1] = body_rate_x
U[2] = body_rate_y
U[3] = body_rate_z
```

最后一层无激活。不要改变通道顺序。

## 5. 网络与归一化

每个网络均为：

```text
12 -> (Linear + ReLU) × (depth - 1) -> Linear(4)
```

`depth` 是 Linear 层总数。推理必须执行：

```text
x_norm = (x_physical - x_mean) / x_std
u_norm = DNN(x_norm)
u_physical = u_norm * u_std + u_mean
```

六模型共享本包 `dataset/normalization.json`。打包器已逐模型按位检查归一化数组一致性：

```text
x_mean: 12
x_std:  12
u_mean: 4
u_std:  4
```

## 6. 数据集与 calibration 规则

数据集共 480,000 条独立状态查询：

```text
train.npz       384,000
validation.npz   48,000
test.npz         48,000
```

输入 `X.shape=[N,12]`，标签 `U.shape=[N,4]`。教师策略在清单中记为：

```text
track_p_independent_gate_state_mpc_queries_v1
```

量化校准只能从 `dataset/train.npz` 的 `X` 构建。不要用 validation/test 做 calibration，也不要根据 test 选择量化参数。

这批教师是项目中的快速线性 MPC，不是后来 ACADO/qpOASES 教师，也不是论文未公开实现的 PAMPC。因此后续报告必须称其为“历史 5 m/s Track-P 12-D 六模型部署实验”，不要称为 PAMPC 的逐数值精确复现。

## 7. 包目录说明

```text
HISTORICAL_5MS_SIX_DNN_UBUNTU_HANDOFF.md
package_manifest.json
SHA256SUMS.txt
dataset/
  train.npz
  validation.npz
  test.npz
  normalization.json
  split_manifest.json
models/
  depth10_width512/
  depth10_width1024/
  depth22_width512/
  depth22_width1024/
  depth34_width512/
  depth34_width1024/
metadata/
source/
```

每个模型目录包含训练指标及：

```text
best_model.pt
controller_scripted.pt
export/weights_fp32.npz
export/model_spec.json
export/normalization.json
export/golden_vectors.npz
export/export_manifest.json
```

若使用 `--without-pytorch` 打小包，则不含两个 `.pt`，但仍含 Ubuntu 重建和量化所需的框架无关 FP32 权重与黄金向量。

## 8. Ubuntu 上的执行顺序

### 8.1 校验传输完整性

```bash
cd /path/to/unpacked_package
sha256sum -c SHA256SUMS.txt
```

随后读取 `package_manifest.json`，断言模型 ID、输入维度和模型数量正确。

### 8.2 对每个模型重建 FP32 网络

对六个 `models/<name>/export/weights_fp32.npz` 分别执行：

1. 按 `linear_00_weight/bias` 至最后一层的编号顺序重建；
2. 隐藏层后 ReLU，最后一层不加 ReLU；
3. 使用该模型 export 目录中的 normalization；
4. 使用 `golden_vectors.npz` 比较 NumPy/PyTorch FP32；
5. 建议 FP32 重建误差要求 `max_abs <= 1e-5`。

### 8.3 转为 Vitis AI 1.2.1 可接受的 Caffe 网络

若 FC 层通过 `1×1 Conv` 等价实现：

- 输入形状必须只表达 12 个元素；
- 权重矩阵顺序不能转置错；
- bias 必须逐层复制；
- ReLU 只放在前 `depth-1` 个 Linear 后；
- 最后一层输出 4 个元素且无激活。

先用全部黄金向量验证 Caffe FP32，再开始量化。

### 8.4 INT8 量化

六个模型必须使用独立量化目录。校准样本从 `dataset/train.npz` 确定性抽取，并保存索引、数量、预处理规则和量化日志。

量化后至少报告：

- INT8 vs Caffe FP32；
- INT8 vs PyTorch/NumPy FP32；
- 每输出维的 MAE/RMSE/max abs；
- 1000 条黄金向量总体 max abs。

### 8.5 编译 ELF

必须读取 AXU3EGB 板上实际 DPU architecture，分别编译六个模型。不要仅根据板卡名称猜架构，也不要复用旧 v4/v6 包的 `.elf`、量化 prototxt、fix info 或 kernel 名称。

建议模型名保持：

```text
falcon_p_d10_w512
falcon_p_d10_w1024
falcon_p_d22_w512
falcon_p_d22_w1024
falcon_p_d34_w512
falcon_p_d34_w1024
```

### 8.6 板端验证与时延

每个 ELF 均应：

1. 启动时只加载一次 kernel/task；
2. 跑 1000 条模型自身的黄金向量；
3. 验证 model ID 与请求一致；
4. 分别记录 DPU 核心时间、DPU+I/O、服务端总时间和 UDP RTT；
5. 报告 median、p95、p99、max；
6. 按 50 ms deadline 统计 miss。

## 9. 与现有 FLCN/v2 服务的关系

当前协议仍是 12 个输入 float 和 4 个输出 float，因此 FLCN/v2 报文结构与模型 ID 映射可以保留。但模型语义、归一化和权重属于本包：

- 可以复用协议代码与 UDP 框架；
- 必须替换六个 DPU 模型；
- 必须替换 normalization；
- 必须替换黄金向量；
- 必须重新量化和编译；
- 必须重新跑板端 6000 条（6×1000）黄金验证。

绝不能因为新旧包都是 12 维，就继续使用以前已编译的 v4/v6 ELF。

## 10. 时延解释与后续调度

Windows TorchScript CPU 推理均值约为 `1.49–5.39 ms`，p95 约为 `1.71–6.15 ms`，均低于 50 ms。但本项目快速线性 MPC 在统一状态采集中的平均求解时间约为 `0.48–0.51 ms`，所以这批 CPU DNN **没有证明比该简化 MPC 更快**。

较复杂 ACADO 流程的既有测量约为：

- solver adapter 平均约 `0.56 ms`；
- planner + solver 平均约 `5.88 ms`，p95 约 `7.44 ms`。

Ubuntu 后续只能使用真实 DPU 测量值构建能力/时延表和调度模型，不能按参数量估算，也不能直接把 CPU 推理时间当作 FPGA 时间。

## 11. 研究报告中必须披露的限制

- 六模型 MSE 不随模型容量单调下降；
- 前三个本地模型具有 resume/warm-start 训练历史，后三个服务器模型为 fresh training，训练来源存在混杂；
- 当前结果为单 seed；
- 教师是项目线性 MPC，不是 ACADO/PAMPC；
- 闭环成功说明模型在当前测试场景可用，但不等于全局鲁棒性证明；
- 不能挑选 checkpoint 人为制造论文中的单调 MSE 趋势。

这些限制不妨碍继续验证“六种 DNN 在 DPU 上的量化、编译、时延与调度流程”，但必须如实写入实验方法和威胁有效性部分。

## 12. Ubuntu Agent 完成标准

- [ ] `sha256sum -c` 全部通过；
- [ ] 六模型 FP32 重建均通过各自黄金向量；
- [ ] 六模型 Caffe FP32 与原 FP32 对齐；
- [ ] 六模型各自完成 INT8 量化并保存报告；
- [ ] 六模型针对实际 DPU architecture 编译 ELF；
- [ ] 板端 6000/6000 黄金请求成功；
- [ ] 每个模型得到真实时延分布与 50 ms miss 统计；
- [ ] 更新 FLCN/v2 服务并重新做 Windows UDP smoke test；
- [ ] 不复用上一批模型的权重、归一化、黄金向量或 ELF。

## 13. 一句话交接

Windows 端现交付一组“12-D、5 m/s、六种 FC/ReLU 架构、AirSim 36/36 闭环成功”的可部署模型；Ubuntu 端应按本包重新完成 FP32 重建、INT8 量化、真实 DPU 架构 ELF 编译、6000 条板端黄金验证和时延测量，同时如实保留其 MSE 非单调与教师实现差异。

## 14. Windows 端打包来源

本包应由项目根目录中的下列程序生成，不要手工拼接六个不同目录：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
& $trainPy .\experiments\paper_track_p_12d_5ms_v1\package_historical_six_for_ubuntu.py --output-root .\ubuntu_transfer --golden-count 1000
```

正常结束时程序打印 `status: PASS`、包目录、ZIP 路径及 ZIP 的 SHA-256。默认包包含 PyTorch checkpoint 与 TorchScript，便于 Ubuntu 端交叉核验。若只需要部署最小包，可增加 `--without-pytorch`；若六个 export 已经完整生成而仅需重新压缩，可增加 `--reuse-exports`。
