# FALCON MPC Imitation DNN：Vitis AI 1.2.1 阶段交接文档

更新时间：2026-09-05  
Windows 工作目录：`F:\FalconReproduction\airsim_control`  
后续目标：把已经通过 AirSim 闭环验证的 FP32 DNN 转入 Vitis AI 1.2.1，完成 INT8 量化、目标 DPU 编译、数值验证和多 DNN 运行。

## 1. 当前结论

Windows 端 FP32 baseline 已完成，可以冻结，不需要重新生成数据或重新训练。

- 模型输入：25 维状态/路径向量。
- 模型输出：4 维控制命令 `[F, wx, wy, wz]`。
- 网络：10 个 Linear 层，前9层输出宽度512并接 ReLU，最后一层输出4维。
- 参数量：2,116,612。
- 教师：项目内工程化线性 MPC，不是论文未公开实现的逐项复刻。
- 单无人机标准闭环：40/40 episode 成功，共200/200次有效穿门。
- 单无人机 3.5 m/s：12/12 episode 成功。
- MPC/DNN 同场景小规模对照：MPC 4/4，DNN 4/4。
- 三无人机 DNN：3架无人机各穿5门，15/15次有效穿门，无运行中碰撞或超时。

因此下一阶段不要继续堆叠低价值 Windows 测试；直接进行部署适配。只有量化误差或 DPU 闭环失败时，才回到数据/训练阶段。

## 2. 不可丢失的模型产物

模型目录：

`F:\FalconReproduction\airsim_control\dnn_runs\iid_path_v1_depth10_width512`

主要文件：

| 文件 | 用途 | SHA256 |
|---|---|---|
| `best_model.pt` | 标准 PyTorch checkpoint/state_dict | `80DF6FB72B6470C9B5783359EA225AA212B03C9C763A6C47E185CD387578BD69` |
| `controller_scripted.pt` | Windows/AirSim 使用的 TorchScript | `7EDE899CDE770CE209A1C0AB337F844DA6C202DBC13A3E1665E0522B6D8FCE9E` |
| `normalization.json` | DNN 输入/输出归一化 | `5066D8A05629EB98B7F2717883B8DC125A7FF1A2366C40421255D89C47E93BEE` |
| `export/weights_fp32.npz` | 框架无关逐层 FP32 权重 | `7D45DE80259879FCD1C12E27AB5E15C6FB533FAA93D397CD5A4C64B90292D933` |
| `export/golden_vectors.npz` | 1000条数值一致性验证向量 | `6A657363A026A1C22E575A19B980475D1F778B88EDE4B062989A15F1F966A786` |
| `export/model_spec.json` | 网络层、输入输出和接口描述 | 见文件 |
| `export/export_manifest.json` | 导出清单和一致性误差 | 见文件 |

导出验证结果：NumPy 与 PyTorch 最大绝对误差为 `1.1920928955078125e-06`。

复制到 Ubuntu 后必须重新执行 SHA256 校验。不要覆盖 Windows 原始产物。

## 3. 数据集和训练结果

原始 IID 教师数据：

`F:\FalconReproduction\airsim_control\teacher_dataset_iid_path_v1\final\teacher_dataset_iid_path_v1.npz`

训练拆分：

`F:\FalconReproduction\airsim_control\dnn_dataset_iid_path_v1`

- train：576,000条。
- validation：72,000条。
- test：72,000条。
- 按 `environment_id` 划分，避免同环境跨集合泄漏。
- 每行是独立状态/路径下的一次 MPC 查询，相邻行不要求时间连续。
- 60个环境，每环境12,000条，共720,000条。
- 状态分为 nominal、moderate、recovery 三档扰动。
- 速度覆盖1.0、1.5、2.0、2.5、3.0、3.5 m/s。

训练：

- CUDA FP32。
- early stopping 于 epoch 1764。
- best epoch 1564。
- validation MSE：`1.2832948414143175e-05`。
- validation RMSE：`0.0035823103971779346`。
- test MSE：`1.2006309589196462e-05`。
- test RMSE：`0.0034650121815502644`。
- test MAE：`0.0020021332893520594`。
- test max absolute error：`0.08821630477905273`。
- 测试集没有用于 early stopping。

完整指标：`dnn_runs/iid_path_v1_depth10_width512/metrics.json`。

## 4. 精确模型结构

模型名：`falcon_dnn_input25_depth10_width512`

```text
input(25)
  -> Linear(25,512) -> ReLU
  -> 8 × [Linear(512,512) -> ReLU]
  -> Linear(512,4)
output(4)
```

总计10个 Linear 层、9个 ReLU。以 `export/model_spec.json` 和实际权重为最终权威来源，不要仅凭上面的简写重建层数。

FP32 权重大约8.5 MB；纯权重 INT8 理论大小约2.1 MB，不含运行时缓冲。

## 5. 25维输入合同

坐标系：AirSim NED。采样周期：0.05 s。MPC reference horizon：20。

输入顺序不可改变：

```text
0  position_error_x
1  position_error_y
2  position_error_z
3  attitude_error_x
4  attitude_error_y
5  attitude_error_z
6  velocity_x
7  velocity_y
8  velocity_z
9  acceleration_x
10 acceleration_y
11 acceleration_z
12 path_offset_1_x
13 path_offset_1_y
14 path_offset_1_z
15 path_offset_5_x
16 path_offset_5_y
17 path_offset_5_z
18 path_offset_10_x
19 path_offset_10_y
20 path_offset_10_z
21 path_offset_19_x
22 path_offset_19_y
23 path_offset_19_z
24 reference_speed
```

`path_offset_k` 是未来第 k 个参考位置相对当前无人机位置的三维向量。当前实现的 offsets 固定为 `[1,5,10,19]`。

输入组装权威代码：`dnn_model_utils.py::build_policy_input`。状态组装权威代码：`state_adapter.py::build_controller_state`。

## 6. 4维输出和归一化合同

输出顺序：

```text
0 collective_thrust_acceleration (F)
1 body_rate_x (wx)
2 body_rate_y (wy)
3 body_rate_z (wz)
```

归一化不在模型图内部：

```text
x_normalized = (x_physical - x_mean) / x_std
u_physical   = u_normalized * u_std + u_mean
```

所有 mean/std 必须从与模型配套的 `normalization.json` 读取，不能重新统计。

运行时安全裁剪：

```text
F             -> [0, 18]
wx, wy, wz    -> [-3, 3]
```

AirSim actuator 映射在 `control_adapter.py`。其中 pitch/yaw rate 在发送到 AirSim 时存在符号变换，DPU 输出本身不要提前改变符号。

## 7. Golden vectors 使用规则

`export/golden_vectors.npz` 包含：

- `X_physical`：25维物理输入。
- `X_normalized`：送入网络的归一化输入。
- `U_true`：MPC教师标签。
- `U_pytorch`：FP32 PyTorch网络的物理输出。

Golden vectors 用来验证转换、量化、编译和 DPU 运行的一致性，不用于训练或 early stopping。

量化 calibration 不要直接使用 test golden vectors。应从 `dnn_dataset_iid_path_v1/train.npz` 提取有代表性的 `X`，使用冻结的 `normalization.json` 转为 normalized input，再形成单独的 calibration set。建议先用10,000至50,000条，覆盖三种 sample family、五个 gate 和全部速度。

## 8. 已完成的闭环证据

### 单机40组

报告：`dnn_runs/iid_path_v1_depth10_width512/closed_loop_40.json`

- 40/40成功。
- 200/200次穿门。
- 20,258个控制周期。
- deadline miss为0。
- 平均CPU推理约1.72 ms。
- 最大横向穿门偏差约0.054 m。
- 最大垂直穿门偏差约0.097 m。

### 3.5 m/s单机

报告：`dnn_runs/iid_path_v1_depth10_width512/closed_loop_speed_3p5.json`

- 12/12成功。

### MPC/DNN对比

报告：`dnn_runs/iid_path_v1_depth10_width512/mpc_dnn_comparison_4.json`

- MPC：4/4。
- DNN：4/4。
- 两者在相同环境和初始状态下均穿过5/5个门。
- 当前工程化线性 MPC 在Windows CPU上约0.07–0.10 ms；DNN约1.5–2.1 ms。DNN当前并不比这个简化MPC的CPU实现快，部署价值在于DPU并行/确定性以及后续多模型调度验证，不应伪造速度优势。

### 三机DNN

报告：`dnn_runs/iid_path_v1_depth10_width512/three_drone_dnn_seed0.json`

- Drone1、Drone2、Drone3使用三个独立DNN实例。
- 三组独立门，共15个门和60个可见门框组件。
- 三机均完成5/5穿门，总计15/15。
- 无运行中碰撞、无超时。
- 573个三机控制周期。
- deadline miss：1，只发生在第一个周期。
- 首周期 Drone1 TorchScript 冷启动推理约39.9 ms；另外两次推理及RPC/状态读取叠加后超过50 ms。
- 稳态平均推理：Drone1 1.46 ms、Drone2 1.19 ms、Drone3 1.16 ms。
- 首样本的 `Ground_5/Ground_6` collision flags 是 AirSim reset/takeoff 残留；脚本只允许首样本忽略，后续没有碰撞。

进入正式实时运行前应对每个实例执行若干次 dummy warm-up，再启动20 Hz deadline计时，以消除首周期冷启动 miss。

三机运行器：`run_three_drones_dnn_gates.py`。

## 9. 后续 Agent 的工作范围

后续 Agent 应直接执行以下阶段，不要重新训练，不要改变25维接口，除非量化/编译证据明确表明必须修改。

### 阶段 A：确认 Vitis AI 1.2.1 环境和目标

1. 记录 Ubuntu 版本、Vitis AI Docker/tag、Python版本和所有 `vai_*` 工具版本。
2. 确认目标 FPGA 板卡、DPU IP名称、DPU fingerprint/architecture JSON、运行时版本和可用内存。
3. 如果用户没有给出目标板卡或 arch 文件，先做框架转换和量化，但不得虚构编译目标。
4. 严格使用 Vitis AI 1.2.1，不得自行升级到新版本并宣称兼容。

### 阶段 B：构建可量化模型

1. 优先从 `best_model.pt` 的 state_dict 重建网络；若旧工具无法读取 checkpoint，则使用 `weights_fp32.npz`。
2. 用1000条 golden vectors 验证重建 FP32 模型，最大误差目标 `<=1e-5`。
3. 如果 Vitis AI 1.2.1 DPU 编译器不支持 Linear/FC 链，将输入 reshape 为 `[N,25,1,1]`，把每个 Linear 等价转换为 `Conv2d(kernel=1x1)`：
   - Linear weight `[out,in]` -> Conv weight `[out,in,1,1]`；
   - bias不变；
   - ReLU位置不变；
   - 输出 reshape 回 `[N,4]`。
4. 任何结构转换都必须先通过 FP32 golden parity，不能直接进入量化。

### 阶段 C：INT8 量化

1. 从 train split 构建代表性 calibration inputs；不要用test标签调参。
2. 使用 Vitis AI 1.2.1 对应的官方量化流程。
3. 保存量化配置、命令、日志、量化模型和工具版本。
4. 对全部1000条 golden vectors 比较：
   - INT8仿真输出 vs FP32 DNN；
   - INT8物理输出 vs MPC `U_true`；
   - 分别报告四个输出的 MSE、MAE、RMSE、最大绝对误差和分位数。
5. 不得只报告总体平均误差，必须单独关注 thrust 和三个 body rates。

### 阶段 D：DPU 编译

1. 使用真实目标 architecture/fingerprint 编译。
2. 保存完整编译命令、compiler report、层映射、CPU fallback信息和最终模型。
3. 若任何层回退CPU，明确报告，不得称为纯DPU模型。
4. 检查 batch=1、输入25维、输出4维和层顺序。

### 阶段 E：DPU 数值和性能验证

1. 先运行1000条 golden vectors，不接入AirSim。
2. 比较 DPU、INT8仿真、FP32 PyTorch 三方输出。
3. 报告：
   - 单次推理 latency 的 mean、P50、P95、P99、max；
   - throughput；
   - warm-up前后差异；
   - 端到端预处理/推理/后处理时间；
   - DPU runner数量和线程配置。
4. 三个控制器在20 Hz下，顺序调度的端到端总预算必须小于50 ms；建议稳态P99留出足够RPC和控制发送余量。

### 阶段 F：多 DNN 和闭环集成

1. 先做一个DPU runner的三组输入顺序调度；再根据目标DPU资源尝试多个runner/并发。
2. 三架无人机必须各自构造独立25维输入，不能把一架机的输出复制给其他无人机。
3. 在真正开始计时前对每个DPU runner做warm-up。
4. 如果DPU不在运行AirSim的Windows主机上，需要定义明确的网络协议：请求必须带 vehicle ID、周期序号和25个float输入，响应必须带相同序号和4个控制输出；必须处理超时、乱序和陈旧响应。
5. 最终复用 `run_three_drones_dnn_gates.py` 的独立路径、门组、穿门和碰撞判定，替换控制器backend为DPU。

## 10. 最低验收标准

后续阶段只有满足以下条件才可标记完成：

1. FP32重建与 golden vectors 最大绝对误差 `<=1e-5`。
2. INT8误差按四个输出分别报告，并证明不会导致控制裁剪或明显系统偏置。
3. DPU实际输出与INT8仿真结果一致到工具链预期范围。
4. 三DNN端到端稳态时延满足20 Hz；不能只报告裸DPU kernel时间。
5. 最终三机闭环实现15/15有效穿门，无运行中碰撞或超时。
6. 保存可复现命令、环境版本、日志、模型、量化/编译报告和最终JSON结果。

若量化后闭环失败，优先检查归一化、输出反归一化、输入顺序、NED坐标、pitch/yaw符号、DPU tensor scale和异步响应对应关系；不要第一时间重新训练。

## 11. 明确的已知边界

- 论文没有公开完整MPC动力学参数、约束、求解器和执行器映射；当前教师是 FALCON-compatible engineering baseline，不是exact paper reproduction。
- 当前 MPC 没有直接使用输入中的 measured acceleration；训练时随机 acceleration 主要让DNN学习对此维度的稳定/不变性。
- 当前Windows三机验证使用同一模型结构和权重的三个独立实例，不包含论文完整的动态优先级调度器。
- 当前验证对象是3架无人机，不代表已经复现论文最大规模并发数量。
- 不知道目标板卡/DPU architecture 时，禁止生成假定目标的最终编译产物。

## 12. Windows 复现实用命令

训练Python：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
```

三机DNN：

```powershell
& $trainPy .\run_three_drones_dnn_gates.py `
  --model .\dnn_runs\iid_path_v1_depth10_width512\controller_scripted.pt `
  --normalization .\dnn_runs\iid_path_v1_depth10_width512\normalization.json `
  --device cpu `
  --epsilon 0.8 `
  --seed 0 `
  --speed 2.0 `
  --group-spacing 6 `
  --gate-count 5 `
  --gate-width 4 `
  --gate-height 4 `
  --gate-clearance 0.25 `
  --per-gate-timeout 30 `
  --gate-asset 1M_Cube `
  --report .\dnn_runs\iid_path_v1_depth10_width512\three_drone_dnn_seed0.json
```

单元/接口测试：

```powershell
& $trainPy -m unittest discover -s .\tests -v
```

当前结果：49/49通过。

## 13. 后续 Agent 首条执行建议

先不要运行训练。先读取本文件、`export/model_spec.json`、`export/export_manifest.json`、`normalization.json`，检查目标 Vitis AI 1.2.1 环境和DPU architecture。随后建立独立的 `vitis_ai_1_2_1_deploy/` 工作目录，复制而非移动冻结产物，完成 FP32模型重建和 golden parity。每个阶段都报告命令、输出路径、数值指标和剩余任务。
