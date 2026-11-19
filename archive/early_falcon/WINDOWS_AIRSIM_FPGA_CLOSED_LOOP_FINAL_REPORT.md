# FALCON Windows AirSim ↔ AXU3EGB DPU 闭环最终报告

日期：2026-09-10  
用途：交接给 Ubuntu/Vitis-AI Agent，作为后续论文级 FALCON 复现的已冻结 Windows 闭环基线。

## 1. 最终结论

Windows AirSim 与 AXU3EGB DPU 的 FLCN/v1 UDP 控制链路已经打通，并按顺序完成：

1. Windows → FPGA 冻结黄金向量烟雾测试；
2. 单机 CPU-DNN 控制、FPGA 并行推理的影子模式；
3. 单机 FPGA DPU 真闭环；
4. 三机共享 FPGA DPU、顺序调度的真闭环。

最终三机实验为 15/15 有效穿门，无运行中碰撞，无任务超时，无 deadline miss；1779/1779 个 UDP 请求得到新鲜合法回包，未发生 retry、协议错误、DPU 错误、控制裁剪或 CPU fallback。三架无人机实际施加的是各自对应的新鲜 FPGA 输出。

这证明了当前限定范围内的端到端闭环：

```text
AirSim state/reference
  → Windows 25-D feature construction and frozen normalization
  → FLCN/v1 UDP
  → AXU3EGB DNNDK/DPU INT8 DNN
  → physical [F, wx, wy, wz]
  → Windows existing control_adapter
  → AirSim actuator command
```

它不等价于“整篇 FALCON 论文已经完整复现”。当前实现是 3 个独立控制状态/参考、同一套 DNN 权重、一个板端 DPU task 的顺序推理基线。论文中若还包含不同网络、调度器、资源分配、更多飞行器、功耗/吞吐/可扩展性或真实飞行实验，需要在下一阶段逐项建立证据。

## 2. 固定软硬件合同

- Windows AirSim：`192.168.77.1`
- AXU3EGB：`192.168.77.2:5005`
- Ubuntu Vitis-AI：`192.168.77.3`，仅构建、部署和分析，不进入实时控制链路
- Vitis-AI：1.2.1 legacy DNNDK/n2cube
- DPU kernel：`falcon_25x4`
- UDP 协议：FLCN/v1
- 网络输入：25 个 big-endian float32，已经按冻结 `normalization.json` 归一化
- 网络输出：4 个物理控制量 `[collective_thrust_acceleration, wx, wy, wz]`
- 控制坐标系：AirSim NED
- 输出裁剪：`F ∈ [0,18]`，`wx/wy/wz ∈ [-3,3]`
- pitch/yaw 符号转换：只允许在既有 Windows `control_adapter.py` 中发生
- 目标控制周期：`0.05 s = 50 ms = 20 Hz`
- UDP runtime timeout：10 ms，最多重试一次相同 datagram
- fallback：同周期 Windows CPU DNN；禁止使用旧 FPGA 回包

## 3. 25 维输入合同

```text
0..2    position_error_xyz
3..5    attitude_error_xyz
6..8    velocity_xyz
9..11   acceleration_xyz
12..14  path_offset_1_xyz
15..17  path_offset_5_xyz
18..20  path_offset_10_xyz
21..23  path_offset_19_xyz
24      reference_speed
```

Windows 端复用 `state_adapter.build_controller_state` 和 `dnn_model_utils.build_policy_input`。未来路径偏移固定为 `future_reference_position[k] - current_vehicle_position`，`k=[1,5,10,19]`，不可改变顺序或符号。

## 4. 控制周期的实现与实测

当前控制周期不是 DPU 的约 1 ms 推理时间，也不是 10 ms UDP timeout，而是完整“读状态→构造特征→三机顺序推理→发送控制→等待下一周期”的 50 ms 调度周期。

`run_three_drones_dnn_gates.py` 中的关键实现：

```python
window = route.reference_window(period_start, dt=0.05, horizon=20)
manager.send_control_all(controls, calibration, duration=0.05)
next_deadline += 0.05
sleep_time = next_deadline - time.perf_counter()
if sleep_time > 0.0:
    time.sleep(sleep_time)
```

它使用绝对 `next_deadline`，每轮只睡剩余时间，避免把每轮计算时间累加成长期漂移；若计算超过 deadline，则不再 sleep，并记录 `deadline_miss=True`。

三机实测控制台的 loop period 基本在约 49.5–50.5 ms，573 个控制周期的 `deadline_misses=0`。三机每周期顺序执行三个后端，CPU+DPU+UDP 单机后端平均时间分别约 3.258、2.896、2.911 ms，总计约 9.07 ms，仍明显小于 50 ms 周期预算。

不要孤立修改一个 `0.05`。该数值同时定义：

- Windows 调度周期；
- AirSim actuator command duration；
- reference window 的时间间隔；
- MPC teacher 的离散时间；
- DNN 训练时未来索引 `[1,5,10,19]` 对应的实际预测时间范围；
- 模型描述中的 `sample_period_seconds`。

因此当前模型只能声明在 20 Hz 合同下得到验证。若要改为 10、25、50 或 100 Hz，应统一参数化上述位置、重新生成/核对 teacher 数据、重新训练/量化，并重新完成全部闭环验收；只改变 sleep 或 command duration 会造成训练与运行语义不一致。

## 5. 阶段 A：Windows UDP 黄金测试

证据：`dnn_runs/iid_path_v1_depth10_width512/fpga_udp_smoke_windows.json`

```text
status                         PASS
FPGA vs Caffe INT8 max_abs     1.2328338617e-7
acceptance limit               2e-6
timeout/retry/protocol error   0 / 0 / 0
single-packet RTT              5.214 ms
DPU time                       1.078 ms
server total                   1.080 ms
```

说明：单包 RTT 包含首次调用开销；后续影子和闭环长序列的稳态 RTT 更低。

## 6. 阶段 B：单机影子模式

证据：

- `dnn_runs/iid_path_v1_depth10_width512/fpga_shadow_single.json`
- `dnn_runs/iid_path_v1_depth10_width512/fpga_shadow_single.jsonl`

此阶段实际控制来自 Windows CPU FP32 DNN；同一份 25 维输入送给 FPGA INT8 DNN，FPGA 输出仅比较和记录。

```text
gates                           5/5
flight cycles                   503
fresh FPGA responses            503/503
fallback/timeout/retry/error    0 / 0 / 0 / 0
clipped/deadline miss           0 / 0
RTT mean/P99/max                1.613 / 1.935 / 3.488 ms
DPU mean                        1.035 ms
FPGA-vs-CPU max_abs             0.114105
```

## 7. 阶段 C：单机 FPGA 真闭环

证据：

- `dnn_runs/iid_path_v1_depth10_width512/fpga_closed_loop_single.json`
- `dnn_runs/iid_path_v1_depth10_width512/fpga_closed_loop_single.jsonl`

```text
controller                      fpga
gates                           5/5
flight cycles                   503
fresh FPGA responses            503/503
fallback/timeout/retry/error    0 / 0 / 0 / 0
clipped/deadline miss           0 / 0
RTT mean/P99/max                1.654 / 1.886 / 3.300 ms
DPU mean                        1.035 ms
FPGA-vs-CPU max_abs             0.116546
```

逐周期 JSONL 证明 `applied_control == fpga_control`；CPU DNN仍在 Windows 计算，用于对照和故障 fallback，但本实验从未采用 CPU fallback。因此准确表述是“FPGA-actuated, CPU-fallback-capable, fallback unused”，而不是“系统中完全没有 CPU 推理”。

## 8. 阶段 D：三机 FPGA 真闭环

证据：

- `dnn_runs/iid_path_v1_depth10_width512/fpga_closed_loop_three.json`
- `dnn_runs/iid_path_v1_depth10_width512/fpga_closed_loop_three.jsonl`

场景：3 架无人机、每机独立 5 门、`epsilon=0.8`、`speed=2.0 m/s`、共享一个 FLCN/v1 socket 和全局 sequence、每周期 Drone1→Drone2→Drone3 顺序请求板端单 DPU task。

```text
completed                       true
valid crossings                 15/15
control cycles                  573
UDP requests incl. warm-up      1779
fresh responses                 1779/1779
timeout/retry/discard/error     0 / 0 / 0 / 0
CPU fallback                    0
control clipping                0
deadline miss                   0
runtime collision               0
elapsed                         35.927 s
```

每架机 FPGA/网络统计：

| Vehicle | RTT mean | RTT P99 | RTT max | DPU mean | FPGA-vs-CPU max abs |
|---|---:|---:|---:|---:|---:|
| Drone1 | 1.612 ms | 1.823 ms | 3.227 ms | 1.035 ms | 0.102186 |
| Drone2 | 1.532 ms | 1.729 ms | 3.255 ms | 1.034 ms | 0.114249 |
| Drone3 | 1.525 ms | 1.690 ms | 3.131 ms | 1.034 ms | 0.152978 |

三个 `initial_collision_flags=true` 均为 AirSim reset/takeoff 后第一采样的 Ground 残留标记，运行时未再发生碰撞；代码只忽略第一采样的该标记。

## 9. Windows 实现文件

- `fpga_udp_client.py`：严格 FLCN/v1、持久 socket、全局 sequence、原报文重试、严格回包校验和统计。
- `fpga_udp_smoke_test.py`：冻结黄金向量测试。
- `fpga_shadow_backend.py`：`cpu_dnn`、`shadow_fpga`、`fpga`，新鲜回包选择、安全 fallback、裁剪和统计。
- `fpga_closed_loop_config.json`：IP、端口、timeout、retry、warm-up、fallback。
- `run_single_drone_dnn_gates.py`：单机影子/FPGA 闭环。
- `run_three_drones_dnn_gates.py`：三机独立输入、共享 socket/全局 sequence、顺序调度。
- `tests/test_fpga_udp.py`：协议与控制源隔离测试。
- `WINDOWS_FPGA_INTEGRATION_STATUS.md`：阶段执行命令和状态。

离线测试：`python -m unittest discover -s tests -p 'test_*.py'`，结果 `Ran 53 tests ... OK`。

## 10. 冻结文件 SHA-256

```text
fpga_udp_client.py
85A0F773E33F2A98A92FC5B3BC66399563072460AEACEB2500CA2286F151AD40

fpga_shadow_backend.py
83AA31B1EC27928AB2792D9A02AE49AACBE7611F6AB06A31DA6F426E84EAFBAD

fpga_closed_loop_config.json
DDB22DD8EF2D3AA0A1100BD97F02A584CA2B804785802D644336BDA410036B0D

run_single_drone_dnn_gates.py
B74B838339574EFC425FF86AABC2A40AA126EA19948DDE02A4692359DD812F6E

run_three_drones_dnn_gates.py
937BFD1E975361195C577A197121369F525CA0A6068750E9BAEDC19062898AE7

normalization.json
5066D8A05629EB98B7F2717883B8DC125A7FF1A2366C40421255D89C47E93BEE

controller_scripted.pt
7EDE899CDE770CE209A1C0AB337F844DA6C202DBC13A3E1665E0522B6D8FCE9E

fpga_udp_smoke_windows.json
C89638D638D2015749E7CEF862938E507C09245F94B93FABAA32BC829825330B

fpga_shadow_single.json
6C49920ABE041B51F6CF2BC948BFAD910B1CAD7EC0205DC44BB7720D516DADA2

fpga_closed_loop_single.json
6CD0A053A2D55D798CEB7F019C1B2BC3AB91F6BCE9F893B1FBC9298F038C65AC

fpga_closed_loop_three.json
897AD617389F4512E69E7E9D9CEEB16CC5CA816DA0336C7BB3D70BBAB998CAA4
```

## 11. 交给 Ubuntu Agent 的下一阶段任务

Ubuntu Agent 应把本报告视为“Windows↔FPGA 三机闭环基线已通过”，不要重新训练、重新量化或改变 FLCN/v1 来重复已经完成的工作。

推荐下一阶段：

1. 将论文中的系统组成、调度算法、硬件映射和实验指标逐项建立 gap matrix，明确哪些已经由当前基线覆盖，哪些尚未覆盖；不得把三机顺序推理基线直接称为完整论文复现。
2. 冻结并归档当前 board package、DNNDK server、INT8 golden、Windows 报告及上述 SHA-256，形成可回退基线。
3. 针对论文要求实现/验证尚缺的 FALCON 调度与多控制器管理机制。当前板端只有一个常驻 `falcon_25x4` kernel/task，三个请求顺序执行；若论文要求多模型、多 DPU task、动态优先级或 deadline-aware scheduling，应在不破坏 V1 基线的独立版本中实现。
4. 建立控制周期压力测试：在 50 ms 合同下逐步增加逻辑控制器数量或负载，报告完整端到端 latency、deadline miss、吞吐和 DPU 利用率；区分 DPU tensor I/O 时间、UDP RTT、Windows 特征/RPC 和完整周期。
5. 根据论文实验表逐项补齐资源、功耗、吞吐、实时性、控制质量和可扩展性证据；只有实际测量的结果才能写入最终复现结论。
6. 如果修改板端协议或服务，保留当前 FLCN/v1 兼容入口和当前冻结包，先做 golden regression，再重新执行 shadow→单机→三机闭环。
7. AirSim 实时闭环仍应保持 Windows `192.168.77.1` ↔ AXU3EGB `192.168.77.2` 直连；Ubuntu `192.168.77.3` 不应被插入每周期实时路径。

## 12. 声明边界

已经证实：FP32 DNN 可用、INT8/DPU golden 正确、Windows UDP 协议正确、单机与三机 AirSim 在 20 Hz 下由 FPGA 新鲜输出完成门轨迹，且本次实验无 fallback。

尚未由本报告证实：任意环境/速度的全面鲁棒性、超过 3 架无人机的可扩展性、论文全部调度算法、多个不同 DNN 的并发映射、功耗/资源结论、真实飞行器闭环，以及整篇 FALCON 的完全复现。
