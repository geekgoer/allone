# native_io 平台接口复用与义务日历接线说明

本文件只说明当前实验副本如何复用现有 Windows 原生 I/O 语义，以及哪些部分还属于下一阶段接线要求。它不是“完整双机义务调度已经通过”的验收报告。

工作目录：`/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001`。

## 1. 实际复用范围，不是编译 Windows DLL

当前参考平台文件为：

`/home/cwc/dpu_work/simplify_windows_framework/route_b_unified_20261116/platforms/windows/native_runtime/native_io.cpp`

没有将这个包含 `windows.h`、`HANDLE`、`__declspec(dllexport)` 的文件原样编译成 Linux 程序，也没有在 PS 上保留 Python → Windows DLL 桥接。迁移复用的是其已经定义的必要平台语义：每架飞机独占一个原生 RPC 客户端和 I/O worker；定时释放不等待上一条请求完成；过期请求、忙请求和完成事件分别记录；命令使用原生 AirSim RPC、物理角速度和 ping 返回代理。

必要平台替换如下。

| Windows 原接口/行为 | PS 实验副本中的接口 | 当前状态 |
| --- | --- | --- |
| `clock_s()`：QPC 单调时钟 | `rpc/airsim_rpc.cpp::monotonic_ns()` 使用 `CLOCK_MONOTONIC`；运行器使用 Linux `steady_clock` 的同一单调时基 | 已实现；AirSim 仿真时间单独记录，不与 PS 绝对时钟相减 |
| Win32 `CreateEventW`、completion/changed event、waitable timer、`WaitForMultipleObjects` | `std::thread`、mutex/condition_variable、单调绝对时间等待；结果交给持有者或日志队列 | 平台替换；不是宣称 Linux 默认线程已经具有硬实时保证 |
| `Runtime::run_loop()` 中每机独立 `MultirotorRpcLibClient` | `rpc/airsim_rpc.hpp/.cpp::Client`；`flight_runtime.cpp::Vehicle` 每机各持有一个 client/worker | 单机固定链已实跑；不把共享客户端用于并发 RPC |
| `snapshot()` 的 state RPC 后 collision RPC | `Client::get_state()` 后 `Client::get_collision()`；分别记录 `state_ready_us`、`collision_ready_us` | 保留 NED、wxyz、仿真 timestamp 和碰撞信息 |
| `Loop::busy` 加 `pipeline_busy`，不允许同机流水线重入 | 固定运行器 `Vehicle::busy` 跨越采样、PS、DPU、下发和成对刷新；日历版本在 `scheduler/live_calendar.*` 中管理逻辑请求归属 | 固定版采用更直接的整条流水线 busy；没有伪装成逐字段照搬 Windows 内存结构 |
| 异步命令后 ping，不 join 命令持续时间 | `Client::move_rates_throttle()` 的 async_call + 同连接 ping | 已实现并用延迟命令 mock 验证；ping 不是物理执行完成确认 |
| `stop()` 唤醒并 join 所有线程 | PS worker/执行器/日志线程的 stop、notify、join 和析构清理 | 已实现；不能硬抢占已经提交的非抢占 DPU 调用 |
| CPU affinity mask | 当前 PS 固定实验未设置 affinity，默认 `SCHED_OTHER` | 明确记录为未启用，不虚构与 Windows mask 相同的性能条件 |

`rpc/` 使用同一交接快照中的 rpclib 2.3.0 和 AirSim RPC 字段契约，而不是另选一套消息格式。PS 程序不需要完整 AirLib 仿真器、Windows DLL 或 Python 客户端运行时。协议 mock 和准备场景 mock 的复跑证据见 `rpc/VALIDATION_20261116.md`。

## 2. 定时释放、忙与过期：保持原本的语义

Windows `release_loop()` 的关键顺序是：

1. 根据单调时钟检查各飞机日历是否到期；`while (next <= now)` 会处理已经错过的每个计划槽，而不是忽略历史槽。
2. 保存该槽的计划释放时间，再调用 `publish_release()`。
3. 在 `publish_release()` 中，先判断 `finished >= planned + period` 是否已经过期；仅在未过期时才判断 `pipeline_busy || busy`。因此“错过期限”不能被一概记成“忙”。
4. 无论该槽成功、过期或忙，原日历都按 `next += period` 前进；不是“上一条完成时间 + period”。
5. 仅成功接纳的槽才提交采样任务。完成事件释放该请求的流水线归属。

固定版 `flight_runtime.cpp` 采用 `release = origin + k * period` 的闭式固定周期形式。固定周期时，它与上述 `next += period` 的计划时间等价：上一条未完成时后续槽仍按原时刻发布，并记录 `dropped_busy_io`；发布时已经过期的活动槽记录 `dropped_release_expired`，检查发生在 busy 之前。飞机已经退役则单独记录 `not_released_vehicle_retired`，不是拿 busy 掩盖退役。发布线程不等待 DPU future、RPC 或 worker 完成。

这只是固定配置阶段的释放器。**义务日历允许改后续周期、相位或计划跳过时，不能继续让这个固定 `for(k)` 成为第二个权威日历。** 接下来的 `adaptive_runtime.cpp` 必须直接以 `LiveCalendar::release_due()` / `next_wakeup_ms()` 驱动实际采样；仅 `status == sampling` 的请求可以交给 worker，其他槽仍须落原始记录。不能只在旁边调用一次 planner、但实际请求仍按旧固定数组发布。

## 3. 完成事件和时间戳如何对应

没有复用 Windows DLL 的二进制 `Event` ABI；复用了事件含义并把它们写成请求记录和本地生命周期调用。

| Windows `Event.kind` | PS 固定链对应记录/动作 | 后续日历运行器必须接入的动作 |
| --- | --- | --- |
| 4，计划释放 | `planned_release_us`、`actual_publish_us`、接纳/过期/busy/退役状态 | `release_due()` 接纳的具体 vehicle/cycle；不可重新编号覆盖 |
| 6，释放触发采样；1，普通采样 | `sample_start_us`、`state_ready_us`、`collision_ready_us`、仿真 timestamp；下发后还有 paired refresh | `observe()` 必须绑定这次状态与当时 held-control version；RAW237 就绪后 `queued()` 冻结实际期限 |
| 2，命令返回代理 | `command_submit_us`、`command_enqueued_us`、`command_ping_start_us`、`command_return_us`，以及实际 effective control | 确实下发且获得返回代理后调用 `command_return()`；即便晚到也更新 held-control version，并使旧证书失效 |
| 3，命令及随后的成对状态采样完成 | `paired_sample_start_us`、`paired_state_ready_us`、`paired_collision_ready_us`、`paired_analysis_ready_us`、`slot_return_us` | 新状态必须与新 held control 配对；随后 `finish()` 解除该请求归属，按真实结果处理补更义务 |
| 5，I/O 错误 | 原始 `error`、请求状态、退役原因；完成的记录由专用 journal 保存 | 在真实错误路径同样解除/退役，不能伪造成功完成或偷偷改用 CPU 推理 |

本地 DPU 另外保留：入队、取队、输入准备起止、`dpuRunTask()` 提交和返回、输出就绪等时间。`dpu_submit_us` 到 `dpu_result_readable_us` 是 **DPU 调用耗时**，不是单独测得的纯硬件执行时间。未实际推理的丢弃请求不得用其空值/零值参与推理耗时统计。

命令 ping 仅延续 Windows 基线的返回代理口径，不证明电机、飞行器或仿真物理已经应用该控制量。不能通过相减 Windows QPC、PS 单调时钟和 AirSim 仿真 timestamp 得到单向物理延迟。

## 4. 物理角速度与 RAW237：两个不能改错的接口

原生 `native_io.cpp` 已明确直接使用：

```cpp
client->moveByAngleRatesThrottleAsync(
    float(task.control[1]), float(task.control[2]),
    float(task.control[3]), float(throttle), 1.f, loop.name);
client->ping();
```

Windows Python control adapter 面向 Python 客户端时会翻转 pitch/yaw，而 AirSim Python 客户端在 RPC 边界又翻转一次。**C++ AirLib/rpclib 不做第二次翻转，因此 PS 必须直接传物理 `wx, wy, wz`。** 当前固定运行器和准备程序均调用 `psfeatures::native_rpc_command(prepared)`，而不是直接拿 Python-facing `prepared.command` 的 pitch/yaw 字段。物理输出限幅、推力到 throttle 的校准，以及实际 effective held control 仍分别保留。

RAW237 构造保留：实际状态、参考轨迹窗口、姿态/速度/参考等原有特征顺序。`rpc/` 不做归一化，特征构造后也不额外归一化。`local_dpu.cpp` 复用冻结 `pool237_runner.cc` 的模型池、归一化、237→240 padding、输入范围检查及 DNNDK 推理入口；因此 RAW237 **只在板端冻结推理路径归一化一次**。没有为了通过实验删除 status=3 越界拒绝、静默裁剪输入或换权重。

## 5. 调度策略来源：义务日历，不是旧 ABC 入口

当前策略参考是用户指定的：

- 原始 `obligation_scheduler.py`：有限控制义务日历、候选方案、PS/PL 联合可行性检查、提交/交付/保持界收紧。
- 原始 `ip37_obligation_runtime.py`：实际释放、队列、未来模型选择、dispatch order、held-control version、计划跳过及 pending recovery 的运行时生命周期。
- 原始 `fly_ip37_joint.py` 中 `--scheduler obligation` 的实际接线和命令/成对刷新回调。
- 整理后的统一包 `source/ps_core/scheduler.py`、`runtime.py`、`lifecycle.py` 是可读参考切片，而不是另一个训练/CPU 仿真控制器。

原始 `fly_ip37_joint.py` 同时保留 ABC 默认参数和多种旧实验入口，不应把该文件默认 `--scheduler abc` 或 `native_io.cpp` 顶部旧注释当成最新策略定义。该快照中 obligation 分支原本还明确限制了 native-sampling 的组合，因此 PS 所做的是把**义务日历的生命周期接到必要的原生平台接口**，不是声称一个原本不存在的 Windows 原生 obligation 全链被原样编译了。

`scheduler/scheduler.*` 和 `scheduler/live_calendar.*` 已有数值/生命周期回归验证，但单独这些测试不能替代双机真实 PS→DNNDK→AirSim 验收。后续 `adaptive_runtime.cpp` 的正确接线要求包括：

- 单一真实日历决定定时释放、模型/周期/相位变化；不维护与 planner 脱节的第二份固定周期表。
- 状态采样、参考与 RAW237、保持/新鲜度/周期/补更期限共同构成安全预算；预测成本必须包含真实 PS、RPC 及返回处理，不能把历史 Windows 固定网络附加值当成本地 DPU 硬件时间。
- 同一个本地 DNNDK executor 非抢占执行；`take()` 决定实际 DPU 派发顺序，真实完成时间进入 `dpu_done()`。
- 安装配置前在同一日历锁下核对 revision、已发布槽、held-control version、有效时限；不能部分安装多机方案。
- 实际命令返回推进控制版本；paired refresh 使用新 held control，随后才完成流水线。
- 跳过默认关闭。跳过/补更仅在双机基础对照通过后单独验证；planned skip 要保留原始槽记录，补更不能因超时、失败或只“计算出了输出”就被误标为偿还。

## 6. 已完成证据与尚未完成的边界

目前已有主机协议/几何 mock、板端数值/命令契约检查和单机固定配置的真实链路运行。早期 20 秒诊断曾保留 400 条 delivered 记录，但最终导出失败；这是**有热路径证据、仍有导出失败的诊断运行**，失败目录和日志必须保留，不能事后改称该次完整交付成功。后续修正导出的单机运行及原始结果由父任务另列验收，不能混写为同一次运行。

此映射文件本身不宣称三轮单机、双机固定/动态对照、计划跳过或补更已经全部验收。尤其是义务核心已通过回归与“真实运行器已经接通并完成实测”是两件不同的事。查看实际 `adaptive_runtime.cpp`、构建记录、板端原始请求及计划/提交日志，才能判定后续阶段完成程度。

本次范围明确不包含：训练、数据集采集、MPC 教师在线求解、CPU 模拟 DPU、CPU 分段 DNN、旧模型实验。参考轨迹生成和 237D 特征构造仍然保留在真实控制链中。

## 7. 冻结来源与 SHA256

统一包根目录：`/home/cwc/dpu_work/simplify_windows_framework/route_b_unified_20261116`。

```text
62eb3a0686e13eec5f106781940268fe57184e1648a2b8b6f71a05776939c04f  platforms/windows/native_runtime/native_io.cpp
78e05d662e502337bf4f7b433580b8e44b10b1072a4509475aaf221407fd3393  source/ps_core/scheduler.py
16aaad98ac7d52a99169feb018e77cd89f0dac9ed0e8deeef194e4a0ed2a666b  source/ps_core/runtime.py
8dad9baa54193bcfd9aa5e177dbb53b230036d39303f9b658c6693b0f3028fb4  reference_assets/deployment/pool237_runner.cc
```

最初已阅读并校验的原始调用链根目录：
`/home/cwc/dpu_work/windows_framework/route_b_ps_handoff_20261116/project/controller_architecture_study`。

```text
9eab45dfef8bea424c25c052e94e632927c83e506707c973e5c4d118a88b2dc8  obligation_scheduler.py
0c81f82391a02a73ae7532780842e83d70d2a8b04ef1eb9ee65e9091fc2e53e4  ip37_obligation_runtime.py
469025a232e9f4355e34e42dc76981b58da8e339edae05742502506b62503e5d  fly_ip37_joint.py
```

来源文件是只读参考；所有本次平台适配与实验修正均在新工作目录中，原模型、服务、源码快照及失败记录均应保留。
