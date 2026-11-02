# 反馈长等待排查：尚未修复，下一步需要内核跟踪

> 2026-10-27更新：内核记录已分析，定位到Torch运行库硬缺页和同核Edge线程竞争；已增加可关闭的驻留候选并完成七机两轮物理救回及固定20ms对照。尾延迟尚未全部修复。详见[后续完整结果](FEEDBACK_KERNEL_FINDINGS_AND_RECOVERY_20261027.md)。下文保留2026-10-26时的排查状态。

## 当前结论

本轮没有得到可接受的修复。全部未通过的功能性候选已回退，保留诊断计时、完整记录和采样工具。原DNN权重、归一化、训练产物没有修改。恢复后的38项相关检查通过。

已经确认，之前Drone1约79ms的连续控制间隔不能解释成DNN计算队列等待：该机优先级最高，关键任务进入后端后首次排队约0.18ms，分段计算累计约3.72ms。原始区间仍见`frontend_order_ablation_20261026_v1/analysis.json`。

目前仍不能把回读后那段32.45ms唯一归因到某个函数、排序算法、GIL或Windows调度。后续记录中，长墙钟时间散落在输入处理、传感分析、消息读取和等待等不同位置；一些几十毫秒操作消耗的线程CPU周期与正常约1ms操作相近。它提示相当部分时间没有用于该线程的实际计算，但不能单凭周期数分清锁等待、I/O等待、缺页或未得到CPU。

因此应停止继续组合调参，获取内核级线程运行、就绪和等待记录，再针对实际阻塞点修改。

## 本轮尝试与结果

本轮共保留43次完整窗口记录，其中包括诊断和失败候选。早期尝试范围过宽，且部分候选组合了多个变化；这些结果不能用来单独证明某一项变化的因果效果。后续工具已加入正常工况门槛和`STOP`标记，失败基线不再继续展开扰动对照。

下表“未按时更新”同时计算缺失周期和原截止时间之后返回的更新，不只统计完成任务的平均响应。每个正常工况应计分2141次更新。各行不是独立随机重复，不能据此建立性能优劣结论。

| 目录后缀 | 试验数 | 正常工况未按时更新 | 正常请求响应P99 | 动态扰动物理通过 |
|---|---:|---:|---:|---:|
| diagnosis：原实现加计时 | 2 | 138 | 37.67ms | 0/1 |
| repair1：管道监听、检查与读取改动 | 6 | 719 | 65.77ms | 0/2 |
| sensor：取消重复回读、复用命令后的观测 | 6 | 259 | 30.17ms | 0/2 |
| deadline_slots：前端EDF并保留迟到周期任务 | 6 | 2006 | 85.15ms | 0/2 |
| priority_deferred：提高私有进程优先级、飞后评分 | 6 | 660 | 63.79ms | 1/2 |
| single_blas：进一步约束数值库线程 | 6 | 110 | 37.76ms | 0/2 |
| short_gil：缩短子进程线程切换间隔 | 6 | 2138 | 4660.48ms | 不用于比较，曾人工暂停 |
| per_loop_io：每机独立RPC进程、共用原I/O核心 | 3 | 938 | 66.27ms | 正常计时回归后停止 |
| native_events：发送方原生事件通知 | 1 | 625 | 48.16ms | 正常计时回归，未展开 |
| wait_chain：等待链采样，诊断专用 | 1 | 553 | 48.86ms | 未展开 |

目录统一位于`F:/FalconReproduction/airsim_control/controller_architecture_study/results/feedback_handoff_20261026_*`。每次的源码、条件、状态、控制和失败记录均保留；未覆盖之前结果。

取消重复回读降低了完成任务的响应时间，却增加了缺失更新，因而未接受。保留迟到任务减少了跳过机会，却可能导致积压，也未接受。提高优先级和飞后评分出现一次物理救回，但没有重复成功，且时序仍失败。多RPC进程增加了解释器和内存开销，没有改善正常基线，已撤销。

缩短子进程线程切换间隔出现严重回归，已撤销。其操作暂停记录在`short_gil/operator_abort.json`；暂停发生在试验期间且没有可靠阻止整个序列，之后记录不能作为公平性能对照。后续已实现试验之间检查`STOP`，不依靠暂停来停止后续试验。

## 进一步排除了什么

- 当前执行作业的CPU配额控制标志为0，没有观察到作业CPU硬配额；查询到的进程节能调度状态也未启用。
- 试验外内存核对约有8.9GB可用、交换区占用约0.26GB；这不能排除试验中的缺页，但不足以声称已发现严重换页瓶颈。
- NumPy和SciPy确实各配置24个BLAS线程，而PyTorch为1。相同核心上的前处理微基准，24线程与1线程的P99约0.62—0.69ms，差异很小。因此不能将几十毫秒长等待直接归因于BLAS线程数。
- Windows等待链采样得到623条记录，没有查询错误，但没有识别出能解释原32ms的具体锁链。该接口对未支持的等待也可能只返回一个线程节点，不能把它的“running”直接当成线程一直占用CPU。[Microsoft接口限制](https://learn.microsoft.com/nl-nl/windows/win32/api/wct/nf-wct-getthreadwaitchain)

微基准：`priority_deferred/blas_microbenchmark.json`。等待链：`wait_chain/wait_chain_samples.json`。分阶段墙钟及CPU周期：各试验`frontend_spans.json`。

## 已恢复的代码状态

六个基础模块与本轮最初的诊断基线逐字节一致：

- `framework_cpu_affinity.py`
- `framework_runtime_wait.py`
- `framework_runtime_wire.py`
- `framework_runtime_worker.py`
- `framework_runtime_io.py`
- `inflight_sensing.py`

所以提高进程优先级、改变BLAS线程数、缩短子进程切换间隔、额外RPC进程、新的管道通知和取消回读均没有留在默认运行中。

`probe_seven_continuous_timing.py`保留可关闭的阶段诊断、线程CPU周期和UTC/单调时钟锚点；`trace_frontend`默认关闭。控制流程、原截止时间、固定模型、输入新鲜度和原周期规则保持基线行为。

新工具`probe_feedback_handoff.py`保存源码与试验条件，增加前端PID/TID、正常工况门槛、试验间停止标记及仅跑正常工况选项。正常工况门槛178来自之前两次`category`正常对照的较差值，仅用于阻止明显回归，绝非严格实时验收通过标准。

## 为什么现在需要管理员PowerShell

已经实际尝试：

```text
wpr.exe -start GeneralProfile -filemode
Failed to enable the policy to profile system performance.
Error code: 0xc5585011
```

这是Windows的系统性能采样权限限制，不是模型、训练或自动审批拒绝。当前执行令牌没有系统性能分析权限。需要从管理员PowerShell启动采样，才能继续区分线程在运行、就绪等待、同步等待和内核/驱动干扰。

已准备脚本：

`F:/FalconReproduction/airsim_control/controller_architecture_study/run_feedback_kernel_trace.ps1`

保持当前12机Play，在**管理员PowerShell**复制下面整段：

```powershell
Set-Location -LiteralPath 'F:\FalconReproduction\airsim_control'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_feedback_kernel_trace.ps1
```

脚本检查管理员身份和记录器空闲状态；只运行一次无扰动的七机正常工况；在`finally`中停止自身启动的WPR记录，并输出结果目录和同名`.etl`路径。它不会停止其他已存在的WPR记录。整段包含模型、进程初始化和约12秒飞行窗口，通常需约1—2分钟，具体取决于机器负载。

本次是诊断采样，不能作为最终性能验收。结果目录包含`plan.json`中的前端PID/TID，以及`report.json`中的控制窗口和时钟锚点，后续只分析该窗口。

## 拿到ETL后的唯一下一步

先查看长等待发生时前端线程究竟处于Ready还是Wait，并检查对应调用栈、唤醒者、缺页及DPC/ISR。若是特定同步点，修改该处；若是CPU调度干扰，处理该资源边界；若是状态/RPC路径，缩短相应链路。拿到这个证据之前，不再扩大模型、门宽、扰动或调度参数网格，也不把更多加速器核心作为已验证的解决方案。

最终仍需同条件正常工况和少量静态/动态配对，分别报告物理通过与时序通过。当前没有任何候选被标记为“修复完成”。
