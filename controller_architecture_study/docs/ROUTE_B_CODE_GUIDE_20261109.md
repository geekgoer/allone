# 路线 B 代码与实验阅读指南

更新于 2026-11-09。本文帮助协作者定位当前真实板端双机实验的源码、数据流和证据。当前用户指示及后续进度记录优先。先读[总体框架](../../docs/FRAMEWORK.md)和[协作进度](../../docs/COLLABORATION_STATUS.md)，再按下列顺序阅读。日期文档记录特定时点的设计或实验，不自动代表当前运行方式。

## 当前研究目标与边界

研究多架无人机共享加速器时，怎样利用状态和时间约束，以低开销调整控制任务周期和执行顺序，兼顾局部安全、总体控制性能与资源需求。FALCON 是研究起点，当前实现并非完整原论文复现。状态相关预算、有限预测和实测条件均有假设，不应把结果写成已证明的物理安全保证。

最近的真实链路是 Windows 上的 C++ 日历与采样、Python 前端与在线决策，经 UDP 使用 Zynq 板端 DPU 的独立 237D 模型。当前模型固定为 ID 11，单板端工作线程按 EDF 或 FIFO 排整任务，当前实时入口的启发式主要改变周期。模型切换、DPU 内部分段抢占、超网上板和 Zynq PS 运行调度器均未在该链路中验证。

## 一条控制任务的调用链

1. [`fly_ip37_joint.py` 的 `run()` 和 `main()`](../fly_ip37_joint.py)读取实验参数、冻结路线、创建飞机和板端客户端。`--native-sampling` 启用 C++ 绝对时间日历与 AirLib 采样。`--split-native-prep` 是可选的实验性前端成本拆分，要求同时启用原生采样。
2. [`native_runtime/native_io.cpp`](../native_runtime/native_io.cpp)中的 `release_loop()`、`publish_release()` 和 `snapshot()` 到点发布并采集状态。`framework_native_io.py` 通过 ctypes 读取事件。每机一个在途名额，过期或忙碌任务有独立状态。
3. [`consume_native_calendar()` 和 `Pilot.run_job()`](../fly_ip37_joint.py)将采样事件转给飞机控制线程。`Pilot.publish()` 通过 [`held_control_safety.py`](../held_control_safety.py)计算持有控制下的预算和绝对期限，`JointRuntime.observe()` 绑定当前控制版本。
4. [`fast_convex_preprocess.py`](../fast_convex_preprocess.py)的 `FastReferenceWindow.window()`、`encode()` 生成参考窗口和 237D 输入。特征顺序与契约见 [`convex_features.py`](../convex_features.py)。
5. [`ip37_joint_runtime.py`](../ip37_joint_runtime.py)的 `enqueue()`、`take()` 在 Windows 端排队，[`fly_ip37_joint.py`](../fly_ip37_joint.py)的 `serve_board()` 使用 [`ip37_shadow.py`](../ip37_shadow.py)的 `predict()` 访问板端。动态加载的 UDP 客户端来自部署包的 `windows/pool237_udp.py`。
6. `Pilot.run_job()` 使用原任务绝对期限检查输入、推理结果和命令下发，再经 [`control_adapter.py`](../../control_adapter.py)与 [`airsim_interface.py`](../../airsim_interface.py)下发角速度和油门。记录中的命令返回采用 RPC/ping 代理，并非物理执行器实际生效时间。

## 在线决策与提交

[`plan_live()`](../fly_ip37_joint.py)运行后台循环。[`JointRuntime.snapshot()`](../ip37_joint_runtime.py)将各机的当前周期、未来释放、已生效控制对应的物理期限、在途任务和历史观测成本组成有限预测场景。状态缺失、旧任务过期或释放已经到期时可能跳过快照，见 `snapshot_skips`。

[`bottleneck_abc.py`](../bottleneck_abc.py)中的 `abc_repair()` 按独占路径来不及、窗口资源需求超载、任务调度阻塞三个方向提出有界修改。它调用 [`bottleneck_joint.py`](../bottleneck_joint.py)生成任务并模拟 PS/PL 阶段，完整检查所有回路及每机工作线程复用。这里的成本为经验值，模拟器可表达分段不代表当前板端 DPU 支持分段抢占。

[`JointRuntime.install()`](../ip37_joint_runtime.py)检查快照版本、决策预算、同机复用和未来释放。原生模式经 [`NativeCommandIO.guarded_batch()`](../framework_native_io.py)与 C++ 日历锁进行整组复核和提交。已发布任务保留发布时的周期与 `deadline = release + period`，失败组不部分安装。当前路线 B 的实际命令仍由 Python 路径负责，阅读旧路线 A 时需分别核对。

## 时间与报告字段

`release_s` 是计划发布时间，`published_s` 是实际发布时刻，`sample_start_s` 是采样开始，`deadline_s` 是本任务的名义周期期限。`command_return_s` 是命令提交及 ping 返回的代理时间，`finished_s` 还包含可能的后处理与工作线程归还。`safety_deadline_s` 是按观测状态及控制估算的绝对物理期限。

`timing_passed` 针对任务名义周期内交付，`physical_deadline_missed` 另记物理预算超期。`flight_passed`、`body_clearance_passed`、`sim_clock_passed` 和 `calendar_coverage_passed` 是不同检查。单次 P99 或历史最大值不能当作认证 WCET。`--split-native-prep` 将采样和线程交接的等待保留在响应路径，将线程接手到输入就绪的实测经过时间计为共享服务估计，该经过时间仍可能包含线程等待。

每次运行目录的 `setup.json` 保存配置和源码哈希，`requests.json` 保存各任务阶段，`decisions.json` 保存候选、预测、拒绝和安装，`result.json` 汇总运行。`evaluation_*.json` 保存独立评价采样，评价实现见 [`ip37_evaluation.py`](../ip37_evaluation.py) 和 [`safety_comparison_core.py`](../safety_comparison_core.py)。报告中的 J 必须结合该次评价范围和公式解释。

## 最近的证据与下一步

[问题日志 B50](ROUTE_B_ISSUE_LOG_20261109.md)记录了原生采样、成本拆分、Windows 线程计时粒度和最后试飞。最终配对位于 `results/ip37_native_prep_worker_wall_20261109_120420/`，同路线和模型，双机从 50/50 ms 开始。合并成本组 1002/1004 按期交付、零周期修改；拆分组 646/646 按期交付、10 次周期修改。两组都 24/24 门、零碰撞。拆分组在第 4 门附近仍有条件 B 超期，没有证明危险阶段加频收益。

拆分组曾在采样完成后约 46 ms 才被 Python 轮询到。第 4 门窗口的无在途快照中，部分首义务到决策时已经过期。下一步应分别核对最早可行动的风险信号与 C++ 到 Python 的交接长尾，再决定 Windows C++ 热路径和 Zynq PS 迁移的范围。

## 必要检查与协作材料

- [`tests/test_ip37_joint.py`](../../tests/test_ip37_joint.py)覆盖发布、期限、在途任务、提交与控制链语义。
- [`tests/test_framework_native_io.py`](../../tests/test_framework_native_io.py)覆盖原生日历、发布竞态、旧期限和整组提交。原生测试需要先构建 DLL，见 [`build_native_runtime.ps1`](../build_native_runtime.ps1)。
- [`tests/test_bottleneck_abc.py`](../../tests/test_bottleneck_abc.py)、[`tests/test_bottleneck_joint.py`](../../tests/test_bottleneck_joint.py)和 [`tests/test_remaining_chain.py`](../../tests/test_remaining_chain.py)覆盖有界启发式、预测、前端等待和工作线程复用。
- [`ROUTE_A_CHECKPOINT_20261106.md`](ROUTE_A_CHECKPOINT_20261106.md)是旧路线恢复点。`incremental_supernet/`、`train_*.py` 和量化脚本属于训练或超网轨道，按需要单独阅读。
- 模型、部署包、路线和 `results/` 下实验报告可能未随 Git 同步，必须核对实际文件、清单及哈希。近期 IP37 候选包此前位于 `F:\FalconReproduction\237D_data\rec\ip37_h4_input8_rc1_20261106`，其 `manifest.json`、归一化和客户端共同构成版本化契约。该路径仅描述原机器，协作者应在本机准备对应资产；不得混用 12D、25D、237D 或超网产物。
- AirSim、板端 IP 和现有服务均是外部运行依赖。复现实飞前核对模型 ID、端口、路线、仿真时间、命令与输出路径。遵循用户当前授权；工作记录与失败报告保留。
