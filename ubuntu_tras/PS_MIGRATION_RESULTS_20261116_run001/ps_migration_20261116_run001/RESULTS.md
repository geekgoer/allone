# 路线B迁移到AXU3EGB PS：实测结果与交付

日期：2026-11-16。实验根目录：`/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001`；板端同名目录位于`/home/root`。

## 结论先行

**主控制链和最新义务日历调度器已迁移并实际跑通；不能宣称长尾已消失，也不能宣称真实跳过—补更验收完成。**

- 单机固定3轮、双机固定/动态各3轮均完成，正式6000条请求全部delivered。
- 另外授权的20ms探索对照和30ms三轮对照全部完成，压力12004条全部delivered；全部17轮共18004条，无输入拒绝、drop、晚返回、记录错误或碰撞。
- 全部这些真实控制请求实际执行的模型均为ID11；正常双机实际周期均50ms。没有实际换档/改周期/跳过，不把策略标签当策略收益。
- T30启用组三轮共89个含skip可行候选，**全部**因上下文版本已变化而未提交；实际planned_skip=0、实际补更=0。安全拒绝正确，但规划/提交活性问题尚未解决。
- 最终原18模型整包校验通过，AirSim只读确认`is_paused=true`。没有更新镜像、驱动、DNNDK库、权重或归一化，没有裁剪RAW237。

## 迁移范围和链路

Windows仅保留AirSim Blocks。PS执行绝对定时释放、状态/碰撞采样、多项式参考、237D构造、held-control保持分析、最新obligation finite calendar与联合搜索、本地真实DNNDK、物理角速度/throttle下发、命令返回与新held版本观测。

链路为`Windows 192.168.77.1:41451 ↔ AXU3EGB 192.168.77.2 PS → 本地DNNDK/DPU`。不再让WindowsPython构造237D/调度，亦不经板端UDP绕回本地推理。旧UDP服务及其包均未删除；不要与本实验同时控制同一无人机/DPU。

重点代码：`adaptive_runtime.cpp`、`adaptive_live.hpp`、`scheduler/live_calendar.*`、`scheduler/scheduler.*`。复用冻结`pool237_runner.cc`中的Pool；必要native_io接口移植为Linux时钟/线程/rpclib，详见MIGRATION_IMPLEMENTATION.md和NATIVE_IO_PORT_MAP.md。不是旧ABC入口换名。

RAW237仅在原Pool归一化一次并补齐240通道；原越界检查保留。C++RPC直接传物理wx/wy/wz，不重复Python客户端pitch/yaw翻转。

## 模型和平台身份

- 实际模型：`main18_plain_mlp_h4_w1024_seed0`，板端numeric_id=11。
- 冻结包：`/home/root/board_package_requant18_release001`。18个模型加载常驻；被选中的ID才实际执行。
- 实际input_scale=16，output_scale=0.0625；wire_tag=`9253dd1fc87f8d7f`。
- normalization SHA256：`fe9c44ee80aef1cf97138536a66a4ee3a126f459f2ef963370db354ada8f19ac`。
- registry SHA256：`040f029bb9bd160bfd1b055748630e87b76e127fe2315f80f73b516dba5bc2e7`。
- Linux5.4.0-xilinx-v2020.1、AArch64、SD第二分区ext4、原DNNDK；最终CPU governor=userspace，观测频率1066666kHz。没有配置PREEMPT_RT/实时线程策略。
- 最终boot_id=`5db7ec45-8983-480d-bc6d-b843c971c773`。不跨运行/启动周期比较绝对PS单调时间。

原快照2613文件、补查统一快照1878文件校验通过，5组关键文件逐字节相同，记录source_proof_001.json。原tar.gz在收到的目录中缺失，不能声明它本身校验通过。

## 前置核对

- 10组特征fixture共2370值最大差0；参考多项式最大差1.421e-14；30条保持分析最大差4.263e-14；3条原生命令线协议通过。
- 18模型各32条本机黄金：576条全部通过，最大差0。每模型100次预热，3轮各100次本地计时共5400条；原3条量程问题向量仍全部status3，未隐藏失败。
- 调度器6组冻结fixture、246条差分案例、43核心检查；最终ARM77生命周期检查及67接线检查通过。后两类单元检查不冒充实际DPU或跳过飞行证据。
- 最终24项Python分析回归通过；延迟读回clock假阴性有专门测试。

## 单机固定正式结果

Drone1，ID11，T50ms，参考匹配初态1.5s，每轮20s、400条。三轮均完整覆盖，5门均观察到机体净空穿越。

以下为**计划发布→命令+ping返回代理**，不是电机实际施加时刻。单位ms。

| 轮次 | 成功/计划 | P50 | P95 | P99 | 最大观测 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_single_002 | 400/400 | 4.418 | 4.71210 | 9.21819 | 9.852 |
| fixed_single_003 | 400/400 | 4.329 | 4.64075 | 9.37817 | 9.657 |
| fixed_single_004 | 400/400 | 4.405 | 4.73560 | 9.19208 | 9.552 |

输入准备/量化P50约0.014ms，DPU调用P50约1.303–1.305ms，输出处理P50约0.003ms，本地DPU服务P50约1.320–1.322ms。完整P50/P95/P99/max在CROSS_STAGE_FINAL_001.csv和各分析REPORT.md，不将分阶段分位数相加当成总分位数。

## 双机正常T50ms正式对照

同一冻结场景、匹配初态、每轮20s；每机400条，共800条。六轮全部通过日历/计时及采样飞行审计，60/60门有机体净空穿越证据。全窗口RMSE约0.165–0.172m（含显式暂停边界代理、采样插值口径，不是连续物理真值）。

| 轮次 | 固定P50/P95/P99/最大(ms) | 动态P50/P95/P99/最大(ms) |
| --- | --- | --- |
| 101 | 4.730 / 5.13510 / 9.19327 / 10.688 | 4.9475 / 6.05345 / 9.98017 / 11.990 |
| 102 | 4.6905 / 5.11900 / 10.11858 / 12.047 | 4.9545 / 5.74435 / 10.03205 / 11.734 |
| 103 | 4.6655 / 4.99105 / 5.21501 / 6.869 | 4.9345 / 5.82605 / 6.60795 / 11.902 |

动态实际安装keep_current为791/793/797次；实际DPU全ID11、实际周期全50ms。规划wall最大17.823/24.987/7.963ms。迁移后仍有网络、Linux调度和规划长尾；没有做匹配的本轮Windows端基线复跑，所以不能量化“比Windows改善多少”。

## 获准的独立压力：真实负结果

固定允许周期集合仅[20]或[30]，其余初始实测成本/候选模型/TRAIN512分数不改，运行中仅按真实观测单调扩大成本。没有人工sleep拖慢推理、篡改模型成本、CPU模型或输入裁剪。受限周期条件不能当正常T50工况。

| 条件 | 关闭/开启轮数 | 每轮总计划数 | 全部成功 | 实际跳过/补更 |
| --- | ---: | ---: | ---: | ---: |
| T20ms/机 | 1/1 | 2000 | 4000/4000 | 0/0 |
| T30ms/机 | 3/3 | 1334 | 8004/8004 | 0/0 |

T20启用组有717条eligible权限，但目标释放只剩0.036–14.286ms，均不足15.432ms规划预算，因此没有进入skip候选检查。T30提供较多提前量，启用组三轮得到6/9/74个含skip可行候选，耗时约9.4–13.2ms；全部安装时context_changed，旧计划被正确拒绝，未形成planned_skip原始记录。

因此本次只验证了资格/期限/CAS拒绝路径，没有验证真实“已跳过→原下一周期实际补更”。后续优先处理有界搜索的执行成本与提交窗口，保持控制版本/期限守卫；不应通过关闭守卫或把候选当执行来凑结果。压力重复已足够揭示问题，不建议盲目继续加压或增加次数。

## 原始数据、版本和复跑

- 单机原始：fixed_single_002/003/004；最终分析analysis_fixed_single_formal_002。
- 双机正式：dual_fixed_101/102/103和dual_adaptive_101/102/103；最终分析analysis_normal_dual_v4_formal_002。
- 压力：pressure20_skip_{off,on}_001，pressure30_skip_{off,on}_{001,002,003}；最终分析analysis_pressure20_pair002和analysis_pressure30_formal_002。
- 每轮保留events.jsonl、requests.jsonl/CSV、plans.jsonl、calendar_final、模型/执行身份、准备场景、初末状态、命令版本及所有状态结果。逐相位CSV、模型/周期/跳过审计CSV在对应analysis目录。
- local_validation_002保留576黄金、5400计时和3条拒绝原始证据；environment_final_001保留系统/库/启动身份；logs保存实际命令、UTC起止、退出码和输出。
- CROSS_STAGE_FINAL_001.csv是17轮跨阶段紧凑汇总，不替代原始日志。

实测二进制：

| 用途 | 本地/板端对应 | SHA256 |
| --- | --- | --- |
| 单机正式 | build_flight_004/ps_fixed_control → bin/ps_fixed_control_v4 | 8034206332be9448b129c8019090c9c73c8d4bd3b76d8f2a46f9d039361eb989 |
| 双机正式 | build_adaptive_004/ps_obligation_control → bin/ps_obligation_control_v4 | b31917c473922eaf6715fff0c334e4e5ce2d99d96b397b5d7a2dcf321e1d1df8 |
| 压力 | build_adaptive_005/ps_obligation_control → bin/ps_obligation_control_v5 | 6b6a4b38e7e967cbd5d169f0df891b665ba9a79fb1cdcb91d17f166834873511 |

复跑命令及重构建依赖见RUNBOOK.md；源码解释见MIGRATION_IMPLEMENTATION.md。精确实测构建源码快照在build_adaptive_004/source与build_adaptive_005/source，不覆盖旧版本。

## 保留的失败及分析修正

本地第一次JSON对象失效、一次上传未完成导致Text-file-busy、fixed_single_001最终汇总导出失败、dual_adaptive_002约10s快照时序错误均保留，不算正式成功。后者已改为锁内采时，ARM回归通过后另跑完整正式轮次。

旧分析以暂停后最终readback观测时间算sim/wall，曾将pressure20-off因约1.06s日志排空延迟判作clock不通过。当前以活动请求首末样本验证clock，最终位置用原始pause_return上界代理对齐；原observed_us和旧分析目录不改，新分析另存。没有修改原始控制结果。活动clock比例约0.999，仍非一向时钟同步或物理执行证明。

## 当前未作的保证

没有证明WCET、硬实时、无限期常驻内存有界、真实电机施加延迟、动态调度收益、真实跳过补更或长尾消除。保持分析是冻结假设下的条件预算，非飞行安全认证。训练、教师求解、CPU DNN、超网及模型重校准均未进入本次迁移范围。
