# 任务释放语义修正：对应FALCON与最新版Framework

日期：2026-10-28。

后续已完成六组碰撞原因对照，见[修正后碰撞原因分析](POST_RELEASE_COLLISION_CAUSE_20261028.md)。本页保留释放机制修正阶段的原始结果。

## 判断

此前十机原型将任务释放与流水线执行许可混在一起，下一任务还受上一任务完整反馈回传的约束，并用上一次实际释放时刻继续推后计划。结果是参数写20ms，但实际任务释放平均约31ms。这个实现不能作为严格周期20ms的FALCON基线；也没有准确落实Framework已承诺的下一次释放时刻。应修正代码，而不是要求修改Framework去适应它。

FALCON原文PDF第6—7页（印刷页411—412），Sec.III-B Eq.(5)(6)与Sec.IV-A.3分别定义周期T、期限D=T和响应R：在kT记录状态，在kT+R施加控制。排队/抢占影响R，不应自动重置周期日历。其FPGA执行时间式不包含网络传输，但硬件在环实验包含通信，不能把本文Windows端到端实测直接等同于论文FPGA层执行时间。

原文件：`E:/BaiduNetdiskDownload/vitis-ai6.1/tmp/pdfs/FALCON.pdf`。

用户`D:/Chrome下载/Framework_latest.md`第七、二十二、二十三节明确采用绝对期限、有限候选释放规划及T_k=r_(k+1)-r_k。动态方案可以主动改变下一次释放时刻；它不要求永远周期20ms。但改变必须是明确规划结果，不能由执行资源空闲与否隐式发生。

## 已实现的修正

1. 新增已释放任务队列。到达计划时刻就登记job arrival，即使3个流水线执行名额已满；名额不足仅推迟dispatch。已释放任务的模型、释放计划、绝对期限冻结。
2. 静态分支以原日历r_0+nT生成下一次任务，不再等待上一任务完成后才生成，也不以晚执行时刻重新计时。晚到的时钟事件单独记录，不伪造准时到达。
3. 动态分支可以在命令确认后使用独立采样结果规划下一job，不强制等待该命令附带的完整反馈处理流程。预算仍要求命令之后的新状态，不为降低开销使用命令之前的状态冒充x_k。
4. 若取得新状态并完成动态决策时，期望的释放时刻已经过去，就不能倒填过去的承诺。保留target_release_s和planning_lateness_s，明确选择不早于当前决策时刻的合法release，并记录selected_release_interval_s。绝对deadline不随之延长；严格模式仍须准入，观察模式的失约不能算安全成功。
5. 端到端R从任务实际到达开始统计，包含队列等待、采样、输入处理、推理、命令传输。额外提供dispatch_wait_ms、release_to_sample_ms、release_jitter_ms、规划迟延、选定T及实际命令更新间隔。
6. 准入预测包含已释放但尚未执行的任务，并与当前每个回路串行I/O、同回路FIFO执行的约束一致。碰撞仅取消该回路的后续任务，其他回路继续。推理仍只在完整两层FC边界切换，不增加中段抢占。

当前异质路线入口`heterogeneous_motivation.py`明确启用`committed_calendar_v1`。其他历史入口保留旧协议并在报告中标记`completion_coupled_legacy`，不能把旧入口的结果当作本次修正后的结果。

主要代码：`framework_releases.py`、`probe_seven_continuous_timing.py`、`framework_runtime.py`；新增回归测试`tests/test_framework_releases.py`。

## 不要再次混淆的四个时间

- 计划释放r_plan：静态日历或动态规划选定的时刻。
- 实际到达r：运行时登记job arrival的时刻；r-r_plan是释放抖动。Windows不能保证该差始终为0。
- dispatch：任务获得流水线执行许可、开始输入服务的时刻；dispatch-r是排队的一部分。
- 命令确认a_hat：AirSim RPC返回，作为生效代理；没有硬件直接测量物理执行时刻。

以实际到达定义R=a_hat-r，则相邻命令更新间隔为：

```
a_hat_(k+1)-a_hat_k = (r_(k+1)-r_k) + R_(k+1)-R_k
```

因此周期20ms、稳定R=8ms时，命令仍每20ms更新，不是每28ms更新。严格安全检查优先使用Framework的绝对期限，不混用不同起点的T+R。

任务释放正确，也不等于采样和控制命令必然在同一时刻发生。当前输入采样仍是流水线服务，可能排队，已用真实时间戳单独记录。不能声称已实现与理想kT同步的零抖动采样，也不能把服务开始时间冒充任务释放。

## 验证结果

100项相关测试通过，覆盖名额占满时按期到达、晚dispatch不改变日历、释放后模式/期限冻结、同回路串行执行、命令前旧状态不可用于新控制预算、动态决策不倒填过去释放等情况。

实飞保存原十机种子4、速度5m/s、门宽1.7m、模型权重及静态配置；本阶段不调路线、不重新训练，也没有重新优化原20候选静态选择。

最终观察批次：`results/release_calendar_repair_20261028_v1/paired_seed4_final`。

| 碰撞前两秒均值 | 静态Drone1 | 静态Drone3 | 动态Drone1 | 动态Drone3 |
|---|---:|---:|---:|---:|
| 计划释放间隔 | 20.00ms | 30.00ms | 25.92ms | 26.11ms |
| 实际到达间隔 | 19.99ms | 30.00ms | 25.93ms | 26.10ms |
| 命令确认间隔 | 20.05ms | 29.98ms | 25.62ms | 25.90ms |

这里动态希望达到的快速目标仍是20ms，但部分决策/执行来不及，使合法选定的T约26ms。报告已把目标、选定T和到达抖动分开，不能把本次动态写成满足20ms。

全任务释放抖动：静态平均0.414ms、p99约1.665ms；动态平均0.267ms、p99约1.323ms。最大尾部仍存在，不形成硬实时证明。

本次静态3架接触门框，动态仍有Drone1/Drone3共2架接触；两组均未通过整体验收。修正时间语义没有自动解决控制器/实际更新范围、整条流水线等待或B的动力学含义。

严格准入批次：`results/release_calendar_repair_20261028_v1/enforce_seed4_final`。在原冻结经验时延包络下，两组均启动前拒绝，闭环请求与施加数均为0，明确记录为未完成/未通过，不能称为零碰撞成功。

开发中的`paired_seed4`保留了一次“拿不到命令后新状态”的错误报告；随后补了状态时刻检查。`paired_seed4_poststate`保留了尚未消除过去释放目标的中间版本。最终结论只使用上述final目录。实飞后仅补齐新到达触发准入重检的标志和状态年龄日志，未改变释放/执行选择；最终代码全部通过上述100项测试。

## 范围与后续

本次解决释放/排队/响应语义偏差，不宣称完整框架或硬实时安全已经完成。经验B、保守端到端包络、模式恢复资格仍有此前已知限制。

后续应在这套正确计时下：给c配套验证过的控制更新时间范围，减少真实dispatch/通信等待，并再检查共享任务可行性。此前静态选择是在旧执行机制下得到的，正式论文对比前必须用相同候选模型池重新做离线选择；不能把旧选择当成新执行机制下的总体最优。

复现观察（明确允许迟到继续飞，不是安全验收）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\FalconReproduction\airsim_control\controller_architecture_study\run_anticipatory_repair.ps1" -DeadlinePolicy observe
```

默认不加`-DeadlinePolicy observe`时启用严格准入，按当前经验模型仍可能拒绝启动。这是明确报告不可准入，不是再次把周期悄悄延长。

审计：`results/release_calendar_repair_20261028_v1/audit.py`、`verification.json`、`before/`、`final_source/`。模型与训练数据均未修改。
