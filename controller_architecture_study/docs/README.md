# 研究分支文档索引

2026-11-11 [Ubuntu Main18 重量化交接](UBUNTU_MAIN18_REQUANTIZATION_HANDOFF_20261111.md)。包含六档诊断、18 个模型重新校准、板端同样本复测和验收口径。

2026-11-10 [路线 B 组会讲稿与汇报图](ROUTE_B_GROUP_MEETING_20261110.md)。包括 A／B／C 启发式公式、动作对照、转弯相位、MPC 增益和板端验证进度。

2026-11-09：[关键任务的前端交付审计与六机对照](ROUTE_B_CRITICAL_ADMISSION_20261109.md)。区分板端排序与命令后的工作槽占用，附固定周期四臂运行入口。

2026-11-09：[IP37 命令空窗审计与第 4 门纠偏时机](ROUTE_B_IP37_GAP_FINDINGS_20261109.md)。实测日志阶段归属、历史状态离线反事实、早晚窗口 AirSim 确认入口。

2026-11-09：[固定周期与模型的局部命令延迟对照](ROUTE_B_LOCAL_DELAY_PROBE_20261109.md)。平缓和转弯各测三档延迟，使用相同模型、门宽和释放周期，真实 AirSim 待运行。

2026-11-09：[主池 18 模型等时延恢复筛选、图表及 A/B/C 公式](ROUTE_B_MAIN18_POOL_HEURISTIC_20261109.md)。包含用户完成的双模型 AirSim 对照、简化动力学全池筛选，以及四模型复核和板端只读回放命令。

2026-11-09 当前源码入口：[路线B代码与实飞阅读指南](ROUTE_B_CODE_GUIDE_20261109.md)。真实板端独立237D模型、Windows C++采样与Python调度的当前边界及 B50 证据见该指南和[问题日志](ROUTE_B_ISSUE_LOG_20261109.md)。下方带日期条目保留历史状态。

2026-11-09协作者任务：[237D gain7教师迁移与微调交接](HANDOFF_237D_GAIN7_FINETUNE_20261109.md)。独立模型和超网权重、旧Train/Validation和校验文件在`results/gain7_handoff_20261109_v1/`；旧标签不是gain7，需先按交接说明重标注并验证。

当前目标与下一步先读[研究目标与执行计划](../../docs/RESEARCH_GOALS_AND_PLAN.md)：保留旧教师DNN，补环境初筛和执行位置判定，再做时序/模型/启发式验证；不默认微调或全部搬到PS。本文后续记录按日期保留，不将旧“优先换教师”建议当作必经步骤。

最新已完成批次：[122次真实板端审计与研究方向复核](IP37_CAMPAIGN_FINDINGS_AND_RESEARCH_RESET_20261108.md)。程序结束不等于216次全做；4机仍无严格配对救回，7机输入拒绝加剧，10/12机无正式配对。原始数据不改，派生证据、测量口径问题、控制/时序分离与下一步收敛建议已记录。

用户最新缩减版：当前批量默认4/7/10/12机×1.9/2.4/3.0m，最多216次，纯飞行约94分钟，目标总耗时约两小时（初始化另算）；新结果目录包含`ip37_campaign_2h`。已通过离线计划入口，命令仍见下方批量说明。

2026-11-08新交付：[板端4～12机×1.7～3.0m完整对照与运行命令](IP37_FULL_CAMPAIGN_20261108.md)。旧模型11、有限静态贪心后冻结、三次动态配对、独立统一评价与断点续跑；74项离线检查通过，真实批量待用户执行。不把这一交付当作已经获得动态安全收益。

当前入口更新于2026-11-02。这里有大量带日期的历史文档；请先读稳定入口，再按任务选择历史资料，不按文件名大小或搜索到的第一份报告判断当前状态。

最新实时试飞见[2026-11-05原生IO分核与后台通知结果](RT_SPLIT_IO_RESULTS_20261105.md)。Agent完成16轮六机试飞，动态周期已实际安装，尾延迟与加频后的丢弃仍需改善。旧v2的零安装结论属于历史状态。

现行动态控制路线的源码、原始试飞及运行依赖见[路线A恢复记录](ROUTE_A_CHECKPOINT_20261106.md)，可在探索启发式或可解释框架期间作为恢复点。

路线B的[关键窗口需求修复：insight、公式、代码及结果](BOTTLENECK_INSIGHTS_AND_HEURISTIC_20261106.md)已形成独立参考实现。8轮历史飞行日志只读审计，另有PS/PL合成时序实验及整任务/分段能力对照；不是新试飞或FPGA实测。四回路结果有希望，十二回路决策开销及贪心漏解仍需解决。

后续的 [A/B/C 决策原型与配对回放](ROUTE_B_ABC_DECISION_20261106.md)分别处理独占路径过期、资源窗口超载和执行阻塞。板上已有 237D 模型时，可按 [Ubuntu 板端时序测量交接](UBUNTU_237D_TIMING_HANDOFF_20261106.md)收集真实成本。

首批板端数据已复算并用于 [串行 FIFO 校准和期限回放](ROUTE_B_BOARD_REPLAY_20261106.md)，含同步与错峰、FIFO/EDF/A/B/C 对照及剩余测量边界。

双模型成本已用于 [同步批次最少降档回放](ROUTE_B_PAIR_BOARD_20261106.md)，同文记录 Windows IP37 黄金探测及板端服务启动命令。

11月7日RC1已完成固定50ms单机实际控制，新的[路线B真实板端多机在线入口与命令](IP37_JOINT_ONLINE_RUNBOOK_20261107.md)也已取得[首次三机有效结果](IP37_JOINT_FIRST_RESULTS_20261107.md)：两组均15/15穿门，动态5组/7次周期改变，请求减35.45%、平均位置MSE增1.19%。下一步补固定80ms，修正成本分解和尾部释放覆盖。

用户随后固定80ms也完成15/15门，最新主线调整为[局部危险识别与安全救回实验](SAFETY_RECOVERY_EXPERIMENT_20261107.md)。已生成六机26秒自然转弯门场景并通过4项离线检查；正式危险对照尚未执行，先修成本模型和接入case/因果日志，不继续以宽门联通或省请求作为主要结论。

阶段A的[同一转弯路线80/30ms配对飞行](IP37_RECOVERY_PAIR_RUNBOOK_20261107.md)现已接入冻结case并交付命令；50项相关检查通过，真实配对待用户运行。此阶段先确认可恢复周期差异，尚未做六机动态救回实验。

上述配对随后已执行：两组均通过前3门后撞第4门左框，碰撞前周期时序通过。见[第4门碰撞诊断与20ms板端／教师对照命令](IP37_GATE4_DIAGNOSIS_20261107.md)。同输入重算不能替代闭环因果；新增诊断真实试飞待用户执行，暂不改门宽或重新量化。

20ms对照现已执行：教师302条命令均按时、仍同门碰撞，板端另有3次迟返和3次忙丢弃。最新诊断第5、6节给出602输入重放、软走廊与门框约束差异、姿态执行过程证据，以及净宽3.0/3.4/3.8m的有限探索命令。路线/速度不变，未改模型或标签；实际筛查待用户执行。

三档宽度随后已完成6次实飞，3.0m双碰撞、3.4/3.8m双通过。用户追问周期为何收益小，见[实际周期、策略语义与教师姿态执行的受控诊断](IP37_PERIOD_EFFECT_DIAGNOSIS_20261107.md)：实际30/80ms已核实，离线仅加入原姿态反馈即复现约1.1m共同偏差及厘米级周期差异。下一步先查教师与执行一致性，不把多机扩展当作解决该问题的替代。

现已交付[教师姿态增益／QP一致前馈四组对照](TEACHER_ATTITUDE_COMPARISON_20261107.md)。原教师文件不改，诊断独立成模块；同30ms同2.1m冻结case，先完成离线模型与46项检查，用户执行真实四组命令。新日志记录目标/实际姿态、角速度、加速度和前馈分量，尚未据离线改善重训或替换部署模型。

四组真实结果与后续[固定教师周期扫描／DNN迁移方案](FIXED_TEACHER_TIMING_AND_TRANSFER_20261107.md)已回填：三改进均13门，尾部漏周期仍在。现在固定gain7做30/50/80/120ms，先研究时间敏感性，不继续追着结果调控制器；脚本待用户运行。新标签微调与从零对照尚未启动。

## 新协作者先读

用户随后完成的[gain5四周期实飞](GAIN5_DIAGNOSTIC_20261107.md)均13门，仍无明显快周期优势；[预演原理与20组有界增益/T/R筛查](TEACHER_MODEL_SCREENING_AND_TIMING_20261107.md)已完成，44项检查通过。K10是后续短/长R实飞候选，原教师和模型保留，模型结果不替代真实飞行。

克隆Git仓库后若缺少 `results/` 下的237D模型和验证数据，按[237D产物交接](SHARE_237D_ARTIFACTS_20261105.md)接收原始文件包并检查。单模型包覆盖残差21×1024、验证数据和默认路线，其他实验所需产物按入口补齐。

1. [整体Framework](../../docs/FRAMEWORK.md)：研究问题、时间定义、模块边界与尚未验证的假设。
2. [协作进度/任务认领](../../docs/COLLABORATION_STATUS.md)：已完成、当前问题、未来任务和产物位置。
3. [易懂实现说明](FRAMEWORK_IMPLEMENTATION_EXPLAINED_20261101.md)：实际控制如何发布、采样、计算、选择和提交。
4. [最新阶段结果](EPOCH_BOUNDARY_RESULTS_20261102.md)：期限修复、六次十机对照与64次补采；新评分器收益尚未成立。[前一阶段](BOUNDARY_TAKEOVER_RESULTS_20261102.md)保留历史结果。

当前用户已运行[期限归属修复与64次边界补采](EPOCH_FIX_AND_BOUNDARY_RUNBOOK_20261102.md)，详见最新结果；该操作说明保留原设计与启动命令，不把运行目标当作已取得的效果。

针对“为何动态没有在危险前生效、开销多高、哪些场景可能胜出”，见[建议拒绝与动态优势诊断](DYNAMIC_ADVANTAGE_AND_ADMISSION_DIAGNOSIS_20261102.md)：只读分析和实验建议，未修改控制机制。

后续[CPU/XPU分工与建议服务修复](CPU_XPU_SERVICE_REPAIR_20261102.md)已完成代码和135项离线检查，真实试飞待用户执行；入口为`run_runtime_service_repair.ps1`，不重复补采。

该入口v1字典异常已修复并通过137项相关检查；用户已完整执行v2。当前最新实测见[CPU服务修复v2结果](RUNTIME_SERVICE_REPAIR_RESULTS_20261102.md)：事实刷新和服务等待改善，新联合建议仍0安装，不是已验证控制收益。

用户希望将在线候选评分改为直接状态反馈决策；交接见[直接资源分配研究与实施计划](STATE_FEEDBACK_ALLOCATION_PLAN_20261103.md)。这是2026-11-03的待实施计划，包含完整开销口径、过渡义务、PS/PL边界及经典方法对照，不代表当前代码已经替换。

原始用户framework逐字快照位于[reference](../../docs/reference/Framework_latest.original.md)，只作为设计来源；其后的讨论及当前实现差异在稳定入口说明。

## 按工作方向找资料

| 工作方向 | 入口 | 阅读重点 |
|---|---|---|
| 当前实时主线 | [Native runtime](NATIVE_RUNTIME_IMPLEMENTATION_20261029.md)、[最新阶段结果](BOUNDARY_TAKEOVER_RESULTS_20261102.md) | 独立发布、IO/前端、分段EDF、期限归属、历史超期 |
| 学习建议/选择 | [解释性联合决策](EXPLAINABLE_JOINT_DECISION_20261030.md)、[上下文评分](CONTEXTUAL_JOINT_DECISION_STAGE_20261031.md)、[修复与新状态复核](EFFECT_AND_FRESH_REVIEW_REPAIR_20261031.md) | 都是对应日期的探索，不是已证明最终算法；最新状态看协作进度 |
| 本轮恢复与接管试验 | [试验设计](BOUNDARY_AND_ACTUAL_RATE_VALIDATION_20261101.md)、[结果](BOUNDARY_TAKEOVER_RESULTS_20261102.md) | 区分计划、采集标签、真实采用及控制收益 |
| 安全时间预算 | [Yang方向研究](YANG_SAFE_PERIOD_RESEARCH_20261030.md)、[恢复预算提案](RECOVERABILITY_BUDGET_PROPOSAL_20261029.md) | 保持控制、假设、状态年龄与绝对期限；没有无条件安全证书 |
| 超网结构/训练 | [架构背景](INCREMENTAL_SUPERNET_ARCHITECTURE_20261012.md)、[实施计划](INCREMENTAL_SUPERNET_IMPLEMENTATION_PLAN_20261012.md)、[源码说明](../incremental_supernet/README.md) | 文档中“尚未训练/量化”可能是历史状态；最新产物看协作进度 |
| 自定义PL/Ubuntu交接 | [M3交接](UBUNTU_INCREMENTAL_SUPERNET_M3_HANDOFF_20261015.md) | 混合精度数值契约、冻结包、质量例外、golden与后续硬件交付 |
| 独立模型池/DPU对照 | [237D模型池交接](UBUNTU_237D_DPU_MODEL_POOL_EXPERIMENT_HANDOFF_20261016.md) | 27模型身份，不能和12D/25D或超网包混用 |
| 文献与研究定位 | [调度研究日志](SCHEDULING_RESEARCH_LOG_20261005.md)、[量化研究](M2_QUANTIZATION_DIAGNOSIS_RESEARCH_20261014.md) | 按记录的阅读范围使用，不把研究设想写成已证新颖性 |
| 早期结构扫描 | [分支README的M1历史部分](../README.md) | 原12D小模型/残差扫描的历史命令，不是当前实时主线 |

## 源码定位

| 代码 | 当前职责 |
|---|---|
| `probe_seven_continuous_timing.py` | 实际多机控制前端与实验执行循环 |
| `framework_native_io.py`、`native_runtime/native_io.cpp` | 原生时间日历、RPC、控制上下文与原子提交桥接 |
| `framework_releases.py`、`framework_runtime_worker.py` | 发布/待派发语义、共享CPU分段执行 |
| `held_control_safety.py` | 保持控制预算及批量分析 |
| `joint_contract_shadow.py` | 后台建议、小评分器候选、最新上下文复核入口；默认不真实提交 |
| `joint_contract_recovery.py`、`joint_commit_guard.py` | 义务更新、全组预测与提交前检查 |
| `controller_effect.py`、`repair_controller_effect.py` | 线性评分器、训练与适用范围 |
| `probe_recovery_boundary.py` | 有界配对恢复采集及无效尝试保留 |
| `run_reviewed_rate_trial.py`、`audit_boundary_takeover.py` | 真实周期建议对照及只读证据汇总 |

路径均相对`controller_architecture_study/`。Windows运行入口通常要在工程根目录以`python -m controller_architecture_study.<module>`调用；不要直接进入子目录后破坏包导入。

新增进展写入[共享看板](../../docs/COLLABORATION_STATUS.md)，新增实验保留带日期报告。不要把这个索引变成第二份进度看板。
