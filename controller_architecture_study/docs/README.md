# 研究分支文档索引

当前入口更新于2026-11-02。这里有大量带日期的历史文档；请先读稳定入口，再按任务选择历史资料，不按文件名大小或搜索到的第一份报告判断当前状态。

## 新协作者先读

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
