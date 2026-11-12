# 协作进度与任务看板

2026-11-11 完成 Ubuntu Main18 重量化交接。沿用 Vitis AI 1.2.1，先在六个代表档位逐级定位，再做全池同权重 PTQ 和固定 480 条样本的真板配对复测。候选量化、编译与上板由 Ubuntu 协作者执行，待回收误差、排序、覆盖率和时延结果。[量化交接与验收](../controller_architecture_study/docs/UBUNTU_MAIN18_REQUANTIZATION_HANDOFF_20261111.md)

2026-11-11 完成 Main18 板端误差与预算选模。18 模型各 960 输入，17,280 次有效回复。建表前沿为 2×128 与 3×256，核对组后者标准化 MSE 降约 28%。相同样本上的模型 MAC 与误差秩相关从 FP32 的 −0.746 反转为 INT8 的 +0.715。板端参与的 AirSim 预算选模四次均通过两门，176 条服务全部在给定预算内。生产多机调度器的剩余预算接入和联合期限验证待完成。[报告与复跑](../controller_architecture_study/docs/ROUTE_B_MAIN18_BUDGET_SELECTOR_20261111.md)

2026-11-10 Windows Agent 完成小模型适用边界复核。五个初态、五个控制器、各三次，共 75 次 AirSim，60 个非参考配对全部匹配，最大采样或施加时刻差约 3 ms。历史早弯两组中 2×128 在第 4 门余量较大，4×1024 与 33×1024 在第 5 门多约 19～21 cm，但小模型两门均通过。独立反向弯和受控偏差 1.20 m/s 全部通过，1.35 m/s 全部未满足第 3 门净空，均无碰撞。尚无小模型失败而其他模型合格的复现状态，未使用板端 INT8 闭环。参考几何和 18 项回归检查通过。[逐次汇总](../controller_architecture_study/results/main18_model_boundary_20261110_212859/summary.json)

2026-11-10 Windows Agent 完成历史弯中状态的等时延模型资格复核。四个历史初态、三个冻结 FP32 模型和原 MPC 教师、各三次，共 48 次 AirSim 步进，统一 T50／R20 ms；36 个非参考配对均匹配，最大时刻差 3.01 ms 以下。两个早弯及一个中段状态，2×128 较 4×1024 多约 11～13 cm 第 4 门净空，晚段差不足 4 mm。三个 DNN 均 12/12 满足目标门判据，教师 6/12，全部零碰撞，1.2 s 窗内均未达到末段恢复阈值。板端原 Main18 5237 的 ID 5、11、18 同输入共 36/36 回包有效，板端未参与闭环。[AirSim 汇总](../controller_architecture_study/results/main18_historical_models_20261110_203410/summary.json) · [板端只读数据](../controller_architecture_study/results/main18_historical_models_20261110_203410/board_readonly.json)

Windows Agent 已澄清路线 B 组会讲稿图 1 的时序统计口径。补充全组按期与工作槽判据、动作列含义和有限枚举无解实例。原数据与图保持原样。[组会讲稿](../controller_architecture_study/docs/ROUTE_B_GROUP_MEETING_20261110.md)

2026-11-10 Windows Agent 完成路线 B 组会讲稿的中文改写。正文先讲完整执行过程和 A／B／C 三个实际回放例，再逐项说明实验设置、读图与结果。公式集中到问答附录，保留四张原汇报图。[组会讲稿](../controller_architecture_study/docs/ROUTE_B_GROUP_MEETING_20261110.md)

2026-11-10 Windows Agent 完成路线 B 组会讲稿、四张 PNG／SVG 图与复现脚本。涵盖 A／B／C 公式与反事实、模型成本、关键更新相位、33 组 MPC AirSim 实测和真板阶段证据。未改运行代码、模型或原始报告。[组会讲稿](../controller_architecture_study/docs/ROUTE_B_GROUP_MEETING_20261110.md)

2026-11-10 Windows Agent 完成教师增益与时序敏感度入口、33 组离线对照及 18 项检查，用户随后完成 33 组 AirSim。增益 3.5 在第 4 门的余量随周期和延迟增大而缩小，增益 10 在 99 ms 压力条件下撞第 3 门下框。未新增 DNN 权重或测试板端。[离线与复跑](../controller_architecture_study/docs/TEACHER_GAIN_TIMING_GRID_20261110.md) · [AirSim 实测](../controller_architecture_study/results/teacher_gain_timing_airsim_20261110_131546/summary.csv)

2026-11-09：ROUTE-B-59 两批真实板端对照完成。接通网线恢复 5238；原批八次 Drone5/6 量程退出，保留为诊断。仅恢复两机参考速度后八次均维持六机进入关键窗口，16/16 关键指令按期且零量化拒绝。完全延后回读前 10 s 两次均零时序违规，整段仍有违规；无目标任务救回证据。FIFO 反序 Drone1 第 4 门身体余量 −4.06 cm，目标两条按时但 2.88/2.96 s 两条连续未交付，值得下一次因果回放。[真实板端结果](../controller_architecture_study/docs/ROUTE_B_CRITICAL_ADMISSION_20261109.md#修正版六机对照结果)。

2026-11-09：Windows Agent 完成 ROUTE-B-59 的历史审计和六机试验准备。18 次同路线双机的 36 条关键指令全按期，板端排队最大约 0.21 ms。122 次批量飞行 12,689 条忙丢弃中，5,561 条记录上一命令已确认但槽仍被配对回读占用。历史最大耗时规则下 5,265 条存在直接腾槽机会，尚无成功交付反事实。六机固定周期 FIFO／EDF／有余量回读／延期回读的八次对照已完成计划、40 项测试和模拟汇总，真实对照等待 5238 和 Blocks。[审计与运行方案](../controller_architecture_study/docs/ROUTE_B_CRITICAL_ADMISSION_20261109.md)。

2026-11-09：Windows Agent 完成 ROUTE-B-58。32 次 AirSim、16/16 有效配对，含历史状态重建与从路线起点连续飞行两批。80 ms 早期连漏两次共 4/4 撞第 4 门左框，正常 0/4，晚期均无碰撞。50 ms 连续飞行早期余量损失 6.3–7.0 cm，晚期只变化毫米级。16 条同路线未参与选择的历史日志也复现了离线相位差。下一步验证关键更新的端到端交付。[实飞确认与边界](../controller_architecture_study/docs/ROUTE_B_IP37_GAP_FINDINGS_20261109.md#airsim-确认结果)。

2026-11-09：Windows Agent 完成 ROUTE-B-57 日志审计与离线回放。批量飞行中长空窗所含漏更新的 92.5% 未进入板端队列。两份历史第 4 门初态的相位扫描显示，约提前 0.94 s 连漏两次会使预测过门余量转负，约提前 0.15 s 则仅毫米级。冻结了 16 次 AirSim 配对确认并完成模拟验证，实际 Blocks 未运行。[结果、限制及试飞命令](../controller_architecture_study/docs/ROUTE_B_IP37_GAP_FINDINGS_20261109.md)。

2026-11-09：用户完成 ROUTE-B-56 的 18 次 AirSim 局部延迟实飞。12/12 配对有效，全部有效过目标门，零碰撞、零丢弃。最长档实测多等约 18–21 ms，转弯过门余量变化为 +1.06～+3.93 mm，峰值横向误差变化正负皆有。该窗口缺少晚到导致安全余量下降的证据。[结果与边界](../controller_architecture_study/docs/ROUTE_B_LOCAL_DELAY_PROBE_20261109.md#实际-airsim-结果)。

2026-11-09：Windows Agent 完成 ROUTE-B-56 的首阶段命令延迟对照入口。固定 4×1024、T50 ms，在平缓／自然转弯第 4 门前仅调整两条命令的施加时间，目标 20／30／40 ms，每组重复三次。9 项针对性检查和 18 次模拟批量输出通过，实际 AirSim 等待用户执行。后续共享 DPU 调度对照以本轮实飞证据为依据。[协议与命令](../controller_architecture_study/docs/ROUTE_B_LOCAL_DELAY_PROBE_20261109.md)。

2026-11-09：用户完成 ROUTE-B-55 完整路线 12 次 AirSim 对照。12/12 全部有效过门、零碰撞、完整时窗且命令数达到计划。恢复标志全为假，源于终点制动后位置与速度未在规定窗口回到误差范围。反向弯及加密组最小过门余量约 29 cm，型号间约差 5 mm。[结果与判据](../controller_architecture_study/docs/ROUTE_B_MAIN18_POOL_HEURISTIC_20261109.md#12-次-airsim-试飞结果)。

2026-11-09：Windows Agent 完成 ROUTE-B-55 的试飞入口与离线检查。冻结平缓 5 门、反向弯 7 门、同参考加密 9 门，四模型共 12 次，统一 50/20 ms 和 2.4 m 净门宽。补齐正常飞行的末端观测并区分完整／公共时窗。参考几何、11 项检查和 12 次模拟批量输出通过，实际 AirSim 由用户执行。[完整路线与运行命令](../controller_architecture_study/docs/ROUTE_B_MAIN18_POOL_HEURISTIC_20261109.md#完整路线批量对照)。

2026-11-09：用户完成四模型 AirSim 等时延复核。1.45 m/s 初始侧向误差下，三个模型在第 3 门碰撞，9×512 无碰撞且恢复跟踪，但第 3 门未达到有效过门余量。另有一条 9×512 低误差试验因末端观测少约 2.94 ms 未获正式恢复标志。当前证据仅涵盖两门局部窗。下一轮建议保持四模型，增加完整自然路线并分开记录覆盖、碰撞与有效过门。[核对记录](../controller_architecture_study/docs/ROUTE_B_MAIN18_POOL_HEURISTIC_20261109.md#用户完成的四模型-airsim-复核)。

2026-11-09：Windows Agent完成 ROUTE-B-54 的全池筛选、用户 AirSim 结果分析和启发式公式交付。18 个模型在前三个固定 T50/R20 ms 初态均恢复，最难一档的 18 个采样几何余量均负。用户双模型 AirSim 在该档均碰同一侧门框，共同时窗 RMSE 差约 1 mm。重建 51 个板端回放输入，5237 原包黄金请求也超时，未取得当前全池服务成本。四模型 AirSim 复核和板端只读回放命令见[主池筛选与启发式规则](../controller_architecture_study/docs/ROUTE_B_MAIN18_POOL_HEURISTIC_20261109.md)。

2026-11-09：Windows Agent完成ROUTE-B-53理想动力学等时延预筛，核对原18模型服务命令。四组共用T50/R20 ms、每侧60次施加且逐条时刻相同，模型33层较4层的RMSE改善约1–3 mm，最重组两侧最低采样几何余量均负。Windows AirSim未运行，已交付步进复跑入口并用模拟RPC核对时序。见[预筛结果与复跑命令](../controller_architecture_study/docs/ROUTE_B_MAIN18_MATCHED_RECOVERY_20261109.md)。

2026-11-09：Windows Agent完成ROUTE-B-52双模型离线反事实回放，65项检查通过。160个受控场景中147个初始失败、131个有限Oracle可修复，37个需T+c。五次检查内原ABC修复130个，带派发保护修复131个，C层修复数相同但模型改动34→11。按实飞3ms提交门槛过滤后保护版为130个。第4门59条历史快照在原记录成本下Oracle为0。见[实验与复现](../controller_architecture_study/docs/ROUTE_B_ACTION_REPLAY_20261109.md)。待测大模型闭环价值和较早危险窗口的动作收益。

2026-11-09：Windows Agent完成DOC-B51，新增[路线B代码与实飞阅读指南](../controller_architecture_study/docs/ROUTE_B_CODE_GUIDE_20261109.md)，梳理当前执行链、决策与提交、指标、实验和依赖，并从入口标明历史状态。原始实验及已有协作者修改保留。结果与源码链接检查已完成。

最新目标计划（2026-11-08）：[RESEARCH_GOALS_AND_PLAN](RESEARCH_GOALS_AND_PLAN.md)。以用户最新决定为准，保留增益3.5板端DNN；按“评价口径→低负载环境初筛→执行位置收益判定→冻结环境→时序差异→模型资格→启发式→小规模扩展”推进。先测Windows原生热路径，PS全量迁移不作前置条件；微调仅在现有控制器确实不满足必要任务时启动。此次只写文档，未改运行代码或开展实验。

最新实测审计（2026-11-08）：`ip37_campaign_2h_20261108_v2`已遍历结束，实际122次（86搜索＋36正式），不是216次全部完成。4机9对可比较J，完整通过回路数未提升；7机动态范围拒绝增多；10/12机没有进入配对。152次输入拒绝、60次飞行保护退出。正式动态28组安装、约96%决策无可行修复，模型历史最大成本过保守；12机主要表现为主机采样/输入路径延迟，而板端内部服务约1.32ms。发现`calendar_coverage_passed`将运行错误与覆盖混用，资格口径需修正。[详细结果与研究取舍](../controller_architecture_study/docs/IP37_CAMPAIGN_FINDINGS_AND_RESEARCH_RESET_20261108.md)，只读派生证据在`results/ip37_campaign_analysis_20261108_v1/`。本轮未修运行代码或训练/实飞；建议先可靠闭环基线、真实执行链及小规模T/R边界，再扩大实验。

用户最新修订：默认4/7/10/12机×1.9/2.4/3.0m，12格、最多216飞行，纯飞行93.6分钟，初始化另算。保留每格12次静态搜索和3组配对，新结果目录`results/ip37_campaign_2h_20261108_v2/`。v2离线计划及三档参考几何审计通过，证据`results/ip37_campaign_2h_offline_plan_20261108_v2/plan.json`。此前1.7/1.9/2.1m批次保留，不混用结果；下文972次为更早的大矩阵设计。

最后更新：2026-11-08（Asia/Shanghai）。这是所有协作者共同维护的状态入口；整体设计见[FRAMEWORK](FRAMEWORK.md)，详细资料见[研究分支索引](../controller_architecture_study/docs/README.md)。

11月8日最新交付：[真实板端完整矩阵入口](../controller_architecture_study/docs/IP37_FULL_CAMPAIGN_20261108.md)。保留旧模型11，4～12机×6档净门宽；每格最多12次静态角色坐标贪心，冻结后3次静态/动态配对，最多972飞行。新增逐机初始周期、固定50ms独立评价、完整失败后真实状态J、上下文绑定续跑和CSV汇总；静态EDF不再暗用B。74项相关检查及6档×12机参考几何审计通过，离线计划保存在`results/ip37_campaign_offline_plan_20261108_v3/`。**真实批量尚未运行**，无新碰撞改善结论；当前只有T/整任务排队调整，无模型切换或DPU抢占，保守历史最大成本估计仍保留并单列拒绝。

**当前结论：Agent完成原生IO分核、后台信号量通知和可恢复的前端线程优先级调整，实飞16轮六机、零碰撞。1.75m场景两轮动态安装5组、平均J提高约0.98%，但丢任务多于静态；1.60m场景新配置安装0组。尚无静态碰撞、动态救回的证据，仍存在尾延迟。详见[11月5日结果](../controller_architecture_study/docs/RT_SPLIT_IO_RESULTS_20261105.md)。**

路线B最新：11月7日真实板端三机固定/启发式均15/15穿门，动态安装5组/7次周期改变，请求1199→774、平均位置MSE约增1.19%。后期1476次拒绝与阶段历史最大值叠加至83.96ms有关；固定组尾部少记录1条计划释放。先补固定80ms对照，再修成本分解和尾部覆盖，见[本轮详细分析](../controller_architecture_study/docs/IP37_JOINT_FIRST_RESULTS_20261107.md)。尚无本路线静态撞门、动态救回证据。

随后固定80ms宽门也完成15/15。自然急弯净宽2.1m的单机80/30ms配对已运行，两组通过前3门后均撞第4门左框，碰撞前没有周期超期；30ms有所改善但未救回。[诊断与下一步命令](../controller_architecture_study/docs/IP37_GATE4_DIAGNOSIS_20261107.md)已交付：同case板端20ms／CPU教师20ms，真实试飞待用户执行。首次旧门框阻断已由实际重跑排除。成本模型的全历史最大值叠加仍未修复，不把本轮固定周期诊断当作动态实验。

20ms能力对照已由用户运行 `ip37_capability_20261107_135028`：两组仍撞第4门；教师302条命令均按时、无周期丢弃，板端3迟返＋3忙丢弃。离线重放602个输入支持教师/姿态执行链在该急弯下存在能力限制，不是仅量化或排队问题。已离线准备净宽3.0/3.4/3.8m、完全相同路线的有限筛查，命令和证据见上述诊断第5、6节；6次实飞待用户运行。未改教师、DNN或量化。

最新：用户已运行三档门宽共6次。3.0m两组撞第4门，3.4/3.8m两组均13门通过，快周期仅增加厘米级余量。进一步[周期作用诊断](../controller_architecture_study/docs/IP37_PERIOD_EFFECT_DIAGNOSIS_20261107.md)确认3.8m实际采样/命令间隔确为30/80ms；离线加入原教师姿态反馈即可复现约1.1m共同偏差及约4cm周期差异，直接执行QP加速度仅有厘米偏差。下一步转为教师/姿态执行一致性诊断，暂缓直接扩大多机；原控制器、权重与量化未改。

已交付[教师姿态响应四组独立对照与命令](../controller_architecture_study/docs/TEACHER_ATTITUDE_COMPARISON_20261107.md)。完整26秒离线模型中，第4门偏移：原版−1.068m、仅增益7为−0.212m、仅QP前馈+0.040m、组合+0.056m。46项相关检查和PS语法通过；实际四组试飞待用户运行。新增诊断模块和显式选项，原教师源码哈希不变；未替换旧DNN或启动真实飞行。

用户已完成 `teacher_attitude_flight_20261107_181345`，[实飞回填](../controller_architecture_study/docs/TEACHER_ATTITUDE_COMPARISON_20261107.md#实飞回填2026-11-07-1813批次)：原版第4门碰撞，gain7／QP前馈／组合均13门、零碰撞，最小过门余量0.331／0.393／0.400m。三组命令均在30ms内返回，但末段新状态安全分析耗时30～63ms，导致每组1～2次漏周期，严格时序仍未通过；未换标签或部署模型。

用户确认回到固定教师下的时间/计算取舍研究。已交付[四周期入口与DNN迁移方案](../controller_architecture_study/docs/FIXED_TEACHER_TIMING_AND_TRANSFER_20261107.md)：先固定gain7，单机30/50/80/120ms，测实际R且保留漏周期，真实试飞由用户执行。暂不训练；未来旧权重只作初始化，换新标签、重置优化器/训练进度并用新验证目标，先代表性架构与从零训练对照。

用户已完成 `teacher_period_sweep_20261107_190128`：四周期均13门，位置RMSE约0.12087～0.12301m，无明显快周期优势；30ms有8条非正常交付状态，其余全交付。已交付自定义周期参数和[六个中间点补测命令](../controller_architecture_study/docs/FIXED_TEACHER_TIMING_AND_TRANSFER_20261107.md#首轮周期实测与细粒度补点)，形成每10ms曲线。补测待用户运行；不预设趋势，若仍平台转独立R实验，暂不训练。

六点补测 `teacher_period_sweep_20261107_191116` 已全部13门且时序通过；合计10个T点仍无明显快周期优势。随后只做[固定T的独立R离线扫描](../controller_architecture_study/docs/TIMING_SENSITIVITY_LIMITS_20261107.md)：增益7、T120固定，R7～100ms使模型RMSE约增加79%，但样本几何余量仍正。不能声称找到安全失效分离；当时建议停止继续细分T、调增益或训练，转向任务变化时间尺度和参考预览作用。

用户随后明确要求测试增益5，194816批次已完成四组均13门零碰撞、最小余量约2.7～3.0cm，仍无明显快周期优势，T30有1迟返和3忙丢弃。后续[预演原理与20组有界增益/T/R筛查](../controller_architecture_study/docs/TEACHER_MODEL_SCREENING_AND_TIMING_20261107.md)已完成，44项检查通过，原冻结教师不变。模型中增益10短R精度较好、长R性能下降；增益14在长R下频繁限幅。未找到单靠增益产生稳定快T优势的候选。下一步宜为固定T的独立R实飞诊断，未启动真实新飞行/训练或改部署模型。

此前11月2日v1在第三组因字典重复字段TypeError中止，原始失败记录保留；错误已修复，137项相关检查通过，v2未再出现异常。v2期限绑定审计96→41、CPU/XPU六输入误差0。新增模块初始backend111.39%、frontend98.16%是经验模型预测，均超过95%；不应视作实测利用率，也不通过放宽预算制造PASS。

本文更新的是进度和任务，不修改原始试验报告。远端Ubuntu/服务器没有回传证据的内容一律写“待回填”，不能根据Windows已有部署包推定已经上板。

## 1. 工程地图与不同轨道

| 位置 | 用途 | 交接时注意 |
|---|---|---|
| 根目录`run_*`、`mpc_controller.py`、`dnn_controller.py` | 历史MPC/DNN/AirSim基础链路 | 不代表最新研究原型的全部实现 |
| `experiments/paper_track_p_12d_5ms_v1/` | 历史12D数据与六模型训练 | 实际周期/速度以配置为准，不从`5ms`目录名推断 |
| `controller_architecture_study/` | 当前主要研究源码：237D、模型池、超网、动态控制与诊断 | 接手实时框架从这里读 |
| `controller_architecture_study/native_runtime/` | C++释放日历和AirSim RPC DLL | 不是FPGA内核，也不是整个框架都已C++化 |
| `controller_architecture_study/results/` | 每次训练/试验/审计产物 | 目录多，先按本文定位，不全量扫描或覆盖 |
| `controller_architecture_study/docs/` | 带日期的设计、实验、文献与交接 | 当前状态以本文为准，历史结论保留原日期 |
| `ubuntu_transfer/` | 冻结部署包和对应manifest | 包存在不等于Ubuntu已完成量化/上板 |
| `colleague_transfer/` | 给协作者的历史压缩包、UE工程及交接 | 注意包的版本，不用旧快照覆盖当前源码 |
| `tests/` | 对应模块的回归检查 | 选择相关检查；真实飞行/板端收益另需证据 |

### 输入和模型不能混用

| 轨道 | 角色与契约 | 当前用途 |
|---|---|---|
| 旧25D | 早期工程化控制输入，包含额外路径信息 | 旧FP32/DPU/UDP闭环经验 |
| 历史12D | 门位置误差、姿态误差、当前速度、当前加速度 | FALCON相关六模型复现轨道 |
| 当前237D独立模型 | 显式状态及未来参考，契约`decision_convex_local_relative_pv_237d_v1` | 当前CPU实时原型、独立模型池基线 |
| 当前237D超网 | 同一237D任务上的因果深宽共享结构 | 已完成FP32训练与整数功能候选，尚未接入当前实时链路 |

所有轨道输出均涉及`[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]`，但仍需各自配套的归一化、坐标约定和输出转换。相同输出维数不意味着模型/协议可直接互换。

237D编码从300D完整参考特征投影得到，不能当作“12D补零”。准确布局看[编码器](../controller_architecture_study/convex_features.py)及[基础特征构造](../controller_architecture_study/decision_dataset.py)，不要在另一端重新猜输入次序。

## 2. 已经完成的工作

| 事项 | 已达到的状态 | 证据/位置 |
|---|---|---|
| 历史FPGA链路 | 用户历史回报旧模型DPU/UDP与Windows/AirSim链路打通 | [旧集成状态](../WINDOWS_FPGA_INTEGRATION_STATUS.md)；不外推到新237D/超网 |
| 237D教师、数据与独立网络族 | 恢复补采样数据与18个主池plain模型、残差及长训对照已形成 | `results/decision_convex_recovery_v1/`、`results/decision_convex_recovery_scenic_falcon18_v1/` |
| 最大残差模型长训 | 33×1024残差追加5000轮完成；作为超网构建的教师来源 | [超网背景](../controller_architecture_study/docs/INCREMENTAL_SUPERNET_ARCHITECTURE_20261012.md)，其中旧进度段是历史记录 |
| FP32二维共享超网 | 17深度×8宽度=136配置，v4追加5000轮完成，导出PASS，最大配置目标达成 | `results/incremental_supernet_237d_v4_grid5000/export/export_report.json` |
| 混合精度整数候选 | W8/head16/A16 LSQ dyadic；整数直接/增量逐位一致，最大/网格质量目标通过，严格切换目标未过 | `results/incremental_supernet_237d_m2_lsq_v3/export/export_report.json`，状态仍为`NUMERICS_PASS_QUALITY_NOT_MET` |
| 冻结超网部署包 | 可交接给自定义PL功能开发，质量例外已记录 | `ubuntu_transfer/falcon_incremental_supernet_m3_v1_20261015/`；`freeze_manifest.json`中FPGA/飞行振荡改善均未验证 |
| 独立237D模型池交接 | 27个FP32独立模型已打包；其中主池18、残差600轮4、长训残差1、早期恢复4 | [237D DPU交接](../controller_architecture_study/docs/UBUNTU_237D_DPU_MODEL_POOL_EXPERIMENT_HANDOFF_20261016.md) |
| 独立任务释放与IO | C++绝对时间日历、每机RPC及Python C ABI接入；与前端反馈处理解耦 | [C++实现说明](../controller_architecture_study/docs/NATIVE_RUNTIME_IMPLEMENTATION_20261029.md) |
| 共享分段执行 | CPU执行器在最多两层FC的段边界按EDF重排 | [执行器源码](../controller_architecture_study/framework_runtime_worker.py) |
| 规则式动态控制 | 有异质多机、周期调整、全组准入、前端保护、单机碰撞退出与日志 | [易懂实现说明](../controller_architecture_study/docs/FRAMEWORK_IMPLEMENTATION_EXPLAINED_20261101.md) |
| 小评分器与提交接口 | 486参数线性动作评分、有限显式选择、最新状态复核、原生原子提交入口 | [后台建议](../controller_architecture_study/joint_contract_shadow.py)、[提交检查](../controller_architecture_study/joint_commit_guard.py) |
| 最新阶段验证 | 120次有效恢复采集＋12次自然十机对照；相关74项检查通过 | [2026-11-02阶段报告](../controller_architecture_study/docs/BOUNDARY_TAKEOVER_RESULTS_20261102.md) |
| 本次协作文档 | 工程入口、框架、看板、历史原文及索引已建立 | 本文与[整体框架](FRAMEWORK.md) |

以上的“已完成”只针对对应交付。超网训练完成不等于超网闭环完成；整数一致不等于全部质量目标通过；零碰撞不等于硬期限全部满足。

## 3. 最新结果：所有协作者必须知道

最新派生汇总为 `controller_architecture_study/results/rt_splitio_stage_20261105/evidence.json`，保留16轮实际试飞及旧共核对照。1.90m首轮分核使两轮静态平均响应降至5.76/6.05ms、丢任务4/3，随后动态实际安装9组。后续修复了后台Event通知的唤醒确认等待，快照长尾由约60ms降至后续各轮0.85–3.48ms。1.75m探索有5组安装和小幅J改善，同时动态丢弃74条、静态0条；1.60m两轮安装0组。详情和全部失败/未运行组见[结果](../controller_architecture_study/docs/RT_SPLIT_IO_RESULTS_20261105.md)。

此前根目录为 `controller_architecture_study/results/runtime_service_repair_20261102_v2/`。用户六次执行完成，详见[历史结果与耗时拆分](../controller_architecture_study/docs/RUNTIME_SERVICE_REPAIR_RESULTS_20261102.md)。当时两组新增模块首次碰撞前共132条建议全部拒绝，管理服务最大间隔约90ms，新规划状态年龄均值72/75ms；保留为历史诊断。

### 3.0 此前期限修复/补采阶段（保留历史）

当前结果根目录为`controller_architecture_study/results/epoch_boundary_20261102_v1/`，详见[期限修复与补采结果](../controller_architecture_study/docs/EPOCH_BOUNDARY_RESULTS_20261102.md)。用户完成了268快照回放、六次自然十机试飞与64次有效补采：

- 重复快照中过期的未发布期限记录412→332，减少80条；是局部改善，不是消除超期或减少碰撞。
- 静态组碰撞2/2，旧动态1/3，新增模块3/1；各累计4架次，六次完整J均缺失。
- 新模块安装1组、7条标记命令完成；周期/释放改变0条，期限改变2条，且安装发生在该轮全部碰撞之后。
- 16个补采状态有3个周期资格差异，全部T80合格而T20不合格；均没有实际碰撞，涉及非常接近0的保守几何余量。模型资格差异仍为0。
- 没有重训或自动启用新模型。过去的动态零碰撞没有在本轮复现，不能把过往样例当作稳定收益证明。

### 3.1 前一阶段历史结果（保留，不作为最新结论）

结果根目录：`controller_architecture_study/results/boundary_takeover_20261101_v1/`。

- 120次有效采集覆盖30个配对状态，另外保留1次最小化窗口造成的基础设施超时。30个状态中，三种DNN及20/80ms比较尚未出现恢复资格差异；不能宣称已得到最低成本合格模型映射。
- 新线性评分器评估40个候选平均约0.072ms、P99约0.182ms。整个后台建议链路平均约6.4–6.7ms，不能只报评分器时间代表总开销。
- 第一轮新增方案安装1组、15条标记控制真正完成，但发生碰撞；之后限定为各机原DNN候选，并将建议处理放在已就绪控制工作之后。
- 修正后静态组两轮碰撞数1/0，旧动态0/0，新增模块组0/0。但修正后新建议实际安装0组，不能把这些零碰撞归功于学习选择器。
- 修正后仍有大量条件预算越期，尚未证明硬实时或严格物理安全。静态配置是历史离线选定项，不是本轮重新求得的全局最优FALCON配置。
- 268次后台决策中137次被未完成超期任务拦截；这137个快照都存在某个任务截止时间不晚于它自己的计划发布时间。部分旧保持控制早已结束，后续计划却仍携带旧期限，这是下一步的关键问题。

源控制DNN与归一化哈希已核对未变。8个网络是当前实时池，27个是导出的独立模型全集，18个是主训练族，136个是超网配置数：这些数字不是同一个集合。

### 最新证据（不要只读旧stage_summary）

| 路径（相对结果根目录） | 含义 |
|---|---|
| `combined_evidence.json` | 两轮对照、源控制器核验与时间归属诊断的合并入口 |
| `boundary/report.json` | 120有效采样与无效尝试、配对差异计数 |
| `ranker/report.json` | 小评分器训练、路线分折与评估 |
| `application/summary.json` | 修正前六次对照，曾实际安装1组 |
| `application_rates_only/summary.json` | 修正后六次对照，实际安装0组 |
| `stage_summary_application_rates_only.json` | 最新修正轮次审计汇总 |
| `application_rates_only/obligation_epoch_audit.json` | 旧控制义务/当前期限的归属诊断 |

`stage_summary.json`默认记录早期轮次，不能只因名字更短就当作最新结果。74项检查是上一轮相应代码的回归结果，不替代后续修改后的验证。

## 4. 当前正在推进的主线

2026-11-06记录两条研究路线。路线A是现行237D动态周期分配与全组复核，源码、依赖和证据已列入[恢复记录](../controller_architecture_study/docs/ROUTE_A_CHECKPOINT_20261106.md)。另一条路线探索启发式或可解释的框架。下方的周期分配和拒绝诊断是路线A的恢复起点。

路线B本轮完成[关键窗口需求修复](../controller_architecture_study/docs/BOTTLENECK_INSIGHTS_AND_HEURISTIC_20261106.md)：审计8轮已有日志，并实现独立PS/PL模型、窗口贡献规则、实际阻塞链诊断及最多5次全组检查。最终v5共480个合成参数案例，四回路两种能力对照均找到74/80可行方案，与该批离散oracle一致；不是飞行安全证明。十二回路加入决策耗时后可选分段分支仅9/80通过，低于只调模型15/80，不能宣称普遍优越。下一步需原生实现与板端成本测量，真实模型资格和半完成任务映射也未补齐。路线A执行代码未动。

目前处于周期分配效果与实时性权衡的核对阶段。分核和通知修复后，周期调整已经实际生效，但加频也增加了任务丢弃。应优先核对需求函数是否把正常的当前任务倒计时误当成未来持续加频需求，再拆分全组提交拒绝原因。

任务截止时间仍须对应正在保持的控制。已结束的旧义务、真实历史超期和当前有效期限继续分别记录。不能通过删除失败或统一放宽期限来提高建议采用率。

恢复边界数据也尚不足以支撑可靠的自动选型。下一轮应有限细化已有临界区域并另留新路线确认，而不是再训练数千轮期待模型自行解决标签缺乏差异的问题。

用户在本次路线B任务中授权Agent写代码并自行运行命令。已运行日志审计和离散时序实验，没有新AirSim试飞或FPGA测量。其他机器是否正在训练、Ubuntu是否已上板，需相应协作者在下面回填。

## 5. 任务认领与后续顺序

2026-11-09：Windows Agent完成ROUTE-B-50前端成本拆分对照，82项检查通过。发现Windows线程CPU计时出现0/15.625 ms量化后，改用控制线程接手至输入就绪的实测经过时间。最终双机配对均24/24门、零碰撞，合并臂1002/1004交付、0修改；拆分臂646/646交付、10修改，4次加频都早于2.11 s，第4门窗口零安装。拆分臂在3.51 s出现48.38 ms前端等待，仍抬高后续估计。第4门同窗口50条无在途快照有28条义务已到期；离线降低成本只能部分恢复可行性。下一步分开查交接长尾与危险前触发时间。见[问题日志B50-01至03](../controller_architecture_study/docs/ROUTE_B_ISSUE_LOG_20261109.md)及`results/ip37_native_prep_worker_wall_20261109_120420/`。

2026-11-09：Windows Agent完成ROUTE-B-49原生日历整组提交，81项检查通过。真实板端两机从同一50/50 ms起步，C++固定、C++动态、Python动态均24/24过门且全部任务按期交付。C++动态两次50→80 ms实装，首个新周期匹配、上一任务期限未改，未观察到危险阶段加频。动态C++仍有1866次预测无可行修复，其中1865次在3 ms内结束。离线重算表明快照对前端成本估计高度敏感；先分解Python交接与计算、保留单机复用保护，再复测危险窗口。见[问题日志B49-01至03](../controller_architecture_study/docs/ROUTE_B_ISSUE_LOG_20261109.md)及`results/ip37_native_adaptive_20261109_115352/`。

2026-11-09：Windows Agent完成ROUTE-B-48固定周期原生释放即采样对照。79项检查通过。真实板端模型11双机50/120 ms两组反序配对，四次均711/711按期交付、24/24穿门、零碰撞。释放至采样P99由2.66/2.72 ms降到0.66/0.73 ms，端到端由18.94/18.95 ms降到15.99/14.93 ms。每轮仍有7次条件物理预算超期，最长请求仍在命令确认阶段等待。在线周期提交接入和高负载验证待做。见[问题日志B48-01至03](../controller_architecture_study/docs/ROUTE_B_ISSUE_LOG_20261109.md)及`results/ip37_native_sampling_20261109_114514/`、`results/ip37_native_sampling_reverse_20261109_114651/`。

2026-11-09：ROUTE-B-47完成平台重审。旧ETL主线程候选在46.14ms区间Waiting约45.68ms；新setup增加发布线程ID。现有C++ mock日历在合成压力下P99约0.52–0.59ms，但Python消费P99仍249–460ms，迁移定时器不足。9项原生、36项IP37测试通过。决定先做Windows C++发布即采样的最小固定周期对照，再按数据决定准备段迁移或PS。见问题日志B47-01/02及`results/native_release_diagnostic_20261109_v1/summary.json`，尚未接入路线B。

2026-11-09：ROUTE-B-46闭环复测628/630交付，飞行/机身通过、严格时序失败、3次安装。参考窗口中位1.350→1.256ms，P99 10.551→13.612ms。两次丢弃来自Drone1 cycle199发布迟46.14ms后准备超期及cycle200线程仍忙；ETL验证参考/编码阶段仍有阻塞。记录见问题日志B46-03及`results/ip37_kernel_20261109_112606/etl_drop_analysis.json`。

2026-11-09：ROUTE-B-46去除板端237D路径未使用的参考角速度求值，70项检查通过，编码逐元素一致。交错离线1000窗口中位0.713→0.598ms，P99 1.228→1.099ms。复核ETL栈定位到Python/NumPy模块级，未识别具体锁；闭环阻塞改善待复测。见问题日志B46-01/02。

2026-11-09：ROUTE-B-45补充用户分段WPR实测。653次全部交付、飞行/机身/时序通过、3次安装，但1932次无可行修复。最大准备30.10ms中参考窗口段22.82ms，ETL确认该段约14.09ms阻塞。见问题日志B45-02，证据`results/ip37_kernel_20261109_110631/etl_reference_analysis.json`。

2026-11-09：ROUTE-B-45完成准备长尾ETL复查和分段插桩。旧样本有多段约4.9–5.95ms阻塞，具体调用尚未定位。沿用WPR脚本采集新增`thread_marks_s`与`prep_phases_ms`；64项相关检查通过，待下一轮板端ETL。逐项证据见`controller_architecture_study/docs/ROUTE_B_ISSUE_LOG_20261109.md`的B45-01。

2026-11-09：ROUTE-B-44补充用户WPR实测，992/1004按时交付，飞行和机身通过，严格时序失败，零安装。新增下发/ping分段有效。ETL确认一次准备长尾在碰撞回读后阻塞约72.86ms，另一次推理丢弃的44.25ms往返中板端服务仅1.319ms。1862次拒绝主要为未找到有效动作。逐项记录见`controller_architecture_study/docs/ROUTE_B_ISSUE_LOG_20261109.md`的B44-05，未改运行策略。

2026-11-09：ROUTE-B-44修复命令期限与worker空闲时刻混用、已返回命令的旧期限重复检查，新增下发和ping分段记录及预测超期任务日志。64项相关测试通过。212快照完整准入29→28，证明仍需继续拆分共享CPU和RPC等待；当前无AirSim RPC监听，待板端复核。逐项问题、证据和状态见`controller_architecture_study/docs/ROUTE_B_ISSUE_LOG_20261109.md`。

2026-11-09：ROUTE-B-44完成237D gain7教师微调交接文档`controller_architecture_study/docs/HANDOFF_237D_GAIN7_FINETUNE_20261109.md`与交付包`results/gain7_handoff_20261109_v1/237D_gain7_handoff.zip`。28个checkpoint（含独立模型和单独标记超网），旧Train/Validation、归一化及来源清单；95个源文件逐项SHA256对比zip通过，整包SHA256见同目录。旧U是gain3.5，不能直接训练gain7；数据转换/验证代码由同事实现，未运行训练、量化、Test或修改旧教师。

2026-11-08：ROUTE-B-43完成经验周期集合接口、60项相关检查及两次双机试飞。默认不限制，指定JSON须匹配case/model/evidence；急弯只允许50、平稳50/120，两轮急弯余量7.14/7.51cm、均机身/飞行通过，但严格时序均失败；r0安装平稳降频一次，r1安装0，长尾/准入未解决。该经验限制非通用状态反馈或认证。证据`results/qualified_observer_20261108_v1/`。新增易读现状说明`docs/CURRENT_IMPLEMENTATION_STATUS.md`，明确部署、控制、独立观测、当前缺口及阶段。

2026-11-08：ROUTE-B-42完成独立RPC版本检查观测原型及两次双机板端试飞。安装13/6次，r0丢推理1次、机身余量-1.20cm不合格，r1无丢弃、机身及严格时序通过；两次无碰撞，unpaired跳过1630/1344。状态更新中位约1.67–2.23ms、P99约7.58–8.15ms，含等待且非WCET。新增默认关闭`--independent-observer`；版本变更期间拒绝观测，旧观测不可覆盖新观测；56项相关检查通过、26项Runtime/Pilot复核通过。报告`controller_architecture_study/docs/INDEPENDENT_OBSERVER_20261108.md`。未证明动态优势或彻底解决缺状态，不能自动认证周期。

2026-11-08：ROUTE-B-41完成5次配对回读时机诊断，均飞行/机身通过、时序失败。全部延后及有余量才立即回读两模式周期安装均0；全部延后出现约2390次unpaired跳过。默认立即回读组安装5次，仍4次丢弃。53项旧相关加新增后为55项通过，随后slack新增分支检查通过；保留默认立即配对，两个新flag仅诊断。报告`controller_architecture_study/docs/REFRESH_POLICY_RESULTS_20261108.md`，证据`results/refresh_policy_20261108_v1/`。需要独立、有预算、版本正确的状态发布路径，尚未实现，不因本结果迁移PS。

2026-11-08：ROUTE-B-40读取用户WPR实测，ETL片段成功提取。双机626交付、1 busy丢弃，均飞行/机身通过、时序失败。Drone4上一任务R61.09ms<80ms但总worker占用92.01ms，故下次release丢弃；命令等待54.33ms中运行1.55ms、Ready0.43ms、Waiting52.35ms。报告`controller_architecture_study/docs/IP37_KERNEL_FINDINGS_20261108.md`，证据原目录`results/ip37_kernel_20261108_192601/etl_thread_analysis.json`。阻塞根因仍未区分GIL/RPC/服务端，不建议立即迁移PS。

2026-11-08：ROUTE-B-39修复`run_ip37_kernel_trace.ps1`在Windows PowerShell 5.1下无BOM UTF-8中文被按本地编码误读的解析错误，脚本提示改为ASCII。实际`powershell.exe` 5.1.26100.9711 ParseFile返回0错误，ASCII字节检查通过；未启动WPR或试飞。此前仅在当前终端解析的检查不足以覆盖5.1编码差异。

2026-11-08：ROUTE-B-38逐线程周期诊断完成，固定/动态各一次均飞完、机身通过、时序失败。74.51ms命令等待仅约1.78倍正常CPU周期，支持明显等待成分但未定位内核/GIL/RPC。新增`thread_cycle_trace.py`与默认关闭的`--trace-thread-cycles`，54项相关检查通过；管理员采样入口`run_ip37_kernel_trace.ps1`语法通过，当前非管理员未采ETL。报告`controller_architecture_study/docs/THREAD_CYCLES_FINDINGS_20261108.md`。继续Windows，不认为PS自动解决；待管理员内核跟踪区分真实根因。

2026-11-08：ROUTE-B-37完成旧日志逐阶段审计和事后P99反事实诊断，见`controller_architecture_study/docs/ONLINE_COST_AUDIT_20261108.md`；P99对照通过数r0 16→87/109，r1 13→57/103，但不是实时上界，不可上线。发现r1约17.28s两机超期相关，粗粒度CPU计时不能定位共同根因。54项相关检查通过；在线模型、成本与周期资格尚未完成修复，问题未解决，需按命令前/后和每机IO资源准确重建并做高精度跟踪。

2026-11-08：ROUTE-B-36完成在途剩余操作修复及历史抽样回放，53项相关检查通过，另新增post阶段回归检查也通过。212个快照原/新通过数均29，新增通过0，故重复计费修复不是本批大量拒绝的充分解释。保留历史成本与期限；周期资格未凭少量试飞自动认证，未新试飞。报告`controller_architecture_study/docs/REMAINING_CHAIN_REPLAY_20261108.md`，证据`results/remaining_chain_replay_20261108_v1/`。

2026-11-08：ROUTE-B-35完成双机启发式拒绝定位，未改在线策略。r0/r1拒绝1838/1797次，no_effective_action分别1837/1797，budget_exhausted仅1/0；历史最大墙钟准备/收尾成本膨胀，且carry阶段可能重复计入已完成前端，周期候选默认全部qualified。详细证据与限制见`controller_architecture_study/docs/ABC_REJECTION_DIAGNOSIS_20261108.md`；下一步按阶段建模和回放，明确控制质量资格。

2026-11-08：ROUTE-B-34在线启发式双机板端探索两轮：均飞完，r0有1次dropped_inference、r1有1次command_return_late、1次dropped_preprocess和2次dropped_busy_io，严格时序均失败；r0 Drone1第四门余量-1.00cm、r1最小余量+5.09cm，均无碰撞。周期变更分别7/4次，全部发生在约0.32–2.38秒，第四门参考时刻5.19秒；危险回路50ms被放宽到80/120ms，没证明危险前成功加频。决策状态no_feasible_repair分别1838/1797次，一个样例为受预算限制而未通过预测，不能直接当成真实全局不可行。证据`results/two_loop_adaptive_20261108_v1/`。需解释代价界、保守安全时域与预算限制并修复后再做闭环；不能声称动态优于固定。

2026-11-08：ROUTE-B-33完成双机复测（每配置累计3轮），用户暂缓PS迁移比较，固定Windows＋板端DPU。两组均3/3飞行及机身通过；两机50ms严格时序1/3通过、3009/3012交付，混合50/120ms为3/3、2133/2133。累计服务占窗口约5.3%/3.7%，平均板前排队0.03–0.35ms，尚非DPU拥塞；危险机余量/响应未稳定改善。详细报告`controller_architecture_study/docs/TWO_LOOP_REPEAT_20261108.md`，逐机证据`results/two_loop_mixed_speed_20261108_v1/per_vehicle_analysis.json`。下一步在线安装检查可在本场景进行，但竞争收益需真实负载证据。

2026-11-08：ROUTE-B-32双机板端初试，原快速两路线中Drone4的237D输入索引7（首参考相对位置x）四次超范围，保留`results/two_loop_allocation_20261108_v1/`。随后保持Drone1快速路线、将Drone4恢复原速，参考几何检查通过；`results/two_loop_mixed_speed_20261108_v1/`中两机50ms及50/120ms各一次均飞行、机身及期限检查通过，分别1004/711次交付；Drone1最小余量6.10/6.37cm，Drone4为89.95/87.84cm。仅各一次，失败回合不可作为完整配对；固定分配不代表动态策略或板端耗时节省。下一步重复有效配对，记录负载和分机响应，再扩展动态策略。

2026-11-08：ROUTE-B-31完成板端探索，`results/fast_route_followup_20261108_v1/`：Drone1快速路线120ms额外等待20ms两次，无碰撞、时序通过，第四门机身余量-4.30/-4.88cm，平均响应29.5/29.4ms；先前自然延迟120ms为-1.79/+0.30cm，50ms为+7.27/+7.48cm。Drone4平稳路线50/120ms各一次均通过，最小余量89.45/90.39cm。少量单机试验，未证明动态方案，负余量非实际碰撞。后续需不同危险路线、多次重复和可调度资源共享对照。

2026-11-08：ROUTE-B-30完成快速路线筛查：70%路线时间缩放0.85，门位置不变，速度峰值约7.19m/s、加速度峰值约10.84m/s²，几何audit通过。真实板端单机50ms和120ms各2次，均飞行及时序通过；50ms机身余量两次约0.07m，120ms分别约-0.02m及+0.003m（一次body检查失败，均未报告碰撞）。结果在`results/fast_route_screen_20261108_v2/`；仅单路线小样本，既非动态方案效果也非严格统计。v1仅离线生成且角色时间元数据未缩放，以v2为准。下一步核查门4余量、重复不同路线并做响应延迟对照。

2026-11-08：ROUTE-B-29完成70%路线瞬态分析和额外中间门试飞，代码`compact_gate_screen.py`，证据`results/compact_gate_screen_20261108_v1/`。原场景最大轨迹误差约1.14–1.17m，集中在约1.4秒起步；命令返回平均间隔与80/120ms设定接近。固定参考轨迹和速度，在两个间隔2秒的门之间加一门，几何audit通过；80/120ms各飞一次，均通过14门及时序，最小穿门余量约0.44/0.42m。新增门未改变连续转弯和参考速度，不能作为更紧凑转弯或周期敏感结论；下一步设计曲率与速度明确、相邻方向变化更快的参考路线并单独审计。

2026-11-08：ROUTE-B-28完成70%路线单机真实板端初筛，`fly_ip37_joint.py`新增默认关闭的推理后延迟，超期仍丢弃；34项相关测试通过。六次飞行均完成且时序通过：80ms、120ms自然延迟以及80ms加20ms各两次，结果在`results/timing_screen_20261108_v1/`；过门最小余量依次约0.42–0.43m、0.37–0.41m、0.38–0.41m，响应均值约8.3–8.4ms、7.3–7.6ms、28.6–29.2ms。仅为单路线少量筛查，命令返回是代理而非物理生效时间，尚不能证明动态调度收益或整个轨迹安全余量；下一步检查瞬态和更紧凑门距等独立变量。

2026-11-08：ROUTE-B-27完成单机板端环境初筛。新增`screen_lateral_cases.py`及2项测试，34项相关测试通过。原3.5增益模型、净宽3m、T=50ms、固定门前后时间：横向比例1.0两次均碰撞同一门；0.7、0.4各两次均完成且时序检查通过。完整结果见`results/lateral_screen_live_20261108_v1/summary.json`；碰撞组余量仅覆盖已穿过的门，不代表碰撞门余量。板端5238黄金输入核验误差约4.3e-9；早前沙箱内ping失败是权限限制。下一步用0.7候选做周期/响应时间对照并记录瞬态，原始1.0为困难参考；不能据本初筛宣称周期敏感或动态策略收益。

2026-11-08：ROUTE-B-26由Windows Agent完成[目标计划](RESEARCH_GOALS_AND_PLAN.md)及协作入口更新，覆盖每阶段工作、交付、验收/停止条件、实验规模、Windows/PS部署选择和Ubuntu开发流程。保留增益3.5板端DNN，不默认微调/全量迁移PS；文档相对链接已检查。未修改运行代码或启动实验。下一实际编码任务是评价口径的最小修复和单机环境初筛入口。

2026-11-08：ROUTE-B-25由Windows Agent完成只读审计和研究定位说明；原始summary哈希及派生结果保存在`results/ip37_campaign_analysis_20261108_v1/`。核实122次、18对、4机9对J可比较，无严格配对救回；错误/前端/资格混用/成本过保守分开归因。核对AirSim接口、采样平台模型、Samii/DECNTR等文献，阅读范围和下一步门槛见上述详细报告。不改原报告、运行代码、教师或模型，不启动新试飞/训练。

2026-11-08：Windows Agent已完成ROUTE-B-24门宽修订，默认净宽调整为1.9/2.4/3.0m，独立v2结果目录；机数、搜索预算和配对次数不变，已通过PowerShell离线计划及三档参考几何审计，真实飞行由用户执行。

2026-11-08：Windows Agent已完成ROUTE-B-24缩减矩阵：默认4/7/10/12机与1.7/1.9/2.1m，新结果目录；搜索预算/配对次数保留。通过实际PowerShell计划入口核对12格、最多216次、纯飞行1.56小时；三档参考几何审计通过。真实执行待用户运行。

2026-11-08：ROUTE-B-24由Windows Agent完成代码与离线验证，**待用户实飞**。涉及`ip37_safety_cases.py`、`ip37_joint_runtime.py`、`fly_ip37_joint.py`及新增`ip37_evaluation.py`、`ip37_campaign.py`、`run_ip37_campaign.ps1`、测试和[说明](../controller_architecture_study/docs/IP37_FULL_CAMPAIGN_20261108.md)。74项检查、PowerShell语法及完整离线计划通过；6档门宽×12机参考几何审计通过。静态配置冻结、独立评价、失败/采用记录、续跑已覆盖模拟集成；真实模型包/教师不改，不预设收益。默认用户运行入口最多972次，已完成原始报告可复用，源码/参数变化必须新建批次。

任务状态使用：`待认领 / 进行中 / 待验证 / 已完成 / 阻塞`。注明具体完成层级，不用一个PASS代替所有结论。当前未来任务均未认领；认领后填Agent/协作者名称和日期。

| ID | 优先级 | 任务与最小交付 | 当前状态/负责人 | 完成条件 |
|---|---|---|---|---|
| ROUTE-B-21 | P0 | 固定gain7的独立R轴离线诊断 | Windows Agent，2026-11-07已完成[证据边界与5条时延模型轨迹](../controller_architecture_study/docs/TIMING_SENSITIVITY_LIMITS_20261107.md)。T=120ms固定，R=7～100ms均小于T；RMSE约0.0825→0.1480m，仍无负几何余量；未改教师或代跑实飞 | 已确认10个周期实飞点均13门、无明显快周期优势。存在离线延迟性能影响，尚无安全失效分离；停止细抠T或增益，下一步先研究任务时间尺度及参考可预测性，不启动训练 |
| ROUTE-B-22 | P0 | 用户指定增益5的独立诊断 | 用户194816批次实飞已完成，Windows Agent于2026-11-07回填；四周期均13门零碰撞、余量约2.7～3.0cm，仍无明显快T优势；T30有1迟返和3忙丢弃，其他三档全交付 | [原始设计和实飞回填](../controller_architecture_study/docs/GAIN5_DIAGNOSTIC_20261107.md)保留，不修改旧模型或数据 |
| ROUTE-B-23 | P0 | 增益5实飞回填与有界模型筛查 | Windows Agent，2026-11-07代码/离线已完成；新增`teacher_timing_screen.py`及独立诊断增益覆盖。20组模型轨迹、4组步长检查、[原理/结果](../controller_architecture_study/docs/TEACHER_MODEL_SCREENING_AND_TIMING_20261107.md)已保存；44项检查通过，原教师不变 | 没有找到稳定快T优势；K10是短/长R实飞候选、不是冻结教师。下一步固定T独立R对照，保留K7；新R飞行入口尚未实现，用户执行真实实验 |
| ROUTE-B-20 | P0 | 固定教师细粒度周期列表 | 用户191116批次补测已完成，Windows Agent于2026-11-07核验：六组均13门、时序通过。合并原四组共有30～120ms每10ms实飞点 | 未发现明显快周期优势；转向ROUTE-B-21独立R轴离线诊断，不再细分当前T或改变增益追结果 |
| ROUTE-B-19 | P0 | 固定教师后的周期敏感性入口与新标签迁移方案 | Windows Agent，2026-11-07已交付[脚本、命令和迁移原则](../controller_architecture_study/docs/FIXED_TEACHER_TIMING_AND_TRANSFER_20261107.md)。PS语法、冻结case路径与CLI入口通过；读取训练器确认init_checkpoint可重置优化器，尚未执行新标签训练 | 固定gain7，四周期与实际R；独立R轴随后实现。真实试飞待用户，主循环/教师/DNN未改；尾部漏周期仍如实记录，未宣称全硬实时或迁移质量保证 |
| ROUTE-B-18 | P0 | 独立教师姿态增益／一致前馈对照 | 用户四组实飞已完成，Windows Agent于2026-11-07核验[实测与时序诊断](../controller_architecture_study/docs/TEACHER_ATTITUDE_COMPARISON_20261107.md)。原版撞第4门，三改进均13/13门，最小余量33～40cm。原教师哈希不变 | 控制能力改善成立于本批次，动态调度收益及严格实时未成立。末段第788条控制后的安全分析30～63ms引起漏周期，待查内部原因；优先保留gain7与组合候选，未重训或替换板端模型 |
| ROUTE-B-17 | P0 | 解释30/80ms实际收益为何较小 | Windows Agent，2026-11-07已完成[实际时序审计与8条受控离线对照](../controller_architecture_study/docs/IP37_PERIOD_EFFECT_DIAGNOSIS_20261107.md)。3.8m采样/下发均为30/80ms；原姿态反馈在理想角速度执行下也出现约1.1m偏差，去掉姿态过程仅约3～8cm；积分细化变化0.0155mm | 有力支持教师预测/姿态执行不一致掩盖周期收益，未称为完整真实动力学辨识。下一步隔离版本做教师响应对照；未改真实控制、训练、量化或启动飞行 |
| ROUTE-B-16 | P0 | 核对20ms教师仍碰门的控制与时序原因 | Windows Agent，2026-11-07完成[602输入重放、诊断及下一轮命令](../controller_architecture_study/docs/IP37_GATE4_DIAGNOSIS_20261107.md#5-20ms实际结果教师及时更新仍碰第4门)。输入重编码与教师输出复算误差均9.54e−7；3档宽度参考审计与布局等价核对通过。6次实飞待用户运行 | 教师没有周期超期仍碰门；软跟踪走廊不等于硬门框约束，姿态/平移链存在滞后证据。探索性宽度3.0/3.4/3.8m、80/30ms；保留全部结果，未称其为J最优静态/动态收益 |
| ROUTE-B-15 | P0 | 分析自然弯道80/30ms同门碰撞 | Windows Agent，2026-11-07已交付[诊断及20ms能力对照](../controller_architecture_study/docs/IP37_GATE4_DIAGNOSIS_20261107.md)。核对 `ip37_recovery_pair_20261107_133948`，278条同输入离线对齐；56项相关检查、PS语法通过，真实对照待用户执行 | 两组均通过3门后碰第4门；周期时序通过。未认定量化/拟合/教师哪项主导。新增可选单机固定周期CPU教师诊断，记录独立来源，不改模型、教师参数或原报告 |
| ROUTE-B-14 | P0 | 修复配对试飞被本实验旧门框阻止 | Windows Agent，2026-11-07已交付。只读RPC确认残留 `FalconJointGate_Drone3_1_top`；新增启动清理与核验、4项边界回归，27项运行检查通过。用户实际重跑待回填 | 仅处理精确匹配 `FalconJointGate_DroneN_门号_部件` 的对象，外来对象保留；保存 `gate_preflight.json`，未确认删除则不飞。原失败两组请求均0，不能归因于板端/控制能力；本轮未删除现场对象或代飞 |
| ROUTE-B-13 | P0 | 接通危险场景的首轮周期能力配对飞行 | Windows Agent，2026-11-07已交付[配对入口与命令](../controller_architecture_study/docs/IP37_RECOVERY_PAIR_RUNBOOK_20261107.md)。23项运行、4项场景和23项启发式检查通过，PS语法通过；未代跑真实飞行 | 同一冻结case单机80/30ms、13门净宽2.1m。已分开命令预留与后续回读、补尾部释放、增加线程CPU分解和逐门余量。ABC全历史最大值预测仍待修，本轮仅测固定周期能力，不宣称动态救回或完整J收益 |
| ROUTE-B-12 | P0 | 面向局部危险纠正的实验设计与自然场景准备 | Windows Agent，2026-11-07方案与离线场景已交付。见[实验设计](../controller_architecture_study/docs/SAFETY_RECOVERY_EXPERIMENT_20261107.md)、`ip37_safety_cases.py`；4项检查通过，生成 `results/ip37_safety_design_20261107/seed0_w21/`，未实际飞行 | 六机26秒、转弯中穿门、前/中/后危险错开。比较最佳找到的静态、始终快更新和动态；先确认周期恢复差异。正式入口仍需case接入、成本分解/预留修正、释放覆盖和统一J观测，不把离线几何通过当作控制器可飞 |
| ROUTE-B-11 | P0 | 分析首次通过的真实板端三机固定/启发式对照 | Windows Agent，2026-11-07已完成只读核验和[分析报告](../controller_architecture_study/docs/IP37_JOINT_FIRST_RESULTS_20261107.md)。用户实际运行 `ip37_joint_20261107_125546`；本轮未改控制代码或代跑试飞 | 两组15/15；动态5组/7次更新、请求减35.45%、平均位置MSE增1.19%。规划P99 0.960ms；1476次拒绝受83.96ms历史最大值合计影响。固定组第三机末尾1条计划释放缺记录；下一步固定80ms、成本分解和日历覆盖 |
| ROUTE-B-10 | P0 | 修复IP37多机预启动空窗及错误展示 | Windows Agent，2026-11-07已交付。18项运行检查及PS语法通过；用户复跑 `ip37_joint_20261107_125546` 两组均15/15穿门、无输入范围拒绝，启动修复实际生效。见[原因与修复](../controller_architecture_study/docs/IP37_STARTUP_FAILURE_20261107.md) | 首轮约0.49s下坠记录保留。新重力回归旧代码先失败、修复后通过；暂停等待至起点、显式悬停保持、终端错误直显。未改模型或量化范围 |
| ROUTE-B-09 | P0 | 将A/B/C启发式接入IP37多机实际控制 | Windows Agent，2026-11-07已交付并取得真实三机采用记录。动态实际安装5组、7次周期改变；详细数字与限制见ROUTE-B-11及[运行说明](../controller_architecture_study/docs/IP37_JOINT_ONLINE_RUNBOOK_20261107.md) | 独立发布、周期超期丢弃、整任务EDF、活动任务保留、状态/控制配对、全组检查与版本提交；仅用RC1模型。成本估计及尾部发布覆盖待修；尚未搜索FALCON式J最优静态基线 |
| ROUTE-B-08 | P0 | RC1 单机实际控制试飞 | 用户实飞完成，Windows Agent 于 2026-11-07 核验 `results/ip37_actual_flight_20261107_114530/result.json`。RC1 实际控制，5/5门、400条输出、零超期、零跳过、零限幅、轨迹RMSE 0.297m | 原影子 wy 偏差在闭环中未变成持续发散，最后4秒平均位置误差0.091m、平均速度0.146m/s。RC1可用于后续原型实验，复杂路线、多机效果另行对照 |
| ROUTE-B-07 | P0 | RC1 独立路线影子验证 | Windows Agent，2026-11-06。env50/route04 与 env49/route02 各 1,200 条候选请求全部按时，旧版同输入共拒绝 70 条。[报告与悬停复现](../controller_architecture_study/docs/IP37_RC1_INDEPENDENT_ROUTES_20261106.md) | RC1 在两条路线末段悬停均持续输出 wy ≈ −0.160rad/s，FP32 接近零。优先修复输出偏差，仍由 MPC 控制 |
| ROUTE-B-06 | P0 | 复验 IP37 小模型重校准候选 | Windows Agent，2026-11-06。1,093 条固定输入和 1,200 条三机实时影子请求全部通过。[复测报告](../controller_architecture_study/docs/IP37_RC1_WINDOWS_RETEST_20261106.md) | 同一批实时输入旧版拒绝 33 条，候选均返回。共同有效样本的 wy RMSE 相对 FP32 从 0.0471 升至 0.1075rad/s，部署精度与控制接管分开验收 |
| ROUTE-B-05 | P0 | Windows IP37 黄金验证与 AirSim 状态影子推理 | Windows Agent，2026-11-06。两模型各 32 条黄金与各 20 条历史输入通过。三机实时影子完成，分别有 36、35 条量化范围拒绝。[报告与复现输入](../controller_architecture_study/docs/IP37_WINDOWS_SHADOW_RESULTS_20261106.md) | 已定位 `x[7]`，采集 37 条 raw237，向两模型回放 74/74 次复现。下一步重校准并复测；控制仍由原 MPC 下发 |
| ROUTE-B-04 | P0 | 双模型成本核验、同步期限选型及 Windows IP37 连通 | Windows Agent，2026-11-06。核验 105 个文件、18,000 条请求、6,000 个配对，完成 840 个回放案例，23 项单元测试通过。[结果与启动命令](../controller_architecture_study/docs/ROUTE_B_PAIR_BOARD_20261106.md) | 12 路同步 50ms 情况至少 9 条换小模型。板端启动后 Windows 黄金已通过，真实输入结果见 ROUTE-B-05 |
| ROUTE-B-03 | P0 | 核对 AXU3EGB 板端数据并进行校准回放 | Windows Agent，2026-11-06。已核验 88 个文件、69,600 条正式请求，完成 1,200 个实测到达与假设期限的回放。[板端校准报告](../controller_architecture_study/docs/ROUTE_B_BOARD_REPLAY_20261106.md) | 同步队列 P99 已复现。单紧迫任务与整任务 EDF 持平，窗口需求超载时都无法修复。下一步补实际控制期限、端到端路径和另一合格模型成本 |
| ROUTE-B-02 | P0 | 实现 A/B/C 决策分支并交接 237D 板端测量 | Windows 原型与交接已交付，2026-11-06。21 项检查通过，完成 480 个旧种子组合与 960 个新种子组合的配对回放。[决策与结果](../controller_architecture_study/docs/ROUTE_B_ABC_DECISION_20261106.md) | 12 回路平均决策变快，但通过率仍低于只换模型。下一步由 Ubuntu Agent 按[测量交接](../controller_architecture_study/docs/UBUNTU_237D_TIMING_HANDOFF_20261106.md)采集实际 PS/PL、队列和端到端数据 |
| ROUTE-B-01 | P0 | 瓶颈感知联合控制的机制、反例与有界启发式 | 阶段完成，Windows Agent，2026-11-06；独立 `bottleneck_*` 代码、15项离线检查、研究文档与v5合成结果；不改路线A执行器 | 已区分Windows实测/合成假设，给出必要条件、局部动作公式和完整搜索对照；480案例最终复核一致，源码/结果哈希保留。12回路开销与漏解未解决，原生/板端/实飞验证未完成 |
| DOC-02 | P0 | 保存现行动态控制路线的恢复点 | 已完成，Windows Agent，2026-11-06。[路线A恢复记录](../controller_architecture_study/docs/ROUTE_A_CHECKPOINT_20261106.md) | `colleague_transfer/route_a_checkpoint_20261106.zip` 保存源码、运行依赖及试飞证据。1895个文件逐项校验通过，解压后的命令入口和6项周期提交检查通过 |
| SHARE-237 | P0 | 补齐Git克隆后的237D独立模型交接与路径迁移 | 已交付代码及zip，Windows Agent，2026-11-05。[交接说明](../controller_architecture_study/docs/SHARE_237D_ARTIFACTS_20261105.md)。新增推理路径解析和内容校验，冻结采集/增广源码保持哈希可核验。原始11文件加默认路线已打包，约241 MiB | `colleague_transfer/residual237d_collaboration_v1_20261105.zip` 及 `.zip.sha256`。换目录解压后来源检查和32条原始验证输入推理PASS。实际试飞和同事机器环境待回填，单包不覆盖六机实时实验的完整依赖 |
| PLAN-04 | P0 | [直接分配计划](../controller_architecture_study/docs/STATE_FEEDBACK_ALLOCATION_PLAN_20261103.md)的周期执行与测量 | 分核与通知修复阶段已完成，Windows Agent，2026-11-05。新增16轮六机试飞全部零碰撞，动态有真实周期改变。[阶段结果](../controller_architecture_study/docs/RT_SPLIT_IO_RESULTS_20261105.md)。最新相关离线检查42项通过 | 下一阶段核对周期需求函数为何给直线回路加频，以及1.60m提交全拒绝的快照。1.75m平均J约提高0.98%但丢弃增加，尚未证明整体安全优势 |
| RT-01 | P0 | 修正保持控制义务的时间归属，先做归档回放与针对性检查 | 归档回放已有局部改善，仍有332条重复过期记录；剩余问题待继续定位 | 旧义务结束与当前义务可区分；历史超期仍保留；当前期限、在途成本及全组检查不被绕过 |
| DIAG-02 | P0 | 核对最新建议的触发、拒绝、提交时间线和总开销，提出可复现异质负载验证方案 | 已完成分析；Windows Agent，2026-11-02；见[拒绝与开销诊断](../controller_architecture_study/docs/DYNAMIC_ADVANTAGE_AND_ADMISSION_DIAGNOSIS_20261102.md)；未改运行机制或试飞 | 首撞前38条全拒绝；发现建议服务缺少执行等待界限、过渡期限阻塞；有条件跳过及公平对照仅为下一步建议 |
| RT-03 | P0 | 分离旧义务重绑与发布保护、给建议处理计费的执行机会、CPU调度与XPU段执行分离 | v2已由用户运行，功能有局部改善、控制收益未成立；见[结果](../controller_architecture_study/docs/RUNTIME_SERVICE_REPAIR_RESULTS_20261102.md) | 事实刷新1960次、服务最大间隔降到约90ms，但新配置安装0；端到端响应与尾延迟仍待解决 |
| RT-03-H1 | P0 | 修复安全分析字典字段重复使第三组中止 | 已完成，v2六次执行未再出现该异常；Windows Agent，2026-11-02 | 137项检查通过，真实完整执行复核；v1失败报告保留 |
| DIAG-03 | P0 | 分析v2六次完整运行的实际采用、CPU服务和CPU/XPU交互时序 | 分析已完成；Windows Agent，2026-11-02；见[结果](../controller_architecture_study/docs/RUNTIME_SERVICE_REPAIR_RESULTS_20261102.md)和v2派生摘要；未改运行算法 | 已区分事实刷新/配置安装/实际交付，拆分发布到派发与段交互，保留初始不可行和完整J缺失 |
| DATA-01 | P1 | 在已有临界范围有限补采配置能力确实不同的状态 | 本轮64次采集完成；3个临界周期差异、0个模型资格差异，稳健性未证 | 预先限定查询数；匹配初态/参考/控制/时序；保留同成同败；独立新路线确认；没有差异也如实报告 |
| SEL-01 | P1 | 基于新证据复核小评分器或更简单的资格映射 | 待认领；依赖DATA-01 | 优于明确默认项的可重复证据；包含准备、评分、检查的总开销；不是只有排序准确率 |
| RT-02 | P1 | 在RT-01后验证新的期限/周期建议真实提交 | 六次对照已运行；仅碰撞后安装，T/释放实际改变0，收益未成立 | 原生安装、带版本的实际命令、真实周期改变三者可追溯；失败结果完整保留 |
| EXP-01 | P1 | 扩展自然异质路线静态/动态对照 | 待认领 | 同模型池/资源；重新说明离线静态选择预算；多种子配对；碰撞、J、时序、未采用建议均报告 |
| HW-01 | 可并行 | Ubuntu自定义PL功能实现与整数golden核验 | 远端状态待回填 | 按冻结包先记录工具/器件；C仿真、综合/协同仿真、板端结果分层给证据；质量例外不变成PASS |
| HW-02 | 后续 | FPGA时延、上下文、DDR/网络与分段成本回填 | 待认领；依赖HW-01 | 当前237D/超网的真实测量，重新标定模型；不使用CPU或旧25D时延替代 |
| BASE-01 | 可并行 | 独立237D模型池的真实DPU切换与命令跳变实验 | 远端状态待回填 | 按27模型交接包核对量化/编译和模型身份；区分时间jitter与控制输出跳变 |
| DOC-01 | 已交付 | 建立协作入口、框架和共享进度 | 已完成；本次Windows Agent；2026-11-02 | 本文、FRAMEWORK、根AGENTS/README、分支索引及原文快照可相互导航 |

范围提示：RT-01主要涉及`joint_contract_recovery.py`、`joint_commit_guard.py`、`joint_contract_shadow.py`与`probe_seven_continuous_timing.py`；DATA-01主要涉及`probe_recovery_boundary.py`与`controller_selector/`；SEL-01主要涉及`controller_effect.py`、`repair_controller_effect.py`。认领时写实际要改的文件，避免多Agent同时编辑实时主循环。

### 暂不作为默认下一步

- 不继续无目标扩大网络或延长训练，不为了得到单调复杂度曲线改变测试划分。
- 不为提高提交率放宽全部deadline，不把新建议未采用的飞行称为学习控制成功。
- 不把中途随意改深宽、无限细抢占、动态增加PL核数同时加入第一版。
- 不把超网替代独立模型池当作已经证明的必要条件；先保留可比较基线。

## 6. 关键产物位置

以下路径相对工程根目录；Ubuntu可使用自己的工程根目录映射，不能照抄Windows盘符。

| 产物 | 相对路径 |
|---|---|
| 当前237D数据 | `controller_architecture_study/results/decision_convex_recovery_v1/datasets/features_recovery/` |
| 主18模型 | `controller_architecture_study/results/decision_convex_recovery_scenic_falcon18_v1/runs/` |
| 残差4模型 | `controller_architecture_study/results/decision_convex_residual4_v1/runs/` |
| 残差33×1024长训 | `controller_architecture_study/results/decision_convex_residual33_long5000_v1/residual_mlp_h33_w1024_seed0/` |
| FP32共享超网 | `controller_architecture_study/results/incremental_supernet_237d_v4_grid5000/export/` |
| 混合精度整数候选 | `controller_architecture_study/results/incremental_supernet_237d_m2_lsq_v3/export/` |
| 自定义PL冻结交接 | `ubuntu_transfer/falcon_incremental_supernet_m3_v1_20261015/`及同名zip |
| 独立池DPU交接 | `ubuntu_transfer/falcon_independent_237d_dpu_pool_v1_20261016/`及同名zip |
| 最新Windows阶段 | `controller_architecture_study/results/boundary_takeover_20261101_v1/` |
| 本次冻结评分器 | 上一目录下`ranker/ranker.npz`，仅代理评分，不是控制器权重 |

数据与模型不是随本文自动附带的。交接给新机器时，按具体任务复制所需数据/权重/manifest，并保持相对路径或明确记录映射；不要默认Git或文档包包含大文件。

## 7. 运行环境与可安全重读的入口

| 项目 | 当前Windows环境 |
|---|---|
| 工程根目录 | `F:\FalconReproduction\airsim_control` |
| 训练/多数研究脚本Python | `D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe` |
| AirSim Python环境 | `F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe`；部分研究入口自行加载客户端路径 |
| Unreal | `E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe` |
| Blocks工程 | `F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject` |
| 多机配置 | `controller_architecture_study/configs/airsim_safety_12.json`，最近对照实际用10回路 |
| 原生DLL | `controller_architecture_study/native_runtime/build/Release/falcon_native_io.dll` |
| 构建入口 | `controller_architecture_study/build_native_runtime.ps1`；已有MSVC/CMake环境 |

不要根据本文PID或过去的Play回复推定当前连接。试飞前核对车辆列表、编辑器未最小化、后台限速设置、CPU配置及是否有并行训练。用户既有偏好是由脚本执行实验，Play通常由用户点击，不操作屏幕；不强制结束有未保存内容的编辑器。

只重新读取本轮结果、不训练也不试飞，可运行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
& $trainPy -m controller_architecture_study.audit_boundary_takeover --root .\controller_architecture_study\results\boundary_takeover_20261101_v1 --application application_rates_only
```

这依赖已有结果文件。首次接手不要直接启动全部训练/试飞脚本；先按所认领任务读取入口和输出位置，避免覆盖别人的运行。相关检查按改动选择；native DLL检查、真实飞行、HLS综合分别报告。

## 8. 协作者怎样共同更新

1. 开工前先读本页和FRAMEWORK，在任务表写负责人、状态、涉及文件与输出目录；有冲突先协调，不覆盖他人的修改。
2. 结束后补一条日志：做了什么、具体证据、未完成什么。更新任务状态，区分“实现完成”和“验证/收益成立”。
3. 影响设计或时间定义的变更同步FRAMEWORK；只影响实验结论的变更写新日期报告并链接，不改历史原始报告。
4. Ubuntu/服务器协作者回填工具版本、目标器件/架构、模型/包hash、命令、产物路径和实测结果；未回传的保持未知。
5. 原模型、manifest、冻结包和失败日志保留。新训练/实验用新版本目录，记录父版本与源码身份。

可复制的更新条目：

```text
日期/时区：
协作者/Agent：
任务ID与状态：
修改文件/设计变化：
执行命令与验证层级：
结果证据路径、版本/hash：
剩余问题：
下一交付与认领情况：
```

### 更新记录

| 日期 | 协作者 | 变化与证据 |
|---|---|---|
| 2026-11-07 | 用户运行 / Windows Agent修复 | 首轮IP37三机两组因启动空窗下坠、继发输入范围拒绝而退出，板端黄金80条正常。保留原报告 `ip37_joint_20261107_124755`；增加重力/暂停时钟回归并修复启动对齐、初始保持和错误汇总，18项运行检查通过。真实复跑由用户执行，详见[修复记录](../controller_architecture_study/docs/IP37_STARTUP_FAILURE_20261107.md) |
| 2026-11-07 | Windows Agent | 交付路线B真实IP37多机在线入口与固定/启发式运行脚本，默认三条异质验证路线、相同RC1模型、同资源对照。48项相关离线检查通过；本轮未启动AirSim或发送板端请求，待用户返回新试飞结果。运行细节、经验成本与安全分析边界见[操作说明](../controller_architecture_study/docs/IP37_JOINT_ONLINE_RUNBOOK_20261107.md) |
| 2026-11-03 | Windows Agent | 按用户要求仅核对并编写[直接状态反馈分配计划](../controller_architecture_study/docs/STATE_FEEDBACK_ALLOCATION_PLAN_20261103.md)。复算376/497过期、完整检查仅16条；区分内部决策0.8ms、后台线程CPU约7–8ms、墙钟约10–15ms。核对经典弹性/反馈调度与过渡安全相关工作；新算法仍未实现，后续由其他模型认领 |
| 2026-11-02 | 用户运行 / Windows Agent分析 | v2六次执行完成，静态10、旧动态9、新增模块9架次碰撞，新建议0安装；事实刷新与CPU服务机制有改善但响应均值约50ms。新增[结果报告](../controller_architecture_study/docs/RUNTIME_SERVICE_REPAIR_RESULTS_20261102.md)及`analysis_summary_20261102.json`，本次未改控制代码或再试飞 |
| 2026-11-02 | 用户运行 / Windows Agent修复 | `runtime_service_repair_20261102_v1`第三组TypeError：两个字典重复传入`is_safety_certificate`；改为字典展开合并，新增CPU服务调用真实分析器的成功/竞态拒绝回归。137项相关检查通过，真实试飞未代跑，v2待用户执行 |
| 2026-11-02 | Windows Agent | 按用户要求修复已知机制，交付[CPU/XPU修复与命令](../controller_architecture_study/docs/CPU_XPU_SERVICE_REPAIR_20261102.md)。135项离线检查通过；真实多进程2模型/6输入误差0，无AirSim连接；耗时审计和六次试飞由用户执行 |
| 2026-11-02 | Windows Agent | 只读分析102条决策及源码，量化前端/后台开销、首撞前38次拒绝与快照空窗；新增[诊断与最小实验路线](../controller_architecture_study/docs/DYNAMIC_ADVANTAGE_AND_ADMISSION_DIAGNOSIS_20261102.md)。生产代码与原始报告未改，未启动试飞；文献核对以DECNTR、弹性周期与状态相关丢任务为主 |
| 2026-11-02 | 用户运行 / Windows Agent分析 | 本轮回放、六次十机对照、64次补采完整结束；没有新增飞行或代码修改，结果与后续聚焦见[本轮结果](../controller_architecture_study/docs/EPOCH_BOUNDARY_RESULTS_20261102.md) |
| 2026-11-02 | Windows Agent | 按用户新流程仅交付期限修复、归档回放和64次边界补采代码，74项离线检查通过；训练/真实回放/试飞未代跑。运行方式与状态见[本轮操作说明](../controller_architecture_study/docs/EPOCH_FIX_AND_BOUNDARY_RUNBOOK_20261102.md) |
| 2026-11-02 | Windows Agent | 汇总120次有效采样和两轮各6次对照，定位旧义务期限问题；详见[阶段报告](../controller_architecture_study/docs/BOUNDARY_TAKEOVER_RESULTS_20261102.md) |
| 2026-11-02 | Windows Agent | 按用户协作要求建立当前入口、框架、看板和原始framework快照。本次仅整理文档，未修改控制/训练/调度代码，未启动训练或试飞 |

本次文档核对：6个入口文档中的63个本地链接均可解析，8个主要产物目录存在，原始framework快照与原文件逐字节一致；12个实时源码文件与上一轮复核记录的哈希一致。后续更新后，这些检查结果只代表本次时间点。
