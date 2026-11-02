# B 恢复余量原型：实现与离线结果

日期：2026-10-29。本页记录初版实现与离线检查。后续已完成十机配对试飞，见[试飞结果与限制](RECOVERABILITY_BUDGET_FLIGHT_RESULT_20261029.md)；尚未通过硬时序约束。

## 本轮改动

Safety Analyzer 现在可以采用恢复余量预算；下一任务的周期、控制器模式、有限抢占和全局准入接口继续沿用。通过配置显式启用新 B，省略配置时使用原几何 B，以便保留可复现的对照。

核心实现是 `controller_architecture_study/recovery_budget.py`。对门框局部坐标中的左、右、上、下四个方向分别计算净距离 d 和有符号接近速度 v，使用 v_+=max(v,0)，不把远离某一侧的速度当成朝该侧运动。

设 b 为假定可实现的纠偏减速度，a 为保持期间向边界加速的上界，l 为命令送达后额外的物理响应滞后。对等待新控制送达的时间 tau，要求

\[
v_+(\tau+l)+\tfrac12a(\tau+l)^2
+\frac{[v_++a(\tau+l)]^2}{2b}\le d.
\]

使用稳定的二次方程正根闭式计算，不做在线轨迹优化或搜索。l 不含采样、排队、DNN计算和命令传输；这些仍单独计入端到端响应 R，避免重复计时。

最终 B 取原几何预算与恢复预算的较小值。新规则不能放宽已有几何期限，也不会将零预算强行抬到 20 ms 最小释放间隔：赶不上时仍标记不可行。

## 有限门框与局部适用范围

门洞不能沿整条航线被当成无限延伸的墙。新分析仅考虑尚未完全通过的门；门的进入检查范围由“下一预算窗口 + 假定响应/制动时间”与纵向最早到达界共同决定，不使用固定的门前 3 m 作为恢复分析范围。原几何对照仍保持原逻辑。

若飞机仍在门前、尚未进入某一侧所定义的门洞接近区域，该侧标记为 `outside_approach_domain`，不把这种正常转弯对齐过程直接判为撞门，也不凭这一条件制造零期限。其他适用方向与原几何条件仍继续检查。

若已经在该局部适用区域内，但剩余间隙不足以覆盖假定的响应与制动过程，则返回零预算及原因。这表示“本局部模型不能给出可等待证书”，不表示任何控制器都必然撞门。

这是一种有限的接近走廊原型，不是完整三维门框可达集。检查范围与上述域处理尚不构成全航线递归安全证明。

## 参数依据必须明确

配置：`controller_architecture_study/configs/recovery_budget_prototype_v1.json`。

| 参数 | 本轮值 | 证据边界 |
|---|---|---|
| 横向/垂向纠偏减速度 b | 2 / 2 m/s² | 研究假设，未证明任何现有 DNN 始终能够提供 |
| 向边界加速度上界 a | 6 / 5 m/s² | 与当前 QP 名义加速度限制一致，不是已认证的 AirSim 动力学界 |
| 命令送达后的物理响应滞后 l | 80 ms | 明确的待测假设，不是计算响应 R，也不是从现有 80 ms 更新上限证明而来 |
| 纵向加速度上界 | 6 m/s² | 同样是名义原型假设 |
| 最大分析窗口 | 150 ms | 保留原型已有上限 |

配置中的 evidence 字段随报告保存。这里没有把教师允许的最大加速度当作 DNN 保证的最小纠偏能力。也没有加入 DNN–MPC 误差变量。

当前分析仍未用实际保持控制推演姿态/执行器动力学，`uses_control_in_bound=false`、`is_safety_certificate=false` 均保持真实。之后接入 Yang 风格分析时，需要完善这部分，而不是修改标志就宣布完成认证。

## 已执行的检查

129 项相关测试通过。覆盖闭式根与整个保持区间的恢复余量、物理滞后的计入、四侧运动方向、旋转门框、门前适用域、绝对期限只收紧、零预算不伪装成 20 ms 可行，以及原独立释放、全局准入和前端保护回归。

离线输入是已有 `global_admission_20261029_v4` 的两个十机试验，不加载权重，不连接 AirSim。5444 个保存状态与原安全分析精确配对；另有 214 条安全分析未找到相同采样时刻的控制状态，明确排除，未用插值补造。所有配对状态的原几何 B 均被重新计算并核对一致。

主报告：`controller_architecture_study/results/recovery_budget_audit_20261029_v3/summary.json`。逐状态结果在同目录的 `seed4_protected_states.json`、`seed5_protected_states.json`。

| 已有轨迹 | 配对状态 | 期限收紧次数 | 其中无法给出正预算 |
|---|---:|---:|---:|
| seed4 | 2687 | 573 | 523 |
| seed5 | 2757 | 598 | 561 |

平稳的 Drone4–Drone8 在两组记录中均没有额外收紧期限。连续转弯与后段转弯回路受到更多限制，说明该规则不是统一缩小所有回路的 B。

但“无法给出正预算”的比例也较高，提示恢复能力假设/局部域可能偏保守，或源轨迹已离开这套假设可认证的区域。不能把这些结果直接当成真实不可恢复，更不能据此声称动态已安全通过。没有为了使数据好看而放宽判据、缩门或调整参数。

## 计算开销

两个离线序列中，原分析平均约 0.14 ms；加入恢复分析后平均约 0.36–0.37 ms，增加约 0.22–0.23 ms。新分析 p99 约 0.65–0.78 ms。这些是在顺序离线调用中测得，不是共享前端的 WCET。

第二组有一次 37.34 ms 调用，其中 36.96 ms 与 Python 垃圾回收重合，GC 回调的实测重合时间已记录在 `slow_calls`。现有飞行主循环在控制窗口内已关闭 GC，本次没有修改该机制。但不能据此保证 Windows 试飞不会有其他长尾。

Windows `thread_time` 的逐调用结果出现约 15.625 ms 量化，故报告只保留累计 CPU 时间，不把其逐调用 p99 误当作算法耗时。

## 运行入口

只重新做离线核对，无需启动 AirSim：

```powershell
Set-Location -LiteralPath 'F:\FalconReproduction\airsim_control'
& 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe' -m controller_architecture_study.audit_recovery_budget
if ($LASTEXITCODE -ne 0) { throw '恢复预算离线核对失败' }
```

之后需要试飞时，使用同一模型池、同一资源、相同初态，对比原动态几何 B 与恢复 B。两组分别按正序/逆序执行；不重新搜索静态基线、不更换 DNN、不加物理扰动，碰撞只退出相应无人机，其他无人机继续。须先启动已有 12 机工程并进入 Play：

```powershell
Set-Location -LiteralPath 'F:\FalconReproduction\airsim_control'
& 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe' -m controller_architecture_study.run_protected_joint --recovery-budget 'F:\FalconReproduction\airsim_control\controller_architecture_study\configs\recovery_budget_prototype_v1.json'
if ($LASTEXITCODE -ne 0) { throw '恢复预算对照未完成，已生成的结果保留' }
```

该入口使用现有观察协议：不可行会被明确记录，不会因为新 B 缺少证书而添加新的提前中止。完成飞行不代表满足硬期限；必须一起看碰撞、准入失败、实际超期、J 和其他回路的变化。新旧模型参数及资源不变，只检验本轮 B 的修改。

后续重点是确认可实现的恢复能力与响应滞后，辨别保守误报和真正提前介入的收益。本节试飞入口现已增加同一冻结规划成本表，两组结果及限制见上述试飞报告；本轮对比的是两种动态B，并非静态FALCON对比。
