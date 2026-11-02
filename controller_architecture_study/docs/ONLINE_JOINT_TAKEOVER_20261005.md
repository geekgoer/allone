# 在线模型/T联合接管（2026-10-05）

## 当前完成与边界

已实现[在线入口与控制器](F:/FalconReproduction/airsim_control/controller_architecture_study/online_joint_flight.py)，复用既有三机共享FCFS推理服务、237D特征、连续轨迹时钟和4个常驻DNN，不修改权重、数据、规划器或既有试飞报告。在线阶段只执行一种策略，不把fixed/period_only/adaptive三种计算同时放在控制关键路径上。

本轮修复后CPU/模拟接口97项相关回归测试通过；真实候选文件只读预检PASS，初始plain4x256、T=D=50ms，共12个已试飞候选组合。本轮没有启动GPU推理、真实AirSim试飞或训练。模拟接口通过不代表实际在线闭环已验收，下一步由用户试飞一次。

## 第71帧准入误中止：定位与修复

原始记录：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_joint_online_20261004T170607_201301Z`。Drone1第71帧选择结束距计划释放30.8405ms，当前D仍为50ms；没有证据把这一准入失败称为当前帧已超时。

原先选中的下一帧`plain4x256_T50`，预测响应49.965992ms；加上实测选择尾部1.003000ms后为50.968992ms，超出下一帧T50。旧逻辑只拒绝这个组合，没有重新检查其他候选，因此中止全部飞机。

使用原记录的候选、响应预测与实测选择耗时进行CPU重算，仍有8个T75/T100组合合格；依照原质量排序，选择`plain6x1024_T75`，预测响应54.324772ms。这一帧门附近状态约束不适用，重算仅证明原观察下还有资源合格选项，不是重新试飞或未来响应保证。单次CPU重新筛选实测约0.065ms，不能把这个单次值当作WCET。

修复范围：

- 对当前策略允许的全部下一帧候选计入选择耗时后重新筛选；fixed不扩展候选，period_only不跨模型。
- 对重新筛选自身耗时再检查；若刚选出的组合因此失效，可继续筛选，次数最多为候选数加一。真正没有候选或当前截止时间耗尽仍中止。
- 预选不修改已提交选择/驻留计时；只有当前帧按时下发，才提交下一帧组合。保持当前模型、当前D、下一次已承诺释放点和门预算不变。
- 下发前分别记录停机标志和精确检查时刻。其他飞机触发停止且本帧未超时，记录`peer_abort`；本帧实际超时记录`late_output_not_applied`。两者同时发生仍计入真实超时，不隐藏失败。

旧Drone2消息把“全局停机”和“本帧超时”合并，旧日志没有精确守卫时刻，不能据此断定是哪一种。本轮不会追改旧报告。原失败目录7个文件SHA256核对未变，原状态仍为INCOMPLETE；模型、数据、规划器和截止时间配置均未修改。

回归覆盖：旧误中止场景先在旧代码上复现，再确认有合格候选时改选；同时检查门预算拒绝、当前截止时间不可延长、重新筛选耗时不遗漏、不重复计费、按时下发才提交、固定策略候选边界、其他飞机触发停止与真正超时，以及三机模拟动态释放完整性。

## 精确时序

在第k次计划释放t_k，当前模型m_k和T_k已经承诺：

- 本次请求只使用m_k，截止时间始终是t_k+T_k，D_k=T_k。
- 本次控制命令duration=T_k，下一次计划释放始终是t_(k+1)=t_k+T_k。
- 读取当前状态、构建237D输入，然后用当前共享队列快照和已完成历史决定下一次的(m_(k+1),T_(k+1))。
- 本次按时下发后才提交这个下一帧配置。下一帧截止时间为t_(k+1)+T_(k+1)。不会重新起算参考轨迹或重排已有释放点。
- 门附近状态有效期检查使用T_k+预计R_(k+1)，不是新选的T_(k+1)+预计R_(k+1)。否则缩短下一帧的T会错误地掩盖已经承诺的等待时间。

选择计算在本次推理提交前完成，实际耗时包含在sample_to_output/sample_to_dispatch中，并单独记录online_selection_ms。预测准入对全部允许候选加上观察截止时刻之后的实际选择尾部耗时，并检查重新筛选的耗时；原选择不合格但其他组合合格时，改选下一帧组合。没有合格选项才中止，绝不加长当前D。预测仍不是WCET。

## 策略和失败处理

默认adaptive同时考虑模型和下一帧T。只使用冻结的已试飞候选、路线质量先验、当前队列和已完成历史；无未来标签、未来响应或未来负载信息。

--policy fixed和--policy period_only保留给后续同场景对照。本轮不要求把三种都跑一遍。当前质量先验来自单条验证路线的活动段位置MSE，不是状态条件下的控制质量/安全界；N*C/T仍是同类任务近似，不是异构三机调度证明。相同源状态不能证明所有未来变化可控。

以下情况中止全部试飞，保留报告，并提交悬停清理：没有合格下一帧组合、状态已违背设定速率上界、选择过慢、共享推理出错、当前释放过期、DNN输出或异步下发超时，以及已有碰撞/飞行保护触发。不会隐藏切回MPC、应用已过期输出或放宽验收。悬停清理是仿真终止措施，不是认证安全恢复。

当前仍使用4m×4m门、0.25m余量、门前后2m监测范围和3m/s横向/垂直误差速率假设。复用state_budget_shadow_v1.json中的数值；其原scope字符串描述旧旁路审计，不代表在线入口仍只做旁路。门外只估计资源可行性，不能声称全路线安全。

## 用户下一步：只试飞一次

手动打开原UE4 Blocks项目并点击Play，确保包含Drone1/2/3；关闭训练和回放，不启动第二个AirSim窗口。若旧运行时门框仍在，先Stop再Play清空它们。启动编辑器仍可使用CMD：

```cmd
"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"
```

点击Play后，在PowerShell复制整段：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
& $trainPy -m controller_architecture_study.online_joint_flight --device cuda --policy adaptive
if ($LASTEXITCODE -ne 0) { Write-Warning '在线接管未通过；报告已保留，请返回末尾输出与报告路径，不要连续重复试飞。' }
```

沿用已试飞路线env48/route0，三机各五门，飞行约20秒，另有模型加载/预热时间。只在初始化reset并空中放置一次，无起飞动画；穿门时不重置、不瞬移。程序连接手动Play实例，不自动启动新窗口。门框默认运行后清理。

路线宽松时始终选择plain4x256/T50完全正常，不强制出现切换，也不能把这种结果解释为动态策略有收益。

## 输出与判定

输出自动保存到新目录：F:\FalconReproduction\airsim_control\controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_joint_online_<时间戳>，不覆盖旧结果。

- report.json：15次穿门、逐请求截止时间、参考活动段覆盖、sim/wall时钟比例和实际动态释放表核对。全部通过才acceptance_passed=true。
- requests.jsonl：本次模型/T/D、位置/速度、选择起止时刻、当前门预算、单策略候选判断、下一帧模型/T和是否按时提交；包含推理提交前的拒绝状态。`joint_online_initial_case`记录未计尾部耗时的原选择，`joint_online_admission`记录选择尾部、重新筛选次数与最终预测；`dispatch_guard_*`区分精确检查时刻的停机标志和真实超时。
- online_selection_report.json：在线策略、决策状态分布、选择耗时、实际下发模型/T分布、已提交下一帧选择数。不是飞行后的三策略建议回放。
- flight_trace.json与state_budget_report.json/state_budget_samples.json：实际轨迹、穿门与事后有效期核对。事后审计不改写截止时间或失败标志。

终端会打印current/next配置、online_selection_ms、model_switches、period_switches、policy_aborts、peer_aborts、schedule_accounting_passed以及三个验收标志。动态请求数量由实际选择决定；另逐帧核对连续帧号、下一帧必须使用上一帧提交的配置、释放间隔为上一帧T、D=T及完整飞行释放区间，不能靠“有多少请求就算计划多少请求”蒙混通过。

此次通过后再做约束活跃场景的固定/只调T/联合调模型-T对照，观察活动路线误差、完整响应、输出年龄和预算违例。当前还没有完成该收益实验，也没有证明超网必要，不需要为本轮重新训练DNN或重复容量扫描。
