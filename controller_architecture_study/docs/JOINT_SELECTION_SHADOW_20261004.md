# 模型/T联合选择：已实现，当前仅旁路建议

## 本轮完成

复用既有资源估计，增加状态有效期筛选，不改变原资源回放选择器及旧结果。R估计包括状态读取/预处理、共享队列、模型推理及异步下发，不只计GPU内核。

每个状态比较fixed、period_only、adaptive：固定策略只检查原静态组合；period_only不换模型；adaptive检查全部已试飞组合。当前修正版在原三机预设切换试飞中只记录位置、速度和时间戳，**飞行结束后**重建当时队列并生成建议，不再在控制线程里计算三种策略。**不应用建议**，实际DNN、T、D、命令duration和连续参考时钟仍遵循原预设计划。

修正版CPU/模拟接口测试80项通过，包含故意拖到60ms的慢分析、采样时间因果性、中止请求保留，以及历史事件不能覆盖当前请求的关联回归检查。真实候选文件的预检PASS，gpu_inference_started=false、airsim_started=false，没有训练或修改权重、教师数据。

## 2026-10-05最新结果

用户已完成移出同步分析后的三机试飞：`airsim_joint_switch_20261004T160131_391468Z`，原飞行、实时性和综合验收均通过，1050次请求全部及时下发，15次穿门通过，不需要重复试飞。

随后发现飞行后回放的历史事件变量覆盖了当前请求，导致该目录原`joint_shadow_report.json`及`joint_shadow_decisions.jsonl`的策略结论失效。这不影响实际飞行报告。关联错误现已修复，新分析单独保存在`.../dynamic_complexity_237d_v1/joint_selection_alignment_fix_20261005_v1`，1050条逐帧关联全部匹配。原始8个文件的SHA256前后一致，旧分析仅保留作审计，不再用于策略比较。

修正后三种策略全部建议plain4x256/T50，无建议切换，无预算或资源耗尽。完整指标、重算命令与未完成范围见[本阶段收尾报告](F:/FalconReproduction/airsim_control/controller_architecture_study/docs/REPLAY_ALIGNMENT_CLOSEOUT_20261005.md)。以下同日早期记录与试飞命令保留作历史说明，不是再次试飞要求。

选择器：`controller_architecture_study/state_joint_policy.py`。旁路/回放：`controller_architecture_study/joint_selection_shadow.py`。物理假设仍在`configs/state_budget_shadow_v1.json`。

## 逻辑和边界

仅考虑已有12个实际试飞组合：4个模型各配50/75/100ms。质量排序使用冻结的活动路线时间加权位置MSE，不把离线DNN控制MSE作为状态安全界。

逐候选检查：

- 资源：`N*C_est/T <= 0.75`，且`release_jitter_est + sample_to_dispatch_R_est <= T`。
- 状态：`T + release_jitter_est + sample_to_dispatch_R_est <= validity_budget`。T是名义释放间隔，另给下一次释放抖动留预算。
- 时序：当前决策还未超过候选对应的下一释放点。
- 合格候选中选冻结路线代价较小者；保留2秒驻留和1%质量增益门槛，但驻留不会阻止立即离开不合格组合。

R_est由已验证试飞的生产者/下发P95、已完成样本响应、当前队列和模型服务成本组成；当前已消耗的预处理时间可提高预算。只用已完成历史响应，不用当前/未来标签。默认30ms响应储备、5ms余量和门附近两轴3m/s上界仍是原型假设，不是WCET/动力学证书。

|状态|含义|建议|
|---|---|---|
|AVAILABLE|门附近资源/状态估计均允许候选|有|
|STATE_NOT_APPLICABLE|远离门，仅估计资源|可有，不能算状态安全通过|
|BUDGET_EXHAUSTED|资源允许一些候选，但状态预算容不下|无|
|RESOURCE_EXHAUSTED|没有候选满足资源/下一释放时序估计|无|
|ASSUMPTION_INVALID|采样状态已违背设定速率上界|无|

无候选时返回recommendation=null，不沿用旧选择、不强制best-effort、不隐藏MPC救援。旁路模式不因此改变实际控制；接管时的停止/恢复措施尚未实现。

N*C/T是同类任务近似，不是异构多机调度证明。门外没有全路线状态约束，执行器真实生效时刻也未测量。P95、MSE和采样速度不是全局安全保证。

## 已有数据回放

源目录：`.../dynamic_complexity_237d_v1/airsim_joint_switch_20261004T140628_043253Z`。

输出：`.../dynamic_complexity_237d_v1/joint_selection_shadow_20261004T152210_033584Z/joint_shadow_report.json`。

1051次状态中172次在门附近。三策略都建议plain4x256/T50，各有172次AVAILABLE和879次STATE_NOT_APPLICABLE，无建议切换或预算/资源耗尽。

回放中三策略合计的每状态选择计算平均约1.071ms、P95约1.519ms、主机单次长尾约24.604ms，不能只看均值。回放没有AirSim/GPU争用，不代表实时试飞的开销。

源报告/请求/轨迹SHA256未变。源旧acceptance_passed=false保留，另存旧统计修复审计不被覆盖。新COMPLETE表示分析完成，不代表重做了一次飞行验收。

各策略看到同一实际基线队列，尚未模拟改变T/模型后的反事实队列或飞行。因此目前只能说**当前状态/负载下无必须切换的证据**，不能报告自适应改善了轨迹或降低超时。

## 153232旧试飞失败和修复范围

旧报告目录：`.../dynamic_complexity_237d_v1/airsim_joint_switch_20261004T153232_782052Z`，不改写原始报告。

Drone3第35帧晚启动7.94ms，读取状态42.22ms，读完已超过50ms；旧旁路再花5.83ms，入队时已经过65.97ms。Drone1第36帧晚启动32.53ms，开始旁路时已经过44.62ms，旁路花5.84ms，入队时变成50.46ms。DNN推理本身分别只有0.77ms和0.35ms。读取状态区间可能包含RPC等待和主机调度，不能仅凭该记录断言显卡驱动故障。

旧实现的“建议不应用”不能保证“分析不干扰控制”：三策略计算同步发生在推理提交之前。这部分已移除，实时阶段不调用选择器、门预算计算或资源反馈快照，也不启动并行分析线程。

实时仅复制位置/速度和单调时钟时间到当前请求，计入`capture_ms`。控制线程和推理服务全部结束后，按记录时间顺序回放入队、完成及实际下发事件。在某个状态的观察时刻，只允许使用该时刻之前发生的事件；晚到且未下发的请求仍影响过去的排队估计，但不进入已下发响应历史。新捕获记录包括中止帧，不把中止帧伪造成成功控制。

此修改只移除同步分析开销，不保证Windows/RPC长尾消失；不放宽T、D或验收，不改模型/规划器，不增加MPC兜底。离线`decision_ms`不能作为上线选择器的实时开销或收益；真正在线选型仍未接管。

旧失败报告兼容性回放也已完成：`.../dynamic_complexity_237d_v1/joint_selection_shadow_20261004T155503_823586Z/joint_shadow_report.json`，覆盖108个已存轨迹状态，status=SOURCE_INCOMPLETE、original_source_acceptance_passed=false。旧源文件SHA256未变；这只是回放兼容检查，不把旧飞行改判为成功，也不证明改变建议会改善控制。

## 修正版试飞命令（已执行，保留作历史说明）

先手动打开原UE4项目并点击Play，不要同时训练/回放或启动第二份AirSim。旧运行时门框可Stop/Play清空；程序不删除其他环境元素，不自动启动新窗口。

需要启动编辑器时，CMD命令：

```cmd
"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"
```

点击Play后，整段PowerShell复制执行；飞行约20秒，另有加载/预热时间：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
& $trainPy -m controller_architecture_study.joint_switch_flight --device cuda --joint-selection-shadow
if ($LASTEXITCODE -ne 0) { Write-Warning '试飞未通过，请保留终端输出和报告' }
```

飞机仍按0/3/6/9秒原计划切换模型/T，三机各飞五门。建议一直T50也不会改变画面中的预设75/100ms阶段。初始化只reset/空中放置一次，无传统起飞动画；穿门不重置轨迹。

新目录：`.../dynamic_complexity_237d_v1/airsim_joint_switch_<时间戳>`。

- report.json：原穿门、逐请求截止时间、时钟比例和活动路线覆盖验收，条件不放宽；增加joint_selection_shadow入口。
- requests.jsonl：joint_shadow_state记录位置/速度，joint_shadow_capture_start_s/snapshot_s记录实时快照耗时和观察截止时刻；不再有在线joint_shadow_start_s/ready_s。
- joint_shadow_report.json：三策略状态/建议分布、建议切换数；execution_mode=post_flight_causal_replay、recommendations_applied=false。capture_ms是实时复制开销，decision_ms是飞行后的分析CPU耗时，二者不混用。
- joint_shadow_decisions.jsonl：各候选资源/状态可行性、估计响应/T上界和建议。
- state_budget_report.json/state_budget_samples.json：PPT预算旁路核对。

只需返回终端最后的flight_passed/realtime_passed/acceptance_passed和新报告路径，不必粘全部状态。终端同时打印release_jitter_ms与release_to_output_ms的均值、P99和最大值，便于识别少数长尾；sample_to_output_ms不包含线程晚启动。异常/未完成结果会保存，建议不会被当作实际控制输出。旧版本没有中止帧状态快照，回放只能覆盖其已存轨迹状态。

## 旧回放重现（已执行，不必重复）

无需Play或GPU推理：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -m controller_architecture_study.joint_selection_shadow --report '.\controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_joint_switch_20261004T140628_043253Z\report.json'
if ($LASTEXITCODE -ne 0) { throw '联合选型回放失败，请查看输出' }
```

## 尚未完成

2026-10-05后续：精简在线选择器及下一帧实际接管代码现已实现，91项CPU/模拟接口测试与真实文件预检通过；真实在线试飞待用户运行。只需一次命令，见[在线接管说明](F:/FalconReproduction/airsim_control/controller_architecture_study/docs/ONLINE_JOINT_TAKEOVER_20261005.md)。下面“实际接管未完成”指尚未完成真实飞行验收及收益对照，不再表示没有代码。

移出同步分析后的新试飞和逐帧关联修复后的回放已完成；在线选择器开销和建议实际接管、预算耗尽后的控制措施、较紧物理边界/真实高共享负载的收益对照、超网/早退/独立模型池公平比较尚未完成。

联合选型核心、实时快照和飞行后因果回放接口已经搭好，不需要重训18个模型或新GPU容量扫描。下一阶段是精简在线选择器、实现实际接管并在约束活跃场景比较收益；不在宽松路线无限重复验证，不为支持超网而强行制造模型切换。
