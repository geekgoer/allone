# 在线联合框架第二阶段：有界反馈、T-only 对照、三机切换机制

## 本次已完成与未完成

代码复用四个已训练完成的237D控制器，不改教师数据、数据划分、模型权重、归一化或参考轨迹。旧版`online_joint_pool_v1`结果完整保留。

已完成代码和轻量检查：

- 在线统计只读最近两秒的完成记录和当前pending数量，完成窗口最多4096条；不再复制、扫描不断增长的完整历史。历史仍写入请求日志。
- 固定策略不搜索其他候选；新增`period_only`，锁定静态最佳模型plain4x256，只允许改变T。
- 模型成本带更新时间。超过两秒的旧观测标为`stale_conservative`，使用旧观测与试飞先验较大者，不冒充当前实时测量。仍不是所有模型同时测量，也不是WCET保证。
- 正序/倒序比较fixed、period_only、adaptive，共两轮六次回放。各请求始终保持自己的D=T。
- 除各自期限下超时，新增统一50/75/100ms响应门槛统计，busy与错过释放都计入。仍应一起看输出年龄和请求数，不能仅比较miss计数。
- 分开记录`decision_ms`与`producer_pause_ms`：后者还包括写日志、终端打印的主线程停顿。统计有界不保证Windows不会出现其他长尾。
- 三机新增预设模型/T切换试飞入口，复用原来的路径、特征、RPC独立线程、共享推理服务和碰撞/过期中止机制。

用户已完成GPU新版回放和一次真实三机预设切换试飞；结果与统计修复见文末。修复后的调度器尚未重新试飞。仍未实现状态相关T_max、状态条件计算收益、在线联合策略对真实飞机的接管、早退或弹性超网。

本阶段首先区分“调整周期的收益”和“同时调整模型的额外收益”；随后验证切换执行层连续性。不要把预设切换当作在线状态感知调度。

本地检查：52项CPU/模拟接口测试通过；回放脚本和切换入口的`--preflight`均PASS，未开始GPU推理或连接AirSim。旧v1四次回放的报告/请求/决策文件哈希与原清单一致。新增测试覆盖有界反馈、历史日志不被在线扫描、成本过期标记、T-only模型锁定、恢复运行、逐请求期限、命令保持时间、一次初始化放置及RPC异常保留报告。实机/仿真性能仍需用户运行下述命令取得。

## 第一步：GPU三策略回放

先关闭AirSim/UE4和其他GPU训练。此命令不会训练、不会启动仿真。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_online_joint_v2.ps1 -Python $trainPy
if ($LASTEXITCODE -ne 0) { throw '回放中止；结果已保留，请检查终端与报告，不要删除已有结果。' }
```

测量时间为216秒，另加加载和预热。输出目录：

`F:\FalconReproduction\airsim_control\controller_architecture_study\results\dynamic_complexity_237d_v1\online_joint_pool_v2`

每策略的`attempt_*/report.json`、`requests.jsonl`、`decisions.jsonl`均保留，六次汇总为`summary.json`。中断后重新运行同一段命令：完成的跳过，中断的另建attempt。非中断异常先定位，不能无限重试。v1结果不覆盖、不合并；代码/配置更改后不得混用旧resume清单。

重点返回最终六行表格即可。判断顺序：

1. 选择/释放开销的长尾是否减少，而不是仅看均值。
2. period_only相对fixed，是否值得用更旧的输出换取更少的超时。
3. adaptive相对period_only，是否存在额外收益。没有收益也保留结果，不强迫切大模型。
4. 统一门槛与各自D=T下的统计是否给出不同结论。

这是推理请求回放，不代表32架无人机实飞，也不包含AirSim状态RPC、路径预处理和下发。

## 第二步：三机预设切换机制试飞

先启动手动UE4项目并点击Play；不要同时运行回放或训练。如果已有运行时门框，Stop/Play一次清理，不删除其他环境元素。程序只连接现有会话，不自动启动另一个AirSim。

启动编辑器的CMD命令：

```cmd
"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"
```

点击Play之后，PowerShell完整命令：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }

& $trainPy -m controller_architecture_study.joint_switch_flight --device cuda
if ($LASTEXITCODE -ne 0) { Write-Warning '切换试飞未通过，请保留报告；不要将穿门成功单独当作完整验收。' }
```

预设切换：

|全局参考时间|配置|
|---|---|
|0秒|plain4x256，T=D=50ms|
|3秒|plain4x512，T=D=75ms|
|6秒|plain4x512，T=D=100ms|
|9秒|plain4x256，T=D=50ms|

该计划可在`F:\FalconReproduction\airsim_control\controller_architecture_study\configs\joint_switch_mechanism_v1.json`调整；只接受原试飞中合格的模型/T组合，所有切换需在活动路线结束前发生。

约20秒，三机独立五门。仿真只在初始化时reset一次、放置三架飞机到空中，因此没有传统起飞动画。参考轨迹沿同一全局时钟前进；切换不重新放置飞机、不重启轨迹、不重新加载模型。每架飞机保留下一次既定释放时刻；后续间隔、命令duration和该次D一起按新T变化，旧请求期限不回改。

超时/碰撞等异常中止并尝试悬停清理，没有隐蔽MPC救援。已有飞行守卫不是形式化安全证书。

报告目录为`...\dynamic_complexity_237d_v1\airsim_joint_switch_<时间戳>`：

- `report.json`：穿门、实时性、模拟/主机时钟比例、活动路线时间加权位置RMSE、切换次数、有效命令样本年龄。
- `requests.jsonl`：每请求模型、T、D、命令duration、下发时刻、前后控制量变化。
- `flight_trace.json`：实测位置、参考位置、连续参考elapsed、模型和T。
- 报告中`model_switches`和`period_switches`按每架飞机累加，正常计划为6次模型切换、9次周期切换，不是全局计划的2和3。
- `switch_control_delta_max_abs_per_output`顺序为推力、wx、wy、wz，单位不同，不能把它们相加。它包含飞机状态变化，不是同一输入下的纯模型差异。

完整通过要求15次有效穿越、无超时、完整测量活动路线窗口、模拟/主机时间比在0.95～1.05。预设切换通过不等于在线联合策略已安全有效。

## 后续主线

若T-only已解释绝大部分资源收益，下一步优先补状态相关控制有效期和实际控制代价，不急着训练超网。若联合选型另有稳定收益，再接入真实负载反馈；最终以相同有效期和资源条件比较常驻模型池、固定宽度早退、弹性共享网络。不能仅凭发生过模型切换证明超网必要。

## 2026-10-04实测与修复进度

新版六次GPU回放已完成，均为零超时。adaptive与period_only都未切换模型；固定策略输出年龄约28.6～28.8ms，自适应/仅周期策略约34.1～35.8ms。目前没有证据表明联合选型优于固定小模型，也没有证明超网必要。该回放不是32架飞机完整实飞。

用户试飞目录：

`F:\FalconReproduction\airsim_control\controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_joint_switch_20261004T140628_043253Z`

原报告为flight_passed=True、realtime_passed=False。实际1051条请求均推理完成并下发，输出与下发期限均零超时，15次有效穿门、6次模型切换、9次周期切换，时钟比例和活动路线覆盖通过。误判来自两处统计缺陷：

1. 旧调度器累加绝对主机时间，而事后计划数量从零时钟重算；浮点端点导致Drone1多一条约20秒处的释放（351/350/350），计划重算只有1050条。
2. 命令年龄函数只识别client0等回放名称，没有匹配Drone1等试飞名称，错误报告100%未服务。

当前代码已将调度累加改为相对时间（未来该计划350/350/350），并显式传入试飞飞机名称。严格请求计数与逐请求D=T检查没有放宽；通用“不要DNN接管”提示改为具体的计时/调度统计失败提示。

58项CPU/模拟接口测试通过，覆盖原实测时钟起点、大时钟起点、漏请求、重复请求、错误配置、真下发超时、未覆盖活动路线和时钟比例失败。文件预检PASS，无GPU推理、无AirSim连接、无训练。

原report.json、requests.jsonl均保留，审计脚本逐条重建旧主机时钟调度、核查下发期限及配置，另存：

`...\airsim_joint_switch_20261004T140628_043253Z\audit_corrected_v1.json`

离线审计PASS；corrected_realtime_passed=True、corrected_acceptance_passed=True。原1050条计划与旧实际1051条分别保留，旧调度端点差异没有隐去；这不是修复后调度器的新飞行结果。原文件SHA256写入审计报告，未修改模型或数据。

修正后的命令输入年龄（按实际下发时刻、三机20秒窗口积分）：均值36.156ms、最大125.816ms，初始尚未收到首条命令的比例0.123%。活动轨迹位置RMSE为0.3594m。最大年龄包含等待下一次控制更新，不等于一次推理响应时间；零响应超时也不等于状态有效期始终满足。

下一实现阶段：接入明确的状态有效期约束，先以影子方式记录状态相关预算、候选T及响应/命令年龄是否可行，暂不直接让未经验证的边界自动接管。状态边界的物理假设、参数和适用范围应先确定，不能将经验预算冒充安全证书。之后比较固定策略、仅T策略和联合模型/T策略的真实控制收益；超网仍是候选实现，不是预先保证优于模型池的结论。
