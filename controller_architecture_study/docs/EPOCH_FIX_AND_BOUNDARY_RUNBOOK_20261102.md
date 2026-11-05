# 期限归属修复与64次边界补采：用户运行说明

> 用户已完成这一轮运行，结果见[本轮分析](EPOCH_BOUNDARY_RESULTS_20261102.md)。以下保留原操作说明，不代表新方案已经取得收益。

2026-11-02。RT-01、DATA-01的代码与必要离线检查已完成，真实归档回放、试飞和补采由用户执行。没有重训或修改控制DNN，没有更改量化包，也没有提交/推送Git。

## 本轮修正的含义

原先只要同一无人机还有采样/推理未完成，就阻止更新后续计划的期限，即使旧保持控制已经被新命令确认替换。这会让未发布的后续任务继续携带旧期限。

现在仅在这些条件下为未发布计划绑定新的保持控制期限：

- 有最新可配对状态及已确认控制，确认时间比该计划记录的旧保持控制更新；
- 尚有足够发布/复核时间，状态时间不晚于分析时刻且不早于该控制确认；
- 没有命令RPC在途或无法识别的在途阶段。

排队采样/推理仍完整计入共享时序预测；已发布任务的ID、期限和名义deadline不改写。新确认发生在旧期限之后时，历史违反仍记录。相同控制的窗口过期不能冒充新控制版本；更旧控制的证据不能覆盖新计划。

实际安装仍需最新状态重建、全组预测和C++原子检查。原生层会拒绝日历改变、控制版本改变或命令仍在途的提交。本轮没有改C++，不需要重编DLL。

这是对保持控制义务归属的局部修正，**不是取消准入，也不是将既往或当前未满足的要求变成安全**。已发布旧任务仍可能导致预测拒绝，不能保证安装率必然提高。命令确认仍是RPC代理，当前模型假设未认证，因此没有新增硬实时/无条件物理安全承诺。

## 三个顺序阶段

1. `replay_obligation_epochs.py`：用修正前冻结源码与修正后函数，在相同归档输入及同一代理时刻比较期限绑定。输出只代表回放不变量，不做完整调度或物理试验。
2. `run_reviewed_rate_trial.py`：同一历史静态、旧动态、新建议辅助，各两次自然十机试飞，顺序正反配对。沿用已冻结小评分器，不先训练新的策略；各机模型、资源和分段EDF保持一致。记录安装版本与真实交付。
3. `refine_recovery_boundary.py`：64次新增有效配对试验，不重新执行原120次。发现路线3400/3402借用旧0.4/0.6锚点，各最多细化4个状态；新确认路线3600/3601各测固定0.425/0.475/0.525/0.575四点。每状态仍为2×128/T20、2×128/T80、3×256/T20、9×1024/T20，首次等待10ms，恢复窗口2秒。

补采只改变预先定义的初始横向偏差/外向速度，不移动门，不添加飞行中扰动，不修改成功判据；确认标签不参与发现集的查询选择。源锚点不重复计入新增64次。所有有效失败保留，基础设施无效记录与重试分开。离散时间点通过不等于整个延迟区间得到保证。

## 启动AirSim

如果已有Unreal，请先确认是否是12机配置；若是且已Play，可跳过启动。若是旧三机配置，正常保存并关闭后再开，避免两个模拟器争用RPC端口。

在PowerShell执行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
Set-Location -LiteralPath $proj
$editor = 'E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe'
$uproject = 'F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$settings = Join-Path $proj 'controller_architecture_study\configs\airsim_safety_12.json'
Start-Process -FilePath $editor -ArgumentList @($uproject, "-settings=$settings") -WindowStyle Normal
```

加载后**手动点击Play**，保持窗口展开、未最小化，并关闭“处于后台时降低CPU使用量”。三机配置不满足本轮要求；入口会核对Drone1–Drone10。不要同时训练或运行另一个控制脚本。

## 一段运行命令

Play后执行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_epoch_boundary_stage.ps1 -Python $trainPy
if ($LASTEXITCODE -ne 0) { throw '本轮未完成；保留末尾错误与新目录，勿删除旧结果。' }
```

入口默认输出：`controller_architecture_study/results/epoch_boundary_20261102_v1/`。需要原20261101阶段的完整本地结果、冻结源码、模型与DLL；仅Git中的小摘要不足以执行全部阶段。

中断处理：离线回放可重算；补采可在同一源码/计划下续跑，有效失败不重试，基础设施尝试保留；六次配对试飞若中断则不从中间拼接，保留旧目录，核对原因后用新的`-OutputRoot`重新完整配对。

若只想先运行不需要AirSim的真实归档回放：

```powershell
Set-Location -LiteralPath 'F:\FalconReproduction\airsim_control'
& 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe' -m controller_architecture_study.replay_obligation_epochs --source .\controller_architecture_study\results\boundary_takeover_20261101_v1\application_rates_only --output .\controller_architecture_study\results\epoch_boundary_20261102_v1\replay
```

## 返回哪些结果

- `replay/report.json`：新旧未发布期限、延期原因、历史记录及绑定不变量。`REPLAY_INVARIANTS_PASS`不等于试飞安全。
- `application/summary.json`：六次真实试飞的碰撞、J、安装组数、带标记的真实交付及周期改变。
- `application/*_planner/report.json`与`installed_groups.json`：建议在哪个阶段被拒绝，是否实际安装。
- `boundary_refinement/report.json`：64次有效配对的资格/碰撞差异。没有差异也属于结果。

终端显示命令完成不意味着新方案更优；需要同时检查实际执行标记、控制/状态窗口、超期和物理结果。新一轮的原始命令期限越期统计仍保留，不应直接解释成互不重复的物理保持控制违反。

## 已做的必要检查

72项相关Python检查通过，覆盖旧/新控制归属、命令在途拒绝、未发布/已发布隔离、历史超期保留、过期计算的资源阻塞、全组准入及补采确认集隔离；另有2项编译DLL的mock检查通过，验证日历/控制版本变化会被原子守卫拒绝。合计74项。

这些是离线检查，没有连接AirSim试飞。真实归档对照与新阶段收益仍待用户运行，不将合成测试结果冒充原137个快照已经修好。
