# 期限归属、CPU建议服务与CPU/XPU分工修复

日期：2026-11-02。代码和离线检查已交付；真实AirSim复核由用户运行。本轮没有训练、连接AirSim或操作Unreal。

## 用户首轮执行后的热修复

用户运行`runtime_service_repair_20261102_v1`时，期限归属审计通过（重复过期future 96→41）、CPU/XPU六输入误差0、12机连接成功。前两次飞行完成，静态碰撞5、旧动态碰撞6；第三次新增模块试验因`TypeError: dict() got multiple values for keyword argument 'is_safety_certificate'`中止。

原因是新增CPU服务分支使用`dict(available=True, **analysis, **timing, ...)`，两个输入都含`is_safety_certificate`，Python禁止函数调用重复关键字。现已改为字典字面量展开，后续时间分析字段覆盖同名字段，与已有后台/复核路径一致，不改变安全判据或模型。

新增测试使用真实`GateHoldBatch`与`safe_deadline`返回值进入`tick`，修复前成功复现异常；修复后覆盖原生上下文接受/拒绝两种结果，137项相关检查通过。之前135项测试未覆盖这条完整字段合并路径，是本次检查遗漏；不能将其解释成用户环境或DNN问题。

v1原始报告不覆盖。重跑下方入口时，增加`-OutputRoot "$proj\controller_architecture_study\results\runtime_service_repair_20261102_v2"`，重新做同成本配对，避免把修复前后试飞拼成一组。

另一个独立警告：第三组初始前端利用率预测为98.16%，超过95%限制，包含3ms/10ms管理预算的最大30%保留量。这不是实测CPU利用率，也不是此次TypeError的原因。该资源可行性问题尚未解决，修复异常不代表全组期限或控制收益已经通过。

## 修改了什么

### 1. 已结束的旧控制义务不再被“临近发布保护”卡住

归档中的典型情况是：旧期限447.9ms，新控制439.8ms已确认，454.5ms处理下一条458.3ms释放的任务。旧输入保持义务已经结束，原实现却因只剩3.8ms、小于6ms规划保留时间，不允许更新其归属，随后按旧期限拒绝。

现在区分两件事：

- 更新已确认新控制对应的期限事实；
- 改变任务释放、T或c。

`prepare_recovery`允许对仍未发布的任务重绑新控制epoch，即使在配置修改保留窗口内；已经到期发布、处于执行阶段或控制正在下发的任务仍受保护。对于没有新ACK的旧控制，不放松原来的限制。

前端CPU服务每次最多分析两条控制epoch已变化的回路，使用已有原生日历/控制token的CAS安装**仅期限归属改变**的版本。它不改变模型、周期、释放或名义期限，不需要等待某次优化成功后才更新事实。竞态失败时Python计划保持原样；已发布任务、过去的miss和旧工作成本不被删除。

这类安装记录在`epoch_metadata_refreshes.json`，与配置建议安装分开统计。它不能被算成调频成功或安全认证。

### 2. 调整建议不再等前端空闲

旧调用位于主循环最后，连续出现输入、交付和释放时容易一直到不了。

新调用位于读取完成事件之后、选择下一项前台工作之前。按10ms间隔检查CPU管理服务，每次目标预算3ms；在预算内处理少量epoch事实，再处理一次结果复核或快照抓取。前端空闲等待也考虑下一服务时刻。

昂贵的联合方案生成仍在独立后台核，不挪进XPU。主CPU调度/输入/提交与模拟XPU继续并行。

记录`cpu_service.json`中的开始、结束、操作、服务迟到及预算超支。这里是Windows/Python合作式预算：已经进入的Python/NumPy调用不能在3ms强制中断。10ms不是已证明的硬响应上界，必须看用户运行后的实际服务间隔与尾延迟。

为避免把管理服务当作免费资源，增加了：

- 共享准入的前端利用率中计入3ms/10ms=30%的最大保留预算；不是说实测一直消耗30%。
- 有限流水线预测中加入后续CPU管理预算窗口；模拟XPU在此期间可以继续计算。
- 当前复核时间仍计入当次响应，不用延长物理期限换取通过。

这可能使部分历史配置不再通过当前预测，应如实保留。基于经验成本的有限预测仍不等于硬实时保证。

### 3. CPU决定下一计算段，模拟XPU只执行指定段

`framework_cpu_xpu.py`实现CPU协调进程和显式段执行协议：

```text
主CPU前端：状态、输入、配置、准入、控制下发
           │ 提交job
           ▼
CPU协调器：按EDF选择下一段（绑定同一host物理核）
           │ 执行指定job的一段
           ▼
模拟XPU：独立物理核、模型和激活常驻
           │ 段完成/最终4D结果
           └─────────────► CPU协调器
```

模拟XPU不自主选任务。CPU只有一个段命令在途，在段完成后才能切到别的任务；既定job的模型、deadline、输入和激活上下文保持一致，不进行段内抢占。

主前端与CPU协调器是两个进程，但绑在同一host核，避免主前端处理某个状态时XPU段完成通知也无人接收；它们的CPU竞争仍存在。后台策略规划另有核，IO/监测也有已有分工。当前并非对Zynq双PS核资源配置的逐项等价模拟，也没有操作系统独占核保证。

段协议使用小消息，激活保留在执行器中，不来回传整层张量。启动时每模型测量3次逐段CPU/XPU往返，在原有计算成本估计上加入往返估计。它包含IPC与模拟CPU执行，不能当作真实FPGA时延或WCET。

六次对照开始前另冻结一份新的往返成本估计，三组共同使用；不会用旧“执行器内部自己选段”的成本假装新通信免费。启动校准段不混入正式飞行段记录。

## 已执行的离线验证

- 135项相关检查通过，包含期限重绑、原生CAS、配对快照不变性、整组预测、协议和段边界调度。
- 单独真实多进程检查：2×128和9×1024各3个输入，共6个job；相对直接推理最大绝对误差0；CPU/XPU绑核不同，CPU命令数与正式执行段数一致。
- 冒烟报告：`results/offline_cpu_xpu_smoke_20261102_b/report.json`。
- PowerShell入口语法、Python CLI help通过。

没有执行真实时序回放、六次试飞或收益验证；不把离线PASS改写为动态减少碰撞。

## 用户运行步骤

AirSim未启动时，PowerShell运行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
Set-Location -LiteralPath $proj
$editor = 'E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe'
$uproject = 'F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
$settings = Join-Path $proj 'controller_architecture_study\configs\airsim_safety_12.json'
Start-Process -FilePath $editor -ArgumentList @($uproject, "-settings=$settings") -WindowStyle Normal
```

加载后点击Play，不最小化窗口，保持后台降CPU选项关闭。已有12机Play则跳过启动命令。

然后整段运行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$proj\controller_architecture_study\run_runtime_service_repair.ps1" -Python $trainPy
if ($LASTEXITCODE -ne 0) { throw '复核中断；结果已保留，请返回末尾报错和报告路径。' }
```

入口依次运行：归档期限重绑审计 → 无AirSim的CPU/XPU一致性检查 → 核对Drone1–Drone10 → 原静态/旧规则动态/修复后的建议辅助动态，共六次配对试飞。没有再次训练或执行64次边界补采。

结果根目录`results/runtime_service_repair_20261102_v1/`。已完成部分保留；完整六次结束后重跑不会覆盖，半途试飞失败则检查后选新的`-OutputRoot`。只想运行期限审计可加`-ReplayOnly`；这不是完整调度回放。

## 回来后先分析哪些指标

1. 归档中过期future和已完成旧义务误拒是否减少，历史失败是否仍保留。
2. `*_planner/cpu_service.json`的实际服务间隔、迟到、预算超支，不能只看平均搜索时间。
3. `epoch_metadata_refreshes.json`中事实更新是否成功；它不算T调整。
4. `installed_groups.json`、`requests.json`中是否在首次碰撞前实际交付了改变周期/释放的命令。
5. `cpu_segment_dispatches.json`中段命令、完成、CPU等待和切换；同时检查新增通信是否反而拉长实际命令间隔。
6. 六次碰撞、完整J缺失、名义与条件物理期限违反，全部保留。

这轮先检验实现修复，不把历史选定30ms配置称为新执行架构下的全局静态最优。通过后才重选公平静态配置、扩大种子和危险段错开实验。

仍固定每架飞机原DNN，未增加drop、重训或切换超网；没有无条件安全保证。后续硬件迁移必须重新测PL、DDR、上下文及网络成本。
