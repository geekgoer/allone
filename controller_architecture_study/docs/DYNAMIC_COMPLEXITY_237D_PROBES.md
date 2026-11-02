# 237D 动态复杂度：先验证收益与预算瓶颈

日期：2026-10-04。此阶段不训练超网，不修改教师、数据划分、归一化或已训练模型。

## 1. 当前实现与边界

默认比较四个已完成的模型：普通 4×256、普通 4×512、普通 6×1024、残差 21×1024。
模型别名依次为 `plain4x256`、`plain4x512`、`plain6x1024`、`residual21x1024`。
残差 33×1024 现已完成，可用 `residual33x1024` 显式加入；旧四模型默认实验不变。代码在加载前检查 completion、237D 输入合同、数据来源、归一化及文件哈希。

最新结果、论文核对与 1～16 路容量扫描命令见：[2026-10-04 容量报告](F:/FalconReproduction/airsim_control/controller_architecture_study/docs/DYNAMIC_CAPACITY_REVIEW_20261004.md)。已有 12 轮 CUDA 回放完成，但原三机影子测试只通过穿门，没有通过 50 ms 实时性验收；不要直接进入下文的 DNN 接管步骤。

新增入口：

- `probe_state_value`：四模型逐状态对齐；只读 Validation 的 9600 条数据，不读取 Test。
- `probe_shared_response`：支持 1～64 个独立周期请求源，共享同一个常驻模型池和 FCFS 单推理工作线程。不启动 AirSim，不等同于试飞。
- `probe_capacity`：串行扫描请求源数量、模型与释放相位，逐轮保存结果；默认 28 个 15 秒试验。
- `probe_airsim_three`：手动 Play 下三机实测；保留完全相同的 237D 编码、已保存 C2 参考轨迹和控制量转换。
- `run_dynamic_probe_offline.ps1`：CPU 对齐或 GPU 12 轮回放的串行命令入口；不启动训练。

所有输出单独保存在：

`F:\FalconReproduction\airsim_control\controller_architecture_study\results\dynamic_complexity_237d_v1`

每次运行创建带 UTC 时间戳的新目录，不覆盖前次结果。原模型目录不写入新报告。

## 2. 先运行逐状态分析（不需要 AirSim）

可以在当前训练继续期间用 CPU 运行；它仍占少量 CPU、内存和磁盘带宽，但不会启动 CUDA 计算。
以下整段复制到 PowerShell。无需改变执行策略，只对本次 PowerShell 进程放行这个本地脚本。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw '训练 Python 路径不存在' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_dynamic_probe_offline.ps1 -Python $trainPy
if ($LASTEXITCODE -ne 0) { throw '分析失败，先查看错误，不继续后续实验' }
```

读取 `alignment_时间戳` 目录：

- `provenance.json`：模型、归一化和 Validation 文件哈希；不复制或改动原检查点。
- `aligned_predictions.npz`：`U_predicted[N,4个模型,4个输出]`、`U_teacher[N,4]`、逐状态误差及原 row/environment/route/family ID。
- `per_state.csv`：同一状态上四模型的标准化 MSE。
- `report.json`：全局/分状态族/速度/位置误差区间、成对改善、最差 5% 状态改善、按 environment 聚类 bootstrap 的探索性区间。

误差默认用未裁剪的输出，与训练 MSE 定义一致；在线控制仍沿用现有物理限幅。
`mean_error_reduction > 0` 表示每一对中的第二个模型更好。标准化 MSE 不能代替安全证明。
`label_aware_oracle` 知道所有模型和教师的答案，只是潜在收益上界，不是可部署选择器；默认尚未计入时间成本。
Validation 已用于选择 checkpoint；此处继续使用属于探索性分析。只有 6 个 Validation 环境，聚类置信区间精度有限。
`--limit 16` 仅用于代码冒烟，不能用其结果推断模型性能。

## 3. 训练全部停止后，运行 GPU 请求回放

关闭训练及 AirSim，保证这一组是无渲染负载基线。代码不会替你终止任何进程；也不把不可见训练干扰自动认定为已排除。
如果以前发生过显卡驱动重置，先运行单次 10 秒小试验，确认环境稳定后再跑完整套件；本工具不进行训练。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -m controller_architecture_study.probe_shared_response --device cuda --clients 3 --model residual21x1024 --duration 10
if ($LASTEXITCODE -ne 0) { throw 'GPU 回放失败，停止' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_dynamic_probe_offline.ps1 -Python $trainPy -RunGpuTiming
if ($LASTEXITCODE -ne 0) { throw 'GPU 计时失败，已完成的报告仍保留' }
```

完整套件 4 种条件，每种 3 次，每次 60 秒（约 12 分钟 + 加载/预热时间）：

1. 单请求源，固定最大候选模型。
2. 三请求源，同步释放，固定最大候选模型。
3. 三请求源，错峰释放，固定最大候选模型。
4. 三请求源，确定性轮换四模型，用于测量常驻池中的服务时间。

所有条件四模型都常驻；轮换顺序、输入序列固定；各轮改变实验执行顺序，减少热状态/顺序偏差。
`cycle` 是调度微基准，不是学会了状态感知的选择策略。
每架无人机最多一个未完成请求；忙时的丢弃和来不及执行的计划采样均单独记录、计入输出期限未满足统计，不静默排出无限旧状态队列。

可选：将一个包含全部模型的 `cycle` 回放报告传入离线分析，用实测平均服务时间构建“已知正确答案”的成本—误差上界。
不含排队，不是动态调度仿真，不是安全选择器；不能选择最好的一轮隐藏其他结果。

```powershell
$root = Join-Path $proj 'controller_architecture_study\results\dynamic_complexity_237d_v1'
$cycleReport = Get-ChildItem -LiteralPath $root -Directory -Filter 'replay_*' | Sort-Object Name -Descending | ForEach-Object {
    $p = Join-Path $_.FullName 'report.json'
    if (Test-Path -LiteralPath $p) {
        $r = Get-Content -Raw -LiteralPath $p | ConvertFrom-Json
        if ($r.device -eq 'cuda' -and $r.request_selection -eq 'cycle' -and $r.status -eq 'COMPLETE') { $p }
    }
} | Select-Object -First 1
if (-not $cycleReport) { throw '没有完整 CUDA cycle 回放报告' }
& $trainPy -m controller_architecture_study.probe_state_value --device cpu --timing-report $cycleReport
```

## 4. AirSim 三机实测

先手动打开 Blocks 项目并点击 Play。设置中必须有 Drone1/2/3，ClockSpeed=1；使用清理过障碍物的 FalconEmpty 地图。
若先前保留了旧门，先 Stop 再 Play。工具不删除其他实验生成的门，发现旧门会要求清理，以免撞到不属于本次路线的门。

此入口不会自动启动另一个 UE 实例。每次试验会 reset 一次、暂停放置三架无人机到空中起点、生成 15 个独立门，然后持续运行。
因此开头的空中定位不是起飞动画；测量期间不再瞬移、不在门前重新启动轨迹时钟，不暂停仿真等待推理。
路线按整条路线的横向包络自动隔开，门、轨迹、机体使用一致的 local-to-world 平移。
全部机体结束后悬停，默认移除本工具生成的门；不改地图资产、settings.json 或原数据。

先运行影子模式，MPC 实际控制；GPU DNN 只接收同一状态并推理。影子计算仍可能竞争 CPU/GPU 资源，因此记录 MPC 下发延迟，不能说它必然不影响 MPC。
下面整段只跑两次影子试验：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
foreach ($phase in @('sync', 'staggered')) {
    & $trainPy -m controller_architecture_study.probe_airsim_three --mode shadow --model residual21x1024 --device cuda --phase $phase --environment-id 48 --route-index 0
    if ($LASTEXITCODE -ne 0) { throw '三机影子测试失败，查看刚输出的 report.json，不继续 DNN 控制' }
}
```

只有影子模式 `acceptance_passed=True`（穿门和实时性同时通过）后，再显式进行真实 DNN 控制。当前已有影子结果不满足这个前提，先按最新容量报告诊断。以下四次试飞用于同一路线的初步筛查，不是最终统计验收。四模型始终常驻，单次三机使用同一个选定模型；尚未做动态模型切换。

```powershell
foreach ($model in @('plain4x256', 'plain4x512', 'plain6x1024', 'residual21x1024')) {
    & $trainPy -m controller_architecture_study.probe_airsim_three --mode dnn --model $model --device cuda --phase sync --environment-id 48 --route-index 0
    if ($LASTEXITCODE -ne 0) { throw "模型 $model 的三机试验未通过，保留报告，先诊断再继续" }
}
```

DNN 模式没有 MPC 悄悄接管；超时、碰撞、过大偏离或速度越界会停止整轮并请求悬停，不把过期 DNN 输出用于飞行。
新的 RPC 异步命令可能在客户端调用返回时尚未物理执行，报告明确将下发时间作为软件代理指标，不称为真实机体执行响应时间。

## 5. 如何读时间报告

`replay_*`、`airsim_shadow_*`、`airsim_dnn_*` 都保存 `report.json` 和逐请求 `requests.jsonl`。
AirSim 额外保存 `setup.json`、`flight_trace.json`。

- `release_jitter_ms`：实际请求采样开始减计划释放时间。
- `queue_ms`：服务开始减入队时间。
- `service_ms`：归一化、H2D、前向、D2H、反归一化，CPU 输出可用为止。
- `gpu_compute_ms`：CUDA events 覆盖前向区间，可能包含设备竞争/主机发射间隙；不等于纯内核活跃时间，也不包含整个软件链路。
- `sample_to_output_ms`：采样 RPC 请求开始到模型结果可用。
- `release_to_output_ms`：计划释放到模型结果可用；包含释放抖动。
- `sample_to_dnn_dispatch_ms`：仅 DNN 实控有值；影子模式为 null，不能冒充实际 DNN 闭环。
- `sample_to_mpc_dispatch_ms`：仅影子模式实际 MPC 命令的下发延迟。
- `output_deadline_misses`：结果晚于计划释放 + deadline，或请求忙丢弃/跳过/失败。
- `control_deadline_aborts`：DNN 消费端已到期，未使用该输出而中止。
- `planned_releases` 与 `released`：中止试验会不同；INCOMPLETE 不能作为完整成功试验。
- `flight.*.sim_seconds_per_host_second`：核查仿真时间与墙钟时间；明显偏离 1 时先检查 ClockSpeed、渲染和系统负载。
- schema v2 另记录 `state_rpc_ms`、`collision_rpc_ms`、`preprocess_wait_ms`、`reference_ms`、`feature_ms`、`sample_to_enqueue_ms`、`mpc_solve_ms`、`command_submit_ms`；包括取样成功但推理请求被忙丢弃的周期。
- `flight_passed` 与 `realtime_passed` 分开，全部满足才 `acceptance_passed=True`。穿门通过而实时性失败退出码为 2，不再作为通过继续执行。

`--preprocess-policy parallel` 默认保留原有预处理方式；`serialized` 仅为显式对照候选，不能假定已经解决实时性。新版本删除了解除暂停后 200 ms 无测量控制的等待，实飞应记录版本差异。

默认周期 50ms、输出期限 50ms，只是本轮实验配置，不等于理论安全的 T_max。
单工作线程 FCFS 是明示的资源共享基线，不代表最优 GPU 调度或真实 DPU 的执行架构。
请求回放不含 AirSim；影子模式不等于 DNN 实控；GPU 时延不能用于替代 FPGA/DPU 时延。

## 6. 判定与下一步

1. 若小模型在重要状态和闭环指标上都足够好，不继续强行做超网。
2. 若大模型仅在部分状态值得计算，而三机预算很宽裕，先研究状态感知的模型池选择器。
3. 若排队/抖动实际侵占控制余量，再研究推理中可调整计算。
4. 只有多出口网络不能达到同等控制—时间折中，而弹性宽度有重复收益时，才升级到弹性超网。

尚未实现/宣称：实际状态选择器、动态 T 策略、多出口/弹性训练、受控干扰实验、控制有效期证明、DPU 测时。
本阶段采集的是第一轮证据。不可用六环境探索性 bootstrap、16 条冒烟数据或少量正常穿门来宣称上述研究假设已经成立。
