# 237D 恢复模型：多 seed 与物理扰动闭环验证

日期：2026-10-03。代码已交付；正式补训及真实 AirSim 扰动试飞由用户执行。

## 本轮目的与边界

当前 seed=0 的 `features_recovery` 四个宽度在固定 Validation 上呈下降趋势，但单 seed 不足以说明趋势稳定，也不能用普通穿门成功替代扰动恢复验证。

本轮不重新生成数据、不重新拆分环境、不修改既有模型、不使用 Test 选择模型。仅对 `features_recovery` 补 seed=1、2：隐藏层数均为 4，宽度为 128/256/512/1024，总共新增八次训练。seed=0 原样复用。

沿用既有 153600 条 Train、9600 条 Validation、冻结归一化、学习率及训练配方。`--epochs 400` 是原数据组的预算口径，恢复组实际每模型训练 **200 epoch、60000 次优化器更新**；不要改成 `--epochs 200`，否则与已完成 seed=0 配置不一致。串行子进程训练，每 epoch 保存；已完成模型跳过，未完成模型自动恢复。

多 seed 汇总保留每个 seed 的指标，并报告均值、样本标准差与同 seed 的相邻宽度差值。不是从三个 seed 中挑一个最佳结果，不保证 loss 必然单调。三个 seed 仍是有限证据，标准差不是置信区间。固定原 Train、独立新状态探针、Validation 的定义保持不变。

## 一整段补训命令

无需启动 AirSim。建议训练时关闭 UE4，以减少与训练竞争的 GPU 负载。整段复制到 PowerShell；中断后仍重跑同一段。

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $proj = 'F:\FalconReproduction\airsim_control'
    $trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
    Set-Location -LiteralPath $proj
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $trainPy)) { throw "找不到训练环境：$trainPy" }
    & $trainPy -m controller_architecture_study.run_convex_recovery --arm features_recovery --seeds 0 1 2 --epochs 400 --device cuda
    if ($LASTEXITCODE -ne 0) { throw '补训停止；检查错误后重跑本段，已保存模型及 epoch 会保留。' }
    $reportPath = '.\controller_architecture_study\results\decision_convex_recovery_v1\runs\comparison_features_recovery_seeds0_1_2.json'
    $result = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $result.models | Select-Object model,status,epochs,best_epoch,fixed_train_mse,fresh_train_mse,validation_mse,gap_absolute | Format-Table -AutoSize
    $result.by_architecture | ForEach-Object {
        [pscustomobject]@{
            width = $_.width
            seeds = $_.seeds_completed
            complete = $_.panel_complete
            val_mean = '{0:E5}' -f $_.validation_mse_mean
            val_std = '{0:E5}' -f $_.validation_mse_std
            fresh_train_mean = '{0:E5}' -f $_.fresh_train_mse_mean
            gap_mean = '{0:E5}' -f $_.gap_absolute_mean
        }
    } | Format-Table -AutoSize
    Write-Host "多 seed 报告：$proj\controller_architecture_study\results\decision_convex_recovery_v1\runs\comparison_features_recovery_seeds0_1_2.json"
}
```

模型目录仍为 `controller_architecture_study/results/decision_convex_recovery_v1/runs/features_recovery/plain_mlp_h4_w<width>_seed<seed>/`，每个模型独立保存 `best_model.pt`、`last_model.pt`、配置、日志、离线指标、固定状态审计和完成标记。新汇总 JSON/CSV 不覆盖上一轮 `runs/comparison.json`。原数据及其被哈希的教师/特征实现未改动。

## 物理扰动测试定义

这不是给 DNN 输入加假噪声，而是用 AirSim `simSetKinematics` 改变模拟飞机实际位置、速度和姿态，随后读回确认。它是人工状态跳变测试，不应称为真实风场或外力实验。

四个固定工况来自 **Validation** 路线：environment=48/route=3 和 environment=51/route=6，各做左/右组合扰动。它们是开发验证，不是未使用 Test 的最终独立验收。

轨迹时间第 3 秒（50ms 控制周期的第 60 步）注入一次：

- 航向坐标下位置偏移 `[0, ±0.6, ∓0.3] m`；第三轴是 NED 向下。
- 速度增量 `[-0.8, ±0.8, ∓0.3] m/s`。
- 机体姿态旋转向量 `[±0.12, -0.12, ±0.12] rad`，用四元数相乘，不是直接相减。

航向由冻结路线给出，不混用无人机局部原点和世界门坐标。短暂暂停用于提交并读回状态，恢复此前暂停状态；轨迹和全局参考时钟不重新规划、不归零。保留碰撞、位置界限和原有控制输出限幅；不切回 MPC。

每个工况先跑一次 QP 教师，然后用完全相同的已保存轨迹测试四个 DNN。教师未通过的工况对 DNN 标为 SKIP，并保留教师失败报告，不假装该工况证明了 DNN 稳定或失败。RPC 连接或状态读回失败停止队列，需要修复后续跑。

本次恢复判据预先固定为：扰动后 3 秒内，位置跟踪误差 ≤0.3m、速度跟踪误差 ≤0.6m/s，并连续保持 ≥0.25 秒；还须完成五个门且无异常。它是工程测试阈值，不是理论稳定性证书。报告分开记录五门通过与恢复通过，避免只在终点悬停才恢复也算通过。恢复时间使用参考时钟；注入操作的实际墙钟开销单独记录，不包含在 DNN 推理时间里，可能影响当周期 deadline。

## 一整段试飞命令

手动打开 UE4 项目并点击 Play，只开一个有 RPC 的 AirSim 实例：

`E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe`

项目：`F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject`。

随后复制以下整段到 PowerShell。程序只连接手动打开的实例，不自动启动第二个窗口。默认四个 seed=0 模型已存在，**不必等待新增 seed 训练结束**，但不要同时开 GPU 训练和渲染试飞。最多 20 次飞行（4 次教师 +16 次 DNN）；失败/中断后重跑会跳过已完成试验。

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $proj = 'F:\FalconReproduction\airsim_control'
    $trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
    Set-Location -LiteralPath $proj
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    & $trainPy -m controller_architecture_study.evaluate_convex_recovery --seeds 0 --vehicle-name Drone1 --resume
    if ($LASTEXITCODE -ne 0) { throw '试飞队列中断/连接或状态读回异常；修复后重跑本段。' }
    $reportPath = '.\controller_architecture_study\results\decision_convex_recovery_v1\runs\recovery_closed_loop_seeds0.json'
    $result = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $result.summary | Format-Table model,completed_trials,successes,failures,skipped_teacher_failed,success_rate,recovery_confirmation_mean_seconds,post_kick_peak_position_error_max_m -AutoSize
    Write-Host "扰动报告：$proj\controller_architecture_study\results\decision_convex_recovery_v1\runs\recovery_closed_loop_seeds0.json"
}
```

报告 `status=complete` 仅表示队列已执行完，不表示所有模型 PASS。主要检查 `successes/failures/skipped_teacher_failed`，以及每次试验的 `recovery`、碰撞、恢复时间、峰值误差和过门偏差。每次飞行另保存 JSON 与 NPZ；teacher 扰动试验不会覆盖此前正常试飞的 `teacher_acceptance.json`。

暂不要求全部 seed 试飞。补训完成后，如第一轮需要扩大证据，可将上述试飞命令的 `--seeds 0` 改为 `--seeds 0 1 2`；会生成独立 `recovery_closed_loop_seeds0_1_2.json`，最多 52 次飞行。不能在已有报告中更换模型/工况继续凑结果。

## 当前已检查 / 尚待执行

已检查：多 seed 默认 seed=0 配方完全保留、正式数据/源码哈希、已有四模型完成状态、新八模型队列、均值/标准差与缺失结果处理、状态注入时刻及坐标旋转、读回不一致报错、连续恢复判据、模拟飞行接口与日志、教师验收文件保护、试验中断恢复。自动测试使用临时数据和模拟客户端；其中 PASS 不是实际 AirSim 飞行证据。

尚待用户执行：seed=1、2 的八次正式补训，以及实际 AirSim 四工况恢复试验。待结果返回后再决定是否扩大结构网格或修订数据，而不是预先保证大模型更好。
