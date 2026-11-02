# FALCON ACADO v3：全局连续轨迹、门与控制器解耦

该目录是独立实验空间。它不会修改或覆盖
`experiments/falcon_paper_fidelity_acado_v2/paper_core_40k_acado_v2`。

## 核心变化

- 五个门只作为上层轨迹规划约束和被动验收对象。
- 每个 episode 开始时一次性生成穿过全部五个门的 C2 分段五次轨迹。
- ACADO MPC 每帧只接收无人机状态和未来 21 个轨迹参考状态。
- MPC 的 2 s 预测窗口可以跨过任意门边界，不再在当前门后冻结。
- 穿门事件不会切换、重建或截断控制参考。
- 仅在第五个门之后的 5 m 退出段平滑减速至零。
- DNN 输入为 12 维轨迹相对量，而不是门中心相对量：
  `[p_ref-p, rotation_error(q_ref,q), v_ref-v, a_ref]`。

## 阶段 1：离线协议验证

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py .\experiments\falcon_paper_fidelity_acado_v3_continuous\validate_protocol.py --json
```

通过标准包括：400 条规划轨迹、2000 个门边界全部 C2 连续，且
`frozen_position_with_nonzero_velocity_pairs` 必须为 0。

## 阶段 2：先看 hard 环境的可视化控制效果

如果使用已经手动打开并点击 Play 的 AirSim，不要添加 `--auto-launch`：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v3_continuous\demo_acado_five_gates.py `
  --initial-state-id 8 `
  --environment-id 38 `
  --duration 15
```

如果希望脚本自己打开并关闭 AirSim：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v3_continuous\demo_acado_five_gates.py `
  --auto-launch `
  --initial-state-id 8 `
  --environment-id 38 `
  --duration 15
```

日志会额外显示每个门平面的实际穿越速度。前四个门应连续通过；允许因
曲率产生平滑速度变化，但不应出现到门前停车再重新起步。第五门后减速悬停正常。

## 阶段 3：可视化 smoke collection

仅在阶段 2 观察确认后执行：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v3_continuous\collect_paper_core.py `
  --auto-launch `
  --render `
  --smoke-test `
  --resume
```

smoke 数据写入独立目录 `paper_core_40k_acado_v3_continuous_smoke`。

## 阶段 4：正式生成 40k 数据

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v3_continuous\collect_paper_core.py `
  --auto-launch `
  --resume
```

中断后重复同一条 `--resume` 命令即可继续。最终数据输出为：

`paper_core_40k_acado_v3_continuous/final/falcon_paper_core_40k_acado_v3_continuous.npz`

不要把 v2 episode 文件复制进 v3；两者的参考轨迹、输入语义和教师策略版本不同。

## 阶段 5：按 FALCON 公开 12 维接口训练六模型

连续轨迹数据本身保存的是轨迹相对输入。论文 Table II 轨道先利用同一批原始状态、
门坐标和完全相同的 ACADO/qpOASES 标签，重建论文明确写出的输入：

`[p_gate-p, Euler_gate-Euler_drone, velocity, acceleration]`。

这一步不启动 AirSim、不重新求解 MPC，也不修改已有 40k 数据：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $trainPy .\experiments\falcon_paper_fidelity_acado_v3_continuous\prepare_paper_gate_12d.py
```

默认按行固定随机种子拆分 80/10/10，这是论文“随机拆分”的通常解释；论文没有公开
拆分单位和种子。若要避免相邻时刻泄漏，可显式使用 `--split-unit episode`，但必须把
它报告为更保守的实现选择。

先训练最小模型验证环境和显卡稳定性：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v3_continuous\train_six_paper_models.py `
  --only 10x512 `
  --device cuda
```

确认完成后串行训练其余五种 Table II 结构（已完成的 `10x512` 会自动跳过）：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v3_continuous\train_six_paper_models.py `
  --device cuda
```

脚本把论文公开的 `20,000 epochs` 作为硬上限，使用学习率 `1e-4`、FC+ReLU 和六种
`depth {10,22,34} × width {512,1024}`。为减少验证集已经进入平台后的无效训练，
默认在连续 400 轮没有相对改善时把学习率乘以 0.3（最低 `1e-6`），至少训练 1500
轮，并在学习率已经退火到最低值、且连续 1200 轮没有超过 `1e-6` 的验证集 MSE
绝对改善时早停。训练结束始终
恢复验证集最优的 `best_model.pt`。这部分属于论文未公开训练细节的工程实现选择。
论文同样未公开的 batch size、优化器、初始化和随机种子分别采用 1024、Adam、
Kaiming/Xavier 和 2025，并记录在产物中。

训练严格串行；每一轮保存 `latest_checkpoint.pt`，每次验证集改善保存
`best_model.pt`。电脑重启或进程中断后重复同一命令即可从当前模型继续，已完成模型
会自动跳过；早停模型也会被正确标记为完成。学习率、训练/验证损失逐轮保存在
`history.csv`，调度器和早停状态随 `latest_checkpoint.pt` 一起恢复。为了避免当前环境的 NumPy/Matplotlib ABI 问题，正式训练使用
`--skip-curve`，完整逐轮数据仍保存在 `history.csv`。
