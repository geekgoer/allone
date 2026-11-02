# FALCON paper-fidelity 12-D DNN 实验空间

> **停用提示（2026-09-13）**：本目录的 `paper_core_40k` 使用 CasADi/IPOPT，实测单次 NMPC 约 `250–1100 ms`，不满足 50 ms 在线控制周期。它只保留作故障分析，不能继续采集或用于论文核心数据训练。请改用 [`../falcon_paper_fidelity_acado_v1/README.md`](../falcon_paper_fidelity_acado_v1/README.md) 中的 ACADO/qpOASES 实验空间。不要把两个目录的 episode 混合。

本空间同时解决两个目标：

1. 用独立的 `paper_core_40k` 尽可能忠实复现论文公开的训练协议；
2. 用独立的 `recovery_extension_60k` 补足专家连续轨迹没有覆盖的恢复状态，降低闭环撞门风险。

先阅读 [FIDELITY_MATRIX.md](FIDELITY_MATRIX.md)。其中明确区分了论文事实、工程假设和鲁棒性扩展。

## 0. PowerShell 初始化

每次重启电脑后：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
```

安装一次 CasADi（只给教师 MPC 使用；DNN 训练不依赖它）：

```powershell
& $py -m pip install -r .\experiments\falcon_paper_fidelity_v1\requirements_collection.txt
```

检查论文公开协议：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\validate_protocol.py
```

期望结果是 `status: PASS` 和 `expected_samples: 40000`。

## 1. 先做可见 smoke test

自动启动可见 AirSim：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\collect_paper_core.py --auto-launch --render --smoke-test
```

如果手工打开 UE4 并已点击 Play，则不要写 `--auto-launch`：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\collect_paper_core.py --render --smoke-test
```

Smoke test 只有 2×2×10=40 条，用于验证 RPC、CasADi、控制输出和原子保存，不用于训练。

## 2. 生成严格的 paper core 40k

建议后台无渲染，速度更稳定：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\collect_paper_core.py --auto-launch
```

断电、Ctrl+C 或中断后继续：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\collect_paper_core.py --auto-launch --resume
```

每个 episode 写完后立即原子保存。中断最多损失当前尚未完成的一个 episode；已完成文件不会重跑。最终文件：

```text
experiments/falcon_paper_fidelity_v1/paper_core_40k/final/falcon_paper_core_40k.npz
```

完成后验证形状：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\validate_protocol.py --dataset .\experiments\falcon_paper_fidelity_v1\paper_core_40k\final\falcon_paper_core_40k.npz
```

## 3. 固定拆分 80/10/10

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\prepare_paper_core.py
```

输出应为 32,000 / 4,000 / 4,000 条，并写入：

```text
experiments/falcon_paper_fidelity_v1/dnn_dataset_paper_core_40k/
```

## 4. 串行训练论文六个 DNN

训练脚本默认串行，避免 Windows 驱动 TDR；每个 epoch 都保存 `latest_checkpoint.pt`，每个模型完成后立刻导出，不会等六个全部结束才保存。

先只训练最小模型做闭环门槛测试：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_v1\train_six_paper_models.py --only 10x512 --device cuda
```

确认流程后训练全部六个：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_v1\train_six_paper_models.py --device cuda
```

脚本严格使用 20,000 epochs 和 `patience=0`。重新运行同一命令会跳过已完成模型，并从各自的 `latest_checkpoint.pt` 继续未完成模型。六个输出分别位于：

```text
experiments/falcon_paper_fidelity_v1/runs_paper_core/depth{10|22|34}_width{512|1024}/
```

每个模型包含 `best_model.pt`、`latest_checkpoint.pt`、`controller_scripted.pt`、`metrics.json`、`history.csv`，以及 `export/weights_fp32.npz`、`model_spec.json`、`normalization.json`、`golden_vectors.npz`。

## 5. Paper-core 单机闭环验收

手工打开 UE4 并点击 Play；此命令不带 `--auto-launch`，会连接你看到的窗口：

```powershell
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'

& $trainPy .\run_single_drone_dnn_gates.py `
  --model .\experiments\falcon_paper_fidelity_v1\runs_paper_core\depth10_width512\controller_scripted.pt `
  --normalization .\experiments\falcon_paper_fidelity_v1\runs_paper_core\depth10_width512\normalization.json `
  --input-target gate `
  --layout-root .\experiments\falcon_paper_fidelity_v1\paper_core_40k `
  --initial-state-id 0 `
  --device cpu `
  --epsilon 1.6 --seed 0 --speed 5.0 `
  --per-gate-timeout 20 --gate-count 5 --x-start 10 --x-spacing 10 `
  --gate-width 4 --gate-height 4 --gate-thickness 0.15 --gate-clearance 0.25 `
  --gate-asset 1M_Cube --keep-gates `
  --report .\experiments\falcon_paper_fidelity_v1\runs_paper_core\depth10_width512\closed_loop_seed0.json

Remove-Item Env:PYTHONPATH
```

必须看到五次 `crossed inside opening`，而不是只看“距离门中心很近”。

## 6. 如果 paper core 闭环不足，再生成鲁棒扩展

这是离线独立状态查询，不需要 AirSim；它覆盖正常、横向恢复、高度恢复、姿态恢复、高速和组合边界六类状态。每条输入都重新生成未来轨迹并重新求解 MPC。

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\generate_recovery_extension.py
```

中断后：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\generate_recovery_extension.py --resume
```

每 500 条原子保存一个 chunk。然后合并成鲁棒训练集：

```powershell
& $py .\experiments\falcon_paper_fidelity_v1\combine_robust_training_data.py
```

先训练一个 10×512 鲁棒模型：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_v1\train_six_paper_models.py `
  --data-dir .\experiments\falcon_paper_fidelity_v1\dnn_dataset_robust_100k `
  --runs-dir .\experiments\falcon_paper_fidelity_v1\runs_robust_extension `
  --only 10x512 --device cuda
```

只有当该模型在未见 seed/epsilon、较大初始偏差和五门闭环中通过，才值得训练其余五种结构。严格 40k 的结果与 robust extension 的结果必须分别报告。
