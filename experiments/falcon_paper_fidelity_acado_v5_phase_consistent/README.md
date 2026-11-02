# FALCON 12-D v5：连续核心 + 相位一致恢复数据

这个目录保留 FALCON 公开的控制器接口：

```text
X = [p_gate-p, wrap(Euler_gate-Euler_drone), velocity, acceleration] # 12-D
U = [collective_thrust_acceleration, wx, wy, wz]                  # 4-D
```

姿态三维量采用论文符号 `r_goal-r` 的逐轴 XYZ Euler 差，并包裹到
`[-pi, pi)`。它与已经部署的 rotation-vector FLCN/v2 六模型不是同一个
数值契约，后续不能混用归一化、权重或板端输入编码。

教师仍为公开 `rpg_mpc` 的 ACADO/qpOASES 生成求解器，预测步数为 20，
AirSim 控制周期为 50 ms，固定标称速度为 5 m/s。

## 为什么需要 v5

v4 的每条独立状态都会从该状态重新生成一条五次多项式，并只保存新轨迹的
第一个 MPC 动作。闭环 DNN 因而可能不断重复保守的 `t=0` 动作并悬停。

v5 做三项修正：

1. 40k 核心仍采用 10 个初始状态、40 个环境、每次 5 s、20 Hz 的连续采集，
   但参考轨迹起点从一开始就带有 5 m/s 的门轴方向速度，不把随机实测低速当作
   隐藏的参考速度。
2. 80k 恢复样本绑定到连续轨迹上的固定 episode/step，相同相位的未来 21 个
   参考节点保持不变，只扰动无人机状态；绝不从扰动状态重新开始规划。
3. 恢复样本在 ACADO 输出空间按 `60% interior / 25% near-limit / 15% saturated`
   平衡，避免大量简单悬停样本或大量饱和常数标签淹没真正的非线性映射。这是让
   六种容量更容易拉开差距的主要数据措施，而不是人为修改测试 MSE。控制桶先被
   确定，再从兼容的扰动族中采样：轻微 `nominal_jitter` 不被强迫产生饱和控制，
   饱和样本由姿态、速度和组合恢复族提供。
4. 40k 核心的十个起始状态限制在可行的巡航邻域（前向速度 `4.7--5.3 m/s`、
   较小姿态和横纵速度扰动）。核心用于采集成功的连续 on-policy rollout；大范围
   低速、姿态和位置偏差由 80k 相位一致恢复集覆盖。旧的宽范围起始状态配合
   `t=0` 固定 5 m/s 参考会让 ACADO 首帧角速度饱和并在低空撞地，不能作为
   论文式核心。旧失败记录保留在 `paper_core_40k_falcon12_v5`，新核心写入
   `paper_core_40k_falcon12_v5_stable`。

40k 是论文式核心；额外 80k 必须在报告中注明为鲁棒性扩展。两者共享同一个
episode级 80/10/10 划分，避免同一轨迹的相邻帧泄漏到不同集合。

## 0. PowerShell 环境

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
```

## 1. 先看修正后的 ACADO 教师

先运行不需要 AirSim 的 400 路线检查：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\offline_sanity_check.py
```

预期 `status=PASS`、`episodes_checked=400`，且首个参考速度恒为 `5.0 m/s`。

手动打开 AirSim 并点击 Play 后运行（不要加 `--auto-launch`）：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v3_continuous\demo_acado_five_gates.py `
  --config .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\paper_core_config.json `
  --layout-root .\experiments\falcon_paper_fidelity_acado_v3_continuous\paper_core_40k_acado_v3_continuous `
  --initial-state-id 8 `
  --environment-id 38 `
  --duration 15 `
  --report .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\teacher_demo_hard.json
```

验收要求：五门全部 `PASS`，第一门前应主动加速，前四门不停车重启。
演示完成后先停止 Play 并关闭手动启动的 UE4；下面的 `--auto-launch` 要独占
AirSim RPC 端口。

## 2. 生成论文式 40k 连续核心

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\collect_paper_core.py `
  --auto-launch `
  --resume
```

中断后重复同一命令即可继续。完成标志：

```text
Paper core complete: X=[40000, 12] U=[40000, 4]
```

## 3. 生成80k相位一致恢复数据

此阶段不需要 AirSim：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\generate_phase_recovery.py `
  --resume
```

该命令逐块保存，可以断点续跑。

## 4. 构建固定训练/验证/测试集

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\prepare_dataset.py
```

预期总量为 120,000：40,000 连续核心 + 80,000 恢复样本。

## 5. 先做容量与闭环 pilot

先只训练最小和最大模型：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\train_six_models.py `
  --mode pilot `
  --seeds 0 `
  --device cuda
```

汇总容量差距：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\summarize_results.py `
  --runs-root .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\runs_pilot
```

只有当最大模型的 test normalized MSE 明显低于最小模型，并且 `10x512`
CPU闭环能连续通过五门时，才应训练全部六种结构。闭环命令中的模型和归一化：

```powershell
$run = '.\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\runs_pilot\seed_0\depth10_width512'
& $trainPy .\run_single_drone_dnn_gates.py `
  --model "$run\controller_scripted.pt" `
  --normalization "$run\normalization.json" `
  --input-target gate `
  --gate-attitude-encoding euler `
  --layout-root .\teacher_dataset_full_route_continuous_v2 `
  --initial-state-id 0 `
  --device cpu `
  --epsilon 0.8 `
  --seed 0 `
  --speed 5.0 `
  --per-gate-timeout 30 `
  --gate-count 5 `
  --x-start 10 `
  --x-spacing 10 `
  --gate-width 4 `
  --gate-height 4 `
  --gate-thickness 0.15 `
  --gate-clearance 0.25 `
  --gate-asset EditorCube `
  --keep-gates `
  --report "$run\closed_loop_seed0.json"
```

## 6. 训练六模型

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\train_six_models.py `
  --mode full `
  --seeds 0 `
  --device cuda
```

最终汇总：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\summarize_results.py `
  --runs-root .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\runs_full
```

如需用随机种子验证趋势稳定性，另行运行 `--seeds 0,1,2`，这会训练18个模型。
训练严格串行，每个模型逐轮保存 `latest_checkpoint.pt`，完成一个就保存一个，
重复命令会跳过已完成模型并恢复中断模型。

## 结果解释原则

- 主容量指标是未参与选模的 `test_normalized_mse`，与训练目标一致。
- `max_to_min_test_normalized_mse_ratio` 衡量六模型是否真正拉开；
  `log_parameters_vs_log_mse_correlation < 0` 表示总体容量趋势方向正确。
- 同时报告 paper core 和 phase recovery 的 physical MSE。
- 更复杂模型通常应在恢复数据上获得更明显优势，但有限样本和随机优化意味着
  不能诚实保证六个点严格单调。
- 禁止根据测试集挑 checkpoint；如出现违反趋势的结构或 seed，应原样报告，
  再从优化稳定性、数据覆盖和多 seed 均值分析原因。

## 7. Paper-only 40k 容量复现轨道

若目标是复现论文 Table II 的六模型容量与 MSE，而不是训练工程恢复控制器，
应只使用 40k 连续核心。80k phase recovery 保留为单独的鲁棒性实验，不混入
论文主 MSE。

构建固定的逐样本 80/10/10 划分：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\prepare_paper40k_dataset.py
```

论文只给出 80/10/10 比例，没有说明划分单位。本轨道采用确定性的逐样本划分，
并在 manifest 中明确披露可能存在连续相邻样本泄漏；原来的 episode-disjoint
120k 数据仍作为更严格的泛化测试保留。

先用较少轮次验证最小/最大模型的容量方向：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\train_six_models_paper40k.py `
  --mode pilot `
  --seeds 0 `
  --epochs 1000 `
  --device cuda
```

汇总 pilot：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\summarize_paper40k_results.py `
  --runs-root .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\runs_paper40k_pilot
```

pilot 通过后，严格按论文的固定 `1e-4` 学习率和 20,000 epochs 串行训练六模型：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v5_phase_consistent\train_six_models_paper40k.py `
  --mode full `
  --seeds 0 `
  --epochs 20000 `
  --device cuda
```

该入口禁用早停和学习率调度，但每轮仍保存最新 checkpoint，并由验证集选择
`best_model.pt`；中断后重复同一命令会恢复，已完成模型会跳过。
