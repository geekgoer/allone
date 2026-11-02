# 控制器结构与动态控制研究分支

本目录现在包含237D教师/数据、独立模型族、深宽共享超网、量化和多机实时原型。接手当前工作先读：

- [整体Framework](../docs/FRAMEWORK.md)
- [协作进度与任务认领](../docs/COLLABORATION_STATUS.md)
- [研究文档索引](docs/README.md)
- [最新阶段结果](docs/BOUNDARY_TAKEOVER_RESULTS_20261102.md)

下文保留早期M1结构扫描的使用说明，属于历史入口；其中“不是训练SuperNet”等仅描述该扫描任务，不代表本目录尚未实现超网。不要直接执行历史训练命令来接续当前实时研究。

## 历史M1：Controller Architecture Scaling

这一步是为后续 compact SuperNet 找合理的控制器复杂度区间，**不是训练 SuperNet**。原有六个模型和 `elastic_supernet_test_a/` 均不改动。

从项目根目录运行（PowerShell 每行单独执行）：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
& $trainPy -m controller_architecture_study.freeze_baseline
& $trainPy -m controller_architecture_study.run_sweep --smoke --device cpu
```

查看冒烟输出：`controller_architecture_study/results/smoke/`。冒烟只使用部分训练/验证样本和两轮迭代，**误差不能用于架构结论**；汇总脚本会忽略它。

正式训练由用户自行启动（请确认 GPU 稳定、数据已在本机；不需要 AirSim/UE4）：

```powershell
& $trainPy -m controller_architecture_study.run_sweep --device cuda
```

逐个架构串行训练；每轮写 `last_model.pt`，最佳验证结果写 `best_model.pt`。中断后执行**同一条命令**，已完成目录跳过、未完成目录从 last checkpoint 续训。每个目录另有 `train_log.csv`、`validation_curve.svg`、`offline_metrics.json`、`complexity.json`。不建议同时启动两个相同 sweep 进程。

全部完成后（中途也可查看已完成模型，但不能得出最终结论）：

```powershell
& $trainPy -m controller_architecture_study.summarize_scaling
```

结果为 `results/architecture_scaling_summary.csv`、三张 SVG 图和 `docs/m1_report.md`。汇总只收录原六模型和已经完成的正式新模型；11 个全跑完后再分析饱和区与选 3–5 个闭环候选。模型数、MAC/Params 和完整训练协议见 `docs/experiment_protocol.md`。

如果改用 Linux/AutoDL：把**项目代码、`experiments/paper_track_p_12d_5ms_v1/dnn_dataset` 和现有六模型的原始运行目录**一起复制，确保项目相对路径不变；在项目根目录执行 `python -m controller_architecture_study.run_sweep --device cuda`。Linux 端只跑新模型时，可不复制六模型目录；但冻结基线与统一汇总应回到拥有六模型目录的环境执行。训练不依赖 AirSim。

M1 的 seed 稳定性、残差网络和大模型低学习率续训诊断是**另一个隔离实验**，见 [`docs/followup_protocol.md`](docs/followup_protocol.md)。
