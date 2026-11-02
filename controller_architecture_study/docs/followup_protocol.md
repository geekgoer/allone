# M1 Follow-up：seed 稳定性、残差消融、优化诊断

此实验是针对 M1 单 seed、深网训练误差较高的结果做**诊断**，不修改原 11 个模型、原六个 FALCON 模型、数据集或 `elastic_supernet_test_a/`。正式结果写入 `results/followup_v1/`；短跑写入其 `smoke/` 子目录且汇总时忽略。

## 三个互相区分的问题

1. **seed 稳定性**：原 seed 0 已有结果；按完全相同的 coarse-sweep 协议新增 seed 1、2，结构为 2×128、4×128、8×128、10×512，共 8 次新训练。每个结构形成三个 seed 的比较。配置顺序优先 4×128 与 10×512。
2. **残差消融**：新增残差 4×128 seed 0 和残差 10×512 seeds 0、1、2，共 4 次。残差网络仍含相同数量和尺寸的隐藏 FC：首层 `12→width`，后续 `hidden_layers-1` 层 `width→width`；每个后续层计算 `ReLU(x + linear(x)/sqrt(hidden_layers-1))`，最后 `width→4` 线性输出。与普通 MLP 的 FC 参数量和 MAC 相同；额外逐元素乘/加单独记录，不是“零成本”。其因子固定，不学习。比较重点是同 seed 的 10×512 普通/残差网络**训练误差**能否拉近。
3. **降低学习率续训诊断**：单独从已有普通 10×512 seed 0 的 `best_model.pt` 初始化，Adam 状态重置，`lr=3e-5`，最多再训练 200 epoch，验证集 60 epoch 不改善则停止。这不是独立 seed，也不是同预算架构比较；它只检验当前大网络是否还有可学空间。绝不写回原 checkpoint。

训练/验证/test 使用原 `dnn_dataset`；没有重切分。输入、标签、归一化和 normalized-MSE 损失均不变。前两组从头训练使用原配置：Adam、`lr=1e-4`、batch 512、最多 500 epoch、至少 100 epoch、patience 50、gradient clip 1.0。早停只看验证集；测试集只在训练完成后评估。输出包括每轮训练/验证误差、梯度范数、clip 频率，以及完整测试指标。原 seed 0 日志没有梯度诊断列，这是新加的**只记录、不改变更新**的指标。

## PowerShell 命令

在项目根目录，确保 `$trainPy` 指向可用的 PyTorch Python：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
& $trainPy -m controller_architecture_study.run_followup --phase all --smoke --device cpu
& $trainPy -m controller_architecture_study.run_followup --phase all --dry-run
```

正式运行可分阶段；中断后执行**相同命令**自动跳过已完成项，并从未完成项的 `last_model.pt` 续训：

```powershell
& $trainPy -m controller_architecture_study.run_followup --phase repeats --device cuda
& $trainPy -m controller_architecture_study.run_followup --phase residual --device cuda
```

第三组为**独立可选**的优化诊断；中断后第二条命令带 `--resume`：

```powershell
& $trainPy -m controller_architecture_study.train_controller --config .\controller_architecture_study\configs\h10_w512_finetune_v1.json --output-dir .\controller_architecture_study\results\followup_v1\plain_mlp_h10_w512_seed0_lr3e-5_finetune --device cuda
& $trainPy -m controller_architecture_study.train_controller --config .\controller_architecture_study\configs\h10_w512_finetune_v1.json --output-dir .\controller_architecture_study\results\followup_v1\plain_mlp_h10_w512_seed0_lr3e-5_finetune --device cuda --resume
```

第一条只在首次运行，第二条只在中断后运行。最后汇总（允许部分完成，但会明确列出 pending）：

```powershell
& $trainPy -m controller_architecture_study.summarize_followup
```

查看 `results/followup_v1/per_run.csv`、`aggregate.csv` 和 `analysis.md`。`aggregate.csv` **不包含**续训诊断；它只汇总独立 seed。单 seed 的残差 4×128 只能作对照，不能估计方差。任何闭环控制结论仍需 AirSim 测试。
