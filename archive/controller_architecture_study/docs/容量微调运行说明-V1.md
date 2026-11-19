# 第一阶段：小、中、大模型低学习率微调对照

本阶段只判断现有容量反转是否与优化设置有关。保留原始教师、48 万条数据、环境划分、归一化、网络结构和旧权重不变。不修改 MPC，不生成新数据，不自动执行函数保持扩展或重新训练 SCENIC 全表。

## 任务与参数

| 顺序 | 隐藏 FC × 宽度 | 起始权重 |
|---:|---|---|
| 1 | 4×128 | results/plain_mlp_h4_w128_seed0/best_model.pt |
| 2 | 8×256 | results/scenic_table3_v1/plain_mlp_h8_w256_seed0/best_model.pt |
| 3 | 12×1024 | results/scenic_table3_v1/plain_mlp_h12_w1024_seed0/best_model.pt |

上述相对路径位于 controller_architecture_study 下。三者就是本次已经试飞的代表模型。

- 从各自最佳权重开始，不从头训练；初次微调重置 Adam 优化器，学习率由 1e-4 改成 1e-5。这是统一微调协议，不是严格只改变学习率而保留全部优化器历史的消融。
- 每个模型最多**追加 300 epoch**；batch=512，seed=0，其余模型/数据配置继承原实验。
- 最少 100 epoch，patience=100；达到追加 300 轮上限或触发早停后进入下一模型。
- 三个模型串行，每个独立子进程；结束一个再启动下一个，以释放该进程的 GPU 资源。
- 初始模型先在完整 train 和 validation 上评估，作为 epoch 0 候选保存。只有验证损失更好才替换 best_model.pt。epoch 0 最好不代表程序出错，而是没有发现更好的微调权重。
- 最后使用同一最佳检查点评估完整 train 和 validation，方便判断优化不足或泛化差距。
- 本阶段不读取 test.npz，不使用测试集早停，不以测试结果决定学习率。
- 每轮打印日志，每个完整 epoch 保存 last_model.pt；新最佳权重立即保存。使用临时文件再替换，减少中断时破坏已有完整文件的风险。

不能保证 GPU 驱动或突然断电不会导致系统/磁盘故障。恢复使用磁盘上最近的完整检查点，未完成的一轮会重跑。不要同时在两个终端启动同一实验目录。

## PowerShell：从头开始

不需要启动 AirSim。建议关闭 UE4 和其他占用 GPU 的任务；关闭电脑睡眠。

在一个 PowerShell 窗口执行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

# 清除当前终端之前为 AirSim 设置的包路径，避免混用两个虚拟环境。
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
```

可选：只检查三个任务和路径，不训练、不创建结果目录：

```powershell
& $trainPy -m controller_architecture_study.run_capacity_finetune --dry-run
```

正式开始，只需一条命令：

```powershell
& $trainPy -m controller_architecture_study.run_capacity_finetune --device cuda
```

该命令在当前终端运行，会按顺序完成三个模型，不是后台任务。指定 cuda 是为了在 GPU 不可用时明确报错，而不是悄悄切到 CPU。

开始时会看到类似：

```text
Warm-started from ...best_model.pt; optimizer state reset
Initial checkpoint retained: validation_normalized_mse=... (epoch=0)
plain_mlp_h4_w128_seed0 epoch=1 train=... val=... best=... stale=...
```

初始完整训练/验证评估会先花一些时间。epoch 从 1 开始表示本次追加轮次，不表示丢弃旧模型从零训练。

## 中断后继续

Ctrl+C 中止后等待命令退出；重启电脑后先重新执行上面的变量和目录设置，然后运行**同一条正式命令**：

```powershell
& $trainPy -m controller_architecture_study.run_capacity_finetune --device cuda
```

自动行为：

1. 有完整 completion.json 的模型跳过。
2. 未完成的模型恢复 last_model.pt 的权重、Adam 状态和 Torch 随机数状态，继续剩余追加轮数。
3. 如果只在初始化阶段中断，重新读取源 best_model.pt 初始化该微调任务。
4. 如果训练已早停但最终评估中断，只补做评估，不再多训练一轮。
5. 若源 checkpoint 的 SHA256 或配置发生变化，拒绝混用旧结果，要求使用新的结果目录。不要手动修改本次 config 后冒充同一次恢复。

异常导致当前模型退出时，队列停止，前面已经完成的模型保留。修复异常后仍用同一命令继续，不会重训完成项。不要删 last_model.pt、best_model.pt、config.json 或 completion.json 来强行继续。

## 输出与反馈

全部新结果位于：

```text
F:\FalconReproduction\airsim_control\controller_architecture_study\results\capacity_finetune_v1
```

每个模型独立保存：

```text
config.json             微调配置和源检查点哈希
initial_metrics.json    原始模型的完整 train/validation 指标
best_model.pt           验证集选出的最佳模型，可能仍为 epoch 0
last_model.pt           最近完整 epoch 的可恢复状态
train_log.csv           每轮在线训练损失、验证损失、梯度统计
offline_metrics.json    最佳模型在完整 train/validation 上的最终指标
completion.json         追加轮次、最佳 epoch、完成标记
complexity.json         参数量和 MACs
validation_curve.svg    本次追加训练的验证曲线
```

注意在线 train_log 的 train 损失是一个 epoch 内不同更新时刻的平均值；诊断泛化差距请使用 initial_metrics/offline_metrics 的固定检查点评估。

顶层 comparison.json / comparison.csv 每完成一个模型就更新。全部完成后打印汇总。也可以只重新汇总，不训练：

```powershell
& $trainPy -m controller_architecture_study.run_capacity_finetune --summarize-only
Get-Content -Raw -LiteralPath '.\controller_architecture_study\results\capacity_finetune_v1\comparison.json'
```

把 comparison.json 发回来即可。主要看微调前后 train/validation MSE、验证改善百分比、best_epoch 是否为 0，以及大模型与小模型的差距是否收敛。不要仅凭训练 MSE 或某一轮最小值宣称大模型更好。

## 后续阶段边界

- 原 scenic_table3_v1 和旧配置不修改。
- 已单独提供 configs/scenic_table3_v2_layercount_corrected.json，其中隐藏 FC 数=(论文 L-1)/2；它只修正层数定义，不宣称复现了论文教师和损失数值。
- 本阶段启动器不会调用这份新全表配置。待低学习率对照结果出来，再决定是否做正确层数的扫描、Net2Net/残差修正器，以及非线性教师/新数据；不同时改变全部因素。
- 本次只执行代码和恢复逻辑测试（合成数据，优化器更新被禁用）与 dry-run；没有启动上述真实模型的训练。
