# 第二阶段：从最佳模型保函数扩展

## 第一阶段结果

这里的 h 是隐藏 Linear 层数，w 是隐藏宽度；MSE 是标准化控制输出上的 MSE。

| 模型 | 固定最佳检查点的训练 MSE | 验证 MSE |
|---|---:|---:|
| h4 w128 | 5.75446e-6 | 6.21185e-6 |
| h8 w256 | 2.55587e-6 | 4.33656e-6 |
| h12 w1024 | 2.82026e-6 | 1.07366e-4 |

中模型验证误差比小模型低约 30.2%，说明优化过程影响了原先的排名。
最大模型验证/训练 MSE 约 38 倍，验证误差约为中模型的 24.8 倍。
这说明单纯拟合训练集已不是主要障碍；但不能仅凭该比例确定具体的泛化失败原因。
此前教师审计也表明当前 MPC 映射较简单，且不使用输入中的实测加速度。
增加容量未必带来收益，不能把强制单调排名作为实验目标。

## 下一阶段做什么

以第一阶段 h8 w256 的 best_model.pt 为共同起点，做三组对照：

1. h8 w256：相同容量继续训练，控制额外训练预算的影响。
2. h8 w512：仅增加宽度，通过复制神经元并分配出边权重保留函数。
3. h12 w256：仅增加深度，在 ReLU 后插入恒等 Linear + ReLU。

复制神经元的出边采用不等比例分配，每组贡献之和仍为一，允许后续产生不同梯度。
扩展不是从头随机训练，也不是已经完成的共享权重超网。
这是检验“大模型较差是否与初始化/优化有关”的诊断实验，不能作为从头训练架构的公平替代。

每个模型使用相同的数据、归一化、seed=0、batch=512、Adam、lr=1e-5；优化器均重置。
最多追加 150 epoch，至少 50 epoch，patience=50。
epoch 0 也参与最佳模型选择，仅使用验证集；最后评估训练集和验证集，不读取测试集。
相同 epoch 不代表相同计算时间。当前仅单 seed，不能据此宣布普遍容量规律。

## 已完成的验证

CPU 单元测试 3 项通过，未进行真实训练。
使用训练集前 4096 条输入，扩展模型与原模型的标准化输出最大绝对差异：

- h8 w512：2.86102294921875e-6（FP32 运算顺序的微小差异）。
- h12 w256：0。

旧数据和旧检查点未修改。

## 用户执行命令

在 PowerShell 中逐行执行。不要复制提示符或把多行赋值连成一行。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

# 已准备好；此命令可重复运行，只做 CPU 输出一致性检查，不训练。
& $trainPy -m controller_architecture_study.prepare_capacity_growth

# 串行训练三组；每完成一个即保存，不等待其余模型完成。
& $trainPy .\controller_architecture_study\run_capacity_finetune.py --config .\controller_architecture_study\configs\capacity_growth_v1.json --results-dir .\controller_architecture_study\results\capacity_growth_v1 --device cuda

# 全部完成后，复制此文件内容反馈。
Get-Content -Raw -LiteralPath '.\controller_architecture_study\results\capacity_growth_v1\comparison.json'
```

停止后重复训练命令即可：完成的模型自动跳过，未完成的从已保存完整 epoch 恢复。
断电可能丢失当前未保存 epoch，不会要求重训已完成模型。
如果已经达到 early stopping，重复命令不会自动再追加 150 epoch。

## 输出和后续判读

全部新产物位于 controller_architecture_study/results/capacity_growth_v1：

- initializations/*：扩展后的初始检查点、config.json、growth_check.json。
- plain_mlp_h8_w256_seed0/、plain_mlp_h8_w512_seed0/、plain_mlp_h12_w256_seed0/：独立训练结果。
- 每组保存 best_model.pt、last_model.pt、train_log.csv、initial_metrics.json、offline_metrics.json、completion.json。
- comparison.json、comparison.csv：汇总训练和验证 MSE。

注意 initializations 内的 best_model.pt 只是兼容既有训练接口的初始权重文件名，并非已经微调完成。

对比扩展模型和同预算 h8 w256 对照的最终验证 MSE：

- 若扩展明显改善，下一步用多 seed 验证是否稳定，再做最终测试集评估和闭环测试。
- 若仅训练误差改善而验证没有改善，优先研究泛化/无关输入，再决定是否改变教师和数据复杂度。
- 若三组都接近同一水平，当前任务可能不需要更大容量，不继续盲目加 epoch。

本阶段不会生成新教师数据、改变数据拆分、进行 AirSim 试飞、量化或 DPU 编译。
