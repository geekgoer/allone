# 深而窄的网络实验与更难控制任务方案

## 现有证据与解释

本文件针对 `paper_track_p_12d_5ms_v1/dnn_dataset` 和 M1 的 29 个正式训练结果。8×128 的 128 是每层宽度；新实验的层数是隐藏 FC 层数，旧 FALCON `depth` 则包含输出 FC。两种命名不能直接按同一个层数比较。

`results/followup_v1/aggregate.csv` 中三个 seed 的归一化测试 MSE 均值：

| 模型 | 均值 | 标准差 |
|---|---:|---:|
| 普通 4×128 | 1.17519e-5 | 4.64716e-6 |
| 普通 8×128 | 9.48200e-6 | 2.08737e-7 |
| 普通 10×512 | 9.28020e-5 | 1.67119e-5 |
| 残差 10×512 | 4.91776e-5 | 4.91662e-6 |

残差 10×512 相对同尺寸普通网络降低约 47% 的平均测试 MSE，仍约为普通 8×128 的 5.2 倍。单次最小测试 MSE 来自普通 4×128 seed 2，但 8×128 在三 seed 下更稳定。它们是已测模型中的结果，不是全局最优证明。

代码证据：

- `mpc_controller.py` 预先求出双积分线性预测模型的控制增益。每次查询使用矩阵乘法得到控制序列，再限幅、计算推力范数、转换姿态和角速度。当前约束是求解后的裁剪，不是在预测时域中联合求解的障碍/执行器约束。
- `track_p_12d_contract.py` 按固定速度在当前门的直线通道上生成参考。当前采集配置速度为 5 m/s，dt 为 0.05 s。
- 测得的三维加速度从 state 取出后没有参与教师计算。正向、固定姿态的门下，纵向门距也被通道投影抵消。
- 本次 100 组离线查询中，仅改变加速度导致控制变化为 0；仅改变正向门距导致最大控制变化为 8.88e-16。这只说明这套教师与门朝向设置的性质，不适用于工程内所有其他控制器。

因此大数据量并不等于复杂函数。较小网络已经能表示这个结构较简单的映射。

对大模型表现的解释目前是**优化方面的假设**：训练曲线本身也较高，残差确实帮助训练，但固定 lr=1e-4、相同 epoch 预算、初始化和残差缩放未必适合所有深度/宽度。当前残差块是 `ReLU(h + Linear(h)/sqrt(L-1))`，采用非零随机分支初始化。跳连改善信号传播，不保证更深的回归模型最终误差更低。不能仅据现有曲线断言过拟合、梯度消失或软件故障。日志的 train MSE 是一轮更新过程中各批次的平均值，也不是最终最佳 checkpoint 在训练集上重新评估的 MSE。

## 可立即运行的架构实验

目的：在同一个教师任务上区分深度、宽度与残差的影响。沿用原 coarse sweep 的数据、拆分、归一化和训练协议：seed 0、Adam、lr=1e-4、batch 512、最多 500 epoch、至少 100 epoch、patience 50、gradient clip 1.0。早停只依据验证集。

先跑普通与残差 22×128 两个模型。完整扫描为以下六种尺寸，各跑普通/残差，合计 12 个新模型：

| 变量 | 尺寸 |
|---|---|
| 固定宽度 128，改变隐藏层数 | 12×128、16×128、22×128、34×128 |
| 固定隐藏层数 22，改变宽度 | 22×64、22×128、22×256 |

22×128 在表中出现两次，实际只训练一次。原来的 4×128、8×128、10×512 等作为已有参照。完整扫描是单 seed 筛选，后续应按验证结果为少数候选补 seed 1/2。训练参数未针对新模型单独调优，故此阶段回答的是同一训练设置下的表现。若大模型 train/validation 均仍持续下降，可另建低学习率续训诊断，不能覆盖本轮权重或把不同训练预算混成架构结论。

新的输出目录为 `controller_architecture_study/results/deep_narrow_v1`。普通与残差 22×128 均为 348,932 参数；8×128 是 117,764，10×512 是 2,372,612。较少参数不保证 GPU batch=1 推理更快，较深网络仍有顺序执行开销。

PowerShell 初始化：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
```

只跑最关键的普通/残差 22×128：

```powershell
& $trainPy -m controller_architecture_study.run_followup --config .\controller_architecture_study\configs\deep_narrow_22x128_v1.json --results-dir .\controller_architecture_study\results\deep_narrow_v1 --phase all --device cuda
```

运行完整 12 项扫描，也可直接从这条命令开始：

```powershell
& $trainPy -m controller_architecture_study.run_followup --config .\controller_architecture_study\configs\deep_narrow_v1.json --results-dir .\controller_architecture_study\results\deep_narrow_v1 --phase all --device cuda
```

串行训练。重复相同命令会跳过已完成模型，并从未完成模型的 `last_model.pt` 继续。先跑两项再执行完整扫描，两个 22×128 的配置相同，因此会跳过它们。每轮原子保存 `last_model.pt`，验证改善即保存 `best_model.pt`；每个模型单独完成测试和产物保存，不等待整组完成。未完成的当前 epoch 会从上一完整 checkpoint 重做。

仅查看任务列表，在上述命令末尾加 `--dry-run`，不会开始训练。不需要运行 AirSim。

汇总已完成结果：

```powershell
& $trainPy -m controller_architecture_study.summarize_followup --config .\controller_architecture_study\configs\deep_narrow_v1.json --results-dir .\controller_architecture_study\results\deep_narrow_v1
```

输出 `per_run.csv`、`aggregate.csv`、`analysis.md`。包含四个原 seed-0 参照和新完成项，未完成项列为 pending。新模型未跑完时也可以汇总。汇总已修正“普通 seed 0 必须在旧目录”的假设，可读取新目录的普通/残差 22×128，并逐尺寸报告匹配对比。

## 更难且有实际价值的下一阶段任务（方案，尚未实现）

目标：获得“计算时间—飞行质量”的可测权衡，让网络应对当前控制器没有覆盖的条件。大网络是否占优是待检验结论，不能通过加随机标签或故意破坏小网络来制造。

独立候选方向：

1. **时序信息完整的非线性约束轨迹控制**：输入当前物理状态、前瞻轨迹、过去动作/执行器估计、控制保持时间 T 和已知延迟，学习预测时域内受约束的非线性最优动作。优点是仍可比较 MLP/残差/SuperNet；代价是专家求解和仿真标定更复杂。
2. **局部规划与控制联合策略**：输入未来几个门和局部障碍信息，由策略选择进门方向、弯前减速和恢复动作。标签可来自有一致决策规则的规划器+NMPC，或先模仿再用闭环奖励优化。代价是必须处理绕障多解，防止相同观测被不同绕行方向标注。
3. **感知加历史控制**：输入深度图/局部点云与 IMU 历史，从观测推断通道与运动状态再控制。这会增加真实的信息处理需求，但需要视觉/时序网络，已明显超出单纯 12D MLP 控制器比较。

**决定：优先方案 1，再按需要加方案 2 的障碍和门序列。** 工程已有独立目录 `experiments/falcon_mpc_complete_300d_v1`，README 描述了当前 10D ACADO 状态、21×10D 参考状态与20×4D参考控制的完整输入，可作为后续核查和复用起点。本轮没有重新验证其求解器与数据质量，也没有修改该目录。增加到 300D 本身不保证函数更难；实际难度来自非线性耦合、约束切换、可观测动态和不同时间尺度。

建议的输入/输出职责：

```text
当前位置/姿态/速度/角速度
+ 足够的未来参考位置、速度、加速度/姿态
+ 最近已执行控制和执行器状态估计
+ 当前选定的控制保持时间 T、可测通信/执行延迟
+ 第二阶段加入的门序列和局部障碍几何
    -> DNN
    -> [collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]
```

教师采用经过验证的非线性模型，并在时域内处理推力、倾角、角速率、动作变化率，以及第二阶段的障碍/通道约束。执行器滞后和风扰动若影响最优动作，必须通过估计量或历史观测对学生可见；不能把专家独有的真实风值藏起来却要求静态网络给出唯一动作。多解规划要提供选定路径上下文或一致的择优规则。专家标签是实际优化出的动作，参考控制是规划前馈，二者不能混淆。

场景逐级增加：先变速直线/平滑转弯/上升下降，再急弯与减速、门朝向变化、约束附近恢复，再加入执行时延/风扰动。可从 2–8 m/s、T=25/50/75/100 ms 的示例网格起步；实际范围由专家闭环可行性决定，不把示例值当验收要求。所有 T 的样本都要由相应离散模型重新求解；不能改 T 后复用旧标签。若比较 T 时保持预测窗口时长，需要明确调整预测步数，不能无意间同时改变窗口时长这个变量。

数据包括可实现的独立状态-参考查询、专家闭环轨迹和学生偏离后的专家重标注。训练/验证/最终留出环境按轨迹、场景和短 rollout 分组；控制饱和、急转弯、恢复等重要阶段单独报告误差，不让长时间容易样本占据总体指标。增加随机扰动前检查状态的一致性与可恢复性。

评价：固定同一场景与扰动 seed，联合测量穿门/避障成功率、轨迹误差、用时、控制代价、端到端时延 P95/P99、deadline miss。先确认不同模型存在有意义的质量/时延差异，再研究动态 T 与复杂度切换、驻留模型与 SuperNet。GPU 上深窄模型可能参数更少但延迟更高，需要实测 batch=1；所有模型都远快于 deadline 时，复杂度切换的收益也可能很小。

反证条件：如果小网络在多个 seed、未见环境、充分训练与等价时延预算下仍处于最好权衡区间，就接受这一结果。不要预设最后必须是 22/34 层更好。真实高水平飞行也可能由较小的低维控制策略实现。

## 有限文献核对

本轮只核对了三篇直接相关的原始论文，不构成完整文献综述或新颖性证明：

- He et al., *Identity Mappings in Deep Residual Networks*: https://arxiv.org/abs/1603.05027 。支持恒等跳连与信号传播对优化的作用，不保证当前任意残差 MLP 都更优。
- Sun et al., *A Comparative Study of Nonlinear MPC and Differential-Flatness-Based Control for Quadrotor Agile Flight*: https://arxiv.org/abs/2109.01365 。讨论敏捷飞行中的非线性、气动与执行器约束，以及性能和计算代价。
- Kaufmann et al., *Champion-level drone racing using deep reinforcement learning*: https://www.nature.com/articles/s41586-023-06419-4 。其 Methods 中的策略和值网络采用两个128单元隐藏层，提醒我们飞行任务复杂不必然要求很大的低维控制 MLP。
