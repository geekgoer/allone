# ACADO / 300D 独立状态数据：96k v2

## 本轮做了什么

保留当前 ACADO 教师、300D 编码和已验证的全局 C2 轨迹时钟；只借鉴旧 48 万条 12D 数据的独立状态、边界重点采样思路，不换回旧的简单 MPC 教师。

新目录：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\decision_nmpc_96k_v2`。
旧 `decision_nmpc_v1` 数据、模型、生成器，以及原 12D 数据和模型均不修改。

| 项目 | 正式数据 |
|---|---:|
| 环境组 | 60 |
| 每环境组的参考轨迹 | 20 |
| 每轨迹的独立状态 | 80 |
| 每环境组样本 | 1600 |
| 参考轨迹总数 | 1200 |
| 样本总数 | 96000 |
| Train / Validation / Test | 76800 / 9600 / 9600 |

这里的“环境组”是固定的数据分区单位，不是一个始终不变的 UE4 地图：每组包含 20 个不同门布局/参考轨迹。每条轨迹及其全部状态仅属于一个组、一个分区。
Train 固定环境组 0–47；Validation 48–53；Test 54–59。不随机打散同一轨迹的状态到不同分区。

每组包含近直线、S 形、升降组合、带偏航航向组合各 5 条不同实现。
沿用 v1 轨迹生成范围，但使用新随机实现，不复用 v1 的 40 条轨迹。轨迹名义巡航/过门速度 5 m/s，起步和停车速度 0；规划可行性速度上限 8 m/s、加速度上限 8 m/s²。

每轨迹的 80 条样本均衡分配到 8 类：起步、跟踪、切段、制动、位置恢复、速度恢复、姿态恢复、组合恢复；全量每类 12000 条。
每行独立抽相位/状态，重新完整初始化 ACADO 并计算标签，行与行不需要连续。
状态扰动不会重新规划并把参考时钟归零；仍取已提交轨迹的未来 21 个参考节点，间隔 0.1 s。

本次重点是参考覆盖，而不是同时把状态范围扩大：

- 常规四类：位置各轴 ±0.08 m、速度各轴相对参考 ±0.15 m/s。
- 位置/组合恢复：位置相对参考 x±0.6、y±1、z±0.6 m。
- 速度/组合恢复：速度各轴相对参考 ±1.5 m/s。
- 姿态/组合恢复：各轴旋转向量 ±0.25 rad。
- 起步包含相位 0 的零速状态，其余起步实际总速度不超过 3.5 m/s；全部实际总速度不超过 8 m/s。
- 每次各轴扰动抽样，有 30% 概率专门落在该范围下端或上端的 15% 边缘带内；其余仍全范围均匀抽样。均匀部分也可能落在边缘，因此总体边缘占比不是恰好 30%。

没有照搬旧 12D 的大范围门误差/当前速度边界：旧教师与 ACADO 任务不同；也不加入当前测得加速度噪声，因为当前 ACADO 反馈状态没有使用它。未来参考加速度通过姿态/前馈编码进入 300D。

## 输入、标签与接受条件

输入契约不变：`decision_nmpc_local_relative_pv_300d_v1`。

| 索引 | 内容（Z-up / FLU） |
|---|---|
| 0:10 | 当前局部位置零向量、四元数 wxyz、当前速度 |
| 10:220 | 21 × [参考相对当前位置 3、参考四元数 4、参考速度减当前速度 3] |
| 220:300 | 20 × 规划器前馈推力/角速度 4；不是优化器输出 |

标签仍是 MPC 的首条控制 `[F, wx, wy, wz]`，物理单位为 m/s²、rad/s，坐标为 AirSim NED / FRD。不会改为执行预测序列最后一步或平均动作。
离线每次独立查询做 4–32 次 RTI；接受阈值不变：KKT ≤1e-3，相邻迭代首动作差异 ≤1e-4，控制在编译边界内且数值有限。
拒绝样本在同一状态类别/同一轨迹内重新采样并记录原因，连续 8 次拒绝停止。不会偷偷放宽阈值或缩小范围。
每轨迹第一条接受样本再完整查询一次，差异必须 ≤2e-5；不是所有行都做了重复查询。
归一化只拟合 Train。

这些条件是数值审计，不是全局最优、所有恢复状态可达/可恢复、DNN 闭环安全的证明。
本次不会保证消除 train–validation gap，也不会预先保证越大的 DNN 越好。

## 1. 正式生成数据

不需要启动 UE4 或点击 Play，也不需要 GPU。这一步是离线 ACADO 标签计算。
在 PowerShell 完整粘贴以下代码；不要粘贴 `(base) PS ...>` 提示符。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

& $py -m controller_architecture_study.decision_dataset_96k generate --resume
if ($LASTEXITCODE -ne 0) { throw '数据生成失败或中断，先查看错误；不要直接训练' }
& $py -m controller_architecture_study.decision_dataset_96k prepare
if ($LASTEXITCODE -ne 0) { throw '数据整理失败，先查看错误；不要直接训练' }
```

每 80 条完整接受样本保存一个数据块，再登记到 manifest。Ctrl+C/掉电后重新运行相同的 `generate --resume` 命令即可：已登记块与轨迹文件校验 SHA256 后跳过，未完成块重算。
源码、DLL、配置发生变化时，拒绝在原目录续接。中断时正在写入的一个块不保证被保留，但不会把未登记块当作完成。
只有全量采集完成后才允许 prepare；重复 prepare 不会重新随机拆分。

预期正式整理结果：`status: PASS`、`samples: 96000`、`routes: 1200`、分区样本 `76800/9600/9600`，分区轨迹 `960/120/120`。
这些是数量/完整性要求，不是全量真实运行已成功的声明。

主要产物都在新目录中：

- `manifest.json`：版本、源码/DLL/块/轨迹哈希、完成状态及拒绝统计。
- `chunks/envXX_routeYY.npz`：输入、标签、环境组/轨迹/状态类别、相位、速度、收敛诊断。
- `routes/envXX_routeYY.json`：门位置/航向、C2 段系数与全局时间，可用于后续一致回放。
- `data_audit.json`：数量、类别、饱和/拒绝情况，以及 Validation 相对 Train 范围诊断。
- `dnn_dataset/`：三个分区 NPZ、`normalization.json`、`split_manifest.json`。

`validation_coverage` 中的逐维范围诊断不检查联合分布密度，不是自动调拆分的依据，也不决定 PASS。Test 不参与这一范围诊断或后续架构选择。

如果只想先检查小流程，可选用独立冒烟目录；不要求额外做一轮正式试飞：

```powershell
& $py -m controller_architecture_study.decision_dataset_96k generate --smoke-test --resume
if ($LASTEXITCODE -ne 0) { throw '冒烟生成失败' }
& $py -m controller_architecture_study.decision_dataset_96k prepare --smoke-test
```

冒烟 60 组 × 1 轨迹 × 8 状态＝480 行，拆分 384/48/48，不能用于正式训练入口。

## 2. 查看正式审计

```powershell
$audit = Get-Content -Raw -LiteralPath '.\controller_architecture_study\results\decision_nmpc_96k_v2\data_audit.json' | ConvertFrom-Json
$audit | Select-Object status,samples,environments,routes,split_counts,split_route_counts,rest_start_samples,saturated_fraction,rejected_queries | ConvertTo-Json -Depth 5
$audit.validation_coverage.route_heading_yaw_range_rad | ConvertTo-Json
```

检查数量之外，关注恢复类别是否大量饱和、拒绝比例是否明显升高、零速起步是否存在，以及参考覆盖。若生成停止并出现 failure.json，先把错误发回来，不忽略错误强行训练。
建议先把此审计结果发回来；如果完整 PASS 且没有明显异常，也可执行下一步。

## 3. 训练三个代表结构

明确指定 `--dataset-version 96k`。不写该参数会使用旧 40k v1 分支，不是本轮数据。
三个网络分别为 **4 个隐藏层 ×128、8 个隐藏层 ×256、8 个隐藏层 ×512**；隐藏层 Linear+ReLU，输出 Linear，输入 300，输出 4。
本轮从头训练，不续接旧 40k 或 12D 模型。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

& $trainPy -m controller_architecture_study.train_decision_pilot --dataset-version 96k --device cuda
```

固定 seed=0、Adam lr=1e-4、batch=512、梯度裁剪 1、最多 400 epoch、至少 100 epoch、patience=80。
同一套数据/归一化/训练规则下比较容量；当前不增加更多模型或 seeds。
一个模型一个子进程，串行运行。每完整 epoch 保存 `last_model.pt`；最佳验证 epoch 保存 `best_model.pt`。初始化另有检查点。
中断后重新运行相同命令：跳过已完成模型，其余恢复已保存 epoch；子进程异常会停止队列，保留前面模型和已有检查点。不保证硬件/驱动崩溃时正在写入的一次保存成功。
只评估 Train/Validation，不用 Test early stopping，也不读 Test 挑模型。

输出目录：

`F:\FalconReproduction\airsim_control\controller_architecture_study\results\decision_nmpc_96k_v2\runs`

每完成一个模型立即更新 `runs/comparison.json`，各模型另有权重、优化器/RNG 恢复状态、训练历史、SVG 曲线、按输出/状态类别的离线指标。

## 4. 把结果发回来

避免复制整份几百行指标，只需先发简表：

```powershell
$comparison = Get-Content -Raw -LiteralPath '.\controller_architecture_study\results\decision_nmpc_96k_v2\runs\comparison.json' | ConvertFrom-Json
$comparison.models | ForEach-Object {
    [pscustomobject]@{
        model = $_.model
        train_mse = $_.train.normalized_mse
        validation_mse = $_.validation.normalized_mse
        gap_ratio = $_.validation.normalized_mse / [Math]::Max($_.train.normalized_mse, 1e-15)
    }
} | Format-Table -AutoSize
```

比较同一个 v2 数据集上的模型；v1/v2 的输出分布及归一化不同，不能把两者标准化 MSE 数值的下降直接当作公平的性能提升。
仍需后续闭环验证。300D 专用试飞入口已在后续补齐，见同目录 `FLY_DECISION_NMPC_96K_V2.md`；不能拿这些权重直接跑旧 12D gate runner。Agent 没有启动实际飞行、正式数据生成、GPU 训练或 DPU 工作。

## 代码验证记录

- 24 项回归通过：临时文件、假教师、CPU 流程测试；涉及训练的测试禁用了优化器更新，没有真实训练。
- 32 条小规模真实 ACADO 接口查询接受；过程中 2 次拒绝后重采。
- 最大 KKT 约 1.2984e-4，重复查询最大控制差异 0。
- 1200 条正式参考轨迹全部通过规划器的数值可行性检查，四种几何各 300 条；没有为此计算全量 MPC 标签或进行真实试飞。
- 此小规模检查没有写数据集，也不代表 96000 行或所有恢复状态已经验证。
