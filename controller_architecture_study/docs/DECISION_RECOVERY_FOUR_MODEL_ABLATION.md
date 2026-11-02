# 96k NMPC：Train-only 补充与四模型对照

## 范围和判断标准

这次不是重新生成整套数据，也不是修改教师来强行得到“大模型更好”。

- 原实验：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\decision_nmpc_96k_v2`，只读。
- 新实验：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\decision_recovery_v1`。
- 输入仍是原有 300D `decision_nmpc_local_relative_pv_300d_v1`；标签仍是同一 ACADO 教师的首步控制。
- 原轨迹系数、全局 C2 参考时钟、5m/s 名义速度、教师收敛要求和控制边界不改。
- 原 Validation 9,600 条逐字节复制；归一化逐字节复制，不重算。
- 原 Test 9,600 条既不重新划分，也不加载、复制或评估，仍在原目录中。
- 原四个 `best_model.pt` 只读。新训练均从相应原权重开始，重置优化器；原权重作为 epoch 0 候选保留。

以同一 Validation 的总 normalized MSE、联合恢复 family 7 MSE、各输出误差和各 family 误差判断改善；不能只比较 Train MSE，也不承诺模型误差随参数量严格单调。
本轮沿用 seed 0，属于诊断性对照，不是多种随机种子的统计结论。闭环验收留到候选选定以后。

## 新增训练数据

仅使用 Train 的 48 个环境组、960 条已保存路线，每条补 32 个独立 MPC 查询，共 30,720 条。
合并后的 Train 为 **107,520 条**；原 76,800 条原样保留，不下采样、不删除困难样本。
新增条目用原 `family_id=7` 加独立 `augmentation_kind` 记录类别；该 family 在新 Train 中占 37.5%，所以这也改变了训练样本权重分布。Validation 的各 family 比例不改。

| 每条路线的补充类别 | 条数 | 目的 |
|---|---:|---|
| 位置 + 速度恢复 | 4 | 填充成对偏差 |
| 位置 + 姿态恢复 | 4 | 填充成对偏差 |
| 速度 + 姿态恢复 | 4 | 填充成对偏差 |
| 连续幅度的联合恢复 | 12 | 填充正常状态与困难状态之间的区域，同时保留少量边缘样本 |
| 饱和控制 Train 状态附近的邻域 | 8 | 在原有困难点附近加密 |

位置偏差最大仍为 `±[0.6,1.0,0.6]m`，速度偏差 `±1.5m/s`，姿态旋转向量分量 `±0.25rad`，实际总速度限制 `8m/s`。
前 12 条固定同一个参考时刻，接着 12 条固定另一个时刻；邻域样本围绕同一路线中的两个原 Train 状态，各 4 条。
**只变无人机状态，不根据扰动重置路径时钟。每条都重新求 MPC 标签。**

邻域锚点按原 Train 标签的饱和控制分量数量选择；本轮没有使用 Validation/Test 错误来选点，也没有运行 DNN 闭环采集。
存档中的 300D 特征是 Z-up/FLU，采样时正确解码回 AirSim NED/FRD，包括位置、速度和四元数。

教师每次独立重置，仍要求 KKT ≤ 0.001、首步控制迭代差 ≤ 0.0001。不收敛就重新采样，同一条最多尝试 8 次；仍失败时保存 `failure.json` 并停止，不伪造标签。
每条路线第一条重新求一次标签检查重复一致性。数据审计只是数值与来源检查，不代表所有扰动状态都经过真实闭环恢复验证。

每完成 32 条就原子保存一个 chunk 和 manifest。中断后用同一命令恢复，跳过已提交 chunk；当前未提交路线最多重做 32 条。
源数据、教师 DLL 或采样代码变更会拒绝混合续跑，需要新输出目录。

## 四个结构和三组实验

四个结构统一 **4 个隐藏层**，宽度为 **64、128、256、1024**，全连接 ReLU，输出层线性。固定深度，避免把宽度与深度的影响混在一起。

| 组名 | 训练数据 | 训练规则 | 追加 epoch 上限 |
|---|---|---|---:|
| `old_control` | 原 Train | Adam，lr=1e-4，无 L2 | 420 |
| `added_only` | 合并 Train | 同上 | 300 |
| `added_regularized` | 合并 Train | Adam L2=1e-5；余弦 lr=1e-4→1e-5 | 300 |

三组都从同一批原 checkpoint 开始，不是从上一组训练完的模型继续训练。
每组 4 个任务，总计 12 个串行任务，每个任务独立子进程，不并行占显存。

batch=512 时，原数据每 epoch 150 次更新，新数据 210 次；`420×150 = 300×210 = 63,000`，匹配最大优化器更新预算。
最少 epoch 与 early-stopping patience 也按更新比例设为旧组 140/112、新组 100/80。early stopping 会使实际更新数不同，而且两组验证频率不同；这里匹配的是上限，不是保证实际计算量完全相同。

`old_control` vs `added_only` 比较补充数据的作用；`added_only` vs `added_regularized` 比较“L2+学习率衰减”这个组合的作用，不能单独归因于 L2 或衰减。
原组、新组的 Train MSE 覆盖不同样本分布，不能直接把它们的 Train/Validation 比值视为同一个指标。

## 从头执行的 PowerShell 命令

无需启动 UE4 或点击 Play。建议关闭 AirSim 和其他 GPU 训练程序。全部入口统一用已有 PyTorch 环境；数据查询在 CPU 上执行，训练才使用 CUDA。

### 1. 设置路径

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Test-Path -LiteralPath $trainPy
```

最后应输出 `True`。不要使用不带 `\.venv\` 的误写路径。

### 2. 补充数据、合并并审计

```powershell
& $trainPy -m controller_architecture_study.decision_recovery generate --resume
& $trainPy -m controller_architecture_study.decision_recovery prepare
```

第一条预期结束于 `added=30720/30720`、`Augmentation complete`。
第二条预期审计中有：`status=PASS`，`old_train_samples=76800`，`added_train_samples=30720`，`train_samples=107520`，Validation/normalization byte-identical 为 true，`test_read=false`。
如果第一条有异常，先不要执行第二条/训练；把错误与 `failure.json` 发回来。

### 3. 先运行数据对照：8 个串行任务

```powershell
& $trainPy -m controller_architecture_study.run_recovery_ablation --arm old_control --device cuda
& $trainPy -m controller_architecture_study.run_recovery_ablation --arm added_only --device cuda
```

每条命令自动按宽度 64、128、256、1024 运行四个任务。可以先完成这两组，发回汇总再决定是否有必要立即跑第三组。

### 4. 训练规则对照：另 4 个串行任务

```powershell
& $trainPy -m controller_architecture_study.run_recovery_ablation --arm added_regularized --device cuda
```

如果确定三组全部连续运行，也可以不用上面三条组命令，改为一条：

```powershell
& $trainPy -m controller_architecture_study.run_recovery_ablation --arm all --device cuda
```

不建议同时开启多个窗口执行这些训练命令。

### 5. 中断后恢复和查看结果

训练中断后，重跑原来的 `--arm ...` 命令即可：已完成模型跳过，未完成模型从最后保存的 epoch、优化器和随机状态恢复。
不需要额外 `--resume` 参数。学习率衰减按 epoch 推导，不会在续跑时重新从大 lr 开始。
Ctrl+C 或断电不会撤销已完成模型；未提交的当前 epoch 可能需要重做。
OOM/驱动错误会停止队列，前面的模型保留，不会尝试不安全的自动重启。当前使用串行执行，但不能保证硬件/驱动不会故障。

```powershell
& $trainPy -m controller_architecture_study.run_recovery_ablation --summarize-only
Get-Content -Raw -LiteralPath '.\controller_architecture_study\results\decision_recovery_v1\runs\comparison.json'
```

`--summarize-only` 不训练，只读取已有日志和结果。可以使用 `--dry-run` 查看将执行的任务；新数据组的 dry-run 也要求已完成 prepare。
不要在同一个结果目录改变 `--epochs`、源 checkpoint 或数据；会拒绝配置不一致的续跑。

## 输出位置

```text
controller_architecture_study/results/decision_recovery_v1/
  manifest.json
  chunks/envXX_routeYY.npz
  data_audit.json
  dnn_dataset/
    train.npz
    validation.npz
    normalization.json
    split_manifest.json
  runs/
    old_control/plain_mlp_h4_w{64,128,256,1024}_seed0/
    added_only/plain_mlp_h4_w{64,128,256,1024}_seed0/
    added_regularized/plain_mlp_h4_w{64,128,256,1024}_seed0/
    comparison.json
    comparison.csv
```

每个训练目录独立保存 `best_model.pt`、`last_model.pt`、`config.json`、`initial_metrics.json`、`offline_metrics.json`、`completion.json`、`train_log.csv`、`validation_curve.svg` 和 `complexity.json`。
regularized 组的 CSV 额外记录每个 epoch 的 learning_rate。
本阶段不自动导出 TorchScript/部署权重，不使用 Test 选模型。

新 config 的 `dataset_source_sha256` 仍表示原 96k 数据的基础来源；额外用 `augmentation_manifest_sha256` 和实际 Train/Validation/normalization 文件哈希记录补充来源。因为编码和归一化未变，后续可继续使用现有 `fly_decision_dnn` 入口和原路线，不需修改在线编码。

## 本地验证边界

- 25 项回归测试通过：模拟教师的数据生成/断点恢复/来源哈希、原文件保留、参考时钟固定、NED/FLU 邻域重建、四模型三组配置、串行失败后续跑，以及禁用优化器更新下的学习率/检查点恢复。
- 真实原 checkpoint 的四模型 dry-run 通过，不执行训练。
- 对一条真实 Train 路线的五种补充类别做了各一条真实 MPC 查询，均收敛，没有写入生产数据。
- 正式 30,720 条补充与 12 个训练任务尚未执行，由用户运行。

下一次请提供 `data_audit.json` 与 `runs/comparison.json`。如果只完成两组，也可以先发该汇总；不要仅发最后一个模型的末尾日志。
