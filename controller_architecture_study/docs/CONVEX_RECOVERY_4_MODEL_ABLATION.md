# 凸 MPC 恢复状态四模型对照

本轮依据 `CONVEX_15_MODEL_GAP_DIAGNOSIS.md` 的诊断，验证输入冗余和独立恢复状态覆盖。教师不变：已试飞的凸平移 QP + 姿态反馈，名义速度 5 m/s、控制周期 50 ms、预测周期 100 ms、20 步未来窗口、全局 C2 轨迹时钟。本轮不重做教师试飞，不修改旧数据/模型，不做量化或 FPGA 编译。

## 四个结构、两组数据

使用 Plain MLP、ReLU、线性 4D 输出，**4 个隐藏层**，宽度分别为 **128、256、512、1024**。固定隐藏层数，只比较宽度，避免把宽度和深度的优化影响混在一起。四个结构各跑两组，共八次独立训练：

| 实验组 | Train | Validation | Test | 最多 epoch/模型 | 最多参数更新/模型 |
|---|---:|---:|---:|---:|---:|
| features_only | 76,800 | 9,600 | 9,600 | 400 | 60,000 |
| features_recovery | 153,600 | 9,600 | 9,600 | 200 | 60,000 |

两组从头训练、seed=0、batch=512、Adam、初始 lr=1e-4、cosine 到 1e-5、weight decay=1e-5、梯度裁剪 1。相同最大更新次数，数据更多的组 epoch 减半；不是“数据更多又额外多训练一倍”。本轮固定预算，不提前结束，但每 epoch 都用 Validation 选择/保存 best，并保存 last 以恢复训练。两组 epoch 粒度的学习率阶梯略有不同，不表示每个更新的样本和学习率都逐项相同。

原 300D 的已有 `4×256、4×512、4×1024` 指标作为参考读入报告；没有旧 `4×128` 时标记 `not_available`，不会伪造基线，也不会再训练全部旧模型。单 seed 下八次训练是定位实验，不足以宣称普遍、严格单调的容量规律。

## 输入和数据构建

输入从 300D 投影到 **237D**：删除局部当前位置的 3 个常量零，以及 20×3 个规划器参考角速度。全部保留索引写进 `normalization.json` 和 `split_manifest.json`，在线和离线调用同一个转换。保持剩余坐标顺序、单位和标签不变。

这一实验删除的是教师对其不直接敏感的坐标，不是 PCA，也不是已经证明的最小充分表示。旧窗口没有直接序列化全部参考加速度向量，网络仍需要从位置/速度窗口间接学习轨迹变化；本轮不把“237D 已解决所有表示问题”当作事实。

追加数据仅来自原 **Train 环境 0～47 的 960 条冻结路线**。每条路线独立新增 80 个状态，复用原采样函数的范围，不扩大极端状态、不按验证失败样本挑选：

- 位置恢复 16 条，合计 15,360；
- 速度恢复 16 条，合计 15,360；
- 姿态恢复 8 条，合计 7,680；
- 位置+速度+姿态组合恢复 40 条，合计 38,400。

每个状态重新调用同一 QP 求标签，检查约束/驻点残差并有限重复求解；不复用原标签，不接受失败求解，不反复抽样掩盖失败。不同随机流在每条 Train 路线上另外生成每族一条，共 **7,680 个新状态探针**，不参加优化和 best 选择，只用于训练后的状态泛化审计。该探针仍是 Train 环境，不代替 held-out 验证。

原 Train 76,800 条始终保留；两组同时保存原 Train 固定审计集。扩充组的训练分布有意增加恢复状态权重，不能将它的混合 Train MSE 与原 Train MSE 直接混为一种总体指标。

**原 Validation/Test 的环境、路线、行顺序、所有标签和非 X 元数据不变**；仅将 X 按相同索引投影，故其 NPZ 文件不会和 300D 原文件字节相同。两组 237D 的验证文件、测试文件和审计文件哈希必须一致。Test 只在准备阶段作确定性投影、保存和核对来源，没有计算测试损失，不用于本轮选择。

两组都冻结原 Train 的归一化：X 均值/标准差取原 300D 对应 237 个坐标；U 均值/标准差原样保留。固定原 Train 子集是合法的 Train-only 归一化来源，也让 normalized MSE 分母与旧结果一致；不因扩充而改变输出尺度来人为降低 loss。

## 一次复制完整运行

不需要启动 UE4/AirSim。使用带 SciPy/Torch 的训练环境，不能把另一个 AirSim 环境的整个 site-packages 放进 PYTHONPATH。

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $proj = 'F:\FalconReproduction\airsim_control'
    $trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
    Set-Location -LiteralPath $proj
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $trainPy)) { throw "找不到训练环境：$trainPy" }
    & $trainPy -m controller_architecture_study.convex_recovery generate --resume
    if ($LASTEXITCODE -ne 0) { throw '恢复状态采集失败；保留已完成分块，修复后重跑本段。' }
    & $trainPy -m controller_architecture_study.convex_recovery prepare
    if ($LASTEXITCODE -ne 0) { throw '数据准备/审计失败，未开始训练。' }
    & $trainPy -m controller_architecture_study.run_convex_recovery --epochs 400 --device cuda
    if ($LASTEXITCODE -ne 0) { throw '训练队列停止；已完成模型和已保存 epoch 保留，修复后重跑本段。' }
    $report = '.\controller_architecture_study\results\decision_convex_recovery_v1\runs\comparison.json'
    $result = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    $result.models | ForEach-Object {
        [pscustomobject]@{
            arm = $_.arm
            model = $_.model
            status = $_.status
            epochs = $_.epochs
            fixed_train_MSE = '{0:E4}' -f $_.fixed_train_mse
            fresh_train_MSE = '{0:E4}' -f $_.fresh_train_mse
            validation_MSE = '{0:E4}' -f $_.validation_mse
            gap_absolute = '{0:E4}' -f $_.gap_absolute
        }
    } | Format-Table -AutoSize
    Write-Host "比较报告：$proj\controller_architecture_study\results\decision_convex_recovery_v1\runs\comparison.json"
}
```

## 保存、恢复及结果解释

全部新产物位于 `controller_architecture_study/results/decision_convex_recovery_v1`：

- `chunks/`、`manifest.json`：按路线/用途保存采集分块及哈希。断电时最多重做当前未提交分块；后续 `--resume` 跳过已完成分块，不是将全部数据重新生成。
- `datasets/features_only/`、`datasets/features_recovery/`：各自的 train/validation/test、归一化、拆分清单；`audit/` 是原 Train，`probe/` 是独立新状态。
- `data_audit.json`：样本数量、输入索引及冻结验证检查结果。
- `runs/<arm>/plain_mlp_h4_w<width>_seed0/`：每个模型独立保存 `best_model.pt`、`last_model.pt`、配置、日志、离线指标、固定状态审计及完成标记。每 epoch 原子保存；故障后不必等其他模型完成才能保住权重。
- `runs/comparison.json` 和 `.csv`：包含八次运行、原 300D 参考、逐状态族及 p95/p99 物理误差。本轮 `test_evaluated=false`。

直接重跑上述整段命令即可继续：已提交分块不求解，已完成模型不再训练，未完成模型从最后成功保存的 epoch 恢复；当前未保存 epoch 可能重跑。保持代码、路径和训练配置不变；若修改预算或配方，应使用新的实验目录，不能静默覆盖旧结果。单 GPU 串行子进程释放训练显存；CPU 后审计不会常驻另一套 GPU 模型。OOM 不会删除已经完成的模型，但无法保证任意硬件上的显存需求。

主要看固定 Validation 的绝对 MSE、组合/速度恢复误差、原 Train 到新状态探针的退化幅度；`gap_absolute` 是 **Validation − 固定原 Train**，而不是 Validation 减去困难样本权重变大的训练混合集。gap 比值变小但 Validation 变大不算改善，四模型不严格单调也不能篡改报告。若恢复数据有效，再展开多个 seed 或单独优化预算对照；不要一开始重复整个十五模型网格。

## 后续试飞接口

训练完成后仍用已有 `fly_convex_policy`；它根据 checkpoint 配置自动加载对应 237D 数据归一化和投影。`--output` 保持指向原冻结路线根目录 `decision_convex_96k_v1`，**不要改为新恢复数据目录**。例如以下是后续可选操作，不包含在本次训练流水线内，需要手动打开 UE4 并点击 Play：

```powershell
& 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe' -m controller_architecture_study.fly_convex_policy --checkpoint .\controller_architecture_study\results\decision_convex_recovery_v1\runs\features_recovery\plain_mlp_h4_w1024_seed0\best_model.pt --environment-id 48 --route-index 3 --device cpu --keep-gates
```

飞行日志保留投影前 300D 原始窗口，模型推理实际使用 237D；报告明确写 `convex_237D_DNN_only`。轨迹时钟、第一步控制执行和原 300D 模型入口保持不变。离线改善不等价于已经证明恢复状态闭环稳定。

当前交付：代码与有限自动测试已完成；正式追加采集、八次训练和新模型试飞尚未运行，由用户执行。
