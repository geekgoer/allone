# 96k / 300D DNN 容量对照实验

## 目标与范围

试飞正常后，在同一份 `decision_nmpc_96k_v2` 数据上比较不同容量的 DNN。
这仍是 ACADO 教师的 300D imitation 分支，不是原 FALCON 的 12D 分支。
不生成新教师数据、不重新拆分、不修改归一化、不调用 AirSim、不读取 Test 选择结构。
也不预设大模型一定更好：本轮检验容量与误差的关系，而不是人为制造单调曲线。

`h` 表示隐藏层数量，不包含输出层；`w` 表示每个隐藏层的宽度。
网络为 `300 -> h 个 Linear+ReLU 隐藏层 -> Linear(4)`，不使用残差连接。

| 对照 | 结构 |
|---|---|
| 固定 h=4，改变宽度 | 4×64、4×128、4×256、4×512、4×1024 |
| 固定 w=256，改变深度 | 2×256、4×256、8×256、12×256 |
| 保留已有较宽深层点 | 8×512 |

去重后共 9 个结构，seed=0。已经完成的 4×128、8×256、8×512 直接复用，只新增 6 个训练任务。
已有目录配置必须完全一致才可复用，否则入口报错，不覆盖原模型。

## 完整训练命令

训练时不需要 UE4/Play，建议关闭 UE4 以避免争用 GPU。不要同时启动其他训练进程。
在 PowerShell 完整复制以下内容，不要复制命令提示符：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

& $trainPy -m controller_architecture_study.train_decision_pilot --dataset-version 96k --grid capacity --device cuda
```

串行运行，一个模型一个子进程。沿用三个已有模型的相同训练规则：

- Adam，学习率 1e-4，batch=512，梯度裁剪 1。
- 最多 400 epoch，至少 100 epoch；验证误差连续 80 epoch 未改善时提前停止。
- 训练集 76800、验证集 9600、测试集 9600；按原环境组划分，归一化只来自 Train。
- 每个完整 epoch 原子保存 `last_model.pt`；验证最优时保存 `best_model.pt`。
- 每个模型完成后立即保存离线评估、计算量和完成标记，再进入下一个模型。
- 异常会停止队列，不删除已完成模型；当前未完成 epoch 可能需要重算。

中断、关机或报错修复后，重新执行同一条训练命令即可：跳过已完成模型，从未完成模型的已保存 epoch 续训。
若需要先看队列而不训练，可选：

```powershell
& $trainPy -m controller_architecture_study.train_decision_pilot --dataset-version 96k --grid capacity --device cuda --dry-run
```

该预览不会启动训练或写入结果。

## 输出与查看结果

输出根目录：

`F:\FalconReproduction\airsim_control\controller_architecture_study\results\decision_nmpc_96k_v2\runs`

每个结构独立目录，如 `plain_mlp_h4_w1024_seed0`，包含：

- `config.json`、`last_model.pt`、`best_model.pt`。
- `train_log.csv`、`validation_curve.svg`。
- `offline_metrics.json`：Train/Validation 总体、分输出、分状态类别指标。
- `complexity.json`、`completion.json`。

九模型汇总为 `comparison_capacity.json`，每完成一个模型更新一次。
原三模型 `comparison.json` 保留不变。以下命令可在训练结束后查看，也可在训练期间读取已完成部分：

```powershell
$reportPath = '.\controller_architecture_study\results\decision_nmpc_96k_v2\runs\comparison_capacity.json'
$comparison = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
$comparison.models | ForEach-Object {
    [pscustomobject]@{
        model = $_.model
        parameters = $_.complexity.parameter_count
        MACs = $_.complexity.mac_count
        weights_MiB = [Math]::Round($_.complexity.fp32_weight_bytes / 1MB, 3)
        epochs = $_.epochs_completed
        best_epoch = $_.best_epoch
        train_MSE = $_.train.normalized_mse
        validation_MSE = $_.validation.normalized_mse
        validation_physical_MSE = $_.validation.physical_mse
        gap_ratio = $_.validation.normalized_mse / [Math]::Max($_.train.normalized_mse, 1e-15)
    }
} | Sort-Object parameters | Format-Table -AutoSize
```

batch=1 MAC 数为 `300*w + (h-1)*w*w + 4*w`；参数量再加 `h*w+4` 个偏置。
`fp32_weight_bytes` 为参数量×4，只有参数，不包含优化器状态、激活、工作区或文件封装。
MAC/参数量是理论计算量，不能直接当作 GPU、CPU 或 DPU 的实测时延。

## 训练之后

先把上面的简表发回来，重点看固定深度/固定宽度两条曲线，以及跟踪/恢复类别是否表现不同。
若最佳 epoch 接近 400 上限，先判断是否尚未收敛，不能直接认定容量不足。
单 seed 结果只用于第一轮筛选，必要时再对候选做多 seed 复验。
随后对候选统一测 batch=1 推理时延、在相同 Validation 路线上试飞，形成误差–实测时延前沿；最后锁定选择再做 Test 验收。
不要仅凭一次试飞或总体 MSE 认定全部恢复状态安全，也不要将本轮 300D 与旧 12D 标准化 MSE 直接横向比较。

新模型可直接使用已有 300D 试飞入口，例如：

```powershell
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
& $trainPy -m controller_architecture_study.fly_decision_dnn --checkpoint .\controller_architecture_study\results\decision_nmpc_96k_v2\runs\plain_mlp_h4_w1024_seed0\best_model.pt --environment-id 48 --route-index 3 --device cpu --keep-gates
```

试飞前仍需手动启动 AirSim 并点击 Play。再次训练前移除 `PYTHONPATH`，如上面的完整训练命令所示。
