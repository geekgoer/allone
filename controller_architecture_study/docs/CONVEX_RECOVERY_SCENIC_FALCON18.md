# 冻结237D恢复数据：SCENIC 12＋FALCON 6结构训练

日期：2026-10-03。代码已交付；耗时训练由用户运行。

## 本轮实验

沿用 `controller_architecture_study/results/decision_convex_recovery_v1/datasets/features_recovery`，不重新生成、不重新拆分、不修改归一化。Train为153600条（原76800＋恢复76800），Validation为9600条，Test为9600条。本轮不读取Test样本、不用Test选模型。

18个网络均为237维输入、4维输出、Linear＋ReLU、线性输出层，从头初始化；不从既有200轮模型接着补到600轮。上一轮模型和报告完整保留。当前是**同一任务上的论文结构对照**，不是原论文12维任务的严格复现，不保证loss随参数量单调下降。

SCENIC使用已修正的12个结构，隐藏全连接层数h与论文层数L满足L=2h+1：

| h | 宽度 |
|---|---|
| 1 | 1024 |
| 2 | 128、1024 |
| 3 | 128、256、1024 |
| 4 | 256、512、1024 |
| 5 | 512、1024 |
| 6 | 1024 |

FALCON为depth=10/22/34、width=512/1024，隐藏层h=9/21/33；depth包含输出全连接层。之前4×128属于额外对照，不计入这18个。

统一配方：seed=0、FP32、Adam、batch=512、学习率1e-4余弦降至1e-5、weight_decay=1e-5、梯度裁剪1。**每个模型实际训练600轮、180000次优化器更新，不折半、不提前停止**。每轮只用Validation选择best_model。相同预算不等于相同耗时，也不意味着每个结构必然收敛；报告保留最后100轮改善与梯度/裁剪统计，再决定是否做独立的追加预算实验。

按参数量从小到大串行运行，每个模型独立子进程。每轮原子保存`last_model.pt`（模型、Adam状态、随机数状态和历史），改善时保存`best_model.pt`。中断后重跑同一命令：完成模型跳过，未完成模型接着最后已保存epoch运行；未保存的当前epoch重做。不要同时启动两份队列，也不要在续训时改变epochs/seed/结构/训练源码。

每个完成模型另外用CPU评估原Train子集和独立新状态探针，避免训练队列父进程持有CUDA模型。审计可能比一次推理耗时长；此时不是卡住，也不在重新训练。

## 整段命令

不需要启动AirSim。建议先关闭UE4及其他GPU训练程序。整段复制到PowerShell；重启/中断后仍用原段，不要换输出目录。

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $proj = 'F:\FalconReproduction\airsim_control'
    $trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
    Set-Location -LiteralPath $proj
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $trainPy)) { throw "找不到训练环境：$trainPy" }
    & $trainPy -m controller_architecture_study.run_convex_scenic_falcon18 --epochs 600 --seeds 0 --device cuda
    if ($LASTEXITCODE -ne 0) { throw '队列停止；检查错误后重跑本段，已保存模型及epoch会保留。' }
    $reportPath = '.\controller_architecture_study\results\decision_convex_recovery_scenic_falcon18_v1\comparison.json'
    $result = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $result.models | ForEach-Object {
        [pscustomobject]@{
            model = $_.model
            group = $_.architecture_group
            status = $_.status
            parameters = $_.parameter_count
            epochs = $_.epochs
            best_epoch = $_.best_epoch
            train = '{0:E5}' -f $_.train_mse
            fixed_train = '{0:E5}' -f $_.fixed_train_mse
            fresh_train = '{0:E5}' -f $_.fresh_train_mse
            validation = '{0:E5}' -f $_.validation_mse
            val_fresh_ratio = '{0:F3}' -f $_.validation_fresh_ratio
        }
    } | Format-Table -AutoSize
    Write-Host "完成：$($result.runs_completed) / $($result.runs_requested)"
    Write-Host "报告：$proj\controller_architecture_study\results\decision_convex_recovery_scenic_falcon18_v1\comparison.json"
}
```

## 产物与读法

新根目录：`controller_architecture_study/results/decision_convex_recovery_scenic_falcon18_v1/`。

- `runs/plain_mlp_h<h>_w<width>_seed0/`：best/last检查点、配置、逐轮CSV、Validation曲线SVG、离线指标、复杂度、完成标记、固定状态审计。已有试飞入口可以加载其中的`best_model.pt`，无需更改237D输入合同。
- `comparison.json`和`comparison.csv`：每完成/中断一个模型更新；缺失结果标为not_started/in_progress，而不是填成零。训练完成但审计中断标为trained_pending_audit，重跑补审计，不重训。
- `job_configs/`：18个独立配置。配置包含数据、归一化、结构目录和训练实现哈希；发现变化拒绝覆盖旧运行。

所有MSE均使用冻结的输出标准化尺度。`train_mse`是153600条实际混合Train在best模型下的评估值，不是逐batch训练日志均值；`fixed_train_mse`是原76800条，`fresh_train_mse`是7680条未训练过但来自Train路线的新状态。Validation来自独立环境。分别看Validation/混合Train、Validation/原Train、Validation/fresh探针，不能把后者接近1说成训练过拟合消失。JSON保留各输出、各状态族和P99误差。

当前先跑seed=0。完成后用Validation分析同宽度深度变化、同深度宽度变化及收敛情况，再挑代表结构补seed；不从单seed排名直接推出稳定结论，不强求深层网络一定胜过浅层网络。本轮不做量化、DPU编译或超网训练。

只查看队列（不训练、不创建结果目录）：在上述相同环境下运行`python -m controller_architecture_study.run_convex_scenic_falcon18 --epochs 600 --seeds 0 --dry-run`。只重建已有结果汇总可用`--summarize-only`；它不触发训练或补审计。

## 交付验证

50项自动测试通过（新增5项＋相关45项）：结构及层数、实际600轮预算、数据区保护、失败停止、已完成跳过、自动续训、缺失结果不填零、三类gap、既有237D检查点与在线输入兼容。涉及训练/飞行的自动测试使用临时微型数据或模拟客户端，不是正式训练或真实飞行证据。

已在正式数据上运行18模型`--dry-run`，数据/教师/归一化哈希验证通过，所有新运行为`new`、actual_epochs=600、updates=180000。预检未创建正式结果目录；未启动正式训练。整段PowerShell命令经语法解析通过，但未执行其训练部分。
