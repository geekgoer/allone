# 二维增量超网：M1 运行说明

2026-10-12。代码已实现；正式训练、量化、PL 和 AirSim 接管尚未执行。

一个共享 FP32 网络，17 个深度 `[1,3,...,33]` × 8 个宽度 `[128,256,...,1024]`。
输入继续使用已冻结的 237D 契约；输出是原 MPC 的 4D 物理控制量。训练直接监督原 MPC 标签，5000 轮密集 DNN 仅用于初始化保留连接和辅助蒸馏。

## 直接执行

关闭 AirSim、其他训练和占用 GPU 的程序；接通电源并保持稳定散热。下面整段可复制到 PowerShell。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_incremental_supernet.ps1 -Python $trainPy -Device cuda -BatchSize 512
if ($LASTEXITCODE -ne 0) { throw '运行未完成；已保存的检查点保留，请查看报错后恢复' }
```

程序在当前终端运行，不开后台任务，也不启动 AirSim。默认固定 420 epoch：最大因果路径适配 100、新 head 训练 20、二维联合训练 300；不早停，不重新划分数据，不读取 test。每个 epoch 评估 136 个配置，训练每个 batch 仅监督最多 6 个配置，共用一次最大图前向。

中断后**重跑同一段命令**。每个完整 epoch 原子保存 `latest_checkpoint.pt`，恢复 optimizer、阶段/cosine 时钟、随机数和二维采样器；未完成的 epoch 重跑。阶段切换时按计划重建优化器，先用适配阶段的最佳最大路径，再训练 heads/joint。

若明确是 `CUDA out of memory`，把 `-BatchSize 512` 改成 `-BatchSize 256` 后恢复。batch 变更写入记录，不保证改变 batch 后与原轨迹逐位相同。驱动黑屏、TDR、illegal memory access 等不是普通 OOM，不应靠不停重试继续。脚本不更新 PyTorch/驱动，不改 TDR 设置。

没有实测此完整训练的 GPU 速度，不承诺耗时。每 100 batch 有进度输出，每 epoch 输出平均配置/最大配置验证 MSE 和 CUDA 峰值分配显存；不需要另开监控脚本。

## 输出

全部新文件位于：

```text
controller_architecture_study/results/incremental_supernet_237d_v2_causal_width/
  config.json
  teacher_predictions.npz       # train/validation 上冻结教师输出，附 hashes
  initial_metrics.json         # 记录加 mask 后与原教师的区别
  latest_checkpoint.pt         # 完整状态，继续训练用
  adapt_best_model.pt           # 阶段一最佳最大路径
  best_model.pt                 # 仅从联合训练中选择的一份共享模型
  last_model.pt                 # 固定预算最后一份共享模型
  history.json
  validation_grid_latest.json
  completion.json
  export/
    controller_scripted.pt     # 单个动态 depth/width TorchScript
    weights_fp32.npz            # 仅有效下三角块和共享 heads，非 136 份权重
    normalization.json
    model_spec.json
    golden_vectors.npz         # 原始/标准化输入，136 配置输出，NumPy 参考
    offline_metrics.json        # 每配置 train/validation、family/通道与尾部误差
    export_report.json          # hashes、数值检查和最大路径质量目标结果
```

训练结束后脚本自动导出。输出 `PASS` 表示导出数值一致，不表示飞行通过或越大一定越好。

`best_model.pt` 选模规则预先固定为：优先最大配置 validation MSE ≤ 冻结教师的 1.05 倍，再最小化 136 配置的等权平均 validation MSE。如果没有配置集合达到最大路径目标，保存实际最佳模型并报告 `maximum_quality_goal_met=false`，不隐藏退化。不同 epoch 的最佳 heads 不拼接到不同骨干上。

若训练已完成、导出中断，可只执行：

```powershell
& $trainPy -m controller_architecture_study.incremental_supernet.export --device cuda
if ($LASTEXITCODE -ne 0) { throw '导出未完成；训练检查点仍保留' }
```

## 推理接口与边界

`controller_scripted.pt` 接收 **原始未标准化**的 `(batch,237)`，返回反标准化后的 `(batch,4)`：

```python
controller = torch.jit.load(".../export/controller_scripted.pt")
physical_u = controller(raw_features, 9, 384)
```

不要再对该 wrapper 的输入/输出重复归一化。它支持动态配置，但不是增量缓存接口；当前既有 flight 入口也不会自动识别新的 checkpoint。先完成 M1，再在 M4 接入此契约，不把该模型直接替换进旧 12D/25D 路径。

需要同一采样的续算时，使用 `load_supernet` 得到 `eval()` 模型，将输入标准化**一次**，创建 `IncrementalContext`；例如：

```python
context = IncrementalContext(model, normalized_x, sample_id="Drone1:100", model_version=checkpoint_sha256)
u_norm, new_nodes = context.grow(9, 128, sample_id="Drone1:100", model_version=checkpoint_sha256)
u_norm, new_nodes = context.grow(9, 256, sample_id="Drone1:100", model_version=checkpoint_sha256)
u_norm, new_nodes = context.grow(17, 256, sample_id="Drone1:100", model_version=checkpoint_sha256)
physical_u = u_norm * u_std + u_mean
```

这复用前面所有层的旧组激活，新增宽度要补算新组的整个前缀。上下文只属于同一输入/固定模型；新采样须新建。输入或权重就地变更会拒绝旧上下文。本次未实现量化、网络协议、硬件抢占或完整控制调度器。

训练采用 dense 矩阵乘固定 mask，实际分配约 33.90M 参数，有效参数约 19.22M。GPU 不会因为零块自动获得理论稀疏加速。FP32 NPZ 只打包有效块；17×8 是候选数，不保证 136 个有用精度台阶。

## 已完成的必要检查

- 6 项 CPU 单元检查：因果前缀及零连接梯度、缓存两种顺序与过期拒绝、动态 TorchScript、非等价教师初始化、网格采样/一致性门控、三阶段小型中断恢复和 NumPy 导出。
- 小型恢复检查在相同 batch/CPU 条件下，恢复后最终权重与不中断训练逐位一致。
- 实际 237D/33×1024：hash 检查、136 输出有限、选定配置与共享前向一致、增量两种顺序一致、TorchScript 一致；仅查询 2 条 validation 输入，没有正式训练或试飞。
- 报告：`controller_architecture_study/results/incremental_supernet_checks/real_237d_check.json`。

如需重跑短检查：`python -m unittest controller_architecture_study.incremental_supernet.test_supernet -v`；真实尺寸检查为 `python -m controller_architecture_study.incremental_supernet.check`。它们不是正式训练前的额外大规模实验。

下一阶段：根据此次共享模型输出进入 M2 的 W8A16 数值规范/QAT，以及 M3 定制 PL 执行器。M1 数值 API 正确不替代之后的控制与时效验收。
