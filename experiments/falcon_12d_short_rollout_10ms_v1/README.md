# FALCON 12-D：10 m/s 五步短 rollout 数据实验

这是隔离的鲁棒性实验，不覆盖现有模型或论文复现目录。

每个随机初态连续生成5条数据。在每个50 ms控制周期中：

1. 从当前真实/传播状态重新生成未来20步参考；
2. 重新调用 ACADO/qpOASES MPC；
3. 只保存第一步四维控制 `u0`；
4. 用零阶保持控制和四阶RK动力学推进50 ms；
5. 在新状态重复以上过程。

因此网络契约仍为12输入、4输出，而不是一次输出5步控制。一个rollout的5行共享同一个 `query_group_id`，以后拆分数据时必须整段进入同一个train/validation/test集合。

目标速度固定为10 m/s。高速度是为了研究10 m/s闭环能力，不是为了美化MSE。物理MSE会随标签尺度改变，跨速度比较应使用normalized MSE、每输出NRMSE和闭环指标。

控制分布设有硬上限：

- saturated不超过3%；
- near-limit不超过30%；
- 若候选rollout导致全局预算超限，它会被拒绝并重采样；
- 最终合并后会再次审计，超限直接失败。

## 第一步：只运行1000条冒烟采集

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py .\experiments\falcon_12d_short_rollout_10ms_v1\generate_short_rollouts.py --config .\experiments\falcon_12d_short_rollout_10ms_v1\short_rollout_smoke_config.json --resume
```

成功时最后一行会显示 `X=(1000,12)`、`U=(1000,4)`、near-limit比例和saturated比例。先检查：

```powershell
Get-Content -Raw .\experiments\falcon_12d_short_rollout_10ms_v1\teacher_short_rollout_10ms_smoke\manifest.json
```

## 第二步：冒烟结果合理后生成120,000条

```powershell
& $py .\experiments\falcon_12d_short_rollout_10ms_v1\generate_short_rollouts.py --config .\experiments\falcon_12d_short_rollout_10ms_v1\short_rollout_120k_config.json --resume
```

该命令支持按chunk续跑。完成后先运行审计：

```powershell
& $py .\experiments\falcon_12d_short_rollout_10ms_v1\audit_short_rollout_dataset.py
```

审计通过后，再按完整五步rollout做80/10/10划分和训练集归一化：

```powershell
& $py .\experiments\falcon_12d_short_rollout_10ms_v1\prepare_short_rollout_dataset.py
```

这一步生成：

```text
dnn_dataset_short_rollout_10ms_120k/train.npz
dnn_dataset_short_rollout_10ms_120k/validation.npz
dnn_dataset_short_rollout_10ms_120k/test.npz
dnn_dataset_short_rollout_10ms_120k/normalization.json
dnn_dataset_short_rollout_10ms_120k/split_manifest.json
```

## 3. Capacity pilot, then the six models

The data has a fixed target/reference speed of 10 m/s.  Each model runs for at
most 800 epochs. Training uses the
normalized control MSE as its selection metric because that is also the
optimization objective; physical-unit MSE is still reported for validation and
the untouched test set.  All runs are serial and resumable.

First train only the smallest and largest architectures:

```powershell
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
& $trainPy .\experiments\falcon_12d_short_rollout_10ms_v1\train_six_models.py --mode pilot --seeds 0 --device cuda --batch-size 512
```

After inspecting both `metrics.json` files, train the remaining architectures
(the already-complete pilot models are skipped):

```powershell
& $trainPy .\experiments\falcon_12d_short_rollout_10ms_v1\train_six_models.py --mode full --seeds 0 --device cuda --batch-size 512
```

The trainer writes a checkpoint after every epoch.  If Windows, CUDA, or the
terminal interrupts a run, rerun the identical command: completed runs are
skipped and the one in progress resumes from `latest_checkpoint.pt`.

暂时不要训练；把审计JSON和split manifest的输出交给我确认后，再配置六模型训练。
