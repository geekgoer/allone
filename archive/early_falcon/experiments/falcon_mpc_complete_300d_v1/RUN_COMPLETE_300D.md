# MPC-complete 300-D 数据、拟合与训练流程

这套流程拟合的是同一个确定性映射：

```text
z = 当前状态
  + 21 个参考状态
  + 20 个由轨迹规划器生成的参考控制

u = ACADO/MPC 在 z 上求解得到的第一个控制量
```

参考控制是轨迹规划器根据参考加速度和参考角速度生成的前馈量，不是 MPC 的求解结果。因此 DNN 部署时可以由状态估计器和轨迹规划器重新构造输入，不需要在线运行 MPC。

## 1. 先做合同和教师检查

在项目根目录执行：

```powershell
& $py .\experiments\falcon_mpc_complete_300d_v1\smoke_test_300d.py
& $py .\experiments\falcon_mpc_complete_300d_v1\smoke_test_300d.py --teacher --report .\experiments\falcon_mpc_complete_300d_v1\smoke_teacher.json
& $py .\experiments\falcon_mpc_complete_300d_v1\preflight_300d.py
```

`preflight_300d.py` 通过后才允许正式采集。

## 2. 生成 300-D MPC 教师数据

这一步不需要打开 AirSim，因为它使用已经保存的 40 个环境轨迹，并离线调用 ACADO 教师：

```powershell
& $py .\experiments\falcon_mpc_complete_300d_v1\collect_300d.py --resume
```

中断后仍然使用同一条命令。完成标志是：

```text
300-D dataset complete: X=(140000, 300) U=(140000, 4)
```

## 3. 构造训练切分

当前推荐的是“每个环境内部随机切分”，而不是把前几个环境给 train、后几个环境给 validation/test：

```powershell
& $trainPy .\experiments\falcon_mpc_complete_300d_v1\prepare_training_dataset.py
```

输出目录：

```text
F:\FalconReproduction\airsim_control\experiments\falcon_mpc_complete_300d_v1\dnn_dataset_300d_140k_v1_within_environment
```

该切分满足：

- train、validation、test 都包含全部 40 个环境；
- 每个环境的样本数量相同；
- 同一个五步 rollout 不会被拆到不同 split；
- normalization 只使用 train 计算；
- 输入仍然是原始 300D MPC 求解输入。

## 4. 先打印六条训练命令

```powershell
& $trainPy .\experiments\falcon_mpc_complete_300d_v1\train_six_models_complete_300d.py `
  --epochs 2000 `
  --batch-size 512 `
  --learning-rate 1e-4 `
  --device cuda `
  --dry-run
```

## 5. 正式训练六个 DNN

确认 dry-run 的数据目录和输出目录正确后执行：

```powershell
& $trainPy .\experiments\falcon_mpc_complete_300d_v1\train_six_models_complete_300d.py `
  --epochs 2000 `
  --batch-size 512 `
  --learning-rate 1e-4 `
  --device cuda
```

六个模型按复杂度递增顺序串行训练：

```text
depth10_width512
depth10_width1024
depth22_width512
depth22_width1024
depth34_width512
depth34_width1024
```

训练策略固定为：

- Adam；
- 学习率始终为 `1e-4`；
- 不早停；
- 不降低学习率；
- 每个模型训练到指定的总 epoch；
- batch size 默认 `512`；
- seed 默认 `0`；
- 一个模型完成后才开始下一个模型。

训练输出：

```text
F:\FalconReproduction\airsim_control\experiments\falcon_mpc_complete_300d_v1\runs_complete_300d_raw_seed0
```

如果中断，重新执行同一条训练命令即可从对应模型的 `latest_checkpoint.pt` 继续；已经完成的模型会自动跳过。

## 6. 实际部署时输入从哪里来

每个控制周期由现有状态估计器提供当前状态，由轨迹规划器提供未来参考状态和参考加速度/角速度，再调用 `build_mpc_complete_300d()` 生成 300D 输入：

```python
from falcon_paper_fidelity.mpc_complete_300d import build_mpc_complete_300d

feature = build_mpc_complete_300d(
    current_state,
    reference_window,
    gravity=9.8066,
)
```

然后使用训练时的 `normalization.json` 标准化，输入 DNN，最后用同一个 `u_mean/u_std` 反标准化输出。

如果部署系统没有独立的轨迹规划器，就不能使用这套 300D 合同；那不是训练问题，而是在线输入缺失。此时必须先让轨迹规划器在没有 MPC 的情况下生成参考窗口。
