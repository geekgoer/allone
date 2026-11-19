# MPC 姿态增益、控制周期与命令延迟对照

本实验用同一个 MPC 教师比较增益 3.5、7、10。每个释放时刻读取状态，教师根据状态和参考轨迹输出推力与三轴角速度。三组共用同一个平动 QP、预测步长、推力限制、角速度限制以及 AirSim 控制接口，仅更改姿态反馈增益。没有开启角速度前馈。实验用 CPU 教师直接闭环。

## 设计

- 单机 Drone1。使用已有的 3.0 m 净门宽自然路线，从参考初态开始，观察前 7.2 秒，覆盖第 4 门。所有组使用相同路线与初态，无外加飞行扰动。
- 核心矩阵为增益 3 档、周期 30／60／120 ms、延迟 9／18／24 ms，共 27 组。周期控制相邻任务的计划释放间隔，延迟控制每次释放后旧命令继续保持多久。固定其中一个变量即可看另一个变量的影响。
- 在周期 120 ms 增加延迟 54／99 ms，共 6 组。较长延迟只用于压力对照，且每档的计划延迟都小于周期。
- 记录第 4 门穿越余量、共同观测窗口的时间加权 RMSE、峰值误差与角速度限幅比例。AirSim 另保存碰撞时间及对象、过门覆盖、每次实际采样和命令发送时的模拟时间、目标延迟兑现情况。提前碰撞而未走完整窗口时，完整窗口 RMSE 留空。
- AirSim 在求解期间暂停，旧控制在模拟时间中继续作用至设定的发送时刻。命令发送时的物理状态是离散步进观测，内环仍按其自身节拍执行。该设计控制的是物理轨迹看到的命令延迟，Windows 实时调度和 DPU 端到端延迟不在这组实验中。

## 离线结果

下表为简化动力学的第 4 门几何余量。正数表示穿越门平面时保留的身体净空，负数表示超出预留身体范围。离线轨迹没有 AirSim 碰撞检测。

| 增益 | 固定 R=9 ms，T=30 / 60 / 120 ms | 固定 T=120 ms，R=9 / 24 / 54 / 99 ms |
| ---: | --- | --- |
| 3.5 | +9.5 / +6.4 / +2.3 cm | +2.3 / +0.4 / −2.4 / −3.2 cm |
| 7 | +78.6 / +78.7 / +79.8 cm | +79.8 / +81.2 / +85.9 / +67.8 cm |
| 10 | +92.4 / +92.7 / +93.5 cm | +93.5 / +94.5 / +96.1 / +29.6 cm |

固定短延迟时，增益 3.5 缩短周期保留了更多第 4 门余量，但其 7.2 秒 RMSE 从 T30 的 0.884 m 降到 T120 的 0.827 m。平均跟踪与最关键位置的净空可能给出不同排序。增益 7、10 在短延迟区间的周期差异约为厘米量级，同时余量较宽裕。

固定 T120 时，增益 10 从 R9 到 R99 的 RMSE 从 0.367 m 升到 0.694 m，角速度限幅从 1.7% 升到 68.3%。增益 7 在同样条件下从 0.394 m 升到 0.539 m，限幅从 1.7% 升到 25.0%。这条路线的简化模型显示，高增益在低延迟更准，在高延迟下退化也更大。六个代表组合从 5 ms 加密至 2.5 ms 积分，目标门余量最大改变 0.068 mm。该检查只覆盖数值步长。

离线原始输出位于 `results/teacher_gain_timing_grid_20261110_offline/`，含冻结输入、33 组轨迹、汇总 CSV 和步长复核。

## AirSim 复跑

打开 Blocks 并点击 Play，保持窗口展开。只需有 Drone1，不用启动板端服务。若上次异常退出并遗留生成的门框，先停止并重新进入 Play。

```powershell
Set-Location -LiteralPath 'F:\FalconReproduction\airsim_control'
$py = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$out = Join-Path (Get-Location) ('controller_architecture_study\results\teacher_gain_timing_airsim_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $py -u -m controller_architecture_study.teacher_gain_timing_grid --backend airsim --output $out
```

每组落盘后会更新 `summary.json`。运行结束生成 `summary.csv`。先核对目标延迟是否实际兑现，再比较各组物理结果。完整飞行还应检查 `full_timing_valid`。默认一轮 33 组，可用 `--repeats 2` 重复并打乱组内运行顺序。教师结果与日后 DNN 结果分别保存。
