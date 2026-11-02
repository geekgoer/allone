# 5 m/s 六模型 baseline：起步参考对照

## 目的与边界

只检查原 baseline 的起步参考是否造成性能指标偏差。不重新训练 DNN，不改 FPGA 权重、12 维输入、门尺寸或通过判定，不扩大 SVR 数据集。

原参考从起点立即以 5 m/s 前进，但飞机从近似悬停开始。这会留下持续的位置滞后。新参考在起步时匹配实测位置和三维速度，以固定 2 秒的五次多项式起步段接上原折线路径；接点位置、速度和加速度连续，此后巡航参考仍为 5 m/s。2 秒对所有模型、周期一致，不根据实测误差拟合时间偏移。后续折线拐角没有平滑，不能称为完整复现论文的轨迹规划器。

重要：现有 DNN 仍只接收门相对位置、相对姿态、实际速度和实际加速度，不接收此时间参考。因此，新参考正常情况下只改变评价目标，不直接改变 FPGA 控制指令；只有发生 MPC 后备控制时，它才会影响后备指令。含后备控制的飞行不得作为纯 FPGA 验收结果。

## 已完成的离线检查

使用原有 216 次无附加延迟的有效飞行日志，保持实际轨迹不变，分别按两个参考重算指标。原日志、标签和 SVR 文件没有改写。

- epsilon=0.8 的 24 组配对中，周期从 30 ms 增至 100 ms：原参考的 J 平均增加 7.29%；新参考的 J 平均下降 16.81%，其中 22/24 组下降。J 越大越好。
- 但模型能力排序未恢复预期：在 T=60 ms、epsilon=0.8 下，模型 1 的新参考 J 约为 0.03568，模型 6 约为 0.01177，最大模型反而更差。模型 6 先前较快追上恒速参考，换成较慢的起步参考后可能更容易超前。
- 两个参考对应不同任务，J 绝对数值不能用来宣称控制器性能提高。不能因为新参考的周期趋势接近论文，就忽略模型排序变差。
- 这说明参考定义会显著影响结果，尚不能认定 baseline 已满足论文趋势，也不应立即扩大其他 SVR 采集。

详细结果：

- `audit_20261015/startup_reference_replay/reference_replay_report.json`
- `audit_20261015/startup_reference_replay/same_flights_reference_comparison.csv`
- `audit_20261015/startup_reference_replay/reference_definition_comparison.png`

## 实飞对照：一条命令，24 次

运行条件：手动打开 Blocks 并点击 Play，使 AirSim RPC 41451 监听；板端继续运行当前历史 5 m/s 六模型服务，Windows 配置端点为 `192.168.77.2:5008`。停止其他正在控制无人机或大量请求 FPGA 的脚本。此步骤会让 FPGA 实际控制 AirSim 无人机，不是影子模式；不自动启动其他 AirSim 实例。

固定方案：模型 1、6；S02（epsilon=0.8，seed=917，initial_state_id=2）；周期 30/60/100 ms；附加延迟 0；原恒速参考/平滑起步参考；每项重复 2 次。合计 `2×1×3×2×2=24` 次。模型和场景不根据实验结果临时更换。

在 Windows PowerShell 中运行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\run_startup_reference_comparison.py --enable-fpga-actuation
```

中断后重新运行同一命令，可以跳过已经完成且存在日志的飞行；未完成的飞行从头重做，不从空中的状态继续。

## 输出与判断

输出位于本目录下 `paper_style_sweep/startup_reference_comparison_v1/`：

- `comparison_manifest.json`：固定实验方案。
- `comparison_summary.json`：24 次飞行的完成情况、J、实测 R、纯 FPGA 有效标记。
- 六个独立子目录：`T030ms_constant`、`T030ms_smooth-start` 等；各自保存参考版本、原始逐周期状态和逐次飞行汇总。

先看是否完成五门、是否碰撞、是否后备/超时/错过截止期。`clean_fpga_episode=true` 才表示该次完成且没有这些问题；完成次数不是纯 FPGA 有效次数。保留失败记录，不只报告成功样本。

然后分别比较各参考下同模型同场景的周期趋势，以及同周期下模型 1 与模型 6 的差别。若更换参考仅改善周期趋势却恶化模型能力排序，应如实保留这个结论，而不是按期望趋势挑参考。

两套参考不得合并成同一个 SVR 的 J 标签。数据构建脚本已经加了参考版本检查，发现混合版本会拒绝写出新数据集。原有参考默认仍为 `constant`，旧流程保持可用。

## 本地验证范围

已完成起步端点、导数连续性、不同周期下相同物理时刻参考一致性、无效参数、参考版本隔离和原门通过/性能指标回归检查，共 18 项通过；对照命令已 dry-run。尚未代替用户运行这 24 次 AirSim/FPGA 实飞。
