# 三机链路优化：批量参考采样与直接 237D 编码

本阶段只优化运行实现，不训练、不改原教师数据、模型权重、归一化参数或拆分。真实 GPU/AirSim 试飞由用户执行。

## 原对照结论

`airsim_chain_20261004T045020_054718Z/summary.json` 中两次均完成 15 次穿门，但都没有满足 T=D=50ms。

| 平均耗时 | parallel | serialized |
|---|---:|---:|
| 参考窗口 | 55.03ms | 18.48ms |
| 特征编码 | 28.30ms | 6.69ms |
| 预处理锁等待 | 约0ms | 42.14ms |
| 采样开始→DNN 输出 | 187.85ms | 218.52ms |

串行化没有解决端到端实时性。CPU 短检查也复现了原预处理的线程竞争。GPU 服务变慢还可能涉及发起推理的 CPU 线程和渲染资源竞争，当前没有完成因果隔离。

## 修改范围

- 新增 `controller_architecture_study/fast_convex_preprocess.py`。
- 在三机入口新增显式 `--preprocess-policy vectorized`；原 parallel/serialized 和原默认值保留。
- 按原连续时间批量计算 21 个参考点及原 1ms 前向差分点。保留完整 p/v/a、姿态和参考角速度，不使用离散查表或重拟合。
- 直接构建现有 237D：当前 q/v 共7维，21个相对位置/参考姿态/相对速度共210维，20个参考推力共20维。坐标与四元数符号规则不变。
- 原 QP MPC、DNN FP32 推理、共享 FCFS 队列、周期、期限、漏期统计和迟到输出保护不变。
- 新报告记录 `preprocess_policy=vectorized` 和实现文件 SHA256。旧结果不覆盖。

这里省掉的是原 300D 中随后被丢弃的中间数据，不是继续减少 DNN 输入维数，也不改变模型学习目标。

## 已完成 CPU 检查

报告：`controller_architecture_study/results/dynamic_complexity_237d_v1/preprocess_check_20261004T050716_291685Z/report.json`。

验证环境48～53，各取 route0，合计216个时间点，包含段边界两侧、边界本身及末尾悬停；不读取原 Test 数据。

| 检查 | 最大绝对差 |
|---|---:|
| 参考 p/v/a | 0 |
| 参考四元数 | 1.11e-16 |
| 参考角速度 | 1.12e-13 |
| 237D 输入及归一化后输入 | 0 |
| 抽样 MPC 控制输出 | 0 |
| 抽查24条 residual21x1024 CPU FP32 输出 | 0 |

短 CPU 基准：单线程原版6.32ms→新版0.81ms；三线程原版45.06ms→新版5.45ms（均为每条请求的预处理平均时间）。不是渲染试飞时的端到端响应，也不是 WCET 保证。

单元测试另覆盖任意姿态、旋转矩阵四个分支、四元数正负号、平移、短窗口补齐、段边界和异常输入。它们是有限数值检查，不是全输入数学等价证明。

本轮30项CPU回归测试通过，包括新旧预处理、MPC、模拟三机客户端线程归属、计时字段和迟到输出保护。未运行真实GPU或AirSim试飞。

如需重复 CPU 检查：

```powershell
& $trainPy -m controller_architecture_study.probe_fast_preprocess
```

## 用户现在只运行这一遍试飞

手动打开 Blocks/FalconEmpty 并点 Play，确认有 Drone1/2/3。停止其他训练和推理实验。不使用 auto-launch，不会另开 AirSim。若有其他实验留下的门，先 Stop/Play。

下面整段复制到 PowerShell。约20秒测量，另加加载、预热和初始化。开始时只 reset 一次，把无人机放到空中起点，所以没有起飞动画；测量中持续仿真，不为推理暂停或移动机体。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
& $trainPy -m controller_architecture_study.probe_airsim_three --mode shadow --model residual21x1024 --device cuda --phase sync --preprocess-policy vectorized --environment-id 48 --route-index 0 --period-ms 50 --deadline-ms 50
if ($LASTEXITCODE -eq 2) {
    Write-Warning '穿门完成，但实时性仍未通过。保留报告，先分析剩余瓶颈；不要进行 DNN 接管。'
} elseif ($LASTEXITCODE -ne 0) {
    throw '连接、试飞或推理异常，请查看终端打印的 report.json。'
}
```

仍然是 CPU MPC 实际控制，GPU DNN 仅影子推理。使用与旧对照相同模型、路线和同步释放，暂不增加无人机数、不更改50ms期限、不切小模型。

## 输出与下一步判断

终端打印一个新的 `dynamic_complexity_237d_v1/airsim_shadow_UTC/report.json`。同目录还保存 `setup.json`、`requests.jsonl`、`flight_trace.json`。

重点看：

1. `flight_passed`、`realtime_passed`、`acceptance_passed`，以及 `output_deadline_misses`。
2. `reference_ms`、`feature_ms` 是否下降；`service_ms`、`queue_ms` 是否随之恢复。
3. `release_to_output_ms` 的 P99/最大值，以及实际完成的周期数。漏期分母仍包括 skipped/busy/error，不仅统计完成的输出。

若预处理明显变快但 GPU 服务仍很慢，下一步隔离推理发起线程与渲染影响；不能直接断定模型太大或重训。链路稳定后才进入动态模型选择基线及共享网络对比。本阶段尚未做 GPU/AirSim 实际优化试飞，也没有实现动态T或超网。
