# 下一步：三机实际链路复测

更新：下述 parallel/serialized 对照已经由用户完成，均穿门成功但实时性失败。后续已实现批量预处理，新的单次试飞命令见 [DYNAMIC_CHAIN_VECTORIZED_20261004.md](DYNAMIC_CHAIN_VECTORIZED_20261004.md)。本页保留原对照设计与命令供追溯，不需要重复运行。

目的：查清三机影子测试中，规划/特征计算/排队/推理分别占多少时间。现有模型已足够，本阶段运行两次对照飞行。

## 已有证据

本轮容量扫描：[summary.json](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1/capacity_20261004T042648_537126Z/summary.json)。28 组全部执行，共 63,600 次计划请求。

| 条件 | 输出漏期 | release→output P99 |
|---|---:|---:|
| 8 路，残差 33×1024，同步 | 0/2400 | 30.90 ms |
| 12 路，残差 33×1024，同步 | 5/3600 | 43.52 ms |
| 16 路，残差 33×1024，同步 | 97/4800 | 52.44 ms |
| 16 路，残差 33×1024，错峰 | 0/4800 | 26.12 ms |
| 16 路，普通 6×1024，同步 | 0/4800 | 34.62 ms |

纯 GPU 请求回放提示释放相位和模型选择都有作用。旧三机影子飞行仍存在明显的链路超时，因此先测量分段耗时。上述每个容量条件仅运行 15 秒，不视为容量认证。

## 本次两次飞行

入口：[probe_airsim_chain.py](F:/FalconReproduction/airsim_control/controller_architecture_study/probe_airsim_chain.py)。

固定：残差 21×1024、env48/route0、三机同步释放、周期和期限均 50 ms，约 5 m/s 的已保存轨迹。选用 21×1024 是为了与旧的三机影子测试保持相同模型。

顺序：

1. `parallel`：三线程各自计算参考轨迹和 237D 特征，保留原执行方式。
2. `serialized`：参考轨迹和特征计算使用同一把锁，检查降低线程竞争后收益是否超过锁等待。

两次均由 CPU MPC 实际控制；GPU DNN 对同一输入并行推理、记录结果。数学控制器、输入合同、规划轨迹保持一致，但两次物理飞行状态可能不同，不作为逐状态配对精度实验。

每次开始都 reset 一次，放置三架无人机到各自空中起点，再生成 15 个独立门。第一遍结束后会开始第二遍；开头定位与两遍之间的复位是实验流程。单遍测量期间不为推理暂停仿真，也不瞬移机体。

## 运行

先手动打开 Blocks 的 FalconEmpty 地图并点击 Play，确认已有 Drone1/2/3。若旧实验门留在场景，先 Stop 再 Play。使用同一个手动打开的 AirSim 实例，停止其他训练和推理实验。

以下一整段复制到 PowerShell。只需点一次 Play，程序依次执行两遍。测量时间约 40 秒，另加模型加载、预热、场景初始化。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
& $trainPy -m controller_architecture_study.probe_airsim_chain --device cuda
if ($LASTEXITCODE -eq 2) {
    Write-Warning '两次对照已完成，但实时性没有全部通过；保留汇总，先分析耗时。'
} elseif ($LASTEXITCODE -ne 0) {
    throw '试飞或连接异常，后续对照已停止；查看保存的 summary.json。'
}
```

碰撞、连接或其他试飞异常会停止剩余试验。单次仅实时性未通过时继续第二个对照；两次跑完但未全部验收通过，入口退出码为 2。`status=COMPLETE` 表示实验完整执行；满足飞行和期限条件须另看 `acceptance_passed`。

## 输出与阅读

根目录：[dynamic_complexity_237d_v1](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1)。每次运行新增 `airsim_chain_UTC时间戳`：

```text
airsim_chain_时间戳/
  summary.json
  parallel/
    setup.json
    report.json
    requests.jsonl
    flight_trace.json
  serialized/
    setup.json
    report.json
    requests.jsonl
    flight_trace.json
```

终端打印两次的飞行/实时性、完成数量、漏期，以及各段 mean/P99 毫秒表。已有结果不覆盖，已写出的单次报告和部分汇总可供中断后查看。

| 字段 | 解释 |
|---|---|
| `reference_ms` | 生成未来参考窗口 |
| `feature_ms` | 编码 237D 输入 |
| `preprocess_wait_ms` | 等待预处理锁 |
| `queue_ms` | 等待共享推理工作线程 |
| `service_ms` | 归一化、传输、DNN 前向、取回和反归一化 |
| `sample_to_output_ms` | 取状态 RPC 开始到 DNN 结果可用 |
| `sample_to_mpc_dispatch_ms` | 取状态 RPC 开始到实际 MPC 命令提交 |

阶段统计的样本集合可能不同，不将各段 P99 相加推导总 P99。`service_ms` 同时包含软件开销，不能直接称为纯 GPU 内核时间。

本次新增入口通过 15 个 CPU 回归测试，包括实时性失败继续对照、连接失败停止、报告保留，以及已有三机探针回归。实际 GPU 对照飞行尚待用户运行。

## 后续路径

1. 依据分段结果修正实际控制链路的瓶颈；只在相关条件下重测。
2. 实现独立模型池的状态/预算选择基线，计入选择开销，禁止使用在线 MPC 真值选择模型。与固定模型、错峰调度比较。
3. 在相同状态来源、预算和控制指标下比较多出口网络与弹性共享网络，计入切换和浪费计算。

当前没有实现实际动态模型选择、动态 T 或超网训练；这些是后续阶段，不能将本次影子结果描述为超网闭环效果。
