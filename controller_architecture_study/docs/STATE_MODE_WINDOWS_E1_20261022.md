# Windows E1：237D 超网六模式恢复能力实验

日期：2026-10-22；2026-10-23 修订为 v2。最新操作与修复原因见 [v2 修复说明](STATE_MODE_WINDOWS_E1_FIX_20261023.md)。旧 v1 数据保留用于诊断，修复后的试验保存到新目录。

v2 交付检查：14 项新旧相关离线测试通过；六配置各两条冻结黄金输入的 TorchScript 输出核对通过，最大物理控制差异 `9.53674316e-7`；PowerShell 入口与计划模式已检查。修复后的实际 AirSim 试验待用户执行。

## 这一步回答什么

在相同初态、参考、旧控制和更新时序下，比较六个 subnet 的恢复表现，观察不同状态下最低已观测合格成本是否变化。这是后续 State-to-Mode 表的先导实验。

不需要重新训练、不需要 Ubuntu、不需要 FPGA。六个模式均来自同一个 FP32 v4 超网，不调用旧的 12D 模型或四个独立 DNN。飞行过程中没有 MPC 回退。

## 六个配置与选取规则

先在已有离线 validation 质量与 effective MACs 中取 frontier，排除 MSE 超过最佳值十倍的明显低质量出口，再按 log(MACs) 均匀选六点，包含最佳离线配置。选择在试飞前完成；原始 136 点与规则都进入 setup.json，不因闭环结果更换。

| 模式 | depth | width | effective MACs | validation normalized MSE |
|---|---:|---:|---:|---:|
| d11_w0256 | 11 | 256 | 553216 | 0.00161858 |
| d11_w0384 | 11 | 384 | 1075584 | 0.000989034 |
| d23_w0384 | 23 | 384 | 2255232 | 0.000719170 |
| d29_w0512 | 29 | 512 | 4710912 | 0.000543828 |
| d27_w0768 | 27 | 768 | 9130752 | 0.000474293 |
| d33_w1024 | 33 | 1024 | 19121152 | 0.000176529 |

MACs 仅是当前阶段的工作量代理，不是实测 FPGA 时延。这里较小的模式仍有 11 层，是因为本超网最浅出口的离线质量明显较低；不能靠故意选坏的小模型证明动态模式选择有用。未选中的模式并未从超网删除。

## 初态、路径与边界

使用已有验证环境 env48 的 route00（接近直线）与 route03（明显转弯）；不读取最终 test。保存的多项式直接回放，不重新规划或在门切换时重启参考时钟。

每条路线选择固定参考相位，从相同参考位置/姿态出发，设置三档横向剩余间隙 0.3/0.6/1.0m、两档间隙收缩率 -0.4/-1.0m/s。总计 12 初态 × 6 模式 × 3 次重复 = 216 个 trial，模式顺序轮换。

这里 `h` 是**随参考移动的虚拟轨迹走廊**的剩余间隙，不是到 UE4 门框 mesh 的精确距离。走廊半宽/半高为 2.0m，扣除假定无人机半径 0.20m 和 clearance 0.25m，允许中心偏差 1.55m。该几何包络是明确的实验假设；实际门框碰撞和穿门平面也另行检查。

初始位置偏差因此为 1.25/0.95/0.55m。参考速度沿用已有 5m/s 路径，局部实际参考速度随多项式变化。旧命令统一为该参考的推力前馈、零角速度，先在暂停准备阶段提交；恢复模式开始前各 trial 的旧命令相同。

状态只在每个 trial 的准备阶段注入并读回，位置/速度/旋转容差分别为 0.02m、0.05m/s、0.02rad。v2 使用瞬移同步物理体和 UE actor，并分别核对两者位置，防止从出生点跨门扫掠。开始后物理仿真持续运行，不能为了等待网络而暂停。只保证可观测初态配对，不声称隐藏的电机/控制状态完全相同。

## 时序与恢复判据

- 默认 release 周期 50ms，统一目标为 release 后 20ms 开始发送命令，比较容差 8ms。错过目标但未过 50ms 截止期时继续试验，标记时序不可比；发送前已过截止期才拒绝下发。
- 分开记录实际采样、推理完成、命令发送和 RPC 返回。RPC 返回晚于20ms比较目标不直接中止试验；若晚于该job截止期，记录无法确认及时性的证据，标记时序不可比，但不倒推命令发送失败。发送时刻与返回时刻均不等于已知的精确执行器生效时间。
- 参考相位使用 AirSim 时间戳推进；调度使用主机单调时钟。仿真/墙钟比需位于 0.95–1.05，最大观测空隙不得超过 100ms。
- 固定观察 3 个仿真秒；要求位置误差 ≤0.3m、速度误差 ≤0.6m/s，连续保持至少 0.25s，并在完整观察窗内没有观测到走廊越界、碰撞或无效穿门。
- 路径进度至少达到同窗参考进度的 80%，排除悬停、倒退换取表面恢复。用固定保存路径上的近邻弧长估计进度，保留原始轨迹供检查。
- 提前碰撞、越界不会因积分窗口变短而被判为更优。恢复表现、时序可比性和两者同时通过分别记录。

50ms/20ms 是本轮比较条件，不是 FALCON 的固定物理限制。v2 默认 CPU；使用12种真实初态与实际切换顺序预热，并关闭本推理路径的 TorchScript 动态优化。若预热后的推理本身已经超过50ms，程序在试飞前报告；统一变更时序时使用新结果目录，不单独给大模型放宽条件。

## 从上到下复制运行

先手动打开 Blocks，在 UE4 中 Stop 后重新 Play，清理之前生成的门；让 AirSim 持续运行。关闭正在进行的训练。实验会 reset 仿真并使用 Drone1，不自动启动另一个 UE4，也不负责控制其他无人机。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_state_mode_probe.ps1 -Python $trainPy -Device cpu -MaxTrials 12
if ($LASTEXITCODE -ne 0) { throw 'E1 中止，已完成数据保留；查看末尾错误后重跑相同命令续跑。' }
```

以上先运行12个新增 trial，覆盖直线和转弯；检查后移除 `-MaxTrials 12` 重跑同一命令，继续余下试验。完整216次的纯观察窗口合计至少10.8分钟；另有 reset、状态准备和门框生成时间，实际总耗时更长。

`-MaxTrials` 限制本次新增数量，不改变实验 manifest。只查看计划使用 `-PlanOnly`，不连接 AirSim；只重新汇总现有结果使用 `-SummarizeOnly`。CUDA 可显式指定，但与 CPU 结果不能混在同一目录。

## 续跑与输出

目录：`F:/FalconReproduction/airsim_control/controller_architecture_study/results/state_mode_admissibility_v2/`。

- `setup.json`：冻结模型/归一化/源码/校准哈希、全部配置、初态、时序与 trial 顺序。
- 每个 trial 下 `attempt_XXX/initial_state.json`、`warmup.json`、`trace.json`、`commands.json`、`report.json`；初态日志包括 actor 和物理体的读回。
- `summary.json`、`summary.csv`：每状态每模式的成功、时序失效和完成数量。
- `recovery_heatmap.svg`：直接用浏览器打开，显示“joint 成功次数/已完成次数”；I 表示时序不可比次数；pending/partial 明示缺失结果。

每次 trial 完成即保存。普通控制失败或时序失败也算已完成，不会在续跑时被自动重试到成功。Ctrl+C、连接失败或初态准备失败的 attempt 保留，下次继续该 trial。模型、源码、校准或协议变化会拒绝沿用旧目录。

三次全部通过仅标为 `pilot_all_passed`；结果混杂、时序无效或未完成标为 unknown。它们不是已认证 admissibility region，也不会直接生成运行中的模式选择表。

汇总还给出每状态最低已观测通过模式，以及若存在的“全部状态均通过的最低成本静态模式”。基于这些点的 hindsight MAC 节约只作潜在收益提示，不计运行开销，不能当作已部署策略效果。

程序整体正常完成但部分 trial 飞行失败时，可以正常退出：这是实验数据，不是脚本崩溃。不要仅看 exit code 判断假设成立。

## 执行后返回什么

返回终端末尾汇总，以及 `summary.json`（或贴出其内容）。据此判断：

1. 同状态下六模式是否有一致差异；
2. 差异是否主要由 timing invalid 造成；
3. 是否存在“简单状态可用低成本模式、较难状态需另一模式”的结构；
4. 下一轮仅挑哪些边界点补 delay/未见状态测试。

如果所有模式都成功，当前条件下复杂度适应的必要性尚弱；如果全部失败，先检查状态可恢复性、时序和输入接口。无需现在就扩大到136模式、重新训练或搬到Ubuntu。
