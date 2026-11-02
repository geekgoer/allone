# Windows E1 v2 修复与运行

2026-10-23。修复后的真实 AirSim 实验已经完成 216/216，另外完成三次教师/无扰动诊断。模型与六配置选择规则沿用原实验。

结果与下一步见 [E1 v2 完整结果分析](STATE_MODE_E1_V2_RESULTS_20261023.md)。90 次控制恢复、84 次联合通过；相同状态下六配置二元恢复结果一致，尚未证明动态复杂度的必要性。下面保留修复说明与原运行步骤供复现，不需要再次运行已完成的 216 次。

## 修正的问题

1. **初态放置扫掠**：v1 将 `simSetKinematics` 设为 `ignore_collision=False`。转弯起点在 x≈20.9m，出生点至该点的直线在 Gate1 平面处为 z≈-2.042m，恰好穿过 z=-2m 的底梁附近。v2 在暂停准备阶段瞬移物理体，并通过 `simSetObjectPose(..., teleport=True)` 将 UE actor 同步到相同世界位姿；两者都读回核对。正式观察阶段继续检查真实碰撞，不忽略首帧的新碰撞。
2. **目标与截止期混淆**：v1 在超过20+8ms比较窗口或RPC返回超出窗口时中止。v2 保留20ms共同发送目标、8ms比较容差、50ms job截止期；超过比较目标只影响时序可比性，继续记录恢复；发送前真正过期才拒绝输出。
3. **RPC返回语义**：发送入口、RPC返回分别记录。31ms返回不代表命令到31ms才生效，也不因此中止；返回超过50ms意味着无法从该代理确认及时性，标为不可比并保留观察。任何代理都不冒充精确执行器时间戳。
4. **预热不足**：改用12个真实初态输入和实际模式切换顺序预热，每次试验准备时再用读回状态预热所选模式。推理路径使用 `torch.jit.optimized_execution(False)`，避免在试飞时触发新的profiling优化。默认改用CPU，减少与UE4争用GPU的影响。

这是实验实现修复，不是把模型失败改成通过：目标偏差仍显示 `timing=False`、不能进入已观测合格集合；碰撞、越界、恢复超时仍保存为失败。

## 已做的检查

- 14项新旧相关回归测试通过。
- 模拟RPC返回晚于比较窗口但发送准时：试验继续、时序合规。
- 超过比较目标而未超过截止期：继续观察、保留时序不可比标志。
- 超过截止期的输出：不下发。
- actor/物理体准备阶段均使用瞬移并读回，恢复窗口内持续推进模拟时间。
- 真实FP32黄金输入核对最大物理误差 `9.53674316e-7`。
- 本次CPU短测，六配置预热后的最大推理耗时约0.64/1.25/2.24/4.74/7.72/14.96ms。这是短测，不是最坏执行时间保证。

## 用户如何运行

如旧命令还在运行，先Ctrl+C停止。UE4中Stop后重新Play，清理旧门框；关闭正在运行的训练。保持使用手动打开的Blocks。

先运行12个新增试验，覆盖六模式的直线与转弯状态：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_state_mode_probe.ps1 -Python $trainPy -Device cpu -MaxTrials 12
if ($LASTEXITCODE -ne 0) { throw 'v2试验中止，已完成记录保留；请返回末尾错误。' }
```

12次结束后返回终端汇总或新目录的 `summary.json`。重点核对转弯组是否仍报远处Gate1碰撞、是否还有大量deadline_misses；不能要求所有模式都成功，否则实验无法观察能力差异。

确认修复后，移除 `-MaxTrials 12` 重跑相同命令，即从已保存结果续跑至216次，无需重复前12次。`MaxTrials` 是每次新增试验数，重复运行带该参数会再增加12次。

## 数据与日志

新目录：`F:/FalconReproduction/airsim_control/controller_architecture_study/results/state_mode_admissibility_v2/`。

旧 `state_mode_admissibility_v1` 保留，不复制旧的成功/失败结果到v2。v2使用新的源码哈希和协议版本，不能与v1混合统计。

终端新增：

- `target_deviations`：超过共同发送目标容差的次数；非零时该trial时序不可比，但继续观察。
- `deadline_misses`：发送前已超过job截止期的次数；该输出不下发。
- `recovery`：控制恢复与进度判据。
- `timing`：发送目标、代理时效、观测间隔、仿真/墙钟比例是否合规。
- `joint`：恢复与时序同时通过。

`commands.json` 保存 `target_s`、`deadline_s`、`command_send_s`、`rpc_return_s`、`rpc_after_send_ms`、`delivery_target_missed`、`deadline_missed`、`rpc_return_after_deadline`。发生碰撞时 `trace.json` 额外保存碰撞时间、位置和impact point，便于区分物理接触与状态同步问题。

`initial_state.json` 中 `actor_readback_position_rotation` 与 `readback_position_velocity_rotation` 必须合格。`warmup.json` 保留实际初态推理的预热数据。

模型、归一化、参考路径、50ms周期、20ms比较目标、8ms容差和恢复阈值保持明确记录。CPU试验用于先识别模式能力，后续GPU/FPGA执行成本在各自平台另测。
