# FALCON paper-core ACADO teacher v2

这是修复后的独立采集空间。`acado_v1/paper_core_40k_acado` 已确认系统性撞地，必须保留为故障证据，不能恢复、合并或训练。

v2 修复了三个实现问题：

1. 每个门段只生成一次五次多项式，并按 50 ms 控制时间推进；不再每帧把实测加速度作为新多项式的硬初值。
2. 由期望加速度计算参考姿态，并由相邻参考姿态计算参考 body rates；ACADO 接收完整的参考状态与参考输入。
3. 不可行 motion primitive 会明确失败；smoke test 保留完整 5 秒，不再用无法暴露发散的 0.5 秒测试。
4. AirSim `simSetKinematics` 后的首个重力瞬态不作为第一段多项式的初始加速度；后续门段仍使用真实闭环加速度。

论文没有公开其多项式边界、速度和重规划触发条件。因此这些仍是明确记录的复现假设；ACADO/qpOASES 求解器来自公开 `uzh-rpg/rpg_mpc`。

## 命令

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py .\experiments\falcon_paper_fidelity_acado_v2\validate_protocol.py
& $py .\experiments\falcon_paper_fidelity_acado_v2\validate_ideal_closed_loop.py

& $py .\experiments\falcon_paper_fidelity_acado_v2\collect_paper_core.py `
  --auto-launch --render --smoke-test
```

Smoke test 是 `2×2×100=400` 条，每个 episode 都跑满论文规定的 5 秒。只有它 4/4 全部完成且无碰撞，才能正式采集：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v2\collect_paper_core.py `
  --auto-launch
```

中断后只恢复 v2：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v2\collect_paper_core.py `
  --auto-launch --resume
```

每个 episode 都以独立文件原子保存。恢复时会跳过已验证完整的 episode，重跑
`failed`、`interrupted`、`running` 和 `pending` 项，不需要从零开始。采集器每完成
100 个 episode 会预防性重启一次它自己启动的 AirSim；遇到 msgpack-rpc timeout
也会重启 AirSim 后原地重试当前 episode。若连续恢复仍失败，整批采集会停止，
避免把后续 episode 错误标记为数据失败。重启次数与原因记录在 manifest 的
`rpc_restarts` 和 `rpc_restart_events` 中。

`--auto-launch` 要求本机 41451 RPC 端口没有其他 AirSim 实例。正式采集前请关闭
手动打开的 UE4/Blocks；如果端口已被占用，采集器会直接报错，不再静默连接到
可能含有 `FalconGate_*` 可视化门框的错误场景。

如果某个范围曾在错误场景中续跑，可先预览再标记为重新采集。例如保留 0--302，
重采 303--399：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v2\prepare_clean_recollection.py `
  --start 303 --end 399

& $py .\experiments\falcon_paper_fidelity_acado_v2\prepare_clean_recollection.py `
  --start 303 --end 399 --apply

& $py .\experiments\falcon_paper_fidelity_acado_v2\collect_paper_core.py `
  --auto-launch --resume
```

`--apply` 会先备份 manifest；原 episode 文件不立即删除，但状态变为 pending，
续跑成功时才逐个原子覆盖。

最终文件为：

```text
experiments/falcon_paper_fidelity_acado_v2/paper_core_40k_acado_v2/final/falcon_paper_core_40k_acado_v2.npz
```

## 五门教师控制器可视化

关闭其他 UE4/AirSim 后，以下命令会启动可见窗口，读取已完成数据集中的第 38 个
高难度环境和第 8 个初始状态，使用与采集完全相同的 ACADO/qpOASES MPC 与五次
多项式规划器，以 5 m/s 依次穿过五个实体门。它不会写入或修改 paper-core 数据：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v2\demo_acado_five_gates.py `
  --auto-launch --initial-state-id 8 --environment-id 38 --duration 15
```

运行结束会逐门打印实际门平面交点的横向/垂直偏差和 `PASS/FAIL`，并汇总
`valid_crossings`、deadline miss 与 MPC 平均求解时间。
