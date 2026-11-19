# FALCON 论文复现：ACADO/qpOASES 教师 MPC

> **已停用（故障留档）**：该版本会逐帧重建 motion primitive，正式 5 秒采集已确认系统性发散和撞地。不要执行本页的采集/恢复命令，也不要用 `paper_core_40k_acado` 训练。请改用 `experiments/falcon_paper_fidelity_acado_v2/README.md`。

这个目录替换了上一版耗时约 `250–1100 ms` 的 CasADi/IPOPT 教师。新后端直接编译并调用 FALCON 引用的公开 `uzh-rpg/rpg_mpc` 仓库中已经生成的 ACADO C 代码和 qpOASES。

它与旧目录完全隔离：

- 旧的故障分析数据保留在 `experiments/falcon_paper_fidelity_v1/paper_core_40k`，不要恢复、合并或用于本轮训练；
- 新数据写入本目录的 `paper_core_40k_acado`；
- 每个 episode 原子保存，支持 `--resume`。

## 复现边界

- FALCON 论文明确给出控制/采样周期 `0.05 s`、预测步数 `K=20` 和 40,000 对训练数据。
- 公开 `rpg_mpc` 生成代码内部使用 `0.1 s × 20 = 2.0 s` 的预测网格。这和 50 ms 的控制调用周期是两个不同概念：控制器每 50 ms 重算一次，但每次优化预测未来 2 s。
- 论文没有发布作者实际使用的 ACADO 二进制、全部权重、随机种子和多项式规划器。本目录使用公开 `rpg_mpc` 默认权重，并把所有工程假设写入配置和 manifest。
- 因此可以称为“公开协议 + 公开 ACADO/rpg_mpc 后端复现”，不能称为作者程序的逐位复刻。

详细来源锁定在 `solver_provenance.json`。

## 0. PowerShell 初始化

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
```

## 1. 构建原生求解器

本机已构建过；只有源码或编译器变化后才需要重建：

```powershell
cmake -S .\native\acado_rpg_mpc -B .\native\acado_rpg_mpc\build `
  -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=D:/mingw64/mingw64/bin/gcc.exe `
  -DCMAKE_CXX_COMPILER=D:/mingw64/mingw64/bin/g++.exe

cmake --build .\native\acado_rpg_mpc\build --config Release --parallel 8
```

输出目录必须同时存在：

```text
native/acado_rpg_mpc/build/bin/libfalcon_acado_mpc.dll
native/acado_rpg_mpc/build/bin/libwinpthread-1.dll
native/acado_rpg_mpc/build/bin/benchmark_falcon_acado.exe
```

## 2. 离线验收（不需要 AirSim）

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v1\validate_protocol.py

& $py .\experiments\falcon_paper_fidelity_acado_v1\benchmark_solver.py `
  --iterations 10000
```

两条命令都必须显示 `PASS`。Python benchmark 的 `p99` 必须低于 50 ms。当前本机原生动态状态 benchmark 约为：平均 `0.236 ms`、p99 `0.432 ms`，不再有 IPOPT 的数百毫秒瓶颈。

## 3. 可见小规模 AirSim 测试

自动打开可见 AirSim：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v1\collect_paper_core.py `
  --auto-launch --render --smoke-test
```

如果你已手动打开 UE4、点击 Play，并希望连接这个窗口，则去掉 `--auto-launch`：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v1\collect_paper_core.py `
  --render --smoke-test
```

Smoke test 是 `2×2×10=40` 条，只检查坐标、控制方向、实时 20 Hz、碰撞和保存，不用于训练。若以前运行过同名 smoke test，删除旧 smoke 目录或给 `--output-root` 指定一个新目录；不要用 `--resume` 接 CasADi 的旧目录。

## 4. 正式生成 40k

Smoke test 确认无人机运动方向和高度正常后，再执行：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v1\collect_paper_core.py `
  --auto-launch
```

Ctrl+C、断电或程序中断后：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v1\collect_paper_core.py `
  --auto-launch --resume
```

最终数据：

```text
experiments/falcon_paper_fidelity_acado_v1/paper_core_40k_acado/final/falcon_paper_core_40k_acado.npz
```

完成后验证：

```powershell
& $py .\experiments\falcon_paper_fidelity_acado_v1\validate_protocol.py `
  --dataset .\experiments\falcon_paper_fidelity_acado_v1\paper_core_40k_acado\final\falcon_paper_core_40k_acado.npz
```

## 5. 什么时候才能训练

只有以下条件同时满足才进入 DNN 训练：

1. 可见 smoke test 中飞行方向正确，没有撞地或发散；
2. `actual_loop_period` 保持接近 50 ms，deadline miss 不是系统性发生；
3. 正式数据达到 `X=(40000,12)`、`U=(40000,4)`；
4. manifest 的 `teacher_policy_version` 是 `falcon_paper_core_quintic_official_rpg_mpc_acado_qpoases_v1`；
5. 数据不包含旧 CasADi episode。

当前阶段先不要运行原目录的 `prepare_paper_core.py` 或训练六个 DNN；要等新 40k 完成并检查统计分布后，再为这个新目录建立固定 episode 拆分。
