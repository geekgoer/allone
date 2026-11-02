# 一条命令连续采集八环境SVR数据

本入口调用原采集器，不再传`--limit`，会依冻结计划依次处理全部4608次试验，自动复位、生成门框、换模型/T/R/环境、保存数据并在结束时导出CSV。不是每飞一次就重新运行；也不是拟合SVR模型或重新训练DNN。

## 启动前

- 保持PS_ETH连接，Windows/板端IP为192.168.77.1/2，当前六模型服务UDP5008运行；六个input scale均为32。
- 原UE4.27.2工程的FalconEmpty已进入Play。正确原工程另包发送，不用官方预编译Blocks替代。
- 已按COLLEAGUE_HANDOFF.md装好Python环境；新包的vendor供AirSim客户端使用。供电稳定、不自动睡眠、无其他控制/FPGA压测程序。
- 新版保持温和八环境及输入检查，布局见MODERATE_V2.md。无需更改SD镜像、模型、板端程序或UE工程。

## 原主机一条采集命令

在项目根目录PowerShell运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\falcon_svr_8env_v1\run_all.ps1 -PythonPath 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
```

默认保存到`experiments/falcon_svr_8env_v1/results/moderate_v2`。原主机已记录的两次会跳过，本次更新没有修改冻结的飞行源码/配置，不必删数据或重飞。不要把旧v1/其他诊断目录传给此命令。

## 同事一条采集命令

新ZIP解压到新项目目录，先进入该目录：

```powershell
Set-Location -LiteralPath 'D:\FalconSVR\falcon_svr_colleague'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\falcon_svr_8env_v1\run_all.ps1
```

脚本优先选新项目的`.venv/Scripts/python.exe`。若venv装在旧目录，可显式`-PythonPath '实际venv的Scripts\python.exe'`，无需复制/重装venv；脚本会使用新项目的vendor。新主机不要复制原主机results来续跑。

## 什么时候继续，什么时候停止

- 成功、真实碰撞、未正确过门、任务超时、输入范围中止：落盘后自动进入下一配置，不重飞到成功。
- UDP/RPC/清理故障：停止，避免长时间生成混入后备控制的数据。连续时序失效也沿用原停止机制。修复后执行同一命令继续，不反复重启板端直到某次成功。
- Ctrl+C/掉电后：先恢复服务与AirSim，再在原目录执行同一命令。未完整落盘的attempt保留，下次另建attempt；已形成result的试验跳过。
- 处理完计划后自动退出，不需要每次手工启动。4608是配置试验数，不保证有4608条有效完整J；失败及越界比例必须报告。

## 主性能SVR用哪份数据

`datasets/E01..E08/svr_success_only.csv`是主性能SVR训练表：输入c/T/实际R，目标为完整成功路线的原J。真实失败保存在all_trials和逐帧日志中，用于标记不可行区域，不拿失败前短轨迹J混比。

`svr_failure_zero.csv`仍保留，但只是失败记0的工程扩展，不能冒充原J性能曲面。输入越界、混合后备、通信/时序失效不加入主性能表。终端沿用旧字段打印的`J=0`是失败惩罚值，并不表示它被加进主性能表。

画图入口默认已改为success_only。若某环境缺乏成功样本，该环境不会凭空形成有支撑的完整性能立方体；不要静默插值填补不可行区域。必要时在后续新批次缩小实验域，而非采集途中自动移动门或改参数。

## 发给同事及结果返还

`package_for_colleague.py`会打包run_all.ps1及全部依赖源码/元数据，不含旧结果、venv、UE引擎、SD镜像或板端运行库。原UE工程包继续单独发送，已经一致则无需重传。安装细节见COLLEAGUE_HANDOFF.md，布局/输入保护见MODERATE_V2.md。

采集结束后返还整个`results/moderate_v2`，包含manifest、profile、progress、trials和datasets；不要只发成功CSV或立方体截图。

此脚本不负责自动开机、启动UE或板端服务，不会自动重试物理失败，也不承诺所有高难配置成功。离线测试不等于已在同事设备上跑完4608次。
