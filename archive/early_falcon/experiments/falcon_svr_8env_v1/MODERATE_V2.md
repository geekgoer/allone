# 温和八环境v2＋Windows INT8输入预检查

> 连续全量采集现已提供`run_all.ps1`，请优先按`RUN_ALL.md`一条命令运行；下文limit命令仅供可选诊断。主性能SVR使用svr_success_only.csv，画图默认也是success_only；failure_zero仅为独立工程扩展。

2026-10-31。不修改板端程序，不重训/量化六个DNN。仍使用current5ms UDP5008、原UE4.27.2工程的FalconEmpty、5 m/s和原门框/碰撞/净空判定。

## 布局及数据口径

新增默认`collection_config_moderate_v2.json`，原`collection_config.json`保留。前四档不变，后四档缓和：

| 环境 | 前向门距/m | 横向变化/m | 高度变化/m |
|---|---:|---:|---:|
| E01 | 10 | 0 | 0 |
| E02 | 10 | 0.8 | 0 |
| E03 | 10 | 1.2 | 0.4 |
| E04 | 8 | 2.4 | 0.4 |
| E05 | 8 | 2.6 | 0.4 |
| E06 | 7 | 2.8 | 0.5 |
| E07 | 6 | 3.0 | 0.5 |
| E08 | 5 | 3.2 | 0.6 |

八环境×两变体×六模型×8T×6R仍为4608次。`layout_preview.png`是旧v1布局，不代表新配置。缓和不保证所有状态可飞、输入永不越界或性能单调。

## 输入保护

本版配置冻结六个输入scale=32，依据当前板端MODEL日志；使用前必须核对，不适用于其他scale/模型包。FLCN/v2不能自动发现scale。

每次profile、预热和正式请求前，全部12维均检查有限性及`-128 <= X_normalized*scale <= 127`。scale32对应保守区间`[-4,3.96875]`。不裁剪、不改变归一化、模型或有效输入；该范围也不代表模型的训练覆盖范围。

越界包不发出，抛出InputRangeError并终止该试验，沿用采集器复位清理。summary/result中的`trial.input_range_violation`保存模型、周期、12维输入、乘scale结果、异常维度、`packet_sent=false`。该周期不派发新控制命令。

outcome=`input_range_exceeded`，不是撞门/网络故障，不记J=0，不加入两套SVR回归目标。记录仍在all_trials/all_recorded_trials，progress及数据集覆盖表单独计数。续跑跳过已记录越界试验，不重飞到成功。若此类排除频繁，应暂停解决部署覆盖，不能隐藏排除项或宣称完整建模安全性。

UDP/RPC故障仍停止；混合MPC后备不作纯FPGA标签。profile阶段越界直接停止，无飞行标签。本保护仅覆盖采用guarded客户端的入口，旧ZIP/独立客户端仍可触发SDK退出；它不是防撞控制器，板端长期仍宜处理坏输入。

## 先在原主机实测一次

若上次SDK退出导致服务终止，重启服务；已经运行则不重复启动：

```sh
cd /home/root/board_package_current5ms_12d_six_v1
./falcon12_current5ms_dnndk_udp_server --port 5008
```

手动打开正确工程、FalconEmpty并Play。PowerShell：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
$v2Root = '.\experiments\falcon_svr_8env_v1\results\moderate_v2'

& $py .\experiments\falcon_svr_8env_v1\collect.py `
    --config .\experiments\falcon_svr_8env_v1\collection_config_moderate_v2.json `
    --enable-fpga-actuation --limit 1 --output-root $v2Root
```

第一个仍是E08_v1/model3/T60/R25，采用新布局。检查outcome及板端是否仍在运行。`input_range_exceeded`表示拦截，不等于通过。单次确认后，把`--limit 1`换成`--limit 95`补完首个T/R切片（八环境×两变体×六模型），核对排除/失败/通信分布，再决定全量：

```powershell
& $py .\experiments\falcon_svr_8env_v1\collect.py --enable-fpga-actuation --output-root $v2Root
```

上述96次计入4608总计划，不是额外重复测试。agent不代运行实际飞行。

默认配置及默认结果目录均已切到v2。不要复用旧full_v1/diagnostic/flight_diagnostic目录；源码哈希变化会拒绝混接，所有旧数据保留。同一电脑/路径/配置中Ctrl+C后重复命令即可继续。

画图默认读取新目录，也可显式`export_and_plot.py --root $v2Root --plot --policy failure_zero`。AirSim解释器没有Matplotlib时使用原训练解释器，不在采集长跑中升级依赖。

## 同事更新

运行`package_for_colleague.py`重新打包，把新ZIP发给同事。解压到新项目目录，沿用已装依赖的venv，重新将PYTHONPATH设为新目录vendor。不要覆盖正在运行的旧工程或将旧结果复制进新批次；$proj/$py换成同事实际路径。

本版已做离线回归、-158输入重放和精简包检查，尚未实测新八档飞行，不能保证全量采集通过。只在用户观察少量实飞后再扩大。
