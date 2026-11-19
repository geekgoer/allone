# FALCON Windows AirSim ↔ FPGA 联调状态与执行流程

更新日期：2026-09-10

## 1. 当前结论

Windows 端 FLCN/v1 客户端、影子后端、单机接入和三机顺序调度代码已经完成，并通过 53 项离线测试。

阶段 A 已于 2026-09-10 通过。板端重新开机并启动服务后，Windows 冻结黄金单包结果为：

- `status=PASS`；
- `max_abs=1.2328338616640622e-07`，满足 `<=2e-6`；
- `timeouts=0`、`retries=0`、`protocol_errors=0`；
- 单包 `RTT=5214 us`、`DPU=1078 us`、`server_total=1080 us`。

阶段 A 证据位于 `dnn_runs/iid_path_v1_depth10_width512/fpga_udp_smoke_windows.json`。

阶段 B 单机 AirSim 影子模式也已通过：5/5 穿门，503/503 个飞行周期均收到新鲜 FPGA 回包，timeout/retry/协议错误/fallback/裁剪/deadline miss 均为 0；RTT mean/P99/max 为 1.613/1.935/3.488 ms，DPU 时间均值为 1.035 ms，FPGA-vs-CPU 最大绝对差为 0.1141。证据为 `fpga_shadow_single.json` 和 `fpga_shadow_single.jsonl`。

阶段 C 单机 FPGA 真闭环已通过：5/5 穿门，503/503 个周期实际施加的控制均来自 FPGA；fallback/timeout/retry/协议错误/裁剪/deadline miss 均为 0。RTT mean/P99/max 为 1.654/1.886/3.300 ms，DPU 时间均值为 1.035 ms。证据为 `fpga_closed_loop_single.json` 和 `fpga_closed_loop_single.jsonl`。这证明 Windows AirSim ↔ AXU3EGB DPU 的单机真闭环已经打通。

阶段 D 三机 FPGA 真闭环已通过：三架无人机 15/15 穿门，573 个控制周期、1779/1779 个 UDP 请求成功；fallback/timeout/retry/协议错误/裁剪/deadline miss 均为 0。Windows↔FPGA 闭环阶段已完成，最终证据、控制周期说明与后续 Ubuntu 工作见 `WINDOWS_AIRSIM_FPGA_CLOSED_LOOP_FINAL_REPORT.md`。

## 2. 已完成的 Windows 文件

- `fpga_udp_client.py`：严格 FLCN/v1、大端编码、全局 sequence、同报文重试、来源/长度/时间戳/status/finite 校验和统计。
- `fpga_udp_smoke_test.py`：冻结黄金向量单包验证，PASS/FAIL 都写 JSON 报告。
- `fpga_shadow_backend.py`：`cpu_dnn`、`shadow_fpga`、`fpga` 三种后端；FPGA 失联时只允许当周期 CPU DNN fallback。
- `fpga_closed_loop_config.json`：板端 IP、端口、10 ms timeout、一次重试、20 次 warm-up。
- `run_single_drone_dnn_gates.py`：新增单机 shadow/FPGA 显式模式和逐周期 JSONL。
- `run_three_drones_dnn_gates.py`：新增三机 shadow/FPGA 模式；共享一个 socket 与全局 sequence，三机各自独立构造 25 维输入并顺序请求 FPGA。
- `tests/test_fpga_udp.py`：协议长度/字节序、原报文重试、shadow 不接管执行器、FPGA 裁剪测试。

验证命令与结果：

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

结果：`Ran 53 tests ... OK`。

## 3. 先恢复 Windows ↔ AXU3EGB 实体链路

在板端确认服务正在运行，并看到类似：

```text
READY protocol=FLCN/v1 bind=0.0.0.0:5005 kernel=falcon_25x4 ...
```

在 Windows PowerShell 检查实体接口：

```powershell
netsh interface ipv4 show interfaces
ipconfig
route print -4
```

要求：

1. 与板卡连接的实体以太网接口必须为 `connected`；
2. 该接口 IPv4 必须为 `192.168.77.1`，子网掩码 `255.255.255.0`，不需要默认网关；
3. 路由表必须出现 `192.168.77.0  255.255.255.0  On-link  192.168.77.1`；
4. 若 Mihomo/TUN 仍劫持该网段，将 `192.168.77.0/24` 设置为直连绕过，或联调时临时退出 Mihomo；
5. Windows 防火墙需允许此 Python 解释器的 UDP 出站和对应回包。

不要只用 `ping` 判断成功；最终以阶段 A 的协议回包为准。

## 4. PowerShell 公共变量

每次重启后执行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
```

## 5. 阶段 A：Windows UDP 黄金烟雾测试

实体网络和板端 READY 后执行：

```powershell
& $trainPy .\fpga_udp_smoke_test.py `
  --host 192.168.77.2 `
  --port 5005 `
  --timeout-ms 1000 `
  --retries 0 `
  --report .\dnn_runs\iid_path_v1_depth10_width512\fpga_udp_smoke_windows.json
```

唯一通过条件：`status=PASS`、无 timeout，并且 `max_abs <= 2e-6`。未通过时不要启动 AirSim 影子或真闭环。

## 6. 启动可见 AirSim

先在 CMD 中打开编辑器：

```cmd
"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"
```

编辑器打开后点击 **Play**，然后回到 PowerShell 执行下面的控制命令。

## 7. 阶段 B：单机 AirSim 影子模式

此阶段实际施加的始终是 CPU FP32 DNN 输出；FPGA 只计算和记录，不会控制无人机。

```powershell
& $trainPy .\run_single_drone_dnn_gates.py `
  --model .\dnn_runs\iid_path_v1_depth10_width512\controller_scripted.pt `
  --normalization .\dnn_runs\iid_path_v1_depth10_width512\normalization.json `
  --controller-backend shadow_fpga `
  --fpga-config .\fpga_closed_loop_config.json `
  --fpga-log .\dnn_runs\iid_path_v1_depth10_width512\fpga_shadow_single.jsonl `
  --layout-root .\teacher_dataset_full_route_continuous_v2 `
  --initial-state-id 0 `
  --device cpu `
  --epsilon 0.8 `
  --seed 0 `
  --speed 2.0 `
  --per-gate-timeout 30 `
  --gate-count 5 `
  --x-start 10 `
  --x-spacing 10 `
  --gate-width 4 `
  --gate-height 4 `
  --gate-thickness 0.15 `
  --gate-clearance 0.25 `
  --gate-asset 1M_Cube `
  --keep-gates `
  --report .\dnn_runs\iid_path_v1_depth10_width512\fpga_shadow_single.json
```

通过条件：5/5 穿门、无碰撞、无协议错误和最终 timeout、FPGA 与 CPU 差异量级符合已知 INT8 基线、无持续饱和/裁剪。

## 8. 阶段 C：单机 FPGA 真闭环（仅 B 通过后）

唯一关键变化是显式指定 `--controller-backend fpga`：

```powershell
& $trainPy .\run_single_drone_dnn_gates.py `
  --model .\dnn_runs\iid_path_v1_depth10_width512\controller_scripted.pt `
  --normalization .\dnn_runs\iid_path_v1_depth10_width512\normalization.json `
  --controller-backend fpga `
  --fpga-config .\fpga_closed_loop_config.json `
  --fpga-log .\dnn_runs\iid_path_v1_depth10_width512\fpga_closed_loop_single.jsonl `
  --layout-root .\teacher_dataset_full_route_continuous_v2 `
  --initial-state-id 0 `
  --device cpu `
  --epsilon 0.8 --seed 0 --speed 2.0 `
  --per-gate-timeout 30 --gate-count 5 `
  --x-start 10 --x-spacing 10 `
  --gate-width 4 --gate-height 4 --gate-thickness 0.15 `
  --gate-clearance 0.25 --gate-asset 1M_Cube --keep-gates `
  --report .\dnn_runs\iid_path_v1_depth10_width512\fpga_closed_loop_single.json
```

每次没有得到匹配的新鲜 FPGA 回包时，同一周期使用 CPU DNN fallback；绝不使用上一周期 FPGA 输出。

## 9. 阶段 D：三机 FPGA 真闭环（仅 C 通过后）

```powershell
& $trainPy .\run_three_drones_dnn_gates.py `
  --model .\dnn_runs\iid_path_v1_depth10_width512\controller_scripted.pt `
  --normalization .\dnn_runs\iid_path_v1_depth10_width512\normalization.json `
  --controller-backend fpga `
  --fpga-config .\fpga_closed_loop_config.json `
  --fpga-log .\dnn_runs\iid_path_v1_depth10_width512\fpga_closed_loop_three.jsonl `
  --device cpu `
  --epsilon 0.8 --seed 0 --speed 2.0 `
  --group-spacing 6 --gate-count 5 `
  --x-start 10 --x-spacing 10 --base-z -2 `
  --per-gate-timeout 30 `
  --gate-width 4 --gate-height 4 --gate-thickness 0.15 `
  --gate-clearance 0.25 --gate-asset 1M_Cube --keep-gates `
  --report .\dnn_runs\iid_path_v1_depth10_width512\fpga_closed_loop_three.json
```

最终验收：15/15 有效穿门、无碰撞、无任务 timeout，并明确报告 CPU fallback 次数。只要 fallback 不为零，结果就不能称为 `FPGA-only`。

## 10. 日志内容

单机和三机 JSONL 每周期记录：无人机、周期、门/phase、25 维物理输入、25 维归一化输入、CPU/FPGA/施加输出、逐维差、sequence、发送/接收单调时间戳、RTT、DPU/服务时间、retry、timeout、fallback、裁剪、deadline miss、碰撞和穿门事件。

最终 JSON report 汇总：周期数、fresh/fallback/裁剪次数、UDP 累计统计、FPGA-vs-CPU 误差，以及 RTT、DPU、服务时间的 mean/P50/P95/P99/max。
