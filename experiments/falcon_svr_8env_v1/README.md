# 八环境 FALCON baseline：可中断实测 SVR 数据采集

> 2026-10-31：默认已切换为温和八环境v2及INT8输入预检查，请先读`MODERATE_V2.md`。下文v1布局/目录为历史说明，旧数据保留，不能用新代码续接旧目录。

现有六个12-D DNN、UDP5008、5 m/s不变。没有重新训练网络，没有给飞行加入避障/安全修正，也没有强制能力轴单调。新增布局、采样网格、失败记录、续跑和绘图在这个独立目录中。

## 1. 采什么

| 环境 | 前后门距 dx / m | 每次横向变化 dy / m | 高度变化 dz / m |
|---|---:|---:|---:|
| E01 直线 | 10 | 0 | 0 |
| E02 缓弯 | 10 | 0.8 | 0 |
| E03 中等混合变化 | 10 | 1.2 | 0.4 |
| E04 大横向折线 | 8 | 2.4 | 0.4 |
| E05 急转 | 6 | 3.2 | 0.6 |
| E06 紧凑急转 | 4.5 | 4 | 0.6 |
| E07 高压力急转 | 3.5 | 5 | 0.8 |
| E08 极端压力 | 2.5 | 6 | 0.8 |

每个环境5个门，门洞仍为4×4 m，厚度0.15 m，安全通过余量0.25 m。门中心从NED z=-3 m起交替向上变化，避免把横梁埋入地面。旧初始状态统一向上平移1 m，保持相对高度条件；不修改旧初始状态文件。

每档包含两条冻结变体：横向正/负镜像，分别使用旧初始状态0/1。对于所有模型、T和R，相同变体的布局和初始条件完全一致。短门距自动缩短参考的门前/门后距离，避免默认2+1 m大于门距而让参考折线倒退；门平面通过判定仍保留。

这是一组**安全压力测试扩展**，不是论文原三档环境的精确重建；难度编号不保证实测性能严格递减，也不保证任何特定方法能通过全部高难场景。不能为了保证撞门而改标签、移动已冻结的门或删除成功试验。

“epsilon_transverse_m”仍保存实际相邻门的平均y/z Manhattan位移；前向门距另存。不要把不同门距的信息全部压成一个epsilon，也不要把E01..E08数字直接当连续物理量。

## 2. 网格与耗时

```text
模型：6个冻结模型，c=1/test_normalized_mse，不按参数量手动排序
T：60,70,80,90,100,120,140,160 ms（8个）
目标R：25,30,35,40,45,50 ms（6个）
环境：8个
轨迹变体：每档2个

6×8×6×8×2 = 4608次完整或失败的实飞试验
每档环境：288个不同c/T/目标R坐标、576次试验
```

所有目标满足R+3 ms容差<T，避免用R>=T的不可行区域填充立方体。c轴只有六个真实模型层，不能靠增加T/R采样制造更多实测c层。绘图输出8张实测立方体，未采样点保持灰色；不插值伪装成实测。

按历史每次18～32秒，约23～41小时；困难场景可能提前失败，也可能耗到25秒任务上限。批处理试验结束后采用AirSim复位而不等待降落，可能更快，但不是实时承诺。`progress.json`按实际墙钟耗时持续更新ETA。建议接电源、关闭系统自动休眠并预留至少10 GB；仅适用于仿真，不连接实体飞行器。

## 3. R怎样得到

第一次运行先对每模型做5次预热、100次冻结输入推理，保存`latency_profile.json`。这是板端+UDP原生时延的经验分布，不是证明的WCET，也不含整个AirSim主机循环。若最大实测RTT加3 ms主机余量已接近最低目标R，程序停止，不悄悄改变某个模型的目标。

实飞每帧先完成状态处理、MPC后备计算和FPGA推理，再等待到“状态获取完成时刻+目标R”才开始发送。等待长度按当帧实际剩余时间计算，不是用离线平均耗时死扣固定睡眠。

实际`R_max_ms/R_mean_ms/R_median_ms/R_std_ms/R_p95_ms/R_p99_ms`来自主机时间戳，包含等待及发送调用；不把目标R冒充实测R，也不能声称它是物理引擎内部精确施加时刻。采样保留整段R序列供以后分析抖动。AirSim `ClockSpeed`必须为1，设置和版本会被冻结；不能通过提高仿真倍速来缩短此实时实验。

## 4. 运行命令

手动打开Blocks并点击Play；网线接PS_ETH；保持正确六模型服务`192.168.77.2:5008`运行；停止其他控制无人机/高频访问FPGA的脚本。采集使用AirSim Python，不使用PyTorch或CUDA训练。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py .\experiments\falcon_svr_8env_v1\collect.py --enable-fpga-actuation
```

一条命令自动测原生时延、冻结计划并采集。它会实际飞行，不是影子模式，不会自动再开一个AirSim，也不会训练DNN或SVR。

想本次只生成计划中的前96个新试验（不是额外重复测试）：

```powershell
& $py .\experiments\falcon_svr_8env_v1\collect.py --enable-fpga-actuation --limit 96
```

之后去掉`--limit`运行原命令继续。只想先测UDP、不启动AirSim飞行：

```powershell
& $py .\experiments\falcon_svr_8env_v1\collect.py --profile-only
```

不访问板端或AirSim，仅检查计划：`& $py .\experiments\falcon_svr_8env_v1\collect.py --dry-run`。

## 5. 中断和失败

- Ctrl+C后重新执行同一命令即可继续；不用加`--resume`。
- 每次完成/真实失败后，先落盘逐帧日志，再原子保存`result.json`作为完成标记。
- 碰撞、越过门洞外、任务超时均记录一次，续跑不会把它重飞到成功。
- 中断/网络故障的原始attempt文件保留；下次用新的attempt文件名重做尚未形成有效结果的试验，不覆盖旧日志。
- RPC/UDP故障停止采集，修好后续跑。连续3个试验发生周期截止期错过也停止，避免长时间生成主机失速数据；已记录的时序不合格试验不自动重试。
- 配置、关键源码、模型目录中的归一化和初始状态有哈希，执行器标定值也会冻结。长跑期间不要修改这些文件；若确需改变协议，使用新的输出目录，不能悄悄续接成同一实验。

默认输出：本目录`results/full_v1/`，包含冻结配置/布局/计划、时延profile、progress、逐试验result、每次attempt和逐周期日志。

## 6. 失败样本怎样用于SVR

每个环境输出两套CSV，不能混淆：

1. `datasets/E01/svr_success_only.csv`等：`J_target`为原`J_paper_style`，仅完整纯FPGA成功路线有效；这是成功条件下的性能，不能代表碰撞风险。
2. `datasets/E01/svr_failure_zero.csv`等：成功沿用原J；物理碰撞、门洞外通过、任务超时显式记0。它是**failure_zero_v1工程扩展**，不是论文未公开失败处理的假定复原。

失败前短轨迹的J仅存为`partial_J_diagnostic`，不能用它奖励提前撞门。网络错误、混合MPC后备和时序失效不伪装成纯FPGA标签；所有记录仍在`all_recorded_trials`或`censored_infrastructure_attempts`中。任务超时与真实碰撞分开计数，超时不自动等于发生了碰撞。

当前没有给FALCON偷偷加入安全控制层。若后续安全方法与baseline比较，必须使用同一测试场景和同一评价口径，不能通过更换标签定义制造收益。两套CSV中的目标政策有明确字段；旧的“只接受正J”训练器不能不加修改直接读取failure_zero数据。

建议后续对每个冻结环境分别拟合`J=C_E(c,T,R)`，得到8个SVR/8个立方体。若合成一个跨几何模型，应显式保留dx/dy/dz，而不是只使用单一epsilon。保留variant做轨迹级留出；两条变体只是有限验证，不是对所有布局的泛化证明。

## 7. 导出和画图

正常结束、中断或故障停止后会自动导出当前已完成部分。画图另用原训练Python执行，避免给AirSim环境添加Matplotlib依赖：

```powershell
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
& $trainPy .\experiments\falcon_svr_8env_v1\export_and_plot.py --plot --policy failure_zero
```

输出8张`results/full_v1/figures/failure_zero/cube_E01.png`至`cube_E08.png`。成功点为圆点、物理任务失败为X、未测为灰点，八图共享色标。原J成功条件图可改`--policy success_only`。请在采集暂停/结束后导出，避免两个进程同时改写派生表。

本目录`layout_preview.png`是冻结布局预览，不是飞行结果。重新生成：

```powershell
& $trainPy .\experiments\falcon_svr_8env_v1\export_and_plot.py --preview-layouts
```

## 8. 与分段执行的关系

论文III-B、式(5)-(7)将连续层组成accelerator segments，并计入计算/搬运/切换工作量。分段不是在每一段结束后都输出飞行控制。

若同一状态输入最终产生完全相同的控制输出，且每帧采样/施加时刻、丢包和整条R时序也相同，分段本身不会改变该确定性闭环。但只保证平均R或最大R相同不够：分段可改变多任务排队、阻塞、抖动及额外搬运开销。

本采集是整模型不可抢占，人工等待建立的是固定延迟性能样本。它可作为以后分段调度的候选性能代理，但不能直接证明分段时的多机控制性能或WCRT。后续需要真实分段profile/RTA、核对中间张量及量化输出，并用实际分段调度的R轨迹做HIL确认。当前一次`dpuRunTask()`不能仅加一个命令参数就变成可抢占分段；本次不更换板端服务。

原文来源：用户提供的`E:/BaiduNetdiskDownload/vitis-ai6.1/tmp/pdfs/FALCON.pdf`，PDF第6页（印刷页411）。

## 9. 已验证与未验证

离线检查覆盖计划数量/均衡、确定性布局、门距/参考几何、原环境合同兼容、失败与RPC故障区分、失败不重飞、哈希续跑和原J/门判定回归；图已作布局检查。尚未代用户运行本协议的真实AirSim/FPGA飞行，不能保证所有高压力环境成功或一定发生撞门。
