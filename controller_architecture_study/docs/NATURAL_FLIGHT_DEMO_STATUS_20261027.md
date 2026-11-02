# AirSim 演示入口与无扰动筛选结果（2026-10-27）

当前结论：已有人工侧推的静态触框／动态救回案例曾重复出现；本轮尚未找到稳定的、无人工扰动的同类案例。不能把一次零碰撞通过包装成已经稳定救回。

## 已有人工侧推演示

在12机工程已经进入Play时，复制以下整段命令。程序使用其中7架，先跑两个无侧推的正常窗口，再跑两组有侧推的静态／动态对照；第二组交换顺序。各轮之间会复位，飞行中不暂停或移动飞机位置。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
$out = Join-Path $proj ('controller_architecture_study\results\pulse_demo_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $trainPy -u -m controller_architecture_study.probe_frozen_feedback_control --output $out
if ($LASTEXITCODE -ne 0) { throw '演示未完整执行，请保留输出目录中的报告' }
```

`normal_static`、`normal_adaptive`是正常窗口；`r1_static`、`r1_adaptive`、`r2_adaptive`、`r2_static`含人工侧推。侧推是在Drone1飞行约1.207秒时额外增加1.95m/s横向速度。它是测试条件，不是framework的组成部分。

该入口使用此前成功序列的冻结源码与配置；先前重复结果不保证每次Windows实测都完全一致。固定模型为237D的9×1024普通MLP，固定周期40ms，动态最小释放间隔20ms。它是固定模型的更新时间／调度对照，不是完整FALCON最优配置复现。

## 本轮自然飞行结果

自然飞行使用已有的环境48保存航线，7机、每机5门、名义速度5m/s。飞机初始化在参考航线的静止起点，之后不添加风、侧推、位置或姿态修改，也没有额外人为延时。静态与动态配对使用相同模型、航线和初态；两边都配备相同的20ms观测服务。推理在CPU上分段运行。

碰撞数量指发生过真实AirSim碰撞事件的无人机数量，而不是保守净空检查的失败数量。

| 航线／门洞净宽 | 静态触碰架数 | 动态触碰架数 | 结果 |
|---|---:|---:|---|
| 三维连续弯道15／2.0m | 7 | 7 | 两边均触框 |
| 平面连续弯道13／2.0m，首次 | 7 | 0 | 动态完成，但保守净空不通过 |
| 三维连续弯道15／1.6m | 7 | 7 | 两边均触框 |
| 平面连续弯道13／2.0m，反序复核 | 7 | 7 | 首次观察不能稳定复现 |
| 平面连续弯道13／2.2m | 0 | 0 | 两边均完成并通过保守净空检查 |

前四行固定周期60ms；最后一行固定周期80ms。动态名义40ms，随状态在20～40ms之间调整。60ms来自此前正常窗口的局部候选选择，未在新航线上重新优化；80ms仅是已有周期候选的一次时序对照。因此不能将本表的静态一栏直接命名为“FALCON最优方案”。

2.0m平面弯道中，门3横向误差约为静态0.54m、动态0.53m，差异只有厘米量级，尚未形成稳定避碰余量。首次动态零碰撞时仍未满足原有的机身包络加净空标准；原始`success=False`和`J=null`均原样保留。没有放宽判据、删去失败复核或把未知状态当成通过。

2.2m对照中的当前评分实现总J为静态0.436036、动态0.467770。动态仍有严格期限违规，不能把“飞过了”称为完整实时验收通过。

筛选在平面弯道13／1.6m的动态轮发生RPC超时后中止，剩余预列候选没有运行。另一个早期启动失败是起始参考时间为负，发生在正式飞行前；已改为所有飞机从参考零时刻静止开始。两类失败记录均保留。

## 自行观察无扰动对照

下面运行2.0m平面弯道。先静态，结束后按Enter，再动态；这是尚未稳定救回的候选，不承诺动态零碰撞。终端会打印首次`COLLISION`的飞机和门框对象；轻微擦框可能仍继续飞行，不一定会明显坠落。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -u -m controller_architecture_study.probe_natural_demo --case "$proj\controller_architecture_study\results\natural_demo_search_20261027_v2\screen_env48_route13_width2\case.json" --arm both --pause-between
if ($LASTEXITCODE -ne 0) { throw '对照未完整执行，请保留末尾输出与报告' }
```

只看其中一轮时，将`--arm both`改为`--arm static`或`--arm dynamic`。保持Unreal已Play、未最小化、后台降频选项关闭；不要同时运行另一组试飞。

## 原始记录

- 结构化汇总：`controller_architecture_study/results/natural_demo_status_20261027.json`
- 首轮筛选：`controller_architecture_study/results/natural_demo_search_20261027_v2`
- 反序复核：`controller_architecture_study/results/natural_demo_confirm_20261027_v1`
- 2.2m对照：`controller_architecture_study/results/natural_demo_screen_20261027_v3`
- 演示源码：`controller_architecture_study/probe_natural_demo.py`

本轮未改训练权重、归一化、原有碰撞与净空判据，也未进行FPGA迁移。
