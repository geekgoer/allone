# 原生IO分核与后台通知修复结果

本轮由Windows Agent在用户授权后直接执行，日期为2026-11-05。完整派生汇总位于 `results/rt_splitio_stage_20261105/evidence.json`，原始试飞记录保存在各自目录。

## 本轮达到的状态

完成16轮六机自然飞行，共96架次、768次有效穿门，实际碰撞均为0。原生采样与命令下发已分到不同物理核，后台快照通知已消除等待消费者确认唤醒的握手。动态周期调整有真实安装及交付记录。

1.75m场景的两轮动态平均J比两轮固定55ms对照高约0.98%，同时增加了周期任务丢弃。1.60m场景的新配置安装为0。本轮尚无静态碰撞、动态救回的证据，也没有全程硬实时结论。

## 修改内容

### 原生采样与命令下发使用不同核

`probe_seven_continuous_timing.py` 在 `period_drop` 且 `period_io_policy=split` 时，将命令通道绑定到第一个IO逻辑核，采样通道绑定到第二个IO逻辑核。`CpuPlacement` 已按物理核心拓扑排除了同核超线程。实测绑定为命令CPU8、采样CPU10，前端CPU12，模拟XPU CPU14。

沿用原先预留的四个物理核。独立监测仍与采样共用CPU10。报告增加 `sample_io_placement` 和 `period_io_policy`，实际掩码可以回读核对。

试验入口默认使用 `--period-io-policy split`，需要复现共核对照时可显式传 `shared`。其他历史入口没有提供此配置时，继续采用原来的共核行为。

### 快照通知使用信号量

`joint_contract_shadow.py` 原先使用 `multiprocessing.Event`。本机Python的 `Event.set()` 调用 `Condition.notify_all()`，后者通过 `_woken_count.acquire()` 等待消费者确认已唤醒。前端因此可能等待后台进程获得执行机会。

现使用 `Semaphore.release()` 通知后台，后台用 `acquire(timeout=.050)` 接收。原来的固定共享内存、版本号和互斥复制继续负责数据完整性，后台只处理最新版本。新增 `snapshot_notification_ms` 单列通知耗时。

后续试飞中，1.75m两轮快照最大耗时为1.72/0.85ms，通知最大耗时为0.029/0.125ms。1.60m两轮快照最大耗时为3.48/1.68ms。测量显示该机制改善了快照长尾；这些仍是经验值。

### 前端线程优先级

该试验入口使用 `CpuPlacement(frontend_priority=1)`，将控制前端线程设为Windows `THREAD_PRIORITY_ABOVE_NORMAL`。进程保持原优先级类别。试验退出后恢复原线程优先级和CPU掩码，成功及异常退出路径均有回归检查。

`cpu_placement.json` 记录实际优先级0→1及恢复状态。该调整后仍观测到零星丢任务。

## 全部试验

下表的丢任务是周期任务未交付数。已丢弃任务保留在分母中。J使用现有脚本的实际飞行窗口评分，越大越好。所有行均零碰撞，物理穿门检查通过。

| 结果目录简称 | 组别 | J | 丢任务 | 计划释放至命令回执平均ms | 新配置安装组数 |
|---|---|---:|---:|---:|---:|
| splitio55 | 静态1 | 7.863949 | 4 | 5.764 | 0 |
| splitio55 | 静态2 | 7.860358 | 3 | 6.045 | 0 |
| splitio55 | 动态1 | 7.987308 | 37 | 8.173 | 3 |
| splitio55 | 动态2 | 7.732339 | 67 | 8.533 | 6 |
| splitnotify55 | 静态1 | 7.763720 | 63 | 13.728 | 0 |
| splitnotify55 | 静态2 | 7.834880 | 0 | 7.721 | 0 |
| splitpriority55 | 静态1 | 7.865282 | 3 | 7.857 | 0 |
| splitpriority55 | 静态2 | 7.838374 | 0 | 7.236 | 0 |
| nextstage175 | 静态1 | 7.843598 | 0 | 6.941 | 0 |
| nextstage175 | 动态1 | 7.890916 | 55 | 10.596 | 4 |
| nextstage175 | 动态2 | 7.941760 | 19 | 9.687 | 1 |
| nextstage175 | 静态2 | 7.835610 | 0 | 6.985 | 0 |
| nextstage160 | 静态1 | 7.850220 | 0 | 7.155 | 0 |
| nextstage160 | 动态1 | 7.865827 | 9 | 9.376 | 0 |
| nextstage160 | 动态2 | 7.844187 | 3 | 8.510 | 0 |
| nextstage160 | 静态2 | 7.817443 | 0 | 6.759 | 0 |

完整目录名如下。

```text
staggered190_6drones_splitio55_20261105_231441
staggered190_6drones_splitnotify55_20261105_232139
staggered190_6drones_splitpriority55_20261105_232541
staggered175_6drones_nextstage55_s6_20261105_232849
staggered160_6drones_nextstage55_s7_20261105_232849
```

旧的共核、前端EDF对照保留在 `staggered190_6drones_frontendedf55_20261103_200252`，其两轮静态丢弃28/29条，计划释放至回执平均12.38/15.12ms。分核首轮的原生命令线程启动等待均值约0.10/0.12ms。

## 验收和探索的区别

1.90m的首个分核试验通过了两轮静态首弯前验收，因此随后进行了动态对照。

加入信号量通知后的第一轮静态出现较多停顿，该次静态验收失败，动态没有启动。提高前端线程优先级后的两轮静态丢弃3/0条，但第一轮首弯前仍有违规，因此动态同样未启动。失败报告完整保留。

后续1.75m和1.60m试验用于探索完整静态/动态行为，执行时没有启用 `--require-static-baseline`，按静态、动态、动态、静态顺序完整记录。这些探索结果不能当作通过了原先的严格静态验收。周期任务截止和丢弃规则仍在执行。

## 动态实际改变了多少

1.75m两轮分别安装4/1组配置。25条带直接修改标记的控制命令完成。原生周期配置安装后还会持续生成后续任务，实际共有3858条成功交付的请求使用了不同于55ms的周期。多数周期处于约29–56ms，具体分布见 `requests.json`。

两轮静态平均J为7.839604，动态为7.916338，提高约0.98%。但静态总丢弃0条，动态总丢弃74条，平均响应也增加。当前不能据此声称动态在控制性能和实时性上同时占优。这里只观察到了一个固定场景、两次重复中的小幅J改善。

1.60m、seed7两轮新配置安装都为0，成功交付的请求仍用55ms。后台主要因流水线预测不可行，以及复核或下次释放所剩时间不足而拒绝。两组J的小差异不归因于新配置。

## 剩余问题和下一步

1. 继续区分控制更新需求与当前任务剩余期限。1.75m中，直线回路也被调到约30–40ms，整体任务量增加，说明周期需求函数还需要核对。应先回放快照检查时间索引和分配依据，再决定是否改变控制律。
2. 对1.60m拒绝快照拆分真实剩余工作、过于保守的经验成本，以及全组提交的释放竞争。不能通过忽略其他回路或放宽期限提高采用率。
3. 下一轮比较应同时报告实际采用、J、丢弃、最长命令间隔和物理结果。当前静态是已有离线模型配置上的固定55ms对照，本轮未重新对当前路线搜索全局最优静态配置。
4. 测试期间整机内存使用率曾达91–94%。只读快照没有同步页故障跟踪，不能把某个具体停顿全部归因于换页。后续复核应记录内存压力。

当前代码仍为Windows原型，独立模型保持不变、段级EDF执行。模型复杂度选择、当前237D超网的实际控制和FPGA时序需分别验证。

## 离线检查和复跑入口

分核改动完成时，周期丢弃、原生IO、CPU绑定和全局重配置的45项检查通过。后续通知和线程优先级改动完成后，CPU绑定、后台进程通信、CPU/XPU、提交保护和周期丢弃的42项检查通过，两组有重叠，不相加为独立测试数量。

保留12机Blocks进入Play，且窗口未最小化后，在工程根目录执行。

```powershell
$py = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
$base = '.\controller_architecture_study\results\boundary_takeover_20261101_v1'
$out = '.\controller_architecture_study\results\splitio55_recheck_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py -u -m controller_architecture_study.run_reviewed_rate_trial --boundary-report (Join-Path $base 'boundary\report.json') --ranker (Join-Path $base 'ranker\ranker.npz') --seed 6 --vehicles 6 --period-ms 55 --period-drop --gate-width-m 1.90 --staggered-turns --spread-releases --joint-only --require-static-baseline --trace-frontend --period-io-policy split --output $out
if ($LASTEXITCODE -ne 0) { throw "试验异常，请查看 $out" }
```
