# PS Planner Profiling Study

**2026-11-17最新：[配对状态事件触发与现场CPU长尾定位](../docs/PS_PAIRED_TRIGGER_AND_LIVE_TAIL_RESULTS_20261117.md)。** 最终可复核版本为 `results/ps_management_trigger_20261117_004/control_source/`；真板双机开/关各1334条交付，事件组23次提前规划，172项ARM回归通过。实际模型/周期/跳更仍未改变。现场异常已定位到`diagnose_impl`单次9.150ms线程CPU，相同输入40次ARM离线重放约0.169ms；尚未找到具体并发根因。旧下文按历史阶段阅读。

**2026-11-17最新：[完整管理计时和有界重验证实测](../docs/PS_MANAGEMENT_REVALIDATION_RESULTS_20261117.md)。** 下文描述早期核心离线剖析；现在新增`live/`和独立真实控制版本，未覆盖原v5。七次真实双机7736条均交付，但没有实际模型/周期/skip修改；新增58ms管理CPU长尾仍待定位。

## 新增源码和证据入口

| 文件 | 职责 |
| --- | --- |
| `live/scheduler.cpp` | 已通过完整差分的报告移动优化；仍是整任务PS/DPU模型 |
| `live/live_calendar.*` | 原日历＋快照/安装分段计时、一次最新状态复核及条件变量等待 |
| `live/revalidate.*` | 在最新日历中应用原未来变更；保留其他回路、固定补更及最终时间/版本检查 |
| `live/revalidate_tests.cpp` | 47项新增正反例；合成生命周期，不是飞行收益 |
| `live/management_bench.cpp` | 实际方法的合成生命周期计时，无RPC/DPU竞争 |
| `prepare_control.py`、`build_control.sh` | 从冻结交付生成独立入口并用原SDK/模型交叉编译 |
| `run_control_trial.sh` | 沿用原双机准备及独占DPU守卫的有限20s试飞入口 |
| `analyze_management.py` | 只读重算管理/日志阶段成本及实际动作 |
| `audit_live_context.py` | 冻结异常输入，逐候选核对ACK后配对观测等待区间 |

实测最终完整源码为`../results/ps_management_repair_20261116_004/control_source/`，其清单和构建/ARM回归在同批结果目录；原始前三轮与未等待的压力轮分别保留002/003批。核对旧工作时不要把47项新测试、50次合成bench、40次当前ARM异常回放和真实7736条DPU请求混为一种证据。

Ubuntu构建根`/home/cwc/dpu_work/windows_framework/ps_management_repair_20261116_001`，最终`control_source_003`/`control_build_003`；板端`/home/root/ps_management_repair_20261116_001/control_003/`。使用既有AXU3EGB SDK和冻结18档ELF，现场所有命令仍为ID11。

离线分析示例（工程根目录，输出必须使用新名称）：

```powershell
& $py controller_architecture_study/ps_planner_repair/analyze_management.py `
  --data controller_architecture_study/results/ps_management_repair_20261116_004/live_raw/proof_on_001 `
  --output controller_architecture_study/results/management_reanalysis_new.json
```

当前不建议直接扩大试飞。真实skip/补更未验收，下一步应先记录live规划内部异常阶段，并按配对状态到达触发管理；不放松原守卫强求触发。

## 早期核心剖析记录

2026-11-16。执行[PS修复计划](../docs/PS_PLANNER_REPAIR_PLAN_20261116.md)第一步的独立离线研究代码；不包含DNNDK、RPC或飞行控制入口。

**最新：[ARM实测结果与边界](../docs/PS_PLANNER_ARM_PROFILE_RESULTS_20261116.md)。** Ubuntu恢复后已完成两轮共14560条ARM核心回放，最终优化平均8.533→4.473 ms、P99 9.135→4.785 ms；原43/77项ARM回归通过。优化仍为独立核心版本，没有替换实时控制程序。

`prepare_profile.py`读取冻结Ubuntu交付，生成原始、阶段计时、报告移动优化、优化后计时四份源码副本，以及91份实际规划输入。原始交付文件保持不变。89份来自真实skip候选，另外两份为最长keep和普通keep。

`profile_replay.cpp`在输入解析完成后重复求解；校验完整决策对象，包括forecast、trace、模型、顺序和补更。浮点比较为绝对1e-10＋相对1e-12容差。核对/结果序列化位于计时区间之外。

`profile.hpp`按栈嵌套切换阶段，只累计互斥时间，避免simulation_report同时计入simulation。Linux使用单调墙钟和线程CPU时钟。探针开销需要与原始版本对照。Windows兼容时钟只用于本地验证，其线程CPU时间有较粗粒度，不能用于ARM归因。

当前阶段覆盖规划函数内部，**不包含**实时取得日历锁、构造快照、install或后台日志线程的成本。后者需要在ARM核心结果出来后，再对真实运行入口添加相应计时；本研究未宣称已经测完全部管理链。

报告移动优化将`simulate()`返回报告的7个已经构造好的JSON容器移入结果，并减少全组检查包装forecast和批准方案包装jobs/forecast的深拷贝。没有删除报告字段、修改调度/期限或省略检查。当前仅部署了独立离线版本，未部署到真实控制程序。

## 已完成

- Windows编译四个变体，91份完整决策逐项核对通过。
- 四变体各两批、每份输入10次，共7280条Windows回放通过；两批按相反版本顺序执行。
- 原版平均约0.893～0.930 ms，移动优化版约0.731～0.732 ms；这是Windows观测，不能外推ARM或实时长尾。
- ARM源码包和交叉编译脚本已用于实际编译/回放，最终源码在`results/ps_planner_profile_20261116_004/payload/`。

记录：[完整输入与源码清单](../results/ps_planner_profile_20261116_002/payload/manifest.json)、[Windows回放汇总](../results/ps_planner_profile_20261116_002/windows_summary.json)。

## 访问中断与恢复记录

此前`vitis-vm`（192.168.125.185）SSH在握手前关闭。用户说明虚拟机刚被挂起并恢复；随后Windows→vitis-vm→axu3egb通道恢复，已完成ARM工作，下面为实际采用的流程参考，不再是当前阻塞。

恢复后按以下顺序继续：

1. 上传最终`results/ps_planner_profile_20261116_004/payload.tar.gz`到Ubuntu全新目录，验证清单并解包。早期Windows反斜杠清单失败已修为POSIX路径，失败包保留。
2. 在Ubuntu执行`bash build_arm.sh <全新构建目录>`；使用现有`/home/cwc/dpu_work/axu3egb_sdk/`，不更换模型/系统。
3. 将四个ARM程序及inputs.json上传到板端全新研究目录，保存哈希、环境和运行身份。
4. 无飞行条件下依次交错运行原版/计时版/优化版/优化计时版，例如`./replay_baseline inputs.json 10 baseline_001.json`；输出文件使用新名字。
5. 回收CPU/墙钟分阶段结果，核对原始完整决策及探针开销；只有ARM结果支持后，再决定是否将优化用于控制版本。

这一步不需要启动AirSim、调用DPU或停止其他控制服务；运行前仍应记录CPU占用，区分独立与竞争条件。

早期Windows原始JSON的scope固定写了ARM，属于元数据文案错误；文件目录及Windows汇总明确标识实际平台，未作为ARM数据使用。后续源码改为平台无关scope，历史原始文件不重写。真实ARM身份由环境/二进制/命令记录核对。
