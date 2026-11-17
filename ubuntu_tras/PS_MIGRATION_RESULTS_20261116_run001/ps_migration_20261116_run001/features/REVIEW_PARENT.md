# 父控制器代码独立审核（2026-11-16）

审核范围：`local_dpu.cpp/.hpp`、`dpu_check.cpp`、`with_exclusive_dpu.sh`。仅只读审核根文件，建议由父Agent修复；本子任务只在`features/`写测试及本记录。以下是审核时快照发现，父文件可能随后已修复，应以复核为准。

## P1：临时JSON子对象的range-for悬垂引用（已实证）

审核时`dpu_check.cpp`存在：

```cpp
for (const auto& row : read_json(problem)["inputs"]) { ... }
```

在C++14中，`operator[]`返回子对象引用，不会延长整个临时JSON的生命周期。进入循环前临时root已析构，迭代器访问释放后的map元素。板端日志`logs/local_dpu_board_validation.log`恰好在18模型黄金数据全部通过后报`ERROR cannot use operator[] with number`，与该路径一致。

已用包内同一个json.hpp独立构造最小复现：`features/test_json_lifetime.cpp`。安全版本默认返回0；`--unsafe`原模式在ASan下明确报`heap-use-after-free`并返回1，不是数值误差或DPU错误。

建议固定为：

```cpp
const auto problems = read_json(problem);
for (const auto& row : problems.at("inputs")) { ... }
```

保留原板端失败目录、日志，不原地覆盖。修复后使用新输出目录复跑3条问题输入，仍应得到原始`status=3`及其维度；不裁剪输入以获得成功。

ASan复现命令（纯主机，不访问DPU）：

```bash
BASE=/home/cwc/dpu_work/windows_framework
NEW="$BASE/ps_migration_20261116_run001"
g++ -std=c++14 -O1 -g -fsanitize=address -fno-omit-frame-pointer \
  -I "$BASE/route_b_ps_handoff_20261116/third_party_reference/AirSim/MavLinkCom/common_utils" \
  "$NEW/features/test_json_lifetime.cpp" -o "$NEW/features/test_json_lifetime_asan"
ASAN_OPTIONS=detect_leaks=0 "$NEW/features/test_json_lifetime_asan"
# 下一条为明确预期失败，用来证明原模式的错误；不是待上板程序。
ASAN_OPTIONS=detect_leaks=0 "$NEW/features/test_json_lifetime_asan" --unsafe
```

`detect_leaks=0`仅因本主机执行器ptrace下LeakSanitizer不支持，不影响ASan的UAF检测。实测日志：`json_lifetime_safe.log`与`json_lifetime_unsafe_expected_failure.log`。

## P1：服务恢复缺少信号/子进程生命周期保证

审核时wrapper仅`trap restore EXIT`，前台直接`"$@"`。异常退出码通常会触发EXIT恢复，但显式SIGINT/SIGTERM、SSH断开或调用者只向wrapper发信号时，没有显式结束/等待子进程再恢复服务的保证。若实验子进程仍持有DPU，恢复的旧UDP进程可能同时访问；若shell直接被信号终止，恢复路径也缺少明确可测的约束。

建议：记录子进程PID；分别处理INT/TERM/HUP，将信号转发给**自己启动的**实验子进程并wait；等其退出且释放DPU后仅恢复已确认的旧服务。不要force-kill未知进程。trap函数加一次性防重复标志，保留原退出码，避免多次恢复。SIGKILL/断电无法保证trap执行，需明确为恢复边界。

审核时`restore`仅nohup并记录新PID，没有验证新服务READY/绑定成功。应在有界时间内确认新PID仍存活、服务日志READY和预期port/模型身份，并落盘`restore_status`；恢复失败也需给用户明确报告，不能仅有PID就写“已恢复服务”。

目前扫描只识别`/home/root/board_package_requant18_release001/pool237_runner`。若另一个模型包服务或本实验上轮程序仍持有`/dev/dpu`，不会被拒绝，不能宣称独占计时。建议只读扫描/proc/*/fd的`/dev/dpu`持有者，遇到非已确认对象则拒绝并报告，不终止未知进程。还可用实验专属锁防止两个wrapper同时扫描后并发启动。

## P2：LocalDpu明确单线程所有权，不是可并发调用接口

`Pool`复用每模型输入/输出tensor，并维护`previous`状态；`LocalDpu::infer`没有锁。若每机网络线程直接调用此接口，会并发写同一tensor及竞态`previous`。推荐一个独立DPU执行线程唯一拥有LocalDpu，各机任务通过有界队列提交。析构必须在工作线程join、最后一次dpuRunTask返回后进行。当前单请求`dpu_check`没有此竞态。

`Pool::infer`非零status下的output是占位值：范围拒绝时为0；某些API失败路径可能是denorm均值。上层必须以`status==0`作为输出可用条件，不能下发占位值或计入精度统计。问题复现JSON目前已含`output_valid`，这是正确方向；CSV读者也须按status筛选，建议CSV同样显式有output_valid或空输出字段。

## P2：计时口径要写清边界

当前冻结Pool时间戳意义：

| 差值 | 真正覆盖的内容 |
|---|---|
| `prep_end-dispatch` | RAW237→归一化、240维补零、量程检查、DNNDK输入FP32→定点/张量写入 |
| `run_end-run_start` | 同步`dpuRunTask()`调用墙钟：包括驱动/调度/搬运/等待，**不是纯PL硬件时间** |
| `end-run_end` | DNNDK输出读取和物理反归一化 |
| `end-start` | 冻结Pool内部服务时间；不含237D参考构造、网络采样、父调度/排队、命令下发 |

`LocalDpu::infer`在Pool时间戳之外还做model lookup、Result字段复制和status3的拒绝维度重算。因此不能把`end-start`称为完整LocalDpu wrapper调用时间；完整PS端到端链要另记录wrapper外层时间戳。`dispatch`也不是硬件开始执行时刻，只是Pool入口中选中任务后的CPU标记。

`dpu_check`使用预先读好的固定237D输入；它不测现场采样或多项式/237D构造。应明确“从RAW237向量就绪开始”。逐轮round列可复算各轮分位数，当前汇总将3轮合并，应保留该说明。

## 已观察到的正确做法

- local_dpu.cpp include明确指向SHA验证过的冻结Pool源码，RAW路径仅归一化一次，未暗改比例/越界检查。
- 量程诊断复现同一float32归一化与scale乘法，边界保持`>=128`及`<=-129`，而不是改成简单[-128,127]浮点判定。
- 输入/输出scale使用DNNDK getter；identity明确写“DPU API wall-clock, not pure hardware time”。
- 板端程序使用新输出目录，不覆盖既有部署与失败记录；逐请求CSV在失败前已写出的记录仍保留。
- 黄金数据模型ID与各自黄金文件逐一对应。日志显示18×32条都通过，后续JSON生命周期失败不能解释成DPU数值失败。

## 等待父实现后的后续检查

`flight_runtime.cpp`尚未完成时不预先声明正确。后续重点审核：绝对释放日历、每机在途槽、过期DPU结果丢弃、命令已发送但晚回执的事实记录、实际施加控制版本、队列/锁所有权、退出时停止/恢复、双机模型切换及全组提交一致性。
