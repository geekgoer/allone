# C++ 实时路径迁移（2026-10-29）

## 这次具体实现了什么

目标是减少 Python IO 线程交接、辅助进程往返和偶发长时间等待，并修正动态任务释放仍依赖上一命令返回的问题。没有修改模型、数据集、控制输出定义和物理预算 B 的公式。

当前结构：

```text
Python：场景初始化、B/模式/周期选择、全局准入、原有分段 EDF 执行、报告
                 │ C ABI（ctypes.CDLL，调用时释放 GIL）
                 ▼
C++ DLL：独立绝对时间释放日历 + 每机常驻 AirLib RPC 客户端 + 状态读取/命令下发
                 │ 与原客户端相同的 AirSim RPC
                 ▼
Windows Unreal / AirSim
```

C++ 接管的是原先最可疑的 IO/释放路径。分段 DNN 执行和调度决策仍沿用现有 Python 实现，不能称为“整个 framework 已全部 C++ 化”。独立观测/评分服务暂时也保留原实现，两组试飞使用相同服务。

## 实时语义

1. 下一次释放时间基于上一次计划释放时间和已提交 T 推进；反馈未返回时，继续使用最近提交配置发布任务。
2. C++ 专用线程用高精度等待定时器发布任务。Python 未读取结果、RPC 阻塞、上一任务未完成，均不阻止该线程发布后续任务。
3. 每个任务分别记录计划释放、C++ 实际发布、Python 收到发布事件、派发、状态 RPC、碰撞 RPC、推理、命令返回等时刻。计划周期不等于操作系统保证的精确唤醒；发布抖动如实计入日志。
4. 反馈只能调整尚未发布的下一任务。任务一经发布，其模型、释放时间和 deadline 不可被新反馈覆盖。
5. 没有新反馈时，动态后继任务不会把旧控制的绝对 B deadline 向后平移。新控制获得新观测后才能重新计算下一物理预算。
6. 新配置及保留旧 c/T 的备选都针对同一新物理期限做全局准入。已发布任务不重新定义期限。
7. 不因“上一任务没完成”偷偷跳过释放，也没有新增自动丢弃/合并请求策略。若持续处理能力不足，真实排队与超期必须显现；这不是硬实时保证。
8. 原有同一飞机 IO 串行限制仍保留，防止控制命令乱序。C++ 命令确认后先发布 acknowledgement，再读取命令后的状态。
9. 命令确认仍采用原有 `moveByAngleRatesThrottleAsync + ping` 返回时刻作为代理，不把它声称为精确物理生效时刻。

过去出现过的约 85ms 新命令空窗不能承诺靠本次改动全部消失。独立任务释放修复了逻辑依赖，C++ 减少了客户端开销，AirSim 服务端/Windows 的长尾以及资源过载仍可能存在。

## 资源与构建

- 本机已有 Visual Studio 2022 / MSVC / CMake，已编译 Release DLL。
- 使用本机 `F:\Airsim\AirSim` 的官方 AirLib C++ 客户端实现，没有重新发明 AirSim 协议。
- rpclib 的构建脚本会在源目录生成文件，所以只在本工程 `native_runtime/build/rpclib_source` 中构建副本，不修改原 AirSim checkout。
- 仅在构建副本中使 rpclib 私有 IO 线程继承调用者的核绑定，避免 C++ 暗中多使用前端/传感核心。
- 沿用现有 CPU 分配、EDF 分段边界、同一模型池和冻结预测成本。普通 Windows 下的核绑定不是独占核心，也不是 WCET 证明。
- 当前版本使用 Windows QPC/事件/定时器；迁移 Zynq Linux 时需要替换这些平台接口，并为 ARM 重新编译。

## 文件

- `native_runtime/native_io.cpp`：C++ RPC、命令转换、定时发布、事件队列。
- `native_runtime/CMakeLists.txt`：只构建客户端所需 AirLib 源文件与本地 rpclib 副本。
- `framework_native_io.py`：固定结构体 C ABI，接入原有前端，无逐帧 JSON 或 Python IO 辅助进程。
- `framework_releases.py`：动态独立后继和只作用于未发布任务的反馈更新。
- `probe_seven_continuous_timing.py`：`loop_config.io_backend='cpp'` 切换接口；两种 IO 后端均修正动态释放依赖。
- `run_native_runtime.py/.ps1`：只读采样对照与十机配对试飞入口。
- `tests/test_framework_native_io.py`：真实编译 DLL 的离线测试；模拟 RPC 延迟用于验证发布独立性。

## 已完成的验证及证据范围

已验证 C++ 编译、C/Python 结构体大小与时钟对齐、90ms 模拟阻塞期间的 20ms 发布、模式修改不覆盖已发布事件、任务 ID 唯一、控制限幅与符号/油门转换、过期命令拒绝。

最终相关回归检查 **165 项通过**。模拟测试不替代真实 RPC/闭环测试。

真实 AirSim 十机只读采样（每组 1000 次、相同 IO 核，先 Python 后 C++，仅作接入检查）：

| 阶段 p99 | Python | C++ |
|---|---:|---:|
| 提交至开始读取 | 5.362ms | 1.098ms |
| 状态/碰撞 RPC 与解码 | 5.611ms | 3.848ms |
| 读取结束至前端收到 | 3.629ms | 2.279ms |

原始记录：`results/native_runtime_io_20261029_v1/io_probe.json` 及两组 `*_io_requests.json`。本轮两组都没有超过 10ms 的采样阶段，不能由此声称原来的 47ms 极端长尾已被消除。

## 使用命令

在 AirSim 已使用独立 12 机配置、点击 Play、关闭后台降速且窗口未最小化后运行。脚本使用绝对路径，可从任意 PowerShell 工作目录调用。

重新编译：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\FalconReproduction\airsim_control\controller_architecture_study\build_native_runtime.ps1"
```

只读采样比较，不 reset、不发飞行命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\FalconReproduction\airsim_control\controller_architecture_study\run_native_runtime.ps1" -IoOnly
```

相同模型和资源下的十机配对动态试飞，seed4 为 Python→C++，seed5 反序；发生碰撞时只退出对应飞机：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\FalconReproduction\airsim_control\controller_architecture_study\run_native_runtime.ps1"
```

仅运行 C++ 版本：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\FalconReproduction\airsim_control\controller_architecture_study\run_native_runtime.ps1" -Backend cpp -Seeds 4
```

每次运行创建新的结果目录；保留中断/失败记录，不覆盖旧数据。此次固定旧离线选出的配置只是迁移对照，并非在新后端重新搜索后的 FALCON 最优 baseline。

## 下一步判据

先确认真实闭环可用，再看：发布抖动、请求派发排队、采样与回传 p99/p99.9/max、完整响应 R、实际命令间隔、碰撞/过门及任务完成。阶段耗时的 p99 不应相加当作整体 p99。

如 IO 明显改善但命令空窗仍长，应根据新日志继续定位前端或请求积压，而不是默认继续把 DNN 计算搬到 C++。尚未实施丢弃、超网早退、PL 加速和最终严格安全证明。

## 最终闭环接入结果

真实试飞前两轮保留在 `native_runtime_flight_20261029_v1/v2`，属于调试记录，不用于性能结论：v1 修正了新候选/保留 c/T 的物理义务匹配；v2 定位到 Python AirSim 客户端内部还会翻转一次 pitch/yaw，而官方 C++ 客户端直接发送。C++ 现在发送与 Python 实际 RPC 完全相同的参数，新增测试直接截获 Python `call_async` 参数进行比较，不只比较控制适配器的表面参数。

有效闭环接入对照在 `results/native_runtime_flight_20261029_v3/`：两个初态、每次十机、seed4 Python→C++、seed5 C++→Python。四次程序均完成、无运行时错误、无清理错误，模型/初态一致。但物理任务与实时验收没有通过。

| 闭环中 RPC 采样子集 p99 | seed4 Python | seed4 C++ | seed5 Python | seed5 C++ |
|---|---:|---:|---:|---:|
| 提交至开始读取 | 3.705ms | 2.739ms | 3.546ms | 3.994ms |
| 状态/碰撞读取与解码 | 5.082ms | 1.173ms | 5.413ms | 2.075ms |
| 读取结束至前端收到 | 4.516ms | 2.839ms | 4.032ms | 3.866ms |
| 碰撞架数 | 3 | 2 | 3 | 4 |

因此不能声称 C++ 已经减少碰撞：两组总碰撞都是 6 架。某些阶段变快，也不意味着所有阶段的尾延迟都降低。发生碰撞后各组剩余负载不同，上表属于接入观察，不是最终受控论文结论。

更关键的发现是**独立发布后暴露的请求积压**：

| 对已应用命令分解的平均耗时 | seed4 Python | seed4 C++ | seed5 Python | seed5 C++ |
|---|---:|---:|---:|---:|
| 任务已发布、尚未派发 | 1712.344ms | 1496.930ms | 2068.538ms | 2028.783ms |
| 派发至开始构造输入 | 3.073ms | 2.184ms | 2.887ms | 3.216ms |
| 输入构造 | 0.467ms | 0.457ms | 0.475ms | 0.481ms |
| 推理及相关传输 | 2.536ms | 2.410ms | 2.396ms | 2.672ms |
| 输出就绪至命令返回 | 2.688ms | 1.803ms | 2.739ms | 2.023ms |

这些是阶段均值，不是最大值。R 的秒级增长主要发生在派发前，不能再归因于某次 C++ RPC 需要几秒。

这不是把旧 R 改一个统计名字：以前动态发布隐含等上一任务返回，实际上减少了到达负载；现在任务持续按日历发布，旧请求仍全部保留，而处理/全局准入没有跟上。已错过期限的未完成任务也会阻碍配置变更准入。继续串行处理全部积压任务无法单靠更快的 RPC 恢复实时性。

两组还在第一次碰撞前的相同时间窗内检查了积压，排除“只是碰撞后退机才形成差异”的解释：seed4 截止 7.177s，Python/C++ 分别已有 753/590 个已发布但未交付请求；seed5 截止 9.583s，分别为 1132/1190 个。

注意：任务 R 大不代表状态输入必然同样陈旧，因为当前仍在真正派发时获取新状态。应分别分析任务完成期限、采样到命令的数据年龄、实际命令间隔。

当前阶段结论：**C++ IO/独立发布路径已经实现并完成真实闭环接入；整个十机实时控制系统尚未通过验收。** 下一项应针对过时排队请求的处理规则与准入开销，不宜直接声称更换语言解决了框架安全性。是否采用“未开始且已被更新请求替代的任务合并/丢弃”，需要作为明确策略记录，不能悄悄删掉请求后仍报告零超期。本次尚未自动丢弃这些请求。

证据：有效运行的 `summary.json`、各组 `requests.json`，以及拆分结果 `pipeline_audit.json`。训练和模型产物未变更；原有阶段报告保持原样。
