# Board PS source snapshot

2026-11-17。这个目录让 GitHub 协作者拿到**真实板端控制程序的完整源码快照**，而不必克隆被 `.gitignore` 排除的实验 `results/`。源码从 `results/ps_capacity_executor_trace_20261117_004/control_source/` 逐文件复制；`20261117_executor_trace/manifest.json` 列出另外 29 个文件的 SHA-256。复制前后已按该清单核对。保留原始字节和相对路径，后续研究新建版本目录，不直接修改这个冻结快照。

## 这是什么版本

- PS 原生 C++ 负责绝对任务发布、AirSim 状态/碰撞 RPC、237D 特征、义务日历和管理决策，调用板端 DNNDK/DPU 执行整条 DNN，再通过 RPC 下发控制。Windows 只运行 AirSim。主入口是 [`adaptive_runtime.cpp`](20261117_executor_trace/adaptive_runtime.cpp)，日历/调度在 [`scheduler/`](20261117_executor_trace/scheduler/)，237D 与 RPC 分别在 `features/`、`rpc/`。
- 这是十二机容量边界调查期间的**诊断源码版本**：相对之前实测版本，仅增强 DPU executor 异常原因的记录，不能称异常已修好。它不是 `OBLIGATION_PROTECTED_BUDGET_FRAMEWORK_20261116.md` 全部研究规格的实现；真实 skip/补更、板端 DNN 分段和超网仍未验收。
- 精确实验结果与实际二进制身份应回到同批原始日志/构建清单核对；源码快照本身不证明运行收益。当前状态见[协作交接](../docs/ROUTE_B_CURRENT_STATUS_AND_HANDOFF_20261117.md)。

## 克隆后还缺什么

[`build_control.sh`](20261117_executor_trace/build_control.sh)是当时 Ubuntu 交叉编译环境的原始脚本，包含 `/home/cwc/dpu_work/...` 的绝对路径。源码包**不含** AXU3EGB 交叉 SDK、AirSim/rpclib 静态库、DNNDK 库、18 个 INT8 ELF、归一化与部署数据、可执行文件、AirSim 工程或原始飞行日志。脚本还链接旧构建目录里的 `local_dpu.o` 和 RPC 静态库；在另一台机器上需按对应 SDK/模型交付重新准备这些依赖，不能期待 `git clone` 后直接运行。不要用旧 12D/25D 契约替代 237D 模型包，也不要改黄金输入来绕过检查。

协作者拿到 GitHub 源码后，如需**阅读或修改调度**，从本快照和 [`ps_planner_repair/`](../ps_planner_repair/) 开始。如需**精确复现实测**，另索取对应的冻结模型部署包、原始结果与哈希；普通 Git 提交只解决源码可见性。未来应在可移植构建脚本和依赖交付验证完成后，再宣称该仓库可独立上板构建。
