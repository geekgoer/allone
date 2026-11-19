# 路线 B 研究目录

当前主线在 Zynq 上运行。工程第一次阅读的三个代码入口见[项目首页](../README.md)。

| 目录 | 用途 |
| --- | --- |
| [ps_planner_repair/live](ps_planner_repair/live/) | 正在维护的 C++ 日历、调度器、复核与回归测试 |
| [ps_board_source](ps_board_source/) | 可随 Git 获取的板端主程序、执行器、RPC、237D 特征和 DPU 接口快照 |
| [ps_planner_repair](ps_planner_repair/) | 生成实验用控制程序、准备场景、分析日志与回放 |
| [configs](configs/) | 包括 Windows Blocks 使用的 [12 机 AirSim 配置](configs/airsim_safety_12.json) |
| [研究文档](docs/研究文档索引.md) | 设计、实测、模型来源及运行边界 |
| [历史档案](../archive/README.md) | Windows 实验、路线 A、早期训练和旧模型的定位方式 |

目录内保留的 Python 文件还供场景生成、Python 对照、数据契约和迁移工具使用。当前板端每个控制周期执行的是 C++ 主链。
