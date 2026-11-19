# 路线 B · 多无人机共享 DPU 实时控制

这个项目研究多架无人机共用一块推理加速器时，怎样让紧急的控制命令及时生效，同时为其他飞机留下完成下一次控制的时间。当前实验在 Windows 上运行 AirSim，Zynq 的 PS 运行 C++ 控制与调度，板载 DPU 执行独立的 237 维输入模型。

刚打开工程目录时，可以先看[根目录说明](#根目录说明)。下面按整理后的实际目录解释每个一级文件和文件夹。

## 第一次读代码

以“一条控制命令如何到达飞机”为线索，按下面三个步骤读即可。

| 顺序 | 打开 | 关注什么 |
| --- | --- | --- |
| 1 | [板端主程序](controller_architecture_study/ps_board_source/20261117_executor_trace/adaptive_runtime.cpp) | 从 `main` 找周期发布、每机工作线程和后台规划线程。采样及命令都经过 [AirSim RPC](controller_architecture_study/ps_board_source/20261117_executor_trace/rpc/airsim_rpc.cpp)。 |
| 2 | [正在维护的调度日历](controller_architecture_study/ps_planner_repair/live/live_calendar.cpp) | 跟随 `queued()`、`take()`，看一个任务何时能排队、交给 DPU、因超期被拒。旁边的 [C++ 测试](controller_architecture_study/ps_planner_repair/live/live_calendar_tests.cpp) 有最小例子。改调度从这里开始。 |
| 3 | [板端执行器](controller_architecture_study/ps_board_source/20261117_executor_trace/adaptive_live.hpp) | 跟随 `Executor` 找 DPU 返回、命令下发和 ACK。结合 [Drone10 的一次实际故障](controller_architecture_study/docs/PS命令保持期限回放与重构记录_20261118.md) 看哪些事件会使一条命令失效。 |

主程序链接指向已冻结的完整源码快照。调度日历链接指向快照之后的维护版本。本地编译回归、上板版本和外部模型包的关系见[板端源码说明](controller_architecture_study/ps_board_source/README.md)。

## 根据任务找材料

| 要做的事 | 入口 |
| --- | --- |
| 理解为什么这样调度 | [框架导读](docs/工程Framework.md) |
| 接续当前任务、了解已验证到哪里 | [协作看板](docs/协作进度.md) |
| 找到实测、模型交付与复现要求 | [研究文档索引](controller_architecture_study/docs/研究文档索引.md) |
| 找早期 FALCON 复现、训练、路线 A、旧交接或原始数据 | [历史归档与还原方法](archive/README.md) |
| 作为 agent 参与修改 | [协作约定](AGENTS.md) |

Git 提供源码和阅读材料。复现实飞还需要匹配的 AirSim 工程、板端 SDK 和库、release001 的 18 个模型以及对应实验输入。Windows 根目录留有少量 Python 模块，供场景生成、标定与历史对照使用。

## 根目录说明

目录清单以 2026-11-19 的本机为准。根目录里的源码和说明大多由 Git 管理。数据、权重、日志和不少交付包是本机资产，单独克隆代码时需要另取。看当前控制调度用 `controller_architecture_study/`，查旧训练和运行链路用 `archive/`，回看某轮试验才需要进入对应的 `results/`。

### 日常会打开的目录

| 根目录条目 | 用途 |
| --- | --- |
| [`controller_architecture_study/`](controller_architecture_study/) | 路线 B 研究区。这里找当前 C++ 调度、冻结板端源码、场景生成和实验结果。常用子目录见[研究区内部](#研究区内部)。 |
| [`docs/`](docs/) | 工程级说明。[`工程Framework.md`](docs/工程Framework.md) 讲设计，[`协作进度.md`](docs/协作进度.md) 讲实现进度，[`实验结果存储与复原.md`](docs/实验结果存储与复原.md) 记录实验结果清理和复原。`reference/` 留存旧框架原文和来源清单。 |
| [`configs/`](configs/) | 早期单机、三机 AirSim 设置，以及教师采集、推力与油门标定配置。现在十二机的 AirSim 设置在[研究区的配置文件](controller_architecture_study/configs/airsim_safety_12.json)。 |
| [`tests/`](tests/) | 当前保留的 Python 回归和输入夹具。包含调度、物理模型、接口、数据契约的测试。板端 C++ 调度回归与实现同放在 [`ps_planner_repair/live/`](controller_architecture_study/ps_planner_repair/live/)。 |
| [`archive/`](archive/) | 整理后移到历史区的研究代码、交接文档、训练入口、测试和本机原始产物。按 [`MOVED_PATHS.tsv`](archive/MOVED_PATHS.tsv) 搜旧文件或目录名。常用分区见[归档区内部](#归档区内部)。 |

### 根目录的 Python 文件

这些是场景、接口、教师和数据处理的共享模块。研究区的 Python 分析和兼容性测试可直接引入它们。查板端每周期的运行过程，从上面的 C++ 阅读入口开始。

| 根目录条目 | 用途 |
| --- | --- |
| [`airsim_interface.py`](airsim_interface.py) | 包装 AirSim Python API，连接指定飞机、读取原始状态、处理车辆局部位置与世界位置的偏移。 |
| [`airsim_process.py`](airsim_process.py) | 为批量采集或试验启动、监控和关闭 UE4/AirSim 进程，核对场景地图。 |
| [`config.py`](config.py) | 定义项目的数据、日志、标定路径，读写悬停油门与推力标定。 |
| [`control_adapter.py`](control_adapter.py) | 检查控制器的四值输出，把推力换算为油门，把角速度和油门转换成 AirSim 可下发的命令。 |
| [`dataset_store.py`](dataset_store.py) | 保存、读取和检查单轮采集数据，记录断点恢复清单，把多轮数据合并并导出元数据。 |
| [`dnn_model_utils.py`](dnn_model_utils.py) | 旧模型及 Python 基线共用的模型构造、特征/归一化、模型说明和导出核对函数。输入维度以模型自己的 `model_spec.json` 为准。 |
| [`environment.py`](environment.py) | 定义门的坐标、姿态和路线难度，根据参数生成多组门。 |
| [`gate_route.py`](gate_route.py) | 把一串门变成连续参考路径，用飞机的移动线段检查是否经过门平面及有效通行区域。 |
| [`gate_visualization.py`](gate_visualization.py) | 在 AirSim 场景中创建和清理可见门框，处理门框的摆放与姿态。 |
| [`initial_state_generator.py`](initial_state_generator.py) | 按种子生成、保存和读取教师采集所需的初始位置、姿态、速度和角速度。 |
| [`mpc_controller.py`](mpc_controller.py) | Python 版模型预测控制教师，根据飞机状态与未来参考计算推力和角速度，为数据采集或对照实验提供控制。 |
| [`prepare_dnn_dataset.py`](prepare_dnn_dataset.py) | 按环境划分训练、验证、测试集，用训练集计算归一化参数，写出 `.npz` 和来源清单。 |
| [`reference_generator.py`](reference_generator.py) | 从固定点、直线或过门路径生成未来目标的位置、速度和参考窗口。 |
| [`state_adapter.py`](state_adapter.py) | 转换 AirSim 和控制器坐标，处理四元数并构建旧 Python 控制器使用的 12 维状态接口。 |
| [`training_environment_generator.py`](training_environment_generator.py) | 按固定种子批量构造不同难度的训练场景，保存和加载环境清单。 |

### 历史实现、外部依赖与交接

| 根目录条目 | 用途 |
| --- | --- |
| [`experiments/`](experiments/) | 根目录保留的早期实验只剩 `falcon_paper_fidelity_acado_v3_continuous/`。里面的连续航线、教师配置和过门辅助函数仍被部分场景准备和对照脚本引用。其他早期实验见[归档实验](archive/early_falcon/experiments/)。 |
| [`falcon_paper_fidelity/`](falcon_paper_fidelity/) | 早期 FALCON 论文复现的 Python 库，含 ACADO/MPC 接入、轨迹生成、教师采集、数据契约和恢复工况。部分参考轨迹组件仍被研究区复用。 |
| [`native/`](native/) | `acado_rpg_mpc/` 里的 C/C++ 教师桥接和基准测试，把 RPG MPC 生成的 ACADO 求解器接到本工程。 |
| [`third_party/`](third_party/) | 外部代码入口。当前 Git 子模块是 `rpg_mpc/`，供 ACADO 教师侧构建。AirSim 工程和板端依赖需要按各自交付准备。 |
| `colleague_transfer/` | 同事交接包的本地输出目录。本次盘点为空，已归档内容从迁移清单查找。 |
| `ubuntu_transfer/` | 历史上发往 Ubuntu 的模型部署资料，含旧 12D、超网 M3、237D 独立 DNN 池及相应运行说明。 |
| [`ubuntu_tras/`](ubuntu_tras/) | 11 月 16 日 Ubuntu → Windows 的 PS 移植交付和验收回传。包含 `PS_MIGRATION_RESULTS_20261116_run001/`、`RESULTS.md`、`RUNBOOK.md`，可查迁移版本的源码、运行命令和证据。 |

### 本机数据、模型与日志

一组数据通常从 `teacher_dataset_*/` 的采集或采样产物生成，再写到对应的 `dnn_dataset_*/`。教师目录优先查 `manifest.json`，训练目录优先查 `split_manifest.json` 和 `normalization.json`。同为 12 维的数据也可能使用不同特征定义和坐标，复用时按各自清单核对。

| 根目录条目 | 用途 |
| --- | --- |
| `teacher_dataset_full_route_continuous_v2/` | 早期连续过门航线的教师控制数据，含初始状态、采集轮次和合并文件。 |
| `dnn_dataset_full_route_v2/` | 由连续航线教师数据生成的 12 输入、4 输出训练、验证和测试划分。 |
| `teacher_dataset_iid_path_v1/` | 从航线环境中采样的 IID 路径教师数据，按环境保留分片和汇总。 |
| `dnn_dataset_iid_path_v1/` | 对应的 25 输入、4 输出训练划分，给早期带路径信息的模型使用。 |
| `teacher_dataset_robust_v1/` | 早期鲁棒场景中的教师采集轮次、环境与合并文件。 |
| `dnn_dataset_robust_v1/` | 基于该批鲁棒教师数据的旧 12 输入、4 输出训练划分。 |
| `teacher_dataset_track_p_12d_robust_v1/` | Track-P 教师数据，12 维特征为门的相对位置、相对姿态、速度和加速度，采用 AirSim NED 坐标。 |
| `dnn_dataset_track_p_12d_robust_v1/` | 按 Track-P 契约拆出的训练、验证、测试数据及归一化。 |
| `dnn_runs/` | 历史基线与冒烟训练产物，例如 `depth10_width512/` 及 IID、鲁棒、Track-P 不同输入契约对应的版本。目录里可核查权重、模型说明和导出样本。 |
| `falcon_dnn_fp32_v1/` | 旧 25 维 FP32 单模型交付，含原始权重、TorchScript、黄金输入向量、归一化和 SHA-256。用于历史模型导出及核对。 |
| `data/` | 早期 AirSim 采集的飞行 CSV，常见文件名按飞机、场景和时间区分，可用于检查原始状态轨迹。 |
| `logs/` | 早期教师采集、UE4 和 AirSim 运行日志。查启动、RPC、采集失败可从这里找。 |
| `tmp/` | 历次手工和脚本试验的临时工作区，含小规模训练、交接 QA、解包等工作副本。 |
| `.tmp/` | 本机自动化工作区，含整理清单、磁盘清理校验、离线回放、编译和检查日志。其中一些记录可用于核对本轮整理。 |

### 工程管理文件与缓存

| 根目录条目 | 用途 |
| --- | --- |
| [`README.md`](README.md) | 项目首页，含读代码的顺序和本目录导览。 |
| [`AGENTS.md`](AGENTS.md) | Agent 的阅读顺序、修改约定、实验与证据规则。 |
| [`pytest.ini`](pytest.ini) | Python 测试发现配置，默认查找 `tests/`，跳过归档测试。 |
| `.git/` | Git 的本地提交、分支、索引及仓库配置。 |
| [`.gitignore`](.gitignore) | Git 默认忽略的生成物、日志、权重、本机交接包和本机归档等路径。 |
| [`.gitattributes`](.gitattributes) | Python 源文件使用 LF，保证跨系统核对冻结特征和教师源码时字节一致。 |
| [`.gitmodules`](.gitmodules) | 记录 `third_party/rpg_mpc` 子模块的仓库地址和本地路径。 |
| `__pycache__/` | Python 导入模块时自动生成的字节码缓存，可再生。 |
| `.pytest_cache/` | pytest 自动生成的测试发现和上次结果缓存，可再生。 |

### 研究区内部

| `controller_architecture_study/` 中的入口 | 用途 |
| --- | --- |
| [`ps_planner_repair/`](controller_architecture_study/ps_planner_repair/) | 当前调度维护目录。其下 `live/` 放日历、调度器、计划复核及对应 C++ 测试，上一级放场景准备、诊断和离线回放脚本。 |
| [`ps_board_source/`](controller_architecture_study/ps_board_source/) | 冻结板端主程序、任务执行器、AirSim RPC、237D 特征构造、DPU 接口和配置。用来读完整运行链、确认历史部署版本。 |
| [`ps_migration/`](controller_architecture_study/ps_migration/) | PS 移植的数据与接口契约、冻结轨迹、交付内容检查和数值一致性工具。 |
| [`native_runtime/`](controller_architecture_study/native_runtime/) | Windows 原生采样与接口 DLL 的 C++ 实现及构建文件，供旧 Windows 路线和接口对照使用。 |
| [`models/`](controller_architecture_study/models/) 与 [`vendor/`](controller_architecture_study/vendor/) | Python 模型结构实现和随工程保存的凸二次规划代码。实际训练得到的权重另存在实验产物中。 |
| [`configs/`](controller_architecture_study/configs/) 与 [`docs/`](controller_architecture_study/docs/) | 十二机设置、实验候选参数、框架规格、板端复飞报告与模型交接索引。 |
| `results/` | 本地实验产物。近期板端实测、回放、模型数据和还被工具读取的交付包都在这里。原始输入和恢复方式见[实验结果存储说明](docs/实验结果存储与复原.md)。 |
| 根部的 Python 与 PowerShell 脚本 | `package_*`、`share_*` 用于打包或交接，`audit_*`、`probe_*`、`replay_*` 用于诊断或回放，`run_*`、`fly_*` 用于相应试验。查某次试验时从研究区文档找到对应脚本、配置和版本。 |

### 归档区内部

| `archive/` 中的入口 | 用途 |
| --- | --- |
| [`early_falcon/`](archive/early_falcon/) | 早期飞控实验源码、训练脚本、12D 交接以及旧实验目录。 |
| [`controller_architecture_study/`](archive/controller_architecture_study/) | 旧研究脚本、路线 A、超网训练和历史报告。归档前的相对路径可用迁移清单还原。 |
| [`history/`](archive/history/) 与 [`tests/`](archive/tests/) | 前者存旧版首页、看板和框架，后者存随历史实现一起移动的回归。 |
| `local/` | 仅本机保存的旧实验、交付包、数据、权重及清理记录。恢复某个已清理产物时先看原位置的 `RESTORE.md`。 |
| [`README.md`](archive/README.md) 与 [`MOVED_PATHS.tsv`](archive/MOVED_PATHS.tsv) | 按主题查归档材料，以及按原路径精确定位搬迁后的文件。 |
