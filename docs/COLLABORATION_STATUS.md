# 协作进度与任务看板

最后更新：2026-11-02（Asia/Shanghai）。这是所有协作者共同维护的状态入口；整体设计见[FRAMEWORK](FRAMEWORK.md)，详细资料见[研究分支索引](../controller_architecture_study/docs/README.md)。

**当前结论：现有规则式动态控制已有自然十机零碰撞样例；新增学习建议链路已实现，但额外收益未成立。下一关键工作是期限归属修复与有效恢复边界证据，不是继续盲目训练更大的DNN。**

本文更新的是进度和任务，不修改原始试验报告。远端Ubuntu/服务器没有回传证据的内容一律写“待回填”，不能根据Windows已有部署包推定已经上板。

## 1. 工程地图与不同轨道

| 位置 | 用途 | 交接时注意 |
|---|---|---|
| 根目录`run_*`、`mpc_controller.py`、`dnn_controller.py` | 历史MPC/DNN/AirSim基础链路 | 不代表最新研究原型的全部实现 |
| `experiments/paper_track_p_12d_5ms_v1/` | 历史12D数据与六模型训练 | 实际周期/速度以配置为准，不从`5ms`目录名推断 |
| `controller_architecture_study/` | 当前主要研究源码：237D、模型池、超网、动态控制与诊断 | 接手实时框架从这里读 |
| `controller_architecture_study/native_runtime/` | C++释放日历和AirSim RPC DLL | 不是FPGA内核，也不是整个框架都已C++化 |
| `controller_architecture_study/results/` | 每次训练/试验/审计产物 | 目录多，先按本文定位，不全量扫描或覆盖 |
| `controller_architecture_study/docs/` | 带日期的设计、实验、文献与交接 | 当前状态以本文为准，历史结论保留原日期 |
| `ubuntu_transfer/` | 冻结部署包和对应manifest | 包存在不等于Ubuntu已完成量化/上板 |
| `colleague_transfer/` | 给协作者的历史压缩包、UE工程及交接 | 注意包的版本，不用旧快照覆盖当前源码 |
| `tests/` | 对应模块的回归检查 | 选择相关检查；真实飞行/板端收益另需证据 |

### 输入和模型不能混用

| 轨道 | 角色与契约 | 当前用途 |
|---|---|---|
| 旧25D | 早期工程化控制输入，包含额外路径信息 | 旧FP32/DPU/UDP闭环经验 |
| 历史12D | 门位置误差、姿态误差、当前速度、当前加速度 | FALCON相关六模型复现轨道 |
| 当前237D独立模型 | 显式状态及未来参考，契约`decision_convex_local_relative_pv_237d_v1` | 当前CPU实时原型、独立模型池基线 |
| 当前237D超网 | 同一237D任务上的因果深宽共享结构 | 已完成FP32训练与整数功能候选，尚未接入当前实时链路 |

所有轨道输出均涉及`[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]`，但仍需各自配套的归一化、坐标约定和输出转换。相同输出维数不意味着模型/协议可直接互换。

237D编码从300D完整参考特征投影得到，不能当作“12D补零”。准确布局看[编码器](../controller_architecture_study/convex_features.py)及[基础特征构造](../controller_architecture_study/decision_dataset.py)，不要在另一端重新猜输入次序。

## 2. 已经完成的工作

| 事项 | 已达到的状态 | 证据/位置 |
|---|---|---|
| 历史FPGA链路 | 用户历史回报旧模型DPU/UDP与Windows/AirSim链路打通 | [旧集成状态](../WINDOWS_FPGA_INTEGRATION_STATUS.md)；不外推到新237D/超网 |
| 237D教师、数据与独立网络族 | 恢复补采样数据与18个主池plain模型、残差及长训对照已形成 | `results/decision_convex_recovery_v1/`、`results/decision_convex_recovery_scenic_falcon18_v1/` |
| 最大残差模型长训 | 33×1024残差追加5000轮完成；作为超网构建的教师来源 | [超网背景](../controller_architecture_study/docs/INCREMENTAL_SUPERNET_ARCHITECTURE_20261012.md)，其中旧进度段是历史记录 |
| FP32二维共享超网 | 17深度×8宽度=136配置，v4追加5000轮完成，导出PASS，最大配置目标达成 | `results/incremental_supernet_237d_v4_grid5000/export/export_report.json` |
| 混合精度整数候选 | W8/head16/A16 LSQ dyadic；整数直接/增量逐位一致，最大/网格质量目标通过，严格切换目标未过 | `results/incremental_supernet_237d_m2_lsq_v3/export/export_report.json`，状态仍为`NUMERICS_PASS_QUALITY_NOT_MET` |
| 冻结超网部署包 | 可交接给自定义PL功能开发，质量例外已记录 | `ubuntu_transfer/falcon_incremental_supernet_m3_v1_20261015/`；`freeze_manifest.json`中FPGA/飞行振荡改善均未验证 |
| 独立237D模型池交接 | 27个FP32独立模型已打包；其中主池18、残差600轮4、长训残差1、早期恢复4 | [237D DPU交接](../controller_architecture_study/docs/UBUNTU_237D_DPU_MODEL_POOL_EXPERIMENT_HANDOFF_20261016.md) |
| 独立任务释放与IO | C++绝对时间日历、每机RPC及Python C ABI接入；与前端反馈处理解耦 | [C++实现说明](../controller_architecture_study/docs/NATIVE_RUNTIME_IMPLEMENTATION_20261029.md) |
| 共享分段执行 | CPU执行器在最多两层FC的段边界按EDF重排 | [执行器源码](../controller_architecture_study/framework_runtime_worker.py) |
| 规则式动态控制 | 有异质多机、周期调整、全组准入、前端保护、单机碰撞退出与日志 | [易懂实现说明](../controller_architecture_study/docs/FRAMEWORK_IMPLEMENTATION_EXPLAINED_20261101.md) |
| 小评分器与提交接口 | 486参数线性动作评分、有限显式选择、最新状态复核、原生原子提交入口 | [后台建议](../controller_architecture_study/joint_contract_shadow.py)、[提交检查](../controller_architecture_study/joint_commit_guard.py) |
| 最新阶段验证 | 120次有效恢复采集＋12次自然十机对照；相关74项检查通过 | [2026-11-02阶段报告](../controller_architecture_study/docs/BOUNDARY_TAKEOVER_RESULTS_20261102.md) |
| 本次协作文档 | 工程入口、框架、看板、历史原文及索引已建立 | 本文与[整体框架](FRAMEWORK.md) |

以上的“已完成”只针对对应交付。超网训练完成不等于超网闭环完成；整数一致不等于全部质量目标通过；零碰撞不等于硬期限全部满足。

## 3. 最新结果：所有协作者必须知道

结果根目录：`controller_architecture_study/results/boundary_takeover_20261101_v1/`。

- 120次有效采集覆盖30个配对状态，另外保留1次最小化窗口造成的基础设施超时。30个状态中，三种DNN及20/80ms比较尚未出现恢复资格差异；不能宣称已得到最低成本合格模型映射。
- 新线性评分器评估40个候选平均约0.072ms、P99约0.182ms。整个后台建议链路平均约6.4–6.7ms，不能只报评分器时间代表总开销。
- 第一轮新增方案安装1组、15条标记控制真正完成，但发生碰撞；之后限定为各机原DNN候选，并将建议处理放在已就绪控制工作之后。
- 修正后静态组两轮碰撞数1/0，旧动态0/0，新增模块组0/0。但修正后新建议实际安装0组，不能把这些零碰撞归功于学习选择器。
- 修正后仍有大量条件预算越期，尚未证明硬实时或严格物理安全。静态配置是历史离线选定项，不是本轮重新求得的全局最优FALCON配置。
- 268次后台决策中137次被未完成超期任务拦截；这137个快照都存在某个任务截止时间不晚于它自己的计划发布时间。部分旧保持控制早已结束，后续计划却仍携带旧期限，这是下一步的关键问题。

源控制DNN与归一化哈希已核对未变。8个网络是当前实时池，27个是导出的独立模型全集，18个是主训练族，136个是超网配置数：这些数字不是同一个集合。

### 最新证据（不要只读旧stage_summary）

| 路径（相对结果根目录） | 含义 |
|---|---|
| `combined_evidence.json` | 两轮对照、源控制器核验与时间归属诊断的合并入口 |
| `boundary/report.json` | 120有效采样与无效尝试、配对差异计数 |
| `ranker/report.json` | 小评分器训练、路线分折与评估 |
| `application/summary.json` | 修正前六次对照，曾实际安装1组 |
| `application_rates_only/summary.json` | 修正后六次对照，实际安装0组 |
| `stage_summary_application_rates_only.json` | 最新修正轮次审计汇总 |
| `application_rates_only/obligation_epoch_audit.json` | 旧控制义务/当前期限的归属诊断 |

`stage_summary.json`默认记录早期轮次，不能只因名字更短就当作最新结果。74项检查是上一轮相应代码的回归结果，不替代后续修改后的验证。

## 4. 当前正在推进的主线

目前处于“期限归属与建议实际落地”的问题定位阶段，而不是新的控制DNN训练阶段。

已定位、尚未修复的中心问题是：**任务的截止时间应对应哪一条正在保持的控制？** 已经结束的旧义务、确实发生的历史超期、当前仍有效的安全期限必须分开。不能通过删除历史失败或统一放宽期限来提高建议采用率。

恢复边界数据也尚不足以支撑可靠的自动选型。下一轮应有限细化已有临界区域并另留新路线确认，而不是再训练数千轮期待模型自行解决标签缺乏差异的问题。

本次协作文档整理未启动新训练或试飞。其他机器是否正在训练、Ubuntu是否已上板，需相应协作者在下面回填。

## 5. 任务认领与后续顺序

任务状态使用：`待认领 / 进行中 / 待验证 / 已完成 / 阻塞`。注明具体完成层级，不用一个PASS代替所有结论。当前未来任务均未认领；认领后填Agent/协作者名称和日期。

| ID | 优先级 | 任务与最小交付 | 当前状态/负责人 | 完成条件 |
|---|---|---|---|---|
| RT-01 | P0 | 修正保持控制义务的时间归属，先做归档回放与针对性检查 | 待认领；已定位 | 旧义务结束与当前义务可区分；历史超期仍保留；当前期限、在途成本及全组检查不被绕过 |
| DATA-01 | P1 | 在已有临界范围有限补采配置能力确实不同的状态 | 待认领 | 预先限定查询数；匹配初态/参考/控制/时序；保留同成同败；独立新路线确认；没有差异也如实报告 |
| SEL-01 | P1 | 基于新证据复核小评分器或更简单的资格映射 | 待认领；依赖DATA-01 | 优于明确默认项的可重复证据；包含准备、评分、检查的总开销；不是只有排序准确率 |
| RT-02 | P1 | 在RT-01后验证新的期限/周期建议真实提交 | 待认领；依赖RT-01 | 原生安装、带版本的实际命令、真实周期改变三者可追溯；失败结果完整保留 |
| EXP-01 | P1 | 扩展自然异质路线静态/动态对照 | 待认领 | 同模型池/资源；重新说明离线静态选择预算；多种子配对；碰撞、J、时序、未采用建议均报告 |
| HW-01 | 可并行 | Ubuntu自定义PL功能实现与整数golden核验 | 远端状态待回填 | 按冻结包先记录工具/器件；C仿真、综合/协同仿真、板端结果分层给证据；质量例外不变成PASS |
| HW-02 | 后续 | FPGA时延、上下文、DDR/网络与分段成本回填 | 待认领；依赖HW-01 | 当前237D/超网的真实测量，重新标定模型；不使用CPU或旧25D时延替代 |
| BASE-01 | 可并行 | 独立237D模型池的真实DPU切换与命令跳变实验 | 远端状态待回填 | 按27模型交接包核对量化/编译和模型身份；区分时间jitter与控制输出跳变 |
| DOC-01 | 已交付 | 建立协作入口、框架和共享进度 | 已完成；本次Windows Agent；2026-11-02 | 本文、FRAMEWORK、根AGENTS/README、分支索引及原文快照可相互导航 |

范围提示：RT-01主要涉及`joint_contract_recovery.py`、`joint_commit_guard.py`、`joint_contract_shadow.py`与`probe_seven_continuous_timing.py`；DATA-01主要涉及`probe_recovery_boundary.py`与`controller_selector/`；SEL-01主要涉及`controller_effect.py`、`repair_controller_effect.py`。认领时写实际要改的文件，避免多Agent同时编辑实时主循环。

### 暂不作为默认下一步

- 不继续无目标扩大网络或延长训练，不为了得到单调复杂度曲线改变测试划分。
- 不为提高提交率放宽全部deadline，不把新建议未采用的飞行称为学习控制成功。
- 不把中途随意改深宽、无限细抢占、动态增加PL核数同时加入第一版。
- 不把超网替代独立模型池当作已经证明的必要条件；先保留可比较基线。

## 6. 关键产物位置

以下路径相对工程根目录；Ubuntu可使用自己的工程根目录映射，不能照抄Windows盘符。

| 产物 | 相对路径 |
|---|---|
| 当前237D数据 | `controller_architecture_study/results/decision_convex_recovery_v1/datasets/features_recovery/` |
| 主18模型 | `controller_architecture_study/results/decision_convex_recovery_scenic_falcon18_v1/runs/` |
| 残差4模型 | `controller_architecture_study/results/decision_convex_residual4_v1/runs/` |
| 残差33×1024长训 | `controller_architecture_study/results/decision_convex_residual33_long5000_v1/residual_mlp_h33_w1024_seed0/` |
| FP32共享超网 | `controller_architecture_study/results/incremental_supernet_237d_v4_grid5000/export/` |
| 混合精度整数候选 | `controller_architecture_study/results/incremental_supernet_237d_m2_lsq_v3/export/` |
| 自定义PL冻结交接 | `ubuntu_transfer/falcon_incremental_supernet_m3_v1_20261015/`及同名zip |
| 独立池DPU交接 | `ubuntu_transfer/falcon_independent_237d_dpu_pool_v1_20261016/`及同名zip |
| 最新Windows阶段 | `controller_architecture_study/results/boundary_takeover_20261101_v1/` |
| 本次冻结评分器 | 上一目录下`ranker/ranker.npz`，仅代理评分，不是控制器权重 |

数据与模型不是随本文自动附带的。交接给新机器时，按具体任务复制所需数据/权重/manifest，并保持相对路径或明确记录映射；不要默认Git或文档包包含大文件。

## 7. 运行环境与可安全重读的入口

| 项目 | 当前Windows环境 |
|---|---|
| 工程根目录 | `F:\FalconReproduction\airsim_control` |
| 训练/多数研究脚本Python | `D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe` |
| AirSim Python环境 | `F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe`；部分研究入口自行加载客户端路径 |
| Unreal | `E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe` |
| Blocks工程 | `F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject` |
| 多机配置 | `controller_architecture_study/configs/airsim_safety_12.json`，最近对照实际用10回路 |
| 原生DLL | `controller_architecture_study/native_runtime/build/Release/falcon_native_io.dll` |
| 构建入口 | `controller_architecture_study/build_native_runtime.ps1`；已有MSVC/CMake环境 |

不要根据本文PID或过去的Play回复推定当前连接。试飞前核对车辆列表、编辑器未最小化、后台限速设置、CPU配置及是否有并行训练。用户既有偏好是由脚本执行实验，Play通常由用户点击，不操作屏幕；不强制结束有未保存内容的编辑器。

只重新读取本轮结果、不训练也不试飞，可运行：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
& $trainPy -m controller_architecture_study.audit_boundary_takeover --root .\controller_architecture_study\results\boundary_takeover_20261101_v1 --application application_rates_only
```

这依赖已有结果文件。首次接手不要直接启动全部训练/试飞脚本；先按所认领任务读取入口和输出位置，避免覆盖别人的运行。相关检查按改动选择；native DLL检查、真实飞行、HLS综合分别报告。

## 8. 协作者怎样共同更新

1. 开工前先读本页和FRAMEWORK，在任务表写负责人、状态、涉及文件与输出目录；有冲突先协调，不覆盖他人的修改。
2. 结束后补一条日志：做了什么、具体证据、未完成什么。更新任务状态，区分“实现完成”和“验证/收益成立”。
3. 影响设计或时间定义的变更同步FRAMEWORK；只影响实验结论的变更写新日期报告并链接，不改历史原始报告。
4. Ubuntu/服务器协作者回填工具版本、目标器件/架构、模型/包hash、命令、产物路径和实测结果；未回传的保持未知。
5. 原模型、manifest、冻结包和失败日志保留。新训练/实验用新版本目录，记录父版本与源码身份。

可复制的更新条目：

```text
日期/时区：
协作者/Agent：
任务ID与状态：
修改文件/设计变化：
执行命令与验证层级：
结果证据路径、版本/hash：
剩余问题：
下一交付与认领情况：
```

### 更新记录

| 日期 | 协作者 | 变化与证据 |
|---|---|---|
| 2026-11-02 | Windows Agent | 汇总120次有效采样和两轮各6次对照，定位旧义务期限问题；详见[阶段报告](../controller_architecture_study/docs/BOUNDARY_TAKEOVER_RESULTS_20261102.md) |
| 2026-11-02 | Windows Agent | 按用户协作要求建立当前入口、框架、看板和原始framework快照。本次仅整理文档，未修改控制/训练/调度代码，未启动训练或试飞 |

本次文档核对：6个入口文档中的63个本地链接均可解析，8个主要产物目录存在，原始framework快照与原文件逐字节一致；12个实时源码文件与上一轮复核记录的哈希一致。后续更新后，这些检查结果只代表本次时间点。
