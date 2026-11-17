# 按板端职责裁剪的源码：不再迁移整个 Windows 入口

补充阅读：[已有C++日历/IO、真实DNNDK与API符号说明](CPP_REUSE_AND_API_CONTRACT_20261116.md)。`controls.py`转换结果属于Python API层，使用AirLib C++时应核对最终RPC，不可直接照搬pitch/yaw参数。

2026-11-16。上一版约0.52 MB源码包保留了旧入口的静态导入依赖，仍带有CPU推理、教师和历史辅助。它解决了传输体积，没有彻底解决职责冗余。本包改为按函数、类和方法裁剪。

**本包是PS运行逻辑的Python参考内核及等价性核对，不是已经交叉编译并在PS执行的完整控制程序。** Ubuntu Agent需据此接本地DNNDK、远程AirSim RPC和板端实时循环。旧工程和此前完整交接包保留。

## 1. 这次具体拿掉什么

| 旧源码/机制 | 新包的处理 |
| --- | --- |
| `dynamic_probe_common.ModelPool/InferenceService`，Torch CPU/GPU推理 | 整个模块不带，不在PS执行CPU版DNN |
| 模型训练、checkpoint加载、FP32离线评估、旧联合搜索实验入口 | 不带 |
| `convex_teacher`、MPC求解器及教师增益试验 | 不带；只抽取原 `MPCReference` 数据类型，改名 `ReferencePoint` |
| `fly_ip37_joint` 等 Windows 飞行/场景/实验入口 | 不带，Ubuntu不要将它们原样搬到PS |
| Windows DLL日历、ctypes、WinMM、编辑器启动、线程cycle诊断 | 不带 |
| `JointRuntime` 的旧ABC快照/安装和原生DLL提交方法 | 移除，只保留新日历使用的任务生命周期方法 |
| `bottleneck_joint` 的枚举Oracle、旧启发式、分段配置生成 | 移除；全组检查只接受已经物化的整任务 |
| 绘制/生成门、训练环境生成、旧轨迹重规划 | 不带；只保留安全分析使用的门框几何和已冻结参考的数值计算 |
| Windows↔板端UDP推理后端 | 不带；保留配置接口，由Ubuntu接现有本地 `Pool::infer()` |

这里的“移除”指本次迁移包。没有删除原工程的历史源码或证据。

## 2. 哪些代码应该在PS执行

`ps_core/`最新保留以下12个模块，不再为了导入一个小函数携带整个旧文件。后续完备性核对补回了轨迹装载和已验证成本表读取，最新交付以 `PS_VERIFIED_HANDOFF_20261116.md` 为准。

| 模块 | PS职责 |
| --- | --- |
| `types.py` | 当前状态、参考点、门及四元数工具；没有MPC求解器或AirSim客户端 |
| `features.py` | 读取已冻结轨迹得到参考，构造原契约RAW237 |
| `trajectory.py` | 校验并装载已冻结的轨迹系数，不重新规划或拟合 |
| `geometry.py` | 门框几何与旋转，供安全分析；不生成UE场景 |
| `hold.py` | 当前有效控制下的条件保持预算和绝对期限 |
| `controls.py` | 物理输出到角速度、油门的转换；不执行RPC |
| `profile.py` | 每档成本/评分数据类型、前沿函数、部署身份与原始成本资料读取；不加载Torch模型 |
| `backend_contract.py` | 配置/后端接口；没有CPU模拟或UDP服务实现 |
| `lifecycle.py` | 周期日历、版本、排队和完成事件；Windows原生接口已裁掉 |
| `runtime.py` | 将观察、未来槽、选模、跳过/补更接到在线决策 |
| `scheduler.py` | 当前有限候选策略、倒推、整组检查和版本提交规则 |
| `forecast.py` | 根据预计耗时计算所有任务能否按期，限定整任务DPU |

**`forecast.py`中的 `simulate()` 是时间预测器，不是CPU模拟DNN。** 它接收“这个任务预计占DPU 1 ms”等数值，算完成时刻；不会执行网络层、读取权重或计算控制输出。这个功能属于在线时序准入检查，应保留在PS。名称沿用原实现方便逐项核对。

`runtime.py`还保留Python的锁、Future和事件状态，作为现有行为参考；PS的C++版应替换为板端事件/队列实现。它没有模拟DPU执行的worker，也没有用sleep伪造DPU耗时的运行入口。

构造器保留原参考版本的经验成本参数，包括默认3 ms网络余量。它们是迁移前的输入配置；本地DPU链路必须重新测量并配置，不能将Windows侧3 ms余量当成硬件固有时间。原模型的经验成本也不能直接改称WCET。

## 3. 什么仍在Windows，什么需要Ubuntu补上

| 位置 | 内容 |
| --- | --- |
| Windows | Unreal/AirSim、画面、实验准备、场景创建、结果分析 |
| PS需要实现 | 独立的绝对日历释放、RPC状态采样、本包数值/调度逻辑、真正的DPU派发、控制RPC、逐槽计时 |
| PL已有 | release001的18档DNNDK DPU模型，整任务不可抢占 |
| 验证目录 | Python对照测试、固定输入与期望值；不进入正式PS控制循环 |

现有本地硬件接口参考是 `board_package_requant18_release001/pool237_runner.cc` 的 `Pool::find(uint32_t)` 和 `Pool::infer(index, raw237, normalized=false)`。Ubuntu已有该文件，不必再带一套CPU推理或WindowsUDP后端。保留量程检查、板端单次归一化和真实DPU调用。模型身份仍为tag `9253dd1fc87f8d7f`，RAW237输入。

当前还缺少：板端本地DPU适配、ARM RPC客户端、PS控制循环/线程分工、构建部署，以及端到端板端实飞。这些不能因为本包的数值检查通过就写成完成。

参考轨迹计算保留 `FastReferenceWindow`；最新版本通过 `trajectory.from_layout()` 校验并装载冻结数据，不依赖旧规划器。验证脚本调用同一运行装载器，避免把必需逻辑只放在fixture里。

## 4. 裁剪怎样追溯，怎样验证

`SOURCE_MAP.json`列出每个保留符号的原始文件、源文件SHA256和原行号；改动主要为本地导入重连、类的方法裁剪、数据类型改名，以及限定整任务检查。`OLD_SOURCE_AUDIT.json`逐一说明上一源码包中的文件是“仅保留指定符号”“只用于核对”还是“整个模块不再携带”。共16个源文件只提取必要符号，3个文件仅用于核对，84个旧源码/测试/导出模块不再携带。这不是对所有历史研究路线的死代码判断。

本包生成代码是移植规格快照。不要在原工程与生成副本长期分别修改；修正后应回到源文件或导出脚本，并重新做对照。原始运行链未改，本包不能自动替代Windows正在使用的代码。

在解包根目录运行（Python≥3.11；数值核对另需NumPy，无Torch/SciPy/OSQP）：

```bash
python3 verification/verify_bundle.py --root .
python3 -m unittest discover -s tests
python3 -m verification.check_ps_scope
```

测试是原有核心日历/运行时测试经导入路径调整后的对照，使用测试替身不执行DNN。轨迹加载测试另需NumPy；纯日历测试可分别以 `-S` 运行两份原核心测试。数值fixture核对237D、参考窗口、控制转换和固定日历。测试不需要板子，不能代替新板端的物理资格或WCET验证。

本轮29项状态机/调度检查通过；10组特征及参考窗口、3组命令转换、6组冻结日历一致。另外1000组合成日历的新旧决策、顺序、选模和已接受方案的slack一致，400组不同姿态/速度/控制的保持界核对一致，预算最大差为0。对应 `verification/original_parity.json`；均为原实现等价性检查，没有新增板端实测或安全收益结论。

下一步请Ubuntu Agent从这个职责清晰的内核开始移植，继续采用此前PS交接文档的阶段与验收要求。旧Windows入口仅在确实需要复查事件细节时回查，不能作为必须搬到板端的整棵依赖树。
