# 路线B唯一迁移交付：Python框架、现有C++、模型、SDK与验收证据

日期：2026-11-16。**本包取代此前分散的完整旧工程、Python裁剪包和C++补充包作为迁移输入。不叠加覆盖旧目录。**

## 1. 这次到底交付什么

这是一个**不依赖旧工程文件的源码与资产交付包**。包括已经有的Windows C++定时释放/采样/命令、Python调用接口、Python调度/安全/输入构造、真实板端C++推理驱动、全部18档ELF/可执行文件、构建所需RPC源码、模型归一化、黄金和原始记录。

**它不是已经完成Linux移植的PS可执行程序。** Windows C++源码虽然提供了相应执行职责，却仍使用Windows时钟、等待和DLL接口。新义务日历与这一原生IO分支也还没有完整接通。把这些文件装进一个包，不能等同于已经完成接线、ARM构建与上板验收。此前称“完整”时没有把这两层讲清楚；本次不再把缺失职责隐藏在外部依赖里。

“独立”在本次验收中意味着：解包后不读取旧工程/旧压缩包，即可导入算法、核对特征/日历/接口、重算黄金/误差/计时记录，并取得原生IO和DPU的全部现有源码、模型与客户端构建依赖。最终PS闭环还需要下面明示的移植工作。系统工具链、DNNDK驱动和Unreal不属于能随源码复制出来的应用文件。

## 2. 按一次控制的顺序找代码

| 步骤 | 本包真实代码位置 | 当前状态与Ubuntu工作 |
| --- | --- | --- |
| 按绝对时刻释放任务 | `platforms/windows/native_runtime/native_io.cpp` 的日历 | 已有Windows实现；迁移QPC/等待/线程亲和，不把release改成上次完成时间加T |
| 采样状态与碰撞 | 同文件 `snapshot()`、采样线程 | 已有AirLib RPC实现；复用16个状态量和时间戳契约；不是完整237D编码 |
| Python/C++事件交接 | `platforms/windows/framework_native_io.py` | 已有ctypes绑定，仅改两行导入指向新内核；保留ABI/版本/丢弃逻辑 |
| 生成参考与RAW237 | `source/ps_core/trajectory.py`、`features.py`、`types.py` | 原本主要是Python，按冻结10组fixture移植；不需要训练或MPC求解器 |
| 危险预算与三期限 | `source/ps_core/hold.py`、`scheduler.py` | 提供参考实现；模型界的物理可信度尚待验证，不能称严格安全证明 |
| 联合选择T/c/顺序与版本提交 | `source/ps_core/scheduler.py`、`runtime.py`、`lifecycle.py` | 有界参考实现和离线/此前Windows实飞证据；PS本地编排待接 |
| 真实DPU执行 | `reference_assets/deployment/pool237_runner.cc`、`pool237_config.h`、`elf/` | 已有DNNDK `Pool`；复用归一化、量程检查和真实推理，替代Windows到板端UDP推理 |
| 命令转换与下发 | `native_io.cpp`、`source/ps_core/controls.py`、`contracts/` | C++现成代码可复用；务必区分Python API和C++/最终RPC的符号 |
| 完成事件与新控制配对 | `framework_native_io.py`、`ps_core/runtime.py`、`lifecycle.py` | 各自逻辑已提供；PS需完成统一事件编排，ACK仍只是生效时间代理 |
| 保存逐槽记录和评价 | `evidence/windows_reference/` 的原始结构与交接要求 | PS需产生同口径新记录；已有记录只用于迁移前参照 |

**没有重写一份C++功能的Python替代品。** Python内核由旧Python函数裁剪/整理；C++ IO和板端runner保留原始字节。调用桥只把 `RawDroneState` 和控制转换的导入换到新内核，两处变化记录于 `PACKAGING_EDITS.json`。不要把 `Pool` 接收RAW237误认为它能从飞机状态构造RAW237。

## 3. 目录及冗余边界

```text
README_FIRST.md                    唯一入口：本文
source/ps_core/                    12个Python职责参考模块
source/config/                    冻结路线、油门标定
source/tests/、verification/      原算法检查与数值fixture
platforms/windows/                原C++ IO、迁移导入后的Python绑定、构建及原生测试
platforms/board/build_runner.sh    原DNNDK链接命令的本包路径版本
reference_assets/deployment/      完整且唯一一份冻结18档部署
third_party/AirSim/               原生IO所需RPC客户端/头文件/依赖源码
contracts/                        Python API—最终RPC—C++符号对照
evidence/                         原始黄金getter、8640条计时/输出、5轮成功/失败日志
verification_inputs/              3条真实量程拒绝输入
tools/                            本包离线验收和旧阅读复用比对
docs/UBUNTU_READING_CORRECTION...   无需推倒重读的增量任务
MIGRATION_MAP.json                旧文件→新职责、符号与源哈希
ASSET_ORIGINS.json                 每份材料的原始来源和哈希
FILE_ROLES.json                    用途分类
HANDOFF_MANIFEST.json              本包逐文件哈希
COMPLETENESS.json                  已提供/尚未实现/未证明
```

以下没有装入：CPU模拟DPU执行器、CPU分段DNN、Torch训练、MPC教师求解、旧25D/12D模型、旧量化包、所有历史实验、Unreal工程、Windows DLL/obj/build缓存。`forecast.py` 是准入所必需的时序计算，不执行神经网络，不是CPU冒充DPU。

Windows代码保留，是因为其定时/采样/下发语义值得复用，并不表示最终在Linux编译它的Windows系统调用。IO内的mock分支和原生测试是离线验收，正式PS不使用mock。第三方保留的是构建头文件及RPC实现，不是另一套控制框架；其中库自带的跨平台分支不应被误称为我们额外的研究实现。

部署中的runner已链接18档ELF，另附ELF用于重新链接；二者用途不同。黄金和validation中部分数据原格式重复，为保持冻结哈希不改写。该部署目录只保留一份，不再另存一份partial deployment或C++源码副本。它的 `PENDING_BOARD_RUN` 冻结元数据不覆盖后续原始getter/正式汇总证据。

## 4. Ubuntu先验收，不连接板子

Python≥3.11及NumPy。不要为了运行这套参考内核安装Torch、OSQP、SciPy。

```bash
sha256sum -c route_b_unified_20261116.tar.gz.sha256
tar -xzf route_b_unified_20261116.tar.gz
cd route_b_unified_20261116
python3 -B tools/check_unified.py
```

该命令在包内逐项执行：全部文件哈希、31项原Python测试、10组输入/参考+3组Python控制+6组日历、576条黄金、8640条板端输出/计时/误差/模型getter、5轮原始日志、完整本包部署哈希，以及Python/C++最终命令符号核对。没有旧文件绝对路径回读，没有访问DPU或AirSim。数值核对和模型哈希不是新硬件试验，也不是WCET证明。

单独读取本包成本/模型身份：

```bash
PYTHONPATH=source python3 -B - <<'PY'
from ps_core.profile import read_profile
modes, identity = read_profile('evidence/release001', deployment='reference_assets/deployment')
print(len(modes), identity)
PY
```

黄金TXT为245列：RAW237 + INT8物理4 + FP32物理4。本版板端输出与第237～240列（0起算）比较。正式MSE沿用原验证float32的 `u_std`；480条是已有工程对照，不是未触碰保护TEST。成本选择用冻结TRAIN512评分，不用480结果重新选候选。

## 5. Windows原生IO能够怎样单独构建和检查

在已装VS2022、CMake、Python、NumPy、pytest的Windows执行。建议把包解压到短目录（例如 `F:\route_b_unified_20261116`），避免MSVC/CMake的深路径问题。脚本默认使用**本包**SDK，DLL是新编译输出，不随冻结交付携带：

```powershell
# 当前目录为解包根目录；不需要旧工程和F:\Airsim
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\platforms\windows\build_native_runtime.ps1
$env:PYTHONPATH = "$((Get-Location).Path)\source;$((Get-Location).Path)\platforms"
python -B -m pytest .\platforms\windows\test_native_io.py -q
```

这组测试运行真正编译的C++日历/事件/命令代码，以mock RPC检查时序和竞态，不执行DNN、不证明完整飞行。Ubuntu无需编译Windows DLL来做PS移植；阅读它的语义和ABI即可。新义务调度器尚未与本包Windows C++桥组成可直接飞行的入口，不能把这项未完成接线写成已经验收。

## 6. Ubuntu/板端实际迁移顺序

1. **保留阅读成果。** 先读增量纠正文档和 `MIGRATION_MAP.json`。已经读过的RAW237、三期限、DPU、义务日历无需重读；原始文件哈希可自动确认。停止追踪旧CPU执行/训练导入链。
2. **本地DPU。** 原 `Pool` 是现有C++，直接复用核心；归一化只做一次，237补零到240，保留量程拒绝。先单进程黄金，然后接本地调用；不是把UDP绕回本机称为PS推理。现有5239服务无需因阅读或离线验收重启。
3. **Linux IO。** 迁移原生绝对日历/异步RPC/工作槽/事件/版本语义；将QPC、WaitableTimer、Windows句柄/绑核换成Linux实现。CPU管理和DPU整任务非抢占分开建模。CPU有核不表示可以中断正在执行的DPU任务。
4. **端到端接线。** 到点发布→采样并绑定实际控制版本→构造参考/RAW237和期限→选择DNN/队列→本地DPU→过期检查→下发→记录ACK→更新配对。新配置只改可修改的未来任务；旧结果晚到直接丢弃，不推迟未来release来隐藏R>T。
5. **先固定，再动态。** 使用同一冻结case、4m门、匹配参考1.5s的接入初态，从单机T50开始，继而双机固定/动态各三次；保留失败和完整逐槽时间。静止起飞另测，不能与飞行中接入混为一组。
6. **最后扩展。** 计划跳过默认关闭。补更资格、相位与保持界尚待完整物理验证；超网仅接口，不把独立18模型当成已完成超网或DPU段级抢占。不要在此次迁移顺手改控制增益、训练/量化、路线或门宽。

本包有原可执行runner和ELF，旧服务可以继续用；若需在Ubuntu重新构建原runner：

```bash
export R18_SDK=/actual/path/environment-setup-aarch64-xilinx-linux
export R18_DNNDK=/actual/path/existing_dnndk
bash platforms/board/build_runner.sh /new/output/runner_build_001
```

这是原链接命令的路径适配，输出到新目录，不改冻结部署。它只构建现有推理服务，不是完整PS控制器。脚本依赖既有匹配的工具链、sysroot、DNNDK；不要安装新版本替代。该构建脚本在本次Windows打包任务中未进行ARM实编译，历史成功命令和getter证据随包保存。

## 7. 命令符号必须按客户端层区分

物理控制为 `[F,wx,wy,wz]`。例如角速度 `[0.3,-0.2,0.1]`：

| 位置 | roll | pitch | yaw |
| --- | ---: | ---: | ---: |
| 我们传给Python客户端 | 0.3 | 0.2 | -0.1 |
| Python客户端内部再次转换后的RPC | 0.3 | -0.2 | 0.1 |
| 我们应传给AirLib C++ | 0.3 | -0.2 | 0.1 |

所以不能把Python fixture的pitch/yaw直接传给C++。`contracts/wire_command_cases.json` 给出最终RPC三组样例，原C++已经处理正确。油门限幅后实际施加的F也要反算后用于保持界，不把原始越限F当成真实控制。

## 8. 回交什么，怎样算成功

交回：实际修改的C++源码、构建/部署命令、工具链/SDK/二进制哈希、PS/DPU核与线程分工、全部黄金结果、每个任务计划/实际释放及各阶段时间戳、模型选择和决策提交/拒绝原因、碰撞/过门/身体余量、所有失败记录。

分别报告：源码与数值一致、黄金通过、完整飞行、期限满足、动态实际安装、相对静态收益。这六项互不替代。已有单/双机Windows成功、双机1条入板前丢弃和量程拒绝都保留；本包没有把它们改成全PASS，也没有新增硬件或安全收益结论。
