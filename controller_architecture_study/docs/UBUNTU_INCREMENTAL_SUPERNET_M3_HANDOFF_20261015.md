# Ubuntu Agent 交接：从 FALCON 基线到可续算的多回路神经控制器

版本：2026-10-15，M3 硬件功能原型交接 v1。

本文既介绍研究背景，也规定下一阶段实现边界。即使接收者没有之前的聊天记录，也应能根据本文及随附文件开始工作。

**当前决策：冻结已经完成的 M2 v3 整数部署包，作为 M3 功能原型候选，推进自定义 PL 执行器。保留 `NUMERICS_PASS_QUALITY_NOT_MET` 原始状态，不改门槛、不宣称全部质量验收通过。**

本交接不要求重新训练，不要求重新生成数据，不要求在 Ubuntu 安装 AirSim，也不要求先开展大规模研究实验。正式综合、实现、上板等耗时步骤请给用户可直接运行的命令；不要未经说明启动长训练或更改现有板卡启动镜像。

## 0. 接收后先做什么

1. 解压压缩包，阅读本文件，运行完整性检查和小规模整数黄金检查。
2. 核实 Ubuntu 中实际安装的 Vivado/HLS 版本、目标器件和现有板卡基础工程，形成工具链记录。
3. 在冻结包之外新建工作目录，开发与整数参考一致的 HLS 核和 C 仿真。
4. 再做综合/实现、板端驱动与黄金验证，得到可暂停、恢复、增深、增宽的真实硬件执行器。
5. 最后接入多回路调度器和 Windows 桥接。集中进行时延、控制效果和切换连续性对比。

不能用「独立编译 136 个网络，然后轮流运行」替代本任务的共享增量执行目标。

## 1. 基于的 FALCON 是什么

### 1.1 原工作

基于的论文是 Siwei Ye、Jintao Chen、Yehan Ma、An Zou 的 **FALCON: FPGA Accelerated Real-Time Intelligent Controller for Autonomous Systems**，RTSS 2025，DOI：[10.1109/RTSS66672.2025.00040](https://doi.org/10.1109/RTSS66672.2025.00040)。

本包附有用户现有论文的文本提取件 `background/FALCON_extracted_text.txt`，便于核查 §III–V；公式提取可能丢符号，精确复现原指标时仍应核对原 PDF。不要把本文的新设计当作 FALCON 原文。

FALCON 的核心不是简单把一个 DNN 放上 FPGA，而是联动考虑：

- 用 DNN 近似路径规划与 MPC 控制策略，提供多个复杂度/误差不同的控制器。
- 将网络逐层执行、连续层组成的执行段及内存搬运纳入执行模型。
- 用性能模型描述控制器误差、控制周期、响应时间、环境和物理系统对实际控制性能的共同影响。
- 在实时性及 FPGA 资源约束下，联合选择网络、周期、优先级和硬件配置。

原实验的 DNN 输入为 12D：门位置与无人机位置之差 3D、门与无人机姿态差 3D、当前速度 3D、当前加速度 3D。输出为 4D 推力加速度和机体系角速度。原文使用深度 `{10,22,34}` 与宽度 `{512,1024}` 组成六种网络；这不意味着「更深网络总会对所有状态更准」是一般定理。

本工程的 `h33` 表示 33 个隐藏 Linear 层；与原文含输出层的深度计数不能直接混用。

### 1.2 FALCON 已经有哪些调度能力

不能说 FALCON 没有调度或没有周期约束。原文采用固定优先级调度、分段执行和响应时间分析，优先级也是优化对象，执行模型使用隐式截止期 `D_i=T_i`。

可以概括其性能模型为：

\[
J_i=C_i(c_i,T_i,R_i,\epsilon_i,\theta_i),\qquad
R_i=R_i(c_i,h,s).
\]

其中 `c_i` 描述控制器质量/复杂度（原文用 `1/β_i` 表征，`β_i` 是 DNN 标签拟合 MSE），`T_i` 是控制周期，`R_i` 是响应时间，`h` 是硬件，`s` 是调度，`ε_i` 是环境条件。原工作通过采样闭环表现并使用 SVM 回归得到性能函数。

**DNN 对 MPC 标签的 MSE 与飞行轨迹误差不是同一个量。** `1/β` 是模型表征，不是直接把 `1/β` 当作最终飞行性能。原文的轨迹性能指标、权重、时间窗口需要单独对齐；不能拿本包的 normalized MSE 与论文表中数值直接比较。

50ms 是采集或某次实验的配置，不是 FALCON 的物理常数。应满足所选时序模型的约束，而不是事后把所有迟到都解释成「50ms 不重要」。放大 deadline 能降低 miss 数字，但不自动改善控制。

### 1.3 本工程对原工作的继承与偏离

| 项目 | FALCON 基线思路 | 当前研究分支 |
|---|---|---|
| 控制任务 | 多回路共享 FPGA | 保留，多无人机是主要场景 |
| 教师与学习 | 模仿控制策略 | 当前为凸平移 QP＋姿态反馈教师，不是原 ACADO 教师 |
| 输入 | 原文 12D，隐含学习规划和控制 | 237D 显式状态/未来参考；保留已有路径规划器 |
| 控制器族 | 六个独立 FC 网络 | 一个因果共享网络，136 个深度/宽度配置 |
| 运行决策 | 协同设计所得配置和调度 | 目标是运行中结合物理状态、计算进度、共享负载调整 |
| 加速器 | FPGA overlay/分段模型 | 新建支持缓存续算的定制 PL 核；旧 DPU 工程保留 |
| 有效期 | 可调度性约束与性能模型 | 增加状态相关控制有效期监测，尚无全局安全定理 |

因此当前工作应称为 **FALCON 思路上的研究扩展**，不是原论文全部细节的逐项复现。旧 12D 模型可作为单独的复现轨道，但不要把两条轨道混写。

## 2. 我们希望解决的问题与候选创新

### 2.1 中心研究问题

> 多条反馈回路共享加速器时，额外计算带来的控制改善，何时值得等待，何时应交付当前结果，何时应让给其他回路，何时应放弃旧观测重新采样？

这里同时存在四类因素：模型近似误差、观测老化、其他回路等待、已经付出的计算。不是只优化推理 MSE 或单任务 latency。

响应时间 `R` 也不是可独立指定的小常数。它由网络/出口、任务顺序、硬件、搬运、队列、通信及系统抖动共同决定。

### 2.2 动态周期与有效期

记第 k 次实际采样时刻为 `s_k`，对应控制实际生效时刻为 `a_k`：

\[
R_k=a_k-s_k,\quad T_k=s_{k+1}-s_k,\quad
\Delta_k=a_{k+1}-s_k=T_k+R_{k+1}.
\]

若当前状态下允许从这次观测到下一次控制生效的最大间隔为 `Δ_max(z_k)`，可使用：

\[
T_k+\widehat R_{k+1}\le\Delta_{\max}(z_k),\qquad
T_{\max,k}=\Delta_{\max}(z_k)-\widehat R_{k+1}.
\]

这是所定义时间模型下的预算表达，不是脱离假设的安全证明。若 `T` 用名义释放周期而非实际采样间隔，必须额外计入释放与采样抖动。

门框剩余间隙减少、横向接近速度增大等可能收紧有效期；响应时间增大也会压缩下一周期预算。实际危险程度不能只用「离门中心距离」表示，需考虑到边界的余量、动力学、旧命令、模型误差和扰动。

更准的控制器可能降低后续误差，允许更晚更新；但为获得它而等待又消耗了余量。这是联合决定下一计算阶段和下一周期的动机，尚待实测支持。

### 2.3 超网为何是候选机制

独立模型池可在请求开始前选网络，但推理开始后改选另一独立模型，通常无法复用已计算激活。当前共享因果网络希望支持：

1. 同一观测上从浅/窄配置继续精修，不从输入重算整个大模型。
2. 在真正支持的阶段边界暂停，先服务其他回路，再恢复上下文。
3. 利用已完成进度估计「还需多少计算」，而不是总把网络视为一个黑盒任务。
4. 通过共同标签、蒸馏和配置关系保持约束，减少不必要的配置间输出分歧。

但请严格区分：

- 独立 DNN 也在近似同一个 MPC 函数；「学同一个函数」本身不是超网独有性质。
- 共享权重不自动保证输出平滑；整数续算一致也不保证输出随深度单调改善。
- **同输入配置切换差异不等于真实时间序列震荡。** 后续仍需测命令变化、动力学和飞行效果。
- 保存/恢复、分组读权重、输出头、DDR 竞争都有开销。续算不是免费计算。

### 2.4 候选学术贡献，而非已经证明的结论

候选主线是「面向动态控制有效期的可恢复计算控制器」：将状态相关质量、观测年龄、已完成计算和后续任务释放纳入多回路资源决策。

可能的贡献包括：

- 建立质量—年龄—下一更新需求的联合可行性/风险模型。
- 在最低可接受控制交付约束下，依据精修收益和队列外部成本调度可选计算。
- 用严格因果连接实现深度和宽度均可增量的执行图，并报告为此付出的容量和缓存代价。
- 联合考虑量化后有效精修收益、DDR 流量与上下文成本，识别「可以续算但不值得续算」的区域。

不能声称这些组合已经构成已证实的新颖性，更不能声称已有稳定性或最优性证明。早退、蒸馏、sandwich、可选计算、状态感知调度和子图缓存均有已有工作。

相关工作的已核对记录在 `background/DYNAMIC_CONTROL_SCHEDULING_FRAMEWORK_20261005.md` 和 `background/SCHEDULING_RESEARCH_LOG_20261005.md`。其中包括 SCENIC、RICH、TimelyNet、SUSHI、TSUF/VCS 等。此处不要求先重做一轮文献检索再开始实现。

**内存不是唯一动机。** 既有 18 个模型的裸参数粗算约 380MiB FP32／95MiB INT8，不能据此说 1GB 必然装不下。真正要量化的是共享进度复用、搬运、时效和闭环收益。

## 3. 当前工程已经做到哪里

### 3.1 历史基础：请不要误当成本包验收

Windows/AirSim 中已经有：MPC 教师、多个独立 DNN、单/三机飞行、模型/周期联合选择及共享 GPU 推理原型。曾出现同步释放、Python 在线旁路分析、计时定义等造成的时序失败，后来进行了错峰释放和计时/旁路修正。

旧模型的 Ubuntu→DNNDK/DPU→UDP→Windows/AirSim 链路也曾跑通，板端 IP 曾为 `192.168.77.2`、UDP 端口曾为 `5005`。这是历史配置信息，不是本次开机已确认的连接状态。

这些说明已有系统经验，但**不能证明当前 237D 超网、当前整数位宽、当前自定义 PL 的时延或闭环效果**。不要复用旧 25D DPU 的时延作当前模型数据。

### 3.2 当前阶段表

| 阶段 | 状态 | 真实含义 |
|---|---|---|
| M0 数据/输入接口 | 已有，冻结沿用 | 237D 特征、教师和拆分不改 |
| M1 FP32 增量超网 | 已完成用户训练与导出 | 一个共享网络，136 配置，软件续算数值检查通过 |
| M2 v3 QAT/整数包 | 已完成，带质量例外 | 数值检查通过；最大/网格 MSE 目标通过；严格切换不退化目标未过 |
| M3 定制 PL | **尚未实现** | 本次 Ubuntu Agent 的主要任务 |
| M4 板端增量调度与新桥接 | **尚未接入当前模型** | 旧 GPU/模型池原型只能借鉴，不能标作完成 |
| M5 集中研究对比 | **未完成** | FPGA、量化闭环和抗震荡优势均待验证 |

包内 2026-10-12 的背景文件保留历史版本，其「正式训练尚未运行」「M2 尚未实现」等进度句已过时。**当前进度以本交接、`freeze_manifest.json` 和 `deployment/export_report.json` 为准；数学接口以冻结代码和 spec 为准。**

### 3.3 数据与教师

当前教师是严格凸的平移加速度 QP，附带四元数姿态反馈分配，将首步加速度转换为 `[F,ωx,ωy,ωz]`。它是新增基线，不是 FALCON 原 ACADO/PAMPC 的等价实现。

教师预测步数 20、预测间隔 0.1s，输入提供 21 个参考状态。**预测步长 0.1s 不等于实际控制周期，改变运行周期不应偷偷改变模型输入契约。** 仍采用滚动优化只执行首个动作的标准形式；参考轨迹使用已提交的全局连续时钟，不要每帧从当前位置重建一个总在「刚启动」的参考而导致不前进。

数据由不同状态/路径条件下独立求教师标签构成，不要求行之间连续；包含 startup、tracking、transition、braking、position/velocity/attitude/combined recovery。恢复补采样用于扩大有效覆盖。不能把随机相邻数据行当成相邻时间步施加「时序平滑」。

Windows 数据位置：

```text
F:\FalconReproduction\airsim_control\controller_architecture_study\results\decision_convex_recovery_v1\datasets\features_recovery
train.npz       153600 行
validation.npz    9600 行
test.npz          9600 行
```

训练环境 ID 为 0–47，验证为 48–53，测试为 54–59。恢复样本增加在训练侧，因此不能不看分布权重就比较总体 train/validation 比值。测试没有用于当前选模/QAT/校准；本次冻结操作也不读取 test.npz。

数据 SHA256：

```text
train.npz:
974e26269053bc9bda664c125332b69623818b8c0b9189470275fb32d37eb1f8
validation.npz:
62da58b5ded2cfbf79e2f00d7ac526fad18b8004a936f2fe8073cb1ed3919a18
normalization.json:
fe9c44ee80aef1cf97138536a66a4ee3a126f459f2ef963370db354ada8f19ac
```

**M3 包不携带数据集，也不需要数据集就能开发黄金向量对应的硬件。** 不要为了综合/驱动开发先要求重传所有训练数据。

## 4. 必须保持的 237D 输入和 4D 输出

### 4.1 输入布局

这是原 300D 特征的明确投影，名称为 `decision_convex_local_relative_pv_237d_v1`。不是原 12D 补零，也不是旧 25D。

| Python 切片 | 维数 | 内容 |
|---|---:|---|
| `X[0:4]` | 4 | 当前规范化、符号对齐的四元数，`[w,x,y,z]`，变换后坐标约定 |
| `X[4:7]` | 3 | 当前线速度，变换后的 Z-up/FLU 约定 |
| `X[7:217]` | 210 | 21 行、每行 10 维的未来参考，按行展平 |
| 每个参考 10D | 3＋4＋3 | `p_ref-p_current`、参考四元数、`v_ref-v_current` |
| `X[217:237]` | 20 | 参考加速度导出的推力大小 feed-forward |

20 个推力是 `norm(a_ref_FLU + [0,0,9.8066])`。这里的特征常数是 **9.8066**，不要擅自替换为教师设置中的 9.80665。当前测得加速度并不在这个 237D 契约中。

从 AirSim NED 状态到该输入使用向量翻转 `[1,-1,-1]`、四元数翻转 `[1,1,-1,-1]`，还包括归一化与相邻参考符号对齐。只看本表不够重写编码器：权威源是随包 `source_snapshot/fast_convex_preprocess.py` 的 `encode`，以及 `convex_features.py` 的投影定义。当前工程有数值等价的向量化路径，后续 Windows 保留该路径。

首版网络桥接由 Windows 构造 RAW 237D 输入，板端 PS 只归一化一次；不要在两端各归一化一次。

### 4.2 标准化与物理输出

```text
x_norm = float32((x_raw - x_mean) / x_std)
q_input = clip(RNE(x_norm / 2**e_A0), -32767, 32767).astype(int16)
u_physical = float32(u_normalized * u_std + u_mean)
```

`x_mean/x_std/u_mean/u_std` 全部来自当前包的 `normalization.json`。不得换用旧 12D/25D 的归一化。

输出顺序固定：

```text
[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]
单位分别为 m/s²、rad/s、rad/s、rad/s
```

`F` 不是牛顿、不是 PWM，也不是 AirSim `[0,1]` throttle。Windows 的 `control_adapter.py` 执行推力标定，并将 AirSim rate 参数转换为 `[wx,-wy,-wz]`。不要板端先翻转一次、Windows 再翻一次。M3 黄金验证先核对转换前的控制输出。

## 5. 共享增量网络的真实结构

### 5.1 一张网络，136 个配置

```text
network_contract = causal_group_residual_depth_width_v2
input_dim = 237
output_dim = 4
group_width = 128
depth_choices = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33]
width_choices = [128,256,384,512,640,768,896,1024]
```

17×8＝136。深度包含 stem，不包含输出 head；深度 33 表示 stem＋32 个残差层。内部偶数层也必须执行，只是没有选作输出深度。

stem 是 237→1024 的 Linear＋ReLU。之后为 32 层按通道组下三角连接的残差层。输出组 g 只依赖上一层组 `0..g`，不依赖未来组；组索引从 0 开始。原始残差比例 `1/sqrt(32)` 已折入量化分支权重和偏置，PL **不能再次乘 alpha**。

17 个深度各有一个 1024→4 线性 head。同一深度下，宽度增加通过新增通道组的 head 贡献实现；head bias 只加一次。不同深度是不同 head，不能将浅层 head 的累计值直接当成深层 head 的前缀。

这是受限连接网络，不等价于原 dense 33×1024 网络的任意切片。有效参数约 19,220,548，训练中的稠密分配参数约 33,900,612。GPU 掩码运算不等于硬件已实现稀疏加速。

### 5.2 增深、增宽与依赖

- 增深：为已选组继续执行尚未完成的层。
- 增宽：沿所需深度补算新增组；依赖的较小组旧激活保持不变。
- 执行 `(layer=d, group=g)` 前，必须具有 `(d-1,0..g)`；stem 则读取完整 237D 输入。
- 可以先深后宽或先宽后深；补齐到同一配置后，结果必须逐位相同。
- 返回较小配置可以读取已经缓存的对应 head，不需要清除更深/更宽的进度。
- 不允许将上一个采样或另一个无人机的激活直接复用为当前状态的激活。

每个上下文维护 `frontier[8]`，表示各通道组已经算到哪一层。不能只保存一个「当前深度/宽度」矩形值，因为中途增宽时不同组可能处于不同深度。

### 5.3 缓存与提交约定

最大隐藏激活缓存为 `33*1024*2 = 67584 bytes = 66KiB`／采样，另加输入、frontier、head 贡献、标记与元数据。8 组、17 个 head、4 个 INT64 原始组贡献全部缓存时约再增加 4352 bytes。不要误以为保存最后一层 2KiB 就足够。

上下文至少绑定 `loop_id, sample_seq, package_version, input_contract`；更新输入、模型或量化尺度后必须使旧上下文失效。候选头的原始组贡献须区分深度和组。

暂停边界是完整输出组结束，且残差/ReLU/重定标及缓存写回已经完成。可以从更细 tile 实现内部流水，但首版不对外承诺 tile 中间任意抢占。

**首版每条采样只提交一个控制命令。** 可以在内存形成多个候选，但不能擅自改成「先控制一次，再把精修结果再控制一次」，那会改变时序/动力学模型。

## 6. 冻结版本及 M2 结果

### 6.1 冻结来源

Windows 源目录：

```text
F:\FalconReproduction\airsim_control\controller_architecture_study\results\incremental_supernet_237d_m2_lsq_v3
```

QAT best epoch 为 300，SHA256：

```text
e6100b0612b8be6ac0a78786632d96b653f7b30fd6cc4a3eb563dd4ecc323bdc
```

它同时是 `quantization_spec.json` 中的 `package_version`。FP32 源为 v4 grid5000 的 best epoch 4983，SHA256：

```text
d9b9c0fd7ffb394f1c11099056e38308d6ef685d9cc99d01e3cd47698f3b2dca
```

量化契约是 **`causal_w8_head16_a16_lsq_dyadic_v3`**，不是旧 W8A8、旧固定 power-of-two v1/v2，也不是 Vitis AI 1.2.1 自动量化的模型。

### 6.2 已完成的训练概要

FP32 网络经过初始联合训练、恢复适配及追加 5000 epoch；M2 经全 W8、W16 head 和当前可学习尺度三个版本逐步改进。

当前 v3 从 mixed v2 best 初始化，正式执行 300 epoch：前 100 epoch 联合学习尺度与权重，后 200 epoch 冻结部署尺度并微调。骨干及最大 head 起始 LR 为 3e-7，其他 head 为 1e-6，尺度 LR 为 3e-4。校准仅用 TRAIN 的 8192 行；保留共同 FP32 教师、逐配置输出保持及相邻配置差异保持项。实际设置完整保存在 `provenance/config.json`。

这不是用 300 epoch 从头学习整个控制器；也不需要在 Ubuntu 再做 300 epoch 才能使用包。

### 6.3 实际结果与明确例外

| 项目 | FP32 参考 | 当前整数 | 解释 |
|---|---:|---:|---|
| 最大配置 validation normalized MSE | 0.000176528589 | 0.000180601990 | 增加约 2.31%，在原定 10% 容许内 |
| 136 配置平均 validation normalized MSE | 0.001551979007 | 0.001486217682 | 改善约 4.24% |
| 固定可信掩码相邻配置差异 MSE | 0.000049907766 | 0.000050404215 | 增加约 0.99%，未达到「完全不增加」 |

135/136 个配置的验证 MSE 比各自 FP32 参考更低；仅最大配置略高。比较是逐配置对各自基线，不能解读为所有小配置都已经达到控制要求。

原质量门槛保持：最大和平均 MSE 不超过 FP32 的 1.1 倍，可信样本上的切换差异不超过 FP32 的 1.0 倍。第三项未通过，因此 `all_quality_goals_met=false`。

最大/网格指标是当前整数验证集结果；报告中的 `same_input_switch_mse` 是配置差异统计，不是时域控制震荡，`extra_switch_mse_vs_fp32` 也不是简单将两个总体 MSE 相减，定义以 `switch_consistency_report.json` 和冻结实现为准。

数值方面：

- 在 32 个 golden × 136 配置上，独立 Torch 整数实现与 NumPy 的宽输出整数逐位相同。
- FP64 fake-quant 模拟与整数物理输出差为 0。
- 当前增量检查为 2 个样本、2 条增长顺序，直接/续算逐位相同，旧前缀不变；不是所有可能执行顺序的穷举证明。
- 全部 9600 验证行、136 配置的整数质量已经计算，**不需要 Ubuntu 再做一遍**。
- 激活裁剪占比为 0；3840/19186688 个有效权重发生裁剪，约 0.0200%，不等于这已经足以证明无影响。
- `FPGA_verified=false`，`flight_oscillation_reduction_verified=false`。

### 6.4 为什么不再因警告阻塞 M3

这是用户明确同意的阶段决策：在整体误差没有崩溃、整数语义已对齐的情况下，允许进行 **带已知质量例外的硬件功能原型开发**。不是把算法质量验收改成 PASS，也不是同意无条件用最小配置控制无人机。

M3 的验收是「硬件是否正确执行冻结整数图，并正确续算」，不等于 M5 的「闭环收益/抗震荡假设成立」。两类状态必须分开记录。

### 6.5 FP32 QAT 代理与整数的差异必须保留

`qat_float32_golden_max_abs_physical_error = 0.0013334813993424177`，因此 `qat_surrogate_within_tolerance=false`。原因定位为 FP32 除法/累加在舍入边界改变少量量化码；不是通过放宽容差将它隐藏。

导出已改为：冻结参数的 FP64 仿真＋独立整数实现逐位核对，同时保留 FP32 代理差异。**板端目标是冻结整数执行，不是 FP32 QAT 训练代理。** 不要用代理输出作为硬件标准答案，也不要把「采用 FP64 验证」误解为 FPGA 需要 FP64 数据通路。

## 7. 冻结包内容、完整性与使用

### 7.1 传输目录

```text
falcon_incremental_supernet_m3_v1_20261015/
  README_UBUNTU_HANDOFF.md           本文件，首先阅读
  freeze_manifest.json              文件大小、SHA256、源路径、批准范围
  deployment/
    weights_mixed.bin               PL/PS 读入的二进制权重与元数据
    weights_mixed.npz               同内容的 NumPy 数组归档
    quantization_spec.json          数学契约与二进制 offset/dtype/shape
    normalization.json
    calibration_manifest.json
    golden_vectors_int.npz
    offline_metrics.json
    switch_consistency_report.json
    export_report.json
  reference/
    integer_reference.py            NumPy-only 权威整数参考
    verify_frozen_bundle.py         完整性与小规模 golden/续算检查
  provenance/
    qat_best_model.pt               冻结 QAT best，追溯用，不是 PL 运行格式
    config.json / completion.json / history.json / initial_metrics.json
    fp32_model_spec.json / fp32_export_report.json
    dataset_split_manifest.json
  source_snapshot/                  量化、网络、训练、特征、控制适配源代码快照
  background/                       历史架构、研究记录、M2 说明及论文文本
```

`weights_mixed.bin` 实际为 **19,665,216 bytes**，约 18.75MiB。`qat_best_model.pt` 为 140,355,582 bytes，包含训练侧参数；不能将该文件大小误算为 PL 裸权重大小。

未包含 train/validation/test 数组、FP32 原检查点、优化器 latest checkpoint、AirSim 工程、Vivado 工程和旧 DPU bitstream。它是完整的 **M3 整数执行交接包**，不是全工程重训环境。

`source_snapshot` 用于理解和追溯，不保证单独拷出其中某个文件就能重跑全套训练/Windows 飞行；其中一些模块依赖完整 Windows 工程。正式 M3 执行仅需 `deployment` 和 `reference`。

冻结是「独立副本＋清单＋归档哈希」，不是硬件写保护。原始 Windows 运行目录未改。请将冻结目录当只读输入，构建输出放到另外的目录；任何权重、布局、舍入、归一化改变都应生成新版本，不能沿用此包的版本和哈希。

### 7.2 Ubuntu 上从上到下运行的接收检查

先将 ZIP 和同名 `.zip.sha256` 放在同一目录。以下命令不训练、不连接 FPGA，也不需要 AirSim。`unzip -n` 避免覆盖已有冻结文件；若同名目录已经被改坏，应另找目录重新解压，不要自动删除。

```bash
sha256sum -c falcon_incremental_supernet_m3_v1_20261015.zip.sha256 && \
unzip -n falcon_incremental_supernet_m3_v1_20261015.zip && \
cd falcon_incremental_supernet_m3_v1_20261015 && \
python3 -B reference/verify_frozen_bundle.py
```

默认检查仅依赖 Python 标准库。`integrity=PASS` 只表示文件正确，不会将模型质量状态改为 PASS。

可用 Python≥3.8 加 NumPy 运行小规模整数检查。优先复用已有独立 Python 环境，**不要升级/破坏旧 Vitis AI 1.2.1 容器**。若已有兼容 NumPy：

```bash
python3 -B reference/verify_frozen_bundle.py --golden --samples 2
```

若系统环境不合适，在已有 Python 3.8–3.11 解释器上建立独立 venv，并安装 `numpy==1.24.4` 即可；安装/联网命令由 Ubuntu Agent 依据实际环境给出，不假定虚拟机已经有该解释器或可访问 PyPI。

参考文件中的 `IntegerExecutor.load()` 使用 Python 3.11 的 `hashlib.file_digest`。附带 verifier 用兼容的分块 SHA256 并直接调用构造器，因此其入口不要求 3.11；旧 Python 不要直接照搬 `.load()` 入口。

### 7.3 核心文件身份

```text
weights_mixed.bin:
8070c9e9c6cd528c4a767a556bb09e80ca6719ba965e21f7a63ceb92fc808f9a
weights_mixed.npz:
b29e1d18fae947bfbae34eccc4b18f73ae6ebe56069301e7476e274dfd1af649
quantization_spec.json:
c854a71f456278d8a9de5291f171948f80e61b2d6f8bdcefe517626e469f1050
golden_vectors_int.npz:
aa88ca376633e0d4047dbc1a413cabd443b83fae0275354aa5246cf83194001f
```

完整哈希以包内 manifest 为准。ZIP 自身的哈希在外部 `.zip.sha256`；manifest 不可能将自己的哈希再无限递归包含进去。

## 8. HLS 必须逐项实现的整数数学

### 8.1 基本位宽与范围

| 对象 | 格式/范围 |
|---|---|
| 输入及激活 | 对称 A16，`[-32767,32767]`；ReLU 后 `[0,32767]` |
| stem/残差权重 | 对称 W8，`[-127,127]` |
| head 权重 | 对称 W16，`[-32767,32767]` |
| bias 与原始 MAC 累加 | 有符号 48 位；包内 bias 用小端 INT64 保存，范围已检查 |
| 每输出通道尺度乘数 M | uint16，1–65535 |
| 尺度基准指数 | 有符号整数，数组保存为 int16 |
| 乘 M、对齐、残差中间量 | 有符号 63 位工作范围，正幅值保守界 `2^62-1` |
| 最终 head 整数 | INT64 存储，**已经包含乘 M**，不饱和到 A16 |

`weight_real = weight_code * M[row] * 2**e_weight[row]`；激活 `activation_real = activation_code * 2**e_A[layer]`。

M 和指数逐输出通道共享，不能针对每个 depth/width 再校准尺度，否则旧前缀编码会改变。输入指数在 `activation_exponents[0]`，各隐藏层为 `[1..33]`。

### 8.2 stem

```text
raw = sum(q_input * W8_row) + bias_row       # 48-bit
scaled = raw * M_row                         # widen BEFORE multiplication
source_exp = e_A0 + e_weight_row
q_stem = saturate_A16(RNE(ReLU(scaled) * 2**(source_exp - e_A1)))
```

### 8.3 残差层

```text
raw = sum(previous_groups[0..g] * W8_row) + bias_row
branch = raw * M_row
branch_exp = e_A_previous + e_weight_row
identity = previous_same_group[row]
identity_exp = e_A_previous
common_exp = min(branch_exp, identity_exp)
combined = (branch << (branch_exp-common_exp))
         + (identity << (identity_exp-common_exp))
q_layer = saturate_A16(RNE(ReLU(combined) * 2**(common_exp-e_A_layer)))
```

先以整数无损对齐、相加，再 ReLU、最后一次 RNE 与饱和。不要先把 branch 舍入到输出尺度再加 identity；不要每个输入组都裁剪；不要再乘一次原残差 alpha。

### 8.4 输出 head

```text
group_contribution[d,g,o] = sum(A16[d,g] * W16_head[d,o,g])  # no bias, raw MAC
total_raw[o] = head_bias[d,o] + sum(group_contribution[d,0..G-1,o])
output_code[o] = total_raw[o] * head_M[d,o]
output_exponent[o] = e_A[d] + head_weight_exponent[d,o]
u_normalized[o] = float32(ldexp(float64(output_code[o]), output_exponent[o]))
u_physical[o] = float32(u_normalized[o] * u_std[o] + u_mean[o])
```

bias 一次、M 一次。跨宽度可复用 raw group contribution；跨深度必须换正确 head。PL 首版可以返回 4 个 INT64 code＋4 个 exponent，由 PS 做末尾浮点转换。

**先逐位比较 code，再比较物理输出。** PS 的中间浮点转换顺序、FMA 合并和编译器 fast-math 可能改变末位；黄金对齐阶段应禁止未经对齐的 fast-math/FMA 变化。浮点近似不能掩盖整数 code 不同。

### 8.5 舍入与溢出

RNE 是 round-to-nearest, ties-to-even，不是 C/C++ `round()` 的 ties-away-from-zero。对负数右移，参考采用算术移位的 floor quotient，加余数和奇偶判定。请照 `integer_reference.rescale()` 编写定点单测，覆盖正负半整数、负数和移位为零。

在移位或相乘 **之前** 检查/推导范围，运算溢出后再检查无意义。HLS 表达式须在乘法/加法之前显式扩位，避免赋给宽变量之前已在窄类型内截断。

`integer_bounds_verified_for_all_A16_inputs=true` 表示参考对当前参数的保守算术范围已检查；它不是 FPGA 时序证明、不是所有物理状态安全证明，也不允许用 INT32 替代指定累加器。

## 9. 二进制布局与黄金向量

### 9.1 权重布局

`quantization_spec.json.array_layout` 对每个数组给出 `offset_bytes/bytes/dtype/shape`。布局为小端、C-order，各数组起始按 64B 对齐。不要根据 `.npz` 内部 ZIP 顺序生成地址，也不要硬编码一套与 spec 脱节的 offset。

主要名称：

- `activation_exponents`。
- `stem_weight/stem_bias/stem_weight_multiplier/stem_weight_exponent`。
- `layer{d}_g{g}_from{j}`，d=2..33，0≤j≤g≤7，形状 `[128,128]`；只打包有效下三角。
- 每个残差层的 `layer{d}_bias/weight_multiplier/weight_exponent`，按输出 row 索引。
- `head{d}_weight`，形状 `[4,1024]`，dtype int16；以及 head bias、乘数、指数。

**一个输出组的不同 from-j 块在文件里是独立数组。** 不能假定把第一个块当 `[128,(g+1)*128]` 连续行就正确；应使用 descriptor 收集 tile，或建立另一个有版本/验证的运行时布局。首版优先直接消费现有数组。

若需要 HLS 常量表/C++ descriptor，将 JSON 转为构建产物而不是修改冻结 bin。数组里原本只需 W8 的上三角零块不应补存、补搬运或补算。

### 9.2 黄金 NPZ 的字段

| 名称 | shape / dtype | 用法 |
|---|---|---|
| `X_raw` | `[32,237]` float32 | PS 输入归一化测试 |
| `X_normalized` | `[32,237]` float32 | 软件中间参考 |
| `X_int16` | `[32,237]` int16 | PL C 仿真与板端主输入 |
| `configurations` | `[136,2]` int32 | 每一行的 depth、width，必须读实际顺序 |
| `outputs_accumulator_int64` | `[136,32,4]` int64 | 权威 head code，包含 M |
| `output_exponents` | `[136,4]` int16 | code 对应的二进制单位 |
| `outputs_normalized/physical` | `[136,32,4]` float32 | 归一化/物理输出 |
| `outputs_torch_accumulator_int64` | `[136,32,4]` int64 | 独立整数实现核对件 |
| `outputs_qat_float32_physical` | `[136,32,4]` float32 | 仅诊断训练代理，不能作为硬件目标 |

32 行来自已有验证集行，行号另存于 `validation_row_indices`；不是新读取的测试集。C++ 不必直接实现 NPZ 解析，可由 Python 生成本地二进制或头文件 fixtures，并保存生成脚本。

## 10. Ubuntu Agent 的具体任务与阶段交付

### U0：冻结包接收及工具链盘点

输入：本包。输出：`TOOLCHAIN_AND_BOARD.md`、完整性检查输出、小规模软件 golden 输出。

只核实必要信息，不展开新的研究扫描：

- `vivado -version`、`vitis_hls -version` 或 `vivado_hls -version` 中实际存在的入口及安装目录。
- 目标 part 的完整字符串、AXU3EGB 板卡修订、现有可用基础工程/XSA/HDF/约束文件。
- 是 Vivado HLS 还是 Vitis HLS；对应 Tcl/接口语法、许可证和可用 C++ 标准。
- 板端 Linux、ARM64 交叉编译工具链、现有驱动、DMA/AXI 接入方式、串口及恢复手段。
- 权重使用 PS DDR 经 AXI HP/HPC 访问，还是已有可用 PL DDR/MIG；首版选择已能可靠访问的路径。
- 宿主 Windows、Ubuntu VM、板卡之间的网段和桥接模式。虚拟机能 SSH 板卡不等于 Windows UDP 已打通。

Vitis AI 1.2.1 是旧 DPU 软件栈版本，**不能据此推断 Vivado/HLS 版本或目标器件**。若某工具未装，只报告缺项和给出匹配方案，不要随意升级现有环境。

### U1：数学正确的 HLS 核与 C 仿真

建议工作结构如下；这是待创建的交付目录，不是声称现在已存在：

```text
falcon_incremental_m3_work/
  TOOLCHAIN_AND_BOARD.md
  hardware/hls/          源码、头文件、testbench、生成 fixtures、Tcl
  host/                 PS 侧布局读取、上下文与驱动
  protocol/             新接口及客户端
  scripts/              可复现构建和运行入口
  reports/              C sim、综合、实现、板端结果
```

先交付一个正确可综合的时分复用 GEMV 引擎：

- 支持 stem W8、残差 W8、head W16、A16、48 位 MAC、M 与指数重定标。
- 支持完整组结束的缓存写回和状态返回。
- 支持选择需要的 layer/group/head，不默认一次计算全部 136 配置。
- 参数来自 DDR/AXI；不要将约 19MB 全部变成常量 ROM 或片上 BRAM。

先实现通用正确性，再在同一设计上加入受控并行和双缓冲。首版可把 descriptor 与依赖管理放 PS，PL 只执行一个或一串合法组；不要求立刻把完整调度器搬进 PL。

必要验证：

1. RNE、乘数、残差对齐、W16 head 的小定点单测。
2. 用 golden 的 X_int16 对齐整数 code，至少覆盖最小/中间/最大配置与两种增深/增宽顺序。
3. 在交付功能版前，自动遍历 32×136 golden 的 C 模型；可利用同一输入的合法缓存减少重复工作。
4. 若采用 HLS 类型/流水改变运算顺序，保证宽整数语义不变；对接 RTL 后保留代表性 co-sim。全量板端组合扫描放 U3/M5，不在每个小改动后重跑全部飞行。

C 仿真通过才继续综合。出现不一致时打印第一个 layer/group/row 和期望/实际 code，不要放宽容差蒙混。

### U2：可实现的 AXU3EGB PL 系统

ALINX 手册所列常见 AXU3EGB 配置为 PS 4GB DDR、PL 1GB DDR；ZU3EG 资源表为 360 DSP、216 个 36Kb BRAM、无 UltraRAM。**请以实际板卡和器件数据库确认，不用此表替代 part/约束核查。** 资料入口：[ALINX 手册](https://www.alinx.com/public/upload/file/AXU3EGB_User_Manual.pdf)、[AMD DS890](https://docs.amd.com/api/khub/documents/dGU6Y~1b8XPqDFk5ulti6g/content)。

架构建议：DDR 权重＋BRAM tile 双缓冲＋激活上下文＋一个复用 MAC 阵列。可评估 8 输出×16 输入通道并行、150–200MHz 的起始方案，但不是硬性承诺；W16 head、宽乘数、DDR 和布线均需资源，不得将所有 DSP 预算都预先分给 MAC。

权重启动时常驻 DDR，不是每次请求从磁盘重载；逐层 DDR→BRAM 搬运仍存在。假设有效 DDR 带宽 2GB/s、最大路径有效权重约 19.12MB，搬运下界量级约 9.56ms；这是估算而非本板实测，不能拿 MAC 理论吞吐直接声称亚毫秒端到端。

多上下文可以将大缓存放可访问 DDR、片上仅保留活跃 tile；需计入读写成本。首版不全部展开 33 层、不做 136 个 bitstream、不依赖动态部分重配置、不先做双 lane 分区。

交付：HLS 工程/Tcl、IP 接口文档、Vivado 工程生成脚本、约束、综合/实现报告、bitstream/硬件平台导出文件及可恢复的加载命令。必须区分 HLS 估计时延、RTL 仿真周期和板端实测。

保护旧 DPU 工程和启动镜像。不要覆盖 SD 卡 BOOT.BIN 或停掉旧常驻服务而不说明目标和恢复方式。自定义 bitstream 与旧 DPU 通常不能被假定同时存在；如需切换，先确认已有备份和串口恢复路径。

### U3：板端 PS 驱动、上下文与整数黄金

PS 程序负责：启动时检查模型哈希并加载权重一次、缓存维护/物理地址管理、descriptor、上下文表、版本与序号、硬件完成事件、输出转换。

最低软件接口可以采用以下语义，函数名只是设计建议：

```text
load_package(path) -> package_version
begin(loop_id, sample_seq, raw_or_qinput) -> context_id
advance(context_id, target_depth, target_width) -> ready/progress
readout(context_id, depth, width) -> codes[4], exponents[4]
release(context_id)
```

每次 advance 只能执行缺失节点；返回合法完成边界，不允许未完成 tile 被标记完成。输入版本错、上下文过期、依赖未满足、输出头不匹配应拒绝，不应静默从头重算却宣称「增量加速」。

硬件上下文切换先实现「一个组结束后换到另一个 context」，不要求不受限硬抢占。记录 descriptor 数、MAC 数、权重字节、DMA/队列/执行/保存恢复时间。至少用两个 context 交错，确认无人机 A 的缓存不会被 B 覆盖。

板端验收：按同一 golden 的 code 和 exponent 逐位核对；验证直接/续算一致、旧前缀不变、跨上下文恢复正确。全部黄金验证不经过网络也应能独立运行，便于隔离错误。

### U4：新版 UDP 服务与 Windows 对接

保留旧 FLCN/v1、12D/25D 服务作基线，不将 237 个浮点数直接塞入旧包结构。新协议要明确版本、消息长度和字节序；权重小端格式不代表 UDP 默认也是小端。

首版可开不同端口（例如 5006，最终以确认配置为准），避免覆盖旧 5005 服务。每请求携带 `loop_id, sample_seq, package_version/短版本标识, input_contract, configuration/预算, payload`，回复带实际配置、原请求身份、控制值/整数 code、进度及各段计时。版本/长度/非有限值错误应明确返回。

237 个 float32 为 948 bytes；设计 header 时尽量保持单个常规 UDP 报文不分片，大调试数据另走日志。不要为每帧加载模型、打开设备、重新映射整个文件。

初版由 Windows 维持路径规划与特征编码，PS 归一化/反归一化，PL 运算。Ubuntu 不需要运行 AirSim。网络黄金先用同一 X_raw 发往板子，收到 4D 物理控制并对照整数参考；额外回传 code 时可以严格核对，网络浮点误差与整数误差应分开。

序号迟到不得覆盖较新控制；timeout、drop、cancel、版本拒绝和 fallback 分别计数。任何 MPC/CPU fallback 不能计入纯 FPGA DNN 成功。

现有 Windows 代码还没有为当前定制核完成该协议适配；需要 Ubuntu Agent 提供协议、可用客户端、启动命令与示例回复，随后由 Windows Agent 接入。

### U5：进度感知调度接口，先完整可运行

不要一上来让 RL 同时学习时限、数值、队列和控制价值。先交付可解释基线：固定配置/周期、仅周期调整、模型池选型、可恢复共享网络。

每条回路保存：采样时间、最新输入、已完成进度、最低可接受配置、候选输出、已承诺 deadline、下一释放时间。初版每回路最多一个活跃推理和一个「尚未启动的最新请求槽」，覆盖/丢弃全部计数。

在真实组边界和新请求到来时做决策：先保护最低可接受结果的可行交付，再比较继续精修与服务其他回路的收益。最低可接受配置须依据控制质量确定，不能默认 d1/w128 安全。

选择器要计入自己的成本，不能为计算下一预算偷偷延长当前已承诺时限。`ΣC/T≤1` 不能被视为包含非抢占段、抖动、网络和切换成本情况下的充分保证。先用小规模保守前瞻/响应估计实现，正式定理另做。

质量收益模型尚未对当前板端量化网络重新拟合。可先用已存的每配置验证质量和实测段成本建立功能接口，但须标注为经验启发式，不能声称它是 FALCON 原 SVM 或经证明最优策略。

## 11. 最终目标与集中验收

### 11.1 一阶段可交付成果

用户当前优先要「阶段成果」，不是每一步都新增一轮必要性实验。第一阶段结束应得到：

1. 一份不变的当前混合整数部署包，全部 136 配置保留。
2. 一个实际综合并上板的共享执行核心，可分组推进、暂停/恢复、增深/增宽。
3. 一个常驻 PS 服务和明确的网络接口，不为每帧重载模型。
4. Windows/AirSim 接入方案与可复制命令；可在多个回路间管理独立上下文。
5. 含实测执行、DDR、上下文和网络成本的调度接口。
6. 每阶段输出路径、命令、版本、报告与未完成事项，失败不丢已完成成果。

这时可以说「功能链路建立」，仍不能说学术收益已经成立。

### 11.2 M5 集中对比的必要内容

公平比较三种能力：常驻独立模型池、固定宽度早退、二维可恢复超网。独立模型池也应有合理蒸馏/一致性与相同位宽、资源和状态条件，不能故意构造弱基线。

比较维度：

- 请求前选定预算与执行中预算变化；固定 N 和动态 N，不能只展示没竞争的三机。
- 同一物理路线/扰动/速度/时长下的轨迹误差、穿门/碰撞、控制命令跳变及必要的平滑度指标。
- 状态采样到控制生效的 R、实际采样 T、输出年龄、超期/丢弃、共同固定时限下的超期统计。
- 模型/配置切换、增量保存恢复、额外 head、DDR 流量、功耗或能耗（有测量条件时）。
- 量化前后有效精修收益，是否有大量不同配置输出完全相同或质量没有收益。
- 不可行/回退的情况，不能只挑通过样例。

最后再回答三个问题：动态复杂度是否确有状态相关价值？共享负载是否造成预算不确定性？可恢复二维结构是否比简单早退和独立模型池带来可重复闭环收益？若没有，也应接受缩小论文结论，而不是强行说超网是必要条件。

### 11.3 性能函数的边界

可以沿用 FALCON 的「闭环性能关于质量、T、R、环境的函数」建模思路，但必须使用当前网络/硬件的数据重新拟合和核验。现有包没有提供可直接部署的、针对当前量化超网与真实 PL 时延拟合好的 SVM。

为避免将来缺数据，日志至少保存实际与参考状态（含坐标/姿态定义）、控制输出、原始采样与生效时间估计、配置/进度、环境/路线、终止原因、队列和各段时延。若没有执行器生效确认，标记时间戳只是发送/接收代理，而不是声称真实 a_k 已知。

复现原性能公式时统一误差权重、物理时间积分/采样权重和倒数处理；终止/碰撞不应通过缩短统计窗口获得更好分数。仅有全局 normalized MSE 不能回填全部原论文控制性能数据。

## 12. 禁止事项与遇到阻塞怎么办

- 不重拆数据，不使用 test 选模，不重训六个或十八个独立 DNN。
- 不将本包重新送旧 Vitis AI 1.2.1 量化器/编译器，替换当前自定义整数语义。
- 不把 head 降为 W8、把激活降为 A8、换舍入、换尺度、重复折叠 alpha 后继续使用相同版本。
- 不为凑「质量 PASS」修改原始报告或降低门槛。
- 不把所有 136 个配置都强制视为已合格闭环控制器，也不删到只剩几个配置冒充完成原目标。
- 不跨采样复用激活，不让 stale output 覆盖新控制，不把 fallback 隐藏为成功。
- 不用仿真周期、理论 MAC 下界或旧 DPU 数据代替当前板端实测。
- 不覆盖旧工程、权重、设备树和启动镜像。工具/接口不匹配时先定位最小缺项。

若硬件资源不足，先缩并行度、调整 tile/cache 布局或时分复用；保持数学不变。若必须改量化格式或拓扑，列出原因和代价并向用户确认，生成新模型/新版本，不偷偷改冻结候选。

## 13. 每次阶段汇报的格式

```text
阶段：U0/U1/…
已完成：实际实现内容
命令：用户可复制的完整命令块，说明在哪个目录/环境运行
产物：绝对路径；源码、日志、bitstream/驱动等
结果：逐位一致性、真实工具版本、综合资源/频率、板端时延等
质量状态：保留原 NUMERICS_PASS_QUALITY_NOT_MET；另列硬件功能状态
未完成/阻塞：具体缺项；不要把计划写成完成
下一步：一个阶段性交付，而不是新增无关验证扫描
```

## 14. 可直接给 Ubuntu Agent 的起始任务

> 请以本文件和冻结包为依据，接手 M3 自定义 FPGA 执行器及后续板端接口。先完成 U0，再实现 U1 的 HLS 整数核心与 C 仿真；不要重训、不要用旧 DPU 编译替代增量执行。保留全部 136 配置，严格照 `causal_w8_head16_a16_lsq_dyadic_v3` 和 NumPy 参考实现，支持同采样的增深/增宽以及不同回路上下文恢复。当前最大/网格精度目标已通过，严格切换差异不退化目标仍差约 0.99%；用户同意作为带已知质量例外的功能原型推进，但不允许更改报告或宣传闭环抗震荡已证实。请先检查实际 Vivado/HLS、part 和基础硬件工程，不猜版本。代码由你编写，长耗时正式命令由用户运行，每阶段给出命令、路径、真实指标、未完成事项。最终形成一个可在多回路间调度的共享可恢复 PL 核、常驻 PS 服务、新 UDP 接口，并与 Windows/AirSim 对接；集中研究评估在功能模块成型后完成。
