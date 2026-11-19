# Ubuntu Agent：237D 独立模型池部署与 DPU 切换／控制抖动实验

日期：2026-10-16。此包为 **独立 FP32 控制器 → Vitis AI 量化 → DPU 多模型池基线**。

**不是旧12D模型，不是旧25D模型，也不是昨天冻结的136配置增量超网。** 三条轨道必须独立目录、独立模型ID、独立协议标识、独立测量记录。

用户让 Ubuntu Agent 完成量化、编译、板端多模型服务和切换实验代码；正式耗时命令由用户运行。Windows Agent 已完成现有模型导出、试飞证据审计及打包，未执行新训练、量化、DPU编译或新试飞。

## 1. 这次要回答的问题

老师要求补充两类实验：

1. 多个独立 DNN 在真实 DPU 上的切换成本，以及模型池扩大后实际控制性能如何变化。
2. 多模型控制的时间抖动和额外控制命令跳变，为后续超网对照建立可信基线。

这些是假设检验，不是预设结论。模型池扩大不保证真实控制性能单调改善；模型常驻后切换可能非常便宜；超网也不能承诺零 jitter。支持、否定和不显著结果都必须保留。

## 2. 与之前12D／25D／超网的区别

| 项目 | 旧12D轨道 | 旧25D轨道 | 本包237D独立池 | 另一个237D超网包 |
|---|---|---|---|---|
| 输入 | 门位置/姿态误差＋速度＋加速度 | 旧12D＋路径上下文 | 当前姿态/速度＋显式未来参考 | 与本包相同237D契约 |
| 教师 | 有多批历史版本，按旧包核实 | 工程化线性MPC | 凸平移QP＋姿态反馈 | 继承同一任务及独立模型教师 |
| 网络 | 六种独立FC/ReLU | 旧独立MLP | plain及dense residual独立网络 | 因果分组残差、17×8配置 |
| 数值 | 旧DPU量化 | 旧DPU量化 | **本次交付只有FP32，尚待量化** | 已冻结W8/head16/A16 dyadic整数包 |
| 部署目标 | 旧DPU服务 | 旧DPU服务 | Vitis AI 1.2.1＋真实DPU架构 | 自定义可恢复PL执行器 |
| 计算复用 | 不默认具备 | 不默认具备 | 独立模型之间不复用激活 | 同一采样上合法增深/增宽复用 |

原FALCON的12D输入和六模型结构只是研究起点；本包是扩展任务，不能称为原论文12D数据格式的严格复现。不要复用旧12D/25D normalization、golden、kernel名称或UDP报文长度。

建议 Ubuntu 项目目录使用 `falcon_independent_237d_dpu_pool_v1`；昨天的 `falcon_incremental_supernet_m3_v1_20261015` 保持原样，旧12D/DPU目录也保持原样。

## 3. 本包到底有哪些模型

共 **27个独立模型**，通过唯一ID区分来源：

| ID前缀 | 数量 | 来源及用途 |
|---|---:|---|
| `main18_` | 18 | 同一恢复数据、seed0、600 epoch的SCENIC 12＋FALCON 6 plain结构，主池候选 |
| `residual600_` | 4 | 4×512、6×1024、21×1024、33×1024，600 epoch dense residual对照 |
| `residual5000_` | 1 | 33×1024残差追加5000轮后的best；与600轮版本不是同一权重 |
| `recovery200_` | 4 | 4×128/256/512/1024，早期恢复补采样版本，有路径关联扰动试飞报告；辅助对照 |

`h` 是隐藏Linear层数，不含输出层。FALCON depth10/22/34对应h9/21/33。不要把编号或参数量当作设备实测快慢。

不包含更早的 `features_only` 消融、重复训练seed1/2和旧12D/25D，避免将同形状的历史权重误认为同一模型。主实验优先使用同一 `main18` 批次；把200轮与600轮混在一起时必须明确训练预算差异。

逐模型结构、参数量、验证MSE、源检查点SHA256、试飞证据和部署状态都在 `model_registry.json`。不要凭同名 `4×512` 判断是同一权重。

### 3.1 推荐起步池

已有哈希匹配飞行证据的四个模型是：

```text
main18_plain_mlp_h4_w256_seed0
main18_plain_mlp_h4_w512_seed0
main18_plain_mlp_h6_w1024_seed0
residual600_residual_mlp_h21_w1024_seed0
```

它们适合先做板端多模型服务与切换工具开发。但residual是否可全部映射到当前DPU，需要编译报告确认，不能偷偷放CPU算残差却称全DPU。

为先得到纯FC/ReLU池，推荐准备以下四个，均来自main18：

```text
main18_plain_mlp_h4_w256_seed0
main18_plain_mlp_h4_w512_seed0
main18_plain_mlp_h4_w1024_seed0
main18_plain_mlp_h6_w1024_seed0
```

其中main18的4×1024 **没有找到完整五门FP32飞行证据**；需要Windows补一次，并在量化后再次确认部署候选能闭环。不能拿recovery200的4×1024扰动报告冒充它。

后续可从main18的2×128、3×128、3×256、3×1024等扩充至6或8个候选。先量化、测质量与时延，再冻结嵌套池和选择规则，不必一开始编译全部27个。

## 4. AirSim试飞审计：有，但不能概括成全部都飞过

### 4.1 哈希匹配的核心四模型

已找到完整的 `joint_model_period_v1/summary.json`，共12项候选：四个模型分别在T=50/75/100ms下试飞。每项是三机DNN实际控制、无MPC fallback，报告标记 `flight_passed/realtime_passed/acceptance_passed=true`，三架各过5门。名义速度5m/s，环境48、路线0。

报告内的检查点SHA256与本次源检查点逐项匹配，相关飞行文件还按历史汇总中的哈希核查。核心权重身份如下：

```text
main18 4×256:
499dc2d02675f1ec0b5685e31c8f24d3f020c82e8440a42d8c3786c26d3a93b4
main18 4×512:
4786b3aa8655de7f37edac40e1e3b9585d94722edce3f47a6d873132d6f186dd
main18 6×1024:
9560a759c3e760ce33cd7d03107845d3ebe7ab2ebeb8910f80835871c0e305f6
residual600 21×1024:
f4e3fe4d43d1e10dc9e0959e8498f1ff7bf8186866b5091a2972494c4103645d
```

所有模型身份检查使用 `model_registry.json` 中的完整哈希与原报告交叉核对，不通过文件名或网络形状猜测身份。

原Windows报告根目录：

```text
controller_architecture_study/results/dynamic_complexity_237d_v1/
  joint_model_period_v1/summary.json
  joint_model_period_v1/trials/*/attempt_001/report.json
  airsim_dnn_20261004T052726_426829Z/report.json
```

包中 `evidence/hashed_flights` 保存相应报告；T50的请求与轨迹日志也附上。

### 4.2 早期四模型：报告PASS，但没有记录检查点哈希

`recovery200`四个seed0模型有左右combined recovery穿门报告，位置在各自 `flight` 子目录。报告写明 `convex_237D_DNN_only`、PASS、5/5穿门；但旧格式只保存检查点路径，不保存权重哈希。

因此注册表标为 `PATH_ONLY_HISTORICAL_PASS_NO_CHECKPOINT_HASH`，不能与前述强证据等同。不要为了补哈希去修改历史报告。

### 4.3 其他模型

对剩余模型，在本次扫描范围内没有找到对应的完整五门验收报告，标为 `NOT_FOUND`。这不证明它们从未飞过，也不意味着一定不能飞，只表示不能根据现有文件声称已验证。

`control_value_geometry_v2_route3`等短窗口报告是转弯/恢复性能测量，明确标记 `short_window_not_five_gate_acceptance=true`，不能当成18模型全套五门验收。

证据总表为 `evidence/flight_audit.json`。FP32 CPU/GPU飞行通过 **不等于** 新DPU INT8版本也通过。量化后仍需必要的数值与闭环检查。

### 4.4 缺失模型的Windows补测命令

先在用户手动打开的UE4项目点击Play；不加 `--auto-launch`，不会新开另一个模拟器。以下脚本只试飞，不训练，也不生成教师数据：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ubuntu_transfer\falcon_independent_237d_dpu_pool_v1_20261016\windows\test_pool237_missing.ps1 -Python $trainPy -Device cpu -Models 'main18_plain_mlp_h4_w1024_seed0'
if ($LASTEXITCODE -ne 0) { throw '试飞未通过，请保留末尾报告路径。' }
```

需要补全部 `NOT_FOUND` 模型时，将 `-Models '...'` 换为 `-AllMissing`。按模型串行试飞，失败停止，已保存报告保留；大网络可显式选 `-Device cuda`，但不要并行训练。

默认环境48、路线0、5m/s、T50、空中静止起始，不施加会造成位置跳变的状态kick。此入口的PASS是功能/五门几何判定，必须另外查看 `deadline_misses`，不能自动当作严格实时验收。报告写入原模型 `flight` 目录。已有核心四模型不要求为了本包再重飞一遍。补测后将新报告另行交给Ubuntu；冻结包内原审计不会自动更新。

## 5. 237D与4D的精确契约

`input_contract=decision_convex_local_relative_pv_237d_v1`，与之前增量超网的输入、教师和归一化相同。

```text
X[0:4]     当前四元数[w,x,y,z]，规范化、符号对齐、变换后坐标
X[4:7]     当前线速度，变换后Z-up/FLU约定
X[7:217]   21×10个未来参考，按行展平
每个10D    [p_ref-p_current(3), q_ref(4), v_ref-v_current(3)]
X[217:237] 20个参考加速度导出的推力feed-forward
```

不包含当前测量加速度。坐标变换、四元数符号和重力常数9.8066不可随意改；权威编码在 `source_snapshot/fast_convex_preprocess.py` 和 `convex_features.py`。路径规划保持已有全局C2时钟，不能每帧重新启动一条从零加速的路径。

教师是严格凸平移加速度QP＋姿态反馈，不是原论文ACADO的等价实现。预测20步、步长0.1s，21个参考点；预测步长不是采样周期。当前名义路线速度5m/s。

所有网络本体均接受标准化237D，输出标准化4D，**不内置归一化**：

```text
x_norm = float32((x_raw-x_mean)/x_std)
u_physical = float32(network(x_norm)*u_std+u_mean)
U顺序 = [collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]
单位 = [m/s², rad/s, rad/s, rad/s]
```

推力不是PWM、不是牛顿。Windows `control_adapter.py` 负责推力标定以及AirSim参数 `[wx,-wy,-wz]`，板端不要提前翻转一遍。

共同 normalization SHA256：

```text
fe9c44ee80aef1cf97138536a66a4ee3a126f459f2ef963370db354ada8f19ac
```

## 6. FP32模型如何重建与量化

每个模型目录包含：

```text
best_model.pt         从原best导出的精简部署检查点，只有权重和必要结构元数据
weights_fp32.npz      全部FP32张量；给旧Caffe/Vitis环境使用的主要入口
model_spec.json       类型、h/w、Linear顺序、权重key、残差定义、身份和数值误差
golden_vectors.npz    32条共同验证输入及各模型FP32/教师控制输出
provenance/           原配置、完成记录、复杂度、离线指标
```

原best通常带Adam状态，因此本包移除优化器以减小传输量。**精简best不是原文件的逐字节副本，不能继续原训练状态**；源检查点哈希与精简文件哈希分别记录，权重张量逐项核对相同，原文件未改。没有包含TorchScript，因为旧Vitis环境不应依赖现代TorchScript加载能力。

plain结构是 `237→[Linear/ReLU]×h→Linear(4)`；residual为stem-ReLU后执行 `x=ReLU(x+alpha*Linear(x))`，最后线性head，`alpha=1/sqrt(max(h-1,1))`。**这里alpha尚未折叠，与昨天整数超网包不同。** 若导出时折叠到W/b，只折叠一次并校验FP32黄金；不得遗漏skip或把残差当plain。

Ubuntu步骤：

1. 根据NPZ和spec重建FP32 Caffe/当前工具支持的图，先核对黄金。
2. 用本包TRAIN校准子集进行量化；不要用test，不要重新用旧12D数据校准。
3. 用完整随包validation比较各模型FP32/量化输出的标签误差、输出差异、逐通道及尾部误差。
4. 使用真实DPU architecture编译。记录工具版本、编译命令、DPU/CPU子图划分、kernel/node名、tensor尺寸和量化尺度。
5. 上板运行共同黄金，先与该模型的量化参考比较，再与FP32比较。不要只看「有4个输出」就称通过。

FC若转换为1×1卷积或有tensor对齐padding，须明确数据布局、补零和权重映射并验证；外部ABI仍是237D，不能改成12D或25D。每个模型的输入/输出量化尺度可能不同，切换时不能直接复用另一模型的INT8输入字节。

量化不理想时先定位转换、尺度和饱和，不默认启动全套QAT重训。残差不被当前DPU原生支持时，单列CPU混合执行结果或先使用plain主池；不能把混合开销隐去。

## 7. 包含的数据、接收检查与限制

```text
shared/normalization.json
shared/calibration_train_4096.npz   X、U、family_id、原TRAIN行号
shared/calibration_manifest.json   TRAIN哈希、固定抽样种子及行号
shared/validation.npz              原9600条，完整复制
shared/dataset_split_manifest.json
reference/pool237_portable.py      NumPy-only推理
reference/verify_pool237_bundle.py 哈希与FP32黄金检查
model_registry.json
evidence/flight_audit.json
bundle_manifest.json
```

TRAIN原有153600条，仅传4096条校准子集，每个family512条，固定seed237016；原TRAIN不修改。validation9600条来自独立环境48–53。test9600条属于54–59，本次未读取也未打包。

黄金输入来自已有validation，每family4条，共32条。`U_fp32_physical`是该模型的输出，`U_teacher_physical`才是教师标签，两者不能混用。

解压及检查命令，在ZIP所在目录执行：

```bash
sha256sum -c falcon_independent_237d_dpu_pool_v1_20261016.zip.sha256 && \
unzip -n falcon_independent_237d_dpu_pool_v1_20261016.zip && \
cd falcon_independent_237d_dpu_pool_v1_20261016 && \
python3 -B reference/verify_pool237_bundle.py
```

完整性检查仅需Python标准库。已有合适NumPy环境时再运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B reference/verify_pool237_bundle.py --golden
```

无需为此升级旧Vitis AI容器。NumPy参考在独立环境运行即可，不需要PyTorch，也不需要AirSim。检查输出的PASS是FP32导出一致性，不是DPU量化/闭环PASS。

## 8. Ubuntu Agent需要实现的板端基础

先注册实际可用的DPU模型，生成 `dpu_model_registry.json`：原FP32源哈希、量化权重哈希、architecture、kernel和输入输出node、模型类型、归一化身份、量化尺度、设备实测时间及内存占用。

常驻多模型服务在启动时执行一次设备初始化，为所选池的各模型加载kernel、创建task；请求只做模型选择、输入处理、推理和回复，不逐帧销毁重载。先用4个并核实内存，不要求一次常驻全部27个。旧DNNDK的kernel/task创建与`dpuRunTask`分开，接口见 [Vitis AI 1.2说明](https://docs.amd.com/r/1.2-English/ug1414-vitis-ai/Using-Advanced-APIs)。实际1.2.1头文件/库为最终依据。

保留旧服务和bitstream；新服务建议另开空闲端口，例如5007，明确 `family=independent237`、237个float32、模型ID和协议版本。不要静默复用旧25D的FLCN/v1载荷，也不要覆盖超网服务计划使用的端口。DPU与自定义PL镜像切换前保留可恢复方案。

日志至少包含：loop/sample序号、requested/actual model、是否切换、各阶段板端单调时钟、输入/输出、量化与归一化身份、queue/run/API wall/service时间、错误/fallback/drop。主计时用普通运行模式，不在计时区内逐行打印；profiling另外标记。

## 9. 实验A：DPU切换成本

### A1 常驻模型之间的有向切换矩阵

先不启AirSim、不走UDP、单线程、固定同一DPU核/优先级、无竞争。每个目标B比较：

```text
前一个模型A → 运行B
前一个模型B → 再运行B
Delta C(A→B) = C(B | previous=A) - C(B | previous=B)
```

使用匹配输入、交错条件和随机测试块顺序。预热每个模型，冷启动单列。首轮每个有向模型对500次、5轮；可先4个模型，再扩展。主要计时为板端服务和run调用墙钟；后者可能包含驱动等待，不能直接称纯硬件时间。若可无侵入获取硬件执行时间，另列。

拆开：句柄/模型分派、输入归一化和量化、run调用、输出处理、全服务。记录P50/P95/P99、观测最大值、每轮样本数和变异；差值可为零或负，不裁掉。不要将B本身比A慢的时间全部当切换成本。

另加一个相同kernel的两个task交替作为句柄/运行上下文对照；不能为了制造开销每帧重新加载模型。按需加载仅作为不同内存预算下的附加模式，加载时间单独报告。

### A2 模型数量与常驻影响

固定相同A↔B请求序列，只增加其他未调用的常驻模型，测驻留数量本身的影响。另测实际各K池选择策略产生的总开销，并记录其调用模型组成和切换频率；两者不要混为因果证明。

代表性切换对再以实际控制周期节奏执行，而不仅是满速连续调用。固定频率/核心配置、温度/频率状态、内存和后台负载都应记录。实验轮次作为重复单位，不能把全部相关请求当独立重复；最大观测值不是已证明WCET。

产物：`switch_pair_samples.csv`、`switch_cost_matrix.json`、`resident_footprint.json`、分布图和完整运行配置。

## 10. 实验B：池大小与实际控制性能

先定义K=1/2/4，具备足够合格模型后扩至6/8。池应嵌套，K1是验证阶段选定的强固定模型，而不是故意选最差的；K≥2时尽量保留相同快/准端点。冻结选择规则与预算轨迹，各K不能分别手调一套策略或使用在线MPC真标签挑答案。

建议先纯plain主池，残差作为扩展；以实测INT8质量—时延筛选，不以层数假定单调。当前注册表的FP32验证MSE不等于量化后的质量排名。

Windows保留状态与规划器，Ubuntu/板端提供可控model_id或选择接口。首轮三类已验证可控场景：平稳跟踪、转弯＋高度变化、转弯/恢复叠加共享负载变化。名义5m/s，匹配初始状态、路线、运行时长、物理时间窗口和超时策略；每组合先3次探索重复，关键最终结果再增加独立重复。

先固定采样周期以隔离模型选择，动态周期作为单独实验臂。N是并发回路数，K是模型池大小，二者独立；不能一边增加K一边增加N却只归因于K。接近临界负载要根据真实服务时间测得，不预设七八架必然过载。

性能至少包括轨迹时间积分误差、穿门/碰撞、完成时间、实际响应R、输出年龄、切换频率、超期/丢弃/fallback。失败不通过缩短窗口得到更好分数。FALCON性能模型 `J=C(c,T,R,epsilon,theta)` 的思路可以沿用，但本包没有针对新DPU池训练好的SVM；未对齐原误差权重和统计窗口时，不把任意倒数MSE冒充原论文J。

更多候选在理想选择下扩大机会，但真实策略、资源和切换成本可令收益饱和或下降。不强制曲线单调，不删掉反例。

## 11. 实验C：jitter与控制命令跳变

### C1 时间jitter

分别记录R的分布及相邻控制提交间隔的波动。周期主动改变不是额外jitter；同时记录实际采样间隔和计划释放间隔。按目标模型分类，并报告整体混合分布，避免将不同模型确定性的耗时差当成全部不稳定性。

Windows的采样、发送、接收、控制提交使用同一客户端单调时钟；板端只用自身单调时钟求分段耗时。没有时钟同步不能直接用Windows时间戳减板端时间戳。只有命令提交/接收代理时，明确标注不是执行器真实生效时间。

### C2 同输入分歧与实际切换分解

先用同一X在两个模型上执行，测 `fB(X)-fA(X)`。可以用共同验证状态或板端黄金，不要求飞行。报告推力和三角速度的分量、P95/P99，汇总时用共同u_std标准化，不把不同单位原值直接相加。

实际从A切到B时：

```text
fB(x_k)-fA(x_(k-1))
 = [fB(x_k)-fA(x_k)] + [fA(x_k)-fA(x_(k-1))]
       切换分量                  输入变化分量
```

对照的旧模型推理在飞行后回放做，避免在线再运行一次DPU污染时延。上述分解针对原始模型输出；实际裁剪/限速/平滑后的命令要另存，所有方法使用相同处理。

**随包历史T50请求和flight_trace日志不包含完整237D原始输入，不能直接补零重构用于该分解。** 新实验需要Windows逐帧记录完整 `X_raw[237]`、模型ID、raw/applied控制、采样/提交时间、参考/状态和配置版本。随包32条黄金是独立验证状态，也不能冒充连续飞行序列。

实际飞行再比较命令变化率、姿态/轨迹误差和切换事件窗口；转弯本来就会改变命令，不把所有变化都归因于模型切换。控制角速度变化率也不是无人机平移jerk，分别命名和注明单位。

## 12. 图与最终对照

1. 一张双子图：共同横轴K，上图实际控制性能，下图实际切换额外时间P50/P99，注明每秒切换次数。避免不解释的双纵轴。
2. 有向模型对切换成本矩阵及必要延迟分布。
3. 时间jitter和命令跳变的事件波形＋多次事件分布，不只挑一段剧烈波形。

未来与超网区分系统比较与机制比较：DPU INT8池对自定义PL混合精度超网是系统级结果，不能将不同位宽、DDR路径和资源差异全部归因于共享结构。机制比较尽量使用同执行核心、同数值格式、同资源预算；独立池也允许合理蒸馏/一致性，不能故意构造弱基线。

## 13. 执行顺序与每阶段汇报

| 阶段 | Ubuntu工作 | 交付 |
|---|---|---|
| P0 | 接收、核查DPU和模型身份、先选4个候选 | 文件校验、候选清单、真实工具与architecture |
| P1 | FP32重建、TRAIN校准量化、编译、板端黄金 | 量化质量、图划分、kernel、逐模型黄金与独立时延 |
| P2 | 常驻多模型服务、A1/A2切换微基准、同输入分歧 | 成本矩阵、内存、时间/输出差异 |
| P3 | 与Windows协作记录完整X，池大小与切换闭环 | 实际性能、jitter、失败统计、结果图 |
| P4 | 超网硬件就绪后复用协议 | 公平对比，允许无优势或收益有限的结论 |

优先交付P0–P2，不要先安排所有27模型的大规模试飞/全部重训。每阶段都列完整可复制命令、绝对输出路径、真实指标、仍未完成内容。数值错误、OOM或编译失败保留已完成产物，按模型单独恢复，不覆盖旧目录。

本包所有 `DPU_compiled=false` 是打包时事实；Ubuntu工作产生的状态应放新DPU注册表，不修改冻结包和Windows历史报告。若某候选尚无FP32完整飞行证据，可先完成纯硬件微基准，但在真正闭环接管前补齐必要确认。
