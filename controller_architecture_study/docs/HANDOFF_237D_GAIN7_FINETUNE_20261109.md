# 给协作者／Agent：237D 控制器的 gain7 教师迁移与微调

日期：2026-11-09。用户明确委托你完成新教师微调；本交接是与现有 Windows/板端实验**并行的独立研究线**。你有工程源码，另有本目录同名权重数据 zip。请先阅读工程根目录 `AGENTS.md`、`docs/FRAMEWORK.md`、`docs/COLLABORATION_STATUS.md` 和本文；任何旧文档中“暂不微调”的历史状态不覆盖这次用户委托。

## 1. 项目和目标

研究多无人机闭环控制中，如何在共享加速器上根据状态分配控制周期、DNN复杂度及合法调度。DNN模仿优化控制器，当前已在Windows＋AirSim＋Zynq DPU联调；本任务只制作新教师模型，不要求你重做整个调度器、HLS或运行大批量多机实验。

现有模型是**237D 输入、4D 物理控制输出**，不是历史12D/25D模型。输出顺序为 `[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]`，分别为 m/s² 和 rad/s；控制适配与坐标约定以源码为准。`X` 是原始物理特征，不是预先标准化数组；`U` 是物理标签。归一化由训练代码读取 `normalization.json` 后应用。相关源码 `convex_features.py`、`fast_convex_preprocess.py`、`eval_offline.py` 和 `models/`。保持237D特征顺序和单位，不混用12D或25D归一化。

用户希望从旧增益3.5教师迁移到**姿态反馈增益7**。优先先跑一个代表性的板端模型 `plain_mlp_h4_w1024_seed0`；再根据结果及预算扩展到主池其他独立模型。高深度/宽度模型资源消耗大，不要默认一次训练全部28个模型。旧超网作为独立可选迁移路线，不能当作18个独立DNN的另一份权重。

## 2. 教师究竟改哪里

必须对照 `controller_architecture_study/convex_teacher.py`：`QPConfig.attitude_gain` 原为3.5；`ConvexTeacher.query`先解平移凸QP，算期望姿态及当前姿态误差，最后 `rates = clip(attitude_gain * rotation, [-3,-3,-2], [3,3,2])`。新教师**仅将增益设为7**，平移QP、参考、限幅和物理输出契约保持一致，不加入额外QP前馈。它不是把整个QP增益改7，也不是旧12D教师。

不要直接修改原 `convex_teacher.py` 默认值或覆盖原manifest：老实验依赖其源码哈希和原语义。新代码显式使用 `ConvexTeacher(replace(QPConfig(), attitude_gain=7.0))`；可对照 `teacher_attitude_probe.py` 的 `DiagnosticTeacher('gain7')`，而不是 `qp_ff_gain7`。新报告记录教师signature、源码哈希、求解器版本、配置及限幅规则。已有gain7教师对单一路线有飞行通过记录，但不代表DNN微调后一定通过或严格实时。

## 3. 文件和版本

zip的 `controller_architecture_study/results/` 使用原相对路径，包含：

- `decision_convex_recovery_scenic_falcon18_v1/runs/`：18个seed0、600轮独立plain模型，各自`best_model.pt`和可找到的配置/完成信息。`plain_mlp_h4_w1024_seed0`是目前真实板端服务的同源架构，但具体已部署INT8身份仍须按hash及部署manifest核验。
- `decision_convex_residual4_v1/runs/`：4个600轮残差模型；`decision_convex_residual33_long5000_v1/`：一个33隐藏层×1024、追加5000轮的残差best。**这个实际路径是33×1024，不要按对话里过去称呼猜成21×1024。**
- `decision_convex_recovery_v1/runs/features_recovery/`：早期200轮4个seed0模型，单独标识、不要混为600轮主池。
- `incremental_supernet_237d_v4_grid5000/best_model.pt`：一个136配置共享超网FP32检查点，仅作为可选、独立迁移路线；它不是已训练为gain7的模型，也不是可直接使用的板端量化包。
- `decision_convex_recovery_v1/datasets/features_recovery/` 的 `train.npz`、`validation.npz`、`normalization.json` 和split manifest：**旧教师标签**，供构建新标签使用；没有交付test。还包括基础及恢复的来源manifest。Train 153600行，Validation 9600行。

每个文件原字节不变，SHA256和字节数见zip中的`MANIFEST_SHA256.json`；路径与原始文件一一对应。解压前确认剩余空间，建议解压到独立目录，不覆盖协作者自己已有results。原manifest/config内有Windows绝对路径、源码hash、旧教师身份：这些是历史证据，不能拿旧manifest直接声称gain7数据已通过原守卫。应在新代码中建立**跨机器可移植的相对路径、独立来源校验**，不要篡改原manifest或关闭哈希校验。

## 4. 首先处理新标签，而不是直接训旧U

旧权重可用作初始化，但原 `train.npz`/`validation.npz` 的U为gain3.5。用新教师生成独立 `gain7` 的Train和Validation标签，原X、row_id、environment_id、route_id及划分保持配对；验证两者无泄漏。保留旧文件，输出到全新目录；增益7 Validation也必须重标注。不可把同一输入的旧、新U混成没有教师标记的矛盾标签。禁止用Test做训练、调参或选best；最终Test是后续单独一次评估。

一个值得验证的快速方案：对**确实由以上同一gain3.5凸教师、相同QP/四元数和相同限幅生成的原始物理U**，姿态增益从3.5翻倍到7时，可推导：推力不变，三个角速度为 `clip(2 * U_old[1:4], [-3,-3,-2], [3,3,2])`。这是因为对称限幅满足 `clip(2*clip(x,-L,L),-L,L)=clip(2*x,-L,L)`；float32会有舍入差。此法**只在上述前提全部成立时有效**，不能用于不同教师、带前馈、输入/输出已标准化的标签或别的限幅。先核对原基础及增广manifest的教师signature/源码和U单位；然后对可重构的基础与恢复样本，按原seed/路线/采样重建真实`state`及21个references，用 `ConvexTeacher` 的3.5和7分别直接重新求解，在不同family、不同环境、限幅边界及已饱和样本上对比原U和换算U，记录逐通道最大差、统计和样本ID。若旧标签无法重构或任何条件/数值对不上，应停止使用该捷径，转而重生成保存完整state/reference的新教师数据；**不能只拿237D X猜测QP需要的完整reference acceleration**。此处是待你实现并验证的任务，仓库未交付现成gain7转换命令。

建议保留旧 `normalization.json` 原字节作为warm-start基准，包括输出mean/std；检查gain7输出分布、标准化范围、饱和比例。若需更新归一化，要明确转换首/末层或接受输出初始化语义变化，分别比较并记录；不能静默替换。新dataset必须写独立manifest，记录输入数据和归一化hash、每份输出hash、教师和转换方法、split与样本数、严格限幅和重现误差；将U替换后重新核查shape、有限值和范围。源split和source hash跨机可重建验证，不要仅以绝对路径相等判有效。

## 5. 微调方法与避免负迁移

`controller_architecture_study/train_controller.py` 已支持相同架构的 `init_checkpoint`、`restore_initial_optimizer=false`，加载模型后重新初始化Adam；训练配置必须与模型的`input_dim=237`、层数、宽度、model_type相符并strict load。新教师是新任务：新输出目录、新optimizer、新学习率计划、新best记录；不能对旧目录用`--resume`冒充迁移。对**新任务被打断**，才在新配置不变时用`--resume`恢复新任务last checkpoint。验证不读test，显式 `evaluation_splits=['train','validation']`。

先选main18 `h4_w1024`：同一份gain7 Train/Validation上做旧权重初始化微调；建议同预算加入同架构随机初始化对照，检查负迁移。学习率/epoch可基于验证集确定并完整记录，不假设某个固定值必然更好。之后选择少量小/中/大代表，使用相同教师、相同拆分和可解释预算；不同结构不要尝试相互strict load。输出best、last、config、日志、验证指标、各输出/按family误差及P95/P99、源和新权重hash。记录训练环境、GPU、时间，保存中断信息。

现有 `train_controller.py` 接收配置：`python -m controller_architecture_study.train_controller --help` 查参数，训练代码依赖新数据规范；请先阅读实现、做少量smoke test再运行。**本交接未提供未经验证的训练命令**；请由你写新gain7数据工具与配置生成器，并在微型数据上验证重标注、strict初始化和断点续训，再提供实际执行命令及耗时预估。

## 6. 验收与交付给用户

1. 数据／教师身份审计：旧3.5与新7签名、label生成方案及数值对照（限幅样本必须覆盖）、划分和哈希，说明哪些新状态未覆盖。
2. 一个代表架构warm-start训练完成，best按gain7 validation选择；如做随机初始化对照，统一数据与预算，展示各输出和困难族误差及训练曲线。用旧3.5模型作为新标签上的迁移前参考；不能把低MSE等同闭环安全。
3. 独立提供新权重、配置、归一化、训练脚本/命令、manifest和报告；明确总训练轮数、选中epoch，绝不覆盖原模型。
4. 之后Windows同事做统一路线闭环和困难条件验证；新的FP32权重要**重新量化、校准、编译并板测**，不能拿旧gain3.5 DPU xmodel或IP37 tag冒充。量化非本阶段默认已完成。新教师对状态分布可能有偏移，必要时在不泄漏test的前提下采集补充数据。
5. 完成后更新 `docs/COLLABORATION_STATUS.md`，报告产物路径、hash、未完成项；若原文件hash或教师签名不匹配，先隔离并报告，不修改原文件绕过校验。

不要把“gain7更高”当作论文算法创新，也不要为了让某个方法好看而先看测试集调整教师。此任务是提供一个可解释的控制器候选。
