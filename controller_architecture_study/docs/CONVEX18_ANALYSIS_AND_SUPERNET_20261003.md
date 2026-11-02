# 18个237D控制器：结果审计、残差对照与超网研究方案

日期：2026-10-03。范围：本地冻结实验的离线结果，以及下一阶段的研究设计。

**状态：18/18完成600 epoch；本次没有重新训练、没有读取Test样本、没有启动AirSim。残差四模型入口已实现，但尚未训练。下文超网是待验证方案，不是已经实现或已证明最优的系统。**

## 1. 结果与实验合同

原报告：`controller_architecture_study/results/decision_convex_recovery_scenic_falcon18_v1/comparison.json`。

已核对每个模型的完成标记、best/last文件存在性、离线指标与训练日志。完整排序表与图：

- [18模型完整排序表](../results/decision_convex_recovery_scenic_falcon18_v1/analysis_20261003/RESULTS.md)
- [可导入Excel的结果表](../results/decision_convex_recovery_scenic_falcon18_v1/analysis_20261003/all18_ranked.csv)
- [复杂度与gap四联图](../results/decision_convex_recovery_scenic_falcon18_v1/analysis_20261003/capacity_and_gap.png)
- [收敛图](../results/decision_convex_recovery_scenic_falcon18_v1/analysis_20261003/convergence.png)
- [机器可读分析](../results/decision_convex_recovery_scenic_falcon18_v1/analysis_20261003/summary.json)

数据为237D特征、4D控制量，Train153600、Validation9600、Test9600，按环境隔离；Train含原76800条及恢复76800条。所有网络使用相同冻结归一化、seed0、600 epoch、batch512、180000次更新、Adam及相同学习率计划。Test没有用于选择结构或epoch。

这是当前凸MPC教师任务上的SCENIC/FALCON结构对照，不是原论文12D任务的严格复现。这里h表示隐藏FC层数；FALCON depth包含输出FC，因此depth10/22/34对应h9/21/33。参数量不能替代设备实测延迟。

## 2. 不是没有趋势，而是加宽和加深的趋势不同

|控制变量|结构|Validation normalized MSE|
|---|---|---:|
|固定h=3|3×128|8.49010e-4|
||3×256|6.11970e-4|
||3×1024|3.32036e-4|
|固定h=4|4×256|5.40051e-4|
||4×512|4.12428e-4|
||4×1024|2.95795e-4|
|固定w=1024|6×1024|2.85564e-4|
||9×1024|3.11224e-4|
||21×1024|3.35463e-4|
||33×1024|3.71670e-4|

实测结论：

1. 固定h=4，加宽256→1024，Validation下降45.23%；固定h=3，128→1024下降60.89%。之前四模型保持相同深度，当然更容易观察宽度趋势。
2. 当前最低Validation是6×1024。33×1024参数约为其6.16倍，Validation反而高30.15%。但这只说明所测结构、配方、seed和预算下的排序，不是所有大型网络必然失败。
3. 33×1024的混合Train误差也比4×1024高24.58%。这不是仅用“过拟合”就能解释的；深层优化、初始化、结构与正则化适配都应检查。全局梯度范数不能证明逐层梯度消失，现有日志没有足够证据把原因断言成梯度消失。
4. 所有best epoch在569～600；最后100轮的最佳Validation仍改善2.68%～8.85%。完成预算不等于完全收敛。但目前没有证据表明只追加epoch就能逆转深度排序。
5. 4×1024与6×1024的Validation只差约3.46%，却增加约62%的参数。单seed不足以确认这种小优势稳定，后续只对少量候选补配对seed，不必立即再训练整套18×3。

按参数量与Validation得到8个未被其他模型同时超过的候选：2×128、3×128、3×256、4×256、4×512、3×1024、4×1024、6×1024。其余模型保留作结构对照。这不是设备延迟Pareto前沿，也不是闭环飞行优劣结论。

## 3. gap仍在，但需要区分是哪一种

以6×1024为例：

|集合|Normalized MSE|含义|
|---|---:|---|
|混合Train|1.16586e-4|实际训练的153600条，含恢复加权|
|fixed_train|8.30583e-5|原Train76800条，见过的固定状态|
|fresh_train|2.69447e-4|来自Train路线、未参与训练的7680条新状态|
|Validation|2.85564e-4|独立环境的新状态|

Validation/fixed_train≈3.44，不能说过拟合消失；Validation/fresh_train≈1.06，说明新环境相对同路线新状态的额外退化较小。全18个模型的后一个比值为1.001～1.179。两个新状态集合都比见过的状态难，主要剩余问题更像有限状态覆盖与泛化，而不是环境划分本身制造了巨大的额外断层。此为探针证据，不是排除所有分布偏移的证明。

不同集合的状态族权重不同，尤其混合Train并不与Validation同分布，不能只看一个总比值判定训练程序错误。误差应继续按状态族、输出通道、饱和边界、速度及偏差范围分层。

**关键困难：最好模型的combined_recovery（族7）只占Validation的12.5%，却贡献约59.89%的归一化平方误差。** 超网若只改善容易直飞样本的平均MSE，可能对真正需要额外计算的情形无益。下一阶段应检查局部误差、P99及闭环恢复，而非强制每个结构的总体loss单调。

## 4. 立即可运行的残差四模型对照

入口：`controller_architecture_study.run_convex_residual4`。

比较4×512、6×1024、21×1024、33×1024四对结构，分别代表小控制器、当前最好控制器、深层及最深层。复用现有残差实现，保持FC参数量、冻结数据、归一化、600 epoch和seed0一致。新输出在独立的`decision_convex_residual4_v1`，不会覆盖18个plain模型。无需重新生成数据。

当前ResidualMLP为：stem-ReLU → 多个`ReLU(h + α·Linear(h))` → 线性输出，α=1/sqrt(残差层数)。它包含缩放跳连及已有初始化配方，所以这是“残差结构配方”对照，不应宣称严格隔离了跳连一个因素。普通残差并不保证泛化改善。新增跳连加法/缩放不计入FC MACs，实际延迟须另测。

串行子进程训练；每个epoch保存last、Validation改善保存best；中断后重跑同一命令续训，完成模型跳过。GPU爆显存会停止队列，已经保存的模型不依赖后续模型成功。系统突然断电时，最后未落盘的epoch可能需要重做，不能保证硬盘或文件系统故障下的数据完整性。

整段PowerShell命令，**由用户执行**；先关闭AirSim及其他GPU训练任务：

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $proj = 'F:\FalconReproduction\airsim_control'
    $trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
    Set-Location -LiteralPath $proj
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $trainPy)) { throw "找不到训练环境：$trainPy" }
    & $trainPy -m controller_architecture_study.run_convex_residual4 --device cuda
    if ($LASTEXITCODE -ne 0) { throw '队列停止。检查错误后重跑本段，已保存模型和epoch保留。' }
    $reportPath = '.\controller_architecture_study\results\decision_convex_residual4_v1\residual_vs_plain.json'
    $result = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $result.models | Select-Object model,status,epochs,plain_validation_mse,residual_validation_mse,validation_reduction_percent,validation_fresh_ratio | Format-Table -AutoSize
    Write-Host "配对报告：$proj\controller_architecture_study\results\decision_convex_residual4_v1\residual_vs_plain.json"
}
```

本次已运行8项小测试（包含旧队列回归、四结构配置、失败保护、小宽度残差前后向）及真实数据合同dry-run；未执行完整训练。模型单独完成时会更新`comparison.json`；队列结束或异常退出时更新`residual_vs_plain.json`。硬关机后可用`--summarize-only`重建配对报告。

## 5. 超网方向：不要把最大的33×1024直接当答案

目标应是**单位响应时间能获得多少有用的控制改进**，不是网络有多大或档位有多少。以下候选在文献核对前分别提出，再按证据取舍；这是AI辅助设计，不是独立专家投票。

|候选|主要价值|主要反例/代价|取舍|
|---|---|---|---|
|A 多模型常驻＋选择器|实现简单，指针切换很便宜|一次小模型推理后升级大模型通常重复计算|必须保留的强基线|
|B 通用弹性宽度/深度超网|共享权重，多档位|宽度变化常需重算前层；已有大量先例|不作为唯一学术贡献|
|C 渐进残差控制超网＋计算停止/采样联合决策|同帧前缀计算可复用；把额外计算收益与状态变旧代价比较|误差估计不准或决策过重可能无收益|推荐第一阶段研究原型|
|D 可中断展开QP优化网络|结构贴近教师，有机会关联优化残差、可行性和控制性能|需轨迹/优化状态标签及约束分析；不等同现有4维回归|备选，当前不并行展开|

判断准则：闭环收益、真实端到端overhead、状态条件误差是否可校准、前缀是否可复用、可否在真实硬件执行、研究范围是否可控。不用没有根据的“新颖性打分”证明最优。

### 推荐的第一版具体结构（待残差结果确认）

固定237D合同，暂不改输入、不改教师。先选择宽度512，以已有4×512控制器作为基础；上接6个双FC残差修正块，每块带一个4维修正头，产生7个出口。对应隐藏FC计算深度4、6、8、10、12、14、16，而不是强行复制论文全部18个形状。若512宽度精度上限不足，改为1024宽度进行同样验证，不能把512称为已知最优。

```text
237D状态与未来参考信息
          ↓
基础控制器/共享特征 ──→ 出口0：u0
          ↓ 残差修正块1
          ├────────→ 出口1：u0 + Δu1
          ↓ 残差修正块2
          ├────────→ 出口2：u0 + Δu1 + Δu2
          ⋮
          ↓ 残差修正块6
          └────────→ 出口6

各出口 → 当前状态的误差/收益估计 + 已耗时间 + 安全余量
       → 继续计算，或提交一次控制并确定下次采样时刻
```

标准化空间中可写为：

\[
h_j=h_{j-1}+\alpha_j W_{2j}\operatorname{ReLU}(W_{1j}h_{j-1}),\qquad
z_j=z_{j-1}+C_jh_j+c_j,\qquad u_j=\mu_u+\sigma_u\odot z_j.
\]

这里“特征残差”和“控制输出增量”是两个不同机制。修正头零初始化使初始出口复用基础输出；训练后不保证每个状态都越深越准。渐进出口并不是将18个独立模型直接拼接，其权重必须共享训练。

初始训练顺序建议：基础模型 → 冻结基础、逐段学习MPC残差 → 全出口联合小学习率微调。每个出口都对原MPC标签监督，不能只训练最大出口。可增加控制敏感度加权误差及跨出口控制差异正则，但过强的一致性会把所有出口压成同一输出；软单调正则也不是逐样本改进证明。先在Train内部独立校准，再决定门控规则；不能用Test反复选出口或阈值。

### 快速切换的关键不是“换模型指针”，而是复用已算内容

同一帧、相同宽度、相同前缀：出口0判断不够后可从保存的隐藏特征继续算到出口1，而不用重新执行基础模型。权重预先常驻，不在控制循环中load/create/copy模型。隐藏特征缓存只在当前输入帧有效；下一帧输入变了，默认重算，不能无依据跨帧复用。

第一版只做深度弹性，暂不做任意宽度/通道组合。普通宽度裁剪在扩宽时不能精确复用先前所有激活，因为新增通道会改变后续已有通道的输入。若要这种复用，需要块三角连接等结构限制，并评估其精度代价。

GPU上先实测分段调用、门控、同步的成本。如果每小段都CPU/GPU同步，overhead可能超过省下的矩阵乘法。比较整模型、预先选定出口和逐出口判断三种路径，不能只测某个Linear层。

DNNDK/Vitis AI 1.2.1阶段仍需核查目标DPU对分段、Add、共享张量和权重的支持。把每个完整出口分别编译成独立kernel，可能又复制前缀权重，并不自动实现“内存里只有一份超网”。跨段特征写DDR、ARM判断及再次启动必须计入R；必要时调整分段粒度或单独实现PL执行器。本设计当前不是DPU可直接运行的交付物。

## 6. 学术问题：继续精算旧状态，还是尽快执行/获得新状态？

相比“调度器给预算→选择一个网络”，建议把核心问题收紧为：

> 在控制状态正在变化、算力有波动时，下一段计算带来的控制改进，何时足以抵偿额外动作延迟与数据变旧？何时应停止推理并缩短下一次采样周期？

记b为出口，T为采样间隔，R_b为完整响应时间，ε_b(x)为当前状态附近的控制近似误差界/估计，d_k为安全余量。R必须包含采集、预处理、排队、网络、推理、选择、安全检查和下发，而不仅是DPU核耗时。

一个需要推导和验证的约束模板是：

\[
\mathcal D_{\rm delay}(x_k,u_{k-1},T,R_b)
+\mathcal D_{\rm approximation}(x_k,T,R_b,\varepsilon_b)
+\mathcal D_{\rm model}(x_k,T,R_b)\le d_k.
\]

各D项必须统一到安全函数/可达状态误差的量纲，不能把四输出混合MSE直接加到距离上。由这个约束求候选T_max(x,b,R)，而不是只由速度求一个固定deadline。

例如，在参考控制轨迹、局部Lipschitz动力学及有效误差界等假设下，状态偏差可用Gronwall型传播界描述：

\[
\rho(H)\le e^{L_xH}\rho_0+
\frac{e^{L_xH}-1}{L_x}(L_u\varepsilon_b+\bar w).
\]

L_x=0时取连续极限H。该式只说明误差与时间如何共同传播；实际执行需把新命令到达前的旧控制阶段、到达后的新控制阶段分别处理，或采用覆盖两阶段的保守界。参考控制本身也须具备已知安全余量。MSE、P99或一个样本上的误差都不能直接充当整个可达域上的ε_b，更不能据此宣称全局安全。

固定周期、无丢包且延迟有界时，最大信息龄可用T+R_max作保守描述；可变周期应使用相应T_k和下一次响应上界，明确队列/覆盖语义。单线程无重叠推理还需R_b≤T；存在流水线/多机时需另做调度和阻塞分析。延迟P99是统计量，不是WCET。

局部决策可以最小化预测控制风险、计算利用率R_b/T及切换引起的控制跳变，并满足上述余量约束。要点有两个：

- 危险不总等于选最小网络。若小网络在组合恢复状态误差很大，稍深网络可能更合适；反之更高精度若来得太晚也可能更差。
- 静态表中的全局MSE不能告诉系统这一帧是否值得多算。需要廉价的状态条件误差/收益估计，并把它自己的延迟算进去。

可考虑的研究贡献组合（全部为待检验假设）：

1. 针对共享前缀神经控制器，建立出口误差、动作时效、采样周期及切换跳变共同影响控制余量的模型，并给出有条件的可行选择规则。
2. 设计“继续修正/立即提交”的停止策略，复用同帧计算，区别于完全独立模型的事前选择；用可测的下一段边际收益决定是否继续。
3. 在闭环恢复、多机共享算力和延迟波动中验证：同等风险下更低计算/能耗，或同等计算下更少约束违反。参数存储减少是附加结果，不是研究必要性的唯一理由。

如果只有工程上的early-exit、ReLU残差和权重共享，没有控制分析及相对强基线的收益，仍然主要是工程集成。若误差估计/门控过重、所有状态都选同一出口，或者最小网络已经足够快且足够准，应接受该设定下超网不必要的结论，而不是刻意削弱基线。

## 7. 必须面对的强基线与失败条件

18个现有模型合计99,732,168个参数。仅裸权重：FP32约380.45MiB，INT8约95.11MiB；不含激活、运行时workspace、编译布局和量化元数据。因此不能声称这些模型仅权重就必然放不下1GB，也不能由此断言板端实际可用内存一定足够。60个模型需按其真实结构另算。

多模型已经常驻时，选择一个模型的指针可能很快。超网必须证明的是共享表示、同帧升级复用或更好的控制资源分配确有收益，而不是拿常驻超网与每帧从磁盘加载模型比较。

基线至少包含：

- 最好静态模型＋调好的固定T，以及轻量静态4×512；
- 当前8个参数/误差候选常驻＋与超网相同的状态感知选择规则；
- 标准多出口网络＋仅deadline规则；
- 建议方案＋状态条件误差和时效联合规则；
- 在真正需要回答“档位够不够”时补密集模型池，而不是一开始先训练60个。

指标：各出口物理通道误差及P99、困难状态误差、恢复成功率、碰撞/越界率、轨迹误差、控制跳变、完整R的p50/p95/p99/max、miss、最大信息龄、显存/板端内存和能耗。GPU batch=1，报告warmup、同步与冷/热缓存条件，测选择器及切换路径；MACs不能替代延迟。

现有18模型是单seed、固定T训练。某些试飞成功不能代替全部18个已验证。若T变化，还须解决当前教师/参考时钟/动作保持时间与训练合同的一致性；不能只把sleep改短就宣称可任意T。可以先限定几个经验证的T，必要时显式加入T/R输入或重新定义教师标签，并将其标为新的输入合同。

不应在训练集附近的简单直飞里刻意制造“超网胜利”。先确认任务确实存在资源紧张或明显状态依赖：同一设备上真实的多机共享、受控且公开的竞争负载或合理更短截止时间；所有基线遭遇同样的负载。若全部R远小于T且任何出口都足够准，细粒度自适应的收益本来就可能很小。

## 8. 文献边界：哪些不是新贡献

这是截至2026-10-03的探索性定向检索，不是穷尽检索或新颖性证明。检索了公开搜索引擎、arXiv、作者PDF、会议官网/作者机构页面；排除只转述论文的聚合页。全文读取失败的项目只依据摘要，不推测实现细节。

|原始来源|已核对范围|与本方案的关系|
|---|---|---|
|[SUSHI, MLSys2023](https://arxiv.org/abs/2306.17266)|作者摘要；会议PDF读取失败|共享超网子图及硬软协同执行已有先例，不能把共享权重/FPGA切换本身称为首次|
|[TimelyNet, EMSOFT2025作者PDF](https://tanrui.github.io/pub/TimelyNet-EMSOFT.pdf)|摘要、系统/方法及相关实验段落|已经结合动态deadline、超网选择与自动驾驶；“动态T＋超网”不是充分的独创性描述|
|[Once-for-All, ICLR2020](https://arxiv.org/abs/1908.09791)|作者摘要|一个网络支持多种结构和渐进训练已有基础|
|[Deep Residual Learning, CVPR2016](https://arxiv.org/abs/1512.03385)|作者摘要|残差是深层优化的已有工具，不保证当前回归控制任务上必然占优|
|[Towards Anytime Classification…, NeurIPS2023](https://arxiv.org/abs/2306.02652)|作者摘要|更多计算不保证逐样本更准，出口单调性需要专门处理|
|[Fast yet Safe: Early-Exiting with Risk Control, NeurIPS2024](https://arxiv.org/abs/2405.20915)|作者摘要|风险控制早退已有研究；其预测风险结论不能直接等同连续闭环安全|
|[Sequence-based Anytime Control, TAC2013](https://arxiv.org/abs/1308.1747)|作者摘要及期刊元数据|随计算资源变化的控制及稳定性分析并非新问题|
|[Anytime Computation and Control…, TCST2021](https://experts.nau.edu/en/publications/anytime-computation-and-control-for-autonomous-systems/)|作者机构摘要及元数据|估计精度/时延和鲁棒控制联合设计也已有先例，不能声称首次进行计算—控制协同设计|
|[Approximate nonlinear MPC with safety-augmented neural networks](https://arxiv.org/abs/2304.09575)|作者摘要|安全增强神经MPC已有相关工作；当前只输出4维单步控制，不自动拥有其控制序列验证机制|

主要检索词记录：`SUSHI FPGA supernet switching dynamic inference paper`；`TimelyNet neural network anytime control paper`；`anytime neural control early exit residual network model predictive control adaptive computation delay`；`neural anytime control early exit`；`approximate MPC neural network safety verification Lipschitz`；`control early-exit deadline safety neural`；`anytime control algorithms computational resources sequence based control Quevedo Gupta 2013`；`control inference accuracy age of information early exit neural network control`；`anytime model predictive control computation time control performance neural early exit`。

投稿前还需针对最终停止准则/稳定性条件做更窄的对比检索。这里没有认定该组合从未出现，也没有把所引用分类/感知论文的保证搬到本控制任务上。

## 9. 阶段决策

1. **已完成**：冻结18模型、整理图表和误差分解。保留完整结果，不能只展示单调子集冒充全体结果。
2. **用户运行**：四个配对残差训练。若深模型明显改善，再对少量关键对照补seed；若无收益，不继续盲目加深。
3. **待实现**：固定宽度的7出口渐进控制原型，先检查每出口控制误差、困难状态及GPU端到端代价，暂不部署FPGA。
4. **有证据再推进**：加入状态条件误差校准和T/出口联合规则，先在模拟安全层下验证；证明条件和工程统计界分开报告。
5. **最后适配**：DPU/PL分段与权重共享。若硬件分段overhead不合算，调整粒度或接受常驻独立模型更优；不要用不真实的模型加载成本制造优势。

验收目标不是“所有大网络必须优于小网络”，而是得到可重复的精度—真实响应时间折中，并证明自适应选择这个折中能改善闭环控制。
