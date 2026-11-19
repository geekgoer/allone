# M2 v2 量化结果、相关研究与下一步判断

日期：2026-10-14，Asia/Shanghai。
范围：针对本工程开展的小范围技术文献检索与代码/结果核对，非穷尽式系统综述。
本轮没有修改训练代码、启动训练、改变门槛或执行 FPGA 工程。

## 1. 本地证据

读取目录：`controller_architecture_study/results/incremental_supernet_237d_m2_mixed_v2`。
300 轮完成；验证选择 epoch 290，SHA256：
`dafbf8353fc1533c0f01ff62fce4bb05000363632193ee504231d1eabc77ed30`。
原 FP32 为 v4 epoch 4983，所有 136 个配置及原始数据拆分保留。

| 标准化验证指标 | FP32 | v2 真整数 | 变化及门槛 |
| --- | ---: | ---: | --- |
| 最大配置 MSE（不是最坏配置/最大逐样本误差） | 0.000176528588669 | 0.000196250855156 | +11.1723%；要求 ≤+10% |
| 136 配置平均 MSE | 0.001551979006706 | 0.001538347568891 | -0.8783%；通过 |
| 固定可信掩码下同输入相邻配置差异 MSE | 0.0000499077659044 | 0.0000513414797036 | +2.8727%；要求不增加 |
| 相对 FP32 的额外配置差异 MSE | 0 | 0.0000012821894740 | 此指标是量化引入的额外差异，不是飞行时序震荡 |

最大配置允许值为 0.000194181447536；当前值比这个允许值高 1.0657%。
这两个未通过条件是本工程之前设定的离线工程目标，不是 FALCON 公式要求、FPGA 可综合性条件或稳定性定理。
不能因为接近门槛就自动放宽，也不能因为未通过门槛就推断板上不可实现。

QAT 并非无效：

- v2 未微调的最大配置 MSE 为 0.000450768821562，QAT 后下降约 56.46%。
- 未微调的额外配置差异 MSE 为 0.0000056294615825，QAT 后下降约 77.22%。
- 300 轮历史中没有任何一轮满足所有原门槛；最佳最大配置也是 epoch 290。
- epoch 300 最大配置 MSE 为 0.000206388252554，不能认为最后一轮一定最好。

真整数执行完成 9600 行、136 配置。导出 golden 32 行的 fake/integer 最大物理输出差为
0.000135437862；已检查 2 行、2 条续算顺序，直接与增量计算逐位一致、缓存前缀未变。
这支持算术契约和计算复用的正确性，不证明 FPGA 实测或闭环安全；导出仍明确 FPGA_verified=false。

权重裁剪 348/19186688，即约 0.001814%。数量很少，不能单凭此数量认定没有影响；
也没有证据将它认定为当前全部退化的主因。激活裁剪为 0。

## 2. 当前方案为何比普通量化案例更难

已核对 `qat_model.py` 的 `exponent_for`：

```
s = 2 ** ceil(log2(max_abs * margin / integer_limit))
```

W8 骨干逐输出通道，W16 输出头逐输出通道，A16 逐层；尺度初始化一次后冻结。
这属于固定尺度的 fake-quant QAT，并未实现 LSQ 可学习步长，也未实现 AdaRound。
纯幂次尺度便于移位，但限制了可选择的量化网格。
33 层残差与多个输出头共同约束，使局部误差可能传递到后续头；哪个层贡献最大仍需敏感性证据。
分类论文的 top-1 几乎不变，不等价于控制回归模型的极低 MSE 几乎不变。

另外，原“同输入切换差绝不增加”是很严格的要求。设量化后配置差异为 Δq=Δf+ε，
则均方差变化为：

```
E[Δq²] - E[Δf²] = 2E[Δf ε] + E[ε²]
```

即使量化扰动很小、均值为零，当交叉项不能抵消时，均方差仍可能增加。
因此完全不增加不能当成任意有限精度模型天然具备的性质。
它可以保留为严格研究目标；实际允许误差最终应依据控制性能/约束确定，不能事后为获得 PASS 随意改阈值。

## 3. 文献范围与筛选

问题：对于共享残差控制超网，哪些量化优化能降低输出误差，并可转成确定性的纯整数推理？
比较：固定尺度 STE QAT；指标：控制回归 MSE、跨配置额外差异、整数数值一致性和最终硬件代价。

检索平台：arXiv、OpenReview、PMLR、CVF 官方论文页；IBM 作者机构页用于交叉核对 LSQ。
时间范围：以 2018–2026 年相关方法为主；英文技术研究；只以作者/会议原始来源支撑方法结论。
纳入：量化尺度学习、舍入/块重建、硬件混合精度、超网与量化关系、FPGA 量化实现。
排除：二手媒体作为证据；将分类结果直接声称为本控制模型保证；没有读到全文时声称已完整复现方法。
预印本页面与正式会议版本按标题/arXiv ID 去重，以下表格每项仅计一个研究。

## 4. 方法及适用性

| 研究 | 证据范围 | 核心做法 | 对本工程的用途与限制 |
| --- | --- | --- | --- |
| LSQ，Esser 等，ICLR 2020 | arXiv 正式会议版全文方法段、作者机构页 | 联合学习权重/激活步长，并缩放步长梯度 | 当前冻结尺度的直接改进候选。原实验是 ImageNet 分类，不能承诺本 MSE 门槛必过 |
| PACT，Choi 等，2018 | 作者 arXiv 摘要 | 学习激活截断阈值 | 可学习激活范围，但当前 A16 分解损失较小，优先级低于权重尺度 |
| AdaRound，Nagel 等，ICML 2020 | PMLR 摘要及论文方法简介 | 根据数据/损失优化向上或向下舍入 | 可在固定尺度下优化整数码，减少架构接口变化；属于 PTQ，并不是 QAT 算法 |
| BRECQ，Li 等，ICLR 2021 | arXiv 正式会议版，§3.2–3.3 | 残差块输出重建、敏感性、混合精度 | 可为本因果残差结构增加块输出对齐；需保留因果 mask 和所有组的共享编码；属于 PTQ 研究 |
| HAWQ-V3，Yao 等，ICML 2021 | PMLR 页、全文 §3.1–3.4 | 硬件感知混合精度；M/2^n 尺度转换；残差纯整数执行 | 整数实现不必只用 2^e。其部署是 TVM/T4，不是本 AXU3EGB 验证 |
| APQ，Wang 等，CVPR 2020 | CVF 官方摘要与引言 | 联合架构、剪枝、量化策略；预训练超网用于搜索 | 表明架构与量化相互影响；不是任意弹性配置可直接共享整数缓存的证明 |
| FINN-R，Blott 等，2018 | 作者 arXiv 摘要 | 针对平台和精度定制 FPGA 推理引擎 | 支持 FPGA 可以使用定制精度的方向；不是本网络性能/资源满足的证明 |
| Model-Preserving Adaptive Rounding，Tseng 等，ICML 2026 | PMLR 官方摘要，未进一步全文核查 | YAQA 关注端到端输出误差与自适应舍入 | 近期候选，当前不直接据此实现或声称控制模型收益 |

## 5. 建议下一版本

优先方向是共享逐通道可学习权重步长（借鉴 LSQ），保留 A16 和 W16 输出头，
继续逐配置蒸馏及相邻配置差异保持。若需要改善深层累积误差，再借鉴 BRECQ 的残差块输出重建。
这些是方法迁移建议，不是本地已完成的实现或性能结果。

部署时所有尺度必须冻结并写入模型版本。采用 M/2^n 的整数尺度转换时，
应在后半段 QAT 中模拟最终乘数/移位/舍入，而不是训练完才未经微调投影为新尺度。
需要重新检查“累加器×整数乘数”的中间位宽；原 48 位 MAC 安全并不自动保证乘数之后也安全。

为了保留精确续算：

1. 同一层/通道组在各深度、各宽度使用相同参数和最终整数编码。
2. 保留严格因果 mask；后续组不改变前缀特征。
3. 不为不同子网配置单独在线校准，也不在运行中更换尺度。
4. 与整数包版本绑定缓存；head 组贡献先按统一数值单位累计，舍入位置一致。

如果需要更稳妥的阶段成果，可先使用更高精度的共享整数网络或仅提高敏感层精度，
再按资源测量降低位宽。这不会取消深度/宽度增量机制，但时延、DSP、带宽必须实际综合/测量。
应避免立即展开 136 套独立量化器或全组合混合位宽搜索。

严格质量门槛保持记录；硬件算术开发可使用当前包对齐数值，同时明确质量仍未验收。
实际控制闭环、量化误差容限及超网相较模型池的优势应在后续集中实验验证。
这轮结果不构成放弃 FPGA 或增量超网的证据。

## 6. 核对过的来源

- LSQ：https://arxiv.org/abs/1902.08153；全文 https://arxiv.org/pdf/1902.08153；作者机构 https://research.ibm.com/publications/learned-step-size-quantization
- PACT：https://arxiv.org/abs/1805.06085（2018 预印本，不将其写成已核验的会议接收版本）。
- AdaRound：https://proceedings.mlr.press/v119/nagel20a.html
- BRECQ：https://arxiv.org/abs/2102.05426；全文 https://arxiv.org/pdf/2102.05426
- HAWQ-V3：https://proceedings.mlr.press/v139/yao21a.html；全文 https://proceedings.mlr.press/v139/yao21a/yao21a.pdf
- APQ：https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.html
- FINN-R：https://arxiv.org/abs/1809.04570
- Model-Preserving Adaptive Rounding：https://proceedings.mlr.press/v306/tseng26b.html

## 7. 检索日志

日期均为 2026-10-14。实际查询串：

| 批次 | 查询 | 最终使用的原始来源 |
| --- | --- | --- |
| 1 | `Learned Step Size Quantization ICLR 2020 paper Esser` | IBM、arXiv |
| 1 | `PACT Parameterized Clipping Activation Quantized Neural Networks arxiv 2018` | arXiv |
| 1 | `HAWQ V3 Dyadic Neural Network Quantization integer-only mixed precision ICML 2021` | PMLR |
| 1 | `BRECQ Pushing the Limit Post Training Quantization Block Reconstruction ICLR 2021 AdaRound` | arXiv |
| 2 | `site.proceedings.mlr.press Up or Down Adaptive Rounding Post Training Quantization Nagel 2020` | PMLR |
| 2 | `site.openaccess.thecvf.com APQ Joint Search Network Architecture Pruning Quantization Policy 2020` | CVF |
| 2 | `FINN-R end to end deep learning framework exploration quantized neural networks FPGA arxiv` | arXiv |

OpenReview LSQ/BRECQ 页和 PDF 触发验证页面，未将其当作已成功读取的正文；
改用作者 arXiv 上传的正式会议版 PDF。2026 YAQA 为查询发现的近期候选，仅读了官方摘要。
没有把搜索结果中的 ResearchGate、博客、第三方摘要站作为方法论证来源。
