# 动态闭环 FPGA/超网调度：检索与构思记录

日期：2026-10-05。类型：探索性范围综述，不宣称穷尽检索或已经证明新颖性。

## 问题和范围

多条无人机控制回路共享计算资源时，如何联合决定采样周期、控制网络计算量和执行顺序，使增加的计算在控制指令失效前产生值得等待的收益？比较固定网络/周期、EDF/固定优先级、状态感知独立模型池、早退网络和弹性共享网络。重点指标为闭环任务误差、动态有效期违例、完整响应时间、计算浪费和实现开销。

检索源：作者/实验室公开页面、会议/出版社原始论文、arXiv；Crossref或其他学术索引用于身份核对。时间以2020年至2026-10-05为主，向前追溯必要的实时调度、反馈调度、自触发控制和imprecise computation基础工作。只以原始来源支持技术结论；明确区分全文、摘要、预印本。非相关同名论文、二手摘要、搜索片段不作为机制证据。

计划检索词：SCENIC capability scheduling co-design intelligent controller；FALCON FPGA accelerated real-time intelligent controller；RICH heterogeneous computing real-time intelligent control；SUSHI supernet FPGA scheduling；TimelyNet dynamic deadline；anytime control imprecise computation scheduling；state-aware control scheduling value of information；multi-DNN FPGA preemption scheduling；supernet inference scheduling dynamic deadlines；control-aware age of information scheduling。

## 初轮构思（检索前冻结，均为待验证想法）

记录者：本对话AI助手。来源：用户已有动态T/超网构想、当前工程测量，以及本轮推理。未组织独立人类评审；下文自我反驳不算独立验证。用户已有SUSHI、TimelyNet等讨论构成先验锚定，须在检索后重新评价。

| ID | 初始想法 | 关键假设 | 可反驳结果 |
|---|---|---|---|
| I1 | 按控制有效期和额外计算的边际闭环价值分配执行顺序/计算量 | 相同额外毫秒在不同状态有不同收益，且能够以足够小开销估计 | EDF加离线子网选择已经达到相同收益 |
| I2 | 将端到端状态年龄纳入调度，而非只处理DNN核执行时间 | 感知、CPU参考生成、通信、排队对预算有实质影响 | 实测这些阶段几乎恒定、可以用固定裕量覆盖 |
| I3 | 把同一采样的推理组织成可续算的有效控制结果和增量修正，在预算变化时保留已完成工作 | 有中间控制头、合法续算路径和可测的上下文成本 | 独立模型切换/普通早退足够；续算频繁失效或被额外访存抵消 |
| I4 | 新状态到来时比较继续旧推理与重启新推理，联合处理过期任务和下一次采样 | 新鲜度损失与精度收益经常出现竞争，存在可测的中间态 | 控制周期长于全部最坏响应，没有有用的跨时刻竞争 |
| I5 | 以可解释的可行性保护和轻量价值选择构成调度，而不是直接先选RL | 约束可近似计算，策略开销小于节省的预算 | 状态/硬件相互作用过于复杂，简单策略稳定劣于学习策略 |

评价准则在看检索结果前固定：问题是否被现有数据支持；相对最接近工作是否存在明确差异；AXU3EGB/现有DPU上的可实现性；能否被公平基线反驳；首轮验证成本。采用定性理由与不确定性，不构造虚假的专家分数或新颖性概率。

## 已有工程证据

三机在线模型/T原型最近一次报告：`results/dynamic_complexity_237d_v1/airsim_joint_online_20261004T173435_107267Z/report.json`。1172次控制下发、15次有效穿门、0截止时间违例；模型和周期各切换4次。该场景门附近估计T_max最小约467ms，远高于50/75/100ms候选。它支持功能可行性，不支持动态约束/超网必要性。237D网络在GPU上的结果不能替代旧25D INT8网络在DPU上的时延证据。

## 检索记录

以下为主题级检索日志，不是逐条数据库命中数/PRISMA流程；没有据此声称穷尽检索。搜索引擎和作者主页在本次访问时可能展示更晚会议的accepted条目，仅纳入已能核实、截至本次日期相关的论文。检索日期均为2026-10-05。

| 轮次/主题 | 代表查询 | 原始来源和阅读范围 | 对构思的影响 |
|---|---|---|---|
| 1 核实三篇起点 | SCENIC capability scheduling co-design; FALCON FPGA real-time controller; RICH heterogeneous real-time intelligent control | 本地原始PDF的英文提取文本：SCENIC重点§V；FALCON重点执行模型、优化问题和算法；RICH重点§IV–VI。出版社DOI/论文首页核对身份。不是采用旁边的中文AI摘要 | FALCON有固定优先级和联合搜索；SCENIC的DQN不能描述成逐帧在线调度；RICH按剩余加速**段数**而非层数排优先级 |
| 2 超网和动态预算 | SUSHI supernet FPGA scheduling; TimelyNet dynamic deadline | SUSHI arXiv全文重点§2–4；TimelyNet作者版与本地TECS正式版重点§4–5，正式版首页DOI | 单独“动态deadline选子网”“缓存共享权重”已经有直接先例 |
| 3 物理状态和计算价值 | state-aware control scheduling value; Performance Optimization Stability Multi-tier; integrated real-time safety control scheduling | RTSS2024 TSUF/VCS原始PDF重点§IV–V；RTAS2025作者PDF和本地原始PDF，定义4、定理2、§IV；已渲染并核对第6页图3和定理2 | 物理状态感知、收益密度、稳定性任务优先，以及安全周期上界均已有先例；须收窄创新主张 |
| 4 分段神经推理 | anytime imprecise computation scheduling; real-time mixed deep learning; Pantheon real-time GPU | Yao等2020 arXiv全文§II-B–E；RT-mDL作者机构摘要；Pantheon作者项目页；DART作者PDF作为背景检索 | “基础输出+可选层+边际收益”也不是新概念；EDF与可选计算必须是强基线 |
| 5 硬件细粒度 | FPGA multi-DNN preemption; IsoSched subgraph isomorphism | IsoSched arXiv预印本§III-C、§IV-A；TCAD2025 Early-Exit FPGA作者机构页面摘要，PDF抓取失败，不将未读正文视为证据 | 区分硬件仿真与上板；早退头的开销、上下文和搬运不能省略；未发现现有DNNDK接口足以支持任意中途抢占的证据 |
| 6 新鲜度与控制价值 | age anytime inference scheduling; value information delay control | VoI/AoI相关arXiv摘要用于背景筛选；Timely ML 2025出版社正文§1–2 | 不能把“关心状态年龄”本身当成创新；全文未细读的VoI论文不支撑具体算法结论 |
| 7 近期反例追索 | control-aware inference scheduling 2025 2026; FPGA early exit control safety | ICCPS2026 Multi-Resolution E2E作者论文§I–IV、作者发表清单；RTSS2020 Algorithmic Priority Inversion作者机构摘要；AIAA2024 Anytime Perception and Control作者来源仅摘要/引言可得；ICRA2024/EMSOFT2025 anytime sensing verification仅核实作者/会议条目，未获得正文 | “单网络动态精度—延迟—安全”及重要输入优先均非空白。形式化anytime sensing安全是下一阶段必须补读的邻近方向，当前不声称首次 |

## 证据目录与原始链接

| ID | 论文/身份 | 可核实来源 | 本轮使用范围 |
|---|---|---|---|
| R1 | SCENIC, RTSS2024 | https://doi.org/10.1109/RTSS62706.2024.00026 | 离线DQN联合配置、全局固定优先级、CPU/加速段模型 |
| R2 | FALCON, RTSS2025 | https://doi.org/10.1109/RTSS66672.2025.00040 | 硬件/复杂度/T/优先级联合优化、R≤D≤T、分段和通信建模边界 |
| R3 | RICH, DATE2025 | https://doi.org/10.23919/DATE64628.2025.10993261 | CCRM离线映射、MRAF动态段级调度 |
| R4 | TimelyNet, EMSOFT/TECS2025 | https://doi.org/10.1145/3762652 ; https://tanrui.github.io/pub/TimelyNet-EMSOFT.pdf | 动态延迟预算下预选子网；不能视为多控制回路的在线执行队列策略 |
| R5 | SUSHI, MLSys2023 | https://arxiv.org/abs/2306.17266 | 查询约束、缓存状态、共享权重驻留；不是跨输入复用激活 |
| R6 | Performance Optimization and Stability Guarantees for Multi-tier Real-Time Control Systems, RTSS2024 | https://doi.org/10.1109/RTSS62706.2024.00025 ; https://digitalcommons.njit.edu/fac_pubs/726/ | TSUF/VCS，稳定任务优先与性能任务价值密度 |
| R7 | Integrated Real-Time Control and Scheduling for Safety Critical Cyber-Physical Systems, RTAS2025 | https://sudvarg.com/publications/RTAS2025_Integrated_Control_and_Scheduling.pdf | 安全delay interval、周期约束和适用前提；第6页定理2视觉核对 |
| R8 | Scheduling Real-time Deep Learning Services as Imprecise Computations, 2020公开稿 | https://arxiv.org/abs/2011.01112 | mandatory/optional、阶段非抢占、EDF基础上的深度优化及输入依赖效用 |
| R9 | RT-mDL, SenSys2021 | https://doi.org/10.1145/3485730.3485938 | 仅机构摘要层面的模型缩放和实时资源联合管理 |
| R10 | Pantheon, MobiSys2024 | https://pantheoninfer.github.io/ | 作者项目介绍层面的GPU细粒度抢占与实时推理 |
| R11 | IsoSched, arXiv2025预印本 | https://arxiv.org/abs/2509.12208 | tile/subgraph抢占和上下文搬运；Verilog综合+周期精确仿真，不是AXU3EGB实测 |
| R12 | Realization of Early-Exit Dynamic Neural Networks on Reconfigurable Hardware, TCAD2025 | https://eprints.soton.ac.uk/499024/ ; https://doi.org/10.1109/TCAD.2024.3519055 | 机构摘要：流水/并行中间分类器、ZCU106验证；正文获取受限 |
| R13 | Timely ML, Real-Time Systems2025，观点文章 | https://doi.org/10.1007/s11241-025-09449-5 | contract/anytime、时间约束与数据年龄的区分；不当作新算法实验结果 |
| R14 | Multi-Resolution End-to-End Deep Neural Network for Optimizing Latency-Accuracy Tradeoff in Autonomous Driving, ICCPS2026 | https://www.ittc.ku.edu/~heechul/papers/multires-e2e-iccps2026-camera-v2.pdf | 单模型多分辨率、CARLA精度/时延/安全权衡；不是237D多回路FPGA队列调度 |
| R15 | On Removing Algorithmic Priority Inversion from Mission-critical Machine Inference Pipelines, RTSS2020 | https://doi.org/10.1109/RTSS49844.2020.00037 ; https://experts.illinois.edu/en/publications/on-removing-algorithmic-priority-inversion-from-mission-critical-/ | 仅原始机构摘要：物理重要输入不应被低重要输入阻塞 |
| R16 | Anytime Perception and Control for Safe and Intelligent Urban Air Mobility, AIAA SciTech2024 | https://doi.org/10.2514/6.2024-2009 ; https://ittc.ku.edu/~heechul/papers/2024-01-Anytime-SciTech-paper.pdf | 摘要/引言背景；网络工具因文件体积未取得全文 |
| R17 | Safety Verification of Closed-loop Control System with Anytime Perception, ICRA2024 | https://doi.org/10.1109/ICRA57147.2024.10611044 | 作者发表页核实存在；未读全文，不据此判定其具体定理覆盖范围 |
| R18 | Star-Set Based Efficient Reachable Set Computation of Anytime Sensing-Based Neural Network-Controlled Dynamical Systems, EMSOFT/TECS2025 | https://doi.org/10.1145/3762658 ; https://github.com/ACM-SIGBED/emsoft/blob/main/articles/emsoft2025.md | 作者/官方会议信息核实，正文未读；列为形式化方向的后续必读 |

主要本地英文原文：`E:/BaiduNetdiskDownload/vitis-ai6.1/tmp/pdfs/extracted/paper_03.txt`（R6）、`paper_05.txt`（R3）、`paper_08.txt`（R1）、`paper_09.txt`（R4）、`falcon.txt`（R2）。对应PDF在 `C:/Users/chen/Documents/论文/无人机` 或FALCON工程原始PDF目录。提取文本用于检索，不将OCR公式错误直接当公式；R7关键时间式另做了视觉核对。

## 相反证据、筛选与未决项

- 排除Scribd研究建议文档、自动聚合“研究故事”、同名FALCON/SCENIC非控制论文；它们不作为技术证据。
- 没有依据抽象中的“state-aware”一词就认定SUSHI感知无人机物理状态；正文指缓存/硬件状态。
- 没有因为搜索未找到完全相同标题就宣布新颖。R6+R8+R7是最强的组合式反例，任何论文定位必须回答为何不能直接组合它们。
- R13是观点文，R11是预印本且采用仿真评估，R9/R12/R15只有机构摘要级使用，R17/R18尚未读全文；证据层级不混用。
- 本轮没有对所有2026会议录执行数据库穷尽检索，也没有专利或严格查新；候选贡献仍需导师和领域专家审阅。

## 检索后决策与自我反驳

| 初始想法 | 处理 | 原因与剩余问题 |
|---|---|---|
| I1 状态价值调度 | 收窄、保留 | TSUF/VCS已有状态相关控制价值；应研究随推理进度变化、按实际交付年龄计价、同时影响下一采样需求的增量控制价值 |
| I2 端到端年龄 | 作为必要测量，非单独创新 | 实时流水线/AoI已有；本工程必须测，但新颖性需来自与控制质量及任务再释放的耦合 |
| I3 可续算共享网络 | 条件保留 | 早退和imprecise computation已有；必须击败相同策略的独立模型池及普通早退，且计入状态保存/访存成本 |
| I4 继续旧样本还是重启 | 作为第二阶段候选 | 信息价值和anytime控制已有基础；在单活跃请求且无紧预算情况下可能根本不触发，先证明现象 |
| I5 可行性保护+轻量策略 | 保留为首个实现方向 | 容易解释、测量和反驳；不是已有实时理论的新命名。正式安全证明需要额外模型/误差界/执行上界 |

优先候选：质量—年龄可行域驱动的增量控制计算调度。备选：独立模型池的状态感知EDF联合选型。暂缓：任意宽度实时扩展、整套RL调度、FPGA部分重配置。排序是基于当前资源和证据的判断，不是打分证明。

最强反驳：如果状态感知EDF+独立模型池已达到相同闭环性能，或普通早退已达到相同收益，就没有足够理由声称更复杂弹性超网是必要的。若控制价值预测/分段开销吃掉收益，应缩小方法而非提高人工负载掩盖问题。状态安全、稳定性与平均跟踪性能是不同命题。

本轮仅形成研究分析和文档；未训练、未试飞、未改动调度或控制代码。模拟实验不等于真机安全许可；实际无人机部署前需要独立安全评估。
