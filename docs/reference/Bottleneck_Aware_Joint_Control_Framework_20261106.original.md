# Bottleneck-Aware Joint Rate–Complexity–Scheduling Framework

> **研究框架草案（2026-11-06）**  
> 面向多 UAV 共享 FPGA 的实时神经控制：用可解释的时间瓶颈与系统干扰关系，引导低开销的联合配置调整。  
> **本文不是已有算法的正式证明或最终定稿**；带有 *假设 / 待验证* 标注的部分，应通过实验与分析决定是否保留。

[TOC]

---

## 1. 一句话概括

**不是使用一个大型黑盒，直接从状态预测最优的 $(T,c,\sigma)$，而是：从物理安全约束中识别时间违例的成因，利用少量可解释的规则提出针对性配置，再用保守的全局可行性检查决定是否执行。**

核心链路：

$$
\boxed{
\text{Physical State}
\to \text{Dynamic Timing Requirement}
\to \text{Bottleneck Diagnosis}
\to \text{Bounded Candidate Generation}
\to \text{Global Feasibility Check}
\to \text{Commit}
}
$$

三个控制变量仍然是：

$$
\boxed{T_i\quad c_i\quad \sigma}
$$

但**不固定为“先调整 $T,c$，后调整 $\sigma$”**。调整动作由当前瓶颈决定；$\sigma$ 是与 $T,c$ 同样重要的调整维度。

---

## 2. 问题与 Motivation

### 2.1 FALCON 给出的基础

FALCON 已经将 DNN complexity、控制周期、响应时间、环境与 FPGA 资源等因素联合纳入控制性能优化。它并不是“不考虑 $R$”或“没有实时性约束”。

当前工作尝试补充的是：**轨迹级或聚合 QoC 不等价于关键时刻的逐点物理安全**。一个全程表现不错的配置，仍可能在门框附近的短时间内遭遇无法及时纠正的偏差。

因此，我们显式建立当前物理状态到下一次有效反馈的时间要求：

$$
\boxed{B_{i,k}=\mathcal B_i(x_{i,k},u_{i,k}^{actual},\Omega_i)}
$$

其中 $\Omega_i$ 是动力学、扰动和状态估计等假设。$B$ 可以是经验性保守时间预算；只有在其上界性质被论证时，才可用于严格的物理安全保证。

### 2.2 为什么动态调整并不自动有效

当 $B_i$ 变小，直觉上可以缩短周期、减小 DNN 或改变优先级。但三种动作各有副作用：

- 缩短 $T_i$，可能使该 UAV 更快更新，却提高共享加速器的未来到达负载；
- 减小 $c_i$，可能节约执行时间，但控制器可能不再具备足够的闭环纠偏能力；
- 改变 $\sigma$，可以重分配等待时间，却不创造计算容量，且可能危及其他 UAV。

因此真正需要解决的是：

> **在多 UAV 的动态安全要求下，如何识别是哪一类时间瓶颈造成违例，快速找到有效调整方向，并保证调整不破坏其他控制回路？**

### 2.3 一个直观例子

假设某 UAV 的简化时间预算为：

$$
B=50\,\mathrm{ms},\quad T=20\,\mathrm{ms},\quad C=8\,\mathrm{ms},\quad W=28\,\mathrm{ms},\quad O=2\,\mathrm{ms}.
$$

当前：

$$
T+C+W+O=58\,\mathrm{ms}>B.
$$

如果把 DNN 从 $8$ ms 降到 $5$ ms，仍有 $55$ ms，修复不了；如果只缩短 $T$ 到 $10$ ms，虽然时间上可能赶得上，却明显提高后续任务到达频率；但若通过合法调度将 $W$ 从 $28$ ms 降为 $18$ ms，就得到 $48$ ms，无须牺牲该 UAV 的模型质量或增大其更新频率。

**这里不是说“一定先调调度”——只有其他 UAV 的截止时间也能满足时，这次调度调整才有效。**

---

## 3. 统一的时间语义和物理约束

### 3.1 三个关键时刻

定义：

$$
\begin{aligned}
r_{i,k}&=\text{control job }k\text{ 的 release time},\\
a_{i,k}&=\text{对应控制命令真正作用于 plant 的时刻},\\
R_{i,k}&=a_{i,k}-r_{i,k},\\
T_{i,k}&=r_{i,k+1}-r_{i,k}.
\end{aligned}
$$

因此相邻两次有效控制的**精确间隔**为：

$$
\boxed{a_{i,k+1}-a_{i,k}=T_{i,k}+R_{i,k+1}-R_{i,k}}
$$

若在 $a_{i,k}$ 时刻，以当时一致的状态 $x_{i,k}$ 和实际生效控制 $u_{i,k}^{actual}$ 计算预算，则下一次有效控制的绝对截止时刻为：

$$
\boxed{d_{i,k+1}^{safe}=a_{i,k}+B_{i,k}}
$$

应检查：

$$
\boxed{r_{i,k+1}+R_{i,k+1}^{ub}\le d_{i,k+1}^{safe}}
$$

由于 $R_{i,k}\ge 0$，在上述时间定义下，常用的：

$$
\boxed{T_{i,k}+R_{i,k+1}^{ub}\le B_{i,k}}
$$

是一个**更保守、更适合讲 Motivation 的简化条件**。论文中的实现和分析应以绝对时间为准，避免多算或漏算历史响应时间。

> **时间戳注意：** Safety Analyzer、选择器、数据传输、控制下发、作动延迟均会消耗真实时间；同一段开销只能计入一次。若 $B$ 由过时的采样状态计算，必须扣除状态年龄或重新验证其适用性。Safety Analyzer 可以返回 “无有效安全时间预算”。

### 3.2 响应时间分解：可解释性的入口

为了理解瓶颈，近似写成：

$$
\boxed{R_i^{ub}\approx C_i^{ub}(c_i)+W_i^{ub}(\sigma)+O_i^{ub}}
$$

- $C_i$：DNN 自身执行需求；
- $W_i$：排队、阻塞、资源干扰；
- $O_i$：切换、保存/恢复、DMA 与其他未计入 $C,W$ 的开销。

这里的拆分是**诊断模型**，不是可以直接取代严格响应时间分析的恒等式；各项不能重复计数。

当前框架保留约定：

$$
R_i^{ub}=R_i^{ub}(\mathbf c,\sigma,h\mid\text{已承诺任务和到达假设})
$$

不把 $T$ 机械地写成单个 job 响应时间函数的直接自变量。但改变 $T$ 会改变未来 job 到达与竞争，因此**全局可调度性与未来响应时间界仍与 $\mathbf T$ 耦合**。

---

## 4. 三个 Knob 并不独立：同一预算的三种调节方式

| Knob | 直接作用 | 能解决的问题 | 主要风险 |
|---|---|---|---|
| $T_i$ | 下一次 release 的时间与未来到达频率 | 下一份新反馈启动太晚 | 任务更密集，可能增加其他 UAV 的 $R$ |
| $c_i$ | 单次执行需求 $C_i(c_i)$ | 自身 DNN 计算过重、总需求过高 | 纠偏/跟踪能力可能下降，模式不是单调变好 |
| $\sigma$ | 执行顺序、阻塞、抢占/恢复 | 等待时间过长或不必要切换 | 等待时间在任务间重分配，可能牺牲其他 UAV |

**Insight 1 — Budget Decomposition（预算分解）：**

$$
\boxed{T_i+C_i+W_i+O_i\lesssim B_i}
$$

把安全时间违例的来源拆成 release、computation、interference 三部分。

**Insight 2 — Local Slack vs Global Interference（局部余量与全局干扰）：**

$$
\boxed{S_i=d_i^{safe}-a_i^{ub}}
$$

表示某 UAV 的保守安全余量；同时定义粗略工作负载指标：

$$
\boxed{U_{load}\approx\sum_i\frac{C_i(c_i)}{T_i}}
$$

$U_{load}$ 用于**识别负载压力和排序候选**，不能代替包含期限、非抢占阻塞、未来到达和切换/访存开销的全局 schedulability check。

**Insight 3 — Actions Have Different Side Effects（动作副作用不对称）：**

- $T_i\downarrow$：可能改善当前 UAV 的反馈时效，却增加未来计算需求；
- $c_i\downarrow$：可降低单次需求，但必须先满足最低控制合格性；
- 调整 $\sigma$：通常主要重新分配等待/阻塞，不能凭空修复真实容量不足；
- 减少抢占：有机会降低 $O$，也可能增加紧急任务的阻塞，必须检查。

**Insight 4 — 看可消除余量，不只看时间占比：** $C$ 占比很大，但已经处于最低合格 subnet 时就无法继续降；$W$ 占比很大，但其他任务更紧急时也未必能降。

---

## 5. 把 Insight 变成有因果解释的 Bottleneck Diagnosis

诊断只负责**引导候选**，不能凭占比直接宣称某个配置安全。

### 5.1 快速压力指标

在简化的 $T+R\le B$ 解释下，定义：

$$
\boxed{
P_T=\frac{T}{B},\qquad
P_C=\frac{C}{B},\qquad
P_W=\frac{W+O}{B}
}
$$

它们告诉我们 $B$ 被哪些环节占用。要求 $B>0$；若 $B\le0$，直接走异常/备用路径。采用绝对时间实现时，仍应使用 $S_i$ 做真实可行性判断。

### 5.2 三类典型失效

**A. Release-limited：下一份任务产生太晚**

假设即使不排队、使用最低合格计算需求，下一次控制仍赶不上截止时刻，那么只调整优先级无效。应考虑更早 release（缩短 $T$），同时核查全局到达负载的副作用。

**B. Compute/capacity-limited：计算需求太重或全局容量不足**

若可用时间区间内的必要总服务需求已超过 FPGA 能力，重新排序无法凭空增加算力。可以优先尝试成本更低、但仍已验证合格的 subnet；必要时与 $T$ 联合调整。不能把“最小网络”当成“最小合格控制器”。

**C. Interference/preemption-limited：计算量够，排队或切换拖慢**

若 $C$ 不大，主要超时来自 $W+O$，优先测试调度相关候选：EDF/limited-preemption 下的合法执行次序、抢占边界或减少不必要切换。前提是其他 UAV 的安全期限不被破坏。

现实中常有**混合瓶颈**。因此类型 A/B/C 只决定候选集合的生成方向，并不把算法锁死在固定顺序。

### 5.3 两个有用的必要性检查

1. **单任务最快完成下界：**若在当前已知条件下，即使立即运行并使用合法最低计算成本，也不可能在绝对 deadline 前产生新控制，则单纯调整 $\sigma$ 无效，应考虑更早 release（如果仍来得及）或 fallback。
2. **容量需求下界：**若某时间窗口内已承诺的必要计算需求超过处理器/加速器可提供的服务量，则仅重排顺序无解。需改变到达、模式、资源接纳或安全策略。

这两种检查非常适合发展成论文中的 **candidate-pruning rules / infeasibility certificates**；严格表达和适用假设需要后续推导。

---

## 6. Framework：Bottleneck-Guided Joint Adaptation

### 6.1 Offline：给在线决策提供少量可靠信息

1. **SuperNet / Mode Pool：**得到合法的 depth/width 子网及 job-boundary mode switching 机制；第一版不做同一 job 中途扩展 subnet。
2. **Timing Profile：**$C(c)$、segment execution、save/restore、DMA/DDR 与作动开销，尽可能建立保守界并明确测量环境。
3. **Controller Admissibility：**验证模式在指定危险状态、更新间隔和控制生效延迟范围内是否有足够的恢复/跟踪能力；未知条件不可默认合格。测试应包含不越界、误差恢复和任务进度，而不是仅看 margin 增加。
4. **Legal Configurations：**允许的 $T$ 范围、可选 $c$、最大基础更新间隔、合法抢占点；不允许最低控制要求被无条件降级。
5. **Optional Evaluation Data：**整体 QoC、能耗、计算量与切换连续性等可用于实验比较，不必在线训练/调用全局 QoC 黑盒模型。

### 6.2 Online：一次决策的完整顺序

```text
当前实际状态 x_i,k + 已生效的 u_i,k
                    │
                    ▼
            Safety Analyzer
       B_i,k 和绝对 safe deadline
                    │
                    ▼
       当前配置 / 任务队列 / segment 状态
                    │
                    ▼
     计算 S_i，诊断 T / C / W+O 压力
                    │
                    ▼
    检查可消除余量与显然无效的动作
                    │
                    ▼
     只生成少量针对性候选动作
      T-only / c-only / σ-only / joint
                    │
                    ▼
   快速筛除控制不合格或明显赶不上的候选
                    │
                    ▼
      全局保守可调度性检查（有界次数）
        所有 UAV + 已承诺 / 未来需求
                    │
             ┌──────┴──────┐
             │             │
           可行           未找到可行
             │             │
      选低代价候选     fallback / 报告
             │
            Commit
             │
       FPGA segment execution
             │
       新控制真正作用于 UAV
```

### 6.3 根据诊断生成少数候选（不是固定调参顺序）

| 诊断结果 | 优先生成的候选（示例） | 解释 |
|---|---|---|
| 当前配置已满足所有要求 | 保持不变 | 避免无意义调整和切换开销 |
| $W+O$ 主导且有可调度余量 | $\sigma'$，或减小不必要的抢占 | 尽量不损失 controller capability |
| $C$ 主导且存在更便宜的合格 subnet | $(T,c^-)$ | 降低执行需求 |
| release 时间主导 | $(T^-,c)$ | 更早获得反馈，但要核查未来负载 |
| release 与 compute 同时受限 | $(T^-,c^-)$ | 缩短周期时同时缓解计算负载 |
| 多 UAV deadline 同时紧张 | 少量跨任务联合候选 | 防止救一个 UAV 却拖坏另一个 |

**候选数设置上限 $K_{cand}$**（例如原型可以先取 3～5，仅是实验参数）。每次生成不是全 $T\times c\times\sigma$ 搜索，而是由瓶颈决定的**有方向的 bounded search**。

### 6.4 可解释的候选排序，但不代替可行性验证

对一个动作 $a$，定义其预计对紧急 UAV 的响应时效改善：

$$
\widehat G_i(a)=\hat a_i^{ub}(\text{before})-\hat a_i^{ub}(\text{after }a).
$$

同时估计它的全局负载变化 $\Delta \hat U(a)$、切换代价 $\hat O_{change}(a)$。可以采用一个**仅用于候选排序**的分数：

$$
\boxed{
\mathrm{score}(a)=
\widehat G_i(a)
-\lambda_U[\Delta\hat U(a)]_+
-\lambda_O\hat O_{change}(a)
}
$$

此式不是需要强行证明的最优目标；不同项量纲和权重必须在实现中规范化，或者干脆先用词典序规则（预计有效时间改善、是否增加负载、切换成本）排序。

**重要原则：一个分数高的动作不一定安全。** 只有通过下面的正式检查才能 commit。

### 6.5 Global Feasibility Check：保证其他 UAV 不被牺牲

对每个候选方案，至少检查：

$$
\boxed{r_{j,k+1}+R_{j,k+1}^{ub}\le d_{j,k+1}^{safe},\qquad \forall j}
$$

并且模式 $c_j$ 仍满足相应状态和延迟条件下的控制/恢复合格性。

实际检查须覆盖：

- 当前 running segment 与非抢占 blocking；
- ready 和已经承诺的 jobs；
- 与预测/假设到达范围有关的未来 jobs；
- segment precedence、切换与上下文恢复；
- DMA/DDR 干扰、作动延迟及决策开销；
- 所有 UAV 的 deadline、模式合法性和基本控制更新上限。

当前可用 EDF / segment EDF 作为基线；EDF-Q/limited-preemption 是下一步值得比较的可选实现。**不要直接把经典 EDF-Q 的静态 task-set 结论搬到动态 deadline 和动态 $c$ 的 FPGA 上。**

对于候选检查设固定计算预算。若检查次数耗尽，仅能说“当前 bounded policy 未找到可行候选”，不能直接声称任务集理论上不可行。

### 6.6 从多个可行候选中选择什么？

第一版采用简单的词典序目标：

1. **所有安全时间约束与控制器合格性必须通过**（硬条件）；
2. 尽量不改变原有配置；
3. 若需要改变，选择额外系统负载和切换代价较低的方案；
4. QoC/能耗作为次级比较或实验指标，不要求在线最大化全局 $J$。

若无可行候选或 $B\le0$，应有明确的 fallback / admission / 安全备用接口。取消过时 job 本身不等于安全保障；这属于后续策略研究，不宜直接当作当前硬保证。

---

## 7. Why This Is a Heuristic, Not a Black-Box Optimizer

| 维度 | 大型在线黑盒搜索 | 本框架的研究设想 |
|---|---|---|
| 决策依据 | 预测整体 QoC / 穷举组合 | 时间违例来源 + 可调整余量 |
| 搜索空间 | $T\times c\times\sigma$ | 少量瓶颈引导动作 |
| $R$ 的处理 | 可能作为预测特征 | 显式的保守调度响应时间界 |
| 对其他 UAV 的影响 | 依赖目标函数/惩罚项 | 逐项检查全局约束 |
| 可解释性 | 配置好坏可能难归因 | 可说明是在修复 release、compute 或 interference 瓶颈 |
| 主要风险 | 决策计算过重 | 可能遗漏可行候选、检查仍可能很贵 |

**低开销不能只由“候选数量少”推断**：需要测量完整开销

$$
\boxed{
O_{decision}=O_{budget}+O_{diagnosis}+O_{generation}+O_{feasibility}
}
$$

并报告平均、P99、最大观测值和候选/检查次数。若全局检查太慢，仍需对其进一步设计有界实现。

---

## 8. 学术 Insight 与可形成的分析结果

下面是**研究目标，不是已经成立的 theorem**。

### 8.1 Action Effect Bounds（动作有效性界）

为 $T^-$、$c^-$、$\sigma'$ 分别推导在已知执行时间与调度假设下，能够改善安全余量的保守上/下界，帮助快速排除不会修复违例的动作。

### 8.2 Feasibility-Driven Pruning（基于必要条件的剪枝）

给出两个低成本必要条件：单任务最快完成约束、时间窗口的总计算需求约束。它们不能证明方案可行，但能够证明某些动作一定无效。

### 8.3 Cross-Loop Interference Sensitivity（跨回路干扰敏感性）

研究 $T_i\downarrow$ 或 $c_i$ 改变对其他任务的 future demand / blocking 的影响，避免局部可行方案伤害全局安全余量。

### 8.4 Bounded-Overhead Heuristic（有界工作量启发式）

明确候选数量、调度模拟/检查次数的上限，并比较相对于小规模离线完整搜索的可行率差距。目标不是强行声称全局最优，而是证明**在有限决策开销下，能找到足够多的可行、控制合格方案**。

> 已有 state-triggered control、dynamic rate scheduling、EDF、limited preemption 和 SuperNet 模式切换等相关工作。**“用了这些组件”本身不是创新。** 可能的贡献要落在可解释的联合决策机制、可证明的排除规则、物理约束与多任务干扰的耦合，以及硬件上的系统性收益；是否足够新仍需和具体文献逐条比较。

---

## 9. 实验计划：让 Insight 由数据支撑

### Stage A：测量时间分解与违例类型

在 AirSim + 简化共享加速器回放中记录：

$$
B_i,\quad T_i,\quad C_i,\quad W_i,\quad O_i,\quad S_i,\quad U_{load}.
$$

观察在危险状态出现时，超时主要来自 release、computation、interference，还是混合原因；检验“某些 knob 确实更有效”的诊断是否成立。

### Stage B：比较启发式与基线

至少包含：

| 方法 | 用途 |
|---|---|
| Fixed $(T,c,\sigma)$ | 基准 |
| $T$-only | 只调周期 |
| $c$-only | 只调复杂度 |
| Scheduling-only | 只调调度/抢占决策 |
| Fixed-order nested adjustment | 验证先 $T,c$ 再 $\sigma$ 是否真的合理 |
| **Bottleneck-guided joint adjustment** | 提议方法 |
| Offline exhaustive / oracle on small instances | 测量启发式漏掉多少可行方案；不作在线方法 |

记录：

- 全局 deadline 满足率和最差 UAV 的 safety slack；
- 某个 UAV 改善是否导致其他 UAV deadline miss；
- controller 合格性、短时恢复成功率、轨迹完成与物理越界；
- 总计算需求、实际更新次数与能耗；
- 抢占次数、save/restore 开销与 blocking；
- 决策总开销、P99、最大观测值、检查次数；
- 相对离线完整搜索的可行解覆盖率。

### Stage C：FPGA + AirSim 端到端

保持 SuperNet、加速器和测试场景一致，比较 fixed、只调一个变量、固定顺序和瓶颈引导方案。逐步添加 segment EDF 与 limited-preemption 机制的对照；不要一开始同时改模型、粒度、调度和安全分析方法，否则难以归因。

---

## 10. 目前明确不做什么

- 不使用在线大型 ML predictor、RL 或完整 $T\times c\times\sigma$ 穷举；
- 不预设“危险程度越大 DNN 一定越大”；
- 不把 Pareto frontier、EDF/EDF-Q、segment scheduling 或简单 $B=d/\bar v$ 当作独立创新；
- 不把 heuristic 的候选排序分数当作安全证明；
- 不将经验性短窗口测试写成对所有物理状态的形式化安全保证；
- 不强制采用任何固定的 $T\to c\to\sigma$ 调整顺序；
- 暂不把预测未来危险、job dropping / weakly-hard、动态 PE 分区、PR 或 DVFS 同时引入主框架。

---

## 11. 最终精炼表述（适合组会）

**Motivation：** 多个智能控制回路共享 FPGA 时，物理风险变化会不断收紧下一次有效控制的时间预算。简单缩短周期可能增加资源竞争，降低 DNN 复杂度可能牺牲纠偏能力，改变调度顺序又可能让其他回路变得不安全。因此，不存在一个始终有效的固定调整次序。

**Key Insight：** $T$、$c$ 与 $\sigma$ 虽然是不同配置维度，却分别影响同一时间预算中的 **release waiting、own computation、interference / switching**，而且各自的全局副作用不同。应根据时间违例成因和可消除余量选择调整动作。

**Method：** 构建 **Bottleneck-Guided Joint Adaptation**：先计算动态安全 deadline 和安全余量，再诊断瓶颈，生成少量针对性 $T/c/\sigma$ 候选；最后使用 bounded global feasibility analysis 验证所有 UAV 的 timing requirement 和 controller admissibility，选择低额外负载的可行配置。

**What to prove / validate：** 剪枝规则的适用性、候选搜索与检查的实际时间上限、相对完整搜索的可行率损失、跨 UAV 干扰及物理恢复表现。目标是**低开销、可解释、全局约束不被牺牲**，而不是仅凭几个公式宣称已经最优或严格安全。

$$
\boxed{
\underbrace{T_i}_{\text{Release Timing}}
+
\underbrace{C_i(c_i)+W_i(\sigma)+O_i}_{\text{Response Time Components}}
\lesssim
\underbrace{B_i(x_i,u_i)}_{\text{State-Derived Time Budget}}
}
$$

> **框架目前的最重要下一步：** 不是继续增加新的模块，而是通过测量验证三个时间瓶颈的可辨识性，以及“瓶颈诊断能否显著降低候选检查次数，同时维持与完整搜索接近的可行率”。
