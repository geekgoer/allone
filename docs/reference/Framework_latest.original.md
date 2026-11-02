# State-Aware Fine-Grained Elastic Control Runtime

> 当前主版本：基于动态安全时间要求、状态相关最低计算需求与共享 FPGA 实时调度的智能控制框架

[TOC]

---

# 一、研究问题

当前工作的核心问题可以压缩为：

> 多条 neural control loops 共享一个 FPGA accelerator 时，不同 physical states 可能对应不同的控制更新时间要求，也可能对应不同的最低 neural computation requirement。如何以较低运行时开销，根据当前状态选择一个成本尽可能低但仍然“够用”的 controller mode，并安排下一次控制更新，使新的控制在动态时间要求内完成？

整个 framework 的核心链可以写成：

$$
\boxed{
x_k
\rightarrow
\begin{cases}
d_{k+1}^{safe}\\
c_{k+1}^{sel}
\end{cases}
\rightarrow
\text{Shared-FPGA Scheduling}
\rightarrow
r_{k+1}
}
$$

其中：

- $d_{k+1}^{safe}$：下一次新的 control command 最迟必须生效的时刻；
- $c_{k+1}^{sel}$：当前状态下计算成本最低的已验证合格 controller mode；
- $r_{k+1}$：下一 control job 的 release time。

当前工作的重点不再是在线最大化全局 trajectory-level QoC，而是：

$$
\boxed{
\text{State-Dependent Minimum Neural Computation}
+
\text{Dynamic Timing Requirement}
+
\text{Low-Overhead Shared-Accelerator Scheduling}
}
$$

---

# 二、从 FALCON 出发

FALCON 已经考虑：

$$
c_i,\quad T_i,\quad R_i,\quad \epsilon_i,\quad h,\quad s
$$

对 closed-loop control performance 的共同影响：

$$
J_i=C_i(c_i,T_i,R_i,\epsilon_i,\theta_i).
$$

其目标本质上是：

$$
\max \sum_i w_iJ_i
$$

并同时考虑实时性与 FPGA resource constraints。

因此 FALCON 已经解决了：

$$
\boxed{
\text{Control Performance}
+
\text{Real-Time Scheduling}
+
\text{FPGA Resource}
}
$$

之间的 co-design。

当前工作的切入点不是“FALCON 没有实时约束”，而是：

> FALCON 主要优化 trajectory-level / aggregate control performance，而没有把 UAV 当前 physical state 对“下一次控制何时必须生效”和“当前至少需要多少 neural computation”这两个要求显式提取出来。

---

# 三、Motivation 1：整体 QoC 好，不等于关键时刻一定安全

一个 UAV 整段轨迹的：

$$
J_i
$$

很高，并不意味着在某个很短的 critical interval 内一定不会出现危险状态。

例如：

```text
大部分飞行阶段：
tracking error 很小

        ↓

门前关键阶段：
横向误差迅速增大

        ↓

进入危险区域
```

因此：

$$
\boxed{
\text{Trajectory-Level QoC}
\not\Rightarrow
\text{Pointwise Physical Safety}
}
$$

即使增加：

$$
J_i\ge J_i^{min},
$$

也只能保证总体 control quality 不太差，并不能直接给出：

> 当前状态下，下一次新的有效控制最迟什么时候必须出现？

所以需要额外从当前 physical state 提取 dynamic timing requirement。

---

# 四、Motivation 2：静态 Worst-Case 配置又太保守

最简单的安全方案是：

> 找到整条轨迹最危险的状态，并始终按照这个 worst case 配置系统。

例如一直按照：

$$
30\text{ ms}
$$

的严格更新时间要求运行。

但真实需求可能是：

$$
150
\rightarrow
100
\rightarrow
60
\rightarrow
30
\rightarrow
120\text{ ms}.
$$

真正需要严格时间要求的可能只有很短一段。

因此：

$$
\boxed{
\text{Static Worst-Case Configuration}
=
\text{Safe but Over-Provisioned}
}
$$

它可能导致：

- 控制任务产生过于频繁；
- FPGA 长期高负载；
- 其他 UAV response time 增大；
- 大量无必要 computation。

因此需要 state-aware runtime adaptation。

---

# 五、Safety Analyzer：从 Physical State 得到动态时间预算

当第 $k$ 次控制真正作用到系统时，记：

$$
x_{i,k}
$$

为分析起点对应的 physical state，

$$
u_{i,k}
$$

为此时真正已经施加到 plant 上的 control command。

定义：

$$
\boxed{
B_{i,k}
=
\mathcal B_i
\left(
x_{i,k},
u_{i,k},
\Omega_i
\right)
}
$$

其中：

- $x_{i,k}$：当前 physical state；
- $u_{i,k}$：当前实际生效的控制命令；
- $\Omega_i$：动力学、扰动范围等 safety assumptions；
- $B_{i,k}$：从当前分析起点开始，下一次新的有效控制最多还能延后多久。

第一阶段可以使用较简单的 conservative model，例如：

$$
B^{simple}(x)
=
\frac{d(x)}{\bar v_e}.
$$

但此时应该把它称为：

$$
\boxed{
\text{dynamic safe-time budget / conservative timing budget}
}
$$

而不是直接声称已经形成严格 physical safety proof。

以后如果需要更强的理论保证，可以把：

$$
\mathcal B^{simple}
$$

替换为 CBF / sampled-data / self-triggered safe-period analysis：

$$
\boxed{
\mathcal B^{simple}
\longrightarrow
\mathcal B^{CBF}
}
$$

后面的 controller-mode selection 和 scheduling framework 可以保持不变。

---

# 六、为什么第一版不单独加入 DNN–MPC Approximation Error

Controller 保持：

$$
u_{i,k}
=
\pi_{c_{i,k}}(z_{i,k}).
$$

Safety Analyzer 使用的是经过：

- 反归一化；
- 限幅；
- control-interface conversion；

后真正作用到 plant 的：

$$
u_{i,k}^{actual}.
$$

因此 safety analysis 直接针对：

$$
(x_{i,k},u_{i,k}^{actual})
$$

进行。

第一版没有必要再额外引入：

$$
u_{\rm DNN}-u_{\rm MPC}
$$

作为新的 runtime variable。

但是 Safety Analyzer 必须允许返回：

$$
\boxed{
\text{No Valid Budget / Infeasible}
}
$$

因为并不是任意状态和任意控制输入下，都一定存在正的 safe horizon。

---

# 七、时间语义：使用 Absolute Safe Deadline

定义：

$$
r_{i,k}
=
\text{第 }k\text{ 个 control job 的 release time}
$$

$$
a_{i,k}
=
\text{第 }k\text{ 个 control command 真正生效的时刻}
$$

以及：

$$
T_{i,k}
=
r_{i,k+1}-r_{i,k}
$$

$$
R_{i,k}
=
a_{i,k}-r_{i,k}.
$$

当第 $k$ 次控制在：

$$
a_{i,k}
$$

真正生效后，根据对应的：

$$
(x_{i,k},u_{i,k})
$$

计算：

$$
B_{i,k}.
$$

然后得到下一次控制的 absolute safe deadline：

$$
\boxed{
d_{i,k+1}^{safe}
=
a_{i,k}+B_{i,k}
}
$$

即：

> 第 $k+1$ 次新的 control command 最迟必须在该时刻之前真正生效。

所以 runtime constraint 是：

$$
\boxed{
a_{i,k+1}
\le
d_{i,k+1}^{safe}.
}
$$

又因为：

$$
a_{i,k+1}
=
r_{i,k+1}
+
R_{i,k+1},
$$

所以：

$$
\boxed{
r_{i,k+1}
+
R_{i,k+1}^{ub}
\le
d_{i,k+1}^{safe}.
}
$$

平时讲 motivation 时仍可以简化成：

$$
T+R\le B,
$$

但严格实现优先维护 absolute deadline。

---

# 八、还需要一个最大控制更新间隔

仅依赖 physical safety budget 存在一个问题：

当 UAV 远离边界时，可能得到非常大的：

$$
B_k.
$$

如果 runtime 一味推迟下一次更新，虽然短时间内不碰撞，但可能：

- tracking error 持续增加；
- 偏离 reference trajectory；
- 无法正常完成飞行任务。

因此需要一个经过离线闭环测试确定的：

$$
\boxed{
T_{\max}^{ctrl}
}
$$

表示：

> 即使没有明显 physical danger，为保证基本 tracking / mission performance，controller 最长也不能多久不更新。

于是下一次控制的有效 deadline 可以写成：

$$
\boxed{
d_{i,k+1}
=
\min
\left(
d_{i,k+1}^{safe},
\;
a_{i,k}+T_{\max,i}^{ctrl}
\right)
}
$$

其中：

- $d^{safe}$：physical safety 给出的限制；
- $T_{\max}^{ctrl}$：基本 control-quality 给出的限制。

这样可以避免“安全预算很宽松，因此一直不更新”的问题。

---

# 九、Controller 与 Control Period 的关系

Controller 保持：

$$
\boxed{
u_{i,k}
=
\pi_{c_{i,k}}(z_{i,k})
}
$$

不把 $T$ 作为 DNN input。

这里 $T$ 是 sampling / release timing variable，只决定多久重新调用一次 controller，而不改变 controller 本身的控制律。

---

# 十、Fine-Grained SuperNet 的定位

SuperNet 提供大量共享权重的 controller modes：

$$
c_1,c_2,\ldots,c_K.
$$

每个 subnet 对应不同：

- depth；
- width；
- execution cost；
- approximation behavior；
- state-dependent recovery behavior。

第一版把 SuperNet 定位为：

$$
\boxed{
\text{Fine-Grained Multi-Mode Controller Library}
}
$$

其价值是提供大量细粒度 controller operating points，而不是让 runtime 在线扫描所有 subnet。

第一版规定：

$$
\boxed{
c\text{ 只在 job boundary 改变}
}
$$

已经开始执行的 job 不会中途：

$$
c_3\rightarrow c_5.
$$

job 内只允许 segment-level pause / resume。

---

# 十一、为什么不再使用 Global \(J\) 驱动 Runtime Selection

之前可以构建：

$$
J(c,T,R,\epsilon)
$$

并在线寻找最大总体 QoC 的配置。

但这仍然是在优化：

$$
\boxed{
\text{trajectory-level global control performance}
}
$$

而当前真正的问题是：

> 当前 UAV 已经处于某种状态时，哪个 controller mode 在给定时间条件下仍然“够用”，而且计算成本最低？

因此第一版 runtime selector 不再运行：

- QoC prediction ML model；
- 全局 $J$ optimization；
- $T\times c$ enumeration；
- Pareto-neighbor search。

全局 $J$ 可以保留作为最终 evaluation metric，但不直接驱动 runtime controller selection。

---

# 十二、核心假设：不同状态可能对应不同的合格 Controller Modes

对每个 subnet：

$$
c_m
$$

定义一个 empirical recovery-admissible region：

$$
\boxed{
\mathcal X_m(\bar\delta)
}
$$

其含义不是：

> 只要从这个状态立即运行 $c_m$，它就能恢复。

而是：

> 当 UAV 状态位于该区域时，在允许的最大等待 / 控制生效延迟不超过 $\bar\delta$ 的条件下，$c_m$ 仍然能够满足规定的 recovery requirement。

也就是说，admissibility 必须与离线验证过的 timing condition 绑定。

这里不假设：

$$
c_1<c_2<c_3
$$

就一定意味着：

$$
\mathcal X_1
\subset
\mathcal X_2
\subset
\mathcal X_3.
$$

DNN complexity 并不天然等价于 recovery capability。

这一关系必须由 offline experiment / analysis 实际确定。

---

# 十三、为什么 Admissibility 必须考虑等待 / 延迟

假设当前：

$$
x_k
$$

看起来较小 subnet 已经足够恢复。

但是新的 controller 并不会立即生效。

从当前分析起点到新控制真正生效之前，UAV 仍然在运动，而且旧控制仍然保持。

因此可能出现：

```text
当前状态：
c2 看起来够用

        ↓ 等待

离边界更近、速度更危险

        ↓

c2 实际已经不够用
```

所以第一版不能：

> 在几乎零等待条件下建立表，然后允许运行时任意推迟控制更新。

第一版最简单的处理方式是统一规定：

$$
\boxed{
0\le\delta\le\bar\delta
}
$$

并在离线 recovery test 中显式覆盖这一 delay 范围。

只有在规定 timing condition 下仍然表现合格的 mode，才能标记为 admissible。

因此运行时还必须满足：

$$
\boxed{
\delta_{actual}
\le
\bar\delta
}
$$

否则该 State-to-Mode Map 的结论不再适用。

第一版不需要把 $\delta$ 再增加成 lookup table 的一个新维度。

---

# 十四、Recovery-Admissible Region 如何得到

第一阶段先使用 empirical short-horizon recovery test，而不是直接做复杂 neural-controller formal verification。

选择代表性危险状态，例如：

- 横向偏差较大；
- 横向速度仍朝向 boundary；
- 已经开始纠偏但 margin 很小；
- 高速接近 gate；
- 转弯过程中出现较大 tracking error。

对完全相同的 initial state，分别使用不同 subnet：

$$
c_m
$$

并统一控制：

- initial position；
- velocity；
- attitude；
- reference trajectory；
- trajectory clock；
- 当前保持命令；
- update interval；
- control actuation delay。

这样才能把不同结果尽量归因于 controller mode 本身，而不是时序不一致。

---

# 十五、Recovery 判据

不能只使用：

$$
h(x(H))-h(x(0))
\ge
\Delta h_{\min}
$$

作为恢复成功判据。

因为 UAV 可能通过：

- 悬停；
- 倒退；
- 绕开目标；

让 safety margin 变大，却没有真正完成任务。

第一版 recovery success 应至少同时满足：

### 1. 整个等待 + 恢复窗口内没有碰撞或越界

$$
h(x(t))\ge0
$$

对整个测试窗口成立。

### 2. 恢复到预先定义的位置 / 速度误差范围

例如：

$$
|e_p|
\le
e_p^{th}
$$

$$
|e_v|
\le
e_v^{th}
$$

并持续：

$$
H_{\rm sustain}.
$$

### 3. 任务进度正常

不能通过明显：

- 悬停；
- 倒退；
- 偏离 reference route；

来换取“恢复成功”。

### 4. Safety margin 作为辅助指标

记录：

$$
h_{\min}
$$

以及：

$$
\Delta h
$$

但不把它们单独作为 recovery success 的唯一判据。

---

# 十六、State Representation 第一版先保持简单

第一版先从少量 risk-related features 开始，例如：

$$
\boxed{
\phi(x)
=
[h(x),\dot h(x)]
}
$$

其中：

- $h(x)$：当前 safety margin；
- $\dot h(x)$：当前正在接近还是远离 safety boundary。

暂时不一开始加入：

- 完整姿态；
- acceleration；
- reference curvature；
- current command；
- 高维 DNN state。

因为过早增加 feature 会使 lookup table 很快变得高维和稀疏。

但第一阶段实验必须主动检查：

> 在相同 $(h,\dot h)$ bin 内，改变姿态、reference trajectory、转弯方向等因素时，是否仍然能找到一致的 admissible mode？

如果不能，再增加最关键的一个 feature，而不是直接扩成高维状态表。

---

# 十七、Admissibility Table 需要三种状态

对每个 state bin 和 controller mode，离线结果不应该只有：

$$
\{\text{success},\text{failure}\}.
$$

还需要：

$$
\boxed{
\{\text{admissible},\text{inadmissible},\text{unknown}\}.
}
$$

例如：

| State Bin | $c_1$ | $c_2$ | $c_3$ |
|---|---|---|---|
| A | ✓ | ✓ | ✓ |
| B | × | ✓ | ✓ |
| C | ? | ? | ✓ |

其中：

- ✓：在规定 timing condition 下经过足够测试后认为合格；
- ×：明确不满足 recovery requirement；
- ?：覆盖不足或结果混杂。

runtime 不能把 unknown 自动当成 admissible。

对于 unknown，可以使用一个经过更多验证的 conservative baseline mode。

---

# 十八、Offline State-to-Mode Map

如果第一阶段实验表明 state-dependent controller admissibility 确实存在结构，则离线构建：

$$
\boxed{
M(\phi)
}
$$

每个 state bin 不存“最小网络”，而存：

$$
\boxed{
\text{minimum-cost admissible mode}
}
$$

即：

$$
\boxed{
c^{sel}(x)
=
\arg\min_{c_m\in\mathcal A(x)}
C(c_m)
}
$$

其中：

$$
\mathcal A(x)
$$

表示当前 state bin 下所有经过验证的 admissible controller modes。

这里不能说：

> “至少需要 $c_5$”。

更准确的说法是：

> 在当前已评估 modes 中，$c_5$ 是满足 recovery requirement 的最低成本 mode。

因为：

$$
c_5\text{ 合格}
$$

并不意味着：

$$
c_6,c_7,\ldots
$$

一定全部合格。

---

# 十九、Runtime Mode Selection

运行时提取：

$$
\phi(x_{i,k})
$$

然后直接查表：

$$
\boxed{
c_{i,k+1}^{sel}
=
M\left(\phi(x_{i,k})\right)
}
$$

正常情况下其开销接近：

$$
\boxed{
O(1)
}
$$

因此 runtime 不需要：

- 枚举 136 个 subnet；
- 调用 ML QoC predictor；
- 做 Pareto search；
- 做复杂 nonlinear optimization。

真正值得验证的是：

> 这个简单 map 是否能够在 unseen states 上可靠地选择一个仍然合格的 mode。

---

# 二十、选完 Mode 后，\(R\) 仍然必须由实时调度处理

选定：

$$
c_{i,k+1}^{sel}
$$

后，可以获得其离线上板 measured execution cost：

$$
C(c_{i,k+1}^{sel}).
$$

但真实 response time：

$$
R_i
$$

还包括：

- execution；
- waiting；
- non-preemptive blocking；
- segment switching；
- memory / DMA contention。

因此：

$$
\boxed{
R_i\neq C_i.
}
$$

运行时必须检查：

$$
\boxed{
r_{i,k+1}
+
R_{i,k+1}^{ub}
\le
d_{i,k+1}.
}
$$

所以：

$$
\boxed{
R
\text{ 没有被忽略，而是直接进入 schedulability condition。}
}
$$

---

# 二十一、Inner Scheduler：第一版采用 Segment-Level EDF

每个 DNN job 被划分为：

$$
S_{i,1},
S_{i,2},
\ldots,
S_{i,M_i}.
$$

segment 内：

$$
\boxed{
\text{non-preemptive}
}
$$

segment boundary：

$$
\boxed{
\text{允许 pause / resume / task switch}
}
$$

每个 ready job 已经拥有 absolute deadline：

$$
d_i.
$$

第一版直接使用：

$$
\boxed{
\text{Segment-Level EDF}
}
$$

即：

> 在合法 segment boundary，优先执行 absolute deadline 最早的 ready job。

EDF 本身不是创新。

第一版也暂时不再引入 Safety-Laxity 作为新的 scheduler contribution。

---

# 二十二、Release Planning：不再追求“精确最晚”，而是 Bounded Feasible Release

之前使用：

$$
r_{i,k+1}^{*}
=
\text{latest feasible release time}
$$

容易让问题重新变成复杂的全局在线优化。

共享 accelerator 下，不同任务的 latest release 不能独立计算。

因此第一版把目标收敛为：

$$
\boxed{
\text{在保护已有调度承诺的前提下，用有限计算量找到一个较晚的可行 release time。}
}
$$

可以只在有限 candidate events 上尝试，例如：

- 当前时刻；
- 下一个 segment completion；
- 下一个 job completion；
- 已经存在的 release event；
- 少量提前量 candidate。

最多检查：

$$
K_{\rm rel}
$$

个 release candidates。

从较晚的 candidate 开始尝试，只要：

$$
r_{i,k+1}
+
R_{i,k+1}^{ub}
\le
d_{i,k+1}
$$

且全局 schedule feasible，就接受。

因此 release planner 的工作量从设计上有明确上限。

如果有限次检查没有找到可行安排，应表述为：

$$
\boxed{
\text{current bounded policy found no feasible schedule}
}
$$

而不是立即宣称：

$$
\boxed{
\text{the task set is theoretically infeasible}
}
$$

---

# 二十三、Control Period 如何得到

一旦确定新的 release time：

$$
r_{i,k+1},
$$

则：

$$
\boxed{
T_{i,k}
=
r_{i,k+1}
-
r_{i,k}.
}
$$

因此第一版中：

$$
\boxed{
T
\text{ 不是在线枚举变量，而是 bounded release planning 的结果。}
}
$$

这仍然保留了 self-triggered / state-dependent update 的思想，同时避免在线扫描多个候选 periods。

---

# 二十四、为什么“尽量晚 Release”只是 Heuristic

不能声称：

> 越晚 release，一定意味着长期总计算量越低。

因为如果更新太晚，下一 physical state 可能变得更困难，导致：

- 下一个 state bin 需要更昂贵的 controller mode；
- 后续需要更频繁更新；
- recovery time 增长。

因此：

$$
\boxed{
\text{late release}
}
$$

只是当前一步减少 unnecessary update 的 heuristic。

论文最终必须测量：

$$
\boxed{
\text{actual total computation over the full mission}
}
$$

而不能只展示某一次 release 被推迟了多少。

---

# 二十五、不再单独设计 Resource Reclamation Module

因为每条回路每次新 job 本来就会重新根据当前 state 选择：

$$
\boxed{
\text{minimum-cost admissible mode}
}
$$

当某个 UAV 状态变得宽松后：

```text
之前危险：
选 c5

状态变宽松：
下一 job 查表得到 c2
```

未来 workload 会自然下降。

所以第一版不需要额外再设计：

$$
\text{Resource Reclamation Module}.
$$

如果要修改已经承诺的 future job configuration，则需要重新执行 global schedulability check。

---

# 二十六、Offline Profiling 现在需要做什么

Offline 阶段主要包含：

### 1. 训练 Fine-Grained SuperNet

得到多个合法 subnet：

$$
c_1,\ldots,c_K.
$$

### 2. FPGA Timing Profiling

测量：

$$
C(c_m)
$$

以及：

- segment execution time；
- pause / resume overhead；
- switching overhead；
- memory / DMA overhead。

### 3. Recovery-Admissible Region Profiling

在 representative dangerous states 上进行 short-horizon recovery tests，并覆盖：

$$
0\le\delta\le\bar\delta.
$$

得到：

$$
\mathcal X_m(\bar\delta).
$$

### 4. 构建 State-to-Mode Map

根据：

$$
\phi(x)
$$

建立：

$$
M(\phi)
=
\text{minimum-cost admissible mode}.
$$

### 5. 确定 Basic Control Update Bound

通过 closed-loop tests 得到：

$$
T_{\max}^{ctrl}.
$$

### 6. 确定 Segment Granularity

实测：

$$
\text{blocking reduction}
\leftrightarrow
\text{switch overhead}
$$

并选择固定合理粒度。

---

# 二十七、完整 Online Runtime Flow

```text
          当前已生效 control u_i,k
                   +
             当前状态 x_i,k
                   │
                   ▼
            Safety Analyzer
                   │
                   ▼
                B_i,k
                   │
                   ▼
      d_i,k+1^safe = a_i,k + B_i,k
                   │
                   ▼
      d_i,k+1 = min(d_i,k+1^safe,
                    a_i,k + T_max^ctrl)
                   │
                   ├──────────────────────────┐
                   │                          │
                   ▼                          ▼
           Risk Feature φ(x)            Dynamic Deadline
                   │                          │
                   ▼                          │
           State-to-Mode Map                  │
                   │                          │
                   ▼                          │
      c_i,k+1^sel = minimum-cost              │
          admissible mode                     │
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                Bounded Release Planner
                    + Segment-Level EDF
                                │
                     try ≤ K_rel candidates
                                │
                         feasible release?
                          /           \
                        YES            NO
                         │              │
                         ▼              ▼
                      commit       conservative /
                                   fallback handling
                         │
                         ▼
                    FPGA execution
                         │
                         ▼
                   new control u_i,k+1
                         │
                         ▼
                 actual actuation a_i,k+1
```

---

# 二十八、Offline / Online 对应关系

| Offline | Online |
|---|---|
| 训练 fine-grained SuperNet | 读取当前 $x_k,u_k$ |
| Profile $C(c)$ | Safety Analyzer 生成 dynamic deadline |
| Recovery tests with bounded delay | 提取 $\phi(x)$ |
| 构建 $\mathcal X_m(\bar\delta)$ | 查 State-to-Mode Map |
| 构建 $M(\phi)$ | 得到 $c^{sel}$ |
| 确定 $T_{\max}^{ctrl}$ | 形成最终 control deadline |
| Profile segment/switch overhead | Segment EDF |
| 确定 segment granularity | Bounded release planning |
| — | 检查 $R^{ub}$ 与 deadline |
| — | commit / conservative handling |

---

# 二十九、当前最核心的 Research Hypotheses

## H1：Dynamic Timing Requirement 确实存在

不同 physical states 应该产生明显不同的：

$$
B_k
$$

或：

$$
d_{k+1}^{safe}.
$$

否则 state-aware timing adaptation 的意义较弱。

---

## H2：不同 Physical States 对 Neural Computation 的最低需求不同

应该存在可重复结构：

$$
\boxed{
\text{state}
\rightarrow
\text{set of admissible controller modes}
}
$$

并进一步存在：

$$
\boxed{
\text{minimum-cost admissible mode}
}
$$

随 state 改变。

这是当前最关键的 research hypothesis。

---

## H3：低维 State-to-Mode Map 能够可靠泛化

使用较少的 risk features，例如：

$$
[h,\dot h]
$$

是否能够在 unseen states 上仍然可靠地识别合格 mode？

如果不能，再增加最关键的新 feature。

---

## H4：Fine-Grained SuperNet 能减少 Over-Provisioning

比较：

$$
K=6,\quad18,\quad64,\quad136
$$

等不同候选粒度。

验证更密集 controller modes 是否真的能在相同 recovery requirement 下减少 computation。

---

## H5：Segment-Level Scheduling 的收益大于切换开销

比较：

- whole-job non-preemptive；
- coarse segment；
- fine segment。

验证 blocking reduction 是否能够抵消 pause / resume / switching overhead。

---

# 三十、第一阶段最应该先做的实验

在继续实现完整 release planner 前，先验证最关键假设：

$$
\boxed{
\text{不同 subnet 是否真的存在可利用的 state-dependent recovery difference？}
}
$$

使用现有 SuperNet 中少量代表性 subnet，例如：

$$
K=6
$$

个配置。

选择一批 representative initial conditions，并保证不同 subnet 使用相同：

- initial position；
- velocity；
- attitude；
- reference trajectory；
- trajectory clock；
- current held command；
- update timing；
- actuation delay。

第一轮主要记录：

| 指标 | 作用 |
|---|---|
| recovery success / failure | 判断 mode 是否合格 |
| collision / boundary violation | safety |
| minimum safety margin | 危险程度 |
| recovery time | 纠偏效率 |
| position / velocity error | 是否真正恢复 |
| path progress | 排除悬停 / 倒退 |
| actual timing | 保证不同 mode 时序可比 |
| execution cost | 后续选择最低成本 mode |

第一轮只需要回答：

> 同一个 initial state 下，不同 subnet 是否存在稳定、可重复的恢复能力差异？

如果没有差异，动态 complexity selection 的收益可能很小。

如果差异存在但低维 state feature 无法解释，则 simple lookup map 可能不可靠。

如果存在稳定结构，则继续建立 State-to-Mode Map。

---

# 三十一、第一阶段之后再补 Delay Validity Test

只对第一轮出现 mode boundary 的少量 states 再增加：

$$
\delta
$$

维度。

即：

```text
当前 initial state
        │
        │ hold old control for δ
        ▼
   new subnet takes over
        │
        ▼
  evaluate recovery
```

用它来确认：

$$
\boxed{
\mathcal X_m(\bar\delta)
}
$$

真正对应运行时允许的 waiting / actuation delay。

没有必要把所有 initial states、所有 subnet、所有 delay 做全组合。

---

# 三十二、创新边界

已有 self-triggered / state-aware scheduling 工作已经研究过：

$$
\boxed{
\text{physical state}
\rightarrow
\text{dynamic deadline}
\rightarrow
\text{scheduling}
}
$$

因此不能单独把这一链条作为当前工作的主要 novelty。

当前更值得验证和强调的是：

$$
\boxed{
\text{不同 physical states 是否对应不同的 minimum neural computation requirement}
}
$$

以及：

$$
\boxed{
\text{能否离线识别这种 controller-mode admissibility，并以极低 runtime overhead 使用它}
}
$$

SuperNet 的作用是提供足够密集的 neural controller modes；

动态 deadline 和 EDF 则是把这些 controller modes 部署到共享 accelerator 上所需的 runtime mechanism。

---

# 三十三、当前明确删除 / 弱化的旧思路

第一版暂时删除或弱化：

### 1. 全局 $J$ 在线优化

$$
\max J(c,T,R,\epsilon)
$$

只保留作 final evaluation，不作为 online selector。

### 2. Latency–QoC Pareto Frontier 作为主要 Runtime Path

不再执行：

$$
\text{walk on Pareto frontier}.
$$

### 3. 在线枚举 / 局部搜索 $(T,c)$

$c$ 通过 State-to-Mode Map 得到；

$T$ 由 bounded release planning 的 release time 决定。

### 4. Safety-Laxity 作为第一版 Scheduler Innovation

第一版直接使用 Segment EDF。

### 5. CLF Recovery Rate $\lambda\rightarrow c$

不作为主算法，因为 controller complexity 与 recovery capability 不保证单调。

### 6. 独立 Resource Reclamation Module

下一 job 自然重新选择 minimum-cost admissible mode 即可。

### 7. Energy 进入 Runtime Objective

Energy 第一版只测量，不参与 online optimization。

### 8. Intra-Job Subnet Expansion

第一版只做：

$$
\boxed{
\text{inter-job controller-mode switching}
}
$$

---

# 三十四、最终 Framework 总结

当前 framework 可以压缩成两条并行链：

## Physical / Control Side

$$
\boxed{
(x_k,u_k)
\rightarrow
B_k
\rightarrow
d_{k+1}
}
$$

以及：

$$
\boxed{
x_k
\rightarrow
\phi(x_k)
\rightarrow
c_{k+1}^{sel}
}
$$

其中：

$$
c_{k+1}^{sel}
=
\text{minimum-cost admissible mode}.
$$

## Real-Time Systems Side

$$
\boxed{
(c_{k+1}^{sel},d_{k+1})
\rightarrow
\text{Segment EDF + Bounded Release Planning}
\rightarrow
r_{k+1}
}
$$

最终：

$$
\boxed{
T_k
=
r_{k+1}-r_k.
}
$$

整个研究主线可以概括为：

$$
\boxed{
\text{Physical State}
\rightarrow
\begin{cases}
\text{Minimum-Cost Admissible Neural Controller}\\
\text{Dynamic Control Deadline}
\end{cases}
\rightarrow
\text{Low-Overhead Shared-FPGA Scheduling}
}
$$

其中：

- Safety Analyzer 回答：**下一次控制最晚什么时候必须生效？**
- State-to-Mode Map 回答：**当前状态下，哪一个已验证 controller mode 成本最低但仍然够用？**
- Segment EDF + bounded release planner 回答：**多个 UAV 共享 FPGA 时，什么时候 release 才能以有限 runtime overhead 满足这些 deadline？**

这就是当前最新、最收敛的一版 framework。
