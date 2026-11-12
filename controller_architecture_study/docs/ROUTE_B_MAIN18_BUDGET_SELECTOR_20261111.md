# Main18 板端误差、服务时间与预算选模

## 板端测量

冻结的 237D validation 数据共有 120 条路线。固定随机种子，每条路线取 8 个输入，全部 18 个模型收到同一批 960 条输入。按环境 48／50／52 和 49／51／53 分成建表和核对两组，各 480 条。数据来自同一个已有的 validation 划分。

从 Zynq 5237 收集了 17,280 次 INT8 回复。两组中的样本均获所有模型有效回复，因此误差在共同样本上直接比较。误差指标是相对 MPC 标签的标准化 MSE。推力与三个体轴角速度各自用冻结训练集标准差归一化，再取四通道和样本的平均平方误差。原始物理单位 RMSE 和每模型的服务 P99 见完整结果。

| 模型 | 建表服务 P99 | 建表标准化 MSE | 核对组标准化 MSE |
| --- | ---: | ---: | ---: |
| ID 5，2×128 | 0.092 ms | 0.009295 | 0.009256 |
| ID 10，3×256 | 0.190 ms | 0.008207 | 0.006670 |
| ID 11，4×1024 | 1.323 ms | 0.021812 | 0.013615 |
| ID 6，33×1024 | 11.783 ms | 0.255126 | 0.187368 |

从 ID 5 到 ID 10，核对组标准化 MSE 降约 28%，服务 P99 增加约 0.098 ms。在这批已部署的 INT8 模型中，建表组只有 ID 5 和 ID 10 构成误差与服务成本的 Pareto 前沿。[18 模型完整结果](../results/main18_board_profile_20261111_v1/profile.json)

相同 480 条核对输入在冻结的 FP32 checkpoint 上再跑一遍。跨 18 个模型，MAC 数和标准化 MSE 的 Spearman 相关系数在 FP32 为 −0.746，在板端 INT8 为 +0.715。33×1024 的误差从 FP32 的 0.000352 增至板端 0.187368，4×1024 从 0.000325 增至 0.013615。结果把排查重点指向当前量化与部署链，具体是哪一层或哪一环需要逐层定位。[相同行 FP32 对照](../results/main18_board_profile_20261111_v1/fp32_comparison.json)

## 选模规则

令 $b_i$ 为调度器给任务 $i$ 分配的剩余板端服务预算，$\widehat C^{99}_m$ 为建表组服务 P99，$\widehat E_m$ 为建表组标准化 MSE。候选必须在建表样本中全部获得有效回复。先按预算筛选，再选误差最小的模型，同误差时选服务时间更低的模型。

$$
m_i^*=\arg\min_{m:\ \widehat C^{99}_m\le b_i}\widehat E_m .
$$

在生产接入时，$b_i$ 应来自安全期限与任务期限中的较早者，扣除当前时刻、剩余输入与下发路径、等待及预留。候选为空时由调用方报告无可行模型。P99 可用于经验选模。硬期限需要可信上界、全组可调度性检查及闭环资格证据。

## 板端参与的 AirSim 闭环

使用历史早弯和独立反向弯两个初态。三种策略分别为固定 ID 5、固定 ID 10、预算选模，每种策略每个初态重复两次。AirSim 用步进时钟，以 50 ms 周期和目标 20 ms 命令延迟隔离各模型本身的闭环效果。预算由实验在 0.4 秒从 0.10 ms 放宽至 0.20 ms。动态策略每次更新调用选择器，实际板端推理并使用返回的 INT8 控制量。

| 策略 | 两门均达标 | 板端服务超给定预算 |
| --- | ---: | ---: |
| 固定 ID 5 | 4/4 | 0/176 |
| 预算选模，ID 5 → ID 10 | 4/4 | 0/176 |
| 固定 ID 10 | 4/4 | 32/176 |

历史早弯中，预算选模与固定 ID 5 的第 4 门平均净空分别为约 11.4 cm 和 20.2 cm，第 5 门分别为约 84.7 cm 和 65.3 cm。独立反向弯中，固定 ID 5 的两门净空都略大。更低的教师拟合误差对应模型保真度，闭环净空仍取决于初态、参考和实际动作。这批数据验证了指定预算下的板端切换与过门结果。剩余预算从真实多任务调度器导出及联合期限验证仍待接入。[逐次闭环记录](../results/main18_board_budget_20261111_v1/summary.json)

## 复跑

Windows 工程根目录运行，板端服务为 192.168.77.2:5237，闭环对照需要 Blocks 已 Play。输出使用新目录。

~~~powershell
$py='D:/files/PYTHON/cv_lab/Yolov8_Beginner_Demo-main/.venv/Scripts/python.exe'
$base='controller_architecture_study/results'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$profile="$base/main18_board_profile_$stamp"
& $py -u -m controller_architecture_study.profile_main18_board --output $profile
if ($LASTEXITCODE -ne 0) { throw '板端建表失败' }
& $py -u -m controller_architecture_study.probe_main18_budget_policy --profile "$profile/profile.json" --results $base --output "$base/main18_board_budget_$stamp"
~~~
