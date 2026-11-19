# ACADO 教师通过后的下一阶段：独立状态查询试点

## 已有证据与边界

你运行的 rest_env38_20261002T104302_882075Z.json：5/5 开口内过门，过门速度约 4.967–5.037 m/s，前 5 秒位移 23.96 m，控制周期超时 0，平均在线求解 0.949 ms。后半程静止是停车悬停。
这是环境 38 的在线一次 RTI、空中静止起步验证，不是所有环境、地面起飞或新 DNN 的保证。

下一步不是重训全部尺寸，而是生成新数据试点，再比较 h4w128、h8w256、h8w512。
容量实验已经证明扩宽有收益，但不能要求任何任务下“每次加大都必然更好”。

## 新数据与旧分支的区别

- 旧 12D 数据、全部旧模型和已有 300D 数据不动；新目录为 controller_architecture_study/results/decision_nmpc_v1。
- 新契约 decision_nmpc_local_relative_pv_300d_v1，不是严格 FALCON 12D 复现；不直接续训旧检查点。
- 当前 p/q/v 状态和完整未来参考一起输入，标签仍是 MPC 首条控制 [F, wx, wy, wz]。四维单位为 m/s²、rad/s、rad/s、rad/s。
- 编码先平移到当前位置原点，再用“参考速度减当前速度”。坐标采用 ACADO 的 Z-up/FLU，标签仍为 AirSim 的 NED/FRD 控制。

输入布局（索引从 0 开始）：

| 范围 | 内容 |
|---|---|
| 0:10 | 当前位置零向量、当前四元数 wxyz、当前线速度 |
| 10:220 | 21 个节点，每个为相对位置 3、参考四元数 4、相对速度 3 |
| 220:300 | 20 个节点的规划器前馈推力和角速度，每个 4 维 |

前馈来自规划加速度/姿态，不是 MPC 优化结果，没有标签泄漏。前 3 维为零是保留完整问题布局的占位，不是三维新信息。
前馈加速度不是当前测得加速度；当前测得加速度没有作为 ACADO 反馈状态，也不添加无效噪声。
保留当前速度后，可以恢复各节点参考速度。当前绝对位置保存在轨迹及相位中，不作为网络条件；当前 DLL 没有障碍物/绝对位置状态约束且感知代价权重为零，允许共同平移。

轨迹使用已有 C2 五次规划器，巡航/过门端点速度 5 m/s，初始速度 0，最终速度 0；速度/加速度可行性上限为 8 m/s、8 m/s²。存在巡航过冲，不能把目标 5 m/s 理解为全程硬限速 5。
参考窗口固定为 21 节点、间隔 0.1 s；未来实际闭环控制仍按 0.05 s 执行首动作。
轨迹只提供可行路径和相位。每条样本独立采样，不连续飞行采集，也不在扰动后重新规划并把时间归零。

40 个独立环境，包含近直线、S 形、升降组合和偏航航向路线；每条 1000 个状态，总计 40000。
8 类均衡采样：起步、跟踪、段切换、制动、位置恢复、速度恢复、姿态恢复、组合恢复。
起步显式包含相位 0 的零速状态，其他起步状态限实际速度 <=3.5 m/s；全部状态实际速度限制 8 m/s。
恢复位置扰动最多 x±0.6、y±1、z±0.6 m；速度各轴±1.5 m/s；姿态用各轴±0.25 rad 的旋转向量扰动。
这些独立状态没有全部经过真实 AirSim 恢复试飞，不能仅由“数据审计 PASS”宣称它们全部可达/可恢复。

## 教师标签与数据质量

新离线查询使用同一个 ACADO DLL/默认权重和边界，但每条查询完整初始化 primal/dual workspace，再做 4–32 次 RTI。
接受条件：首动作相邻迭代最大变化 <=1e-4，记录的 KKT <=1e-3，控制满足已编译边界，全部数值有限。
这是局部数值一致性筛查，不是非线性全局最优证明；也不等于已经通过在线一次 RTI 的同一个逐帧控制器。
未达条件的状态记录拒绝原因，在同一类别内重新采样；连续 8 次拒绝停止并保留 failure.json。不会偷偷缩小状态范围或放宽阈值。
每个环境第一条接受样本重复完整查询，首动作差异必须 <=2e-5。其他样本不冒充已经逐条重复验证。
本轮少量真实 DLL 查询：32 条接受样本，重复差异最大 0，KKT 最大约 4.008e-4；这是局部接口检查，不是全量数据质量指标。

拆分固定为环境 0–31 train、32–35 validation、36–39 test，各集合都含四种几何类别。
正式数量分别为 32000、4000、4000。每个完整轨迹只属于一个集合，归一化只用训练集。
prepare 会读取数据检查质量和写出三个分区，不会训练模型或用 test 控制 early stopping。

## 1. 先生成 320 条冒烟数据

无需 UE4/Play，也不用 GPU。使用包含 NumPy 和已有 ACADO DLL 支持的 AirSim Python。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

& $py -m controller_architecture_study.decision_dataset generate --smoke-test --resume
if ($LASTEXITCODE -ne 0) { throw '冒烟数据生成失败，请先查看报错' }
& $py -m controller_architecture_study.decision_dataset prepare --smoke-test
```

输出独立保存在 decision_nmpc_v1_smoke。应显示 status=PASS、samples=320、分区数量 256/32/32。冒烟数据不用于正式训练。

## 2. 生成正式 4 万条，再整理

冒烟成功后：

```powershell
& $py -m controller_architecture_study.decision_dataset generate --resume
if ($LASTEXITCODE -ne 0) { throw '正式数据生成失败，请先查看报错' }
& $py -m controller_architecture_study.decision_dataset prepare
```

每 100 条原子保存一次。Ctrl+C/掉电后重新运行同一 generate --resume 命令即可；已登记的数据块会校验 SHA256 并跳过，未完成块重做。
代码、DLL、样本数不能在同一目录内更改后混用。完全生成后才允许 prepare；准备步骤也可重复运行，不重新随机拆分。

主要输出：

- manifest.json：采集配置、源码/DLL 哈希、数据块哈希、拒绝统计。
- chunks/：独立状态标签与环境、相位、状态类别、速度、求解诊断。
- environments.json：门布局、C2 轨迹系数和时间，可用于后续一致的闭环回放。
- data_audit.json：覆盖/收敛/饱和情况，不是 DNN 成绩。
- dnn_dataset/：train.npz、validation.npz、test.npz、normalization.json、split_manifest.json。

把正式 data_audit.json 发回来，重点检查拒绝是否过高、起步是否有零速样本、低速覆盖和饱和分布；不能只看一个 PASS。

## 3. 数据确认后才启动三个代表结构

本轮已经准备好训练入口，但没有启动真实训练。先检查数据审计，再执行这一阶段。

```powershell
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -m controller_architecture_study.train_decision_pilot --device cuda --dry-run
& $trainPy -m controller_architecture_study.train_decision_pilot --device cuda
```

三个模型从头训练、seed=0、标准化输出 MSE、Adam lr=1e-4、batch=512、最多 400 epoch、至少 100 epoch、patience=80。
每个模型一个子进程，串行运行，不并行抢显存；每个完整 epoch 保存 last_model.pt，最佳验证 epoch 保存 best_model.pt。
中断后重新运行同一命令：跳过有 completion.json 的模型，其他从保存的完整 epoch 恢复，未完成 epoch 重做。第一 epoch 之前另有初始状态检查点。
异常停止整个队列，保留已完成模型和当前已有检查点；不能保证在硬件/驱动崩溃时正在写入的一次保存成功。
只评估 train/validation（包括状态类别分组指标），不读取测试集来挑架构。runs/comparison.json 每完成一个模型即更新。

## 本阶段仍未完成

- 正式采集和真实 GPU 训练由你执行，不是已经完成。
- 更复杂新环境和离线多次收敛教师尚未在 AirSim 中全量闭环验证。
- 300D 新编码的导出/闭环接入下一阶段完成；不能把这些检查点直接交给旧 12D gate runner。
- 等三个代表模型验证指标、闭环结果和时延都可用，再扩展尺寸和 seeds；不是先训练全部模型后才确认任务可用。
- 共享权重超网、动态 T/R 调度、Vitis AI/DPU 适配均不在本次修改范围。

本轮代码验证：19 项临时文件/CPU 流程回归通过（优化器更新被禁用），含旧 12D 容量实验的恢复测试；没有运行 AirSim 或真实训练。
