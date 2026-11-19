# 四个残差模型、服务规模与下一步容量实验

日期：2026-10-04。范围：237D 控制器的模型结果核验、两篇原论文定向核对、计时探针修订和用户运行命令。未启动训练、GPU 基准或真实 AirSim 飞行；未修改原始模型、数据、归一化、划分或地图配置。

## 1. 结论先行

增加到七八个甚至更多控制任务，是合理的容量实验，不应把三机当作 FALCON/SCENIC 的理论上限。但“更多任务需要资源管理”不等于“必须使用超网”：错峰、独立模型选择、准入控制可能已经足够。

当前应并行推进两个证据链：

1. **纯推理容量**：固定周期/期限，增加周期请求源，观察排队、尾延迟、丢弃与漏期；分别对比模型复杂度和同步/错峰。
2. **实际链路开销**：修正三机影子测试的判定与计时，排查规划、特征计算、Python 线程竞争、渲染与 GPU 推理之间的影响。未通过实时性验收前，不做该探针的 DNN 接管。

这两组实验不能混称“八架无人机已实时飞行”。

## 2. 四个残差模型全部完成

来源：[residual_vs_plain.json](F:/FalconReproduction/airsim_control/controller_architecture_study/results/decision_convex_residual4_v1/residual_vs_plain.json)，并检查了模型 completion 和输入/归一化来源；原有四模型加新 residual33 候选的 provenance 检查为 PASS。

全部为 600 epoch、seed=0。以下是标准化控制输出的验证 MSE，不是轨迹误差。

| 隐藏层×宽度 | 普通网络验证 MSE | 残差网络验证 MSE | 残差相对变化 | 验证/fresh-train |
|---|---:|---:|---:|---:|
| 4×512 | 4.12428e-4 | 5.51872e-4 | 增加 33.81% | 1.087 |
| 6×1024 | 2.85564e-4 | 3.31293e-4 | 增加 16.01% | 1.064 |
| 21×1024 | 3.35463e-4 | 2.49841e-4 | 降低 25.52% | 1.200 |
| 33×1024 | 3.71670e-4 | 2.35511e-4 | 降低 36.63% | 1.189 |

因此“浅层普通网络、深层残差网络”目前是有证据支持的候选结构，而非预先假定。33×1024 比残差 21×1024 再降低约 5.74%，是否值得额外耗时尚待测量。残差方案还包含相应初始化，不能将所有变化只归因于跳连；单 seed 也不足以确认普遍规律。

Gap 没有消失：验证误差仍高于 fresh-train 约 6%～20%，且高于 fixed-train 约 2.5～3.9 倍。它不像早期几十倍的差距，但不能声称无泛化误差。未读取正式 Test 来选择模型。

## 3. 重读原论文后的纠正

### FALCON

原论文：[FALCON, RTSS 2025](https://doi.org/10.1109/RTSS66672.2025.00040)。本次直接阅读本地 14 页原 PDF，不依赖此前生成的摘要。

- 建模面向 N 个控制任务；三机是案例规模，不是理论上限。
- PDF 第 8 页式 (12) 已包含 `R_i ≤ D_i ≤ T_i`，并联合考虑周期、网络和硬件资源等。不能将动机写成“FALCON 完全不约束周期/响应时间”。
- PDF 第 9 页 V-C：3、6、8 个任务时，优化耗时分别为 7.23 秒、56.85 秒、69.72 分钟。这里是**离线配置优化**，不是每次控制输出的时延，也不是八架无人机飞不动的证据。
- PDF 第 10 页实际案例同时控制三架无人机。

### SCENIC

原论文：[SCENIC, RTSS 2024](https://doi.org/10.1109/RTSS62706.2024.00026)。直接阅读本地原 PDF。

- 一般 N 任务的能力—调度协同设计，涉及模型、CPU/GPU 分配与优先级；不能把“多任务/优先级调度”本身当作新贡献。
- PDF 第 9 页 V-B 区分离线配置搜索和运行时执行。报告了 6/9/12 任务条件下的优化耗时：1892/3271/6257 秒；这不是每帧推理时间。
- 同页还报告 3 任务优化从 82 分钟降到 308 秒的折中，平均性能损失 9.8%，并把更简化的运行时协同设计列为未来方向。
- PDF 第 10 页实际案例仍是三架四旋翼。

上述事实支持研究“任务集合和状态紧迫程度变化时，如何快速更新决策”，但没有证明超网是唯一解。超网共享权重也不会自动消除调度搜索的组合复杂度，甚至可能扩大候选动作空间。

## 4. 已有计时其实给出了两种不同结果

原始记录根目录：[dynamic_complexity_237d_v1](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1)。20261003 的 12 轮 CUDA replay 现已全部落盘，共 36,000 次计划请求，全部完成，输出漏期为 0。

| 条件 | 三轮 release→output P99 范围 | 本组漏期 |
|---|---:|---:|
| 单路，残差 21×1024 | 20.84～21.33 ms | 0 |
| 三路同步，残差 21×1024 | 27.78～28.18 ms | 0 |
| 三路错峰，残差 21×1024 | 20.76～21.08 ms | 0 |
| 三路，四模型确定性轮换 | 22.32～22.91 ms | 0 |

默认 T=D=50 ms。残差 21×1024 平均服务时间约 3.8～4.2 ms。粗略负载估计 `rho ≈ N*C/T`：若 C≈3.9 ms，则 N=8 时约 0.624。**这是均值估计，不是可调度性证明**；释放抖动、排队尾部和其他工作均未被这个简单式子覆盖。新残差 33×1024 的 GPU 延迟还没有测，不能拿 21×1024 的数值代替。

反之，[原三机影子报告](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1/airsim_shadow_20261003T165223_655150Z/report.json) 虽然 15 门通过，但：

- 控制命令来自 MPC；DNN 下发次数为 0。
- 1200 次计划释放中，708 次跳过，179 次因未完成请求而丢弃，只有 313 个 DNN 结果。
- 以 50 ms 为期限，1200 次均未按时得到 DNN 结果。
- MPC 实际下发 492 次，均超过该期限；采样到下发平均 122.17 ms。
- DNN 采样到结果平均 243.29 ms，P99 354.08 ms。

所以不能说“三机实时控制完全不受影响”。正确说法是：MPC 在该次负载下仍完成了几何穿门，但该探针没有满足配置的实时期限。

## 5. 本次代码变更与验证

- `dynamic_probe_common.py`：增加 residual33 别名；旧四模型默认集合不变；报告 schema v2 加入分段时间、`realtime_passed`。
- `probe_shared_response.py`：支持 1～64 个独立周期请求源；输入索引按实际 N 生成，记录 `input_order_version=2`，不能把旧版与新版输入序列称作完全一致。
- `probe_state_value.py`：可显式选择模型列表，旧四模型分析保持默认。
- `probe_capacity.py`：新增容量扫描；同一五模型池常驻，同一 N 的模型/相位对照采用同一输入序列；batch=1、单 FCFS 工作线程。每轮独立保存，套件 summary 增量写入；Ctrl+C 后已完成结果保留，但目前不是自动续跑器。
- `probe_airsim_three.py`：分开记录状态 RPC、碰撞 RPC、预处理等待、参考生成、237D 编码、MPC 求解、入队/推理及命令提交；取消解除暂停后额外等待 200 ms；原数学控制器与参考时钟不变。
- 新的三机判定：`flight_passed` 只表示无错误且 15 门通过；`realtime_passed` 还要求完整计划释放、零输出漏期和零命令下发漏期；两者同时通过才 `acceptance_passed=True`。飞行通过但计时失败退出码为 2，报告仍保存。

CPU 微基准（env48/route0，每线程 20 次，无 GPU、无 AirSim、没有周期等待）发现：

| 参考生成 + 特征计算 | 平均工作耗时 | 含锁等待的平均总耗时 | 含锁等待总耗时 P95 |
|---|---:|---:|---:|
| 单线程 | 6.73 ms | 6.73 ms | 7.37 ms |
| 三线程同时计算 | 41.74 ms | 41.74 ms | 46.49 ms |
| 三线程，预处理串行化 | 7.29 ms | 18.88 ms | 65.56 ms |

这提示 Python/小 NumPy 运算线程竞争值得查，但不能解释全部实飞开销。串行化改善均值而锁等待尾部仍大，所以只作为显式候选 `--preprocess-policy serialized`，默认 `parallel` 保留原基线；不是已经验证的实时性修复。

本地已验证：36 个 CPU 回归测试；五候选 provenance PASS；8 路 CPU 回放 32 请求完成；residual33 两样本 CPU 加载/推理冒烟；容量入口的两路 CPU 同步/错峰完整执行，五模型常驻加载正常；本文两段 PowerShell 命令通过语法检查。冒烟不用于估计容量或模型精度。没有运行新的 GPU 扫描、没有修改正式训练/测试数据。

## 6. 先运行这一整段：容量扫描

关闭训练与 AirSim，保持电源/显卡模式一致。不要同时跑其他 GPU 实验。这里不需要 Play，不生成门、不控制无人机。默认 28 轮 × 15 秒，即 7 分钟测量时间，另加模型加载/预热；不是硬性完成时间保证。若显卡再次报驱动错误，停止基准并保留日志，不自动重试压测。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
& $trainPy -m controller_architecture_study.probe_capacity --device cuda --clients 1 3 6 7 8 12 16 --models plain6x1024 residual33x1024 --duration 15 --rounds 1 --period-ms 50 --deadline-ms 50
if ($LASTEXITCODE -ne 0) { throw '容量扫描中止；已完成的单轮报告和部分 summary 仍保留，请先检查错误' }
```

终端最后输出比较表与 `capacity_时间戳/summary.json`，其中每项链接对应 `replay_时间戳/report.json`。`status=COMPLETE` 表示实验完整执行，**不表示全部满足期限**；应看 `realtime_passed`、`output_deadline_miss_fraction`、`response_p99_ms` 和 `worst_client_miss_fraction`。

选 plain6 与 residual33 是对照“已有较好精度且推理较快的普通网络”和“目前最低验证误差的深残差网络”；不是重新训练两个模型。全部五个候选始终常驻，避免池大小随所测模型变化。多个请求源不会复制 N 份模型权重。

若发现边界，例如 6 路无漏期、8 路有漏期，再在边界附近重复 3 轮、每轮 60 秒；若 16 路仍有余量，可以增加到 24/32，不能为了得到过载而暗中修改期限。超过当前场景合理规模的人工压力测试须单独标注。各配置不是自动的安全准入上限。

## 7. 再单独运行：三机实际链路候选修复

容量扫描结束后，手动打开现有 Blocks/FalconEmpty 并 Play；若遗留其他实验门，先 Stop/Play。下面不会打开第二个 AirSim 实例，不改 settings。它只运行 MPC 影子模式，用新增分段计时检查串行预处理的实际效果。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -m controller_architecture_study.probe_airsim_three --mode shadow --model residual21x1024 --device cuda --phase sync --preprocess-policy serialized --environment-id 48 --route-index 0
if ($LASTEXITCODE -eq 2) { Write-Warning '穿门通过，但实时性未通过。把 report.json 发回来，不切换 DNN 接管。' }
elseif ($LASTEXITCODE -ne 0) { throw '影子测试异常，先查看保存的报告' }
```

如需同版消融，再把 `serialized` 改为 `parallel` 重跑；比较各段时间，不能把路径状态变化下的两次实飞当作逐状态完全配对实验。当前不提供自动八机实飞：那还需要扩展 AirSim 实体配置、出生点、独立路线及安全边界，应在计时问题厘清后单独实施。

## 8. 实验结果怎样决定下一步

| 观察 | 能支持什么 | 还不能支持什么 |
|---|---|---|
| 大模型同一状态误差更低，但所有负载都宽裕 | 状态相关精度收益 | 资源紧张/推理中调整的必要性 |
| 同步过期、错峰正常 | 释放相位是有效杠杆 | 必须用超网 |
| 大模型过期、较小模型正常 | 复杂度—及时性折中 | 共享权重必优于常驻独立模型池 |
| 实飞预处理占大头，纯推理宽裕 | 先改控制链路实现 | 把性能下降都归因于模型太大 |
| 在线任务/危险程度改变后，提前选择常常失配 | 值得比较推理中调整 | 多出口一定不够用 |
| 在同一误差、控制指标和预算下，弹性网络优于早退/模型池，且开销计入 | 超网的增量实证价值 | 对所有平台/任务的普遍必要性 |

建议问题表述：**任务数量与状态紧迫程度变化时，如何在线分配控制计算，使误差收益不被结果过期抵消？** T、模型复杂度、释放相位/调度是决策变量；超网是候选实现，不是前提。

后续需要依次实现：基于状态且不使用真值标签的独立模型选择器；任务动态加入/离开的可复现实验；固定宽度多出口基线；再对比弹性结构。必须计入选择、排队、切换、检查点/出口和已浪费计算开销，并以同一路线的穿门、恢复与控制有效期指标验证，不能只比较离线 MSE。控制有效期模型/阈值也尚未在本探针中证明。

尚未完成：真实多于三架无人机闭环、修订后实飞实时性验收、状态感知选择器、动态 T、早退/超网训练与中途调整、当前 237D 网络的 DPU 时延。旧 25D DPU 结果不能代替这里的测量。

## 9. 定向文献核查记录

2026-10-04，问题限定为两篇论文的服务规模、优化/运行时开销和调度约束；不是完整系统综述，也不能据此宣称新方案全球首创。

检索词：`FALCON FPGA Accelerated Real-Time Intelligent Controller Autonomous Systems pdf SCENIC`；`SCENIC scalable scheduling control neural inference controllers drones real time pdf`；`"SCENIC" "Capability and Scheduling" pdf`；`"FALCON" "FPGA-Accelerated" controller pdf`；`"SCENIC: Capability and Scheduling"`。

网络用于确认题名/DOI，网页访问不完整处没有猜测正文；随后读取本地原 PDF，并渲染检查两篇第 9 页表述：

- FALCON：`E:\BaiduNetdiskDownload\vitis-ai6.1\tmp\pdfs\FALCON.pdf`，重点 PDF 第 8～10 页。
- SCENIC：`C:\Users\chen\Documents\论文\无人机\SCENIC - Capability and Scheduling Co-Design for Intelligent Controller on Heterogeneous Platforms_RTSS_CCF-A_2024.pdf`，重点 PDF 第 3、9～10 页。

原文事实、现有工程数据和待验证推论已分节区分。先前生成的 SCENIC 文本摘要仅用于定位原文件，不作为证据来源。
