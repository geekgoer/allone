# SCENIC 命名批次：容量反转诊断与 AirSim 试飞

日期：2026-10-02。原始数据、训练脚本、旧模型、旧配置均未修改。本次没有训练，也没有启动 AirSim 飞行；只做 CPU 诊断、论文核对与三个模型的独立导出。

## 结论与证据边界

不能把“大 DNN 的测试 MSE 必须更低”作为必须做出的结果。当前有三个具体问题：SCENIC 层数映射错误；教师控制律比论文中的非线性动力学/路径规划组合简单得多；现有大模型同时存在优化不足与泛化差距。增加 epoch 有可能改善，但没有证据能保证翻转排名。

### 1. 论文并非给出了严格单调定律

FALCON Table II 的 L=10 两个模型已经是反例：宽度 512 的 MSE=0.094，宽度 1024 的 MSE=0.117。作者的概括不能解释为对所有架构、数据和训练配置成立的定理。该论文还用 1/beta 量化 c；这种精度能力指标也不能直接等同参数量。

SCENIC 第 7 页（印刷页 207）明确把激活函数计为层；第 8 页（208）举例 L=23 包含 12 个 FC 和 11 个 activation。因此普通 FC+ReLU、输出 FC 不接 ReLU 时，隐藏 FC 数 h=(L-1)/2。当前 `configs/scenic_table3_v1.json` 和原训练说明误写成 h=L-1。

| 论文 L | 当前批次隐藏 FC 数 | 按 SCENIC 定义的隐藏 FC 数 |
|---:|---:|---:|
| 13 | 12 | 6 |
| 11 | 10 | 5 |
| 9 | 8 | 4 |
| 7 | 6 | 3 |
| 5 | 4 | 2 |
| 3 | 2 | 1 |

旧权重仍然是可用的架构探索结果，不应删除或改名成另一种架构；但不应再宣称它们是精确的 Table III 结构复现。此错误不单独解释同一隐藏层数下加宽后误差升高。

原论文使用欧拉角差，当前 Track-P 使用 rotation vector。试飞当前模型必须沿用其训练编码，不能为了形式贴合论文就把推理参数改成 euler。

### 2. 48 万独立查询并不意味着控制映射复杂

审查 `track_p_12d_contract.py` 与 `mpc_controller.py`：参考轨迹是固定 5 m/s 的门轴直线；双积分器预测、固定 Q/R、20 步、dt=0.05。控制增益预计算，不是每条样本重新求解含状态/输入约束的非线性优化问题。限幅发生在求出的控制量之后。

训练集所有门四元数都是 [1,0,0,0]。在此配置中，限幅前的期望加速度可化为：

```text
a_x = 2.1322388176 * (5 - v_x)
a_y = 2.0643291692 * e_y - 2.1322388176 * v_y
a_z = 2.0643291692 * e_z - 2.1322388176 * v_z
```

其中 e 是门中心减无人机位置；随后各分量裁剪至 [-6,6]，再经推力向量模长、小角度姿态映射、姿态误差反馈和角速度限幅生成四维控制。整个映射不是全局线性函数，但远比完整 NMPC 简单。

CPU 回放 256 条随机训练样本：

| 检查 | 最大输出绝对差 |
|---|---:|
| 重算教师 vs 已存标签 | 8.9271e-7 |
| 仅改变三个加速度输入 | 0 |
| 仅改变门前纵向距离 | 1.7764e-15 |
| 整体平移门与飞机 | 3.5527e-15 |

12 维中至少 4 维在当前教师下不影响输出；其余维度也有明显低阶结构。扩大无关维度取值、整体移动门、增加同类环境数量，不会自动增加映射复杂度。环境 ID 分离仍然有价值，但此处不是对 60 种不同控制规律的测试。

这不说明旧数据损坏。它说明旧数据适合模仿当前简单控制律，不适合用来证明大容量一定有益。

### 3. 同一检查点重评：优化和泛化都有问题

抽样种子 20261002；train 与 validation 各随机 4096 行；同一个 best checkpoint、eval 模式、相同归一化、CPU。没有重新读取 test。

| 隐藏 FC × 宽度 | Train normalized MSE | Validation normalized MSE |
|---|---:|---:|
| 4×128 | 9.1521e-6 | 9.5029e-6 |
| 4×1024 | 2.3458e-5 | 8.4273e-5 |
| 8×256 | 9.0796e-6 | 1.2021e-5 |
| 12×1024 | 5.5460e-5 | 1.4846e-4 |

大模型训练误差本身更高，不能只说过拟合；同时其验证/训练差距更大，不能只说优化没收敛。本表为诊断子集，不是替换已有完整 48000 行验证/测试指标。

当前是固定 Adam lr=1e-4，没有学习率衰减。12 个模型中 8 个达到 500 epoch 上限，有些最佳 epoch=492/498；确有进一步优化空间。12×1024 则在 357 轮因 patience=50 停止。直接把上限改成 20000，未必改变这种停止行为。新训练的模型最后 50 轮记录的 clip_fraction=0，不支持“训练后期一直被梯度裁剪卡住”的解释。普通 MLP 已有 Kaiming 初始化，也不是完全没做初始化。

每轮为 384000/512=750 次更新；500 轮约 375000 次更新。与论文比较时不能只比 epoch，要同时比较样本数、batch size、优化更新次数、墙钟时间和终止条件。

### 4. 其他既有分支不能忽略

`experiments/falcon_paper_fidelity_acado_v6_diverse40k/runs_full/capacity_summary.json` 中，同深度从宽度 512 增至 1024 的测试 normalized MSE 改善分别约 2.7%、9.0%、3.1%（depth 10/22/34）。因此不是所有分支都出现“大模型更差”。不过各运行预算并不完全相同，不能据此宣称严格容量定律。

`experiments/falcon_mpc_complete_300d_v1/audit_generalization/DIAGNOSIS.md` 已记录另一个问题：该分支同检查点完整训练/验证误差相差约 16.15 倍，绝对位置与参考位置高度相关，输入覆盖又缺少低速启动。因此不能把新方案简化为“直接改成 300D、样本更多即可”。不要混淆这些数据源与本次 12D 快速线性教师。

## 推荐下一步：区分三个可被否定的假设

1. **优化瓶颈**：先保持数据和结构不动，为小/中/大代表各建独立 warm-start 运行，从 best checkpoint 以 lr=1e-5 追加最多 300 轮；再根据验证集决定是否试 3e-6。小模型也给同样微调机会。原脚本支持 init_checkpoint，但不能直接改旧 config 后 --resume。保留微调前模型作为验证候选，以免越训越差。全部由用户执行，不自动启动。
2. **随机初始化造成深网络退化**：测试函数保持的扩宽/加深（Net2Net），或者冻结小模型并加一个末层零初始化的残差修正器 `u_small(x)+delta_u(x)`。先验证扩展前后输出一致，再继续拟合同一个 MPC 标签；不是仅用小模型伪标签代替 MPC。此方法保证初始函数不退化，不保证新测试集严格改善。它也不同于当前从随机参数重新训练整个 ResMLP。
3. **真实任务缺乏容量需求**：如果充分优化的小/大模型仍接近同一精度，接受小模型足够。若研究目标确实是更复杂控制，独立建立新任务：非线性姿态/推力耦合、优化内部约束、变曲率/爬升/制动参考、可行恢复状态和约束边界。先保证教师可靠再采样，不为制造难度而加入不可观测随机量。

更复杂教师优先复用并审查已有 ACADO/PAMPC 分支，不重写一个未经验证的求解器。参考公开实现：https://github.com/uzh-rpg/rpg_mpc 。函数保持扩展参考：https://arxiv.org/abs/1511.05641 。上述是候选实验，不是已经证实的新结果。

新数据仍可逐条独立采样，但应满足：

- 速度、姿态、控制历史和扰动状态满足所选动力学的可行性；不能只对输入加噪声却复用旧控制标签。
- 教师实际依赖的参考速度、未来轨迹、周期 T、模型参数/扰动估计，要么固定，要么由输入充分确定，要么显式加入输入。不要混入同 X 不同标签的隐藏条件。
- 对同一查询检查重复求解、收敛状态、约束违反、可行性及必要的冷/热启动差异。随机查询边界与短闭环恢复均应覆盖；历史相关 RTI 不能被假装成无记忆标签。
- 把普通跟踪、恢复、急转/制动、约束切换分层记录并预先定义评估权重；保持验证选型，测试不用于早停/调参；连续 rollout 的相关样本按来源组隔离。
- 固定深度的宽度扫描与固定宽度的深度扫描分开；先用 3 个 seed、3 个代表规模做试点，不直接训练几十个大模型。
- 最终评估整体 MSE、分输出/分状态族 MSE、尾部误差、闭环通过率与同设备端到端延迟。不要只保留有利于大网络的子集。

## AirSim：先试三个已训练模型

已导出到独立目录 `controller_architecture_study/deploy/scenic_flight_20261002`：4×128、8×256、12×1024。TorchScript 与原网络归一化输出比较最大差为 0；AirSim 实际运行包装器反归一化后的比较最大差不超过 9.54e-7。未重新训练。4×128 是 12 个候选中复用旧粗扫描的模型，所以原 checkpoint 位于 results 根目录，而不是 scenic_table3_v1 子目录。

先手动打开 UE4 Blocks 并点击 Play，只保留一个 AirSim RPC 实例；不要同时运行采集器或其他飞行脚本。下面连接手动打开的实例，没有 --auto-launch。每次测试会 reset 仿真、重新布置门，但不修改训练数据或权重。使用 Drone1；如果场景车辆命名不同，应调整 vehicle-name。

在 PowerShell 完整运行一次：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
$deploy = Join-Path $proj 'controller_architecture_study\deploy\scenic_flight_20261002'
$flight = @(
  '--device', 'cpu', '--controller-backend', 'cpu_dnn',
  '--input-target', 'gate', '--gate-attitude-encoding', 'rotation_vector',
  '--vehicle-name', 'Drone1',
  '--layout-root', '.\teacher_dataset_full_route_continuous_v2',
  '--initial-state-id', '0', '--epsilon', '0.8', '--seed', '0',
  '--speed', '5', '--per-gate-timeout', '30', '--gate-count', '5',
  '--x-start', '10', '--x-spacing', '10',
  '--gate-width', '4', '--gate-height', '4', '--gate-thickness', '0.15',
  '--gate-clearance', '0.25', '--gate-asset', '1M_Cube', '--keep-gates'
)
function Test-ScenicFlight {
  param([string]$Name)
  $modelDir = Join-Path $deploy $Name
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
  $report = Join-Path $proj "controller_architecture_study\results\scenic_flight_reports\${Name}_${stamp}.json"
  & $trainPy .\run_single_drone_dnn_gates.py @flight --model "$modelDir\controller_scripted.pt" --normalization "$modelDir\normalization.json" --report $report
  Write-Host "Report: $report"
}
```

随后分别执行，一次一个模型：

```powershell
Test-ScenicFlight 'plain_mlp_h4_w128_seed0'
Test-ScenicFlight 'plain_mlp_h8_w256_seed0'
Test-ScenicFlight 'plain_mlp_h12_w1024_seed0'
```

这是 DNN CPU 控制、UE4 实时渲染，不是训练或 FPGA 控制。模型的目标巡航速度为训练时固定的 5 m/s，不保证瞬时实测速度恒为 5 m/s；修改 --speed 不会让没有目标速度输入的旧模型学会新速度。目录 5ms 的语义是 5 m/s，实际控制周期仍为 50 ms。

通过一次五门飞行只能说明该场景闭环可运行，不代表其整体精度或鲁棒性最佳。记录 success、crossings、碰撞、控制周期 miss 及推理时间；慢模型超时属于此次实机 CPU 设置，不可直接当成 GPU/DPU 延迟。

## 复查诊断

```powershell
& $trainPy -m controller_architecture_study.audit_capacity_20261002 --report .\controller_architecture_study\results\scenic_capacity_audit_20261002.json
```

读取原数据和模型，仅写诊断报告。不连接 AirSim，不训练，不使用 test split。
