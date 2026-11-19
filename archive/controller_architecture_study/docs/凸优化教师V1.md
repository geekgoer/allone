# 稳定教师、独立状态数据与四种 DNN 容量对比

## 本次改了什么，没改什么

这是独立的新控制基线，不是把原 ACADO/PAMPC 原样复现。旧 ACADO 教师、12D 数据、300D NMPC 数据、已有模型都保持不动。

上轮诊断发现，当前 NMPC 某些邻域存在不同收敛解，偏航第一拍能从 +2 跳到 -2 rad/s。增加 epoch 不能保证学好这种分支。此次先改为严格凸的**平移加速度 QP + 显式四元数姿态反馈**：正定目标保证加速度优化解唯一；姿态控制不再通过非凸优化选择偏航分支。

这是为研究“可学控制律的容量—精度—时延关系”建立一个新基线，不能称为已经修复/验证原 NMPC，更不能把它的新 normalized MSE 与旧任务直接比较。

QP 仍有真实内部约束：每轴加速度限制、35° 倾角锥的保守八面近似、未来轨迹的软走廊约束；不是优化后单纯裁剪一个无约束 LQR。使用 20 步、预测步长 0.1 s，轨迹巡航速度 5 m/s，AirSim 控制周期 50 ms。软走廊允许 slack；**不代表硬避障或已证明安全**。倾角约束针对目标推力方向，不是实际无人机倾角的硬保证。

姿态内环跟踪 QP 优化后的目标推力姿态，不直接叠加规划器的角速度前馈：后者微分的是规划器自己的姿态，可能与优化后的姿态不同。300D 编码保留以便后续对照，其中测量加速度、当前角速度不进入标签；规划器角速度也不是本教师内环的附加命令。不能宣称全部 300 个坐标都是独立有效变量。

只执行第一个优化动作仍是正确的 MPC 用法。已提交的全局 C2 轨迹时钟每拍前进 50 ms，不会每拍从当前位置重新规划、把参考起点重新归零。静止启动也会得到前进控制。

原理依据：[OSQP 的凸 QP 问题形式](https://osqp.org/docs/solver/index.html)、[OSQP Python 接口](https://osqp.org/docs/interfaces/python.html)。唯一解依据本代码目标 Hessian 严格正定，不是仅凭求解器返回 solved。

## 文件与数据

代码：

- `controller_architecture_study/convex_teacher.py`：QP、物理控制输出与数值证书。
- `controller_architecture_study/convex_dataset.py`：小规模预检、独立查询、分块续跑、固定环境拆分。
- `controller_architecture_study/fly_convex_policy.py`：手动 Play 的教师/DNN 试飞。
- `controller_architecture_study/train_convex_capacity.py`：4 个宽度，固定 4 个隐藏层，串行训练、每 epoch 保存、自动续跑。
- `tests/test_convex_teacher.py`：解析预测矩阵、正定性、符号、重复性、平移不变性、软约束、续跑与损坏检测。

独立结果根目录：

```text
F:\FalconReproduction\airsim_control\controller_architecture_study\results\decision_convex_96k_v1
    preflight.json                数值预检，不是飞行证明
    teacher_acceptance.json       教师 AirSim 试飞报告
    manifest.json                 采集状态、配置、代码与分块哈希
    chunks\                       每条路线 80 个独立样本
    routes\                       精确保存的 C2 轨迹系数
    data_audit.json                数据数值、约束激活与各状态族审计
    dnn_dataset\
        train.npz                 76800 行，环境 0～47
        validation.npz             9600 行，环境 48～53
        test.npz                   9600 行，环境 54～59
        normalization.json        只拟合 Train
        split_manifest.json       来源与文件哈希
    runs\plain_mlp_h4_w{宽度}_seed{种子}\
        best_model.pt             Validation 最好的模型
        last_model.pt             最近 epoch，包含优化器和随机数状态
        config.json
        train_log.csv
        completion.json
        offline_metrics.json      同一个 best 检查点的 Train/Validation 指标
    runs\comparison.json          单模型、逐族 gap、多 seed 均值/标准差
```

总共 60 个环境组 × 20 条路线 × 80 次独立状态查询 = 96000 行。沿用已有 96k 实验的路线生成方式与环境编号划分，但**重新采样全部状态，重新计算全部标签**。没有读取旧 Validation 的失败状态来补 Train，也没有随机拆散同一环境。

每个环境、每条路线，八个状态族数量相同：启动、跟踪、门间过渡、制动、位置恢复、速度恢复、姿态恢复、组合恢复。每次查询独立选择轨迹相位与状态；不是学习连续五门飞行流程。恢复扰动最大位置 `[±0.6, ±1, ±0.6] m`、速度 `±1.5 m/s`、旋转向量 `±0.25 rad`，并加入密集内部尺度。实际速度上限 8 m/s；巡航参考是 5 m/s，但参考窗口包含真实起步、转弯和制动的速度/加速度变化。25% 启动查询选择相位零、实际速度零。当前测量加速度不是本教师反馈状态，不添加无效噪声。

输入保持 300D 在线编码：当前局部状态 10D + 21 点未来位置/姿态/相对速度 210D + 20 点规划器推力/角速度前馈 80D。坐标系为 Z-up/FLU；标签 `[F,wx,wy,wz]` 为 AirSim NED/FRD 对应物理命令。**未来前馈来自规划器，不含优化后的教师标签**。编码代码沿用 `build_features()`，合同单独命名为 `decision_convex_local_relative_pv_300d_v1`，不混用旧模型。

## 1. PowerShell 初始化

逐行复制，不要复制提示符 `PS ...>` 或 `>>`：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Test-Path -LiteralPath $trainPy
```

不要把 AirSim 虚拟环境的整个 `site-packages` 放进 `PYTHONPATH`，否则可能把 NumPy 1.26 换成 2.4、使 SciPy 1.10 出错。新试飞入口只单独加载 AirSim、msgpack、msgpackrpc 和它们匹配的 Tornado，不改变训练环境的 NumPy/SciPy/PyTorch 文件。

本机已安装项目独立目录中的 OSQP 1.0.5，原 PyTorch 环境未修改。只有换机器、该目录不存在时才需要：

```powershell
& $trainPy -m pip install --no-deps --only-binary=:all: --target .\controller_architecture_study\vendor\convex_qp osqp==1.0.5
```

本机已验证 NumPy 1.26.4、SciPy 1.10.1、PyTorch 2.4.1+cu121。若迁移 Linux，不能使用 Windows 编译的 vendor 文件；在 Linux 环境安装 OSQP 1.0.5、SciPy，加载器会使用 Linux 自身的包。

## 2. 教师预检与试飞（先做这一步）

```powershell
& $trainPy -m controller_architecture_study.convex_dataset preflight
```

数值预检需要 `status=PASS`。检查相同输入重复输出、先解邻近问题再回到原问题、1 cm 位置扰动、约束残差；仅使用 Train 环境。它不是“全部状态平滑”的证明。

用 PowerShell 启动现有编辑器，再手动点击 Play；若已经开着就不用再次启动：

```powershell
& 'E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe' 'F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject'
```

在 PowerShell 执行教师试飞，路线 3 包含横向变化、高度变化与非零航向：

```powershell
& $trainPy -m controller_architecture_study.fly_convex_policy --environment-id 48 --route-index 3 --keep-gates
```

不传 checkpoint 就是 **QP 教师控制，不是 DNN**。不会启动另一套 Unreal，也不会生成训练数据。无人机会被放到空中 `z≈-4 m` 的静止起点，再开始飞，不是从地面执行 takeoff。

需要 `status=PASS`、`valid_crossings=5`、没有碰撞/状态异常；同时检查门内穿越、速度、位置误差和 deadline misses。如果失败，先将完整报告返回，不要继续大规模采集。完整采集程序会检查本次配置匹配的教师试飞 PASS，避免“还未证明能飞就先采集训练”。这一次试飞只证明该路线，不能当作全面鲁棒性证明。

## 3. 独立状态数据生成和准备

教师通过后可以关闭 Unreal。数据生成是离线求解，不需要 AirSim：

```powershell
& $trainPy -m controller_architecture_study.convex_dataset generate --resume
& $trainPy -m controller_architecture_study.convex_dataset prepare
```

必须前一条正常结束再执行下一条。采集中 Ctrl+C、断电后，再执行相同 `generate --resume` 即可：已记录完整且哈希正确的路线直接跳过；中断的当前 80 行分块重新生成。不重复追加已完成数据。未收敛查询会保存 `failure.json` 并停止，**不换一个更容易的状态来掩盖失败**。

`failure.json` 是历史失败记录；恢复完成状态以 `manifest.json.complete=true` 为准。改变教师代码、采样、数值库版本会拒绝混合续跑，需要新目录；不应为了续跑删除原 manifest。准备阶段要求数据全齐，检查行数、环境划分、标签数值证书，计算 Train-only 归一化。

准备后确认：

```powershell
Get-Content -Raw -LiteralPath '.\controller_architecture_study\results\decision_convex_96k_v1\data_audit.json'
```

需要 `status=PASS`，行数 `96000`，split 数 `76800/9600/9600`。约束激活比例说明任务实际有多少受限决策；不能单凭模型训练 loss 判断数据合理。

## 4. 四个容量模型先跑 seed 0

不用旧检查点，四个模型全部从头训练；固定四个隐藏层，宽度 `64、128、256、1024`。这样先隔离宽度影响，不混入“大模型又更深”的优化困难。

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -m controller_architecture_study.train_convex_capacity --seeds 0 --epochs 400 --device cuda --dry-run
& $trainPy -m controller_architecture_study.train_convex_capacity --seeds 0 --epochs 400 --device cuda
```

先确认 dry-run 列出四个正确目录，再执行下一条。单进程/单模型串行，上一模型结束立即保存完整产物并退出释放显存；不等待四个全部结束才保存。每个 epoch 原子保存 `last_model.pt`；Ctrl+C、断电、OOM 后重新运行相同命令会跳过已完成模型，继续未完成模型。当前 epoch 会重跑，之前完整 epoch 不丢失。串行降低显存压力，但不保证解决显卡驱动/TDR 故障。

默认最多 400 epoch，最少 150 epoch，连续 100 epoch 没改善时早停，学习率余弦衰减 `1e-4 → 1e-5`，Adam L2 `1e-5`，batch 512。所有模型同一配方；大模型不保证需要同样训练时间，也不保证 Validation 严格单调。

已有运行恢复时不要改变 `--epochs`；学习率时间轴和配置会变化，程序会拒绝默默续跑。若要独立新预算，请用新实验目录/明确的新训练配置，不要覆盖旧报告。

## 5. 看结果与纯 DNN 试飞

```powershell
$comparison = Get-Content -Raw -LiteralPath '.\controller_architecture_study\results\decision_convex_96k_v1\runs\comparison.json' | ConvertFrom-Json
$comparison.models | Select-Object model,status,parameter_count,train_mse,validation_mse,gap_absolute,validation_train_ratio,best_epoch | Format-Table -AutoSize
```

四个模型使用各自同一 best 检查点、eval 模式、相同归一化评估 Train/Validation。报告同时含逐输出 RMSE、p99 绝对误差以及八种状态族的 gap。总体 gap 缩小但恢复族仍很差，不算已经解决。比值在 Train 很小的时候会显得很大，要同时看绝对差、物理误差及闭环。

训练结束后打开同一个编辑器、点击 Play，先试一个模型：

```powershell
& $trainPy -m controller_architecture_study.fly_convex_policy --checkpoint .\controller_architecture_study\results\decision_convex_96k_v1\runs\plain_mlp_h4_w256_seed0\best_model.pt --environment-id 48 --route-index 3 --keep-gates
```

这里是纯 DNN 输出，没有 MPC 回退；路径规划器和教师试飞是同一个全局 C2 轨迹。要测试其他宽度，只替换 checkpoint 的 `w256`，不要换归一化。报告分别放在每个模型自己的 `flight` 中。

## 6. 加 seed 1、2，不以一次结果下结论

第一组结果和飞行合理后：

```powershell
& $trainPy -m controller_architecture_study.train_convex_capacity --seeds 0 1 2 --epochs 400 --device cuda
```

seed 0 已完成会跳过；只训练剩余八个。`comparison.json` 保留全部完成种子并统计每个宽度的 Validation 均值及样本标准差。不要挑一个恰好让大模型最好看的 seed。

最后再冻结架构与配方，单独做 Test 评估；此轮训练入口**不读取 Test 样本，也不按 Test 选模型/早停**。生成 Test 文件、数据完整性检查不等于用 Test 做模型选择。

## 完成标准与边界

### 可选：扩展为 SCENIC 12 个结构 + FALCON 3 个结构

2026-10-03 新增 `--architecture-set scenic_falcon15`，默认四宽度队列不变。这里只借用网络规模，仍是同一份 **300D 凸 QP 教师数据**，不是原论文 12D 数据或论文损失的精确复现。不会读取旧 12D 模型权重。

SCENIC 使用已有 `configs/scenic_table3_v2_layercount_corrected.json`，不使用早期 v1 的错误层数映射：该配置将论文 FC/ReLU 总层数映射为 `hidden_layers=(L-1)/2`。12 个隐藏层数×宽度为：

```text
6×1024  5×1024  5×512  4×1024  4×512  4×256
3×1024  3×256   3×128  2×1024  2×128  1×1024
```

FALCON 三个模型选择项目已有的 **总 FC 深度 10/22/34、宽度 512** 口径，对应 `9×512、21×512、33×512` 隐藏层结构。FALCON 这个 `depth` 口径与上述 SCENIC 的 FC/ReLU 总层数口径不同，报告统一列出实际隐藏层数、参数量和 MACs，不混淆二者。

```powershell
& $trainPy -m controller_architecture_study.train_convex_capacity --architecture-set scenic_falcon15 --seeds 0 --epochs 400 --device cuda
```

seed 0 共 15 个结构，按参数量从小到大串行执行；协议仍为最多 400 epoch、最少 150、patience 100、batch 512、余弦学习率和同一正则化配方。大深度模型更慢，不能保证 400 epoch 足够优化，亦不能保证 Validation 单调改善。每个模型立即保存，失败停止队列；重跑同一命令自动恢复未完成模型、跳过已完成模型。

数据生成/准备命令不变。原四宽度队列中 `4×256、4×1024` 若已经训练过，配置完全相同，可直接恢复/复用；其余旧四宽度模型不计入这次 15 个结构。结果目录仍为 `runs/plain_mlp_h{隐藏层数}_w{宽度}_seed0`。

独立汇总为 `runs/comparison_scenic_falcon15.json`，不会覆盖原 `comparison.json`。新报告按 **隐藏层数和宽度二元组** 统计种子均值，不把不同深度但相同宽度的网络混为一组。若以后使用 `--seeds 0 1 2`，则是 15 种结构 × 3 个种子 = 45 个训练运行，不再是 15 个。

本次代理只完成代码与有界离线检查，没有运行正式 96k 生成、GPU 训练或真实 AirSim 飞行。已通过 24 项相关测试（含新教师测试与旧入口回归）；64 组重复/邻域预检 PASS，重复差为零，最大约束违反约 `1.05e-5`，平均基础查询约 `1.57 ms`。另外 640 条随机 QP 查询全部收敛，平均约 `1.50 ms`，p95 约 `2.22 ms`，观察到的最大值约 `44.94 ms`。这些有限样本的结果不代表所有查询的最坏时延或飞行成功率。

1. 教师数值预检和真实 AirSim 试飞通过，静止起步正常；否则停止大规模生成。
2. 96k 数据完整、划分不变、状态族平衡、标签每条重新计算，续跑和完整性检查通过。
3. 四个网络均报告真实 Train/Validation 误差和逐族 gap，多 seed 复核容量趋势。
4. 纯 DNN 闭环也正常；正常穿门不代替恢复状态测试。

**不能预先保证“大模型的 Validation loss 必定更低”。** 本实验消除已定位的非凸偏航分支源、增加启动/组合恢复内部覆盖，给容量比较一个更明确的前提。如果大模型只降低 Train，而 Validation 仍不改善，应据实继续检查覆盖、优化和归纳偏置，不能删困难验证点、随机换 split、用 Test 调参或改报表来制造单调结果。
