# M2 v3：共享可学习步长与 dyadic 整数尺度（2026-10-14）

本版是在 v2 的 300 轮结果之上适配量化器，正式训练由用户执行。
初始化：v2 的 best_model.pt，epoch 290，SHA256
`dafbf8353fc1533c0f01ff62fce4bb05000363632193ee504231d1eabc77ed30`。
冻结参考：原 v4 FP32 best_model.pt，epoch 4983，SHA256
`d9b9c0fd7ffb394f1c11099056e38308d6ef685d9cc99d01e3cd47698f3b2dca`。
v2 中 residual alpha 已折入权重；加载时不会二次折叠。

输出目录：`controller_architecture_study/results/incremental_supernet_237d_m2_lsq_v3`。
输入、数据、环境拆分、归一化、全部 136 个配置保持原契约；不读取测试集。
本阶段不需要 AirSim、板子或 Vivado。

## 算法及训练步骤

1. 校验 FP32、v2 初始化检查点、数据 SHA256 及模型契约。
2. TRAIN-only 8192 行校准共享 A16 激活尺度；激活尺度一直固定。
3. 加载 v2 的骨干和输出头浮点训练权重，重新初始化逐输出通道权重步长。
4. 第 1–100 轮：联合更新网络权重和 log(step)。骨干 W8、输出头 W16。
5. 第 101 轮开始：在 CPU FP64 中编码并冻结各通道的 uint16 整数乘数；继续微调到第 300 轮。
6. 只允许冻结尺度的检查点成为导出候选：冻结初始候选 epoch 0，或第 101–300 轮。
7. 最后用 validation 9600 行、全部 136 配置进行真整数验证、导出及增量续算检查。

采用 [LSQ](https://arxiv.org/abs/1902.08153) 风格的 STE 和步长梯度缩放，
但这是适配本因果超网的逐通道实现，不声称完整复现原论文。
log(step) 梯度经 `1/sqrt(有效行长度 * 正整数上限)` 缩放；因果块只计有效前缀长度。
逐通道权重步长编码为 `s = M * 2**e`，M 为 uint16 正整数，e 为该通道固定的基准指数。
初始 M 约在 [16384,32768)，学习期间用 STE 将 M 限制到 [1,65535] 并舍入。
因此从第一轮开始就模拟最终的乘数网格，不在导出时才突然把任意尺度投影到整数。
冻结后的尺度缓冲区不再依赖 CUDA/CPU exp 的浮点舍入差异。

这借鉴了 [HAWQ-V3](https://proceedings.mlr.press/v139/yao21a.html) 的 dyadic 硬件尺度思想，
但未实现其 Hessian/ILP 混合精度搜索。没有引入每配置单独尺度、在线校准或 136 个独立模型。

默认预算与初始学习率：

| 项目 | 设置 |
| --- | --- |
| 总轮次 | 300，无 early stopping |
| 学习尺度阶段 | 前 100 轮 |
| 冻结尺度阶段 | 后 200 轮 |
| batch | 512 |
| 骨干＋最大深度输出头 LR | 3e-7 |
| 其他输出头 LR | 1e-6 |
| log(step) LR | 3e-4；冻结后停止更新 |
| LR 衰减 | 余弦到初值的 0.1 |
| weight decay | 0 |

保留 MPC 标签、共同最大 FP32 教师、逐配置 FP32 蒸馏。
相邻配置差异保持项权重为 2，最大配置标签项权重为 2、最大配置教师项为 1。
这比 v2 更强调最大配置准确性和配置关系保持，不能保证门槛必过。
训练行独立，不伪造时序样本；配置关系保持不等于已证明闭环震荡减少。

## 整数接口与增量计算

契约：`causal_w8_head16_a16_lsq_dyadic_v3`。
交付 `weights_mixed.npz` / `weights_mixed.bin`，并新增每层逐通道 `*_weight_multiplier` uint16 数组。

骨干运算：

```
raw_mac = dot(A16_codes, W8_codes) + bias_code        # 48-bit
scaled_branch = raw_mac * M                          # checked signed 63-bit working range
branch unit = 2 ** (input_activation_exponent + weight_base_exponent)
align branch and identity -> add -> ReLU -> RNE -> A16 saturation once
```

head 运算：各通道组 W16/A16 原始 MAC 缓存并累计，加 bias 一次，最后乘 M 一次。
golden 的 `outputs_accumulator_int64` 包含乘 M 后的整数值，`output_exponents` 为相应基准单位指数。
读出用 FP64 小矩阵累加模拟，避免 W16×16 位尺度乘数重建权重时被 FP32 有效位数截断。
训练的大骨干 fake-quant 点积仍是 FP32。导出使用 FP64 模拟检查原 0.0005 物理输出容差，
并使用另一套独立 PyTorch 整数实现逐位核对 NumPy 整数累加器。
训练 FP32 模拟与整数的差异另行披露，不与 FP64 验收值混用。
部署参考运算全部为整数；PS 归一化和输出反归一化保持 float32。

执行器在运算前检查 MAC、乘数乘积及残差对齐的保守范围，覆盖所有 A16 编码输入。
48 位 MAC 安全不代表新增乘法天然安全；若界限不满足，导出停止。
PL 需要额外的整数乘数及移位接口，此版尚未综合或测试 FPGA 时延。

同一观测、同一模型版本的共享尺度与因果前缀才允许复用。
增深只补后续层，增宽只补后续组；整数乘数不随配置改变。
直接计算与续算必须逐位相同，已缓存前缀必须不变。

## 从上到下运行

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_incremental_supernet_m2_lsq.ps1 -Python $trainPy -Device cuda -BatchSize 512
if ($LASTEXITCODE -ne 0) { throw 'M2 v3 未全部通过；检查点和报告已保留，请查看末尾输出。' }
```

中断后重跑同一段，从最后完整 epoch 恢复；尚未完成的 epoch 会重放。
冻结阶段中断也会恢复相同整数乘数，不重新校准。
遇到明确 CUDA OOM 可改 BatchSize 为 256；记录变化，不保证相同训练轨迹。
初始化后不要修改配置 JSON。export 中断后重跑只重新导出，不重训已完成阶段。

`latest_checkpoint.pt` 保留优化器、尺度状态、RNG 和历史；`best_model.pt` 为冻结候选；
`last_model.pt` 为第 300 轮。best 为 epoch 0 时也属于合法初始候选，并非未训练。
最终 export 目录包含整数包、规范、归一化、TRAIN 校准记录、golden、全配置离线指标及报告。
原有三个质量门槛保持：最大/网格 MSE 不超过 FP32 1.1 倍，固定可信掩码的切换差不增加。
数值通过但质量未过仍保留产物、返回质量未通过；不能声称板上或闭环验收完成。

## 本次代码检查

29 项 CPU 检查通过，包括旧版兼容、步长梯度、冻结后编码不变、位宽溢出检查、
整数直接/增量一致性、打包/二进制布局及恢复。
新增微型训练检查从旧 v2 格式初始化，分别在学习阶段和冻结阶段中断恢复，
最终全部参数/缓冲区与连续运行逐位一致，源文件 hash 保持不变。
真实 v2 模型的 8 条 validation 输入、136 配置只读检查通过：
fake/integer 最大物理输出差 0.0003111693，两条续算路径逐位一致，
全 A16 输入的保守 MAC/乘积/残差界通过。此初始包有 10 个被裁剪的权重，正式报告会继续披露。
以上不是正式 300 轮训练或全量质量结果。

## 2026-10-15：训练完成后的导出数值修复

用户完成 300 轮，最佳 epoch 为 300。已完成的 9600 行真整数指标：
最大配置 MSE=0.000180601990233（比 FP32 高约 2.31%，通过），
网格 MSE=0.001486217681938（比 FP32 低约 4.24%，通过），
同输入切换差 MSE=0.0000504042148032（高约 0.99%，原“不增加”目标尚未通过）。

原导出在 golden 的 FP32 模拟检查处中断，差值 0.00133348139934。
定位到：6 个输出头共 8 个权重在 FP32 除法的半整数边界上选出了与 FP64 打包不同的整数码；
FP32 点积从第一层起有少量 A16 编码偏差，沿残差层传递。
这不改变已完成的真整数全量指标，但说明 FP32 QAT surrogate 不能用于逐位部署参考。

修复仅作用于导出验证，不改变训练输出、检查点、部署整数运算或质量门槛：

- 将同一冻结模型的离线模拟提升为 FP64，继续检查原 0.0005 容差。
- 新增 `torch_integer.py`，从原始检查点参数独立生成标准整数码，
  MAC 用受 48 位界限约束的 FP64 精确整数点积，乘数、对齐、残差和 RNE 用受检查的 INT64。
  不从 NumPy 打包数组读取权重，避免自我比较。
- 逐位比较 PyTorch / NumPy 的所有 golden 累加器，并比较标准化输出。
- 原 FP32 偏差保留在 `qat_float32_golden_max_abs_physical_error`，
  `qat_surrogate_within_tolerance` 会如实为 false；不自动放宽原容差。
- `fake_quant_emulation_dtype=float64` 明确验收模拟的精度。
- golden 增加独立 PyTorch 累加器/输出，以及原 FP32 模拟输出。

报错的同一组 32 行、136 配置已只读检查：独立整数累加器逐位相同，FP64 模拟误差为 0。
31 项 CPU 检查通过，新增权重除法舍入边界和独立数值参考的回归用例。
FP64 仅用于软件校验，不意味着在 PL 上使用浮点乘法。

现在只重跑导出，无需重训（这段会重新进行完整整数验证）：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -m controller_architecture_study.incremental_supernet.quantize export --config .\controller_architecture_study\configs\incremental_supernet_237d_m2_lsq_v3.json --device cpu --golden-count 32 --integer-batch-size 32
if ($LASTEXITCODE -eq 2) {
    Write-Warning '整数数值检查通过，但严格质量目标仍未全部通过；请查看 export_report.json。'
} elseif ($LASTEXITCODE -ne 0) {
    throw '导出未完成，请保留末尾报错。'
}
```

预计仍会因原严格切换门槛而得到 `NUMERICS_PASS_QUALITY_NOT_MET`；这是已知约 0.99% 的切换差增加，
与本次浮点数值不一致报错不同。当前不据此宣称飞行震荡改善或 FPGA 验收完成。
