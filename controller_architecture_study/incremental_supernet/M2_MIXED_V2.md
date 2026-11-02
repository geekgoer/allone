# M2 v2：W8 骨干、W16 输出头、A16 与 FP32 配置关系保持

本版从冻结 v4 FP32 的 epoch 4983 重新初始化，输出到独立目录
`controller_architecture_study/results/incremental_supernet_237d_m2_mixed_v2`。
保留旧 M2 实验、原数据/拆分、归一化和全部 136 个配置。不需要 AirSim、板子或 Vivado。

## 修改依据与边界

旧 M2 100 轮已完成，但全量整数验证没有通过原质量门槛。
在 512 条按 family 分层选择的验证样本上，从 FP32 出发的只读误差分解显示：
仅将输出头变为 W8，使最大配置 MSE 从 8.9063e-5 升至 1.04745e-3，
同输入相邻配置差异 MSE 从 5.0849e-5 升至 5.14023e-4。
仅 A16 激活的对应指标为 9.8139e-5 和 5.1213e-5。
这用于定位输出头敏感性，不是完整验证，也不证明骨干 W8 已达到质量要求。

17 个输出头保留 W16，裸权重比 W8 多 17×4×1024=69632 字节（68 KiB）。
PL 输出头需要 A16×W16 乘法，不能用 W8 乘法器接口静默截断。
有效骨干仍为 W8；所有层/配置的尺度共享并在校准后冻结。
权重尺度的安全余量从 1.25 改为 1.0，激活余量仍为 1.25。
权重舍入仍使用每输出通道 2 的幂尺度，整数平台不增加任意浮点乘法。

## 训练步骤

1. 校验冻结 FP32 和训练/验证数据的 SHA256；不读取测试集。
2. 仅用 TRAIN 的 8192 行校准共享 A16 尺度，固定权重逐通道尺度。
3. 固定 300 epoch，无 early stopping；batch 默认 512。
   骨干及最大深度输出头学习率 3e-7，其他输出头 1e-6，余弦下降到初值的 0.1；weight decay=0。
4. 每个 batch 在冻结 FP32 上一次计算全部配置，得到对应教师输出；冻结教师不更新、不折叠权重。
   QAT 模型另行复制并折叠 residual alpha，避免教师被初始化过程修改。
5. 每轮保存最新训练状态，按验证质量选 best，全部完成保存 last。
6. 最后集中运行 9600 行×136 配置的真整数验证、导出和直接/增量一致性检查。

设量化输出为 q_c，冻结同配置 FP32 输出为 f_c，原 MPC 标签为 u，最大配置为 M。
所有 MSE 在标准化输出空间计算；损失为：

```
mean_c MSE(q_c, u)
+ 0.1 * mean_c MSE(q_c, f_M)
+ 2.0 * mean_c MSE(q_c, f_c)
+ 1.0 * mean_(a,b) MSE(q_a - q_b, f_a - f_b)
+ 1.0 * MSE(q_M, u)
+ 0.25 * MSE(q_M, f_M)
```

使用全部 247 条相邻增深/增宽边，不再使用随 QAT 误差变化的训练门控。
切换差约束保留 FP32 必要的配置差异，不要求所有配置输出相同。
两端 QAT 输出都接收梯度；FP32 参考完全 detach。
原有原始切换差验收门槛不变，同时增加 `extra_switch_mse_vs_fp32` 以显示量化额外引入的差异。

## 运行与恢复

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_incremental_supernet_m2_mixed.ps1 -Python $trainPy -Device cuda -BatchSize 512
if ($LASTEXITCODE -ne 0) { throw 'M2 v2 未全部通过；检查点和报告已保留，请查看末尾输出。' }
```

重跑相同命令从上一完整 epoch 恢复。若导出中断，重跑重新导出，不重训完成的 300 轮。
仅在明确 CUDA OOM 时，可将 `-BatchSize 512` 改为 `-BatchSize 256` 再运行；
更换 batch 会记录，训练轨迹不保证逐位相同。不要在初始化实验后修改配置文件。
训练可能选择 epoch 0 为 best，这是合法的量化初始候选，并不意味着未运行全部轮次。

## 整数交付和下一步

新版契约为 `causal_w8_head16_a16_pow2_rne_v2`。
权重使用 `weights_mixed.npz` / `weights_mixed.bin`，输出头数组 dtype=int16，其他权重为 int8。
`quantization_spec.json` 明确声明权重文件、每个数组的 dtype、偏移及 head_weight_bits=16。
旧版 W8 包和检查点仍可以用原配置读取/恢复。
MAC/bias 仍为有符号 48 位：1024 项 W16A16 极值乘加约需 41 个有符号位，
执行器在加偏置前检查保守界。残差对齐使用受检查的 INT64 工作量。
同一样本、同一模型/量化版本才能复用缓存；增深/增宽不能改动已缓存前缀。

export 包含：混合整数权重、量化规范、归一化、TRAIN 校准清单、精确累加器 golden vectors、
全配置/分 family 离线指标、切换差异报告、export_report.json。
数值校验通过但质量不通过仍返回 `NUMERICS_PASS_QUALITY_NOT_MET`，所有文件保留。
最大配置/平均 MSE 不超过 FP32 1.1 倍，固定可信样本切换差不超过 FP32，才报告质量 PASS。
本次不会通过减少配置数量或放宽阈值让结果通过。

质量通过后固定整数契约，进入 M3 HLS/PL 实现。FPGA 实际时延、闭环震荡改善和独立模型池对照仍未验证。
配置差异保持并不保证稳定性，也不等价于时序控制平滑；当前数据行是独立样本。

## 本次实现检查（2026-10-14）

20 项 CPU 检查通过，覆盖旧版 W8 与新版混合精度的数值规则、增量缓存、
损失梯度方向、2 轮微型训练的中断恢复逐位一致、导出二进制布局和源文件保护。
另在真实 v4 模型的 8 条验证输入、全部 136 个配置上只读检查：
fake quant / 整数的物理输出最大绝对差为 0.0002835813（容差 0.0005），
两条增深/增宽续算路径与直接整数执行逐位一致，权重裁剪数为 0。
这不是新版 300 轮训练或全量质量验收；正式结果由上述命令产生。
