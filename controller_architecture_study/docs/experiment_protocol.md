# M1 实验协议

## 固定数据与输入输出

- 数据：`experiments/paper_track_p_12d_5ms_v1/dnn_dataset`，`train.npz` 384000 条、`validation.npz` 48000 条、`test.npz` 48000 条；沿用现成的 environment_id 拆分，不重新切分。
- 输入 12D：`[p_gate-p_vehicle, rotation_error, velocity, acceleration]`，AirSim NED；输出 4D：`[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]`。
- 读取数据集原有 `normalization.json`；训练输入和标签分别做 `(x-x_mean)/x_std`、`(u-u_mean)/u_std`；预测物理控制量是 `u_pred_norm*u_std+u_mean`。
- 新模型和六个旧模型的训练损失均为四维归一化输出的均方误差。只用验证集选最优 checkpoint/early stopping，test 只在训练结束评估。
- 六个旧模型文件不修改、不再训练；`results/falcon_baseline.csv` 引用原始 checkpoint 和原始 test 报告。归一化 JSON 的逐项差异不超过 `2e-7`，属保存精度差异；基线冻结的容差为 `1e-6`。

## 架构定义和复杂度

新框架的 `hidden_layers` 是真正的隐藏 FC 数；旧框架 `depth` 是 FC 总数，故旧 depth22 对应新 `hidden_layers=21`。所有模型为 `Linear+ReLU` 重复、末层 `Linear`；无残差。MAC 是 batch=1 时所有 FC 权重乘加数，Params 另外包含偏置；这不是 GPU 或 FPGA 延迟测量。

## 统一新模型协议

所有 11 个候选由 `configs/coarse_sweep.json` 控制：seed 0、Adam (`foreach=False`)、初始学习率固定 `1e-4`、batch 512、最多 500 epoch、至少 100 epoch、验证 normalized MSE 连续 50 epoch 未改善则停止、gradient clip norm 1.0。隐藏层 Kaiming ReLU 初始化，输出层 Xavier 初始化。每架构单独保存 best、last、日志、验证曲线、验证/测试指标与复杂度；运行器串行执行以避免显存争用。中断后用同一命令续训，last checkpoint 保存优化器和随机数状态。

## 与现有六模型不可忽略的差异

现有六模型并非同一受控训练实验：seed 有 123 和 42；训练上限分别约 100、400、300、600、500，以及最后一个从已有权重续训到 744 epoch；最后一个续训学习率为 `5e-5`，其余为 `1e-4`。原训练的 early-stopping 设置也不一致。故同一图上的六个旧点是外部参考，不能把新旧单点误差差异全归因于架构。新 11 点使用一致协议，适合比较架构趋势。每个新架构暂时只有一个 seed；最终候选需补三 seed 和闭环验证。

## 验收边界

M1 只测控制器规模与离线指标；不做 SuperNet、量化、DPU、FPGA 或调度器。离线误差低不代表闭环安全。候选 saturation region、3–5 个 AirSim 闭环候选必须在 11 模型结果出来后基于真实 MAC–误差曲线确定。
