# 各阶段 DNN 训练记录汇总

读取日期：2026-09-18。来源：各模型目录的 metrics.json、history.csv、normalization.json；未启动训练或重新评估模型。排除 smoke 功能测试，包含47次正式或中断运行。

口径：选择 metrics.json 的 best_epoch；缺少完成指标文件时取现存 history 中验证指标最优轮次。Train 是该轮在线训练的 normalized MSE，Validation 是该轮末的 normalized MSE，二者不是同一冻结权重的双集合重评。标记 † 的旧模型验证 normalized MSE 根据逐输出 physical MSE / u_std² 后取四输出平均还原（近似）；未保存足够信息者留空。不能从不同数据集的绝对loss直接判断模型优劣。

完成判据为存在最终 metrics.json；“仅检查点”不意味着当前进程仍在运行。best/last 表示选中轮次/现存历史末轮。300D原诊断epoch332和这里最新完成模型epoch383不同，故数值不同。

## 早期12D连续轨迹

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/dnn_runs/depth10_width512/metrics.json) | 953/1153 | 1.904141e-4 | 7.755614e-4 † | 5.198955e-6 | 已生成最终产物 |

## 12D robust_v1

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/dnn_runs/robust_v1_depth10_width512/metrics.json) | 259/459 | 1.077197e-3 | 4.378818e-3 † | 3.936518e-5 | 已生成最终产物 |

## 25D iid_path_v1

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/dnn_runs/iid_path_v1_depth10_width512/metrics.json) | 1564/1764 | 2.000494e-5 | 1.116072e-5 † | 1.283295e-5 | 已生成最终产物 |

## 12D track_p_robust_v1

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/dnn_runs/track_p_12d_robust_v1_depth10_width512/metrics.json) | 849/999 | 5.113946e-5 | 1.313843e-5 † | 2.996718e-5 | 已生成最终产物 |

## 历史5m/s六模型（后续部署采用的版本）

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth10_width512/metrics.json) | 1083/1283 | 5.194980e-5 | 1.745576e-5 † | 5.435930e-5 | 已生成最终产物 |
| [10x1024](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth10_width1024/metrics.json) | 1092/1292 | 3.772653e-5 | 1.430193e-5 † | 4.909140e-5 | 已生成最终产物 |
| [22x512](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth22_width512/metrics.json) | 1706/1906 | 7.634058e-5 | 2.310879e-5 † | 8.202509e-5 | 已生成最终产物 |
| [22x1024](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs_server/paper_track_p_12d_5ms_v1_server_depth22_width1024/metrics.json) | 1721/1921 | 3.825869e-5 | 3.892602e-4 † | 1.463503e-3 | 已生成最终产物 |
| [34x512](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs_server/paper_track_p_12d_5ms_v1_server_depth34_width512/metrics.json) | 3430/3630 | 4.614806e-5 | 4.637859e-4 † | 1.714889e-3 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs_server/paper_track_p_12d_5ms_v1_server_depth34_width1024/metrics.json) | 664/864 | 3.301286e-4 | 5.345212e-4 † | 2.013379e-3 | 已生成最终产物 |

## ACADO v3：连续40k

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v3_continuous/runs_paper_gate_12d_euler/depth10_width512/metrics.json) | 15114/20000 | 1.094360e-4 | 3.490802e-3 † | 1.267586e-3 | 已生成最终产物 |
| [10x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v3_continuous/runs_paper_gate_12d_euler/depth10_width1024/metrics.json) | 947/2701 | 5.508021e-4 | 3.264821e-3 † | 1.264444e-3 | 已生成最终产物 |
| [22x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v3_continuous/runs_paper_gate_12d_euler/depth22_width512/metrics.json) | 1709/3463 | 6.084189e-4 | 3.374627e-3 † | 1.361085e-3 | 已生成最终产物 |
| [22x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v3_continuous/runs_paper_gate_12d_euler/depth22_width1024/metrics.json) | 806/2140 | 3.363791e-4 | 3.275719e-3 † | 1.228484e-3 | 已生成最终产物 |
| [34x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v3_continuous/runs_paper_gate_12d_euler/depth34_width512/metrics.json) | 1565/2885 | 2.345121e-4 | 3.664106e-3 † | 1.419671e-3 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v3_continuous/runs_paper_gate_12d_euler/depth34_width1024/metrics.json) | 1348/3102 | 6.032248e-4 | 3.410264e-3 † | 1.263226e-3 | 已生成最终产物 |

## ACADO v4：独立120k

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v4_independent/runs_balanced_120k_seed0/seed_0/depth10_width512/metrics.json) | 489/500 | 3.535275e-4 | 1.750329e-2 | 2.186188e-2 | 已生成最终产物 |
| [10x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v4_independent/runs_balanced_120k_seed0/seed_0/depth10_width1024/metrics.json) | 496/500 | 3.583039e-4 | 1.474286e-2 | 1.808500e-2 | 已生成最终产物 |
| [22x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v4_independent/runs_balanced_120k_seed0/seed_0/depth22_width512/metrics.json) | 370/450 | 8.574403e-5 | 1.542959e-2 | 1.778549e-2 | 已生成最终产物 |
| [22x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v4_independent/runs_balanced_120k_seed0/seed_0/depth22_width1024/metrics.json) | 337/417 | 6.610460e-5 | 1.182697e-2 | 1.202196e-2 | 已生成最终产物 |
| [34x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v4_independent/runs_balanced_120k_seed0/seed_0/depth34_width512/metrics.json) | 242/322 | 2.788966e-4 | 1.543492e-2 | 1.673751e-2 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v4_independent/runs_balanced_120k_seed0/seed_0/depth34_width1024/metrics.json) | 182/262 | 1.296918e-4 | 1.203721e-2 | 1.170782e-2 | 已生成最终产物 |

## ACADO v5：40k+80k recovery

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v5_phase_consistent/runs_pilot_stable/seed_0/depth10_width512/metrics.json) | 22/300 | 1.154777e-1 | 1.915338e-1 | 6.926405e-1 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v5_phase_consistent/runs_pilot_stable/seed_0/depth34_width1024/metrics.json) | 58/300 | 3.560172e-3 | 1.848897e-1 | 6.515083e-1 | 已生成最终产物 |

## ACADO v5：仅40k

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v5_phase_consistent/runs_paper40k_pilot/seed_0/depth10_width512/metrics.json) | 488/1000 | 7.867306e-3 | 1.992012e-2 | 9.691907e-4 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v5_phase_consistent/runs_paper40k_pilot/seed_0/depth34_width1024/metrics.json) | 468/1000 | 4.678052e-3 | 2.108217e-2 | 1.012402e-3 | 已生成最终产物 |

## ACADO v6：diverse40k

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v6_diverse40k/runs_full/seed_0/depth10_width512/metrics.json) | 19798/20000 | 1.829397e-4 | 1.148558e-2 | 7.790721e-4 | 已生成最终产物 |
| [10x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v6_diverse40k/runs_full/seed_0/depth10_width1024/metrics.json) | 14863/16241 | 9.907992e-5 | 1.080321e-2 | 7.343931e-4 | 已生成最终产物 |
| [22x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v6_diverse40k/runs_full/seed_0/depth22_width512/metrics.json) | 102/1000 | 6.201245e-3 | 1.229643e-2 | 1.184169e-3 | 已生成最终产物 |
| [22x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v6_diverse40k/runs_full/seed_0/depth22_width1024/metrics.json) | 253/1000 | 6.269437e-3 | 1.123931e-2 | 8.924404e-4 | 已生成最终产物 |
| [34x512](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v6_diverse40k/runs_full/seed_0/depth34_width512/metrics.json) | 217/1000 | 5.161948e-3 | 1.259691e-2 | 1.109882e-3 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_paper_fidelity_acado_v6_diverse40k/runs_full/seed_0/depth34_width1024/metrics.json) | 144/1000 | 7.466197e-3 | 1.211709e-2 | 1.005492e-3 | 已生成最终产物 |

## 10m/s short rollout 120k

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_12d_short_rollout_10ms_v1/runs/seed_0/depth10_width512/metrics.json) | 275/469 | 1.519153e-3 | 6.803557e-2 | 2.628690e-2 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_12d_short_rollout_10ms_v1/runs/seed_0/depth34_width1024/history.csv) | 152/300 | 2.191033e-3 | 6.491826e-2 | 2.391219e-2 | 仅检查点，缺最终指标 |

## 10m/s hybrid160k 原pilot（大模型存在跨数据版本续训经历）

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_10ms_hybrid_160k_v1/runs_local_pilot_300_seed0/seed_0/depth10_width512/metrics.json) | 299/300 | 5.404827e-3 | 1.983439e-1 | 3.468156e-1 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_10ms_hybrid_160k_v1/runs_local_pilot_300_seed0/seed_0/depth34_width1024/history.csv) | 164/167 | 3.070880e-2 | 9.362856e-2 | 2.024613e-1 | 仅检查点，缺最终指标 |

## 10m/s hybrid160k v4从头训练

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_10ms_hybrid_160k_v1/runs_local_pilot_300_seed0_v4_fresh/seed_0/depth10_width512/metrics.json) | 275/300 | 6.438992e-3 | 1.176579e-1 | 2.616895e-1 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_10ms_hybrid_160k_v1/runs_local_pilot_300_seed0_v4_fresh/seed_0/depth34_width1024/history.csv) | 261/263 | 3.739268e-3 | 9.902070e-2 | 2.123678e-1 | 仅检查点，缺最终指标 |

## 300D 140k

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/falcon_mpc_complete_300d_v1/runs_pilot/seed_0/depth10_width512/metrics.json) | 383/400 | 2.669611e-3 | 5.031072e-2 | 4.260369e-2 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/falcon_mpc_complete_300d_v1/runs_pilot/seed_0/depth34_width1024/history.csv) | 81/84 | 1.172405e-2 | 5.084641e-2 | 4.968069e-2 | 仅检查点，缺最终指标 |

## 历史5m/s早期/中断尝试（不属于部署采用六模型）

| 模型 | best/last epoch | Train normalized MSE | Validation normalized MSE | Validation physical MSE | 状态 |
|---|---:|---:|---:|---:|---|
| [10x512](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs/paper_track_p_12d_5ms_v1_depth10_width512/history.csv) | 426/434 | 9.614806e-5 | 未记录 | 1.421632e-4 | 仅检查点，缺最终指标 |
| [10x1024](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs/paper_track_p_12d_5ms_v1_depth10_width1024/history.csv) | 205/227 | 1.585102e-4 | 未记录 | 3.323871e-4 | 仅检查点，缺最终指标 |
| [22x512](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs/paper_track_p_12d_5ms_v1_depth22_width512/history.csv) | 256/281 | 3.381668e-4 | 未记录 | 7.215934e-4 | 仅检查点，缺最终指标 |
| [22x1024](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs/paper_track_p_12d_5ms_v1_depth22_width1024/metrics.json) | 115/315 | 1.000009e+0 | 1.004315e+0 † | 3.127930e+0 | 已生成最终产物 |
| [22x1024](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth22_width1024/history.csv) | 13/13 | 4.857679e-3 | 未记录 | 1.858645e-2 | 仅检查点，缺最终指标 |
| [34x512](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs/paper_track_p_12d_5ms_v1_depth34_width512/metrics.json) | 115/315 | 1.000010e+0 | 1.004316e+0 † | 3.127930e+0 | 已生成最终产物 |
| [34x1024](F:/FalconReproduction/airsim_control/experiments/paper_track_p_12d_5ms_v1/runs/paper_track_p_12d_5ms_v1_depth34_width1024/history.csv) | 115/285 | 1.000009e+0 | 未记录 | 3.127930e+0 | 仅检查点，缺最终指标 |

## 解读边界

- 历史5m/s六模型混合本机与服务器运行，且先前发现归一化文件不完全一致；容量因果比较需要额外控制，不能把所有差异归咎于模型规模。
- v6浅层模型训练约1.6–2万轮，深层模型约1000轮，训练预算不一致。
- v4具有模型间差别，但同样存在很大的train/validation gap；差别更大并不等于训练或闭环更可靠，且验证集并非严格单调。
- 300D最大模型当前仅84轮记录，小模型完成400轮，不能据此判定收敛后的容量趋势。
- 25D与早期部分12D运行的在线train高于epoch末validation，不直接构成异常；训练期权重不断变化、样本难度也不同。
- 最早的MPC/门框联调和后续FPGA部署/影子测试不是独立的新DNN训练阶段，不重复统计相同权重。

