# 七机：静态性能搜索与状态框架的配对结果

这是CPU有限实验，使用经验名义/尾部预测，不是WCET、CBF安全证明或完整FALCON复现。
运行目录：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\framework_runtime_v10_residual_cost\n7_net2`。版本：`framework_runtime_v10_residual_cost`。

## 正常工况的静态候选搜索

已完成候选：40；通过全部可比条件：12。
J为原式形式的完整窗口状态跟踪评分（Wx=I、公共20ms网格），不是DNN对MPC输出的离线MSE。
恢复日志覆盖不足的候选不等同于观察到控制错误；完整原始报告保留。

| 候选 | 完整可比 | 总J | 中止/其他未通过条件 |
|---|---|---:|---|
| plain_mlp_h9_w512_seed0_T25 | False | — | expired_not_applied |
| plain_mlp_h9_w512_seed0_T50 | True | 4.680515 | 无 |
| plain_mlp_h9_w512_seed0_T75 | False | — | expired_not_applied |
| plain_mlp_h9_w512_seed0_T100 | True | 4.093660 | 无 |
| plain_mlp_h9_w512_seed0_T125 | False | 3.775555 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w512_seed0_T150 | False | 3.426627 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w512_seed0_T25 | False | — | command_return_late |
| plain_mlp_h21_w512_seed0_T50 | True | 4.732904 | 无 |
| plain_mlp_h21_w512_seed0_T75 | True | 4.360029 | 无 |
| plain_mlp_h21_w512_seed0_T100 | True | 4.111145 | 无 |
| plain_mlp_h21_w512_seed0_T125 | False | 3.781785 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w512_seed0_T150 | False | 3.469853 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h33_w512_seed0_T75 | True | 4.329051 | 无 |
| plain_mlp_h33_w512_seed0_T100 | False | 4.067958 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T125 | False | 3.690662 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T150 | False | 3.420430 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w1024_seed0_T50 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w1024_seed0_T75 | True | 4.347840 | 无 |
| plain_mlp_h9_w1024_seed0_T100 | True | 4.098063 | 无 |
| plain_mlp_h9_w1024_seed0_T125 | False | 3.727161 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w1024_seed0_T150 | False | — | expired_not_applied |
| plain_mlp_h21_w1024_seed0_T25 | False | — | expired_not_applied |
| plain_mlp_h21_w1024_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h21_w1024_seed0_T75 | True | 4.419521 | 无 |
| plain_mlp_h21_w1024_seed0_T100 | True | 4.107326 | 无 |
| plain_mlp_h21_w1024_seed0_T125 | False | 3.787019 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w1024_seed0_T150 | False | 3.469988 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T50 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T75 | False | — | expired_not_applied |
| plain_mlp_h33_w1024_seed0_T100 | True | 4.002078 | 无 |
| plain_mlp_h33_w1024_seed0_T125 | False | 3.688322 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w1024_seed0_T150 | False | 3.375791 | 恢复状态或日志覆盖未通过 |
| neighbor0 | True | 4.763707 | 无 |
| neighbor1 | False | — | delivery_budget_exhausted |
| neighbor2 | False | — | expired_not_applied |
| neighbor3 | False | — | delivery_budget_exhausted |

选中静态候选：`neighbor0`；选择状态：`ADAPTIVE_NOMINAL_FAILED`。
选中静态J=4.763707；动态正常工况J=—；比率=—。
这是观测值，不是统计非劣或总体最优证明。
动态正常复核：有效安全分析23次，其中物理B先于80ms更新上限生效0次。
若这里为0，正常工况只能说明集成/性能观察，不能用它证明状态物理期限的安全收益。

## 预声明扰动配对

发现集完成：False；已完成配对：0/16。

| 场景 | 初态契约一致 | 静态碰撞 | 静态完成 | 动态完成 | 动态J | 干净安全优势 |
|---|---|---:|---|---|---:|---|

核对初态后的干净安全优势数：**0**。
静态碰撞而动态中止，不算安全任务完成；基线撞毁后的不完整J不与完整动态J作大小比较。
相邻扰动确认/消融完成：False；记录数：0。
所有失败、预测拒绝、实际时序违规均保留，不能只挑选有利场景。

## 当前证据边界

只有发现集完成且存在独立确认，才能据此讨论可重复的安全优势。没有优势时应报告负结果。
模式仅在任务边界变化，抢占仅在段边界；CPU时延不适用于FPGA。B仍是启发式。
上述基线是受限模型/周期族的经验静态搜索，没有SVM和全硬件搜索，不能写成完整FALCON的全局最优解。
