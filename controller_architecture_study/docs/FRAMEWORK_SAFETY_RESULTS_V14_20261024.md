# 七机：静态性能搜索与状态框架的配对结果

这是CPU有限实验，使用经验名义/尾部预测，不是WCET、CBF安全证明或完整FALCON复现。
运行目录：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\framework_runtime_v14_stage_visibility\n7_net2`。版本：`framework_runtime_v14_stage_visibility`。

## 正常工况的静态候选搜索

已完成候选：40；通过全部可比条件：10。
J为原式形式的完整窗口状态跟踪评分（Wx=I、公共20ms网格），不是DNN对MPC输出的离线MSE。
恢复日志覆盖不足的候选不等同于观察到控制错误；完整原始报告保留。

| 候选 | 完整可比 | 总J | 中止/其他未通过条件 |
|---|---|---:|---|
| plain_mlp_h9_w512_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w512_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h9_w512_seed0_T75 | True | 4.354235 | 无 |
| plain_mlp_h9_w512_seed0_T100 | True | 4.102904 | 无 |
| plain_mlp_h9_w512_seed0_T125 | False | 3.766535 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w512_seed0_T150 | False | 3.454983 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w512_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w512_seed0_T50 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w512_seed0_T75 | True | 4.409248 | 无 |
| plain_mlp_h21_w512_seed0_T100 | True | 4.053706 | 无 |
| plain_mlp_h21_w512_seed0_T125 | False | 3.734135 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w512_seed0_T150 | False | 3.344776 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h33_w512_seed0_T75 | False | — | expired_not_applied |
| plain_mlp_h33_w512_seed0_T100 | False | 4.045239 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T125 | False | 3.671745 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T150 | False | 3.354022 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w1024_seed0_T50 | True | 4.682293 | 无 |
| plain_mlp_h9_w1024_seed0_T75 | True | 4.314539 | 无 |
| plain_mlp_h9_w1024_seed0_T100 | False | 4.039898 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w1024_seed0_T125 | False | 3.702016 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w1024_seed0_T150 | False | 3.425077 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w1024_seed0_T25 | False | — | expired_not_applied |
| plain_mlp_h21_w1024_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h21_w1024_seed0_T75 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w1024_seed0_T100 | True | 3.849271 | 无 |
| plain_mlp_h21_w1024_seed0_T125 | False | 3.699594 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w1024_seed0_T150 | False | 3.369734 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w1024_seed0_T25 | False | — | expired_not_applied |
| plain_mlp_h33_w1024_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h33_w1024_seed0_T75 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T100 | True | 3.784437 | 无 |
| plain_mlp_h33_w1024_seed0_T125 | False | 3.548052 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w1024_seed0_T150 | False | 3.253447 | 恢复状态或日志覆盖未通过 |
| neighbor0 | True | 4.646971 | 无 |
| neighbor1 | False | — | delivery_budget_exhausted |
| neighbor2 | False | — | expired_not_applied |
| neighbor3 | True | 4.617856 | 无 |

选中静态候选：`plain_mlp_h9_w1024_seed0_T50`；选择状态：`ADAPTIVE_NOMINAL_FAILED`。
选中静态J=4.682293；动态正常工况J=—；比率=—。
这是观测值，不是统计非劣或总体最优证明。
动态正常复核：有效安全分析338次，其中物理B先于80ms更新上限生效0次。
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
