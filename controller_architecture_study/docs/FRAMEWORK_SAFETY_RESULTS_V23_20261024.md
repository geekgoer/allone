# 七机：静态性能搜索与状态框架的配对结果

> **碰撞有效性更正：** 此版本使用了当前AirSim未填充的状态包collision字段。表中的“碰撞0”只是旧接口记录，不能证明无真实碰撞；相关安全验收和模式表需用专用碰撞接口重采。原始数值保留以追溯，不能引用为安全优势或无碰撞证据。

这是CPU有限实验，使用经验名义/尾部预测，不是WCET、CBF安全证明或完整FALCON复现。
运行目录：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\framework_runtime_v23_precise_release\n7_net2`。版本：`framework_runtime_v23_precise_release`。

## 正常工况的静态候选搜索

已完成候选：40；通过全部可比条件：20。
J为原式形式的完整窗口状态跟踪评分（Wx=I、公共20ms网格），不是DNN对MPC输出的离线MSE。
恢复日志覆盖不足的候选不等同于观察到控制错误；完整原始报告保留。

| 候选 | 完整可比 | 总J | 中止/其他未通过条件 |
|---|---|---:|---|
| plain_mlp_h9_w512_seed0_T25 | False | — | deadline_expired_before_submit |
| plain_mlp_h9_w512_seed0_T50 | True | 4.745125 | 无 |
| plain_mlp_h9_w512_seed0_T75 | True | 4.445271 | 无 |
| plain_mlp_h9_w512_seed0_T100 | True | 4.129162 | 无 |
| plain_mlp_h9_w512_seed0_T125 | False | 3.785689 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w512_seed0_T150 | False | 3.494658 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w512_seed0_T25 | False | — | deadline_expired_before_submit |
| plain_mlp_h21_w512_seed0_T50 | True | 4.718418 | 无 |
| plain_mlp_h21_w512_seed0_T75 | True | 4.454018 | 无 |
| plain_mlp_h21_w512_seed0_T100 | True | 4.137741 | 无 |
| plain_mlp_h21_w512_seed0_T125 | False | 3.816216 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w512_seed0_T150 | False | 3.465897 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T50 | True | 4.662778 | 无 |
| plain_mlp_h33_w512_seed0_T75 | True | 4.321682 | 无 |
| plain_mlp_h33_w512_seed0_T100 | True | 4.057438 | 无 |
| plain_mlp_h33_w512_seed0_T125 | False | 3.698979 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w512_seed0_T150 | False | 3.453271 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w1024_seed0_T50 | True | 4.703915 | 无 |
| plain_mlp_h9_w1024_seed0_T75 | True | 4.425580 | 无 |
| plain_mlp_h9_w1024_seed0_T100 | True | 4.104753 | 无 |
| plain_mlp_h9_w1024_seed0_T125 | False | 3.792626 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h9_w1024_seed0_T150 | False | 3.472819 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w1024_seed0_T25 | False | — | expired_not_applied |
| plain_mlp_h21_w1024_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h21_w1024_seed0_T75 | True | 4.332643 | 无 |
| plain_mlp_h21_w1024_seed0_T100 | True | 4.045398 | 无 |
| plain_mlp_h21_w1024_seed0_T125 | False | 3.691718 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h21_w1024_seed0_T150 | False | 3.404710 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T50 | False | — | expired_not_applied |
| plain_mlp_h33_w1024_seed0_T75 | True | 4.221780 | 无 |
| plain_mlp_h33_w1024_seed0_T100 | True | 3.958709 | 无 |
| plain_mlp_h33_w1024_seed0_T125 | False | 3.585945 | 恢复状态或日志覆盖未通过 |
| plain_mlp_h33_w1024_seed0_T150 | False | 3.263506 | 恢复状态或日志覆盖未通过 |
| neighbor0 | True | 4.730506 | 无 |
| neighbor1 | True | 4.600260 | 无 |
| neighbor2 | True | 4.741084 | 无 |
| neighbor3 | True | 4.652480 | 无 |

选中静态候选：`plain_mlp_h9_w512_seed0_T50`；选择状态：`READY`。
选中静态J=4.745125；动态正常工况J=4.727496；比率=0.996285。
这是观测值，不是统计非劣或总体最优证明。
动态正常复核：有效安全分析456次，其中物理B先于80ms更新上限生效0次。
若这里为0，正常工况只能说明集成/性能观察，不能用它证明状态物理期限的安全收益。

## 预声明扰动配对

发现集完成：True；已完成配对：16/16。

| 场景 | 初态契约一致 | 静态碰撞 | 静态完成 | 动态完成 | 动态J | 干净安全优势 |
|---|---|---:|---|---|---:|---|
| i0_sign-1_v0.6 | True | 0 | True | True | 4.345917 | False |
| i0_sign-1_v1 | True | 0 | True | False | — | False |
| i0_sign-1_v1.4 | True | 0 | False | False | — | False |
| i0_sign-1_v1.8 | True | 0 | False | False | — | False |
| i0_sign+1_v0.6 | True | 0 | True | False | — | False |
| i0_sign+1_v1 | True | 0 | True | False | — | False |
| i0_sign+1_v1.4 | True | 0 | False | False | — | False |
| i0_sign+1_v1.8 | True | 0 | False | False | — | False |
| i6_sign-1_v0.6 | True | 0 | True | False | — | False |
| i6_sign-1_v1 | True | 0 | True | False | — | False |
| i6_sign-1_v1.4 | True | 0 | False | False | — | False |
| i6_sign-1_v1.8 | True | 0 | False | False | — | False |
| i6_sign+1_v0.6 | True | 0 | True | True | 4.444730 | False |
| i6_sign+1_v1 | True | 0 | True | False | — | False |
| i6_sign+1_v1.4 | True | 0 | False | False | — | False |
| i6_sign+1_v1.8 | True | 0 | False | False | — | False |

核对初态后的干净安全优势数：**0**。
静态碰撞而动态中止，不算安全任务完成；基线撞毁后的不完整J不与完整动态J作大小比较。
相邻扰动确认/消融完成：False；记录数：0。
所有失败、预测拒绝、实际时序违规均保留，不能只挑选有利场景。

## 当前证据边界

只有发现集完成且存在独立确认，才能据此讨论可重复的安全优势。没有优势时应报告负结果。
模式仅在任务边界变化，抢占仅在段边界；CPU时延不适用于FPGA。B仍是启发式。
上述基线是受限模型/周期族的经验静态搜索，没有SVM和全硬件搜索，不能写成完整FALCON的全局最优解。
