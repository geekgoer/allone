# 7机：静态性能搜索与状态框架的配对结果

这是CPU有限实验，使用经验名义/尾部预测，不是WCET、CBF安全证明或完整FALCON复现。
运行目录：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\framework_runtime_v27_stable_static\n7_net2`。版本：`framework_runtime_v27_stable_static`。

碰撞来源：专用API及新事件时间戳。

## 正常工况的静态候选搜索

已完成候选：52；通过全部可比条件：22。
J为原式形式的完整窗口状态跟踪评分（Wx=I、公共20ms网格），不是DNN对MPC输出的离线MSE。
恢复日志覆盖不足的候选不等同于观察到控制错误；完整原始报告保留。

| 候选 | 完整可比 | 总J | 中止/其他未通过条件 |
|---|---|---:|---|
| plain_mlp_h9_w512_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w512_seed0_T30 | True | 4.947552 | 无 |
| plain_mlp_h9_w512_seed0_T35 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w512_seed0_T40 | True | 4.836874 | 无 |
| plain_mlp_h9_w512_seed0_T45 | True | 4.768708 | 无 |
| plain_mlp_h9_w512_seed0_T50 | True | 4.698504 | 无 |
| plain_mlp_h9_w512_seed0_T75 | True | 4.409420 | 无 |
| plain_mlp_h9_w512_seed0_T100 | True | 4.103435 | 无 |
| plain_mlp_h21_w512_seed0_T25 | False | — | deadline_expired_before_submit |
| plain_mlp_h21_w512_seed0_T30 | False | — | deadline_expired_before_submit |
| plain_mlp_h21_w512_seed0_T35 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w512_seed0_T40 | True | 4.824690 | 无 |
| plain_mlp_h21_w512_seed0_T45 | True | 4.783855 | 无 |
| plain_mlp_h21_w512_seed0_T50 | True | 4.683543 | 无 |
| plain_mlp_h21_w512_seed0_T75 | True | 4.388171 | 无 |
| plain_mlp_h21_w512_seed0_T100 | True | 4.109449 | 无 |
| plain_mlp_h33_w512_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T30 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T35 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T40 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T45 | True | 4.705463 | 无 |
| plain_mlp_h33_w512_seed0_T50 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w512_seed0_T75 | True | 4.302094 | 无 |
| plain_mlp_h33_w512_seed0_T100 | True | 4.037952 | 无 |
| plain_mlp_h9_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w1024_seed0_T30 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w1024_seed0_T35 | False | — | delivery_budget_exhausted |
| plain_mlp_h9_w1024_seed0_T40 | True | 4.817573 | 无 |
| plain_mlp_h9_w1024_seed0_T45 | True | 4.769614 | 无 |
| plain_mlp_h9_w1024_seed0_T50 | True | 4.673525 | 无 |
| plain_mlp_h9_w1024_seed0_T75 | True | 4.402546 | 无 |
| plain_mlp_h9_w1024_seed0_T100 | True | 4.091399 | 无 |
| plain_mlp_h21_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w1024_seed0_T30 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w1024_seed0_T35 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w1024_seed0_T40 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w1024_seed0_T45 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w1024_seed0_T50 | False | — | delivery_budget_exhausted |
| plain_mlp_h21_w1024_seed0_T75 | True | 4.323195 | 无 |
| plain_mlp_h21_w1024_seed0_T100 | True | 4.022565 | 无 |
| plain_mlp_h33_w1024_seed0_T25 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T30 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T35 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T40 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T45 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T50 | False | — | delivery_budget_exhausted |
| plain_mlp_h33_w1024_seed0_T75 | False | — | expired_not_applied |
| plain_mlp_h33_w1024_seed0_T100 | True | 3.903843 | 无 |
| neighbor0 | False | — | expired_not_applied |
| neighbor1 | False | — | delivery_budget_exhausted |
| neighbor2 | False | — | delivery_budget_exhausted |
| neighbor3 | False | — | delivery_budget_exhausted |

上述初次候选来自保留的搜索：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\framework_runtime_v26_fine_static\n7_net2`；本批没有重飞全部候选。
此前已复核候选被排除，不重复等成功：plain_mlp_h9_w512_seed0_T30, plain_mlp_h9_w512_seed0_T40, plain_mlp_h21_w512_seed0_T40。

## 静态候选重复复核

本批三个候选各额外运行两次；三次均通过才可选型，按三次平均J排序。

| 候选 | 初次通过 | 两次额外通过 | 三次平均J | 可参与最终选型 |
|---|---|---|---:|---|
| plain_mlp_h9_w1024_seed0_T40 | True | True/True | 4.841886 | True |
| plain_mlp_h21_w512_seed0_T45 | True | True/True | 4.793085 | True |
| plain_mlp_h9_w1024_seed0_T45 | True | True/True | 4.746907 | True |

选中静态候选：`plain_mlp_h9_w1024_seed0_T40`；选择状态：`READY`。
静态选型分数J=4.841886；动态正常工况J=4.803077；比率=0.991985。
有重复复核时，静态分数是三轮均值，动态此处是一轮观察；该比率不是配对效应估计。
这是观测值，不是统计非劣或总体最优证明。
初次候选来自先前52个配置的冻结搜索：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\framework_runtime_v26_fine_static\n7_net2\calibration_results.json`。当前选择与追加复核记录均保留。
动态正常复核：有效安全分析523次，物理B先于基本更新及模式等待上限生效0次，模式等待上限先行生效0次。
若物理B激活为0，正常工况只能说明集成/性能观察，不能用它证明物理B本身带来的安全收益。

## 预声明扰动配对

发现集完成：True；已完成配对：16/16。

| 场景 | 初态契约一致 | 静态碰撞 | 静态完成 | 动态完成 | 动态J | 干净安全优势 |
|---|---|---:|---|---|---:|---|
| i0_sign-1_v0.6 | True | 0 | True | True | 4.400620 | False |
| i0_sign-1_v1 | True | 0 | True | False | — | False |
| i0_sign-1_v1.4 | True | 0 | False | False | — | False |
| i0_sign-1_v1.8 | True | 1 | False | False | — | False |
| i0_sign+1_v0.6 | True | 0 | True | False | — | False |
| i0_sign+1_v1 | True | 0 | True | False | — | False |
| i0_sign+1_v1.4 | True | 1 | False | False | — | False |
| i0_sign+1_v1.8 | True | 1 | False | False | — | False |
| i6_sign-1_v0.6 | True | 0 | True | False | — | False |
| i6_sign-1_v1 | True | 0 | True | False | — | False |
| i6_sign-1_v1.4 | True | 1 | False | False | — | False |
| i6_sign-1_v1.8 | True | 1 | False | False | — | False |
| i6_sign+1_v0.6 | True | 0 | False | True | 4.534176 | False |
| i6_sign+1_v1 | True | 0 | True | False | — | False |
| i6_sign+1_v1.4 | True | 1 | False | False | — | False |
| i6_sign+1_v1.8 | True | 1 | False | False | — | False |

核对初态后的干净安全优势数：**0**。
静态碰撞而动态中止，不算安全任务完成；基线撞毁后的不完整J不与完整动态J作大小比较。
相邻扰动确认/消融完成：False；记录数：0。
所有失败、预测拒绝、实际时序违规均保留，不能只挑选有利场景。

## 当前证据边界

只有发现集完成且存在独立确认，才能据此讨论可重复的安全优势。没有优势时应报告负结果。
模式仅在任务边界变化，抢占仅在段边界；CPU时延不适用于FPGA。B仍是启发式。
上述基线是受限模型/周期族的经验静态搜索，没有SVM和全硬件搜索，不能写成完整FALCON的全局最优解。
