# 7机：静态性能搜索与状态框架的配对结果

这是CPU有限实验，使用经验名义/尾部预测，不是WCET、CBF安全证明或完整FALCON复现。
运行目录：`F:\FalconReproduction\airsim_control\controller_architecture_study\results\framework_runtime_v26_fine_static\n7_net2`。版本：`framework_runtime_v26_fine_static`。

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

## 静态候选重复复核

本批三个候选各额外运行两次；三次均通过才可选型，按三次平均J排序。

| 候选 | 初次通过 | 两次额外通过 | 三次平均J | 可参与最终选型 |
|---|---|---|---:|---|
| plain_mlp_h9_w512_seed0_T30 | True | False/False | — | False |
| plain_mlp_h9_w512_seed0_T40 | True | False/True | — | False |
| plain_mlp_h21_w512_seed0_T40 | True | True/False | — | False |

选型尚未通过：`NO_REPEATEDLY_VALID_STATIC_CANDIDATE`。未启动后续挑战不等于安全完成。

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
