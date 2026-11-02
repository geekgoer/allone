# 三机联合切换与旁路回放：阶段收尾（2026-10-05）

## 结论

三机预设模型/T切换试飞已通过；飞行后回放中的请求关联错误已修复。用同一次成功试飞重算，1050条状态与请求全部一一匹配，三种策略都建议plain4x256/T50，没有切换。这完成了“预设切换、轻量状态捕获、因果回放”阶段，但没有完成真正的在线自适应控制，也没有证明超网必要。

本轮没有训练、GPU推理或AirSim试飞，没有改变控制逻辑、模型、规划器、T/D、原始数据或验收条件。用户当前不需要再运行命令。

## 原始飞行与保存范围

- [原飞行报告](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1/airsim_joint_switch_20261004T160131_391468Z/report.json)：flight_passed、realtime_passed、acceptance_passed均为true，errors为空。
- 1050次请求全部及时完成并下发；15次穿门通过；预设模型切换6次、周期切换9次。切换计数是实际脚本执行，不是在线建议被应用。
- 实际脚本为0秒plain4x256/T50、3秒plain4x512/T75、6秒plain4x512/T100、9秒plain4x256/T50，D=T。
- 活动路线位置误差RMSE约0.359m。sample_to_output平均3.911ms、P99为7.093ms、最大27.556ms；release_to_output平均11.807ms、P99为21.075ms、最大32.172ms。
- 实时快照捕获平均约0.00132ms、P99约0.00230ms、最大约0.00340ms。选择器在飞行后才运行，不占本次飞行响应时间。

原目录的report.json、requests.jsonl、flight_trace.json、setup.json、state_budget_report.json、state_budget_samples.json以及原joint_shadow报告/决策，共8个文件的SHA256前后一致。永久核对记录见[源文件保留与独立关联审计](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1/joint_selection_alignment_fix_20261005_v1/source_preservation_audit.json)。

## 修复了什么

[回放代码](F:/FalconReproduction/airsim_control/controller_architecture_study/joint_selection_shadow.py)在处理历史入队/完成/下发事件时复用了当前请求变量，导致当前状态被关联到历史请求。原成功试飞目录中的joint_shadow_decisions.jsonl存在1049条关联不一致、1050条记录只有1049个唯一飞机/帧号对。因此该目录原joint_shadow_report.json及joint_shadow_decisions.jsonl的策略结论失效，不能再用它们说明需要切换、资源耗尽或自适应收益；实际飞行和独立状态预算报告不受影响。

现在历史事件使用独立变量，并在回放保存前逐条检查飞机名、帧号、采样时间、计划释放时间、观察时刻、实际模型和实际T；重复飞机/帧号或错配会使分析失败，不再默默输出结论。

[回归测试](F:/FalconReproduction/airsim_control/tests/test_state_joint_policy.py)先重现旧错误，再验证修复。相关7个测试文件合计80项通过；只有既有pynvml弃用警告。

## 重算结果

- [新旁路报告](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1/joint_selection_alignment_fix_20261005_v1/joint_shadow_report.json)
- [新逐帧决策](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1/joint_selection_alignment_fix_20261005_v1/joint_shadow_decisions.jsonl)

request_alignment=PASS；期望记录1050、匹配1050、唯一飞机/帧号对1050、错配0。还通过独立源请求关联核对。

|策略|建议模型|建议T|建议数|建议切换|预算/资源耗尽|
|---|---|---|---:|---:|---:|
|fixed|plain4x256|50ms|1050|0|0|
|period_only|plain4x256|50ms|1050|0|0|
|adaptive|plain4x256|50ms|1050|0|0|

175条门附近状态为AVAILABLE，其余875条为STATE_NOT_APPLICABLE。后者只估计资源可行性，不代表状态安全通过。

状态预算报告中T_max最小约466.922ms；50/75/100ms在全部175条适用状态上均满足当前条件估计。门尺寸4m×4m、余量0.25m、横向/垂直误差速率上界3m/s均为原型假设，只有门平面前后2m监测范围；不是全路线安全证书或实测WCET。

三策略合计每状态离线CPU决策耗时平均1.051ms、P99为1.849ms、最大22.462ms。这不是实际在线选择器开销，也不能据此保证上线截止时间。

三策略使用同一实际基线队列；没有重建“若应用不同建议”的飞行与排队结果。recommendations_applied=false。质量排序仍是单条验证路线冻结的位置MSE，不是状态条件下的控制质量或安全界。该结果只能说明本场景没有必须切换的证据，不能声称自适应降低了超时、改善了飞行或证明超网必要。

## 已执行命令（无需再次执行）

以下只重算旧记录，不要求Play、训练或GPU推理。输出目录已存在，不应覆盖；需要复算时另选新目录。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $trainPy -m controller_architecture_study.joint_selection_shadow --report '.\controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_joint_switch_20261004T160131_391468Z\report.json' --output-dir '.\controller_architecture_study\results\dynamic_complexity_237d_v1\joint_selection_alignment_fix_20261005_v1'
if ($LASTEXITCODE -ne 0) { throw '联合选型重算失败' }
& $trainPy -m pytest tests/test_state_joint_policy.py tests/test_state_budget_shadow.py tests/test_joint_switch_flight.py tests/test_online_joint_policy.py tests/test_dynamic_probes.py tests/test_joint_controller_sweep.py tests/test_fast_convex_preprocess.py -q
if ($LASTEXITCODE -ne 0) { throw '回归测试失败' }
```

## 下一阶段：精简在线选择器实际接管

先实现一个真实在线策略，而不是在飞行关键路径上同时比较三种策略。在线估计只使用当前状态和已完成历史；把选择开销计入完整响应，使用既有连续参考时钟、释放/下发截止时间以及D=T语义。必须明确本次输出与下一次释放分别使用哪个T，不能只把recommendations_applied改成true。

无合格候选时，必须记录不可行并采用明确的仿真中止/悬停终止措施；不得应用过期输出、隐式放宽截止时间或藏入MPC救援。终止措施不等于经过认证的安全恢复。

必要验证只覆盖实际模型/T切换、时间记账、无候选和过期请求处理。代码就绪后由用户运行一次实际在线接管，再在约束活跃的真实状态或实测共享负载下，以相同场景比较固定、只调T、联合调模型/T；用完整时延、输出年龄、约束违例和活动路线误差评价，不只比较miss数。

当前阶段不用重训18个DNN、不用重复宽松路线、不用再扫GPU容量。超网/早退/常驻独立模型池的公平比较留到确有动态复杂度收益之后；不能为得到切换结果强行制造负载或把回放建议当作控制改进。
