# 状态有效期模块：第一阶段完成

## 当前进度与剩余范围

已有模型训练、同状态误差比较、共享推理回放、三机固定模型/T组合和预设切换机制已经完成。旧试飞统计缺陷已经另存审计修正，原始报告没有改写。

本阶段完成的是**门附近状态预算的因果旁路回放**，而不是在线自适应控制：用已有状态和过去已完成的响应估计T_max，事后核对下一次下发时的年龄。开启该功能的新试飞仍由既有DNN及预设切换计划控制，预算分析只在飞行结束后运行。这与`probe_airsim_three --mode shadow`的“MPC控制、DNN旁路推理”不是同一个模式。

剩余一个实现阶段：在线状态/资源联合选型，并执行一次必要的三机对照。当前模块的结束标准是单位/时序正确、决策不使用未来信息、旧请求D不回改、切换不重置参考时钟、对照报告完整。预计剩余代码与轻量检查约1～2小时，你执行必要试飞含启动/整理约10～20分钟；异常修复、物理上界重新标定、超网训练和论文级重复实验不包含在此估计中。

超网/早退与独立模型池的比较是后续模块，目前没有证明超网必要。

## PPT公式和实现假设

来源为`D:\杂用\组会\20260923\组会.pptx`第1页；仅读取，未修改。

采样时刻t_k，控制实际生效时刻t'_k；R_k=t'_k-t_k，H_k=t'_(k+1)-t'_k。于是：

`H_k = T_k + R_(k+1) - R_k`

`R_k + H_k = T_k + R_(k+1)`

若横向偏差变化率在整个区间始终不超过v_y_bound，剩余横向间隙为d_y，则条件为`T_k + R_(k+1) <= d_y / v_y_bound`。R_k在推导中抵消，因此不是T_k+R_k。

实现把同样条件独立扩展到垂直门洞，取两个轴的较小预算。控制MSE没有加到米上，前向巡航速度没有被当作横向误差变化率，瞬时横向速度没有被当作未来速度上界。

当前参数集中在`controller_architecture_study\configs\state_budget_shadow_v1.json`：

- 门几何沿用试飞：宽/高4m，横梁厚0.15m，有效间隙0.25m。允许中心偏移半宽/半高为`4/2 - 0.15/2 - 0.25 = 1.675m`。0.25m是现有穿门判定的有效半机身尺寸加余量假设，不是新测量的飞机尺寸。
- 仅门局部x方向±2m内监测；远处对准门的状态显示“不适用”，不算安全通过。门姿态和每机局部坐标偏移参与换算。
- 横向/垂直偏差变化率假设各不超过3m/s，未证明是动力学上界。采样速度超出时标记失效，不提供可用候选周期。
- 下一响应估计为`max(30ms, 最近32个已下发响应最大值 + 5ms)`。只用该次采样之前已完成的响应，包括共享服务的其他飞机；当前/未来输出不进入估计。30ms和5ms是原型参数，不是WCET。
- `T_max = min(d_y/v_y_bound, d_z/v_z_bound) - R_next_estimate`，负值保留，不抬升为最小候选T。
- 事后使用真实采样间隔加下一条响应，等价于下一条命令下发时刻减本次采样时刻；不是名义T加本次响应。
- 主机读状态请求开始及异步命令提交完成时间近似采样/生效；没有真实传感器年龄和执行器生效确认。

这些条件不包含模型近似误差、未知扰动的额外界及门外障碍约束。全部记录满足也**不是全路线安全证书**，更不代表可以直接使用数百毫秒周期。下一阶段只从已试飞候选中选型。

## 已有试飞回放结果

源目录：

`controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_joint_switch_20261004T140628_043253Z`

最终代码的旁路报告：

`controller_architecture_study\results\dynamic_complexity_237d_v1\state_budget_shadow_20261004T145621_620440Z\state_budget_report.json`

状态1051个，门附近适用172个，其他879个不作门洞预算判定。未发现采样速度超出所设上界、中心越过内开口界、名义周期超预算或事后下一次下发年龄超预算。估计T_max最小467.887ms、平均509.184ms；适用状态到下一次下发的实际年龄最大118.145ms。50/75/100ms在这172个状态下均满足相同响应估计条件，这不是不同模型的独立响应预测。

原报告、请求、轨迹SHA256逐一核对，未改写；没有启动AirSim/GPU推理或训练。68项CPU/模拟接口测试通过，含坐标旋转、R_(k+1)、未来信息隔离、负预算、假设失效、无适用状态、缺少下一下发、接口集成及原切换调度回归。文件预检PASS。

结论：当前路线较宽松，状态约束未成为瓶颈，不能据此证明动态周期或超网必要。不能为了制造切换而事后修改分析参数；之后应在预先设定的更紧门洞/可恢复偏差或真实共享负载下观察边界是否活跃。仅修改分析参数不等于更换物理门。

## 完整命令

旧报告回放已经执行，不必重复。如需自己重现，无需AirSim，输出为新时间戳目录：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
& $trainPy -m controller_architecture_study.state_budget_shadow --report '.\controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_joint_switch_20261004T140628_043253Z\report.json'
if ($LASTEXITCODE -ne 0) { throw '预算回放失败，请查看输出' }
```

如需新试飞顺便记录预算，只需一次约20秒三机预设切换。先手动打开原UE4项目点击Play，不要同时训练或回放；旧运行时门框可Stop/Play清空。程序不自动启动另一份AirSim，只在初始化reset/空中放置一次。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }
& $trainPy -m controller_architecture_study.joint_switch_flight --device cuda --state-budget-config '.\controller_architecture_study\configs\state_budget_shadow_v1.json'
if ($LASTEXITCODE -ne 0) { Write-Warning '试飞未通过，请保留报告，不能只看穿门成功' }
```

新试飞目录包含原有report.json/requests.jsonl/flight_trace.json，加上state_budget_report.json及state_budget_samples.json。逐状态记录保留R_next估计、T_max、假设失效标志、周期判定，以及事后采样间隔/下一响应/年龄。最后一个状态没有下一下发时显示未知，不算通过。预算结论不改写既有穿门/实时性验收标志。

历史报告缺少局部门位姿时，通过SHA256匹配的原路线与每机参考位置恒定平移恢复，逐状态验证；新试飞直接保存每机局部门位姿。

## 同日后续进度

联合选型核心和旁路回放接口已经实现。完整命令和未完成边界见`controller_architecture_study/docs/JOINT_SELECTION_SHADOW_20261004.md`。`joint_switch_flight --joint-selection-shadow`修正版仅在飞行时捕获位置/速度/时间戳，飞行结束后因果回放fixed/period_only/adaptive建议，不应用建议，不在控制线程计算策略；实际控制仍按预设计划。1051个旧状态回放中三策略均建议plain4x256/T50，尚未证明动态策略收益。

2026-10-05更新：用户已完成移除同步分析后的试飞，飞行/实时性/综合验收均通过；1050条联合选型回放关联也已修复并重算，三策略均建议plain4x256/T50。无需重复试飞、重新训练18个模型或启动超网训练。详见[阶段收尾报告](F:/FalconReproduction/airsim_control/controller_architecture_study/docs/REPLAY_ALIGNMENT_CLOSEOUT_20261005.md)。在线选型开销、实际建议接管和边界活跃场景尚未完成；若现有场景不触发约束，应记录“收益未证实”，而非无限追加重复验证或宣称超网必要。
