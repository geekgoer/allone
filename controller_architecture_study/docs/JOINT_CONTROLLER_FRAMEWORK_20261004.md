# 从三机接管到周期—复杂度联合控制框架

日期：2026-10-04。当前分支为237D凸MPC imitation，不是原FALCON的12D控制器。

## 1. 已达到的阶段

用户实跑报告：

[三机错峰DNN接管](F:/FalconReproduction/airsim_control/controller_architecture_study/results/dynamic_complexity_237d_v1/airsim_dnn_20261004T052726_426829Z/report.json)

- residual21x1024，FP32，RTX4060 Laptop，三机共享一个推理服务。
- T=D=50ms，三机错开约16.67ms；现有全局C2参考轨迹、标称5m/s。
- 1200/1200请求及DNN命令完成，MPC命令0，15/15次穿门，零超时、零错误。
- 计划释放至输出：平均15.14ms，P99 24.03ms，最大观测28.50ms。
- 排队：平均0.073ms；实际采样至输出：平均7.206ms，P99 11.493ms。
- 这是一个20秒场景的观测结果，不是WCET、全局安全或全部模型的飞行保证。

上一次同步接管在约16秒时出现一个53.90ms响应，超过当次50ms期限，试验被中止。错峰是重要调度基线；不能把一次同步漏期解释为必须使用超网。

## 2. 现在要搭建的整体框架

```text
飞行状态 + 已知全局参考轨迹 + 活跃任务/资源状态
                       |
                       v
状态条件的控制有效期/误差收益估计 [待实现、待校准]
                       |
                       v
模型 × 周期候选的性能/响应表       [本次实现采集入口]
                       |
                       v
联合选择器
  - 静态、同构三机候选选择         [本次实现]
  - 在线、逐无人机模型/T选择       [下一阶段，未实现]
                       |
                       v
共享推理服务 + 调度 + 控制下发    [已有三机实跑通过]
                       |
                       v
状态、误差、延迟、期限反馈        [已有逐帧日志]
```

先使用常驻独立模型池，超网是后续替换的计算后端，而不是现在的前提。保留同样的参考生成、输入合同、输出转换、调度和记录口径，才能公平比较模型池、固定宽度多出口和弹性共享网络。

研究问题仍是“何时额外精度值得等待，状态/任务变化时如何分配计算”，不是“强行证明共享网络一定最好”。

## 3. 本次代码能做什么

入口：[joint_controller_sweep.py](F:/FalconReproduction/airsim_control/controller_architecture_study/joint_controller_sweep.py)。

默认四模型：plain4x256、plain4x512、plain6x1024、residual21x1024。

默认三个周期：50、75、100ms；每个候选均D=T。一共12个候选，复用现有residual21x1024/50ms报告后只需11次新试飞。

- 复用现有manual-Play三机入口，每次三机同模型、同周期、同相位策略。
- 周期与控制命令保持时间一起改变，不只是放宽deadline。参考预测间隔仍为0.1秒，不改教师/特征合同；新周期的闭环效果由实际试飞评估，不能预先保证。
- 四模型始终常驻。不会训练、改权重、改数据、改归一化、改划分或打开第二个UE4实例。
- 每次飞行独立保存。汇总和选择结果增量保存；再次运行同一命令会跳过已完成的候选，包括已记录的期限失败。
- 中断未完成的尝试保留原文件，续跑使用新attempt目录，不删除或覆盖旧报告。
- 普通deadline中止作为不可行候选保留，继续下一个配置；碰撞、连接失败、CUDA异常、飞行保护异常停止套件，不自动反复压测。
- 校验模型、归一化、参考路线、快预处理、运行设备和精度的一致性。已完成报告、轨迹和请求文件保存SHA256，续跑检查文件未改变。

### 选择口径

1. 先要求当次完整飞行/实时验收通过，且三机sim/wall比例均在0.95～1.05之间。
2. 在相同参考时钟区间 `[0, route.duration]`，积分平方位置误差并除以区间长度；三架无人机等权平均。报告MSE的单位为m²、RMSE的单位为m。
3. 实际采样不均匀，按时间积分，而非把不同周期的样本均值直接比较。积分不包含末尾长时间悬停。首个观测值向零时刻常值延伸；中止且未覆盖完整区间的轨迹不补齐，不参与选择。
4. 在观测可行候选中选择上述误差最小的配置。另报响应P99/max、超时、裁剪命令数和平均服务负载估计，不把MACs当实际延迟。

这个性能代理**不是原论文J，也不是SVM性能函数**。先得到可执行、可比较的静态基线，不人为组合量纲不同的控制MSE和距离。

`selection.json`只建议一个观测候选，不自动控制飞机；`coverage_complete`说明候选是否全部测完。一个validation场景、每配置一次，不能宣称统计显著最优、跨场景最优或完整FALCON硬件协同优化结果。

## 4. 用户运行：一整段命令

手动打开现有Blocks/FalconEmpty并Play，保持ClockSpeed=1，停止GPU训练。若还有其他实验生成的门，先Stop/Play一次。不要同时运行另一个控制程序。

11次新试飞的测量时间约220秒，另加模型加载、预热、场景重置和记录开销。不承诺总运行时间；发生异常会提前停止。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'

Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $trainPy)) { throw 'Python 路径不存在' }

& $trainPy -m controller_architecture_study.joint_controller_sweep --device cuda --phase staggered --models plain4x256 plain4x512 plain6x1024 residual21x1024 --periods-ms 50 75 100 --environment-id 48 --route-index 0 --reuse-report .\controller_architecture_study\results\dynamic_complexity_237d_v1\airsim_dnn_20261004T052726_426829Z\report.json --resume

if ($LASTEXITCODE -eq 2) {
    Write-Warning '候选采集完成，但没有观测可行配置。请保留报告。'
} elseif ($LASTEXITCODE -ne 0) {
    Write-Warning '套件中止；已完成结果保留。请先查看异常，再决定续跑。'
}
```

输出目录：

`F:\FalconReproduction\airsim_control\controller_architecture_study\results\dynamic_complexity_237d_v1\joint_model_period_v1`

- `summary.json`：所有成功/失败候选、统一积分指标、完整来源和套件状态。
- `selection.json`：当前观测最好的静态配置；未测完时会标注覆盖不完整。
- `trials/<模型_T周期>/attempt_NNN/`：原三机入口的报告、逐帧请求和飞行轨迹。

中断后复制同一整段命令，不清空目录，不重新训练。若是碰撞/连接/硬件错误，应先处理；续跑不会自动忽略已有此类错误。更改模型来源、代码或配置需要新输出目录，不能与原套件冒充同一实验。

### 仅读取性能表作选择，不飞行

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
& $trainPy -m controller_architecture_study.joint_controller_sweep --select-only --period-cap-ms 75
```

这里75ms是调用者提供的周期上限，用于测试决策接口，**不是系统计算出的T_max，更不是安全保证**。没有候选满足上限时返回不可行，不偷偷扩大上限。

## 5. 下一阶段，不再把项目困在重复验证里

1. **本轮完成后**：查看12配置的实际闭环—响应折中，冻结观测静态联合基线。先不重复全部18模型、不大规模多seed、不立即训练超网。
2. **在线模型池决策**：将活跃任务数量、观测负载与状态条件收益接入联合选择器；动态加入/退出先用请求回放，再在必要的实际闭环场景检查切换和更新周期。选择器自己的耗时、切换控制跳变、任务丢弃均计入。
3. **有效期模型**：单独推导/校准状态条件T上界及其与响应时间的关系，明确假设、量纲与误差界。禁止把DNN输出混合MSE直接当距离余量；禁止把P99当WCET。
4. **共享计算后端**：先固定宽度多出口，再弹性宽度/结构；沿用相同选择预算和控制指标比较。多个独立模型的权重不能直接拼接就称为已训练超网，出口也不能默认越深每个状态越准。
5. **必要性判断**：如果优化T和错峰已解决实际负载问题，就接受该场景不需要超网；只有共享或中途调整产生可重复的额外控制收益，才推进其训练和FPGA实现。

## 6. 本次本地验证边界

- 新入口的CPU单元测试覆盖：时间积分、悬停排除、未覆盖窗口、D=T检查、来源/设备检查、期限失败保留、异常停止、续跑/孤立已完成报告恢复和成功报告复用。
- 原动态探针和快预处理回归测试共同通过；没有修改原三机入口、推理服务或控制器数学实现。
- 已对用户成功报告完成真实文件导入/评分冒烟：活动参考区间三机综合RMSE约0.3661m；无新试飞、无GPU推理。
- 75/100ms、新候选模型、完整12配置和在线选择收益均尚未实跑，留给用户执行。
