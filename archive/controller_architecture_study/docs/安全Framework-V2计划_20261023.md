# 多机框架 v2：修复范围与连续执行协议

日期：2026-10-23。v1 的 8 机静态配置可行，但动态释放把前端等待漏进了模型：6 架截止时间中止，不能称为安全完成。v1 结果不覆盖，三个源码文件已按原 setup 哈希备份到 `results/safety_framework_comparison_v1/n8_net2/frozen_source`。

## 不改变的条件

既有 237D DNN 权重、归一化、参考轨迹、初始状态族、门洞净宽、机体碰撞几何、距离/速度预算 B 和截止时间过期判据全部不变。无 MPC fallback，不把悬停或中止算成功；不操作电脑屏幕。默认 8 架实际参与飞行（Play 中总共启动 12 架）。本轮每架目标仍为 gate3/gate4，统一任务窗口 3 秒。

## v2 的最小修复

1. **时间记账**：下一请求规划用命令 RPC 返回后的时刻，不能继续用发送前的旧 now。物理截止时间仍锚定原观测时间加 B，不因晚了而延期。
2. **完整任务成本**：从新状态采集开始，到命令 RPC 返回结束，包含状态/碰撞 RPC、特征生成、IPC、DNN、输出处理和命令 RPC。保留各阶段时间戳。
3. **一个公共完整任务服务资源**：双方都最多一个在途完整 job；其余请求按静态固定优先级/动态 EDF 等待。这样用同一个资源模拟完整-job 成本，不再用“只排 DNN 队列＋固定前端裕量”假装完整响应。
4. **代价估计**：从本轮有效静态名义飞行获得各模型完整任务的 p95，使用 `max(5ms,1.25*p95+2ms)` 做预测成本，并同时保留样本数、最大观测值。该值明确不是 WCET，也不是严格响应上界。实际截止时间继续监控；预测不可行时退到立即尝试而不是延长截止时间，这一状态单独记录，不当作安全认证。
5. **更强的静态搜索**：三种网络×六个共同周期（25/50/75/100/125/150ms），再做相位、优先级、末位回路网络/相邻周期的局部邻域，选择最高实测总 J 的可行候选。

这版是保守的完整任务调度先导，不宣称利用了多任务层间抢占。已有分段能力和单元测试仍在，但在一个完整任务在途限制下，不能用本轮结果证明层级抢占收益。之后若本轮框架有效，再分开扩展双资源流水与分段抢占，避免把正确性问题同时摊到太多模块。

## 自动执行顺序

1. 单元测试：旧时刻回归、完整任务代价、全局已承诺任务检查，加已有输入/分段/评分测试。
2. `calibrate`：完成三模型六周期和邻域静态试飞，保存完整成本档案，随后验证相同选定配置的新方法名义飞行。若失败停止，不继续安全优势比较。
3. `discover`：只有校准通过后才跑原先声明的 16 个初始偏差×两个方法；交替顺序、保留所有正负结果。
4. `confirm`：有干净差异才做三组新重复（初值 ±1cm），并加入最快已测可行静态配置（若与总 J 最优配置不同）。没有差异就如实报告，不制造确认结果。
5. 输出结果报告与下一阶段判断，检查 AirSim 可响应且临时门已清理。

## 判断边界

J 是 Wx=I10、共同网格插值的 Eq.(4)-form 近似指标，原文权重和完整设计空间未复现；基线必须称为 FALCON-style 受限近似。不能把小模型集合/离散周期没搜全、Windows 队列噪声或稀采样误差当作学术优势。

阶段性正面证据需要：静态方法发生实际碰撞，动态方法完整穿门，双方时钟与执行正常且无实际截止时间违约。仅动态 miss 更少、仅零碰撞但中止、仅一个事后挑选的胜例，都不够。即使确认成功，也只是给定场景的有限实验证据，不是形式化安全证明，更不是 FPGA 结果。

v2 结果目录：`controller_architecture_study/results/safety_framework_comparison_v2`。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\controller_architecture_study\run_safety_comparison.ps1 -Python $trainPy -Vehicles 8 -NetWidth 2
if ($LASTEXITCODE -ne 0) { throw '阶段未通过；结果保留，先检查而非盲目重跑。' }
```
