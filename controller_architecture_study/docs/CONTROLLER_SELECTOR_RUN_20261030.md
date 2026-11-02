# 选择器第一阶段：配对恢复数据与小模型训练

这是可运行的数据/训练流水线，没有接管现有多机飞行策略，也不修改任何控制DNN权重。

## 数据协议

使用现有8个237D FP32模型，同一个起始状态和旧控制分别测试全部候选。默认68个独立路线/状态组：训练32、验证8、校准20、测试8。每组8模型×3个初次接管延迟（20/50/80ms），共1632次配对短恢复试验。

每次观察2秒，按绝对模拟时间日历每50ms下发，命令间隔上限60ms；后续每次输入额外施加10ms模拟延迟，实测输入年龄不得超过20ms。实际初次接管延迟、后续输入年龄和控制间隔均保存，误差超出协议则标签无效。

这是暂停/步进物理后施加指定延迟的控制能力实验，CPU推理/RPC耗时另记；不把暂停期间的推理当成真实零时延系统，也不把这组数据用作端到端实时性成绩。20/50/80ms是延迟采样点，不宣称覆盖所有更短延迟或所有更新模式。命令时间是暂停物理下的RPC接受代理，不是电机实际生效测量。

起始状态含位置、速度、姿态偏差，用于学习恢复资格，属于离线系统辨识/恢复试验；不是“自然飞行时给静态方案额外加扰动”的比较实验。另11架不参与恢复标签、不会借此冒充多机压力成绩。

恢复余量综合有限门框间隙、末段持续250ms位置/速度误差，以及沿参考任务的前进程度。碰撞、控制恢复失败产生负样本；基础设施、初态匹配、时钟或延迟执行问题产生无效样本。一个配对组只有全部候选/延迟数据有效、初始241D特征匹配时才进入训练。

每个试验最多两次尝试，仅重试无效数据；正常控制失败不重试以挑成功。所有attempt目录保留。源码和模型身份冻结，采集代码变化时要求新数据目录，避免悄悄混入不一样的标签协议。

## 运行

先保持12机工程处于Play，关闭编辑器后台降频。命令可从任意PowerShell目录执行。

小批采集两个状态组（48次），可以重复运行以跳过已保存有效试验：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\FalconReproduction\airsim_control\controller_architecture_study\run_controller_selector.ps1" -Stage Pilot
```

全部采集后自动训练并导出：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\FalconReproduction\airsim_control\controller_architecture_study\run_controller_selector.ps1" -Stage All
```

训练只使用CPU上的8673参数小模型，默认600轮，验证集选择best，保存latest支持同参数续跑。没有启动GPU训练。默认不评估测试集，也不把测试标签用于模型、校准或时间输入范围选择。

若只想继续采集或只训练，使用 `-Stage Collect` 或 `-Stage Train`。新协议/改源码的试验用 `-Data` 指定新目录。测试集只在规则冻结后显式运行训练模块的 `--evaluate-test`，测试结果不得反过来挑参数。

## 产物

默认根目录：`controller_architecture_study/results/controller_selector_237d_v2`。

- `data/plan.json`：分组、初态、路线、旧控制、模型身份、延迟与恢复协议。
- `data/trials/*/attempt_*`：初态回读、实际命令、轨迹、控制/时序结果。
- `data/collection_summary.json`：有效/无效/缺失情况与可训练完整组。
- `data/paired_rows.json`：仅完整配对组的241D输入、实际延迟、候选描述与余量标签。
- `predictor/best_model.pt`、`latest_checkpoint.pt`：小预测器与续训状态。
- `predictor/weights_fp32.npz`：纯NumPy运行，归一化仅由训练数据计算。
- `predictor/selector_scripted.pt`：TorchScript版，输入先使用配套归一化。
- `predictor/selector_spec.json`：冻结候选池、描述、计算成本和校准修正。
- `predictor/training_report.json`：余量误差、误放率、选择覆盖率、选中模型与计算成本差距。

`controller_selector.predict.Selector`在初始化时加载一次权重，后续调用无磁盘模型重载。它只提出推荐，不安装配置、不发送飞行命令；全局准入、输入生效状态覆盖和解析安全检查仍必须通过。

独立校准组数不够时不会捏造有限修正量，推理返回无已校准推荐；这不自动要求停止所有无人机。预测器尚未取得物理安全或反馈分布下统计保证。模型种类或权重改变后需要重新校准，不能把结构描述相同当成函数相同。

## 当前阶段的判定

代码检查、导出一致性和测试标签泄漏检查使用合成数据；这些测试产生的小模型不能当成真实恢复选择器。必须等完整配对实飞数据足够后，才有可评价的训练结果。

若发现多数状态所有模型都成功，仍报告该结果，不通过挑选失败场景强迫选择器显得必要。当前有效旧策略作为后续比较基线保留。
