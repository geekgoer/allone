# 迁移分支进度

2026-11-16：已认领PLAN中的四个互不重叠源码目录。交接快照2613文件HASH_PASS；压缩包缺失单列。Windows实际IP/监听经用户确认，PS→AirSim TCP连接实测成功。开始C++数值fixture、AirSim原生RPC、有限日历调度核心和本地DNNDK接口。尚无新闭环/调度/跳过收益结论。

## 本轮实际进展（持续更新）

- 仅补查缺失native_io/接口的新统一快照，1878个清单文件通过；模型、冻结case、校准、数值fixture与原快照相同。未改原模型和系统镜像。
- ARM特征/参考/保持界与10组fixture对齐；2370值误差0，3组native物理角速度线协议对齐。18模型每个32条黄金全PASS；每个300次本地服务计时完成。3条历史越界全部status3复现。
- 单机诊断`fixed_single_001`：400条delivered，但最终汇总有旧JSON/C++14求值顺序错误；原始events完整、failure保留，不算正式轮次。修复版为build_flight_004，板端bin/ps_fixed_control_v4。
- 三轮正式`fixed_single_002/003/004`：各400条delivered，未记录拒绝、drop、late、碰撞；各5门观察通过；约0.166–0.171m全窗口跟踪RMSE（边界时间代理、采样插值口径）。结果在analysis_fixed_single_formal_001。RPC返回是ping代理，不是物理电机确认。
- 真实ARM调度器：6组冻结fixture、43核心及51生命周期检查通过。300条离线规划计时完成，不冒称飞行；由其和1200条正式单机记录生成config_dual_001.json，经验系数1.25，非WCET。
- 双机义务日历实际接线/测试进行中；计划跳过默认关闭。尚不宣称双机或跳过闭环验收。

## 最终更新：2026-11-16

- 正式单机3轮、双机固定3轮/义务日历3轮均通过实际DNNDK链路、日历/计时和采样飞行审计，正式请求合计6000条。正常双机全部ID11/T50，无实际模型/周期变化，不宣称调度收益。
- 另获用户授权的压力实验：T20ms关闭/开启各1轮，T30ms关闭/开启各3轮。模型、成本初始配置和归一化不改，无伪造延迟；受限周期集合单独标识。压力合计12004条，均delivered。17轮总18004条。
- **未完成的验收：真实计划跳过→原下一周期实际补更。** T20启用组717条eligible权限全部距目标释放不足15.432ms预算；T30启用三轮89个通过检查的含skip候选全部context_changed未安装。实际skip=0，不能把开启选项或单元测试视作已验证。
- 正常自适应规划最大wall17.823/24.987/7.963ms，长尾没有被证明消除。后续应优化规划热路径和合法提交窗口，保持CAS/控制版本/期限守卫，不强行提交旧计划。
- 增加锁内当前时钟快照，修复双机v2诊断002中的早停，保留失败原始记录；v4为双机正式版，v5另含有限窗口补更后继守卫，仅用于压力。
- 最终分析器修正暂停后读回延迟造成的clock假阴性；原数据与旧分析目录保留。最终分析用analysis_fixed_single_formal_002、analysis_normal_dual_v4_formal_002、analysis_pressure20_pair002、analysis_pressure30_formal_002。24项分析回归通过，ARM77生命周期/67接线检查通过。
- 最终原release001整包校验通过；只读probe确认AirSim is_paused=true。当前没有常驻DPU控制/UDP服务进程（实验前此阶段也未运行，独占脚本没有擅自启动）。见environment_final_001和logs/final_*。
- 入口说明见RUNBOOK.md；结论与限制见RESULTS.md；实现解释见MIGRATION_IMPLEMENTATION.md；跨阶段结果见CROSS_STAGE_FINAL_001.csv。
