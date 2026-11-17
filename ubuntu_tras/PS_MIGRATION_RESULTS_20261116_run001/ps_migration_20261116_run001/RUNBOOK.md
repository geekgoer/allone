# 路线B PS迁移：运行与边界

本目录是迁移实验副本。原release001、18个模型、SD镜像、DPU驱动、归一化参数均保留。任务主入口为最新义务日历，不是旧ABC runner；ABC只是该日历内部有界候选搜索的一部分。

## Windows端的作用

目前Windows仅运行AirSim Blocks：启动Blocks并点Play，以太网`192.168.77.1`，RPC监听`0.0.0.0:41451`。不要同时启动旧的Windows控制framework，不要同时让其他客户端控制Drone1/Drone2。无需Windows替PS计算特征、排队、调用UDP推理或选择模型。

此迁移链为：Windows AirSim状态 → 以太网原生RPC → 板端PS参考/237D/保持分析/日历 → 本地DNNDK DPU → PS控制转换 → 原生RPC回AirSim。它仍有AirSim RPC网络开销，不是纯板内端到端系统。

`native_io.cpp`中的WindowsQPC/HANDLE定时/线程平台接口已做Linux适配，细节见`NATIVE_IO_PORT_MAP.md`。没有将旧Win32 DLL直接放到ARM运行。真实推理通过`local_dpu.cpp`包含原冻结`pool237_runner.cc`的Pool实现；不走CPU DNN、教师求解或VART/xmodel。

## 符号、输入和计时

- C++原生RPC直接发送物理`wx,wy,wz`；不复制Python客户端内部第二次pitch/yaw翻转。
- 特征构造输出RAW237。归一化仅由原Pool做一次，再补齐240通道；输入越界仍status3拒绝，不裁剪。
- 所有本地耗时为PS单调时钟；AirSim模拟时间戳另列。DPU调用时间不是纯硬件时间；命令+ping返回只是同原参考实现的代理，不是实测电机施加时刻。
- 每机一个在途控制槽，单DPU不可抢占。发布日历独立于请求完成；忙/过期/跳过/取消均需记录。
- 新的`config_dual_001.json`由实际ARM/DNNDK计时和三轮固定单机数据生成。TRAIN512选择分数不变，final480分数仅描述，不能据其重选模型。经验最大值×1.25不是WCET保证。

## 已验证的单机复跑

Ubuntu主机：

```bash
PS_WORK=/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001
bash "$PS_WORK/build_flight.sh" "$PS_WORK/build_flight_NEW"
```

将新构建另名上传后，修改脚本中的版本路径再复跑。当前**已实测**版本`build_flight_004/ps_fixed_control` SHA256：
`8034206332be9448b129c8019090c9c73c8d4bd3b76d8f2a46f9d039361eb989`。

板端当前实测入口：

```sh
sh /home/root/ps_migration_20261116_run001/run_single_trial.sh 005
```

编号必须全新，脚本拒绝覆盖已有prepare/guard/raw目录。`002/003/004`是正式三轮；`001`为导出失败但完整journal保留的诊断轮。每轮重新设置冻结参考匹配初态1.5s，固定ID11/T50ms/20s/Drone1；不是静止起飞验收。

场景准备只删除本实验`PSMIG20261116_`前缀的对象，遇到外部`Falcon.*Gate.*`会拒绝，不删除其他实验的门。结束/故障后暂停物理；未知DPU占用会拒绝独占实验，不强杀。

回收与分析：

```bash
scp -r axu3egb:/home/root/ps_migration_20261116_run001/fixed_single_005 "$PS_WORK/"
python3 -B "$PS_WORK/features/analyze_flight.py" \
  --data "$PS_WORK/fixed_single_005" --output "$PS_WORK/analysis_fixed_single_005"
```

必须回收整个目录，不仅summary；`events.jsonl`是连续日志，`requests.jsonl`/CSV是最终导出。程序退出0不等于飞行通过，分析也会检查错误、碰撞、过门、窗口完整性和模拟时钟比例。

## 义务日历后续入口

`adaptive_runtime.cpp`是实际义务日历接线；编译/板端实测状态见`STATUS.md`，不得把源代码存在或fixture通过冒称双机已跑完。

规划通过snapshot→bounded search→原子install；实际发布由LiveCalendar独占管理，派发时take选择的模型真正传给DNNDK。实际返回后dpu_done；实际命令确认后command_return更新控制版本，再以新状态配对分析，最后finish释放在途所有权。

计划跳过默认关闭。只有上一阶段通过后才单列启用实验；跳过必须有同控制版本的新鲜保持预算与稳定阶段资格，补更必须通过实际DPU和命令路径，不能把计划事件当作补更成功。

## 保留与恢复

所有命令、退出码及合并输出在`logs/*.json`与`logs/*.log`。旧错误不删除：本地DPU测试第一次JSON临时对象失效、一次上传未结束导致Text-file-busy、单机诊断最后导出计数错误均保留。

独占脚本仅识别冻结release001的既有5239服务，正常停机后恢复相同模型/端口并用新日志；未知任务拒绝。若实验前该服务本来未运行，实验不会擅自启动它。当前服务状态以环境快照/最终记录为准，不能依据旧PID推断。

## 已完成的双机与独立压力条件

正常双机正式版为板端`bin/ps_obligation_control_v4`，对应本地`build_adaptive_004`；固定/动态101–103各3轮均完成。Windows只运行Blocks/Play，不启动旧Windows控制器。确保无人争用Drone1/Drone2和DPU后，选全新编号复跑：

```bash
ssh axu3egb 'sh /home/root/ps_migration_20261116_run001/run_dual_trial_v4.sh fixed 104'
ssh axu3egb 'sh /home/root/ps_migration_20261116_run001/run_dual_trial_v4.sh adaptive 104'
```

每条命令会重置本实验场景、匹配参考初态、验证/预热真实模型、运行20s并暂停。不能同时启动两条。

压力版为板端`bin/ps_obligation_control_v5`，对应`build_adaptive_005`。它另加有限窗口内必须保留原补更后继的守卫。仅改独立压力配置，不改冻结模型/归一化，不添加假推理耗时：

```bash
# T30ms/每机，允许周期仅[30]。off和on各001–003已完成。
ssh axu3egb 'sh /home/root/ps_migration_20261116_run001/run_pressure_trial.sh off 004 30'
ssh axu3egb 'sh /home/root/ps_migration_20261116_run001/run_pressure_trial.sh on 004 30'
# T20ms/每机：001为单次探索对照，参数末尾改为20。
```

这些不是正常T50工况，也不是故意注入CPU/DPU延迟。实际结果均未提交跳过，所以**不建议只继续增加重复次数来宣称完成第三阶段**；应先处理约10ms搜索过程中上下文过期的问题。

最终分析（目录名称须全新）：

```bash
PS_WORK=/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001
scp -r axu3egb:/home/root/ps_migration_20261116_run001/dual_adaptive_104 "$PS_WORK/"
python3 -B "$PS_WORK/features/analyze_obligation.py" \
  --data "$PS_WORK/dual_adaptive_104" --output "$PS_WORK/analysis_dual_adaptive_104"
```

`calendar_final.json`、`plans.jsonl`、`events.jsonl`、`requests.*`、`execution_identity.json`、初末状态及prepared_source都需要保留。跳过验收要求`skip_recovery_experiment_validated=true`且至少有实际planned_skip；本次所有真实试验此字段均false。

## 从交付包重新构建

交付包有源码、逐请求原始数据和本轮已测ARM程序，不含Yocto SDK、DNNDK开发库或冻结交接依赖的大文件副本。仍需本机已校验的`../route_b_ps_handoff_20261116`、既有SDK和`vitis_ai_dnndk_axu3egb`；对应模型/外部依赖哈希见各build的FROZEN_EXTERNAL_SHA256及source_proof_001.json。它不是可替换SD卡的镜像。

```bash
PS_WORK=/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001
cd "$PS_WORK"
bash build_local.sh "$PS_WORK/build_local_repro001"
bash rpc/build.sh arm
PS_LOCAL_BUILD="$PS_WORK/build_local_repro001" \
  bash build_flight.sh "$PS_WORK/build_flight_repro001"
PS_LOCAL_BUILD="$PS_WORK/build_local_repro001" \
  bash build_adaptive.sh "$PS_WORK/build_adaptive_repro001"
```

构建目录必须不存在。重构建产物应以新版本名上传，并检查/运行ARM测试；不要覆盖正在执行的文件。原实测二进制SHA在RESULTS.md。顶层构建脚本最后增加了可指定依赖目录及生命周期测试编译，默认运行算法未改变；实测版本的构建源码快照保存在各build/source中。
