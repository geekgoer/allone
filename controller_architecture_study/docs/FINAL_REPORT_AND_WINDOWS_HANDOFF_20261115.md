# Main18 PTQ新部署：实际板端验收与Windows交接

日期：2026-11-15（Asia/Shanghai）。本轮已实际执行完毕，不是待执行计划。

## 1. 结论

- 18/18原始237D模型完成候选冻结、同样本480条评测、B1152 ELF编译和ARM交叉链接。
- 18×32=576条模型自有黄金全部通过，板端与新INT8参考逐值一致。
- 18×480=8640条正式UDP请求全部成功，拒绝0、超时0、其他状态错误0、协议/传输错误0、INT8一致性错误0。
- 在这8640条中，板端与新CPU定点参考的最大物理输出绝对差为**0**。
- 全18模型的新板端MSE均低于旧版，降低约10.28%–96.36%。
- FP32与INT8模型误差排序的Spearman由旧版**−0.246646**变成新版**+0.762642**。
- MAC与INT8误差的Spearman由旧版**+0.715170**变成新版**−0.403509**。
- 预先约定的工程目标全部达成，但并非“每个更大的模型必然更准”，也没有达到FP32原精度。
- 本轮新INT8最低MSE为ID8（3×1024）约**0.003406430**，不是最大模型；这个结果保留，不人为排序。

**测试服务已停止。** 本次只启动并停止了新包PID779、端口5239的服务。结束时DPU为Idle；旧包、旧服务配置、自启动、启动镜像、驱动、FP32权重、网络设置未修改。

## 2. 18个模型实际板端MSE

统一口径：`mean(((物理预测输出 − 物理MPC教师输出) / u_std)^2)`，对完全相同的480条输入和四个通道计算，无量纲。

| ID | 隐藏层×宽度 | MAC | FP32 | 旧板端INT8 | 新板端INT8 |
|---:|:---|---:|---:|---:|---:|
| 5 | 2×128 | 47,232 | 0.001343308 | 0.009256300 | 0.008304936 |
| 9 | 3×128 | 63,616 | 0.000796323 | 0.017062206 | 0.009418897 |
| 10 | 3×256 | 192,768 | 0.000593740 | 0.006669738 | 0.005467397 |
| 1 | 1×1024 | 246,784 | 0.002888587 | 0.011328934 | 0.009370963 |
| 12 | 4×256 | 258,304 | 0.000478308 | 0.008715536 | 0.005570292 |
| 13 | 4×512 | 909,824 | 0.000422885 | 0.007232159 | 0.003986975 |
| 15 | 5×512 | 1,171,968 | 0.000419491 | 0.009607078 | 0.004611680 |
| 4 | 2×1024 | 1,295,360 | 0.000498545 | 0.016022502 | 0.004470136 |
| 18 | 9×512 | 2,220,544 | 0.000451885 | 0.018937640 | 0.004259422 |
| 8 | 3×1024 | 2,343,936 | 0.000392548 | 0.020003695 | 0.003406430 |
| 11 | 4×1024 | 3,392,512 | 0.000324922 | 0.013615234 | 0.003574870 |
| 14 | 5×1024 | 4,441,088 | 0.000286501 | 0.016144636 | 0.003693311 |
| 3 | 21×512 | 5,366,272 | 0.000573786 | 0.041812656 | 0.005713577 |
| 16 | 6×1024 | 5,489,664 | 0.000271972 | 0.009203549 | 0.003893476 |
| 7 | 33×512 | 8,512,000 | 0.000467889 | 0.106901695 | 0.007054096 |
| 17 | 9×1024 | 8,635,392 | 0.000305372 | 0.038286510 | 0.003515724 |
| 2 | 21×1024 | 21,218,304 | 0.000287820 | 0.081738690 | 0.004335227 |
| 6 | 33×1024 | 33,801,216 | 0.000352461 | 0.187368458 | 0.006826872 |

原始精度、物理通道RMSE/P95/P99/max及未截断数值见`results/release001_actual_board_analysis/board_model_metrics.csv`和`board_analysis.json`。

![实际FPGA三组MSE](results/release001_actual_board_analysis/board_mse_vs_macs.png)

灰色为同源FP32、橙色为旧板端INT8、绿色为本次真实FPGA结果。横纵轴均为对数坐标，点旁数字为板端ID。

## 3. 数据与方法边界

- 每个模型480条，全部18模型共同有效行也是480条，不是不同模型各自选一批容易的输入。
- 480条是冻结前已锁定的历史工程对照，源自旧Windows5237基线；**不是一份全新未触碰的独立TEST**。真正TEST文件未使用。
- 候选在TRAIN4096内以3072校准、512尺度搜索、512方案选择冻结；没有根据这480条更换候选或重新搜索尺度。
- 沿用原FP32权重、237D输入契约与归一化，不做QAT/微调。FP32逐层W/b与源NPZ核对，未把subnormal权重清零；原始Caffe FP32重新计算480条。
- 新定点参考使用已部署的固定W/b与官方FixedNeuron，先与官方ConvolutionFixed图在32条黄金上核验，再上板测量；不是普通浮点图冒充INT8。
- 并行只发生在不同模型的CPU评测/编译之间；每个模型仍单线程、batch32。板端正式回放一次只有一条未完成请求，不是并发负载或切换成本实验。
- 板端CSV中的`run_api_wall_us`是DPU调用墙钟时间，不是单独的纯硬件执行时间；没有把主机时钟与板端时钟相减。
- 新量化改变了模型输出，旧版切换/控制连续性实验不能直接改标签称为本版本结果。那些实验如需对比本新包，应该另行重跑。

## 4. 新部署身份（Windows必须更新）

| 项目 | 实际值 |
|---|---|
| family / protocol | `independent237` / `IP37/1` |
| 板卡 | `192.168.77.2`，AXU3EGB |
| 新端口 | UDP `5239` |
| 新wire tag | `9253dd1fc87f8d7f`（64位十六进制） |
| 模型ID | 原numeric ID1–18不变 |
| ID11新kernel | `rq_73c051a35e68` |
| 输入 | 237个有限RAW float32；板端归一化并补3个零到240 |
| 输出 | 4个物理控制量：推力加速度m/s²、wx/wy/wz rad/s |
| 全18模型实际输入比例 | DNNDK getter读回`16` |
| 全18模型实际输出比例 | DNNDK getter读回`0.0625` |
| 硬件 | 单核DPU B1152，v1.4.1，150MHz |
| DNNDK | Vitis AI for Edge DPU1.2；N2Cube4.2 |
| 主机镜像ID | `sha256:b9d2842bf7328f07a0d3d9c4c80c783529fb8eeb27b0b1db2066b703e65fabae` |
| 冻结SHA256 | `9214c4fff3d7d1c2eeeff72e4f5238f80e1ce2f67cddba086282440362a22f8d` |
| normalization SHA256 | `fe9c44ee80aef1cf97138536a66a4ee3a126f459f2ef963370db354ada8f19ac` |
| B1152 DCF SHA256 | `cbd84eb836ef0de8c16ee90fd59994cb83237995d56f81156f5cb30f3c98b1da` |

主机包：`/home/cwc/dpu_work/237D_work/main18_requant_20261111/board_package_requant18_release001`。

板端包：`/home/root/board_package_requant18_release001`。18个ELF均已链接进新的`pool237_runner`；不要换回旧runner只复制新松散ELF。

不要沿用旧5237的tag，也不要把5238的单模型输入比例8身份混进本池。

## 5. Windows如何重新接板

### 5.1 先启动新服务（在板子终端执行）

本次验证后服务已停止，下面是**后续需要时**的启动方式。不要在新旧服务都处理控制请求时比较洁净时延。

```sh
cd /home/root/board_package_requant18_release001
netstat -anu | grep ':5239'
```

如果端口被其他进程使用，先核对归属，不要直接kill。如果空闲，选择全新日志名：

```sh
mkdir service_windows_001
./pool237_runner serve --port 5239 --inputs inputs_raw.txt \
  --warmup 100 --out service_windows_001/serve \
  >service_windows_001/server.log 2>&1
```

保持进程运行；另一终端查看`service_windows_001/server.log`直到READY。此命令不安装开机自启动。

### 5.2 Windows复制整个新包，先固定输入回放

在解压后的`board_package_requant18_release001`目录，用Windows现有Python运行，不需要Vitis/CUDA/Torch：

```powershell
python .\windows\board_validate_client.py --host 192.168.77.2 --port 5239 `
  --models 11 --group golden --count 1 --output .\windows_smoke_001
```

客户端从本包registry读新tag；不要只复制一个脚本而漏掉registry、normalization、refs与codec。正式复核：

```powershell
python .\windows\board_validate_client.py --host 192.168.77.2 --port 5239 `
  --models main18 --group final480 --output .\windows_board480_001
```

若只做Windows网络复核，可不传Ubuntu的golden日志；但本轮实际getter证据已归档在结果目录，不要把“未提供getter日志”冒称再次测得比例。

### 5.3 AirSim桥接程序的接口

使用包内`windows/pool237_udp.py`的`Client`，原型是：

```python
client = Client("192.168.77.2", 5239, int("9253dd1fc87f8d7f", 16), timeout_ms=1000)
reply = client.infer(11, raw237)  # 必须是原契约的237维RAW输入
if reply["status"] != 0:
    # 交给现有安全回退/拒绝处理；不要把错误响应当成有效控制输出。
    ...
else:
    controls = reply["U_raw_physical"]
client.close()
```

输入只在板端归一化一次。发来的必须仍是原`decision_convex_local_relative_pv_237d_v1`特征顺序/参考坐标，不得根据新的MSE表改feature语义。先影子测试，不直接替代MPC闭环。

## 6. 重要：37条起飞边界输入问题仍需单列

**本轮不是此前比例8的ID11重校准包。** 新Main18赢家仍是输入比例16，原`scaled >= 128 || scaled <= -129`拒绝检查完整保留，没有clip。

480条全部通过不代表37条历史起飞问题输入通过；本轮没有重跑它们，也不宣称修复。Windows恢复三机影子时应继续保留范围审计与安全回退，单独验证起飞加速阶段。不要因为工程MSE目标通过就把范围问题视为解决。

## 7. 原始数据、代码、命令在哪里

主根目录：`/home/cwc/dpu_work/237D_work/main18_requant_20261111`。

| 内容 | 根目录下的路径 |
|---|---|
| 新模型包、ELF、ARM runner及源码 | `board_package_requant18_release001/` |
| 每模型480条原始板端CSV | `results/release001_board480/requests.csv`（8640行数据） |
| 板端逐模型计数/物理通道误差/时延 | `results/release001_board480/summary.json` |
| 真实FPGA18模型表/图 | `results/release001_actual_board_analysis/` |
| CPU对照表/图/排名 | `results/release_001_final480_analysis/` |
| 板端32条×18黄金/getter/内存证据 | `results/release001_board_goldens/` |
| 本次常驻服务加载/最终计数/退出证据 | `results/release001_board_service/` |
| 冻结候选、完整原始评测/编译产物 | `release_001/` |
| 本次实际命令、日志和环境 | `execution_20261115_run001/` |
| 完整复跑SOP | `MAIN18_RELEASE_RUNBOOK_20261115.md` |

`execution_20261115_run001/run_host_jobs.py`只在不同模型ID之间并行；`run_post_validation.py`保存打包、编译、scp、黄金、UDP、停止本次进程等原始命令到`post_commands.json`。`analyze_board_release.py`从逐请求CSV重新计算结果，不根据480条挑选模型。原DNNDK C++推理逻辑和wire codec逐字节复用，头文件只更新新模型与新tag。

本轮曾人为中止最初的串行ID2评测以改为并行，原始日志中的`KeyboardInterrupt`是这次调度动作，不是模型数值失败。已完成批次通过身份/哈希校验后恢复，最终全部18模型DONE且真实板端PASS。

交接压缩包以部署包、源码、原始测量和报告为主，不内嵌Docker镜像或整个原始训练集；完整原权重、所有候选与大体积中间产物仍保留在原Ubuntu工作目录。完整重新训练/量化还需要原bundle及相应哈希依赖。不得把结果包误称为独立封存测试集。
