# 96k / 300D DNN：手动 AirSim 渲染试飞

## 使用边界

这是新 300D 专用入口，不使用旧 12D gate runner，不需要先导出 TorchScript。
直接读取本轮 `best_model.pt`，核对模型/数据来源，使用同一套 300D 编码与训练集归一化。
重放 `routes/envXX_routeYY.json` 中保存的 C2 多项式，不重新拟合参考轨迹；参考时钟每周期前进 0.05 s。
未来窗口为 21 点、间隔 0.1 s。DNN 每次输出 4 维物理控制，不调用 ACADO 求解、不由 MPC 回退控制。
输出限幅沿用教师编译范围 `[2,-3,-3,-2]` 到 `[20,3,3,2]`；报告记录限幅次数。

入口只连接手动打开并点击 Play 的实例，不启动第二个 Unreal 窗口。
它会重置当前 AirSim 仿真、清理旧 Falcon 生成门并创建本次 5 个门。因此不要同时运行采集器或其他飞行控制脚本。
不会修改数据集、模型或训练历史，只新增模型目录下的 `flight/` 日志。

Drone1 初始状态来自参考轨迹的空中静止起点，约 z=-4 m，并非地面起飞过程。
参考巡航/过门速度 5 m/s，起步、制动时速度变化。测试默认为至少 20 s，结束后调用 AirSim hover()；碰撞/越界/严重周期超时也会结束循环并尝试 hover()。
hover() 是结束清理，不是飞行中使用 MPC 替代 DNN。

本轮三个模型的 Validation 标准化控制 MSE：

| 隐藏层结构 | Validation MSE |
|---|---:|
| 4×128 | 0.029516 |
| 8×256 | 0.040490 |
| 8×512 | 0.037135 |

恢复类误差仍较大，不能由离线指标保证不撞。先测 4×128 是按验证指标排序，不代表它的真实飞行一定最好。
选环境组 48（Validation），路线 0 为近直线，路线 3 为偏航/升降组合；不是读取 Test 挑模型。

## 1. 打开 AirSim

若尚未打开，在 CMD 中运行：

```cmd
"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"
```

打开清空障碍物的 FalconEmpty 地图并点击 Play。只保留一个运行实例；如果已经手动打开并 Play，不再运行此启动命令。

## 2. PowerShell 初始化

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
$runs = '.\controller_architecture_study\results\decision_nmpc_96k_v2\runs'
```

这里沿用已验证可加载 AirSim 和 PyTorch 的本地依赖路径；无需安装或修改虚拟环境。
重启 PowerShell 后需重新定义这些变量。复制代码时不要包含 `(base) PS ...>` 提示符。

## 3. 先试 4×128

```powershell
& $trainPy -m controller_architecture_study.fly_decision_dnn --checkpoint "$runs\plain_mlp_h4_w128_seed0\best_model.pt" --environment-id 48 --route-index 0 --device cpu --keep-gates
```

开始前会暂停仿真 2 s，门生成后再开始飞行；不需重复点 Play。
控制期内终端可能仍打印原循环的 `paper ... solve=...` 字段，此处实际测量的是 DNN 编码/推理，不是调用 MPC。

若这条路线正常，再试含转弯、升降和航向偏移的路线：

```powershell
& $trainPy -m controller_architecture_study.fly_decision_dnn --checkpoint "$runs\plain_mlp_h4_w128_seed0\best_model.pt" --environment-id 48 --route-index 3 --device cpu --keep-gates
```

## 4. 用同一路线换模型

```powershell
& $trainPy -m controller_architecture_study.fly_decision_dnn --checkpoint "$runs\plain_mlp_h8_w256_seed0\best_model.pt" --environment-id 48 --route-index 0 --device cpu --keep-gates

& $trainPy -m controller_architecture_study.fly_decision_dnn --checkpoint "$runs\plain_mlp_h8_w512_seed0\best_model.pt" --environment-id 48 --route-index 0 --device cpu --keep-gates
```

建议分别运行、观察和保存结果；若失败，不把失败当作已经通过。定位后再决定是否扩展场景。
将 `--route-index 0` 改为 `3` 即可对比组合路线。三者使用完全相同的保存轨迹和初始化规则。
没有添加改变目标速度的参数，避免本轮比较中同时改变训练任务。

## 5. 查看结果

每次写到对应模型目录：

`...\runs\plain_mlp_hX_wY_seed0\flight\env48_route00_时间戳.json`

同名 NPZ 保存实际 300D 输入、限幅前/后控制、位置、速度、参考位置、推理耗时。异常也保留已经完成推理的部分轨迹。
JSON 中 `status=PASS` 表示运行期间未检测到错误且五个门都从有效开口穿过；不是所有环境或恢复状态的保证。
关注：`valid_crossings=5`、`error=null`、`clipped_cycles`、`max_position_error_m`、推理延时与 `deadline_misses`。
失败时进程退出码为 1，JSON 给出碰撞对象、越界或超时原因；将报告发回来即可。

完成本轮试飞后，在继续训练/生成数据之前移除 AirSim 的临时依赖路径：

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
```

## 实现验证，不等于真实试飞结果

- 14 项相关回归通过：存储轨迹重放、编码一致性、归一化/限幅、模型来源、失败部分日志和既有采样流程；没有真实模型训练或 AirSim 连接。
- 三个真实检查点各取 32 条 Validation 输入，单帧运行时与离线批推理最大物理输出差异分别约 3.81e-6、5.72e-6、2.86e-6。
- 本次没有由 agent 启动飞行，真实门穿越结果由你运行后获得。
