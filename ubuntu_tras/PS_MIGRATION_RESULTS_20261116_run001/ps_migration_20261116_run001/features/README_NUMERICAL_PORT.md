# 237D数值端口与保持分析

本目录只在新实验目录开发。冻结交接包、模型权重、归一化、服务及失败样本均未修改。本模块不进行DPU量化/推理，不连接板子或AirSim；父控制程序负责调用和逐段计时。

## 已验证的范围

- `feature_parity_host.json`：10组固定状态，每组237个float32特征，共2370个逐值比对；最大绝对误差0。另从冻结五次多项式重建21步参考，位置/速度/加速度/姿态最大差异`1.4210854715202e-14`；完整多项式→RAW237链同样最大差异0。
- 3组命令转换逐字段比对，阈值`1e-12`。另核对四元数整体符号翻转、当前加速度/角速度不进入特征、油门饱和后的真实物理推力反馈、有限门框保持分析边界和时钟顺序。
- `hold_reference_parity_host.json`：10状态×3实际控制，30组保持分析与原始Python源函数逐值核对，最大差异`4.263256414560601e-14`。比对保持预算、最约束门框、全部门框预算/分离面、名义加速度、网格余量和速度，以及证据标志。

上述为**主机离线数值核对**，不是PS执行/飞行测试，也不证明假设的动力学误差界已认证。PS编译后的同一测试程序可按下面命令复跑。

## 父控制程序的稳定API

`features.hpp`中接口均位于`psfeatures`，使用C++14。`json.hpp`使用包内`third_party_reference/AirSim/MavLinkCom/common_utils/json.hpp`，不依赖安装系统JSON库。

```cpp
auto trajectory = psfeatures::Trajectory::from_layout(case_json["layouts"][vehicle]);
auto calibration = psfeatures::Calibration::from_json(calibration_json);
psfeatures::State state;       // local AirSim NED: p/v/a; q=(w,x,y,z)
// ... 用同一vehicle的实测状态填充state ...
auto refs = trajectory.window(reference_time_s, .1, false);
auto raw237 = psfeatures::encode(state, refs);    // float32[237], 未归一化
// ... 交给既有Pool::infer，仅在既有DPU入口归一化一次，不裁剪输入 ...
psfeatures::Control physical_dnn_output;          // [F,wx,wy,wz]
auto prepared = psfeatures::prepare_live_command(physical_dnn_output, calibration, 1.);
// ... 下发prepared.command中的AirSim参数 ...
// 下发未执行/被丢弃时不得更新保持控制；命令确认后才安装prepared.effective。
auto holding = analyzer.analyze(state, actual_effective_held_control);
auto deadline = psfeatures::safe_deadline(sample_s, now_s, last_actuation_ack_s,
                                         holding, maximum_update_s);
```

`Trajectory::window`固定21点，间隔默认0.1s，不重新拟合路径。`window`的时间是全局轨迹时间（参考匹配接入试验为`sample-origin+1.5s`）；不能每次采样重置轨迹。默认不计算教师才需要的body-rate feed-forward；这不影响RAW237。`encode`也提供`vector<Reference>`重载，复现Python不足21点重复最后一点、多于21点截断的行为；空序列拒绝。

`GateHoldAnalyzer(gates, case_width-.15, bounds, 4.-.15)`对应现有飞行链，而不是直接拿中心4m宽度充当净开口。有限门框四条杆及飞机半径/clearance按原函数构建。`bounds`默认原开发假设：加速度误差2m/s²、位置误差0.01m、速度误差0.05m/s、保持上限0.15s、barrier_rate=10。`plant_bounds_verified=false`，因此即使有正预算，`is_safety_certificate`也为false。0预算只是这些假设下无证书，不代表已碰撞。

`safe_deadline`返回`min(sample+budget, last_actuation+maximum_update-uncertainty)`，不重复扣响应时间。上层另外合并发布时冻结的周期期限、输入新鲜度期限和欠补更期限。

## 三种不同常数，不能统一替换

| 使用位置 | 原值/含义 |
|---|---|
| 参考姿态与237D参考推力特征 | `9.8066` m/s² |
| 保持分析名义重力 | `9.80665` m/s² |
| 油门标定悬停推力/工程斜率 | `9.81` / `19.62` |

`Calibration`结构默认值保持原`config.py`回退值（hover_throttle=0.5）；**生产必须读取包内标定文件**，本包实测hover_throttle为0.6，已在fixture中核对。不能把此标定文件省略后以默认值飞行。

## 坐标与特征布局

外部状态和参考都保持AirSim local NED，四元数wxyz。编码器内部执行原`VECTOR_FLIP=[1,-1,-1]`和`QUATERNION_FLIP=[1,1,-1,-1]`转换：当前姿态4＋当前速度3＋21×(相对位置3＋姿态4＋相对速度3)＋20个参考推力=237。姿态序列先按原w正分支及相邻点积保持符号连续，再使当前姿态与首参考同半球，最后转换坐标并转float32。当前测量加速度、角速度不进入特征；RAW237不进行范围修补、输入裁剪、缩放或归一化。

AirSim命令为`roll=wx, pitch=-wy, yaw=-wz`。`convert_command`只做原`control_adapter.py`的符号和油门转换，不额外限物理控制；`prepare_live_command`先复现`fly_ip37_joint.py`的物理限幅`[2,-3,-3,-2]..[20,3,3,2]`，然后油门限幅。其`effective`为**标定/油门限幅后**的物理控制；例如输入F=20时，hover=0.6会使throttle饱和1.0，实际用于保持分析的F应为17.658，而非20。

## 复跑命令

```bash
BASE=/home/cwc/dpu_work/windows_framework
SRC="$BASE/ps_migration_20261116_run001/features"
HANDOFF="$BASE/route_b_ps_handoff_20261116"
g++ -std=c++14 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  -I "$HANDOFF/third_party_reference/AirSim/MavLinkCom/common_utils" \
  "$SRC/features.cpp" "$SRC/test_features.cpp" -o "$SRC/test_features_host"
"$SRC/test_features_host" "$HANDOFF" "$SRC/feature_parity_host.json"
/home/cwc/dpu_work/.venv_falcon_fp32_cpu/bin/python -B \
  "$SRC/check_hold_reference.py" --handoff "$HANDOFF" \
  --cpp-report "$SRC/feature_parity_host.json" \
  --output "$SRC/hold_reference_parity_host.json"
```

ARM64交叉编译用同一`features.cpp`，替换g++为现场SDK编译器。无`-ffast-math`；使用`-ffp-contract=off`避免ARM融合运算改变边界舍入。JSON仅在初始化及测试读取，不进入参考/编码热路径。保持分析的返回结果含按固定门数有界的vector，不含Python/教师求解器。

Python参考检查仅在Ubuntu主机运行：由于宿主参考环境为Python3.6，`check_hold_reference.py`用AST读原有数值辅助函数、去除类型注解后在内存中执行，再加载原`held_control_safety.py`；不改原文件、不更改数值函数体。其报告保存全部四个原文件SHA256。最终PS控制链不需要该脚本或Python依赖。
