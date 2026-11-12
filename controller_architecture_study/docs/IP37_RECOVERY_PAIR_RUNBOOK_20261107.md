# 自然转弯穿门：首轮80/30ms配对操作

2026-11-07。板端沿用RC1模型11、5238端口，不更换量化包。代码与50项相关离线检查已完成，真实飞行由用户执行。

后续修复：`ip37_recovery_pair_20261107_133527`两组在起飞前被旧门框阻止（请求数0）。只读RPC确认一个残留部件 `FalconJointGate_Drone3_1_top`。入口现在自动清理精确匹配本实验命名的门框，并确认消失；外来对象不会删除，未确认清理则列名报错。记录在新目录的 `gate_preflight.json`。新增4项回归后27项运行检查通过；保持当前Play即可重跑同一命令，实际清理和飞行结果待用户回传。

## 本轮回答什么

固定同一个DNN和完整自然路线，是否存在慢周期难以安全通过、而实际可实现的快周期能通过的局部区间。先确认恢复机会，再测试动态能否及时识别和采用。此次两组都为固定周期，不把其差异直接写成动态收益。

使用已冻结六机场景中的Drone1路线：13个门，两个门位于实际转弯段，净宽2.1m；从起点静止出发，26秒任务。其余飞机不参与本轮控制，不增加虚构背景负载。两次均重新初始化，参考轨迹和门框一致，不注入外扰。

## 本轮完成的必要修改

- `fly_ip37_joint --case`读取已保存的完整轨迹、净宽和各机门数，保存原case的逐字节副本和哈希；不再假定每机5门。
- 实际下发前保留明确的2ms经验预留，与命令后的额外状态回读成本分开。后续回读长尾仍完整记录，不再自动让之后的30ms任务停止下发。
- 保留不可抢占的实际板端请求、周期期限和过期输出丢弃。命令已发出但返回晚仍记为超期，2ms经验预留不是硬WCET保证。
- 在窗口终点补处理尚应释放的尾部任务；不在终点额外生成下一周期。
- 增加真实线程CPU时间，与准备、RPC、板端等待的墙钟分开记录，为后续成本模型修正提供数据。本轮固定周期不使用尚未修正的ABC历史最大值预测。
- 逐门保留穿越点横/纵偏差，以及按0.5m机体半径+0.025m额外余量计算的过门截面余量；这不是完整连续扫掠认证，物理接触仍独立记录。

## 执行

保持板端5238服务运行；Blocks点击Play，窗口不要最小化。只需要Drone1存在，已有3机/12机工程均可。无需重启板端或改变设置。

```powershell
Set-Location -LiteralPath 'F:\FalconReproduction\airsim_control'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'F:\FalconReproduction\airsim_control\controller_architecture_study\run_ip37_recovery_pair.ps1'
if ($LASTEXITCODE -ne 0) { Write-Warning '结果已保留，请返回末尾输出与结果目录。' }
```

自动执行固定80ms、固定30ms，各约26秒，加初始化和黄金预热，通常约1分钟。输出新目录 `results/ip37_recovery_pair_<时间>/`，原始模型及已有结果均保留。慢周期碰撞或异常后仍会测快周期；技术失败保留并标为不可据此判断恢复能力，不裁剪输入或临时修改规则。

## 看什么

| 输出 | 含义 |
|---|---|
| `summary.json` | 两组完整结果；碰撞、穿门、时序与开销均在其中 |
| `classification.json` | 首轮筛查结论，不是重复验证结论 |
| `fixed80/case_frozen.json`及`fixed30/case_frozen.json` | 同一冻结场景；核对哈希一致 |
| `min_body_margin_m` | 各已穿越门的最小保守机体截面余量 |
| `timing_before_contact_passed` | 碰撞终止前是否已有丢弃/超期；接触终止本身不混成计算超期 |
| `calendar_coverage_passed` | 活跃回路直到任务终点、或碰撞退出前的日历是否有未处理释放 |

分类：

- `candidate_period_recovery_gap`：慢组失败/余量不足，快组完整通过且时序可比。仍需交错顺序重复，检查毫米级差异，随后再测动态。
- `both_pass_no_gap_at_this_width`：这个宽度尚未发现能力差异；按预定顺序试净宽1.8m，再视结果试1.5m，不无限收窄。
- `both_failed_no_recovery_demonstrated`：没有证明此控制器在这条路线下能救回，先分析原因。
- `faster_period_did_not_improve`：快周期并未改善，不能强行宣称短周期更安全。
- `timing_inconclusive`：周期未被实际执行好、或时钟/日历不合格，不能据此归因于控制能力。
- `runtime_or_input_inconclusive`：通信、输入范围或其他运行异常，需要看逐任务日志。

如需下一档，只新建场景并指定其文件，旧场景不能覆盖。例如：

```powershell
$py = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
$caseDir = '.\controller_architecture_study\results\ip37_safety_design_w18_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $py -m controller_architecture_study.ip37_safety_cases --seed 0 --net-width 1.8 --output $caseDir
if ($LASTEXITCODE -ne 0) { throw '几何检查未通过，保留报告，不启动飞行。' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\controller_architecture_study\run_ip37_recovery_pair.ps1' -CaseFile (Join-Path $caseDir 'case.json')
```

本轮没有接入六机在线动态修复，也没有完成正式同频观测J评价。完整研究步骤仍见[危险纠正实验设计](SAFETY_RECOVERY_EXPERIMENT_20261107.md)。
