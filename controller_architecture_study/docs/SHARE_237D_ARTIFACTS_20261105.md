# 同事协作的237D模型与数据交接

维护日期为2026-11-05。

## 为什么克隆Git仓库后缺文件

本工程的 `.gitignore` 排除了 `results/`、权重 `.pt`、数据 `.npz` 和压缩包。Git仓库提供源码，运行时产物需要单独传输。同事列出的11个文件在原机器均存在，合计约251 MiB。旧12D包和当前237D契约不同。

本次交付一个独立残差控制器 `residual_mlp_h21_w1024_seed0`、9600行237D验证数据、来源清单、归一化和默认五门路线。该网络训练600轮，最佳权重来自第598轮。包中保留原始 `best_model.pt`，包含原训练检查点状态。

这个文件组支持单模型离线推理检查、单模型验证分析及指定路线的单机试飞。当前六机实时试验还依赖其他独立模型、评分器、历史配置及原生IO库，须按其入口另行补齐产物。超网和DPU实验也有独立交接包。

## 本次路径修复

原来的恢复清单把 `base` 绑定到 `F:\FalconReproduction\airsim_control\...`，训练配置中的 `data_dir` 也是原机器绝对路径。现在推理入口按当前工程位置解析记录路径，并用基础采集清单哈希、增广清单哈希、环境划分、237D特征索引及归一化哈希核对数据身份。

原始清单、训练配置和权重的字节保持原样。新增推理校验位于 `convex_artifacts.py`。被冻结清单记录自身哈希的 `convex_recovery.py` 保持原样，原训练、生成入口的检查继续保留。

`.gitattributes` 为Python源文件规定LF换行，避免Windows克隆自动转换CRLF后破坏源码哈希。出现 `source code changed` 时应先核对Git版本及换行，不能改写清单中的期望哈希来通过检查。

## 包内布局

解压到Git仓库根目录后，应出现下列文件。相对路径以该根目录为起点。

```text
237d_handoff_manifest.json
READ_237D_HANDOFF.md
controller_architecture_study/results/
  decision_convex_96k_v1/
    manifest.json
    dnn_dataset/split_manifest.json
    dnn_dataset/normalization.json
    routes/env48_route00.json
  decision_convex_recovery_v1/
    manifest.json
    data_audit.json
    datasets/features_recovery/
      split_manifest.json
      normalization.json
      validation.npz
  decision_convex_residual4_v1/runs/residual_mlp_h21_w1024_seed0/
    config.json
    completion.json
    best_model.pt
```

`237d_handoff_manifest.json` 记录每个产物的大小、SHA-256和打包前检查。压缩包旁的 `.zip.sha256` 用于核对传输完整性。这里额外加入路线文件，因为单机试飞会读取它。

## 原机器生成交接包

Agent已生成的zip可直接发送。需要再次打包时使用新文件名，在工程根目录执行下列命令。

```powershell
$py = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'F:\FalconReproduction\airsim_control'
& $py -m controller_architecture_study.share_237d_artifacts package --output .\colleague_transfer\residual237d_collaboration_v1_20261105.zip
if ($LASTEXITCODE -ne 0) { throw '打包失败，请查看输出' }
```

打包入口先核对来源和验证数据，加载指定权重执行32条验证输入，再检查压缩后的每个产物哈希。它读取原始产物并创建新zip。

## 同事更新代码和解压

同事先保存自己的代码修改，再更新Git仓库。以下以 `D:\allone` 为同事本机仓库示例，按实际位置修改两处路径。`$py` 指向同事已经配置的Python环境。离线检查所需导入包括NumPy、PyTorch、SciPy。

```powershell
$proj = 'D:\allone'
$zip = 'D:\Downloads\residual237d_collaboration_v1_20261105.zip'
$py = 'python'
Set-Location -LiteralPath $proj

git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw '代码更新失败，请先处理本机修改或分支差异' }

$expected = ((Get-Content -LiteralPath "$zip.sha256" -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw '压缩包哈希不一致，请重新传输' }

Expand-Archive -LiteralPath $zip -DestinationPath $proj
& $py -m controller_architecture_study.share_237d_artifacts check
if ($LASTEXITCODE -ne 0) { throw '237D产物检查失败，请保留完整输出' }
```

解压时若提示已有文件，先核对本机是否已有该版本，保留自己的实验产物。检查输出中的 `status=PASS` 表示来源、产物哈希及32条CPU推理通过，报告还会给出237D输入形状和权重哈希。包内记录的权重哈希与接收文件进行比较。

Linux同样将zip解压到仓库根目录，再执行 `python -m controller_architecture_study.share_237d_artifacts check`。记录中的Windows分隔符由新路径解析器处理。

## 单模型离线验证

在检查通过后执行下列已有分析入口。这里显式选择包中的一个模型。

```powershell
& $py -m controller_architecture_study.probe_state_value --preflight --models residual21x1024
if ($LASTEXITCODE -ne 0) { throw '模型来源检查失败' }
& $py -m controller_architecture_study.probe_state_value --device cpu --models residual21x1024
if ($LASTEXITCODE -ne 0) { throw '离线验证失败' }
```

验证集用于复核已有模型输出。包中没有Train或Test文件，后续训练、泛化评估应另行安排数据交接。

## AirSim单机试飞

只有离线检查通过后再连接同事本机AirSim。先配置AirSim Python客户端，使 `python -c "import airsim"` 能运行。自动搜索原机器 `F:/Airsim/AirSim/PythonClient` 的历史回退路径不能替代同事本机的客户端安装。实际试飞还会用到OSQP，需在该Python环境中安装。Unreal中加载Blocks、进入Play，并确认存在 `Drone1`。

```powershell
& $py -m controller_architecture_study.fly_convex_policy --checkpoint .\controller_architecture_study\results\decision_convex_residual4_v1\runs\residual_mlp_h21_w1024_seed0\best_model.pt --output .\controller_architecture_study\results\decision_convex_96k_v1 --environment-id 48 --route-index 0 --device cpu
if ($LASTEXITCODE -ne 0) { throw '试飞未通过，请返回完整终端输出和报告路径' }
```

本次Windows Agent完成离线迁移检查，真实飞行由使用者运行。离线推理通过与实际控制时序、飞行成功需要分别记录。

## Git与产物分工

源码修复和本文通过Git推送。同事需要先取得本次代码更新，再解压zip。zip和同名 `.zip.sha256` 可以通过网盘、文件传输或手工GitHub Release附件提供，仓库的结果目录忽略规则继续保留。
