# 独立运行指南：AXU3EGB + Windows AirSim 八环境 SVR 数据采集

> 连续全量采集的最新入口和数据口径见`RUN_ALL.md`：一条命令处理全部计划、支持中断续跑，主性能SVR使用完整成功路线J。无需逐次执行limit=1。以下保留安装/历史排障说明。

> 2026-10-31：新ZIP已采用温和八环境v2和输入保护，采集命令及目录以`MODERATE_V2.md`为准。下文SD/网络/安装步骤仍适用；旧20261015 ZIP没有输入保护。环境必须用原UE4.27.2工程的FalconEmpty，官方预编译Blocks不能默认等价。

版本：2026-10-15。本文可脱离原聊天记录使用。只做仿真采集，不控制实体无人机。

## 0. 任务与交付物

目标：复用已经训练、量化、编译的六个12维DNN，在同事自己的Windows + AXU3EGB上采集八档环境的闭环性能数据。不是训练DNN，不需要在同事电脑安装PyTorch、CUDA、Vitis AI或交叉编译器。

控制链路：Windows读取AirSim状态 → UDP发送归一化12维输入 → 板端DPU运行指定模型 → 回复四维物理控制量 → Windows等待到目标R再发送AirSim命令。

必须拿到下面三类材料，不能只拿采集脚本：

| 材料 | 提供方 | 用途 |
|---|---|---|
| `falcon_axu3egb_working_sd.img`，可另用7-Zip压缩 | 原设备持有人从已工作SD卡读取 | 启动分区、Linux、设备树、DPU硬件/驱动等；前提是它们确实在SD镜像中 |
| `falcon_current5ms_board_*.tar.gz`及SHA-256、原板启动/存储检查输出 | 原设备持有人从运行中的板端备份 | 当前六模型服务、ELF/内核、归一化/黄金向量等完整板端包 |
| `falcon_svr_colleague_时间.zip` | 原设备持有人运行本文第1节打包 | Windows采集代码、配置、冻结模型元数据/黄金输入、AirSim Python客户端源码、本文 |

另外需要一套可实际运行的**同版本Blocks环境**。只有AirSim插件源码、`Blocks.uproject`单文件或下载了一个不同环境，不等于环境已就绪。如果已有下载环境与原环境不同，优先让提供方补发同一套可运行Blocks发布包；整个包解压运行，不是把exe单独拎出来。不需要复制整个UE安装目录。

Windows ZIP不包含SD镜像、板端运行库/可执行程序、仿真器、训练权重、训练集、旧实验结果、密码或原Python虚拟环境。DNN实际由板端已编译程序执行，因此Windows采集不需要`.pt`权重。

### 固定身份，切勿混包

```text
模型包：falcon_current_5ms_12d_six_20260920_202227
板端目录：/home/root/board_package_current5ms_12d_six_v1
板端服务：falcon12_current5ms_dnndk_udp_server
板端地址：192.168.77.2
Windows专用有线网卡：192.168.77.1 / 255.255.255.0
协议：FLCN/v2，UDP 5008
仿真API：Windows本机 TCP 41451
网口：板子 PS_ETH，不是 PL_ETH
硬件：与原板相同型号/版本的AXU3EGB及兼容DDR配置，现有DPU为B1152
```

旧5005/5006/5007服务、旧25维模型及其他12维模型不是本实验。不能因为端口能回包就认定权重正确。

## 1. 原设备持有人：先准备交接材料

### 1.1 板子仍开机时，确认“SD里到底有什么”

在**原板Linux终端**执行并保存输出给同事/负责板端的Ubuntu agent。这些是检查命令，不会刷写介质：

```sh
uname -a
cat /etc/os-release
cat /proc/cmdline
mount
df -h
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT
ip addr
ls -l /dev/dpu*
lsmod
printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
ls -l /home/root/board_package_current5ms_12d_six_v1
ldd /home/root/board_package_current5ms_12d_six_v1/falcon12_current5ms_dnndk_udp_server
```

BusyBox系统如果没有`lsblk`或`ldd`，保留其他输出，不要为此升级整个系统。还应提供原板SD启动拨码照片及正常冷启动串口日志。

解释：

- 根文件系统、引导文件均在SD，DPU随冷启动加载：整卡镜像通常可以迁移，再核对板端包。
- 根在eMMC：**SD克隆不包含eMMC根文件系统**，需让Ubuntu agent提供匹配的可启动系统镜像/正确的根文件系统迁移方案。
- 根是ramfs/tmpfs/initramfs/overlay：克隆会带走启动镜像中原有内容，但开机后临时装进去的库、程序、驱动、手工挂载内容可能不在镜像里。板端包要另备份；库/驱动仍需与原内核匹配，不能随意混装新版DNNDK。
- 使用TFTP/NFS网络启动或JTAG手动下载DPU：要把对应启动文件、根文件系统、bitstream和加载步骤一并移交；不能承诺只靠SD克隆就独立启动。

不要根据`mmcblk0/mmcblk1`的数字猜SD和eMMC。检查结果没有明确前，不刷eMMC/QSPI，不改引导环境。

### 1.2 单独备份当前板端运行包

先停止旧采集，服务在前台时按Ctrl+C；不必卸载DPU。原板上：

```sh
cd /home/root/board_package_current5ms_12d_six_v1
sha256sum -c SHA256SUMS.txt
falcon_backup="/home/root/falcon_current5ms_board_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$falcon_backup" -C /home/root board_package_current5ms_12d_six_v1
sha256sum "$falcon_backup"
printf 'Archive to copy: %s\n' "$falcon_backup"
```

若目录/校验清单不符，先找对current5ms包，不要改成旧historical包。上面备份只归档指定运行包，**不是整个Linux备份，不自动包含外部共享库和驱动**。

原Windows PowerShell把最后打印的实际文件名替换到下面的`远端实际文件名`位置：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
New-Item -ItemType Directory -Force -Path "$proj\colleague_transfer" | Out-Null
scp root@192.168.77.2:/home/root/远端实际文件名.tar.gz "$proj\colleague_transfer\"
```

密码通过可信私下渠道传递，不写进文档。复制后核对Windows `Get-FileHash -Algorithm SHA256 -LiteralPath '实际归档路径'`与板端输出一致。**先拷出归档，再关机**，因为原根目录可能是易失性内存。

### 1.3 制作整张SD镜像，而不是复制Windows看得到的文件

推荐从[Win32 Disk Imager项目页](https://sourceforge.net/projects/win32diskimager/)取得工具。它支持可移动介质的原始镜像读写；不要从仿冒下载站拿安装器。

1. 原板执行`sync`、`poweroff`，等待关机后断电，再拔SD卡。不要热拔运行中的根文件系统。
2. SD卡接读卡器插原Windows。如果弹“需要格式化才能使用”，**取消**。
3. 在“磁盘管理”核对这张可移动卡的容量/分区，记住盘符；其他可移动存储尽量拔掉。
4. 管理员打开Win32 Disk Imager。Image File选择电脑上的新文件`falcon_axu3egb_working_sd.img`，不能存回源SD卡；Device选择已确认的**源SD卡**。
5. 点击 **Read**：方向是“SD卡 → 电脑镜像”。读取整张介质，不只备份FAT启动分区。
6. 结束后计算`Get-FileHash -Algorithm SHA256 -LiteralPath '镜像实际路径'`，记录镜像字节数和SHA-256。镜像可能与SD整卡容量相当，电脑要预留空间。

镜像包含整卡数据，可能连带系统密码散列、SSH密钥、Wi-Fi/网络凭据和已删除数据残留。只交给被授权的同事，不公开上传。它也不是“只有实验文件”的精简包。

### 1.4 打包Windows采集材料

原Windows PowerShell：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
& $py .\experiments\falcon_svr_8env_v1\package_for_colleague.py
```

终端会打印`colleague_transfer`下新ZIP的完整路径、大小和SHA-256。将该ZIP、SD镜像、板端tar.gz、各自校验值、原板检查/启动日志、本文及相同Blocks环境交给同事。

## 2. 同事：把空SD卡恢复成可启动板卡系统

先确认板子与原板硬件兼容。目标SD的**实际字节容量不得小于源镜像**；“同为32GB”也可能略小，优先准备更大的卡。

1. 核对收到镜像的SHA-256。解压压缩文件后，对真正的`.img`核对提供方的镜像哈希。
2. 接入**同事的空SD卡**，用磁盘管理按容量、读卡器和盘符再次确认。不要插着源卡操作。
3. 管理员Win32 Disk Imager选择收到的`.img`，Device选这张**目标SD卡**。
4. 点击 **Write**。这一步会覆盖目标卡原分区和全部数据；选错系统盘/其他U盘会造成数据丢失。确认目标后才执行，不使用本指南猜测磁盘号。
5. 写入后做工具提供的读取校验；不要把“文件复制成功”当作镜像写入成功。取消Windows后续格式化提示并安全弹出。
6. 板子断电时插卡，设置与原板/该硬件版本手册一致的**SD启动拨码**，再上电。不要猜拨码数字，也不要刷写eMMC/QSPI。

连接USB-UART观察启动并登录。串口设置优先照原板记录；通常115200、8N1、无流控，若原板不同以原板为准。用户名/密码由提供方给出，本文不假定厂商默认密码。

本指南没有替你实际读取/刷写SD，也未确认原板根文件系统位置；第1.1节的条件检查不可省略。若新板不能独立冷启动Linux或没有DPU，不要继续跑Windows实验。

## 3. 同事：接线和IP配置

接线：`Windows专用有线网口（或USB网卡） ↔ 网线 ↔ AXU3EGB PS_ETH`。UART另接USB。不要把网线接PL_ETH；本服务运行在PS Linux上，PL_ETH不等于自动可用的Linux网卡。板卡手册也区分PS/PL两路Ethernet。

### 3.1 Windows

保持Wi-Fi正常联网。给**连接板子的那块网卡**设置静态IPv4：

```text
IP：192.168.77.1
掩码：255.255.255.0
默认网关：留空
DNS：留空
```

不要改Wi-Fi的IP，不要设置网桥/共享上网。若VPN、Wi-Fi或其他设备已占用192.168.77.0/24，先解决网段冲突；不要只改一端IP。

PowerShell只读检查：

```powershell
Get-NetAdapter
Get-NetIPConfiguration
```

### 3.2 板端

UART登录后先看`ip link`、`ip addr`。通过原板记录/插拔网线后的link状态确认PS网卡名，不能假设一定叫eth0。下面仅在确认名称后运行：

```sh
falcon_if='替换成已确认的PS网卡名'
ip link set "$falcon_if" up
ip addr replace 192.168.77.2/24 dev "$falcon_if"
ip addr show dev "$falcon_if"
ping -c 4 192.168.77.1
```

以上IP设置可能重启后失效；先不要盲改不熟悉的网络配置文件，重启后重新检查即可。如果镜像已经自动配置192.168.77.2，不必重复设置。

Windows运行`ping 192.168.77.2`。Ping不通时先检查接线/IP/ARP/防火墙；ICMP可能被防火墙拦截，不能单凭Ping判UDP服务死活。**`Test-NetConnection -Port 5008`测的是TCP，不能验证UDP5008。**

克隆卡可能继承静态MAC/主机名/SSH密钥/IP。两套设备不要放在同一共享网段用相同IP；本方案每套PC和板子各自直连隔离。新板SSH指纹告警时先通过UART核对身份，不盲目忽略或删除known_hosts。

## 4. 同事：恢复并启动正确板端服务

串口/SSH检查：

```sh
uname -a
ls -l /dev/dpu*
ls -ld /home/root/board_package_current5ms_12d_six_v1
```

如果整卡恢复后该目录已在，先核对，不必重复覆盖。目录缺失时，把收到的板端归档传入新板：

```powershell
scp '收到的板端归档实际路径.tar.gz' root@192.168.77.2:/home/root/
```

板端先用`sha256sum`核对归档，再`tar -tzf 实际归档文件名.tar.gz`查看内容确实为指定运行包。如果目标目录已存在且不一致，先保留旧目录并让提供方确认，**不要直接覆盖**。确认目标目录不存在后：

```sh
tar -xzf /home/root/实际归档文件名.tar.gz -C /home/root
cd /home/root/board_package_current5ms_12d_six_v1
sha256sum -c SHA256SUMS.txt
ldd ./falcon12_current5ms_dnndk_udp_server
```

有库`not found`、`/dev/dpu`缺失、架构/驱动不匹配时停止：向提供方要与原Linux内核、DPU bitstream、DNNDK一致的系统/库/驱动。不要通过随意升级Vitis、复制别处libc或换其他DPU架构解决。

首次移植建议做一次板端六模型黄金检查，成本很小，不重做旧AirSim测试：

```sh
cd /home/root/board_package_current5ms_12d_six_v1
for model in depth10_width512 depth10_width1024 depth22_width512 depth22_width1024 depth34_width512 depth34_width1024
do
  ./falcon12_current5ms_dnndk_golden "$model" "golden/${model}_golden_vectors.txt" 1000 || break
done
```

六个都应出现`board_int8_consistency=PASS`。若发行包不含该验证程序/黄金文件，请提供方补齐同版包，不用其他模型的黄金输出替代。

先确保没有其他5008服务实例，再启动：

```sh
cd /home/root/board_package_current5ms_12d_six_v1
./falcon12_current5ms_dnndk_udp_server --port 5008
```

成功标志：MODEL id=1..6，输入维度12、输出4，并出现`READY protocol=FLCN/v2 ... 5008 ... models=6`。映射：1=10×512，2=10×1024，3=22×512，4=22×1024，5=34×512，6=34×1024。

长时间运行如需关闭SSH窗口，先Ctrl+C停止前台实例，再用下面方式替代启动，**不要同时开两个**：

```sh
nohup ./falcon12_current5ms_dnndk_udp_server --port 5008 > falcon5008.log 2>&1 < /dev/null &
head -n 12 falcon5008.log
```

`nohup`不等于开机自启动；板子掉电重启后必须确认网络和服务重新启动。

## 5. 同事：解压Windows包并建环境

把ZIP解压到本地短路径，例如`D:\FalconSVR\falcon_svr_colleague`。这一级目录应直接看见`config.py`、`experiments`、`vendor`、`transfer_manifest.json`。不要把项目放云盘同步目录，后续运行不要移动它。

安装Windows x64 CPython **3.11**，推荐3.11.9。不要复制提供方`.venv`，不要用Python3.13/3.14直接尝试旧RPC依赖。提供方原AirSim解释器是3.11.9/NumPy2.4.6；此交接包选用已通过离线回归的NumPy1.26.4，便于和Matplotlib3.7.2兼容。它作为新主机的新批次单独记录，不与原主机结果无标识混接。

PowerShell：

```powershell
$proj = 'D:\FalconSVR\falcon_svr_colleague'
Set-Location -LiteralPath $proj
py -3.11 -m venv .venv
$py = Join-Path $proj '.venv\Scripts\python.exe'
& $py -m pip install -r .\experiments\falcon_svr_8env_v1\requirements-colleague.txt
if ($LASTEXITCODE -ne 0) { throw 'Python依赖安装失败，先解决安装错误，不要开始采集' }
$env:PYTHONPATH = Join-Path $proj 'vendor'
& $py .\experiments\falcon_svr_8env_v1\verify_transfer.py --runtime
& $py .\experiments\falcon_svr_8env_v1\collect.py --dry-run
```

如果系统没有`py`启动器，使用已安装的3.11解释器完整路径创建venv。必须先把pip安装问题解决；不要升级tornado到6或同时安装新版`msgpack`来覆盖旧`msgpack-python`。AirSim客户端源码随包位于`vendor/airsim`，不用再`pip install airsim`。

检查应显示哈希PASS、AirSim1.8.1、4608次计划，且不会访问硬件。此处不需要GPU训练环境。安装依赖需要联网；ZIP不是离线wheel仓库。

## 6. 同事：准备AirSim

先退出AirSim/停止Play，备份自己现有的`Documents\AirSim\settings.json`。使用实际“文档”文件夹，不假定它未被OneDrive重定向。将项目内`configs/settings_three_drones.json`复制为该目录的`settings.json`，不要直接覆盖未备份的配置。

它创建Drone1/2/3；本采集只控制Drone1。保持SimpleFlight、碰撞检测有效、ClockSpeed=1（该配置默认值为1），不设置`IgnoreCollision`、不启用不同风场，不修改车辆物理参数。`hover_throttle=0.6`等执行器标定随项目保留，不能针对某模型临时调参。

手动启动相同Blocks并点击Play；发布包则进入可运行仿真状态。只开一个实例，不使用`--auto-launch`。UE4编辑器里不要开启“失去焦点时大幅降低CPU使用”等会使后台仿真明显降速的选项。不要提高ClockSpeed来省时间。

每个新PowerShell窗口先恢复变量：

```powershell
$proj = 'D:\FalconSVR\falcon_svr_colleague'
Set-Location -LiteralPath $proj
$py = Join-Path $proj '.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $proj 'vendor'
Get-NetTCPConnection -State Listen -LocalPort 41451
& $py -c "import airsim; c=airsim.MultirotorClient(timeout_value=8); c.confirmConnection(); print('vehicles=',c.listVehicles()); print('settings=',c.getSettingsString()); print('EditorCube available=', 'EditorCube' in c.simListAssets())"
```

预期：41451监听、Drone1存在、设置正确、EditorCube可用。API已连接但没有可用立方体资产时，先换回同版本Blocks；不要仅更改资产名而不校准尺寸，否则门框大小会变化。

## 7. 同事：Windows→FPGA profile，然后开始采集

先只发UDP测六模型原生时延，不飞行：

```powershell
& $py .\experiments\falcon_svr_8env_v1\collect.py --profile-only
```

该步骤会冻结本机实验并保存`latency_profile.json`。参考板端DPU+I/O约1.0～11.8 ms，Windows RTT会稍大；不是硬性时延承诺。最大实测RTT加3 ms余量必须小于最低目标R=25 ms，否则程序停止。

注意：根目录旧`falcon12_fpga_smoke_test.py`内置的是旧5006模型黄金向量，本ZIP刻意不带它。不要改成5008就拿来验证本批权重。正确身份由当前板端包SHA/黄金验证配合这里的六模型UDP检查确认。

UDP失败先看板端服务、网线、IP、网络路由。Windows若提示网络访问权限，允许**本项目的`.venv\Scripts\python.exe`**。防火墙按需要仅允许该程序与192.168.77.2的UDP5008通信；不要关闭整个防火墙。AirSim本地41451错误与板端UDP5008错误是两个问题。

开始前确认接电、系统不自动睡眠、磁盘至少预留10GB、板子供电散热稳定、无其他FPGA压测/无人机控制程序。锁屏不必然停止程序，但睡眠/休眠会中断；AirSim在后台也必须正常推进。

建议先运行计划中的前96个新试验，**这96个本来就属于4608总数，不额外浪费**：

```powershell
& $py .\experiments\falcon_svr_8env_v1\collect.py --enable-fpga-actuation --limit 96
```

确认连接/运行正常后，同一窗口继续全部剩余数据：

```powershell
& $py .\experiments\falcon_svr_8env_v1\collect.py --enable-fpga-actuation
```

也可首次直接运行无`--limit`命令。粗估全量23～41小时，以`progress.json`和终端实测ETA为准。不会训练DNN或自动拟合SVR。

### 中断与继续

- Ctrl+C中断，重新执行上面同一命令继续；**不用`--resume`**。
- 关机后先恢复板端服务、AirSim和PowerShell四行变量，再执行原命令。
- 已完成的成功/碰撞/未正确过门/任务超时跳过，不会重飞到成功。
- 中途未完整完成的attempt保留，再跑使用新文件名，不覆盖旧轨迹。
- UDP/RPC故障会停止批处理；排除故障后续跑。不要把网络失败手工改成碰撞。
- 配置、初始状态、标定、关键代码及AirSim设置会被冻结；采集开始后不要升级依赖、修改代码/场景/配置或移动项目。确需更换实验条件，另开输出目录，不把旧结果复制过去冒充同批。
- 同事主机是**新批次**，不复制提供方`results/full_v1`来接着跑；换电脑会改变时延/仿真条件，而且绝对路径哈希不同。

## 8. 数据含义和八个立方体

固定网格：6模型 × 8T × 6目标R × 8环境 × 2冻结变体 = 4608次；每环境288个不同坐标、576次试验。T=60/70/80/90/100/120/140/160 ms，目标R=25/30/35/40/45/50 ms；所有R<T。两个变体是固定镜像布局/不同初始状态，不是大量随机独立重复。

环境(dx,dy,dz)/m：E01(10,0,0)、E02(10,0.8,0)、E03(10,1.2,0.4)、E04(8,2.4,0.4)、E05(6,3.2,0.6)、E06(4.5,4,0.6)、E07(3.5,5,0.8)、E08(2.5,6,0.8)。门洞4×4 m，五门，名义速度5 m/s。c=冻结test normalized MSE的倒数，不是手动按参数量制造单调。

先测原生时延，再在每帧推理后等待到“状态获取完成+目标R”发命令；回归记录的是实际测量R而非目标值。保留全时延序列。采集为整网络不可抢占，**没有启用FALCON分段调度**。

高难环境允许失败，不保证特定方法能通过。不得放大门洞、忽略碰撞、改判定、删掉失败或反复重飞某一配置直到成功。保持提供方基线协议。

默认结果目录：`项目\experiments\falcon_svr_8env_v1\results\full_v1\`。

```text
experiment_manifest.json / frozen_files_sha256.json
runtime_identity.json / latency_profile.json
benchmark.json / initial_states.json / model_manifest.json / plan.json
progress.json
trials/<试验ID>/attempt_*.jsonl / attempt_*.summary.json / result.json
datasets/all_recorded_trials.csv / all_recorded_trials.jsonl
datasets/censored_infrastructure_attempts.csv / dataset_manifest.json
datasets/E01..E08/all_trials.csv / svr_success_only.csv / svr_failure_zero.csv
```

`svr_success_only.csv`仅成功完整路线原J；`svr_failure_zero.csv`把物理碰撞、门洞外通过、任务超时显式记0，是声明过的工程扩展。部分失败轨迹J只诊断，不当奖励。混合MPC后备、截止期失效和网络故障不当纯FPGA标签；它们原始记录仍保留。不能拿只成功的数据声称已经建模碰撞风险。

采集暂停/结束后，再安装画图依赖并导出，别与采集同时改派生表：

```powershell
& $py -m pip install -r .\experiments\falcon_svr_8env_v1\requirements-colleague-plot.txt
& $py .\experiments\falcon_svr_8env_v1\export_and_plot.py --plot --policy failure_zero
```

八图在`results/full_v1/figures/failure_zero/cube_E01.png`～`cube_E08.png`；未测点灰色，真实失败X，成功圆点，共用色标，不插值伪装实测。原成功J图可换`--policy success_only`。

## 9. 完成后返还什么

停止采集后，返还**完整`results/full_v1`目录**，不要只发CSV或截图。附本机CPU/GPU/Windows版本、Blocks来源/版本、板卡型号/版本、SD镜像和板端包SHA、服务启动日志、依赖版本（`& $py -m pip freeze`输出），注明有无掉电/中断和手工改动。

PowerShell压缩示例，输出新文件，若已经存在请改名：

```powershell
$result = Join-Path $proj 'experiments\falcon_svr_8env_v1\results\full_v1'
$resultZip = Join-Path $proj ('falcon_svr_results_colleague_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.zip')
Compress-Archive -LiteralPath $result -DestinationPath $resultZip
Get-FileHash -Algorithm SHA256 -LiteralPath $resultZip
```

若归档工具遇到大文件限制，改用7-Zip归档整个结果目录，别拆掉原始逐帧日志。然后由实验负责人审核覆盖率、排除项与失败分布，再训练八环境SVR。本任务的完成不等于已经证明论文全部数值、单调趋势、安全性或多机分段调度。

## 10. 最短排障表

| 现象 | 优先检查 |
|---|---|
| 新板不启动 | 镜像SHA、写入目标、SD启动拨码、硬件兼容；原板是否实际用eMMC/TFTP/JTAG |
| 有Linux但无DPU/缺库 | 启动bitstream、设备树、同内核驱动、DNNDK共享库，不是Windows代码问题 |
| UDP超时 | PS_ETH、192.168.77.1/2同网段、5008服务READY、Windows程序防火墙 |
| TCP41451拒绝连接 | AirSim未运行/未Play、不是带AirSim插件的环境、多实例或端口不符 |
| Drone1或EditorCube缺失 | settings.json位置/加载顺序、错误Blocks环境；不要偷改门尺寸 |
| profile原生时延过大 | 其他FPGA请求、网络/CPU负载、电源模式；不能把目标25ms硬说成已达到 |
| E07/E08碰撞/未过门 | 可能是预期压力测试结果，保存记录；先别改控制器 |
| 哈希/续跑条件不匹配 | 文件、配置、设置或路径改变；保留原批次，确认原因后另开新目录 |

## 11. 参考与验证边界

- [Win32 Disk Imager项目说明](https://sourceforge.net/projects/win32diskimager/)：可移动介质原始读写/备份。
- [ALINX AXU3EG手册](https://www.alinx.com/public/upload/file/AXU3EG_User_Manual.pdf)：PS/PL网口为不同接口；具体硬件版本以实物配套手册为准。
- [Microsoft AirSim settings](https://github.com/microsoft/AirSim/blob/main/docs/settings.md)：settings加载顺序、ClockSpeed、API端口。
- 精简包已进行本地离线打包/解包、哈希/导入和计划检查；不是在同事空板上实测成功的声明。没有访问或刷写同事SD卡，没有替同事执行网络配置或启动真实采集。
