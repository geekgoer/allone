# FALCON SVR：`J_paper_style` 复现定义

本文档固定当前复现实验中 SVR 的标签、输入、数据划分和训练协议，避免把工程代理指标误当成论文中的性能指标。

## 1. 论文式性能指标

FALCON 论文 Eq. (4) 给出的性能形式为：

\[
J = \frac{1}{\displaystyle\sum_{k=0}^{K}
\left\|\psi(k)-\psi_{\mathrm{ref}}(k)\right\|_{W_x}^{2}}
\]

本项目采用下面这个明确、可复现的实现：

- `K = 20`；
- 因为论文求和是 `k=0,...,K`，所以每个窗口使用 **21 个状态点**；
- `W_x = I_10`，即 10 个状态分量等权；
- 状态向量为：

  ```text
  psi = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
  ```

- 四元数比较前做符号对齐：如果 `dot(q_actual, q_ref) < 0`，将其中一个四元数乘以 `-1`。这是必要的，因为四元数 `q` 和 `-q` 表示同一个姿态。
- 实际状态与参考状态按同一个控制周期时间戳对齐。MPC 继续使用未来参考窗口，但计算论文指标时不使用向前偏移一个周期的 `window[0]`。

对第 `w` 个连续窗口，定义：

\[
P_w=\sum_{k=0}^{20}\left\|\psi_w(k)-\psi_{\mathrm{ref},w}(k)\right\|_2^2
\]

完整 route 使用连续、非重叠的完整窗口。最后不足 21 个点的窗口舍弃，并在数据生成日志中记录舍弃数量：

\[
P_{route}=\frac{1}{N_w}\sum_{w=1}^{N_w}P_w
\]

\[
\boxed{J_{paper\_style}=\frac{1}{P_{route}}}
\]

也就是说，当前 route-level 聚合顺序是：

```text
每个窗口先求代价
        -> 平均所有完整窗口代价
        -> 最后取倒数
```

这与“先平均窗口代价再取倒数”的约定一致。它不是声称恢复了论文作者未公开的全部实现细节，因此标签名称使用 `J_paper_style`，而不是 `J_exact`。

## 2. SVR 输入

每个完整闭环 episode 对应一行 SVR 数据：

\[
\boxed{[c,\ T,\ R,\ \epsilon]\longrightarrow J_{paper\_style}}
\]

其中：

- `beta_test_normalized_mse`：对应 DNN 的独立 test normalized MSE；
- `c = 1 / beta_test_normalized_mse`；
- `T = period_ms`，控制周期；
- `R = R_max_ms`，即一个 episode 内从状态采样到对应控制命令 dispatch 的实测总响应时间最大值；
- `actuation_delay_ms` 是人为注入的 delay，它叠加在原有响应路径上；
- `measured_R_max_ms/measured_R_p95_ms/measured_R_p99_ms` 是同一组实测总响应时间的显式诊断字段；
- `epsilon`：实际 gate layout 的环境难度参数；
- 动力学参数 `theta` 固定，因此当前版本不作为输入。

这里的 `R` 不是单独的 DPU 推理时间，也不是单独的 UDP RTT，而是完整的状态采样到命令 dispatch 延迟。它包含原有的 FPGA/主机通信、推理、调度和 AirSim dispatch 时间，也包含人为注入的 `actuation_delay_ms`。因此可以近似理解为：

```text
R ≈ R_native + actuation_delay
```

当前没有多任务资源竞争时，人工 delay 用来扩展可观测的 R 范围；将来做局部搜索时，再用目标平台的 WCET/资源占用模型估计候选配置的 R。

## 3. 输入归一化

只对 `c`、`T`、`R` 做 maximum normalization：

\[
c'=c/c_{max},\qquad T'=T/T_{max},\qquad R'=R/R_{max}
\]

`epsilon` 保持原始数值，不做归一化。

归一化上限必须只由训练数据计算。对交叉验证而言，每一个 fold 都只使用该 fold 的训练子集计算上限；对最终 holdout 模型而言，只使用 grouped train split 计算上限；最终使用全部有效数据训练的模型才使用全部有效数据计算上限。

SVR 的目标值直接使用原始 `J_paper_style`，不做 target standardization。

## 4. 数据有效性和防泄露

训练 CSV 必须至少包含：

```text
record_id
source_directory
beta_test_normalized_mse
period_ms
R_max_ms
epsilon
J_paper_style
valid_for_svr
```

只保留 `valid_for_svr=true` 的 episode，并排除 fallback、deadline miss、timeout、collision 或 error episode。每个 episode 是一行，而不是每个 control cycle 一行。

训练集和测试集按 `source_directory` 分组划分：同一个实验目录内的多个 repeat 只能整体进入 train 或整体进入 test，不能随机按行拆开。交叉验证同样使用 `GroupKFold`。

这一步是防止数据泄露的关键。否则同一组环境、周期和 delay 的重复飞行可能同时出现在训练和测试中，得到的指标会过于乐观。

## 5. `J_proxy` 与 `J_paper_style` 的区别

旧数据集中的：

```text
tracking_cost
J_proxy
```

来自工程闭环控制代价，主要使用位置误差、速度误差和控制周期加权。它没有保存计算 Eq. (4) 所需的完整 10D 实际状态与参考状态序列，因此不能通过改列名直接变成论文式 `J`。

新闭环采集会为每个周期保存 `paper_psi_actual`、`paper_psi_reference` 和逐维平方误差，并在 episode summary 中保存窗口代价及 `J_paper_style`。新训练脚本只接受 `J_paper_style`，这样可以避免误用旧 `J_proxy` 数据。旧结果必须重新采集，不能只重新运行数据集构建脚本。

## 6. 论文未公开、因此不能声称完全复现的部分

论文没有完整公开以下细节：

- 完整的 `W_x` 数值；
- route-level 窗口边界和聚合细节；
- 所有候选控制周期列表；
- SVR 的完整 kernel 超参数和训练划分；
- 目标硬件上所有延迟测量的具体时间戳定义。

因此本项目的正确表述是：

```text
按照论文 Eq. (4) 和公开信息实现的 J_paper_style 复现版本
```

而不是声称得到论文作者内部使用的 `J_exact`。

## 7. 重新采集论文式数据

必须使用新的输出目录；旧目录的 manifest 没有 `paper_metric` 合同，不能用 `--resume` 混入新数据。

先用 `pilot` 场景集做两个 episode（S01、S06）的 smoke test：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\run_fpga_closed_loop.py `
  --enable-fpga-actuation `
  --fpga-config .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\fpga_current5ms_config.json `
  --scenario-set pilot `
  --model-ids 1 `
  --repeats 1 `
  --period-ms 50 `
  --output-dir .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\results\paper_style_smoke_50ms_model1
```

如果中断，在相同命令末尾增加 `--resume`。已完成的 `trial_key` 会跳过，未完成的 partial trial 会从头重跑。不能对旧的 `closed_loop_full_*` 目录使用这个恢复方式，因为旧 trial 没有论文式状态记录。

所有 period/delay 实验完成后，使用 `build_falcon_svr_dataset.py` 汇总；输出的训练文件名是：

```text
paper_style_episode_dataset_train_ready.csv
```

### 7.1 完整、可恢复的四维 sweep

总控脚本：

```text
run_paper_style_svr_sweep.py
```

默认实验网格为：

- 6 个板端 DNN，即 6 个 `c`；
- `T = {30,40,50,60,70,80,90,100} ms`；
- 人工 delay 按控制周期比例取值，例如 `{0,0.2,0.4}`，实际 delay 等于 `T × 比例`；
- `full` 场景集 S01--S06，对应 `epsilon = {0.0,0.8,1.6}`，其中 0.8 和 1.6 各含多个独立 layout；
- 每个 `(c,T,delay,scenario)` 重复 5 次。

因此共有 24 个独立 run，每个 run 有 `6 × 6 × 5 = 180` 个 episode，总计 4320 个 episode。SVR 的 `R` 使用每个 episode 实测的 `R_max_ms`，其中包含原有响应时间和人为 delay。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
$env:PYTHONPATH = (Get-Location).Path

& $py .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\run_paper_style_svr_sweep.py `
  --output-root .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\paper_style_sweep\total_R_fraction_v1 `
  --periods-ms '30,40,50,60,70,80,90,100' `
  --delay-fractions '0,0.2,0.4'
```

可以随时按 `Ctrl+C`。恢复时原样重跑上面的同一条命令：完整 run 会跳过；已有 manifest 的未完成 run 会自动使用 `--resume`，已完成 episode 会跳过，中断时正在执行的 episode 会从头重跑。若改变 period、delay、模型、场景或 repeats，脚本会拒绝写入原目录，必须换一个新的 `--output-root`。

全部采集完成后，脚本会自动构建：

```text
paper_style_sweep/total_R_fraction_v1/svr_dataset/paper_style_episode_dataset_train_ready.csv
```

## 8. 可恢复训练

训练脚本：

```text
train_falcon_svr_paper_style.py
```

它会：

- 按 `source_directory` 固定 grouped train/test split；
- 手动遍历 RBF-SVR 超参数候选；
- 每完成一个候选就写入 `checkpoint.json` 和 `cv_scores.csv`；
- Ctrl+C 后保留已完成候选；
- 使用完全相同的命令重跑时校验数据集 SHA256、seed、test size 和 CV 配置，然后跳过已完成候选继续；
- 搜索完成后分别保存 grouped-holdout 模型和全数据模型。

示例命令：

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
$env:PYTHONPATH = (Get-Location).Path

& $py .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\train_falcon_svr_paper_style.py `
  --dataset .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\svr_dataset\paper_style_all_periods_v1\paper_style_episode_dataset_train_ready.csv `
  --output-dir .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\svr_runs\paper_style_seed42 `
  --seed 42 `
  --test-size 0.2 `
  --cv-splits 5
```

如果中断，直接再次执行同一条命令即可继续。旧的 `phase1_all_periods_v2` 数据只有 `J_proxy`，不能作为这个脚本的输入。
