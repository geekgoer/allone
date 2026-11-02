# FALCON reproduction and dynamic control research

## 协作入口（更新：2026-11-02）

本工程已从AirSim/MPC与FALCON模型复现，扩展到**状态相关控制期限、低开销配置决策与多回路共享加速器调度**。当前主要研究源码在`controller_architecture_study/`；下面的早期MPC命令继续保留，但不能据此判断整个工程的最新状态。

- [整体Framework](docs/FRAMEWORK.md)：研究目标、时间语义、超网/FPGA定位及当前实现差距。
- [协作进度与任务认领](docs/COLLABORATION_STATUS.md)：已经做了什么、当前问题、后续工作、产物和协作更新方式。
- [Agent入口](AGENTS.md)：阅读顺序与协作约定。
- [研究分支文档索引](controller_architecture_study/docs/README.md)：实现、实验、量化与Ubuntu交接的导航。

最新阶段已完成120次有效恢复采集与12次十机对照。规则式动态有零碰撞样例；新增评分器的额外收益、可靠的最低成本合格DNN选择、全程硬期限及新超网上板效果尚未成立。具体结论看[阶段结果](controller_architecture_study/docs/BOUNDARY_TAKEOVER_RESULTS_20261102.md)。

25D、12D、237D独立模型和237D超网是不同轨道；不要混用数据、归一化或部署包。大文件不随文档自动提供，协作时按看板中的产物位置核对。

## Historical AirSim MPC teacher pipeline

This project implements the AirSim + UE4 part of the FALCON reproduction
scope: state acquisition, a 12-D controller state, an MPC-compatible teacher,
the `[F, wx, wy, wz]` control interface, visible gates, and teacher-data CSV
logging. It also prepares leakage-safe dataset splits and trains the paper's
fully-connected DNN controller family. FPGA/DPU deployment, quantization, SVM
performance modelling, and FALCON scheduling are outside this historical
teacher pipeline; later research and deployment work are indexed above.

## Run the single-drone checks

Start Blocks in UE4.27, click **Play**, then use the AirSim Python virtual
environment:

```powershell
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location 'F:\FalconReproduction\airsim_control'
& $py -m unittest discover -s tests -v
& $py run_single_drone_mpc.py --duration 15 --dx 3
& $py run_single_drone_gates.py --epsilon 0.8 --seed 0 --per-gate-timeout 30
```

`run_single_drone_gates.py` creates four static cube bars for each gate. The
bars use the Blocks asset `1M_Cube`, with width 4 m, height 4 m, and thickness
0.15 m. Use `--keep-gates` when the objects should remain after landing;
otherwise the script removes only its own `FalconGate_` objects.
`--gate-width`, `--gate-height`, `--gate-thickness`, `--gate-asset`,
`--gate-count`, `--x-start`, and `--x-spacing` are available for adjustments.

The gate generator guarantees x coordinates 10, 20, 30, 40, and 50 m and the
requested y/z Manhattan displacement for epsilon 0.0, 0.8, or 1.6. The paper
does not publish a unique procedure for reconstructing the original y/z
coordinates, so this is a deterministic **paper-compatible environment
generator**, not a claim of exact original gate coordinates.

## Three-drone run

`configs/settings_three_drones.json` is a ready-to-copy AirSim settings file.
Back up the current file, copy this file to
`C:\Users\chen\Documents\AirSim\settings.json`, stop and start UE4, and
click **Play** again. Then run:

```powershell
& $py run_three_drones_mpc.py --duration 15 --dx 3 --keep-gates
```

The three vehicles are independently controlled at 20 Hz. The script also
creates three independent gate groups by default: 15 gates and 60 cube bars.
Use `--no-gates` for a pure three-drone point-tracking test. Their generated
teacher files are `data\Drone1.csv`, `data\Drone2.csv`, and `data\Drone3.csv`.
The default corresponding gate centers have at least 6 m separation in y,
including the worst-case y excursion caused by epsilon. Adjust this with
`--group-spacing`; gate count and x placement use `--gate-count`, `--x-start`,
and `--x-spacing`. The implementation rejects a gate width that would overlap
the requested group spacing.

To make all three vehicles traverse the five gates simultaneously, run:

```powershell
& $py run_three_drones_gates.py --epsilon 0.8 --seed 0 --keep-gates
```

Each drone follows its own five gate centers. The gate width in this script
defaults to 4 m and can be changed with `--gate-width`. The previous 8 m
width with 6 m group spacing was unsafe because neighboring gate frames
overlapped; that combination is now rejected.

The reference trajectory is generated from the gate coordinates at runtime.
Each gate contributes a point 2 m before its plane and a point 1 m after it;
all points are joined into one constant-speed polyline. Intermediate points
retain non-zero reference velocity, so vehicles no longer stop before every
gate. A gate is counted only when the measured vehicle segment crosses the
gate plane in the forward direction and the interpolated crossing point is
inside the clear width/height after frame thickness and `--gate-clearance` are
applied. Multi-vehicle controller targets remain in each vehicle's local NED
frame, while visible gate actors are translated into UE world NED with a
runtime-measured per-vehicle offset.

## Teacher Dataset Collection

The automated collector uses one vehicle (`Drone1` by default), one UE4/AirSim
process, and the Cartesian product of 10 initial states and 40 deterministic
environments. Each episode records exactly 100 samples at `dt=0.05 s`; the
target output is therefore 400 episodes and 40,000 pairs with
`X.shape=(40000, 12)` and `U.shape=(40000, 4)`.

The collector launches `FalconEmpty` directly in UE4 game mode, polls the
AirSim RPC, resets the vehicle, removes and recreates the five current gates,
and saves each episode atomically before continuing. It does not require
opening the editor or clicking Play. The current standalone `FalconEmpty`
asset registry contains `EditorCube` rather than `1M_Cube`, so the dataset
config uses `EditorCube`; change `configs/teacher_collection.json` if the
registered asset changes.

First run the live smoke test:

```powershell
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location 'F:\FalconReproduction\airsim_control'
& $py collect_teacher_dataset.py --config configs\teacher_collection.json --auto-launch --smoke-test
& $py validate_teacher_dataset.py --dataset-root teacher_dataset_continuous_v2_smoke
```

For the complete collection, use the same command without `--smoke-test`:

```powershell
& $py collect_teacher_dataset.py --config configs\teacher_collection.json --auto-launch --resume
```

The five-second configuration above is the paper-compatible short-window
baseline. It normally reaches only the first gate when the gates are spaced
10 m apart. For a complete five-gate route, use the separate extended-route
configuration; it keeps the baseline dataset intact, records 700 samples at
20 Hz per episode, and rejects an episode unless all five target gates were
observed. The extended teacher collection uses the generated gate positions
as virtual geometry by default, so physical gate bars cannot corrupt the MPC
teacher trajectory through an incidental collision; visible bars remain
available in the interactive gate scripts. It also uses UE4 `-nullrhi` mode
because teacher collection does not require rendering and this avoids runtime
shader-worker failures on the current machine:

```powershell
& $py collect_teacher_dataset.py --config configs\teacher_collection_full_route.json --auto-launch --resume
& $py validate_teacher_dataset.py --dataset-root teacher_dataset_full_route_continuous_v2 --json
```

The extended collection uses a 35-second, 700-sample route window and writes
to `teacher_dataset_full_route_continuous_v2\`; it contains 400 episodes and
280,000 samples when complete. The 35-second window leaves roughly nine seconds
after a nominal five-gate traversal while avoiding the hover-heavy tail observed
in the 60-second pilot. The old `teacher_dataset_full_route\` remains a
stop-and-go baseline and is never resumed into the v2 policy. Manifests carry
a teacher-policy version so episodes from different controllers cannot be
mixed. The collector still saves each episode atomically and can be resumed.

## Robust teacher-data pipeline

The separate paper-interface 12-D state-query pipeline is documented in
[`TRACK_P_12D_TRAINING.md`](TRACK_P_12D_TRAINING.md). It keeps the validated
25-D Track-E model intact and generates independent
`[p_gate-p, r_gate-r, velocity, acceleration] -> MPC control` queries with a
fixed, deterministic reference rule.

### Independent path-aware MPC queries (recommended)

The current recommended dataset no longer treats a complete five-gate flight
as one training example sequence.  Every row is an independent MPC query:
a deterministic environment and random route progress define the future
reference window, while bounded nominal/moderate/recovery deviations define
the vehicle position error, attitude error, velocity and acceleration.  MPC
is recomputed for every sampled row; labels are never copied from another
state.

Independent does not mean physically arbitrary. The sampler keeps position,
attitude and velocity deviations inside three configurable envelopes and
constructs every reference window from a valid five-gate polyline. This avoids
impossible/extreme combinations while covering recovery states that nominal
closed-loop demonstrations rarely visit. Acceleration is randomized as an
input-invariance test; the current engineering MPC teacher does not directly
use measured acceleration when computing its label.

The learned input is 25-D: the legacy 12-D controller state, four future
reference positions relative to the vehicle (offsets 1/5/10/19), and one
reference-speed scalar.  The output remains `[F, wx, wy, wz]`.  The default
configuration creates 60 environment shards with 12,000 independent samples
each (720,000 total).  Shards are atomic and resumable, and this offline stage
does not start AirSim.

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

# Small end-to-end dataset check.
& $trainPy .\generate_iid_mpc_dataset.py --config .\configs\teacher_iid_mpc_v1.json --smoke-test
& $trainPy .\validate_iid_mpc_dataset.py --dataset-root .\teacher_dataset_iid_path_v1_smoke_v2 --label-checks 128

# Full independent dataset. If interrupted, repeat with --resume.
& $trainPy .\generate_iid_mpc_dataset.py --config .\configs\teacher_iid_mpc_v1.json
& $trainPy .\generate_iid_mpc_dataset.py --config .\configs\teacher_iid_mpc_v1.json --resume
& $trainPy .\validate_iid_mpc_dataset.py --dataset-root .\teacher_dataset_iid_path_v1 --label-checks 1000

# Leakage-safe environment split and path-aware training.
& $trainPy .\prepare_dnn_dataset.py --dataset .\teacher_dataset_iid_path_v1\final\teacher_dataset_iid_path_v1.npz --output-dir .\dnn_dataset_iid_path_v1
& $trainPy .\train_dnn_controller.py --data-dir .\dnn_dataset_iid_path_v1 --output-dir .\dnn_runs\iid_path_v1_depth10_width512 --depth 10 --width 512 --epochs 20000 --batch-size 1024 --learning-rate 0.0001 --patience 200 --device auto
```

The original trajectory collections below remain available as baselines, but
they are not the recommended source for the next path-aware model.

After the path-aware model passes single-vehicle closed-loop evaluation, run
three independent copies of it against three independent five-gate lanes:

```powershell
& $trainPy .\run_three_drones_dnn_gates.py --model .\dnn_runs\iid_path_v1_depth10_width512\controller_scripted.pt --normalization .\dnn_runs\iid_path_v1_depth10_width512\normalization.json --epsilon 0.8 --seed 0 --speed 2.0 --group-spacing 6 --gate-count 5 --gate-width 4 --gate-clearance 0.25 --keep-gates
```

The runner loads three DNN instances, constructs a separate 25-D state/path
input for every vehicle at 20 Hz, validates every physical gate-plane
crossing, checks collisions, and writes a JSON timing/result report.

The original `teacher_dataset_full_route_continuous_v2\` is preserved. The
new robust collection is a separate 12-D/4-D teacher dataset: eight wider
initial pose conditions crossed with 60 deterministic environments, five
reference speeds (1.5--3.5 m/s), and three gate spacings (8/10/12 m). Each
episode also records four future reference positions as `path_context`; this
does not change `X.shape=(N,12)` and `U.shape=(N,4)`.

Run the following from PowerShell. The first command starts a new collection;
if it is interrupted, repeat it with `--resume`. `--render` is optional and
makes UE4 visible; omit it for the faster headless collection.

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location $proj

# New robust nominal MPC data: 480 episodes x 700 samples.
& $trainPy .\collect_teacher_dataset.py --config .\configs\teacher_collection_robust_v1.json --auto-launch

# Continue after Ctrl+C, terminal closure, or an interrupted episode.
& $trainPy .\collect_teacher_dataset.py --config .\configs\teacher_collection_robust_v1.json --auto-launch --resume

# After completion, verify the collection.
& $trainPy .\validate_teacher_dataset.py --dataset-root .\teacher_dataset_robust_v1 --json
```

After the nominal collection is complete, generate MPC-relabelled state
perturbations offline. This stage does not connect to AirSim and leaves the
nominal directory unchanged. It can also be resumed; it creates a combined
nominal-plus-perturbation directory with `data_source=0` for nominal rows and
`data_source=1` for perturbation rows.

```powershell
& $trainPy .\generate_perturbed_teacher_dataset.py --source-root .\teacher_dataset_robust_v1 --output-root .\teacher_dataset_robust_v1_augmented

# If the terminal is interrupted, continue the same output.
& $trainPy .\generate_perturbed_teacher_dataset.py --source-root .\teacher_dataset_robust_v1 --output-root .\teacher_dataset_robust_v1_augmented --resume

& $trainPy .\validate_teacher_dataset.py --dataset-root .\teacher_dataset_robust_v1_augmented --json
```

The DAgger collector is intentionally a later stage: it requires a first DNN
checkpoint. It applies the DNN inside a safety envelope, evaluates MPC on the
same measured state, and writes the MPC value as the label (`data_source=2`),
with both the applied student action and shield flag retained for analysis.
The model is pinned in the manifest, so a resumed run cannot silently mix two
different checkpoints. The command is provided here for the next stage; it
must not be run before the first robust DNN checkpoint exists:

```powershell
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
& $trainPy .\collect_dagger_teacher_dataset.py --model .\dnn_runs\robust_v1\controller_scripted.pt --source-root .\teacher_dataset_robust_v1_augmented --output-root .\teacher_dataset_dagger_v1 --render

# Continue a stopped DAgger collection.
& $trainPy .\collect_dagger_teacher_dataset.py --model .\dnn_runs\robust_v1\controller_scripted.pt --source-root .\teacher_dataset_robust_v1_augmented --output-root .\teacher_dataset_dagger_v1 --render --resume
Remove-Item Env:PYTHONPATH
```

The preparation script now supports both the old 40-environment dataset and
the new 60-environment dataset without random row-level splitting. It holds
out complete environment IDs (48/6/6 for 60 environments) and preserves the
optional robust metadata in the split files:

```powershell
& $trainPy .\prepare_dnn_dataset.py --dataset .\teacher_dataset_robust_v1_augmented\final\teacher_dataset_robust_v1_augmented.npz --output-dir .\dnn_dataset_robust_v1
```

## Prepare and train the DNN controller

Do not randomly split individual 20 Hz rows: adjacent samples from one flight
would leak into both training and test data. The preparation script holds out
entire gate environments while preserving the paper's 80/10/10 ratio:
environment IDs 0--31 train, 32--35 validation, and 36--39 test. With the full
35-second collection this produces 224,000 / 28,000 / 28,000 samples.

```powershell
& $py .\prepare_dnn_dataset.py

# The validated Windows training environment for this project.
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'

# Short pipeline check before a long run.
& $trainPy .\train_dnn_controller.py --depth 10 --width 512 --epochs 3 --batch-size 1024 --device cuda --output-dir .\dnn_runs\smoke_depth10_width512

# Resumable depth=10, width=512 baseline; --epochs is the total target epoch.
& $trainPy .\train_dnn_controller.py --depth 10 --width 512 --epochs 20000 --batch-size 1024 --learning-rate 0.0001 --patience 200 --device cuda --output-dir .\dnn_runs\depth10_width512

# If interrupted, continue from the latest atomic checkpoint.
& $trainPy .\train_dnn_controller.py --depth 10 --width 512 --epochs 20000 --batch-size 1024 --learning-rate 0.0001 --patience 200 --device cuda --resume --output-dir .\dnn_runs\depth10_width512
```

The trainer standardizes inputs and outputs using training-only statistics,
selects the checkpoint with the lowest validation MSE, evaluates physical-unit
metrics on the untouched test set only after training, and writes
`best_model.pt`, `latest_checkpoint.pt`, `controller_scripted.pt`,
`history.csv`, `per_output_metrics.csv`, `per_gate_metrics.csv`,
`test_predictions.npz`, `model_spec.json`, `normalization.json`, and
`metrics.json`. A CUDA-capable PyTorch installation is strongly recommended
for the paper-scale network.

The framework-neutral export and fixed golden-vector check are run with:

```powershell
& $trainPy .\export_dnn_artifacts.py --model .\dnn_runs\depth10_width512\best_model.pt --output-dir .\dnn_runs\depth10_width512\export --golden-count 1000
& $trainPy .\evaluate_dnn_controller.py --model .\dnn_runs\depth10_width512\best_model.pt --split validation --device cuda
& $trainPy .\evaluate_dnn_controller.py --model .\dnn_runs\depth10_width512\best_model.pt --split test --device cpu
```

The DNN closed-loop runner uses the same gate layout, continuous polyline
reference and physical gate-plane validation as the MPC runner; it replaces
only the MPC solve call:

```powershell
& $trainPy .\run_single_drone_dnn_gates.py --model .\dnn_runs\depth10_width512\controller_scripted.pt --device cpu --epsilon 0.8 --seed 0 --per-gate-timeout 30 --keep-gates
```

For automated held-out layouts (environment IDs 36--39, all ten initial
states) and the MPC/DNN comparison report:

```powershell
& $trainPy .\evaluate_dnn_closed_loop.py --model .\dnn_runs\depth10_width512\controller_scripted.pt --max-episodes 1 --auto-launch --render
& $trainPy .\evaluate_dnn_closed_loop.py --model .\dnn_runs\depth10_width512\controller_scripted.pt --auto-launch
& $trainPy .\compare_mpc_dnn.py --model .\dnn_runs\depth10_width512\best_model.pt --offline-only
& $trainPy .\compare_mpc_dnn.py --model .\dnn_runs\depth10_width512\controller_scripted.pt --auto-launch
```

The current Windows phase intentionally stops before Vitis AI quantization or
DPU compilation. The resulting FP32 checkpoint, model description,
normalization and NPZ weights are the bridge for the later Vitis AI 1.2.1
Ubuntu/Docker phase.

For a live AirSim run, the AirSim Python source and its RPC dependency path
must be added while retaining the training venv's PyTorch:

```powershell
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
& $trainPy .\run_single_drone_dnn_gates.py --model .\dnn_runs\depth10_width512\controller_scripted.pt --device cpu --epsilon 0.8 --seed 0 --gate-asset EditorCube --auto-launch
Remove-Item Env:PYTHONPATH
```

The first completed baseline run achieved the offline metrics above, but the
pure DNN closed-loop test passed 0/40 held-out combinations while the same
scene MPC comparison passed 5/5 gates. This is recorded in
`dnn_runs\depth10_width512\BASELINE_REPORT.md`; do not proceed to DPU
deployment until an off-trajectory robustness phase improves that result.

The output is stored under `teacher_dataset\`: `manifest.json`, the generated
`initial_states.json` and `environments.json`, one atomic NPZ per episode, the
merged `final\teacher_dataset.npz`, `final\teacher_metadata.csv`, and
`summary.txt`. If the process is interrupted, run the same command with
`--resume`; valid completed episodes are skipped and only missing/corrupt or
unfinished episodes are collected again. RPC failures trigger a bounded
retry and a restart of the UE4 process that the collector owns.

The paper specifies the counts, five-second window, 20 Hz period, 12-D input,
and 4-D output. The random ranges, seeds, gate coordinates, difficulty
generator, asset choice, retry policy, and timeout are engineering
assumptions because the paper does not publish those details. The generated
layouts are deterministic and paper-compatible, not claimed to be the exact
FALCON layouts. Use `validate_teacher_dataset.py` to verify shapes, finite
values, episode lengths, state/environment coverage, and summary statistics.

## Data contract and assumptions

Each CSV row contains the paper-compatible 12-D input `x0 ... x11`, the
four-value teacher output `F, wx, wy, wz`, raw AirSim state, reference,
solver time, actual loop period, and deadline-miss flag. `dt=0.05 s` and
`horizon=20` are the paper-specified interface values.

The paper does not provide enough information to reproduce its exact MPC:
complete dynamics parameters, constraints, actuator mapping, and solver
configuration are absent. `mpc_controller.py` is therefore explicitly an
engineering **FALCON-compatible MPC teacher**, not the exact FALCON/PAMPC
implementation. The current thrust mapping is anchored at the measured
hover throttle in `configs\thrust_calibration.json`; its local throttle slope
remains a configurable engineering assumption.

The linear prediction model now tracks all 20 future position/velocity samples
rather than discarding samples 2-20. AirSim controller state is expressed in
each vehicle's local NED frame; visible scene actors use UE world NED and are
translated explicitly. The attitude error is `q_target * inverse(q_current)`,
converted to a rotation vector; quaternions are never subtracted directly.

## M1-M12 acceptance status

The implementation covers all requested M1-M12 interfaces and workflows. The
local unit/integration checks currently pass 49/49, including legacy 12-D and
path-aware 25-D inputs, the 4-D control, epsilon generation, 15 independent
visible gates, three named vehicles, and CSV schema.

The evidence boundary for live AirSim acceptance is:

- M1-M6: previously validated for state acquisition, hover calibration,
  single-drone MPC hover/point tracking, and the 20 Hz loop.
- M9-M11: previously validated for three named vehicles, simultaneous
  lifecycle, and three independent point-tracking MPC loops.
- M12: CSV files and the required fields have been generated and checked.
- M7-M8: the gate generator and the revised alignment/approach/crossing logic
  are implemented and unit-tested, but the revised 3-drone gate run still
  needs one fresh AirSim validation for all five gates and the selected
  epsilon. The earlier log used the old center-distance behavior and is not
  sufficient evidence of physical gate passage.
