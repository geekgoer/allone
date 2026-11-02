# Track-P 12D MPC imitation training

This pipeline creates a new paper-interface model without modifying the
validated 25-D Track-E dataset, checkpoints, exports, or FPGA package.

## Exact contract

Each independent row has:

```text
X[0:3]   = current gate center - vehicle position (AirSim NED, m)
X[3:6]   = gate target attitude - vehicle attitude (rotation vector, rad)
X[6:9]   = current linear velocity (AirSim NED, m/s)
X[9:12]  = current linear acceleration (AirSim NED, m/s^2)
U        = [collective thrust acceleration, body rate x, body rate y, body rate z]
```

The MPC internally still receives a 20-sample future reference at 50 ms. To
avoid hidden-label ambiguity, this reference is generated deterministically
from the 12-D query: a fixed 2 m/s lane goes through the target gate along its
passage axis. Reference speed is deliberately fixed because it is not one of
the 12 student inputs. Actual vehicle velocity is broadly varied from reverse
motion through fast forward motion.

Rows are not consecutive flight samples. Six balanced families cover nominal
tracking, lateral recovery, vertical recovery, attitude recovery, fast
near-gate states, and combined recovery. Thirty percent of coordinates are
sampled near an envelope edge to prevent the easy central region from
dominating. MPC is called again for every row; labels are never reused after
perturbing an input.

The current engineering MPC uses position, attitude, and velocity in its
control law but does not directly feed measured acceleration into its
double-integrator optimization. Acceleration is still varied so the student
learns the same invariance. This is recorded in the manifest rather than being
presented as an unverified property of the paper's unavailable MPC code.

## Commands (PowerShell)

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj
```

Run the small pipeline check first. It does not start AirSim:

```powershell
& $trainPy .\generate_track_p_12d_dataset.py --config .\configs\teacher_track_p_12d_robust_v1.json --smoke-test --resume
& $trainPy .\validate_track_p_12d_dataset.py --dataset-root .\teacher_dataset_track_p_12d_robust_v1_smoke --label-checks 128
```

Generate the full 480,000-query dataset. The same command starts a new output
or resumes it; completed environment shards are retained:

```powershell
& $trainPy .\generate_track_p_12d_dataset.py --config .\configs\teacher_track_p_12d_robust_v1.json --resume
& $trainPy .\validate_track_p_12d_dataset.py --dataset-root .\teacher_dataset_track_p_12d_robust_v1 --label-checks 2000
```

Create deterministic environment-level 80/10/10 splits. No row-level random
split is used and test data is not used for early stopping:

```powershell
& $trainPy .\prepare_dnn_dataset.py `
  --dataset .\teacher_dataset_track_p_12d_robust_v1\final\teacher_dataset_track_p_12d_robust_v1.npz `
  --output-dir .\dnn_dataset_track_p_12d_robust_v1
```

Train the first depth-10, width-512 FP32 model. The trainer infers input size
12 from the split and checkpoints atomically. `--epochs` is a ceiling;
validation early stopping normally ends sooner:

```powershell
& $trainPy .\train_dnn_controller.py `
  --data-dir .\dnn_dataset_track_p_12d_robust_v1 `
  --output-dir .\dnn_runs\track_p_12d_robust_v1_depth10_width512 `
  --depth 10 --width 512 `
  --epochs 5000 --batch-size 2048 `
  --learning-rate 0.0001 --patience 150 `
  --device auto
```

Resume an interrupted training run without restarting:

```powershell
& $trainPy .\train_dnn_controller.py `
  --data-dir .\dnn_dataset_track_p_12d_robust_v1 `
  --output-dir .\dnn_runs\track_p_12d_robust_v1_depth10_width512 `
  --depth 10 --width 512 `
  --epochs 5000 --batch-size 2048 `
  --learning-rate 0.0001 --patience 150 `
  --device auto --resume
```

## Safe six-model sweep

The paper architecture matrix is `depth={10,22,34}` by
`width={512,1024}`.  Train it with one isolated CUDA process at a time:

```powershell
& $trainPy .\train_track_p_12d_sweep.py `
  --data-dir .\dnn_dataset_track_p_12d_robust_v1 `
  --runs-root .\dnn_runs `
  --models all `
  --epochs 5000 --batch-size 2048 --min-batch-size 256 `
  --learning-rate 0.0001 --patience 150 `
  --device cuda
```

The already completed `depth10_width512` run is detected and skipped. Each
architecture has its own directory, `latest_checkpoint.pt` is updated after
every completed epoch, and `best_model.pt` is updated on every validation
improvement. A CUDA out-of-memory failure is retried at half batch size and a
repeated invocation resumes partial models. Completed models are never
discarded because a later architecture fails. Overall progress is recorded in
`dnn_runs/track_p_12d_sweep_status.json`, while each model has a separate
`sweep_training.log`.

The Track-P model deliberately has no reference-speed input. Supplying a
different `--speed` to the AirSim gate runner does not command a different
cruise speed; matched evaluation must use `--speed 2.0`.

After training, export framework-neutral artifacts and golden vectors:

```powershell
& $trainPy .\export_dnn_artifacts.py `
  --model .\dnn_runs\track_p_12d_robust_v1_depth10_width512\best_model.pt `
  --data-dir .\dnn_dataset_track_p_12d_robust_v1 `
  --output-dir .\dnn_runs\track_p_12d_robust_v1_depth10_width512\export `
  --golden-count 1000
```

Do not overwrite or rename `dnn_runs\iid_path_v1_depth10_width512`; that is
the validated 25-D Track-E model and remains the engineering fallback.

## Later AirSim use

The Track-P checkpoint must be fed gate-center errors, not the legacy current
reference-point errors. The single- and three-drone runners now expose this as
an explicit option. Use `--input-target gate` and keep `--speed 2.0`, matching
the fixed teacher reference speed:

```powershell
$env:PYTHONPATH = 'F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages'
& $trainPy .\run_single_drone_dnn_gates.py `
  --model .\dnn_runs\track_p_12d_robust_v1_depth10_width512\controller_scripted.pt `
  --normalization .\dnn_runs\track_p_12d_robust_v1_depth10_width512\normalization.json `
  --input-target gate --speed 2.0 --device cpu --keep-gates
```

Omitting `--input-target gate` would give this model the wrong semantics even
though the array shape would still be 12. A different commanded cruise speed
requires a separate fixed-speed 12-D model or an additional speed/path input.
