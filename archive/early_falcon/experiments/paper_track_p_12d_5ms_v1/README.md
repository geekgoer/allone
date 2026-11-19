# Paper-interface Track-P 12-D, fixed 5 m/s

This experiment is isolated from the earlier fixed-2 m/s Track-P baseline and
the validated 25-D Track-E controller.

## Evidence boundary

FALCON Sec. III-A (PDF page 5) specifies the 12-D input
`[p_g-p, r_g-r, v, v_dot]`, the 4-D output `[F, Omega]`, six independently
trained FC/ReLU architectures, a 0.05 s control period, 20,000 epochs, and a
1e-4 learning rate. It states that path planning produces a reference
trajectory from current state and gate environment, but does not publish a
cruise speed or enough trajectory-generator details to reproduce its exact
speed profile. The 5 m/s reference in this experiment is therefore an explicit
reproduction choice, not a value claimed by the paper.

The dataset keeps the paper's 12-D interface while using independent robust MPC
queries. Actual velocity is broadly varied around and beyond 5 m/s. Reference
speed is fixed at 5 m/s because it is not present in the 12-D student input.

## Generate and validate

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $trainPy .\generate_track_p_12d_dataset.py `
  --config .\experiments\paper_track_p_12d_5ms_v1\teacher_config.json `
  --smoke-test --resume

& $trainPy .\validate_track_p_12d_dataset.py `
  --dataset-root .\experiments\paper_track_p_12d_5ms_v1\teacher_dataset_smoke `
  --label-checks 128

& $trainPy .\generate_track_p_12d_dataset.py `
  --config .\experiments\paper_track_p_12d_5ms_v1\teacher_config.json `
  --resume

& $trainPy .\validate_track_p_12d_dataset.py `
  --dataset-root .\experiments\paper_track_p_12d_5ms_v1\teacher_dataset `
  --label-checks 2000

& $trainPy .\prepare_dnn_dataset.py `
  --dataset .\experiments\paper_track_p_12d_5ms_v1\teacher_dataset\final\teacher_dataset_track_p_12d_5ms_v1.npz `
  --output-dir .\experiments\paper_track_p_12d_5ms_v1\dnn_dataset
```

## Train all six networks

Close AirSim/UE4 before training. Adaptive mode starts up to three independent
CUDA jobs according to currently free VRAM. If parallel execution produces a
CUDA OOM, it drains the other active jobs and continues serially. A serial OOM
halves only that model's batch size. Every model has independent epoch and best
checkpoints, so completed work survives a later OOM, interruption, or power
loss.

```powershell
& $trainPy .\train_track_p_12d_adaptive_parallel.py `
  --data-dir .\experiments\paper_track_p_12d_5ms_v1\dnn_dataset `
  --runs-root .\experiments\paper_track_p_12d_5ms_v1\runs `
  --run-prefix paper_track_p_12d_5ms_v1 `
  --models all `
  --max-parallel auto `
  --epochs 20000 `
  --batch-size 2048 --min-batch-size 256 `
  --learning-rate 0.0001 --patience 200 `
  --device cuda
```

Run the same command again after an interruption. Completed models are skipped
and partial models resume from `latest_checkpoint.pt`.

`--patience 200` selects the best validation checkpoint and normally stops far
before the 20,000-epoch ceiling. To reproduce the paper's literal fixed epoch
count, use `--patience 0`; this is much slower on the 480,000-query robust
dataset and the paper does not disclose its checkpoint-selection details.

Overall status is written to
`runs/paper_track_p_12d_5ms_v1_adaptive_status.json`. Each model directory
contains `adaptive_training.log`, `latest_checkpoint.pt`, `best_model.pt`,
`controller_scripted.pt`, `normalization.json`, `model_spec.json`, and
`metrics.json` after successful completion.

## Stable serial recovery after a Windows CUDA driver reset

If the adaptive run reports `illegal memory access` or Windows logs an
`nvlddmkm` error, reboot Windows and use the strict serial launcher below. It
writes to a new root, leaves the failed adaptive run untouched, resumes only
old checkpoints that demonstrably beat the validation mean-output baseline,
and initializes collapsed deep models afresh with Kaiming/ReLU weights.

```powershell
& $trainPy .\train_track_p_12d_serial_recovery.py `
  --data-dir .\experiments\paper_track_p_12d_5ms_v1\dnn_dataset `
  --runs-root .\experiments\paper_track_p_12d_5ms_v1\runs_serial_v2 `
  --run-prefix paper_track_p_12d_5ms_v1_serial_v2 `
  --seed-runs-root .\experiments\paper_track_p_12d_5ms_v1\runs `
  --seed-run-prefix paper_track_p_12d_5ms_v1 `
  --models all `
  --epochs 20000 --batch-size 2048 --min-batch-size 256 `
  --learning-rate 0.0001 --patience 200 `
  --max-validation-baseline-ratio 0.25 `
  --cooldown-seconds 30 `
  --device cuda
```

The same command is restartable. Healthy source runs contribute only their
best model weights; Adam state is deliberately reset so a checkpoint from a
previous driver-failure session cannot carry stale optimizer state. A fatal CUDA error stops the entire sweep so
that no later model is trained in a suspect driver session. OOM alone reduces
the batch size and retries the same model serially.
