# 10 m/s hybrid 160k ACADO dataset

This is a new, isolated dataset pipeline.  It does not alter prior 5 m/s,
10 m/s short-rollout, historical FPGA, or strict-paper experiment folders.

## Dataset contract

Every sample has exactly the same 12-D trajectory-relative contract:

```text
[p_ref - p, rotvec(q_ref * inverse(q)), v_ref - v, a_ref]
```

The fixed reference speed is 10 m/s.  The four labels are the first control
from the official public `rpg_mpc` ACADO/qpOASES solver:

```text
[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]
```

The final data size is exactly 160,000 rows:

| Source kind | Rows | Role |
|---|---:|---|
| 0: AirSim continuous core | 40,000 | 10 initial states x 40 environments x 5 s x 20 Hz |
| 1: phase-consistent recovery | 40,000 | low-speed, cruise, lateral, vertical, attitude and combined recovery |
| 2: 5-step phase-consistent rollout | 40,000 | 8,000 local 250-ms rollouts |
| 3: critical phase query | 40,000 | near-gate and gate-transition states |

The core provides real AirSim states.  The other 120k rows use the immutable
global reference trajectory from the core, perturb the measured state, and
relabel the resulting state with ACADO.  They never replan a new trajectory
from the perturbation, so the reference phase remains well-defined.

## 1. Preflight the ACADO library

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py -c "from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig,AcadoRtiMPC; c=AcadoRtiMPC(AcadoMPCConfig(library='native/acado_rpg_mpc/build/bin/libfalcon_acado_mpc.dll')); print('ACADO ready, horizon=',c.config.horizon)"
```

## 2. Optional visible core smoke test

Manually start Blocks and click **Play**, then run:

```powershell
& $py .\experiments\falcon_10ms_hybrid_160k_v1\collect_core.py `
  --smoke-test `
  --render `
  --resume
```

This writes only the independent `core_40k_airsim_acado_10ms_v2_safe_start_smoke` directory.

The actual core starts near the 10 m/s reference speed (`v_x=9.4--10.6`
m/s), rather than from rest.  This is deliberate: setting a 0--3 m/s
AirSim plant state against a trajectory whose boundary velocity is 10 m/s
caused a physically unsafe initial catch-up transient and Ground collisions.
Low-speed takeoff and recovery are instead supplied by the separately
relabelled `launch_recovery` extension rows, where ACADO sees the same 10 m/s
trajectory phase but a realistic 0--3.5 m/s plant velocity.

## 3. Generate the immutable 40k AirSim core

For visible execution, keep the manually started AirSim session running:

```powershell
& $py .\experiments\falcon_10ms_hybrid_160k_v1\collect_core.py --resume
```

For managed headless collection instead:

```powershell
& $py .\experiments\falcon_10ms_hybrid_160k_v1\collect_core.py `
  --auto-launch `
  --resume
```

The task is resumable.  Do not change `core_40k_config.json` after data exists;
the configuration hash intentionally prevents incompatible resume.

## 4. Audit the core before extensions

```powershell
& $py .\validate_teacher_dataset.py `
  --dataset-root .\experiments\falcon_10ms_hybrid_160k_v1\core_40k_airsim_acado_10ms_v2_safe_start `
  --json
```

The final core must report `X=(40000,12)` and `U=(40000,4)`.

## 5. Optional offline extension smoke test

This does not require AirSim once the core exists.

```powershell
& $py .\experiments\falcon_10ms_hybrid_160k_v1\generate_extensions.py `
  --smoke-test `
  --resume
```

## 6. Generate 120k extensions

```powershell
& $py .\experiments\falcon_10ms_hybrid_160k_v1\generate_extensions.py --resume
```

The generator writes complete chunks atomically.  An interruption can be
resumed with exactly the same command.

The v3 extension sampler assigns each query a desired control bucket before
sampling: 90% interior, 7% near-limit and 3% saturated for independent
queries; short five-step rollouts are restricted to interior/near-limit
commands.  This prevents the prior v2 failure mode, where early low-speed
launch samples consumed the full global saturation allowance.  The old v2
partial directory is deliberately retained only as a diagnostic artifact and
is not read by the v3 pipeline.

## 7. Merge, split, and audit 160k rows

```powershell
& $py .\experiments\falcon_10ms_hybrid_160k_v1\prepare_hybrid_dataset.py

& $py .\experiments\falcon_10ms_hybrid_160k_v1\validate_hybrid_dataset.py --json
```

The final training data appears in:

```text
experiments/falcon_10ms_hybrid_160k_v1/dnn_dataset_hybrid_160k_10ms_v3_bucket_balanced/
  train.npz
  validation.npz
  test.npz
  normalization.json
  split_manifest.json
```

The split is environment-disjoint: environments 0--31 train, 32--35
validation, and 36--39 test.  Normalization is computed from train only.

## 8. Optional unattended collection-to-training pipeline

After starting the visible 40k core collection in one PowerShell window, run
this in a *second* PowerShell window.  It waits on the atomic completed-core
manifest rather than trying to parse another terminal's text.  It then creates
the 120k offline extensions, merges and audits 160k rows, and serially trains
the six models in this order: 10x512, 34x1024, 10x1024, 22x512, 22x1024,
34x512.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\experiments\falcon_10ms_hybrid_160k_v1\wait_for_core_then_train.ps1
```

After the completed core passes its audit, the pipeline closes only the
`UE4Editor.exe` process whose command line identifies `Blocks.uproject`; this
frees GPU/CPU memory before offline extension generation and CUDA training.
To keep that AirSim window open, append `-StopAirSimAfterCore $false`.

Training is intentionally fixed at 8000 epochs per model with a constant
`1e-4` learning rate: it has neither early stopping nor a learning-rate
scheduler.  It is one CUDA process at a time, uses batch size 256, `workers=0`,
disabled pin-memory, gradient clipping, epoch-level atomic checkpoints, a
90-second cooldown between models, and up to two automatic resume attempts
after a child process fails.  The pipeline transcript is written under
`pipeline_logs/`.
