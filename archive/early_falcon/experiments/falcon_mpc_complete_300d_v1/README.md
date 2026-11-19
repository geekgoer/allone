# MPC-complete 300-D DNN experiment

This experiment is independent of every existing 12-D data set and model.
Its input contains the full online-varying ACADO trajectory-tracking problem:

```text
current ACADO state                  10
21 ACADO reference states       21 x 10 = 210
20 ACADO reference controls     20 x  4 =  80
-------------------------------------------
total                                      300
```

The 20 reference controls are planner feed-forward values derived from planned
acceleration and planned body rates.  They are not the MPC solution.  The label
is the first control of a deterministic, reset-and-refined ACADO query.  A
query makes at least four RTI updates and stops once the maximum first-control
change is at most ``1e-3`` or after sixteen updates.

Run the offline contract check:

```powershell
& $py .\experiments\falcon_mpc_complete_300d_v1\smoke_test_300d.py
```

Run the local DLL/teacher repeatability check:

```powershell
& $py .\experiments\falcon_mpc_complete_300d_v1\smoke_test_300d.py --teacher --report .\experiments\falcon_mpc_complete_300d_v1\smoke_teacher.json
```

## Generation gate and collection

Before generating anything, run the real-label preflight:

```powershell
& $py .\experiments\falcon_mpc_complete_300d_v1\preflight_300d.py
```

The checked configuration uses an environment-disjoint,
difficulty-stratified split rather than allocating a prefix of environment IDs
to training. It plans 140,000 rows: 1,000 recomputed continuous-core rows,
1,500 independent recovery rows, and 1,000 rows arranged as 200 five-step
short rollouts per environment. The manifest and each 2,000-row chunk are
written atomically, so an interrupted run resumes with `--resume`.

```powershell
& $py .\experiments\falcon_mpc_complete_300d_v1\collect_300d.py --resume
```

Collection is offline: it does not open UE4/AirSim. It needs the ACADO DLL and
the immutable 40k core source dataset. For a separate 360-row disposable
pipeline test:

```powershell
& $py .\experiments\falcon_mpc_complete_300d_v1\collect_300d.py --smoke-test
```

## Training split

The collector produces a single provenance-preserving dataset. For training,
use the separate **within-environment** split command below. Every one of the
40 environments appears in train, validation, and test; source kind, control
bucket, and all five rows of each short rollout remain balanced/group-safe.
This avoids an artificial train/validation distribution gap caused only by
withholding entire environments.

```powershell
& $trainPy .\experiments\falcon_mpc_complete_300d_v1\prepare_training_dataset.py
```

The resulting `dnn_dataset_300d_140k_v1_within_environment` directory holds
`train.npz`, `validation.npz`, `test.npz`, train-only `normalization.json`,
and `split_manifest.json`.
