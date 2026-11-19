# 300D generalization audit

Date: 2026-09-17. Diagnostic only: no training, dataset, controller, or native solver changes. Test split was not evaluated by these diagnostics.

## Scope and main finding

CPU evaluation of one loaded `depth10_width512/best_model.pt` checkpoint (epoch 332), with identical inference mode and normalization on train and validation. The path may subsequently be updated by training; these results refer to the checkpoint loaded for this audit.

| Dataset | Samples | Normalized MSE |
|---|---:|---:|
| Train | 112000 | 0.0031742134 |
| Validation | 14000 | 0.0512630543 |

Ratio: 16.1498. This is a real generalization gap, not merely the online-training versus end-of-epoch evaluation distinction. Checkpoint/data normalization differences are zero for x_mean, x_std, u_mean, u_std.

Normalized MSE is the mean over samples and four outputs of `(prediction_physical - target_physical)^2 / train_u_std^2`. Physical MSE averages unnormalized squared errors across outputs with different units; it is not a single physical-unit tracking error.

## Error localization

| Source | Train normalized MSE | Validation normalized MSE |
|---|---:|---:|
| Continuous-core queries | 0.003045 | 0.014113 |
| Independent recovery queries | 0.003632 | 0.074447 |
| Short-rollout queries | 0.002617 | 0.053636 |

Validation normalized per-output MSE: thrust 0.0065254, roll rate 0.0106423, pitch rate 0.1310589, yaw rate 0.0568255. Pitch contributes 63.91% of normalized SSE and yaw 27.71%. This weighting is intentional normalization, not an arithmetic error. Output standard deviations are approximately [2.83524, 1.37441, 0.86631, 0.23594].

Nominal-family validation MSE is 0.01532, combined-recovery-family 0.15701. The worst 1% of validation rows accounts for 25.23% of SSE. Tail cases matter substantially, but do not explain the entire gap: median validation row MSE is 0.01230 versus training 0.001929.

## Split and coverage

All 40 environments appear in both splits. Source proportions are identical. The maximum per-feature mean difference is only 0.03657 training standard deviations; this does NOT prove identical joint distributions or adequate local coverage.

Query-group overlap is zero. However, 400 base episode IDs and 8804 base row IDs occur on both sides; five exact X rows overlap. Thus group separation is not full trajectory/provenance separation. Related perturbations and nearby core frames can cross splits. This would generally make validation optimistic, not explain why it is much worse. Claims of complete leakage freedom would be too strong.

Training core contains 32000 rows but 31985 distinct base row IDs. Retries can replace a selected core row with another row. This is a limited duplication issue, not evidence that most labels are corrupted.

Actual speed <=3.5 m/s: zero samples in both train and validation. Reference-position-error norm train median/p99/max: 0.1189/0.4484/0.8861 m; validation 0.1213/0.4563/0.8680 m. Therefore, this is mainly a cruise-neighborhood dataset, not a demonstrated takeoff/large-recovery dataset. This limitation is a separate closed-loop deployment risk; it does not by itself explain an in-distribution validation gap.

## Teacher replay

300 stored problems were reconstructed directly from their 300D inputs: 100 random and up to 50 highest-error rows from each split. The normal query stopping rule was replayed, then compared with 64 refinement iterations and with changed query ordering.

| Diagnostic | Mean max-absolute control difference per row | Maximum |
|---|---:|---:|
| Replayed label vs stored label | 6.47e-6 | 1.90e-4 |
| 64 iterations vs normal stopping | 1.63e-4 | 0.00256 |
| Changed query ordering | 6.30e-6 | 0.000281 |

These differences are much smaller than DNN errors in the inspected cases. Teacher nondeterminism or insufficient refinement is not supported as the dominant explanation by this audit. This is not an exhaustive proof for all 140000 labels.

Code caveat: reset reconstructs the primal guess but does not fully clear all dual workspace state. The stopping rule checks first-action change, not full KKT convergence. Some early KKT residuals were large (maximum 0.308), while 64-iteration residual maximum was 7.21e-7. These are real numerical-quality caveats, but measured first-control effects here were small.

## Input conditioning

The current 300D contract contains the current 10D state, 21 reference states (210D), and 20 reference/feedforward controls (80D), under fixed solver parameters. Including numerical problem information does not guarantee finite-data learnability, global optimality, closed-loop stability, or no overfitting.

Absolute x-position std is 14.0978 m and first-reference x std is 14.1496 m; their correlation is 0.999983. Yet their difference std is only 0.09696 m. A plain fully connected network must extract small tracking errors from highly correlated absolute inputs, together with a redundant reference window.

A lossless candidate recoding keeps current state and all quaternion/control entries, while replacing each reference p/v by reference-minus-current p/v. Original features can be reconstructed exactly before floating-point rounding. This changes representation, not the teacher problem or available information.

Exploratory CPU kNN diagnostic on 1000 randomly selected validation rows: eight-neighbor normalized MSE was 0.32272 in the original standardized representation and 0.24573 after lossless relative-p/v recoding (23.9% reduction). Both are worse than the DNN. This supports testing the recoding, but is NOT proof that retrained DNN error will improve by that amount.

## Interpretation and next controlled experiment

The small model has 2257412 parameters and fits training rows much better than unseen rows. The observed curve is consistent with overfitting/generalization limitation, especially in combined-recovery and angular-rate regions. Simple insufficient optimization of training loss is not the main evidence here. Root-cause fractions cannot be assigned from this audit alone.

Recommended order, not implemented in this audit:

1. Preserve raw data, current split and baseline results. Do not regenerate all labels or train all six models yet.
2. Test the lossless relative-p/v 300D recoding against original inputs with identical split, initialization seed, optimization budget and output normalization. Fit preprocessing statistics only on training data; no old-checkpoint resume across changed preprocessing.
3. Compare same-checkpoint train-eval/validation per-output and per-source metrics, not only the scalar total. Do not use test to select these changes.
4. Separately strengthen provenance grouping while retaining within-environment splitting. Keep short rollouts and perturbations of the same anchor together; use episode-level or suitably separated trajectory blocks for the core. Do not hide leakage by removing only exact duplicates.
5. If gap persists, isolate modest regularization and reference-window encoding experiments one at a time; do not simultaneously change model, split, labels and optimizer.
6. Before takeoff-to-cruise deployment, explicitly cover low-speed/startup states or define and validate a separate startup controller. Current zero low-speed coverage is not acceptable evidence of autonomous takeoff capability.

Neither 300D inputs nor any of these changes guarantees monotonically improving test MSE with model size. Use honest validation-selected checkpoints and unchanged final evaluation protocols.

## Artifacts

- `report.json`: same-checkpoint evaluation, subgroup errors and split statistics.
- `teacher_replay.json`: individual replay/refinement records and summaries.
- `feature_geometry.json`: lossless-recoding kNN diagnostic and conditioning statistics.
- `train_errors.npz`, `validation_errors.npz`: per-row normalized prediction errors.
