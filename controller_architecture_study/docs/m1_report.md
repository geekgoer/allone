# M1 — Controller Complexity Characterization

## 1. Motivation

Measure the accuracy–complexity tradeoff before designing a compact elastic controller.

## 2. Immutable FALCON baseline

Six existing checkpoints and reported held-out test metrics are inventoried in `../results/falcon_baseline.csv`; none were retrained.

## 3. Experiment protocol

See `experiment_protocol.md`. New models share the same dataset, split, normalization and normalized-MSE objective.

## 4. Tested architectures

0/11 completed; pending: plain_mlp_h10_w256_seed0, plain_mlp_h10_w512_seed0, plain_mlp_h2_w128_seed0, plain_mlp_h2_w64_seed0, plain_mlp_h4_w128_seed0, plain_mlp_h4_w256_seed0, plain_mlp_h4_w64_seed0, plain_mlp_h6_w128_seed0, plain_mlp_h6_w256_seed0, plain_mlp_h8_w128_seed0, plain_mlp_h8_w256_seed0. New models use seed 0.

## 5. Offline metrics

See `../results/architecture_scaling_summary.csv` and each completed run's `offline_metrics.json`.

## 6. Complexity metrics

MACs and parameters count FC operations at batch size one, excluding bias additions; see the summary CSV.

## 7. MAC–quality Pareto

See `../results/mac_vs_physical_mse.svg` and `../results/parameters_vs_physical_mse.svg`. Baselines used different seeds and training budgets, so the plot is descriptive, not a controlled architecture-only comparison.

## 8. Candidate saturation region

Pending full 11-model sweep and examination of measured marginal gains; no absolute threshold or predetermined winner.

## 9. Candidate models for closed-loop evaluation

Pending measured sweep. Select a small, knee-point, slightly-overprovisioned and FALCON reference (3–5 models) only after reviewing the complete curves.

## 10. Limitations

One seed per new architecture, offline imitation errors do not establish closed-loop safety, and FALCON baseline training recipes vary. No hardware latency is inferred from MACs.

## 11. Next step

Complete the coarse sweep, review the Pareto curves, then choose candidates for AirSim closed-loop validation. Do not select the SuperNet backbone yet.
