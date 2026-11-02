# Elastic MLP Test A

This package evaluates whether one trained FALCON MLP can become a useful shared-weight, width-elastic family. It does not require AirSim and should be run on the GPU server used for DNN training.

Run commands from the repository root so Python can import `elastic_supernet_test_a`.

## Paths used by the first experiment

```bash
MODEL=experiments/paper_track_p_12d_5ms_v1/runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth22_width512/best_model.pt
NORM=experiments/paper_track_p_12d_5ms_v1/dnn_dataset/normalization.json
TEST=experiments/paper_track_p_12d_5ms_v1/dnn_dataset/test.npz
TRAIN=experiments/paper_track_p_12d_5ms_v1/dnn_dataset/train.npz
VAL=experiments/paper_track_p_12d_5ms_v1/dnn_dataset/validation.npz
```

## Smoke test

```bash
python -m elastic_supernet_test_a.evaluation.evaluate_baseline \
  --model "$MODEL" --normalization "$NORM" --dataset "$TEST" \
  --device cuda --max-samples 2048 \
  --output elastic_supernet_test_a/results/baseline_fixed_model_smoke.json

python -m elastic_supernet_test_a.evaluation.evaluate_subnets \
  --model "$MODEL" --normalization "$NORM" --dataset "$TEST" \
  --device cuda --max-samples 2048 \
  --widths 64,128,256,384,512
```

## Complete pre-fine-tuning evaluation

```bash
python -m elastic_supernet_test_a.evaluation.evaluate_baseline \
  --model "$MODEL" --normalization "$NORM" --dataset "$TEST" --device cuda

python -m elastic_supernet_test_a.evaluation.evaluate_subnets \
  --model "$MODEL" --normalization "$NORM" --dataset "$TEST" --device cuda
```

## Fine-tune the shared supernet

```bash
python -m elastic_supernet_test_a.training.finetune_supernet \
  --model "$MODEL" --normalization "$NORM" \
  --train "$TRAIN" --validation "$VAL" \
  --device cuda --epochs 100 --batch-size 2048 --patience 15
```

The fixed controller remains frozen as an immutable teacher. Candidate subnets
learn from both MPC labels and fixed-controller distillation. Checkpoint
selection includes an explicit maximum-subnet retention penalty, so a gain in
small-subnet quality cannot silently destroy the original full model.

## Complete post-fine-tuning evaluation

```bash
python -m elastic_supernet_test_a.evaluation.evaluate_subnets \
  --model "$MODEL" --normalization "$NORM" --dataset "$TEST" --device cuda \
  --elastic-checkpoint elastic_supernet_test_a/checkpoints/elastic_supernet_best.pt
```

The generated CSV files use MAC count as the compute axis. GPU timing is intentionally excluded from the primary Test A claim.

## Supplementary resident-model switching benchmark

This benchmark compares a preloaded fixed-model pool against runtime elastic
configuration changes. It reports static and switched CUDA/wall-clock latency
distributions. It is useful for detecting a large software switching penalty,
but it must not be presented as FPGA latency evidence.

The supplied manifest currently contains the three completed fixed models. Add
the other three entries after their training finishes.

```bash
python -m elastic_supernet_test_a.evaluation.benchmark_switching \
  --fixed-pool-manifest elastic_supernet_test_a/configs/fixed_pool_current3.json \
  --elastic-model "$MODEL" --elastic-normalization "$NORM" \
  --dataset "$TEST" --device cuda --iterations 1000 \
  --output elastic_supernet_test_a/results/gpu_switching_current3.json
```

After supernet fine-tuning, add:

```bash
--elastic-checkpoint elastic_supernet_test_a/checkpoints/elastic_supernet_best.pt
```
