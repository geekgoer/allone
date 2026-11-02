# FALCON elastic MLP Test A report

Status: implementation and CPU smoke test complete; measured full Test A pending GPU execution.

The immutable baseline is the trained 12-D, depth-22, width-512 Track-P controller documented in `model_inventory.md`. Test A uses physical control MSE as its primary offline error and active MAC count as its primary compute proxy. GPU latency is deliberately not used as the primary cost axis.

The software supernet owns one maximum set of weights. A subnet selects prefix slices independently for every hidden layer. The initial controlled search enumerates 13 uniform, tile-aligned widths; it does not yet claim that arbitrary per-layer combinations are Pareto useful.

Required measurements still pending:

1. fixed-baseline reproduction;
2. exact maximum-subnet equivalence;
3. all subnet errors before fine-tuning;
4. sandwich-style shared-weight fine-tuning;
5. all subnet errors after fine-tuning;
6. Pareto and near-Pareto analysis;
7. six-fixed-model overlay after all six FALCON models are available;
8. final Go/No-Go recommendation.

No FPGA conclusion is made before those measurements are complete.

## CPU smoke evidence (not the final Test A)

- Dataset subset: first 2,048 samples of the immutable test split.
- Fixed baseline physical MSE: `8.2783357e-05`; RMSE: `0.00909854`.
- Maximum elastic subnet vs fixed model maximum absolute difference: exactly `0.0`.
- Zero-training width-64/128/256/384 subnet MSE: approximately `3.08` to `3.30`.
- Interpretation: arbitrary prefix slicing of the conventionally trained weights is unusable; supernet-aware training is necessary.
- A one-epoch, 512-sample pipeline smoke test completed. It is not used as quality evidence.
- The fine-tuning selector now saves epoch zero first and explicitly penalizes maximum-subnet drift, ensuring that an unsuccessful fine-tune cannot overwrite the exact fixed-model baseline.

The supplementary GPU benchmark separately measures static and switched
resident-model inference for a fixed pool and an elastic supernet. These GPU
numbers are not treated as FPGA cycle predictions.
