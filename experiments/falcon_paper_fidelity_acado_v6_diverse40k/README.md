# FALCON 12-D diverse 40k route (v6)

This route preserves the published-size protocol: 10 initial plant states,
40 environmental conditions, 5 seconds per episode, 50 ms sampling, 40,000
labeled pairs, the 12-D gate-relative FALCON input, ACADO/qpOASES teacher, and
the six Linear+ReLU architectures.

The paper does not publish the random initial-state ranges.  V6 broadens those
ranges relative to v5 while retaining one fixed 5 m/s planner setting.  Its
global C2 trajectory starts from the measured initial velocity and converges
to 5 m/s, avoiding an artificial first-frame velocity discontinuity.  This is
intended to make the capacity comparison informative without adding the 80k
non-paper recovery extension.

Run in this order:

1. `preflight_audit.py` (no AirSim required).
2. `collect_paper_core.py --smoke-test --auto-launch --render`.
3. Full resumable 40k collection.
4. `audit_collected_core.py`.
5. Build the deterministic row-wise 80/10/10 paper split using the v5 generic
   preparation utility and the v6 core/environment files.
6. Train only 10x512 and 34x1024 for a 1000-epoch pilot, with gradient clipping.
7. Summarize the pilot.  Proceed to all six models only if the largest model is
   better on test normalized MSE and no collection-health issue is present.

The local wrappers are `collect_paper_core.py`, `prepare_dataset.py`,
`train_six_models.py`, and `summarize_results.py`.  Pilot training defaults to
1000 epochs; full training defaults to the paper's 20,000 epochs.  Every model
has an independent output directory and resumable `latest_checkpoint.pt`.

For practical full runs, the v6 wrapper now keeps 20,000 as the hard maximum
but requires at least 1,000 epochs and then early-stops after 500 epochs without
a normalized-validation-MSE improvement larger than `1e-5`.  An interrupted
run with a satisfactory best checkpoint can be completed without more epochs
using `finalize_model.py --depth D --width W`.

Because repeated Windows/NVIDIA TDR resets were observed on the laptop RTX
4060, the v6 wrapper uses a conservative batch size of 1024 and disables
DataLoader pinned memory.  Checkpoints remain per-epoch and resumable.

V5 artifacts are immutable comparison baselines and are not overwritten.
