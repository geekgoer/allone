# Analyze native Route-B obligation-calendar trials

This is a separate lifecycle-aware extension of `analyze_flight.py`. It imports
that analyzer only for original request timing, numerical tracking and gate
observations. It does not execute a scheduler, DNN, AirSim client, CPU DPU
simulation, or teacher. The fixed-period analysis and previous results remain
unchanged.

```bash
python3 -B /home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001/features/analyze_obligation.py \
  --data /absolute/trial1 /absolute/trial2 /absolute/trial3 \
  --output /absolute/NEW_analysis_directory \
  --label dual_fixed_vs_obligation_skip_disabled
```

Read inputs: `execution_identity.json`, request journal/final export,
`calendar_final.json`, `plans.jsonl`, prepared scene, optional boundary states,
runtime summary and failure records. No raw file is changed; output directories
must be new. Exit 0 means raw audit passed, not flight or strategy acceptance.

## Release coverage differs from the fixed-period analysis

An adaptive trial can change both periods and phase; the number of releases is
**not** inferred from `ceil(duration/initial_period)`. The analyzer verifies each
record's `calendar_release`, contiguous per-vehicle cycles, increasing planned
times, the released period and `deadline=release+period`, cutoff admission, and
`next_release=previous_release+current_period` following actual calendar edits.
The final loop's cycle must equal the raw release count. Its `next_ms` must have
reached the measurement cutoff for full nonretired coverage. Retired loops have
an explicit coverage exemption, **not** a successful complete trial. Final busy
ownership and uncertain native command-pending state are checked separately.

Actual inference model IDs are checked against the candidate list and the
calendar dispatch's selected configuration; release-default IDs are a separate
histogram because a queued plan may change the actual dispatch choice. Actual
nonpreemptive DPU intervals may not overlap. Per-vehicle held versions advance
exactly once after actual guarded command acknowledgement. A planner decision
never constitutes acknowledgement or a known held-control change.

`plans.jsonl` is parsed as an event stream, not deduplicated by vehicle/cycle.
Installed versus rejected plans, actions, mode edits, release edits, wall/CPU
decision costs, and the bounds of 8 checks/12 candidates are reported. An adaptive
label with no actual mode/period/skip effect is explicitly not evidence of an
adaptation advantage. Installed future edits are reported separately from
actually inferred non-default models, released changed periods and consumed
skips. Skip/recovery experiment validation additionally requires that at least
one skip actually occurred; merely enabling the option cannot pass that test.

## Planned skip and mandatory successor recovery

A `planned_skip` is an intentional release lifecycle outcome, not a packet drop.
It must have explicit opt-in and a successfully installed permission; its row
must not submit DPU work or a control command. Its recovery is the immediate
same-vehicle **existing successor cycle**, not a synthetic extra request. The
successor's recovery deadline must match the committed held-control deadline.
Debt clears only after a real command acknowledgement before the minimum of the
recovery and actuation deadlines; the raw `recovery_on_time` flag is cross-checked.
Missing successors and commitments at the end of the finite measurement window
remain outstanding/not proved, never silently counted as recovered.

Calendar/timing acceptance allows `delivered` and valid `planned_skip` outcomes,
not necessarily all-delivered rows. It additionally requires nonretired cutoff
coverage, drained workers/known command state, no runtime/journal/planner failure,
no late commands, and no outstanding skip/recovery obligation. Flight acceptance
still separately requires actual tracking/gate/collision/clock evidence; neither
timing nor an installed plan proves a good flight.

Outputs: `analysis.json`, `REPORT.md`, per-request phase timing CSV, tracking CSV,
command lifecycle/held-version CSV, skip-recovery CSV, installed release-update
CSV, model-edit CSV, run-summary CSV and vehicle-summary CSV. Each run remains individually identified with file
hashes. Empirical timing envelopes are not WCET or physical actuator guarantees;
the DPU interval remains call wall-clock rather than pure hardware execution.

Synthetic tests only:

```bash
cd /home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001/features
python3 -B -m unittest -v test_analyze_obligation
```
