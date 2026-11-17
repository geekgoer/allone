# Obligation-calendar live integration audit

This audit concerns the latest Route-B obligation scheduler, not the previous
ABC runner. The bounded ABC search is an internal heuristic of the obligation
algorithm; invoking an old ABC runtime is not equivalent to this integration.

## Sources checked

- Unified `source/ps_core/runtime.py`: `ObligationRuntime`, including actual
  future-mode installation, reorder, period editing, skip permissions, and
  recovery debt.
- Unified `source/ps_core/lifecycle.py`: independent releases, ownership,
  observations, held-control versions, and one nonpreemptive DPU job.
- Original frozen `project/controller_architecture_study/fly_ip37_joint.py`:
  `Pilot.publish`, `Pilot.run_job`, `execute_board`, `plan_live`, and the main
  release loop. The unified package preserves the relevant numerical/runtime
  core; the original runner supplies the real integration call sequence.
- Unified runtime tests, not training, teacher solving, CPU inference, or old
  model experiments.

All C++ calendar timestamps are **absolute CLOCK_MONOTONIC milliseconds**.
Duration fields are milliseconds. Do not mix seconds, AirSim simulation
timestamps, wall-clock UTC, or Windows QPC epochs with these values.

## Required runtime sequence

1. While AirSim is paused, load the unchanged frozen DNNDK models, validate the
   chosen model's own golden examples, warm up, prepare exact reference/layout,
   prime calibrated physical hover, and establish an initial state/held-control
   certificate. Priming is outside the released measurement window. The initial
   known held control is version 0; no inference completion is fabricated for it.
2. Construct `LiveCalendar` with actual vehicle names, common origin, initial
   periods, supported periods, default model ID, candidate IDs, and measured local
   DNNDK mode costs. Preserve model identity and normalization provenance.
3. One independent publisher calls `release_due(now_ms, end_ms)`. Record **every**
   returned row. Only `status == "sampling"` is handed to the corresponding
   vehicle worker. The function advances from each planned release, not from
   completion. Late releases, busy-worker drops, and planned skips stay in the
   raw record. Sleep to `next_wakeup_ms()`; retiming may change that value.
4. A vehicle worker samples AirSim state/collision, generates the same reference,
   computes the held-control safety analysis, and builds **RAW237**. Under a
   short shared-PS accounting/serialization convention, record the actual sample,
   reference, analysis, and feature-ready timestamps. Publish
   `observe(vehicle, sample_ms, hold_deadline_ms, held_version, stable, evidence)`
   only with the same held-control version used by the analysis.
5. `queued(vehicle, cycle, sample_ms, feature_ready_ms, hold_deadline_ms)` freezes
   the actuation deadline as the minimum of slot deadline, hold deadline,
   sample+freshness, and an explicit recovery deadline. On a cancelled return,
   record/drop/finish without submitting inference. RAW237 goes through the
   frozen Pool input normalization exactly once; do not normalize in features.
6. The **sole DPU executor** calls `take(now_ms)`. It may get:

   - `null`: idle, active DPU, or waiting for a promised feasible order head;
   - `dispatch == false`: an expired/rejected row. Resolve its worker's future
     as cancelled and keep draining; **do not call DPU**;
   - `dispatch == true`: use this returned `model_id`/`configuration_id`, not the
     worker's original default. This is where an installed model choice reaches
     real DNNDK inference.

7. Invoke the actual local DNNDK pool. Its return timestamp, including normal
   wrapper result availability, becomes `dpu_done(..., result_ready_ms, success)`.
   Fulfill the waiting future only after calling `dpu_done`. Cancellation or an
   expired deadline does **not** preempt active DPU hardware; never call `finish`
   on an active DPU row before the real call returns.
8. The vehicle worker checks status, cancellation, range checks, and remaining
   command-reserve budget. Send no command for a nonzero DNNDK status. Keep input
   out-of-range rejection; no clipping, fallback teacher, or CPU model is allowed.
9. Immediately before RPC, call
   `command_begin(vehicle, cycle, actual_begin_ms)`. A false return means the
   row is cancelled or has insufficient frozen-deadline/reserve budget: do not
   send, record `dropped_before_command`, and finish normally. A true return
   invalidates the old observation, marks native `command_pending`, and changes
   the revision under the same calendar lock, without yet advancing the known
   held-control version. It is the Linux calendar counterpart of
   `native_io.cpp`'s guarded-batch `command_pending` check: an old-held snapshot
   cannot commit while a new command may already be reaching AirSim.

   Send native AirSim RPC using **physical wx, wy, wz directly** and calibrated
   throttle. Python-layer pitch/yaw signs are not native RPC signs. After the
   existing command+ping submission-completion proxy returns, call
   `command_return` **even if late**. It increments the held-control version and
   invalidates the previous certificate. Record late return rather than claiming
   it was never sent. An RPC exception after submission is uncertain delivery,
   not a confirmed unsent request; retire the affected vehicle and preserve it.
   Pending state persists without a real acknowledgement; `finish()` cannot
   clear it or re-enable old-held observations. Even if a caller accidentally
   forgets retirement, later slots remain `dropped_uncertain_command` rather
   than running against an unknown held control.
10. Update the stored held control using the actual clipped/calibrated command.
    Pair a **new** AirSim state with that new held control and publish `observe`
    with the new version. Alternatively, explicitly defer refresh to the next
    planned sample; the planner must wait while the certificate is absent.
    Call `finish` only after postprocessing/refresh finishes. Finishing clears
    busy ownership but does not move the next periodic release.

Rows returned by the C++ adapter are JSON **copies**, not mutable shared runtime
objects. Updating a log row does not update the calendar; use the lifecycle
methods. Keep futures/payloads indexed by `(vehicle, cycle)` in the application.

## Planner and release ownership

Run the planner separately at the documented bounded interval (the reference
runner uses 10 ms). `snapshot` captures the complete revision and control
versions under a lock; `plan_snapshot` runs bounded search outside the lock;
`install` reacquires the lock **before** sampling commit time and performs a
revision/elapsed-time guarded atomic install. Preserve rejected plans in logs.

The publisher must consult the calendar's updated `next_ms` and `period_ms`.
Keeping a separate fixed `for(k)` release sequence while claiming adaptive
period/skip operation would not exercise the migrated policy. A fixed stage-1
runner remains a useful separate baseline, but is not a complete stage-2 result.

`adaptive=false` is the fixed baseline. The C++ adapter defensively refuses
installing an adaptive decision in that mode, even if an application accidentally
calls the planner. Skip must remain `enable_skip=false` until the separate
stage-3 validation.

## Skip and mandatory recovery

The development stable-phase qualifier from the real runner is
`tracking_error < 0.15 m` and maximum reference-speed-vector change over its first
six reference points `< 0.3 m/s`. It is **not itself a safety proof**. A fresh,
same-control-version hold certificate must separately cover the entire deferred
interval. Unknown/stale/unstable phase, an active worker, or existing recovery
debt disables a new skip.

An installed skip consumes one exact existing periodic release, leaving its
period unchanged. The immediate existing successor, not an extra synthetic
request, carries `recovery_due_ms`. It must pass the normal queue/dispatch/DPU/
command path. Debt is removed only after a real acknowledged on-time command;
missing/late recovery remains visible. The C++ adapter additionally requires
the acknowledgement to meet the already frozen actuation deadline, a
conservative tightening of the Python runtime's hold-debt-only finish check.
It never fabricates successful recovery from a scheduled event.

## Costs and claims

Use locally measured model service costs and measured PS preparation, command
return, and rearm costs. Do not retain the historical Windows UDP `network_ms=3`
addition after switching to local DNNDK. AirSim RPC is still network I/O and must
be charged/logged in its proper preparation/post envelope rather than assumed
free. The current adapter increases observed envelopes monotonically.

Those empirical envelopes are not certified WCETs. A running DPU job that already
exceeds its predicted envelope disables replanning rather than pretending its
unknown remainder is zero. Report empirical misses and all rejected contexts;
a feasible finite forecast is conditional on the supplied costs and hold model.

## Defensive changes and tests in this audit

- Removed duplicate CMake executable/test target declarations.
- Finite timestamp and causal queue/completion guards; NaN cannot bypass budget
  checks.
- A command-return event requires an actually completed successful DPU row and
  a prior successful `command_begin`, and cannot be recorded twice. Late real
  acknowledgements are still accepted.
- Native-command begin atomically invalidates stale held-state planning;
  failed/no-ack command delivery cannot silently restore the old certificate.
- Invalid multi-field cost updates have no partial state effect.
- Fixed-mode adaptive installs are refused.
- Regressions additionally cover native command pending, old snapshot CAS
  rejection during command, blocked stale observation, uncertain cleanup,
  late actual acknowledgement, and the atomic pre-command reserve guard.
- Regressions include explicit skip consumption,
  existing-successor recovery, actual dispatch, and debt clearing on real ack;
  core scheduler regressions cover 43 checks.

These API guards do not change normal valid numerical scheduling decisions.
The CMake binary is absent from the host PATH; the supported Bash build was
used. That missing build tool is not a scheduler-regression failure.

Rebuild on the host:

```bash
cd /home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001
bash scheduler/build_native.sh
scheduler/build/live_calendar_tests
scheduler/build/scheduler_tests
```

Build ARM objects and test programs using the existing SDK (no `set -u` while
sourcing its environment):

```bash
cd /home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux
BUILD_DIR="$PWD/scheduler/build_arm" bash scheduler/build_native.sh
```

The command-pending audit uses fresh output directories:
`scheduler/build_command_guard_host_001` and
`scheduler/build_command_guard_arm_001`. Its exact commands and results are in
`logs/calendar_command_guard_*_build_001.{json,log}`; this preserves the earlier
51-check build as separate evidence.

The runtime must link the current ARM `scheduler.o` **and**
`live_calendar.o` with pthread. Building these objects or
running the regression binaries alone does not claim AirSim/DPU stage-2 flight
completion.
