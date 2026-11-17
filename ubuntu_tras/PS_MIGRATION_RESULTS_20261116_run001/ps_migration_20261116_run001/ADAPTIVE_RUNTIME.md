# Latest Route-B obligation-calendar PS runtime

`adaptive_runtime.cpp` is the live application adapter for the already verified
`scheduler/LiveCalendar` port. It does not invoke the old ABC entry point. The
bounded ABC heuristic internal to `obligation_scheduler.py` remains an internal
search step of that newer algorithm.

The frozen feature/reference implementation, native RPC transport, and
`pool237_runner.cc`-backed local DNNDK wrapper are reused. The original Windows
`native_io.cpp`'s platform ownership is mapped onto Linux: one independent
absolute-release publisher, a separate native RPC client/worker per vehicle,
the same command-submission-plus-ping return proxy, explicit busy ownership,
and completion events. Windows QPC/waitable timers are not used on PS.

## Inputs and identity

The executable requires a freshly prepared, paused scene and a measured profile:

```bash
./ps_obligation_control \
  --prepared /home/root/ps_migration_20261116_run001/prepared_dual_001/prepared_case.json \
  --config /home/root/ps_migration_20261116_run001/inputs/config_dual_001.json \
  --policy fixed \
  --duration-s 20 \
  --out /home/root/ps_migration_20261116_run001/fixed_dual_001
```

Use a NEW prepared/reset scene and NEW output directory for every run. For the
matched adaptive run change only `--policy adaptive` and those two paths. The
same frozen configuration must be used for both policies. The parent workflow
uses the exclusive-DPU guard before invoking this executable; do not run the
old UDP service and the local experimental DNNDK executor simultaneously.

Default candidate IDs are the original `[5,10,13,8,11]`; initial model is 11;
initial period is 50 ms; supported periods are `[20,30,40,50,60,80]` ms. Explicit
configuration values take precedence. Initial names are taken from the prepared
scene, and origin from actual PS monotonic time. The profile is made by
`prepare_live_config.py` from measured local DNNDK, successful fixed trials,
and the actual ARM scheduler benchmark—not a historical Windows UDP addition.

At startup the program compares actual compiled DNNDK wire and normalization
identity with the configuration and registry. It checks the freeze identity and
unchanged TRAIN512 scores; final480 check MSE is descriptive, never a selection
objective. Each candidate's own 32 golden inputs must pass, then each candidate
gets 100 warmup requests. Those operations are outside released timing. All 18
unchanged model ELFs remain linked/resident; the actual dispatched ID is logged.

## Actual request lifecycle

1. While paused, verify the prepared position/velocity/orientation, read the
   collision baseline, and prime calibrated physical hover. After unpausing,
   do not relabel the earlier paused state as a new live observation. Planning
   waits for the first real sampled state/held-control pair.
2. The publisher alone calls `release_due(now,end)`, records every returned row,
   and submits only `sampling` rows to the associated native worker. Busy/late
   releases and planned skips remain explicit records. It consults the calendar
   again at most 250 microseconds later, so a retimed earlier release cannot be
   missed by sleeping to an obsolete fixed schedule. This is ordinary Linux
   scheduling, not a certified 250-us wake latency.
3. The worker performs native state/collision reads, reference generation,
   complete finite/flight guards, held-control analysis, and RAW237 encoding.
   Pure numerical feature/hold work is serialized by the shared PS mutex.
   Full frontend wall cost including native RPC is conservatively charged to
   the empirical preparation envelope. No additional normalization is applied.
4. Publish the observation using the exact held-control version analyzed. Put
   the RAW payload/future in the executor registry before `queued()` can expose
   its calendar row. Queue deadline is the minimum of period, hold, freshness,
   and an explicit recovery obligation.
5. The sole DPU executor calls `take()`. A rejected dispatch resolves its future
   without inference. A valid dispatch's returned `configuration_id` is passed
   to real `LocalDpu::infer`; it is not replaced by the worker's initial ID.
   Once real inference returns, call `dpu_done()` before completing the future.
   An active DPU call is never cancelled or preempted by a software timeout.
6. Read the authoritative calendar row again after inference: a later periodic
   release may have cancelled it while DPU was active. Reject range/status
   failures and expired output; retain the existing input checks, with no
   clipping, teacher solve, CPU DNN, or automatic model replacement.
7. Before native command submission, invalidate the old held certificate through
   the command-pending lifecycle guard. Native RPC gets physical `wx,wy,wz`
   directly and calibrated throttle. Do not copy the Python-layer pitch/yaw
   pre-flip into native RPC.
8. Every real successful command+ping return is sent to `command_return()`, even
   if late. Update effective held control/version, sample a NEW state, repeat
   flight/hold analysis, publish that new version's observation, then `finish()`.
   An exception after possible command submission is uncertain delivery and
   retires that vehicle; the record is not falsely marked confirmed-unsent.
9. Finish clears busy ownership, not the next periodic release. Peer vehicles
   may continue after a reported retirement; such a trial is not a clean dual
   acceptance run merely because the process exits normally.

## Planner and skip policy

A separate planner worker performs snapshot → bounded search → atomic install,
then waits 10 ms after the call, matching the real `plan_live` integration. It
logs snapshot, decision, accepted/rejected install, wall time and thread CPU
time. Search uses the core's 8 checks/12 candidates bounds. A call can exceed
10 ms; no timing guarantee is inferred from its service interval. A stale,
elapsed-budget, control-version, or changed-release install is rejected intact.

`--policy fixed` does not start an adaptive planner and never installs dynamic
choices. It uses the same native IO, feature, budget and real DNNDK chain.

Skipping is OFF unless `--enable-skip` is explicitly supplied with adaptive
policy, irrespective of an accidental enabled field in an input JSON. Stage 3
should only run after the fixed/adaptive baseline is accepted. Qualification is
the original development test (`error<0.15m`, first-six reference-velocity-vector
change `<0.3m/s`) AND a fresh same-held-version hold certificate. It is not a
plant-certified safety bound. A skipped release is consumed from the existing
calendar; its existing immediate successor carries recovery debt. Debt clears
only after an actual on-time acknowledged recovery, not on a synthetic event.

## Raw outputs and units

- `requests.jsonl`: every actually published calendar row, full RAW237 and
  physical output for executed jobs, independent status/collision/error fields,
  plus the lifecycle copies at release/queue/dispatch/completion.
- `requests.csv`: scalar timestamps and identities for analysis; JSONL retains
  vector data and scheduler details.
- `events.jsonl`: incrementally flushed completed raw rows. A separate writer
  formats data; bounded queue failure or write failure is visible and invalidates
  the run. Nothing is silently discarded. On fatal interruption this file may
  contain only completed events; it must not be mistaken for a complete window.
- `plans.jsonl`: every planner call, including context-not-ready/rejected calls.
- `actual_dpu_identity.json`, `golden.json`, `execution_identity.json`,
  `config_source.json`, `prepared_source.json`, `initial_states.json`,
  `final_states.json`, `calendar_final.json`, `summary.json` and, on exceptions,
  `failure.json` preserve provenance and acceptance context.

All raw `*_us` values are absolute **PS CLOCK_MONOTONIC microseconds**. Calendar
fields ending `_ms` are the same epoch in milliseconds. RPC native `*_ns` are
converted by division by 1000. AirSim `sim_timestamp_ns` is a different clock and
must never be subtracted from PS timestamps. Hold-analysis seconds are converted
to milliseconds at the calendar boundary. `DPU submit → result-readable` is DPU
**API wall time**, not independently measured pure hardware execution. Native
command+ping return is a submission-completion proxy, not physical actuation.

Relevant intervals include release→feature-ready, queue→dispatch, DNNDK input
preparation, DPU API call, output conversion, output-ready→command-return,
command-return→paired-refresh/finish, and complete release→command-return or
release→slot-return. Discarded rows without those stages must be excluded from
phase-duration statistics, not counted as zero-time inferences.

## Rebuild and offline tests

```bash
cd /home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001
bash build_adaptive.sh "$PWD/build_adaptive_NEW"
```

`build_adaptive.sh` refuses an existing build directory, snapshots its new source,
builds ARM objects using the existing SDK, and links the frozen 18 ELF files and
existing native RPC/DNNDK libraries. It produces `ps_obligation_control`,
`test_adaptive_live`, and SHA256 records. Source scheduler regressions and the
feature/native-command fixtures remain separately required.

`test_adaptive_live.cpp` is deliberately a CPU-only injected-inference plumbing
test. It covers queue visibility, actual chosen ID reaching the callback,
dispatch rejection without inference, future completion after `dpu_done`,
nonpreemptive cancellation, late ack/version invalidation, exception propagation,
post-failure enqueue rejection, and journal draining/failure visibility. Passing
it does **not** count as a CPU replacement for real DPU or AirSim acceptance.
