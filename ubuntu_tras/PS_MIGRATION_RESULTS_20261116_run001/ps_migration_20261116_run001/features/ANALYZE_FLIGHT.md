# Native PS flight analysis

`analyze_flight.py` is a Python 3.6+ standard-library-only, read-only analyzer. It
does not import training/teacher code, run a CPU DNN, or connect to AirSim/DPU.
It never modifies the raw run or overwrites an existing analysis directory.

Example after copying the complete board output directory to Ubuntu:

```bash
python3 /home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001/features/analyze_flight.py \
  --data /absolute/path/to/run001 /absolute/path/to/run002 /absolute/path/to/run003 \
  --output /absolute/path/to/NEW_analysis_directory
```

Outputs are `analysis.json`, `REPORT.md`, `timing_per_request.csv`,
`tracking_samples.csv` and `gate_crossings.csv`. Per-run identities and SHA256 checksums of analyzed files
are embedded in the JSON. Runs remain separate: early failures are not pooled
with successful flights to improve an aggregate score.

The analyzer prefers final `requests.jsonl`; otherwise it reports the continuous
`events.jsonl` as a partial journal. If both exist their parsed row sets must
match. Exception: when `failure.json` exists and the final export is an identical
strict subset of the journal (including an empty export), analysis uses the
actual journal, sets `partial_final_export=true`, and retains both file hashes,
row counts and an export-mismatch warning. This never turns the failed run into
formal completion. A divergence without a recorded runtime failure, or exported
rows that disagree with journal content, remains an audit error. Optional
`--label diagnostic_first_trial_not_formal` can classify a diagnostic run without
changing acceptance. Duplicate vehicle/cycle rows, malformed lines and negative phase intervals
are audit errors. It preserves runtime `failure.json` and `summary.json` in its
report; parsing success is explicitly not flight success. Exit code 0 means the
log audit passed, not that flight or timing passed; 2 means an audit error.

## Timing

Every interval subtracts two PS monotonic microsecond timestamps and is reported
in milliseconds, with count, minimum, mean, P50/P95/P99 and maximum observed.
Percentiles use linear interpolation at `(n-1)*p`, matching NumPy's default.
The report separates publication lateness, state/collision RPC, PS-lock waiting,
reference generation, safety budget, feature construction, DPU queue, original
runner input preparation/quantization, DNNDK DPU call, output processing, command
conversion/RPC proxy, paired-refresh phases, and request end-to-end times.

Missing endpoints do not become zero-duration samples. Failed requests can still
have measured earlier phases, so sample counts differ by phase. DPU call time is
host wall-clock around `dpuRunTask`, **not pure hardware time**. Pure hardware
time remains null. Command return is a native RPC `ping` proxy, **not the time the
aircraft physically actuated the command**. AirSim simulation timestamps are never
subtracted from PS timestamps.

Statuses are counted independently from error annotations. For example,
`status=delivered,error=collision_after_command` is counted as a delivered-status
row and as a collision, not a clean success. A request whose command may have been
sent but lacks a confirmed return is counted as delivery unknown. `status=3`
input-range rejections, drops, late commands and retired-vehicle slots are shown.

## Tracking and gates

Reference position/velocity are taken from a recorded reference if available.
Otherwise the exact frozen polynomial is evaluated at
`reference_start_s + (sample_start_us-origin_us)/1e6`, using the same segment
selection and clamping as C++ `Trajectory::evaluate`. The source of the reference
is recorded. Sample error RMSE is descriptive even for a failed short run. A
full-window time-integral MSE/RMSE is null unless telemetry brackets `[0,T]` with a
strictly increasing sample clock; the analyzer never extends a last observation
at 19.95 s to claim coverage through 20 s. Actual `initial_states.json` and
`final_states.json` can supply boundary observations: the initial paused state is
aligned to origin; the final paused readback uses `observed_us` as an explicit
time proxy. These are labeled in every sample and report, not presented as exact
physics sampling instants. The integral is an observational piecewise-linear
estimate with boundary-time uncertainty, not ground-truth continuous tracking
cost. Sim/wall ratio is separately reported. Final-readback errors or new
collisions block the observed flight flag even when all slot rows say delivered.

Gate intersections use normalized wxyz quaternion inverse rotation and the exact
original `gate_route.segment_gate_crossing` forward rule:
`local_previous.x <= 0 < local_current.x`. The intersection is linearly
interpolated between actual logged positions. The first forward crossing per gate
matches the original reporting convention; all observed forward crossings are
also retained. Two distinct margins are reported:

- Point-clearance margin: `min(W/2-thickness/2-.25-|lateral|,
  H/2-thickness/2-.25-|vertical|)`, matching the latest joint-flight point gate test.
- Body margin: same expression with `.525` (body radius `.5` + clearance `.025`),
  matching its additional body-clearance report. Physical post-center W/H and
  thickness come from prepared scene metadata (defaults 4 m, 4 m, .15 m).

The intersection calculation is exact **for the sampled straight-line segment**,
not for the unobserved physics path. Crossing intervals over 100 ms are flagged
as insufficiently sampled. Missing crossings are not treated as impossible
crossings or proof of a collision. The observed flight flag additionally requires
full tracking-window coverage, all body-clear gate crossings with acceptable
sample gaps, a 0.95–1.05 observed sim/wall ratio, and no recorded collision/error or
top-level failure. It is not a safety certificate.

## Test

```bash
cd /home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001/features
python3 -B -m unittest -v test_analyze_flight
```

Tests are synthetic only: quantiles, quaternion frames, forward crossing and
body margins, polynomial reconstruction, partial failures, phase-clock errors,
unknown command delivery, malformed/duplicate journal audit, and output
non-overwrite. No actual flight result is invented by these tests.
# Clock-evidence correction (2026-11-16)

The current analyzer uses **only active request observations** to validate the
simulation/PS clock ratio. A paused final state may be read back much later
after logger/calendar export; that delay is not slow physical simulation. When
the raw final row provides causal `pause_return_us`, its recorded upper-bound
proxy is used for the final boundary position/reference alignment, while
`observed_us` is preserved. Both remain proxies, not an exact state acquisition
timestamp. `sim_wall_ratio_with_paused_proxies` is a separate descriptive value
using these corrected boundaries, not the old delayed-readback denominator.
Old analysis directories are retained; `*_002` final summaries use this fix.
The delayed-readback regression is part of the24 analyzer tests.

