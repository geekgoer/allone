# Current 5 m/s Track-P 12-D: Windows AirSim FPGA shadow-test handoff

## Verified deployment

Target package: `falcon_current_5ms_12d_six_20260920_202227`.

The current six models have passed:

`FP32 -> Caffe FP32 -> Vitis-AI 1.2.1 INT8 -> B1152 ELF -> AXU3EGB DPU -> Ubuntu UDP golden`

All six model-specific 1,000-vector board/UDP checks passed with board-versus-Caffe-INT8 maximum absolute error about `1.94e-6 <= 2e-6`. This proves compiled model identity, physical board inference and Ubuntu transport. It does not yet prove AirSim feature construction or closed-loop behavior.

Use only:

    Windows AirSim: 192.168.77.1
    AXU3EGB:        192.168.77.2
    FLCN/v2 port:   UDP 5008

Never point this current package at old UDP 5005, 5006, or 5007 services.

## Measured isolated board profile

| ID | Model | Mean board DPU+I/O ms | p99 board DPU+I/O ms | Mean Ubuntu RTT ms |
|---:|---|---:|---:|---:|
| 1 | depth10_width512 | 1.035 | 1.041 | 2.429 |
| 2 | depth10_width1024 | 3.119 | 3.123 | 4.468 |
| 3 | depth22_width512 | 2.392 | 2.400 | 3.714 |
| 4 | depth22_width1024 | 7.443 | 7.450 | 8.964 |
| 5 | depth34_width512 | 3.750 | 3.759 | 5.152 |
| 6 | depth34_width1024 | 11.769 | 11.774 | 13.252 |

These are one-request-at-a-time Ubuntu measurements, not Windows multi-model WCRT. Windows must profile actual sweep timing again.

## Immutable AirSim contract

The package name says `5ms`, but its manifest defines `0.05 s`: use a **50-ms / 20-Hz** controller period and nominal speed 5 m/s.

From one consistent AirSim NED state snapshot:

    X[0:3]   = p_gate - p_vehicle
    X[3:6]   = shortest_rotation_vector(q_gate * inverse(q_vehicle))
    X[6:9]   = current linear velocity
    X[9:12]  = current linear acceleration

The orientation term is not Euler-angle subtraction and not quaternion-component subtraction.

Input feature order is gate position error xyz, gate attitude error xyz, velocity xyz, acceleration xyz. Use the current package's `models/<model>/export/normalization.json` and float32 arithmetic:

    X_normalized = (X_physical - x_mean) / x_std

Expected normalization SHA-256:

    0f1d14a3622c8b8119ec751c1bbb8d1c3eb361e76c2a52722b4efea96dd323c7

Model ID mapping:

    1 depth10_width512     2 depth10_width1024
    3 depth22_width512     4 depth22_width1024
    5 depth34_width512     6 depth34_width1024

Reuse the existing FLCN/v2 binary client. Requests contain **normalized** 12-float input and a unique sequence. Replies are already **physical** four-float output:

    [collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]

Do not output-de-normalize again. Reject/log a reply with wrong peer, model ID, or sequence.

## Phase 1 — shadow test only

### Safety invariant

During the entire shadow test, actual AirSim actuation uses only the existing teacher/MPC output. FPGA replies are recorded but never actuated. A timeout, late reply, malformed reply, or wrong sequence is a measurement event only.

### One 50-ms cycle

1. Take one consistent vehicle and active-gate snapshot.
2. Build `X_physical` and `X_normalized` exactly as above.
3. Calculate and retain `U_teacher`.
4. Send the same normalized state to all six current model IDs on `192.168.77.2:5008`.
5. Validate and log every response.
6. Apply **only** `U_teacher` and advance AirSim.

The board has one DPU and serves requests arrival-order/non-preemptively. Serial six-model requests are acceptable for profiling. Measure the full six-request sweep time; do not assume isolated latency measurements imply a 50-ms Windows guarantee.

Run a short smoke episode first, then run the same current AirSim scenarios, initial states and gate configuration used to create this current package. Do not silently replace them with historical-package scenarios.

## Required records

Use append-only JSONL, Parquet, or NPZ with a metadata manifest. Every model reply record must include:

    experiment_id, run_id, scenario_id, seed, environment_epsilon, cycle_index,
    state_timestamp_us, X_physical[12], X_normalized[12], U_teacher[4],
    model_id, model_name, sequence, request_timestamp_us, reply_timestamp_us,
    U_fpga_physical[4], rtt_us, board_dpu_plus_io_us, board_server_total_us,
    fresh_reply, retry_count, timeout, protocol_error, wrong_sequence,
    late_after_50ms, period_ms=50, teacher_applied=true

The manifest must record package name, endpoint and port, model mapping, normalization SHA, feature contract, client revision, AirSim settings, scenario/gate configuration, and fallback policy.

For every model/scenario report reply, timeout, retry, protocol-error and late counts; RTT/DPU/server timing mean/median/p95/p99/max; FPGA-versus-teacher output MAE/RMSE/max; optionally FPGA-versus-local-current-TorchScript error; and the full six-request sweep deadline-miss count.

FPGA-versus-teacher error is expected to be nonzero because FPGA is INT8 and teacher is MPC. The `2e-6` threshold applies only to the frozen board/Caffe golden vectors, not arbitrary live states.

## Stop point and next phase

Return these artifacts after shadow testing:

    current5ms_shadow_manifest.json
    current5ms_shadow_cycles.(jsonl|parquet|npz plus metadata)
    current5ms_shadow_summary.json
    README.md

The README must say that the teacher/MPC, not FPGA, actuated AirSim.

Only after a clean shadow report should FPGA-actuated closed-loop trials begin. Those trials use **one fixed model per episode**, one request per 50-ms cycle, and a declared safe fallback. Do not request all six then choose the output numerically closest to teacher.

## SVM clarification

FALCON uses SVM **regression**, not a per-frame six-class model selector:

    J = C(c, T, R, epsilon, theta)

Its later SVR data comes from FPGA-actuated closed-loop trials: controller quality/complexity proxy `c`, period `T`, end-to-end state-to-actuation response `R`, environment difficulty `epsilon`, and declared closed-loop performance `J`.

Shadow logs are communication/inference evidence, not the SVR training set. With only the frozen 50-ms trials, a later model can assess a conditional slice at `T=50 ms`; it cannot establish full period dependence without separately designed and measured period-sweep trials.

## Reporting boundary

Call this the **current 5 m/s Track-P 12-D six-model FPGA deployment experiment**. Its teacher is `track_p_independent_gate_state_mpc_queries_v1`, not the paper's unreleased PAMPC implementation.

