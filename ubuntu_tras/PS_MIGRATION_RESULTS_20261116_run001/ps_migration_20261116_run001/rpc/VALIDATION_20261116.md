# Native RPC / scene-preparation validation

Scope: new experimental `rpc/` only. The original handoff, model weights,
DNNDK runner, running board, and real AirSim instance were not modified by these
tests. Loopback mocks validate protocol and preparation logic, not flight
performance, safety, DPU timing, or successful live flight.

## Executed checks

From `/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001`:

```bash
bash rpc/build.sh host
python3 record_command.py --name rpc_protocol_mock_20261116_001 -- rpc/build_host/test_rpc
python3 record_command.py --name rpc_prepare_mock_20261116_001 -- python3 rpc/test_prepare.py
bash rpc/build.sh arm
```

The two mock test commands require permission to bind their **127.0.0.1-only**
test ports. They do not contact Windows or the board.

Both host tests passed. Exact commands, UTC timestamps, exit codes and stdout
are in `../logs/rpc_protocol_mock_20261116_001.{json,log}` and
`../logs/rpc_prepare_mock_20261116_001.{json,log}`. Detailed preparation fixtures
and results are retained in `prepare_tests_ivjf2y23/`.

The RPC protocol mock checks state and collision decoding, NED/quaternion order,
management argument order, physical angular rates passed **without hidden sign
changes**, explicit invalid-command rejection, six-argument object spawning,
and refusal to delete foreign objects. A delayed command plus immediate ping
confirms that command-return proxy timestamps are not physical completion.

The preparation test checks exact single/dual translated geometry, polynomial
reference position/velocity at 1.5 s, lane offsets, gate bar geometry, paused
initialization and readback, calibrated hover priming, cleanup of an existing
owned actor, refusal of foreign actors before any reset/deletion, and owned
prefix guards. Offline preparation must not emit `prepared_case.json`.

## Corrections made during this review

- Added the missing host mock-server build target; it is not a board runtime.
- Replaced a hard-coded prefix comparison length with the actual owned-prefix
  length and restricted prefix characters so regex/path metacharacters cannot
  broaden the actor-cleanup query.
- Marked deletion of an already-owned actor as simulator mutation, including
  when later initialization fails. Partial mutation records are preserved.
- Native preparation explicitly uses `native_rpc_command(prepared)` rather than
  Python-facing pitch/yaw fields. Hover has zero rates, but the explicit call
  avoids carrying the wrong convention into future nonzero-rate setup.

No model calibration, weights, normalization, feature ordering or raw-input
range checks were changed. RAW237 normalization still occurs once in the
frozen local DNNDK inference path, not in RPC.

## Completed ARM build

The PetaLinux2020.1 SDK build completed with exit code 0; log:
`logs/build_arm_20261116T002445_2.log`. The products are AArch64 ELF binaries,
not host executables. SHA256 at the completion of this review:

```text
1ceab8db31fadc05aac48af0791da978ea75f66863398e6b2c51fd53c8ce36d9  build_arm/prepare_case
0c5d4d6e99b27cb8ce4f538e1c3bb6dccac8686f7bdcac211ba95d6c727b6f9a  build_arm/airsim_probe
043e0cec001afa7e08b9ece7dc1961c1241e5997b436d3b0a4d7726b026c4512  build_arm/test_rpc
```

Compilation is not execution evidence. Deployment and actual board/flight
results are owned by the parent experiment and recorded separately.

## Limits and live-use gates

Do not infer a live `PREPARED_PAUSED` result from mock success. The parent
experiment must deploy the ARM binary, record its hash, validate the actual
Windows address and server, then run native preparation against the real
simulator in a new output directory. A failed setup writes `setup.json`, not a
successful prepared artifact, and attempts to leave physics paused after any
mutation. Foreign actors are never removed automatically.

The default owned prefix is intentionally reused between controlled repeats so
previous owned gate actors can be removed. Custom prefixes are explicit owner
identities, not a request to clean up other prefixes.
