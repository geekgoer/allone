# Native AirSim RPC binding for Zynq PS

This directory is a new experimental copy. Frozen source/model/deployment trees
are read-only inputs and were not changed. The binding uses the bundled rpclib
2.3.0 and its original MessagePack/ASIO implementation; no Python, Eigen,
TensorFlow, Torch, CUDA, or full AirLib vehicle simulator is linked.

## Interface and wire contract

`airsim_rpc.hpp` is a dependency-light C++14 header. Link `libpsairsim.a`, then
`librpc.a`, and `-pthread`. Create one `psairsim::Client` per owning worker; do
not concurrently share a client without external serialization. The parent
control program decides CPU affinity, queueing, reference, feature encoding,
model selection, control adaptation, deadlines and cancellation policy.

Readonly methods: `ping`, `list_vehicles`, `get_state`, `get_collision`, and
`get_ground_truth_kinematics`. State positions/velocities remain local NED;
orientation is `[w,x,y,z]`; body rates are not converted to world rates.
Simulation timestamps are separate from Linux `CLOCK_MONOTONIC` timestamps.

`move_rates_throttle(vehicle, wx, wy, wz, throttle, duration)` sends native
`moveByAngleRatesThrottle` asynchronously, then makes a synchronous `ping` on
the same connection, as the frozen Windows path does. It does **not** wait for
the entire command duration. The returned `CommandProxy` records:

- `submit_ns`: immediately before enqueueing the asynchronous command.
- `enqueued_ns`: after `async_call` returned, not proof the server executed it.
- `ping_start_ns` / `ping_return_ns`: the synchronous round trip around ping.
- `ping_ok`: successful ping response, **not** an actuation acknowledgement.

An RPC return/ping proxy is not a measured physical motor application time;
no one-way network time can be formed by subtracting PC and PS absolute clocks.
The binding does not clip outputs, change signs, or normalize features. The
frozen control adapter must perform its own documented limits/sign mapping.
Invalid throttle/duration/NaN arguments fail explicitly.

In particular, the Python AirSim client negates pitch/yaw when converting its
arguments to the RPC wire. Native AirLib and this binding do not. The native
control path therefore passes physical `wx, wy, wz` directly, using
`psfeatures::native_rpc_command(prepared)`; do not pass the Python-facing
`prepared.command` pitch/yaw fields directly. RAW237 normalization is unrelated
to this RPC layer and happens once inside the frozen DNNDK runner.

`pause`, `enable_api`, `arm`, `set_pose`, `set_kinematics` and `reset` are
management methods only; never call them in the timed control hot path.
`airsim_probe` requires `--allow-management` for any mutating mode.

Scene management additionally provides `get_object_pose(name)` (world NED,
`Pose{position,orientation_wxyz}`), `list_scene_objects(regex)`, and
`spawn_object(name,asset,pose,scale,physics_enabled=false)`. Match a vehicle's
world actor pose to its simultaneously observed local state before computing
lane offsets. Unknown object poses fail explicitly on NaN rather than inventing
zero offsets. `destroy_object(name,owned_prefix="PSMIG_")` rejects an empty or
nonmatching owned prefix before making RPC; it must never remove unrelated
actors. The parent experiment owns gate layout and tracks every spawned name.

## Rebuild

From Ubuntu (these commands do not contact the board or AirSim):

```bash
RPC=/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001/rpc
bash "$RPC/build.sh" host
"$RPC/build_host/test_rpc"
python3 "$RPC/test_prepare.py"
bash "$RPC/build.sh" arm
```

`PS_BUNDLE` and `PS_SDK` may override the immutable input tree and SDK path.
Every attempt, including failed compilation, gets its own log in `logs/`.
Source changes always retain original failed logs; there is no clean/reset.
The build deliberately does not execute vendor CMake because it configures
headers back into the frozen vendor source directory.

Host output is x86-64 and is **not** board evidence. ARM output is built using
the existing PetaLinux2020.1 SDK and is dynamically linked only to normal SDK
Linux runtime libraries (rpclib and the binding themselves are static).

## Readonly probe

After separately copying `build_arm/airsim_probe` to an experimental board
directory, the board command is:

```sh
./airsim_probe --host ACTUAL_WINDOWS_AIRSIM_IP --port 41451 \
  --vehicle Drone1 --timeout-ms 1000
```

By default this performs only ping/list/state/collision. It prints JSON,
including failure details, and returns nonzero on failure. The parent runbook
must record the actual host/interface, server settings and firewall rule.

Example isolated setup calls (only when requested by experiment management):

```sh
./airsim_probe --host ACTUAL_WINDOWS_AIRSIM_IP --mode pause --allow-management
./airsim_probe --host ACTUAL_WINDOWS_AIRSIM_IP --vehicle Drone1 \
  --mode enable --allow-management
./airsim_probe --host ACTUAL_WINDOWS_AIRSIM_IP --vehicle Drone1 \
  --mode set-kinematics --input initial_state.json --allow-management
./airsim_probe --host ACTUAL_WINDOWS_AIRSIM_IP --mode unpause --allow-management
```

State JSON contains `position[3]`, `velocity[3]`, `orientation_wxyz[4]`, optional
`acceleration[3]` and `angular_velocity[3]`. Position initialization uses the
original reference-matched diagnostic, not a silent replacement for normal
stationary takeoff acceptance. `set-kinematics` requests collision checking.

## Offline test and limits

`test_rpc.cpp` starts a local C++ mock server. It checks exact RPC names/argument
order, map decoding, quaternion/NED order, management calls, unchanged rate
signs and explicit invalid-command rejection. A deliberately delayed command
and immediate ping demonstrate why the proxy must not be called physical
completion. Mock results do not prove AirSim flight or board execution.

Actual PS readonly/flight results belong to the parent experiment directory,
not to a synthesized PASS in this library's logs.

`prepare_case.cpp` is an explicit scene-management program, not a hot-path
sampler. It validates the frozen layout before mutation, refuses foreign
`Falcon.*Gate.*` actors, only removes its own safe `PSMIG20261116_` prefix,
initializes translated reference-matched states while paused, creates the gate
bars, primes calibrated physical hover, and verifies the paused readback. A
successful run writes `prepared_case.json` with status `PREPARED_PAUSED`; a
failure only writes `setup.json`, retains partial mutation/actor records, and
attempts to pause the simulator. `--offline --offsets FILE` only generates
geometry and never writes a misleading prepared-live artifact.

`test_prepare.py` checks single/dual geometry against the frozen polynomial
case, translated offsets, initial reference and gate bar positions; its native
mock also checks RPC initialization, owned-actor cleanup, foreign-scene refusal
before any reset/deletion, and safe-prefix validation. Mock success is neither
flight success nor proof of real-time safety.

## Build failures retained

The first build exposed rpclib's renamed dependency namespaces. Passing its
own documented `ASIO_STANDALONE`, `RPCLIB_ASIO=clmdep_asio`,
`RPCLIB_FMT=clmdep_fmt`, and `RPCLIB_MSGPACK=clmdep_msgpack` fixed this without
editing vendor files. The next probe build exposed the old vendored JSON
header's lack of `get<std::array>` support; strict length-checked elementwise
decoding replaced that unsupported convenience call. Original host and ARM
failure logs remain in `logs/`.

The first host protocol mock could not create a loopback listening socket in
the sandbox (`open: Operation not permitted`). A subsequent explicitly allowed
loopback-only run passed. This infrastructure failure is retained separately
and is not reported as an AirSim or RPC wire failure.
