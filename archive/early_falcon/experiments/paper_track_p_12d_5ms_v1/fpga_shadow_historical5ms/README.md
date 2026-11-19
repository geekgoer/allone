# Historical 5 m/s Track-P: MPC-actuated FPGA shadow test

This directory implements Phase 1 from the deployment handoff.  It uses the
historical six-model FLCN/v2 service on UDP port **5007**.  It does not use the
older services on ports 5005 or 5006.

For each 50-ms AirSim control cycle, the script builds the frozen 12-D
gate-relative state once, calculates a command with the project's fast linear
MPC teacher, sends the normalized state to all six FPGA model IDs serially,
and applies **only the MPC command**.  FPGA replies are measurement data only.

## Preconditions

On AXU3EGB, keep the new historical service running:

```sh
cd /home/root/board_package_historical5ms_12d_six_v1
./falcon12_historical5ms_dnndk_udp_server --port 5007
```

On Windows, start Blocks manually, click Play, and verify RPC port 41451 is
listening.  Do not start a second UE4 process with `--auto-launch`.

## Pilot

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$py = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $py .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\run_fpga_shadow.py `
  --scenario-set pilot `
  --output-dir .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\results\pilot
```

This runs frozen scenarios `S01` and `S06`, keeps the teacher MPC in control,
and creates:

```text
shadow_manifest.json
shadow_cycles.jsonl
shadow_summary.json
```

Only after a clean pilot should `--scenario-set full` be used.  A nonzero
FPGA-vs-teacher error is expected: it compares the learned INT8 DNN to MPC,
not FPGA output to an INT8 golden vector.

## FPGA-actuated pilot

After the shadow pilot passes, `run_fpga_closed_loop.py` performs the next
phase. It queries one selected FPGA model per 50-ms cycle; only a fresh reply
arriving before that cycle's deadline is applied. A late or missing reply uses
the same-cycle linear MPC teacher command. This is intentionally explicit and
requires `--enable-fpga-actuation`.

Run all six models on the two pilot scenarios, three repeats each:

```powershell
& $py .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\run_fpga_closed_loop.py `
  --enable-fpga-actuation `
  --scenario-set pilot `
  --repeats 3 `
  --output-dir .\experiments\paper_track_p_12d_5ms_v1\fpga_shadow_historical5ms\results\closed_loop_pilot
```

The resulting `fpga_closed_loop_trials.jsonl` includes the declared fallback
source and host-side state-acquisition-to-command-dispatch latency. AirSim
does not expose a precise physics-application timestamp, so that limitation is
recorded in `closed_loop_manifest.json` rather than overstating the result.
