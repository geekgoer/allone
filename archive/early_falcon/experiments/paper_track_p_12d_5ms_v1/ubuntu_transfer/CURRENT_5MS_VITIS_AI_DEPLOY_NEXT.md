# Current 5 m/s Track-P 12-D: Vitis-AI 1.2.1 deployment

## Scope

This workspace deploys only:

```text
/home/cwc/Downloads/true_final/falcon_current_5ms_12d_six_20260920_202227
```

It is independent from the old 25-D service (UDP 5005), prior 12-D service (UDP 5006), and historical 5 m/s service (UDP 5007). This package uses UDP **5008**. Never reuse any old Caffe files, INT8 calibration, ELF, board golden vectors, reports, or board package.

The name includes `5ms`, but the source manifest says `control_period_seconds=0.05`: it is a **5 m/s, 50-ms / 20-Hz** experiment, not a 5-ms controller.

## Contract

```text
ID 1: depth10_width512     ID 2: depth10_width1024
ID 3: depth22_width512     ID 4: depth22_width1024
ID 5: depth34_width512     ID 6: depth34_width1024

X[0:3]   = p_gate - p_vehicle
X[3:6]   = shortest_rotation_vector(q_gate * inverse(q_vehicle))
X[6:9]   = linear velocity
X[9:12]  = linear acceleration
(frame: AirSim NED)

U = [collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]
```

The network uses normalized input and produces normalized output internally. The FLCN/v2 service receives 12 normalized `float32` values and returns four already de-normalized physical `float32` values. Do not de-normalize a board reply again.

## A. Host validation and calibration

```bash
SOURCE=/home/cwc/Downloads/true_final/falcon_current_5ms_12d_six_20260920_202227
WORK=/home/cwc/dpu_work/falcon_current_5ms_12d_deploy
PYTHON=/home/cwc/dpu_work/.venv_falcon_fp32_cpu/bin/python

chmod +x "$WORK"/current5ms_*.sh
SOURCE_ROOT="$SOURCE" WORK="$WORK" PYTHON="$PYTHON" \
  bash "$WORK/current5ms_host_preflight_fp32_calibration.sh"
```

This verifies the original CRLF SHA manifest without changing the source package; checks per-entry model IDs; rebuilds all six networks from `weights_fp32.npz`; compares 1,000 model-specific FP32 goldens with `max_abs <= 1e-5`; and selects 12,000 deterministic INT8 calibration inputs from **train.npz only**.

Do not enter Docker until this stage passes.

## B. Caffe FP32 export, INT8 quantization, ELF compile

Start the existing legacy CPU-only Vitis-AI image:

```bash
sudo docker run --rm -it \
  --entrypoint /bin/bash --network=host \
  -v "$WORK":/workspace/current5ms \
  -v "$SOURCE":/workspace/source:ro \
  -v /home/cwc/dpu_work/axu3egb_model_compile:/workspace/axu3egb:ro \
  -w /workspace/current5ms \
  local/vitis-ai-cpu:1.2.1
```

Inside the container run sequentially:

```bash
bash /workspace/current5ms/current5ms_run_caffe_fp32_exports.sh
bash /workspace/current5ms/current5ms_quantize_and_measure_int8.sh
bash /workspace/current5ms/current5ms_compile_six_b1152.sh
```

The scripts activate `vitis-ai-caffe`, configure the known glog preload, never request GPU mode, and pin the AXU3EGB B1152 DCF SHA-256:

```text
cbd84eb836ef0de8c16ee90fd59994cb83237995d56f81156f5cb30f3c98b1da
```

Expected ELF files are under `compiled_b1152/<model>/dpu_falcon_p_*.elf`.

## C. Board package and hardware checks

Exit Docker, then execute:

```bash
"$PYTHON" "$WORK/current5ms_prepare_board_goldens.py" \
  --source-root "$SOURCE" \
  --quantized-root "$WORK/quantized" \
  --output-dir "$WORK/board_goldens"

bash "$WORK/board_current5ms/build_aarch64.sh"
bash "$WORK/board_current5ms/stage_board_package.sh"

scp -r "$WORK/board_package_current5ms_12d_six_v1" axu3egb:/home/root/
```

On AXU3EGB:

```sh
cd /home/root/board_package_current5ms_12d_six_v1
sha256sum -c SHA256SUMS.txt

for model in \
  depth10_width512 depth10_width1024 \
  depth22_width512 depth22_width1024 \
  depth34_width512 depth34_width1024
do
  ./falcon12_current5ms_dnndk_golden \
    "$model" "golden/${model}_golden_vectors.txt" 1000 || exit $?
done

./falcon12_current5ms_dnndk_udp_server --port 5008
```

Every run must end with `board_int8_consistency=PASS`.

## D. Ubuntu-to-board UDP check

Keep the new service running. On Ubuntu:

```bash
for model in \
  depth10_width512 depth10_width1024 \
  depth22_width512 depth22_width1024 \
  depth34_width512 depth34_width1024
do
  python3 "$WORK/current5ms_udp_golden_client.py" \
    --model "$model" --host 192.168.77.2 --port 5008 \
    --vectors "$WORK/board_goldens/${model}_golden_vectors.txt" \
    --count 1000 --timeout-ms 1000 \
    --report "$WORK/results/udp_v2_${model}_1000.json" || exit $?
done
```

The acceptance threshold is `udp_board_vs_caffe_int8.max_abs <= 2e-6` for all six. Only then create a separate Windows 5008 shadow-test handoff.

## Reporting constraint

Call this the **current 5 m/s Track-P 12-D six-model FPGA deployment experiment**. Its teacher is `track_p_independent_gate_state_mpc_queries_v1`, not the paper's unreleased PAMPC implementation. Preserve measured INT8 error, DPU/UDP timing, and all limitations in the report.

