#!/bin/sh
# Only used in a dedicated T,c,s deployment on Zynq.
set -eu
kit=${1:?kit path required}
out=${2:?new result path required}
config=${3:?configuration filename required}
trace=${4:?trace base required}
base=/home/root/ps_migration_20261116_run001
test "$(readlink -f "$kit")" = "$kit"
test "$(dirname "$kit")" = /home/root
case "$(basename "$kit")" in tcs_command_trace_*) ;; *) exit 2 ;; esac
test "$(dirname "$out")" = "$kit"
case "$(basename "$out")" in run_[a-zA-Z0-9_]*) ;; *) exit 2 ;; esac
case "$config" in config_natural_fixed20.json|config_natural_retime20.json|config_ping.json|config_receipt.json) ;; *) exit 2 ;; esac

require_idle() {
    for fd in /proc/[0-9]*/fd/*; do
        target=$(readlink "$fd" 2>/dev/null || true)
        case "$target" in /dev/dpu|/dev/dpu[0-9]*) echo "DPU in use by $fd; abort" >&2; exit 1 ;; esac
    done
}

require_idle
mkdir "$out"
"$base/bin/prepare_case_v2" --host 192.168.77.1 --port 41591 --case "$kit/case.json" \
    --calibration "$base/inputs/thrust_calibration.json" --vehicles "$(cat "$kit/vehicles.txt")" \
    --reference-start-s 1.5 --out "$out/prepare" > "$out/prepare.log" 2>&1 || { cat "$out/prepare.log"; exit 1; }
require_idle
"$kit/ps_obligation_control_repair" --host 192.168.77.1 --port 41591 --trace-base "$trace" \
    --prepared "$out/prepare/prepared_case.json" --config "$kit/$config" \
    --policy fixed --out "$out/trial" --duration-s 32 > "$out/controller.log" 2>&1 || { tail -35 "$out/controller.log"; exit 1; }
tail -12 "$out/controller.log"
