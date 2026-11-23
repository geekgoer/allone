#!/bin/sh
# Run named ACK recovery arms, defaulting to the original paired pilot.
set -eu
KIT=${1:?prepared kit directory}
OUT=${2:?new result directory}
OLD=/home/root/ps_migration_20261116_run001

(cd "$KIT" && sha256sum -c inputs.sha256)
mkdir "$OUT"
cp "$KIT/inputs.sha256" "$OUT/"
shift 2
if [ "$#" -eq 0 ]; then
    set -- drone2_off drone2_on drone5_on drone5_off
fi
for job in "$@"; do
    case "$job" in ''|*[!a-z0-9_]*) echo "Invalid run name: $job" >&2; exit 1;; esac
    test -r "$KIT/config_$job.json"
    run="$OUT/$job"
    mkdir "$run"
    echo "RUN $job"
    if ! "$OLD/bin/prepare_case_v2" --host 192.168.77.1 --case "$KIT/case.json" \
        --calibration "$OLD/inputs/thrust_calibration.json" --vehicles "$(cat "$KIT/vehicles.txt")" \
        --reference-start-s 1.5 --out "$run/prepare" > "$run/prepare.log" 2>&1; then
        cat "$run/prepare.log"
        exit 1
    fi
    if ! sh "$OLD/with_exclusive_dpu_v2.sh" "$run/guard" "$KIT/ps_obligation_control_repair" \
        --prepared "$run/prepare/prepared_case.json" --config "$KIT/config_$job.json" \
        --policy fixed --out "$run/trial" --duration-s 32 > "$run/controller.log" 2>&1; then
        tail -35 "$run/controller.log"
        exit 1
    fi
done
printf 'Command-retime comparison saved in %s\n' "$OUT"
