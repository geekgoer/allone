#!/bin/sh
# Two repeats at each capacity point, mirrored to reduce run-order bias.
set -eu
KIT=${1:?directory containing the binary and frozen inputs}
OUT=${2:?fresh result directory}
OLD=/home/root/ps_migration_20261116_run001

(cd "$KIT" && sha256sum -c inputs.sha256)
mkdir "$OUT"
cp "$KIT/inputs.sha256" "$OUT/"
for job in n12_t20_1 n10_t20_1 n12_t30_1 n12_t30_2 n10_t20_2 n12_t20_2; do
    arm=${job%_*}
    run="$OUT/$job"
    mkdir "$run"
    echo "RUN $job"
    if ! "$OLD/bin/prepare_case_v2" --host 192.168.77.1 --case "$KIT/case.json" \
        --calibration "$OLD/inputs/thrust_calibration.json" --vehicles "$(cat "$KIT/vehicles_${arm}.txt")" \
        --reference-start-s 1.5 --out "$run/prepare" > "$run/prepare.log" 2>&1; then
        cat "$run/prepare.log"
        exit 1
    fi
    if ! sh "$OLD/with_exclusive_dpu_v2.sh" "$run/guard" "$KIT/ps_obligation_control_repair" \
        --prepared "$run/prepare/prepared_case.json" --config "$KIT/config_${arm}.json" \
        --policy fixed --out "$run/trial" --duration-s 32 > "$run/controller.log" 2>&1; then
        tail -35 "$run/controller.log"
        exit 1
    fi
done
printf 'Capacity comparison saved in %s\n' "$OUT"
