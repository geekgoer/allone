#!/bin/sh
# Same binary and cases, with the snapshot builder switched on/off in ABBA order.
set -eu
KIT=${1:?directory containing the binary and frozen inputs}
OUT=${2:?fresh result directory}
OLD=/home/root/ps_migration_20261116_run001
N=Drone1,Drone2,Drone3,Drone4,Drone5,Drone6,Drone7,Drone8,Drone9,Drone10,Drone11,Drone12
(cd "$KIT" && sha256sum -c inputs.sha256)
mkdir "$OUT"
cp "$KIT/inputs.sha256" "$OUT/"
for job in off_1 on_1 on_2 off_2; do
    run="$OUT/$job"
    mkdir "$run"
    echo "RUN $job"
    if ! "$OLD/bin/prepare_case_v2" --host 192.168.77.1 --case "$KIT/case.json" \
        --calibration "$OLD/inputs/thrust_calibration.json" --vehicles "$N" \
        --reference-start-s 1.5 --out "$run/prepare" > "$run/prepare.log" 2>&1; then
        cat "$run/prepare.log"
        exit 1
    fi
    if ! sh "$OLD/with_exclusive_dpu_v2.sh" "$run/guard" "$KIT/ps_obligation_control_repair" \
        --prepared "$run/prepare/prepared_case.json" --config "$KIT/config_${job%_*}.json" \
        --policy adaptive --out "$run/trial" --duration-s 32 > "$run/controller.log" 2>&1; then
        tail -35 "$run/controller.log"
        exit 1
    fi
done
printf 'Snapshot comparison saved in %s\n' "$OUT"
