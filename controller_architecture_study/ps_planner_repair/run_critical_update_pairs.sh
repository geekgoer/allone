#!/bin/sh
# Pilot only. Intentionally omit the frozen updates in the experimental arm.
set -eu
KIT=${1:?directory with experiment binary and frozen inputs}
OUT=${2:?new result directory}
OLD=/home/root/ps_migration_20261116_run001
BIN="$KIT/ps_critical_update_probe"
INPUT="$KIT/inputs"
N=Drone1,Drone2,Drone3,Drone4,Drone5,Drone6,Drone7,Drone8,Drone9,Drone10,Drone11,Drone12
test -x "$BIN"
(cd "$INPUT" && sha256sum -c inputs.sha256)
if [ -f "$INPUT/vehicles.txt" ]; then N=$(cat "$INPUT/vehicles.txt"); fi
mkdir "$OUT"
sha256sum "$BIN" > "$OUT/binary.sha256"
cp "$INPUT/inputs.sha256" "$OUT/"
for job in early_1_normal early_1_omit late_1_omit late_1_normal \
           late_2_normal late_2_omit early_2_omit early_2_normal; do
    run="$OUT/$job"
    cfg="$INPUT/${job%%_*}_${job##*_}.json"
    mkdir "$run"
    echo "RUN $job"
    if ! "$OLD/bin/prepare_case_v2" --host 192.168.77.1 --case "$INPUT/case.json" \
        --calibration "$OLD/inputs/thrust_calibration.json" --vehicles "$N" \
        --reference-start-s 1.5 --out "$run/prepare" > "$run/prepare.log" 2>&1; then
        cat "$run/prepare.log"
        exit 1
    fi
    if ! sh "$OLD/with_exclusive_dpu_v2.sh" "$run/guard" "$BIN" \
        --prepared "$run/prepare/prepared_case.json" --config "$cfg" --policy fixed \
        --out "$run/trial" --duration-s 32 > "$run/controller.log" 2>&1; then
        tail -35 "$run/controller.log"
        exit 1
    fi
done
echo "Pilot runs saved in $OUT"
