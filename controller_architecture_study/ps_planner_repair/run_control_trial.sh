#!/bin/sh
# Finite two-UAV acceptance using unchanged frozen case/model artifacts.
set -eu
MODE=${1:?fixed, measure, repair, pressure_off or pressure_on}
LABEL=${2:?fresh label required}
OLD=/home/root/ps_migration_20261116_run001
ROOT=$(CDPATH= cd "$(dirname "$0")" && pwd)
POLICY=adaptive
CONFIG="$ROOT/config_dual_001_repair.json"
SKIP=off
case "$MODE" in
  fixed) POLICY=fixed; CONFIG="$ROOT/config_measureonly_001.json" ;;
  measure) CONFIG="$ROOT/config_measureonly_001.json" ;;
  repair) ;;
  pressure_off) CONFIG="$ROOT/config_pressure30_001_repair.json" ;;
  pressure_on) CONFIG="$ROOT/config_pressure30_001_repair.json"; SKIP=on ;;
  trigger_off) CONFIG="$ROOT/config_pressure30_001_paired_off.json"; SKIP=on ;;
  trigger_on) CONFIG="$ROOT/config_pressure30_001_repair.json"; SKIP=on ;;
  *) exit 2 ;;
esac
test ! -e "$ROOT/$LABEL"
test ! -e "$ROOT/prepare_$LABEL"
sha256sum "$ROOT/ps_obligation_control_repair" "$CONFIG"
"$OLD/bin/prepare_case_v2" --host 192.168.77.1 --case "$OLD/inputs/case.json" \
  --calibration "$OLD/inputs/thrust_calibration.json" --vehicles Drone1,Drone2 \
  --reference-start-s 1.5 --out "$ROOT/prepare_$LABEL"
set -- "$ROOT/ps_obligation_control_repair" --prepared "$ROOT/prepare_$LABEL/prepared_case.json" \
  --config "$CONFIG" --policy "$POLICY" --out "$ROOT/$LABEL" --duration-s 20
if test "$SKIP" = on; then set -- "$@" --enable-skip; fi
sh "$OLD/with_exclusive_dpu_v2.sh" "$ROOT/guard_$LABEL" "$@"
