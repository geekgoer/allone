#!/bin/sh
# Frozen slower route, unchanged PS controller and real DPU model pool.
set -eu
ROOT=${CAPACITY_TRIAL_ROOT:-/home/root/ps_capacity_edge_followup_20261117_002}
CONTROL=/home/root/ps_capacity_edge_20261117_001
BIN=${CAPACITY_BINARY:-$CONTROL/ps_obligation_control_repair}
OLD=/home/root/ps_migration_20261116_run001
CASE=${1:?case_w3.8_r0.8.json or case_w3.8_r0.7.json}
DURATION=${2:?32 or 36}
VEHICLES=${3:?comma-separated vehicle names}
PERIOD=${4:?20, 30, 40, 50, or 80}
POLICY=${5:?fixed or adaptive}
LABEL=${6:?new trial label}
case "$CASE:$DURATION" in case_w3.8_r0.8.json:32|case_w3.8_r0.7.json:36|case_w2.4_r0.8.json:32) ;; *) exit 2 ;; esac
case "$PERIOD" in 20|30|40|50|80) ;; *) exit 2 ;; esac
case "$POLICY" in fixed|adaptive) ;; *) exit 2 ;; esac
case "$LABEL" in *[!a-zA-Z0-9_-]*|'') exit 2 ;; esac
test ! -e "$ROOT/$LABEL"
test ! -e "$ROOT/prepare_$LABEL"
sha256sum "$ROOT/$CASE" "$CONTROL/config_edge_T$PERIOD.json" "$BIN"
"$OLD/bin/prepare_case_v2" --host 192.168.77.1 --case "$ROOT/$CASE" \
  --calibration "$OLD/inputs/thrust_calibration.json" --vehicles "$VEHICLES" \
  --reference-start-s 1.5 --out "$ROOT/prepare_$LABEL"
sh "$OLD/with_exclusive_dpu_v2.sh" "$ROOT/guard_$LABEL" \
  "$BIN" --prepared "$ROOT/prepare_$LABEL/prepared_case.json" \
  --config "$CONTROL/config_edge_T$PERIOD.json" --policy "$POLICY" \
  --out "$ROOT/$LABEL" --duration-s "$DURATION"
