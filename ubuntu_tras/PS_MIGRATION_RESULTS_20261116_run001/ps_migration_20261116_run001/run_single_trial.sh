#!/bin/sh
# Board only. Each ID names NEW prepare/guard/raw directories; never overwrite.
set -eu
PS_TRIAL=${1:?three-digit new trial ID required}
case "$PS_TRIAL" in [0-9][0-9][0-9]) ;; *) exit 2 ;; esac
PS_ROOT=/home/root/ps_migration_20261116_run001
test ! -e "$PS_ROOT/prepare_single_$PS_TRIAL"
test ! -e "$PS_ROOT/fixed_single_$PS_TRIAL"
sha256sum "$PS_ROOT/bin/ps_fixed_control_v4" "$PS_ROOT/bin/prepare_case_v2"
"$PS_ROOT/bin/prepare_case_v2" --host 192.168.77.1 \
  --case "$PS_ROOT/inputs/case.json" \
  --calibration "$PS_ROOT/inputs/thrust_calibration.json" \
  --vehicles Drone1 --reference-start-s 1.5 --out "$PS_ROOT/prepare_single_$PS_TRIAL"
sh "$PS_ROOT/with_exclusive_dpu_v2.sh" "$PS_ROOT/guard_fixed_single_$PS_TRIAL" \
  "$PS_ROOT/bin/ps_fixed_control_v4" \
  --prepared "$PS_ROOT/prepare_single_$PS_TRIAL/prepared_case.json" \
  --out "$PS_ROOT/fixed_single_$PS_TRIAL" --model 11 --period-ms 50 --duration-s 20
