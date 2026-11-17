#!/bin/sh
# Board only. New ID per condition/repetition. Same prepared case/profile.
set -eu
PS_POLICY=${1:?fixed or adaptive required}
PS_TRIAL=${2:?three-digit new trial ID required}
case "$PS_POLICY" in fixed|adaptive) ;; *) exit 2 ;; esac
case "$PS_TRIAL" in [0-9][0-9][0-9]) ;; *) exit 2 ;; esac
PS_ROOT=/home/root/ps_migration_20261116_run001
PS_BINARY=${PS_OBLIGATION_BINARY:-"$PS_ROOT/bin/ps_obligation_control_v4"}
PS_LABEL=dual_${PS_POLICY}_${PS_TRIAL}
test ! -e "$PS_ROOT/prepare_$PS_LABEL"
test ! -e "$PS_ROOT/$PS_LABEL"
sha256sum "$PS_BINARY" "$PS_ROOT/config_dual_001.json" "$PS_ROOT/bin/prepare_case_v2"
"$PS_ROOT/bin/prepare_case_v2" --host 192.168.77.1 \
  --case "$PS_ROOT/inputs/case.json" \
  --calibration "$PS_ROOT/inputs/thrust_calibration.json" \
  --vehicles Drone1,Drone2 --reference-start-s 1.5 --out "$PS_ROOT/prepare_$PS_LABEL"
sh "$PS_ROOT/with_exclusive_dpu_v2.sh" "$PS_ROOT/guard_$PS_LABEL" \
  "$PS_BINARY" --prepared "$PS_ROOT/prepare_$PS_LABEL/prepared_case.json" \
  --config "$PS_ROOT/config_dual_001.json" --policy "$PS_POLICY" \
  --out "$PS_ROOT/$PS_LABEL" --duration-s 20
