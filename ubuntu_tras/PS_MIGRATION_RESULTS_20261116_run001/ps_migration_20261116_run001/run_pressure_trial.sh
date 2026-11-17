#!/bin/sh
# Separately authorized stress case: no model/normalization change or fake DPU.
set -eu
PS_SKIP=${1:?off or on}
PS_TRIAL=${2:?three-digit fresh ID}
PS_PERIOD=${3:-20}
case "$PS_PERIOD" in 20|30) ;; *) exit 2 ;; esac
case "$PS_SKIP" in off|on) ;; *) exit 2 ;; esac
case "$PS_TRIAL" in [0-9][0-9][0-9]) ;; *) exit 2 ;; esac
PS_ROOT=/home/root/ps_migration_20261116_run001
PS_LABEL=pressure${PS_PERIOD}_skip_${PS_SKIP}_${PS_TRIAL}
test ! -e "$PS_ROOT/prepare_$PS_LABEL"
test ! -e "$PS_ROOT/$PS_LABEL"
PS_CONFIG="$PS_ROOT/config_pressure${PS_PERIOD}_001.json"
sha256sum "$PS_ROOT/bin/ps_obligation_control_v5" "$PS_CONFIG"
"$PS_ROOT/bin/prepare_case_v2" --host 192.168.77.1 \
  --case "$PS_ROOT/inputs/case.json" --calibration "$PS_ROOT/inputs/thrust_calibration.json" \
  --vehicles Drone1,Drone2 --reference-start-s 1.5 --out "$PS_ROOT/prepare_$PS_LABEL"
set -- "$PS_ROOT/bin/ps_obligation_control_v5" \
  --prepared "$PS_ROOT/prepare_$PS_LABEL/prepared_case.json" \
  --config "$PS_CONFIG" --policy adaptive \
  --out "$PS_ROOT/$PS_LABEL" --duration-s 20
if test "$PS_SKIP" = on; then set -- "$@" --enable-skip; fi
sh "$PS_ROOT/with_exclusive_dpu_v2.sh" "$PS_ROOT/guard_$PS_LABEL" "$@"
