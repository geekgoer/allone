#!/bin/sh
# One finite AirSim/real-DPU trial. The caller supplies a frozen case and profile.
set -eu
ROOT=$(CDPATH= cd "$(dirname "$0")" && pwd)
OLD=/home/root/ps_migration_20261116_run001
CASE=${1:?case_w3.8.json}
CONFIG=${2:?config_edge_T50.json}
POLICY=${3:?fixed or adaptive}
VEHICLES=${4:?comma-separated vehicle names}
LABEL=${5:?new trial label}
case "$CASE" in case_w3.0.json|case_w3.4.json|case_w3.8.json) ;; *) exit 2 ;; esac
case "$CONFIG" in config_edge_T20.json|config_edge_T30.json|config_edge_T40.json|config_edge_T50.json|config_edge_T80.json) ;; *) exit 2 ;; esac
case "$POLICY" in fixed|adaptive) ;; *) exit 2 ;; esac
case "$LABEL" in *[!a-zA-Z0-9_-]*|'') exit 2 ;; esac
test ! -e "$ROOT/$LABEL"
test ! -e "$ROOT/prepare_$LABEL"
sha256sum "$ROOT/ps_obligation_control_repair" "$ROOT/$CASE" "$ROOT/$CONFIG"
"$OLD/bin/prepare_case_v2" --host 192.168.77.1 --case "$ROOT/$CASE" \
  --calibration "$OLD/inputs/thrust_calibration.json" --vehicles "$VEHICLES" \
  --reference-start-s 1.5 --out "$ROOT/prepare_$LABEL"
sh "$OLD/with_exclusive_dpu_v2.sh" "$ROOT/guard_$LABEL" \
  "$ROOT/ps_obligation_control_repair" --prepared "$ROOT/prepare_$LABEL/prepared_case.json" \
  --config "$ROOT/$CONFIG" --policy "$POLICY" --out "$ROOT/$LABEL" --duration-s 26
