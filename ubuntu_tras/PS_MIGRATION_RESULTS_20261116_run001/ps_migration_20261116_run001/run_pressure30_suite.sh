#!/bin/sh
set -eu
PS_ROOT=/home/root/ps_migration_20261116_run001
for PS_SPEC in on:002 off:002 off:003 on:003; do
  PS_SWITCH=${PS_SPEC%:*}
  PS_TRIAL=${PS_SPEC#*:}
  sh "$PS_ROOT/run_pressure_trial.sh" "$PS_SWITCH" "$PS_TRIAL" 30
done
