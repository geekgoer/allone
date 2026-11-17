#!/bin/sh
set -eu
PS_ROOT=/home/root/ps_migration_20261116_run001
# adaptive101 is the separately observed first post-fix trial. Remaining trials
# interleave policies. Every item is a fresh, complete20s scene/reset.
for PS_SPEC in fixed:101 adaptive:102 fixed:102 adaptive:103 fixed:103; do
  PS_POLICY=${PS_SPEC%:*}
  PS_TRIAL=${PS_SPEC#*:}
  sh "$PS_ROOT/run_dual_trial_v4.sh" "$PS_POLICY" "$PS_TRIAL"
  python3 - "$PS_ROOT/dual_${PS_POLICY}_${PS_TRIAL}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text())
assert not s['early_stop'] and not s['planner_failed'] and not s['executor_failed'] and not s['journal_failed'],s
assert len(s['vehicles'])==2 and all(not v['retired'] and v['counts']=={'delivered':400} for v in s['vehicles']),s
f=json.loads((p/'final_states.json').read_text())
assert all(not x.get('error') and not x.get('collision_new') for x in f),f
print('NORMAL_TRIAL_COMPLETE',p.name,'delivered=800 preliminary_status_check_only')
PY
done
