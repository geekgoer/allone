#!/bin/bash
# Generated archive of source, actual raw evidence and selected tested binaries.
# Original models/SDK are referenced by exact path/hash; no system is modified.
set -euo pipefail
PS_WORK=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PS_DEST=${1:?absolute new .tar.gz path outside the experiment directory required}
case "$PS_DEST" in /*.tar.gz) ;; *) exit 2 ;; esac
case "$PS_DEST" in "$PS_WORK"/*) echo 'Archive must be outside the measured source directory' >&2; exit 2 ;; esac
test ! -e "$PS_DEST"
test ! -e "$PS_DEST.sha256"
PS_STAGE=$(mktemp -d /tmp/ps_migration_manifest.XXXXXX)
cd "$PS_WORK"
find . -type f -print0 | sort -z | while IFS= read -r -d '' PS_FILE; do
  case "$PS_FILE" in */__pycache__/*|*/.git/*) continue ;; esac
  case "$PS_FILE" in
    *.py|*.cpp|*.cc|*.hpp|*.h|*.sh|*.md|*.json|*.jsonl|*.csv|*.txt|*.log|*/SHA256SUMS|*/SHA256SUMS.sha256)
      printf '%s\0' "$PS_FILE" ;;
    ./build_flight_004/ps_fixed_control|./build_adaptive_004/ps_obligation_control|./build_adaptive_005/ps_obligation_control|./build_local_002/ps_dpu_check|./build_adaptive_005/test_live_calendar_77|./build_adaptive_005/test_adaptive_live|./rpc/build_arm/prepare_case|./rpc/build_arm/airsim_probe|./rpc/build_arm/airsim_probe_final)
      printf '%s\0' "$PS_FILE" ;;
  esac
done > "$PS_STAGE/files.nul"
xargs -0 sha256sum < "$PS_STAGE/files.nul" > "$PS_STAGE/DELIVERY_FILE_SHA256.txt"
tar -czf "$PS_DEST" --transform='s,^,ps_migration_20261116_run001/,' \
  --null -T "$PS_STAGE/files.nul" -C "$PS_STAGE" DELIVERY_FILE_SHA256.txt
sha256sum "$PS_DEST" > "$PS_DEST.sha256"
tar -tzf "$PS_DEST" > "$PS_DEST.contents.txt"
echo "DELIVERY_CREATED $PS_DEST"
echo "MANIFEST_STAGING_PRESERVED $PS_STAGE"
