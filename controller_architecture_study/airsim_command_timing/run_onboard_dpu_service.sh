#!/bin/sh
# Start release001 model 11 from the read-only reference package.
set -eu
base=/home/root/board_package_requant18_release001
if netstat -lun | grep -q ':5240[[:space:]]'; then
    echo 'UDP port 5240 is already in use' >&2
    exit 1
fi
for fd in /proc/[0-9]*/fd/*; do
    target=$(readlink "$fd" 2>/dev/null || true)
    case "$target" in /dev/dpu|/dev/dpu[0-9]*) echo 'DPU is already in use' >&2; exit 1 ;; esac
done
out=$(mktemp -d /run/tcs_onboard_dpu_20261124_XXXXXX)
cd "$base"
nohup ./pool237_runner serve --pool main18_plain_mlp_h4_w1024_seed0 --port 5240 \
    --inputs inputs_raw.txt --warmup 100 --out "$out/service" >"$out/service.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$out/pid"
for i in 1 2 3 4 5 6 7 8 9 10; do
    if grep -q '^READY.*port=5240' "$out/service.log"; then
        printf '%s\n' "$out"
        exit 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
cat "$out/service.log" >&2
kill "$pid" 2>/dev/null || true
exit 1
