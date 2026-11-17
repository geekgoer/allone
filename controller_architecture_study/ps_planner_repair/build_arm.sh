#!/bin/bash
# Build only; does not link DPU/RPC libraries or start a controller.
set -eo pipefail
study=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=${1:?"new build directory required"}
test ! -e "$build"
mkdir "$build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux
flags=(-O2 -ffp-contract=off -std=c++14 -pthread -I"$study")
for variant in baseline profile move_baseline move_profile; do
    extra=()
    if [[ "$variant" == *profile ]]; then extra=(-DPS_PROFILE_ENABLED); fi
    $CXX "${flags[@]}" "${extra[@]}" -I"$study/$variant" \
        "$study/$variant/scheduler.cpp" "$study/profile_replay.cpp" \
        -o "$build/replay_$variant"
done
sha256sum "$build"/replay_* > "$build/SHA256SUMS.txt"
file "$build"/replay_*
