#!/bin/bash
set -eo pipefail
study=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=${1:?"new build directory required"}
mkdir "$build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux
flags=(-O2 -ffp-contract=off -std=c++14 -pthread -I"$study")
for source in scheduler live_calendar revalidate; do
    $CXX "${flags[@]}" -c "$study/$source.cpp" -o "$build/$source.o"
done
for executable in scheduler_tests live_calendar_tests revalidate_tests management_bench; do
    $CXX "${flags[@]}" "$study/$executable.cpp" "$build/scheduler.o" \
        "$build/live_calendar.o" "$build/revalidate.o" -o "$build/$executable"
done
sha256sum "$build"/scheduler_tests "$build"/live_calendar_tests \
    "$build"/revalidate_tests "$build"/management_bench > "$build/SHA256SUMS.txt"
