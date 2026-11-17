#!/bin/bash
set -eo pipefail
study=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=${1:?"new build directory required"}
mkdir "$build"
original=/home/cwc/dpu_work/windows_framework/ps_migration_20261116_run001
bundle=/home/cwc/dpu_work/windows_framework/route_b_ps_handoff_20261116
package="$bundle/project/controller_architecture_study/docs/main18_requant_release001_delivery_20261115/board_package_requant18_release001"
dnndk=/home/cwc/dpu_work/vitis_ai_dnndk_axu3egb
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux
flags=(-O2 -ffp-contract=off -std=c++14 -Wall -Wextra -pthread -I"$study" -I"$bundle/third_party_reference/AirSim/MavLinkCom/common_utils")
for source in scheduler live_calendar revalidate; do
    extra=()
    if test "$source" = scheduler; then extra=(-DPS_PROFILE_ENABLED); fi
    $CXX "${flags[@]}" "${extra[@]}" -c "$study/scheduler/$source.cpp" -o "$build/$source.o"
done
$CXX "${flags[@]}" -c "$study/features/features.cpp" -o "$build/features.o"
elfs=("$package"/elf/dpu_*.elf)
test "${#elfs[@]}" -eq 18
$CXX "${flags[@]}" "$study/adaptive_runtime.cpp" "$original/build_adaptive_005/dependencies/local_dpu.o" \
    "$build/features.o" "$build/scheduler.o" "$build/live_calendar.o" "$build/revalidate.o" \
    "$original/build_adaptive_005/dependencies/libpsairsim.a" "$original/build_adaptive_005/dependencies/librpc.a" "${elfs[@]}" \
    -L"$dnndk/n2cube/build" -Wl,-rpath-link,"$dnndk/n2cube/build" \
    -ln2cube -lhineon -ldpuaol -pthread -ldl -o "$build/ps_obligation_control_repair"
for name in scheduler_tests live_calendar_tests revalidate_tests; do
    $CXX "${flags[@]}" -I"$study/scheduler" "$study/scheduler/$name.cpp" \
        "$build/scheduler.o" "$build/live_calendar.o" "$build/revalidate.o" -o "$build/$name"
done
sha256sum "$build/ps_obligation_control_repair" "$build"/*_tests > "$build/SHA256SUMS.txt"
file "$build/ps_obligation_control_repair"
