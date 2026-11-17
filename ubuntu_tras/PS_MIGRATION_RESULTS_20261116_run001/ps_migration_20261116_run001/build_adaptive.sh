#!/bin/bash
set -eo pipefail
PS_WORK=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PS_BUNDLE="$PS_WORK/../route_b_ps_handoff_20261116"
PS_PACKAGE="$PS_BUNDLE/project/controller_architecture_study/docs/main18_requant_release001_delivery_20261115/board_package_requant18_release001"
PS_DNNDK=/home/cwc/dpu_work/vitis_ai_dnndk_axu3egb
PS_JSON="$PS_BUNDLE/third_party_reference/AirSim/MavLinkCom/common_utils"
PS_LOCAL_BUILD=${PS_LOCAL_BUILD:-"$PS_WORK/build_local_002"}
PS_RPC_BUILD=${PS_RPC_BUILD:-"$PS_WORK/rpc/build_arm"}
PS_BUILD=${1:?"new build directory required"}
mkdir "$PS_BUILD"
mkdir -p "$PS_BUILD/source/features" "$PS_BUILD/source/rpc" "$PS_BUILD/source/scheduler" "$PS_BUILD/dependencies"
cp "$PS_WORK/adaptive_runtime.cpp" "$PS_WORK/adaptive_live.hpp" "$PS_WORK/test_adaptive_live.cpp" "$PS_WORK/build_adaptive.sh" "$PS_BUILD/source/"
cp "$PS_WORK/local_dpu.hpp" "$PS_WORK/local_dpu.cpp" "$PS_BUILD/source/"
cp "$PS_WORK/features/features.cpp" "$PS_WORK/features/features.hpp" "$PS_BUILD/source/features/"
cp "$PS_WORK/rpc/airsim_rpc.hpp" "$PS_WORK/rpc/airsim_rpc.cpp" "$PS_BUILD/source/rpc/"
cp "$PS_WORK/scheduler/scheduler.cpp" "$PS_WORK/scheduler/live_calendar.cpp" "$PS_WORK/scheduler/scheduler.hpp" "$PS_WORK/scheduler/live_calendar.hpp" "$PS_BUILD/source/scheduler/"
cp "$PS_WORK/scheduler/live_calendar_tests.cpp" "$PS_BUILD/source/scheduler/"
cp "$PS_LOCAL_BUILD/local_dpu.o" "$PS_RPC_BUILD/libpsairsim.a" "$PS_RPC_BUILD/librpc.a" "$PS_BUILD/dependencies/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux
PS_FLAGS=(-O2 -ffp-contract=off -std=c++14 -Wall -Wextra -pthread -I"$PS_JSON" -I"$PS_BUILD/source")
$CXX "${PS_FLAGS[@]}" -c "$PS_BUILD/source/features/features.cpp" -o "$PS_BUILD/features.o"
$CXX "${PS_FLAGS[@]}" -c "$PS_BUILD/source/scheduler/scheduler.cpp" -o "$PS_BUILD/scheduler.o"
$CXX "${PS_FLAGS[@]}" -c "$PS_BUILD/source/scheduler/live_calendar.cpp" -o "$PS_BUILD/live_calendar.o"
$CXX "${PS_FLAGS[@]}" "$PS_BUILD/source/test_adaptive_live.cpp" "$PS_BUILD/scheduler.o" "$PS_BUILD/live_calendar.o" -o "$PS_BUILD/test_adaptive_live"
$CXX "${PS_FLAGS[@]}" -I"$PS_BUILD/source/scheduler" "$PS_BUILD/source/scheduler/live_calendar_tests.cpp" "$PS_BUILD/scheduler.o" "$PS_BUILD/live_calendar.o" -o "$PS_BUILD/test_live_calendar"
PS_ELFS=("$PS_PACKAGE"/elf/dpu_*.elf)
[ "${#PS_ELFS[@]}" -eq 18 ]
$CXX "${PS_FLAGS[@]}" "$PS_BUILD/source/adaptive_runtime.cpp" "$PS_BUILD/dependencies/local_dpu.o" "$PS_BUILD/features.o" \
  "$PS_BUILD/scheduler.o" "$PS_BUILD/live_calendar.o" \
  "$PS_BUILD/dependencies/libpsairsim.a" "$PS_BUILD/dependencies/librpc.a" "${PS_ELFS[@]}" \
  -L"$PS_DNNDK/n2cube/build" -Wl,-rpath-link,"$PS_DNNDK/n2cube/build" \
  -ln2cube -lhineon -ldpuaol -pthread -ldl -o "$PS_BUILD/ps_obligation_control"
find "$PS_BUILD/source" "$PS_BUILD/dependencies" -type f -print0 | sort -z | xargs -0 sha256sum > "$PS_BUILD/SOURCE_AND_DEPENDENCIES_SHA256.txt"
sha256sum "${PS_ELFS[@]}" "$PS_JSON/json.hpp" > "$PS_BUILD/FROZEN_EXTERNAL_SHA256.txt"
sha256sum "$PS_BUILD/ps_obligation_control" "$PS_BUILD/test_adaptive_live" "$PS_BUILD/test_live_calendar" > "$PS_BUILD/SHA256SUMS.txt"
file "$PS_BUILD/ps_obligation_control" "$PS_BUILD/test_adaptive_live"
sha256sum "$PS_BUILD/ps_obligation_control" "$PS_BUILD/test_adaptive_live"
