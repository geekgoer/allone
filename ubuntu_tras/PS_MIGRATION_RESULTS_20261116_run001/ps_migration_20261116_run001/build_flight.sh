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
cp "$PS_WORK/flight_runtime.cpp" "$PS_BUILD/flight_runtime.cpp"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux
PS_FLAGS=(-O2 -ffp-contract=off -std=c++14 -Wall -Wextra -pthread -I"$PS_JSON" -I"$PS_WORK")
$CXX "${PS_FLAGS[@]}" -c "$PS_WORK/features/features.cpp" -o "$PS_BUILD/features.o"
$CXX "${PS_FLAGS[@]}" "$PS_WORK/test_native_contract.cpp" "$PS_BUILD/features.o" -o "$PS_BUILD/test_native_contract"
$CXX "${PS_FLAGS[@]}" "$PS_WORK/features/test_features.cpp" "$PS_BUILD/features.o" -o "$PS_BUILD/test_features"
PS_ELFS=("$PS_PACKAGE"/elf/dpu_*.elf)
$CXX "${PS_FLAGS[@]}" "$PS_BUILD/flight_runtime.cpp" "$PS_LOCAL_BUILD/local_dpu.o" "$PS_BUILD/features.o" \
  "$PS_RPC_BUILD/libpsairsim.a" "$PS_RPC_BUILD/librpc.a" "${PS_ELFS[@]}" \
  -L"$PS_DNNDK/n2cube/build" -Wl,-rpath-link,"$PS_DNNDK/n2cube/build" \
  -ln2cube -lhineon -ldpuaol -pthread -ldl -o "$PS_BUILD/ps_fixed_control"
sha256sum "$PS_BUILD/ps_fixed_control" "$PS_BUILD/test_features" "$PS_BUILD/test_native_contract"
