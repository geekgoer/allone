#!/bin/bash
set -eo pipefail
PS_WORK=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PS_BUNDLE="$PS_WORK/../route_b_ps_handoff_20261116"
PS_PACKAGE="$PS_BUNDLE/project/controller_architecture_study/docs/main18_requant_release001_delivery_20261115/board_package_requant18_release001"
PS_DNNDK=/home/cwc/dpu_work/vitis_ai_dnndk_axu3egb
PS_JSON="$PS_BUNDLE/third_party_reference/AirSim/MavLinkCom/common_utils"
PS_BUILD=${1:-"$PS_WORK/build_local_001"}
mkdir "$PS_BUILD"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source /home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux
PS_ELFS=("$PS_PACKAGE"/elf/dpu_*.elf)
[ "${#PS_ELFS[@]}" -eq 18 ]
$CXX -O2 -ffp-contract=off -std=c++14 -Wall -Wextra \
  -I"$PS_DNNDK/n2cube/include" -I"$PS_PACKAGE" -I"$PS_JSON" -I"$PS_WORK" \
  -c "$PS_WORK/local_dpu.cpp" -o "$PS_BUILD/local_dpu.o"
$CXX -O2 -ffp-contract=off -std=c++14 -Wall -Wextra \
  -I"$PS_JSON" -I"$PS_WORK" "$PS_WORK/dpu_check.cpp" "$PS_BUILD/local_dpu.o" "${PS_ELFS[@]}" \
  -L"$PS_DNNDK/n2cube/build" -Wl,-rpath-link,"$PS_DNNDK/n2cube/build" \
  -ln2cube -lhineon -ldpuaol -pthread -ldl -o "$PS_BUILD/ps_dpu_check"
file "$PS_BUILD/ps_dpu_check"
sha256sum "$PS_BUILD/ps_dpu_check"
