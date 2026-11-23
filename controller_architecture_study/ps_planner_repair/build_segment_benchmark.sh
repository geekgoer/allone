#!/bin/bash
# Run on the existing Ubuntu cross-build host.
set -eo pipefail
test "$#" -eq 2 || { echo 'Usage: build_segment_benchmark.sh SEGMENTS NEW_BUILD_DIR' >&2; exit 2; }
segments=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd -- "$(dirname -- "$0")" && pwd)
sdk=/home/cwc/dpu_work/axu3egb_sdk
dnndk=/home/cwc/dpu_work/vitis_ai_dnndk_axu3egb
pool=/home/cwc/dpu_work/windows_framework/route_b_ps_handoff_20261116/project/controller_architecture_study/docs/main18_requant_release001_delivery_20261115/board_package_requant18_release001
export LD_LIBRARY_PATH
source "$sdk/environment-setup-aarch64-xilinx-linux"
header=$(find "$dnndk" -path '*/dnndk/dnndk.h' -print -quit)
test -n "$header"
mkdir "$build"
$CXX -O2 -std=c++14 -I"$(dirname "$(dirname "$header")")" "$here/benchmark_segments.cpp" \
    "$pool/elf/dpu_rq_73c051a35e68.elf" "$segments/rb11_seg_a/dpu_rb11_seg_a.elf" \
    "$segments/rb11_seg_b/dpu_rb11_seg_b.elf" -L"$dnndk/n2cube/build" \
    -Wl,-rpath-link,"$dnndk/n2cube/build" -ln2cube -lhineon -ldpuaol -pthread -ldl \
    -o "$build/benchmark_segments"
file "$build/benchmark_segments"
sha256sum "$build/benchmark_segments"
