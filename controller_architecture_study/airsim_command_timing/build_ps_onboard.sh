#!/usr/bin/env bash
# Usage: build_ps_onboard.sh SDK_ENV DNNDK_LIB_DIR RELEASE RPC_HEADERS PS_RUNTIME NEW_BUILD
set -eo pipefail
[[ $# == 6 ]]
script=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
libs=$(realpath -e -- "$2")
models=$(realpath -e -- "$3")
rpc_headers=$(realpath -e -- "$4")
runtime=$(realpath -e -- "$5")
source "$1"
mkdir -- "$6"
build=$(cd -- "$6" && pwd)
src="$runtime/source"
deps="$runtime/build_deps"
elfs=("$models"/elf/dpu_*.elf)
test "${#elfs[@]}" -eq 18
$CXX -O2 -ffp-contract=off -std=c++14 -Wall -Wextra -pthread -I"$src" -I"$deps" -I"$rpc_headers" \
    "$script/ps_onboard.cpp" "$src/features/features.cpp" "$deps/local_dpu.o" \
    "$deps/libpsairsim.a" "$deps/librpc.a" "${elfs[@]}" -L"$libs" -Wl,-rpath-link,"$libs" \
    -ln2cube -lhineon -ldpuaol -ldl -o "$build/ps_onboard"
(cd "$build" && sha256sum ps_onboard > SHA256SUMS.txt)
