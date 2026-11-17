#!/usr/bin/env bash
# Rebuild original DNNDK runner into a NEW directory; do not alter frozen assets.
set -eo pipefail
R18_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
R18_PACKAGE="$R18_ROOT/reference_assets/deployment"
R18_OUTPUT=${1:?"usage: bash build_runner.sh /new/output/directory"}
: "${R18_SDK:?Set R18_SDK to environment-setup-aarch64-xilinx-linux}"
: "${R18_DNNDK:?Set R18_DNNDK to the existing DNNDK distribution}"
test -f "$R18_SDK"
test -f "$R18_DNNDK/n2cube/include/dnndk/dnndk.h"
(cd "$R18_PACKAGE" && sha256sum -c SHA256SUMS.txt)
# Existing source snapshot uses the same command as the recorded successful build.
mkdir "$R18_OUTPUT"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source "$R18_SDK"
R18_ELFS=("$R18_PACKAGE"/elf/dpu_*.elf)
[ "${#R18_ELFS[@]}" -eq 18 ]
$CXX -O2 -ffp-contract=off -std=c++11 -Wall -Wextra \
  -I"$R18_DNNDK/n2cube/include" -I"$R18_PACKAGE" \
  "$R18_PACKAGE/pool237_runner.cc" "${R18_ELFS[@]}" \
  -L"$R18_DNNDK/n2cube/build" -Wl,-rpath-link,"$R18_DNNDK/n2cube/build" \
  -ln2cube -lhineon -ldpuaol -lpthread -ldl -o "$R18_OUTPUT/pool237_runner"
sha256sum "$R18_OUTPUT/pool237_runner" > "$R18_OUTPUT/binary.sha256"
echo 'RUNNER_BUILD_COMPLETE; no hardware command was run; this is not the full PS controller.'
