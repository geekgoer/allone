#!/bin/bash
set -e -o pipefail
RPC_DIR=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:-host}
BUNDLE=${PS_BUNDLE:-/home/cwc/dpu_work/windows_framework/route_b_ps_handoff_20261116}
VENDOR="$BUNDLE/third_party_reference/AirSim"
RPCLIB="$VENDOR/external/rpclib/rpclib-2.3.0"
SDK=${PS_SDK:-/home/cwc/dpu_work/axu3egb_sdk/environment-setup-aarch64-xilinx-linux}
BUILD="$RPC_DIR/build_$TARGET"
mkdir -p "$BUILD/obj" "$RPC_DIR/logs"
LOG="$RPC_DIR/logs/build_${TARGET}_$(date +%Y%m%dT%H%M%S)_$$.log"
exec > >(tee "$LOG") 2>&1
if [ "$TARGET" = arm ]; then
    unset LD_LIBRARY_PATH
    source "$SDK"
elif [ "$TARGET" = host ]; then
    CXX=g++
    AR=ar
else
    echo "Usage: bash rpc/build.sh [host|arm]" >&2
    exit 2
fi
set -u
read -r -a COMPILER <<< "$CXX"
FLAGS=(-std=c++14 -O2 -g -fPIC -pthread -DRPCLIB_LINUX -DRPCLIB_CXX_STANDARD=14
       -DASIO_STANDALONE -DRPCLIB_ASIO=clmdep_asio -DRPCLIB_FMT=clmdep_fmt
       -DRPCLIB_MSGPACK=clmdep_msgpack
       -I"$RPCLIB/include" -I"$RPCLIB/dependencies/include"
       -I"$VENDOR/MavLinkCom/common_utils" -I"$RPC_DIR")
printf 'UTC build date: '; date -u --iso-8601=seconds
printf 'Compiler: '; printf '%q ' "${COMPILER[@]}"; printf '\n'
"${COMPILER[@]}" --version
printf 'Flags: '; printf '%q ' "${FLAGS[@]}"; printf '\n'
SOURCES=(lib/rpc/dispatcher.cc lib/rpc/server.cc lib/rpc/client.cc
         lib/rpc/this_handler.cc lib/rpc/this_session.cc lib/rpc/this_server.cc
         lib/rpc/rpc_error.cc lib/rpc/detail/server_session.cc
         lib/rpc/detail/response.cc lib/rpc/detail/client_error.cc
         lib/rpc/nonstd/optional.cc dependencies/src/format.cc dependencies/src/posix.cc)
OBJECTS=()
for relative in "${SOURCES[@]}"; do
    src="$RPCLIB/$relative"
    obj="$BUILD/obj/${relative//\//_}.o"
    if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ] || [ "$RPC_DIR/build.sh" -nt "$obj" ]; then
        printf 'COMPILE %s\n' "$relative"
        "${COMPILER[@]}" "${FLAGS[@]}" -c "$src" -o "$obj"
    fi
    OBJECTS+=("$obj")
done
"$AR" rcs "$BUILD/librpc.a" "${OBJECTS[@]}"
"${COMPILER[@]}" "${FLAGS[@]}" -c "$RPC_DIR/airsim_rpc.cpp" -o "$BUILD/airsim_rpc.o"
"$AR" rcs "$BUILD/libpsairsim.a" "$BUILD/airsim_rpc.o"
"${COMPILER[@]}" "${FLAGS[@]}" -I"$RPC_DIR/../features" \
    "$RPC_DIR/prepare_case.cpp" "$RPC_DIR/../features/features.cpp" \
    "$BUILD/libpsairsim.a" "$BUILD/librpc.a" -o "$BUILD/prepare_case"
"${COMPILER[@]}" "${FLAGS[@]}" "$RPC_DIR/airsim_probe.cpp" \
    "$BUILD/libpsairsim.a" "$BUILD/librpc.a" -o "$BUILD/airsim_probe"
"${COMPILER[@]}" "${FLAGS[@]}" "$RPC_DIR/test_rpc.cpp" \
    "$BUILD/libpsairsim.a" "$BUILD/librpc.a" -o "$BUILD/test_rpc"
if [ "$TARGET" = host ]; then
    # Protocol-only test peer; never included in the board runtime.
    "${COMPILER[@]}" "${FLAGS[@]}" "$RPC_DIR/mock_prepare_server.cpp" \
        "$BUILD/librpc.a" -o "$BUILD/mock_prepare_server"
fi
file "$BUILD/airsim_probe" "$BUILD/libpsairsim.a"
sha256sum "$BUILD/airsim_probe" "$BUILD/libpsairsim.a" "$BUILD/librpc.a"
echo "BUILD_PASS target=$TARGET log=$LOG"
