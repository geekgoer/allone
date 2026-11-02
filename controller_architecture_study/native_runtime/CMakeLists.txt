cmake_minimum_required(VERSION 3.20)
project(falcon_native_io LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT WIN32)
  message(FATAL_ERROR "This first native runtime uses the existing Windows QPC/wait contract")
endif()
set(AIRSIM_ROOT "F:/Airsim/AirSim" CACHE PATH "Existing AirSim checkout")
# rpclib's CMake writes generated headers into its source tree. Work on a local
# build copy so building this library never edits the user's AirSim checkout.
set(RPC_SOURCE "${CMAKE_BINARY_DIR}/rpclib_source")
if(NOT EXISTS "${RPC_SOURCE}/CMakeLists.txt")
  file(COPY "${AIRSIM_ROOT}/external/rpclib/rpclib-2.3.0/" DESTINATION "${RPC_SOURCE}")
endif()
set(RPCLIB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RPCLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
# rpclib has one private event-loop thread per client. Windows otherwise gives
# it the whole process mask, unlike our Python helper pinned to one I/O core.
# Apply this small affinity inheritance change ONLY to the build-local copy.
file(READ "${RPC_SOURCE}/lib/rpc/client.cc" RPC_CLIENT)
if(NOT RPC_CLIENT MATCHES "falcon_owner_affinity")
  string(FIND "${RPC_CLIENT}" "std::thread io_thread([this]() {" RPC_ANCHOR)
  if(RPC_ANCHOR EQUAL -1)
    message(FATAL_ERROR "Unexpected rpclib client source; cannot preserve I/O CPU placement")
  endif()
  string(REPLACE "#include <thread>" "#include <thread>\n#define WIN32_LEAN_AND_MEAN\n#include <windows.h>" RPC_CLIENT "${RPC_CLIENT}")
  string(REPLACE "std::thread io_thread([this]() {"
    "GROUP_AFFINITY falcon_owner_affinity{};\n    GetThreadGroupAffinity(GetCurrentThread(), &falcon_owner_affinity);\n    std::thread io_thread([this, falcon_owner_affinity]() {\n        SetThreadGroupAffinity(GetCurrentThread(), &falcon_owner_affinity, nullptr);"
    RPC_CLIENT "${RPC_CLIENT}")
  file(WRITE "${RPC_SOURCE}/lib/rpc/client.cc" "${RPC_CLIENT}")
endif()
add_subdirectory("${RPC_SOURCE}" "${CMAKE_BINARY_DIR}/rpc")
get_target_property(RPC_DEFINITIONS rpc COMPILE_DEFINITIONS)
list(REMOVE_ITEM RPC_DEFINITIONS "_WIN32_WINNT=0x0501")
set_property(TARGET rpc PROPERTY COMPILE_DEFINITIONS ${RPC_DEFINITIONS} _WIN32_WINNT=0x0A00)
add_library(falcon_native_io SHARED native_io.cpp
  "${AIRSIM_ROOT}/AirLib/src/api/RpcLibClientBase.cpp"
  "${AIRSIM_ROOT}/AirLib/src/vehicles/multirotor/api/MultirotorRpcLibClient.cpp"
  "${AIRSIM_ROOT}/AirLib/src/common/common_utils/FileSystem.cpp")
target_include_directories(falcon_native_io PRIVATE "${AIRSIM_ROOT}/AirLib/include"
  "${AIRSIM_ROOT}/AirLib/deps/eigen3" "${AIRSIM_ROOT}/MavLinkCom/include"
  "${RPC_SOURCE}/include")
target_compile_definitions(falcon_native_io PRIVATE _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE
  _WIN32_WINNT=0x0A00 RPCLIB_MSGPACK=clmdep_msgpack)
target_link_libraries(falcon_native_io PRIVATE rpc ws2_32 shell32)
if(MSVC)
  target_compile_options(falcon_native_io PRIVATE /EHsc /bigobj /wd4996 /wd4244)
endif()
