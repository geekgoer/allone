cmake_minimum_required(VERSION 3.21)
project(airsim_command_timing LANGUAGES CXX)

set(AIRLIB "" CACHE PATH "AirLib directory from the existing Windows Blocks plugin")
if(NOT MSVC OR NOT EXISTS "${AIRLIB}/lib/x64/Release/AirLib.lib")
    message(FATAL_ERROR "Use x64 MSVC and set AIRLIB to a built Windows AirLib directory")
endif()

add_executable(probe_command_timing probe_command_timing.cpp)
target_compile_features(probe_command_timing PRIVATE cxx_std_17)
target_compile_definitions(probe_command_timing PRIVATE NOMINMAX _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_WARNINGS)
target_include_directories(probe_command_timing PRIVATE "${AIRLIB}/include" "${AIRLIB}/deps/eigen3")
target_link_libraries(probe_command_timing PRIVATE
    "${AIRLIB}/lib/x64/Release/AirLib.lib"
    "${AIRLIB}/deps/MavLinkCom/lib/x64/Release/MavLinkCom.lib"
    "${AIRLIB}/deps/rpclib/lib/x64/Release/rpc.lib"
    ws2_32 mswsock shell32)
