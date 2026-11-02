cmake_minimum_required(VERSION 3.20)

project(falcon_acado_rpg_mpc LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RPG_MPC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/rpg_mpc")
set(ACADO_GENERATED "${RPG_MPC_ROOT}/model/quadrotor_mpc_codegen")
set(QPOASES_ROOT "${RPG_MPC_ROOT}/externals/qpoases")

set(QPOASES_SOURCES
  "${QPOASES_ROOT}/SRC/Bounds.cpp"
  "${QPOASES_ROOT}/SRC/Constraints.cpp"
  "${QPOASES_ROOT}/SRC/CyclingManager.cpp"
  "${QPOASES_ROOT}/SRC/Indexlist.cpp"
  "${QPOASES_ROOT}/SRC/MessageHandling.cpp"
  "${QPOASES_ROOT}/SRC/QProblem.cpp"
  "${QPOASES_ROOT}/SRC/QProblemB.cpp"
  "${QPOASES_ROOT}/SRC/SubjectTo.cpp"
  "${QPOASES_ROOT}/SRC/Utils.cpp"
  "${QPOASES_ROOT}/SRC/EXTRAS/SolutionAnalysis.cpp"
)

add_library(falcon_acado_mpc SHARED
  falcon_acado_c_api.cpp
  "${ACADO_GENERATED}/acado_qpoases_interface.cpp"
  "${ACADO_GENERATED}/acado_integrator.c"
  "${ACADO_GENERATED}/acado_solver.c"
  "${ACADO_GENERATED}/acado_auxiliary_functions.c"
  ${QPOASES_SOURCES}
)

target_include_directories(falcon_acado_mpc PRIVATE
  "${ACADO_GENERATED}"
  "${QPOASES_ROOT}"
  "${QPOASES_ROOT}/INCLUDE"
  "${QPOASES_ROOT}/SRC"
)

target_compile_definitions(falcon_acado_mpc PRIVATE FALCON_ACADO_BUILD_DLL)

if(MSVC)
  target_compile_options(falcon_acado_mpc PRIVATE /O2 /W3)
else()
  target_compile_options(falcon_acado_mpc PRIVATE -O3 -Wall -Wextra)
  if(MINGW)
    target_link_options(falcon_acado_mpc PRIVATE
      -static-libgcc -static-libstdc++ -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic)
  endif()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(falcon_acado_mpc PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark_falcon_acado benchmark_falcon_acado.cpp)
target_link_libraries(benchmark_falcon_acado PRIVATE falcon_acado_mpc)

set_target_properties(falcon_acado_mpc benchmark_falcon_acado PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# MinGW's C++ runtime is linked statically above, but the toolchain's OpenMP /
# threading stack can still retain a dependency on libwinpthread-1.dll.  Keep
# that dependency beside our DLL so Python does not depend on a globally
# configured MinGW PATH.
if(MINGW)
  get_filename_component(MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
  add_custom_command(TARGET falcon_acado_mpc POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${MINGW_BIN_DIR}/libwinpthread-1.dll"
      "$<TARGET_FILE_DIR:falcon_acado_mpc>"
    COMMENT "Bundling the MinGW pthread runtime next to the ACADO DLL"
  )
endif()
