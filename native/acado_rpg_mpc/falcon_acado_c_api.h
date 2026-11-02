#pragma once

#if defined(_WIN32)
#  if defined(FALCON_ACADO_BUILD_DLL)
#    define FALCON_ACADO_API __declspec(dllexport)
#  else
#    define FALCON_ACADO_API __declspec(dllimport)
#  endif
#else
#  define FALCON_ACADO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  FALCON_ACADO_HORIZON = 20,
  FALCON_ACADO_STATE_SIZE = 10,
  FALCON_ACADO_INPUT_SIZE = 4,
  FALCON_ACADO_REFERENCE_NODES = 21
};

/*
 * The upstream generated solver uses a 0.1 s discretization and a 20-node
 * prediction horizon.  This is independent of FALCON's 0.05 s control task
 * period.  The paper does not publish whether its ACADO solver was regenerated
 * with a different discretization, so callers must record this value.
 */
FALCON_ACADO_API double falcon_acado_prediction_dt(void);

FALCON_ACADO_API int falcon_acado_initialize(void);

FALCON_ACADO_API int falcon_acado_configure(
    const double state_weights[5],
    const double input_weights[3],
    const double input_limits[4]);

FALCON_ACADO_API int falcon_acado_reset(const double state[10]);

/*
 * state:             [px, py, pz, qw, qx, qy, qz, vx, vy, vz]
 * reference_states:  21 consecutive state vectors with the same layout
 * reference_inputs:  21 consecutive [F, wx, wy, wz] vectors
 * output:            first optimal [F, wx, wy, wz]
 * diagnostics:       optional [KKT, objective, qp_nWSR]
 */
FALCON_ACADO_API int falcon_acado_solve(
    const double state[10],
    const double reference_states[21 * 10],
    const double reference_inputs[21 * 4],
    double output[4],
    double diagnostics[3]);

FALCON_ACADO_API const char* falcon_acado_status_string(int status);

#ifdef __cplusplus
}
#endif

