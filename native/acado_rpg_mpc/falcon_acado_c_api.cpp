#include "falcon_acado_c_api.h"

#include <algorithm>
#include <cmath>
#include <cstring>

extern "C" {
#include "acado_auxiliary_functions.h"
#include "acado_common.h"

ACADOvariables acadoVariables;
ACADOworkspace acadoWorkspace;
}

namespace {

constexpr int kState = ACADO_NX;
constexpr int kInput = ACADO_NU;
constexpr int kRunReference = ACADO_NY;
constexpr int kTerminalReference = ACADO_NYN;
constexpr int kNodes = ACADO_N + 1;
constexpr float kGravity = 9.8066F;

bool g_initialized = false;
bool g_has_primal_guess = false;

bool finite_array(const double* values, int count) {
  if (values == nullptr) {
    return false;
  }
  for (int index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

void set_diagonal(float* matrix, int size, int diagonal, float value) {
  matrix[diagonal * size + diagonal] = value;
}

void configure_online_data() {
  for (int node = 0; node < kNodes; ++node) {
    float* od = &acadoVariables.od[node * ACADO_NOD];
    od[0] = 0.0F;
    od[1] = 0.0F;
    od[2] = -1000.0F;
    od[3] = 0.0F;
    od[4] = 0.0F;
    od[5] = 0.0F;
    od[6] = 1.0F;
    od[7] = 0.0F;
    od[8] = 0.0F;
    od[9] = 0.0F;
  }
}

void fill_state_guess(const double state[10]) {
  for (int node = 0; node < kNodes; ++node) {
    for (int item = 0; item < kState; ++item) {
      acadoVariables.x[node * kState + item] = static_cast<float>(state[item]);
    }
  }
  for (int stage = 0; stage < ACADO_N; ++stage) {
    acadoVariables.u[stage * kInput + 0] = kGravity;
    acadoVariables.u[stage * kInput + 1] = 0.0F;
    acadoVariables.u[stage * kInput + 2] = 0.0F;
    acadoVariables.u[stage * kInput + 3] = 0.0F;
  }
}

void set_references(const double* states, const double* inputs) {
  for (int stage = 0; stage < ACADO_N; ++stage) {
    float* y = &acadoVariables.y[stage * kRunReference];
    for (int item = 0; item < kState; ++item) {
      y[item] = static_cast<float>(states[stage * kState + item]);
    }
    y[10] = 0.0F;
    y[11] = 0.0F;
    for (int item = 0; item < kInput; ++item) {
      y[12 + item] = static_cast<float>(inputs[stage * kInput + item]);
    }
  }

  const double* terminal = &states[ACADO_N * kState];
  for (int item = 0; item < kState; ++item) {
    acadoVariables.yN[item] = static_cast<float>(terminal[item]);
  }
  acadoVariables.yN[10] = 0.0F;
  acadoVariables.yN[11] = 0.0F;
}

void set_feedback_state(const double state[10]) {
  for (int item = 0; item < kState; ++item) {
    acadoVariables.x0[item] = static_cast<float>(state[item]);
  }
  float quaternion_dot = 0.0F;
  for (int item = 0; item < 4; ++item) {
    quaternion_dot += acadoVariables.x0[3 + item] * acadoVariables.y[3 + item];
  }
  if (quaternion_dot < 0.0F) {
    for (int item = 0; item < 4; ++item) {
      acadoVariables.x0[3 + item] = -acadoVariables.x0[3 + item];
    }
  }
}

}  // namespace

double falcon_acado_prediction_dt(void) {
  return 0.1;
}

int falcon_acado_initialize(void) {
  std::memset(&acadoVariables, 0, sizeof(acadoVariables));
  std::memset(&acadoWorkspace, 0, sizeof(acadoWorkspace));
  const int status = acado_initializeSolver();
  if (status != 0) {
    return status;
  }
  configure_online_data();
  const double state_weights[5] = {200.0, 500.0, 50.0, 10.0, 0.0};
  const double input_weights[3] = {1.0, 1.0, 1.0};
  const double limits[4] = {2.0, 20.0, 3.0, 2.0};
  g_initialized = true;
  g_has_primal_guess = false;
  return falcon_acado_configure(state_weights, input_weights, limits);
}

int falcon_acado_configure(
    const double state_weights[5],
    const double input_weights[3],
    const double input_limits[4]) {
  if (!g_initialized || !finite_array(state_weights, 5) ||
      !finite_array(input_weights, 3) || !finite_array(input_limits, 4)) {
    return -100;
  }
  if (input_limits[0] < 0.0 || input_limits[1] <= input_limits[0] ||
      input_limits[2] <= 0.0 || input_limits[3] <= 0.0) {
    return -101;
  }

  std::memset(acadoVariables.W, 0, sizeof(acadoVariables.W));
  std::memset(acadoVariables.WN, 0, sizeof(acadoVariables.WN));

  for (int stage = 0; stage < ACADO_N; ++stage) {
    float* weight = &acadoVariables.W[stage * kRunReference * kRunReference];
    set_diagonal(weight, kRunReference, 0, static_cast<float>(state_weights[0]));
    set_diagonal(weight, kRunReference, 1, static_cast<float>(state_weights[0]));
    set_diagonal(weight, kRunReference, 2, static_cast<float>(state_weights[1]));
    for (int item = 3; item < 7; ++item) {
      set_diagonal(weight, kRunReference, item, static_cast<float>(state_weights[2]));
    }
    for (int item = 7; item < 10; ++item) {
      set_diagonal(weight, kRunReference, item, static_cast<float>(state_weights[3]));
    }
    set_diagonal(weight, kRunReference, 10, static_cast<float>(state_weights[4]));
    set_diagonal(weight, kRunReference, 11, static_cast<float>(state_weights[4]));
    set_diagonal(weight, kRunReference, 12, static_cast<float>(input_weights[0]));
    set_diagonal(weight, kRunReference, 13, static_cast<float>(input_weights[1]));
    set_diagonal(weight, kRunReference, 14, static_cast<float>(input_weights[1]));
    set_diagonal(weight, kRunReference, 15, static_cast<float>(input_weights[2]));
  }

  set_diagonal(acadoVariables.WN, kTerminalReference, 0, static_cast<float>(state_weights[0]));
  set_diagonal(acadoVariables.WN, kTerminalReference, 1, static_cast<float>(state_weights[0]));
  set_diagonal(acadoVariables.WN, kTerminalReference, 2, static_cast<float>(state_weights[1]));
  for (int item = 3; item < 7; ++item) {
    set_diagonal(acadoVariables.WN, kTerminalReference, item, static_cast<float>(state_weights[2]));
  }
  for (int item = 7; item < 10; ++item) {
    set_diagonal(acadoVariables.WN, kTerminalReference, item, static_cast<float>(state_weights[3]));
  }
  set_diagonal(acadoVariables.WN, kTerminalReference, 10, static_cast<float>(state_weights[4]));
  set_diagonal(acadoVariables.WN, kTerminalReference, 11, static_cast<float>(state_weights[4]));

  for (int stage = 0; stage < ACADO_N; ++stage) {
    acadoVariables.lbValues[stage * kInput + 0] = static_cast<float>(input_limits[0]);
    acadoVariables.ubValues[stage * kInput + 0] = static_cast<float>(input_limits[1]);
    acadoVariables.lbValues[stage * kInput + 1] = static_cast<float>(-input_limits[2]);
    acadoVariables.ubValues[stage * kInput + 1] = static_cast<float>(input_limits[2]);
    acadoVariables.lbValues[stage * kInput + 2] = static_cast<float>(-input_limits[2]);
    acadoVariables.ubValues[stage * kInput + 2] = static_cast<float>(input_limits[2]);
    acadoVariables.lbValues[stage * kInput + 3] = static_cast<float>(-input_limits[3]);
    acadoVariables.ubValues[stage * kInput + 3] = static_cast<float>(input_limits[3]);
  }
  return 0;
}

int falcon_acado_reset(const double state[10]) {
  if (!g_initialized || !finite_array(state, kState)) {
    return -102;
  }
  fill_state_guess(state);
  set_feedback_state(state);
  g_has_primal_guess = false;
  return 0;
}

int falcon_acado_solve(
    const double state[10],
    const double reference_states[21 * 10],
    const double reference_inputs[21 * 4],
    double output[4],
    double diagnostics[3]) {
  if (!g_initialized) {
    const int initialize_status = falcon_acado_initialize();
    if (initialize_status != 0) {
      return initialize_status;
    }
  }
  if (!finite_array(state, kState) ||
      !finite_array(reference_states, kNodes * kState) ||
      !finite_array(reference_inputs, kNodes * kInput) || output == nullptr) {
    return -103;
  }

  set_references(reference_states, reference_inputs);
  set_feedback_state(state);
  if (!g_has_primal_guess) {
    fill_state_guess(state);
    acado_initializeNodesByForwardSimulation();
    g_has_primal_guess = true;
  }

  const int preparation_status = acado_preparationStep();
  if (preparation_status != 0) {
    return preparation_status;
  }
  const int feedback_status = acado_feedbackStep();
  if (feedback_status != 0) {
    return feedback_status;
  }

  for (int item = 0; item < kInput; ++item) {
    output[item] = static_cast<double>(acadoVariables.u[item]);
  }
  if (diagnostics != nullptr) {
    diagnostics[0] = static_cast<double>(acado_getKKT());
    diagnostics[1] = static_cast<double>(acado_getObjective());
    diagnostics[2] = static_cast<double>(acado_getNWSR());
  }
  return 0;
}

const char* falcon_acado_status_string(int status) {
  if (status == 0) {
    return "success";
  }
  if (status == -100) {
    return "not initialized or non-finite configuration";
  }
  if (status == -101) {
    return "invalid input limits";
  }
  if (status == -102) {
    return "invalid reset state";
  }
  if (status == -103) {
    return "invalid solve input";
  }
  return acado_getErrorString(status);
}

