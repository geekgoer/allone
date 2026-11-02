#include "falcon_acado_c_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
  const int repetitions = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1000;
  if (falcon_acado_initialize() != 0) {
    std::cerr << "failed to initialize ACADO solver\n";
    return 1;
  }

  double state[10] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double references[21 * 10] = {};
  double inputs[21 * 4] = {};
  for (int node = 0; node < 21; ++node) {
    references[node * 10 + 0] = std::min(2.0, node * falcon_acado_prediction_dt());
    references[node * 10 + 3] = 1.0;
    references[node * 10 + 7] = node < 20 ? 1.0 : 0.0;
    inputs[node * 4 + 0] = 9.8066;
  }

  double output[4] = {};
  double diagnostics[3] = {};
  if (falcon_acado_reset(state) != 0 ||
      falcon_acado_solve(state, references, inputs, output, diagnostics) != 0) {
    std::cerr << "initial solve failed\n";
    return 2;
  }

  std::vector<double> milliseconds;
  milliseconds.reserve(repetitions);
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    const double phase = 0.013 * static_cast<double>(iteration);
    state[0] = 0.15 * std::sin(phase);
    state[1] = 0.10 * std::cos(0.7 * phase);
    state[2] = 0.05 * std::sin(0.5 * phase);
    state[7] = 0.20 * std::cos(phase);
    state[8] = 0.10 * std::sin(0.7 * phase);
    const auto start = std::chrono::steady_clock::now();
    const int status = falcon_acado_solve(state, references, inputs, output, diagnostics);
    const auto end = std::chrono::steady_clock::now();
    if (status != 0) {
      std::cerr << "solve failed: " << status << " "
                << falcon_acado_status_string(status) << "\n";
      return 3;
    }
    milliseconds.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::sort(milliseconds.begin(), milliseconds.end());
  const double mean = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
                      static_cast<double>(milliseconds.size());
  const auto percentile = [&](double value) {
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(value * static_cast<double>(milliseconds.size()))) - 1;
    return milliseconds[std::min(index, milliseconds.size() - 1)];
  };

  std::cout << std::fixed << std::setprecision(6)
            << "backend=official_rpg_mpc_acado_qpoases\n"
            << "horizon=" << FALCON_ACADO_HORIZON << "\n"
            << "prediction_dt_s=" << falcon_acado_prediction_dt() << "\n"
            << "repetitions=" << repetitions << "\n"
            << "mean_ms=" << mean << "\n"
            << "p95_ms=" << percentile(0.95) << "\n"
            << "p99_ms=" << percentile(0.99) << "\n"
            << "max_ms=" << milliseconds.back() << "\n"
            << "output=" << output[0] << "," << output[1] << ","
            << output[2] << "," << output[3] << "\n"
            << "kkt=" << diagnostics[0] << "\n"
            << "objective=" << diagnostics[1] << "\n"
            << "qp_nwsr=" << diagnostics[2] << "\n";
  return 0;
}
