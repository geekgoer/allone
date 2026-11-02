# Historical 5 m/s 12-D physical benchmark v1

This folder freezes a new held-out AirSim benchmark for the six historical
Track-P controllers. It does not overwrite the original test-set MSE and must
not be presented as the paper's Table II protocol.

The benchmark has two complementary tracks:

1. `run_common_state_benchmark.py`: an MPC teacher flies each scenario once.
   Every recorded 12-D state is then evaluated by all six DNNs offline, giving
   a fair common-state MSE against exactly the same teacher labels.
2. `run_closed_loop_benchmark.py`: each DNN flies every frozen scenario from a
   reset state. This measures actual gate-passage success and deadline misses.

The scenario list, model paths, fixed 5 m/s speed, controller period and gate
geometry are frozen in `benchmark_config.json`. Do not edit them after seeing
results. Create a versioned `benchmark_config_v2.json` for future changes.

Run `pilot` first. It contains S01 and S06 (straight/easy and displaced/hard).
If the teacher pilot is valid, run `full`, which contains all six scenarios.

