from elastic_supernet_test_a.analysis.compute_cost import mlp_active_parameters, mlp_macs


def test_known_macs() -> None:
    assert mlp_macs(12, (16, 16), 4) == 12 * 16 + 16 * 16 + 16 * 4


def test_parameters_include_biases() -> None:
    expected = 12 * 16 + 16 * 16 + 16 * 4 + 16 + 16 + 4
    assert mlp_active_parameters(12, (16, 16), 4) == expected

