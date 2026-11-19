def test_elastic_max_matches_synthetic_fixed_model() -> None:
    import torch

    from elastic_supernet_test_a.models.elastic_mlp import ElasticMLP

    fixed = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )

    class Info:
        input_dim = 3
        output_dim = 2
        hidden_layers = 2
        width = 8

    class Wrapper:
        pass

    wrapper = Wrapper()
    wrapper.torch = torch
    wrapper.model = fixed
    wrapper.info = Info()
    elastic = ElasticMLP.from_fixed_controller(wrapper)
    x = torch.randn(17, 3)
    torch.testing.assert_close(fixed(x), elastic(x, elastic.max_config), rtol=0.0, atol=0.0)

