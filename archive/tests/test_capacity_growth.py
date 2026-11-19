"""Verify functional equality and usable extra width, without real training."""

import unittest

import torch

from controller_architecture_study.models.plain_mlp import build_plain_mlp
from controller_architecture_study.prepare_capacity_growth import grow_model


class CapacityGrowthTests(unittest.TestCase):
    def test_width_depth_and_combined_growth_preserve_outputs(self):
        torch.manual_seed(4)
        source = build_plain_mlp(torch, input_dim=12, output_dim=4,
                                 hidden_layers=3, hidden_width=8)
        with torch.no_grad():
            for layer in source:
                if isinstance(layer, torch.nn.Linear):
                    layer.bias.uniform_(-0.5, 0.5)
        original = {key: value.clone() for key, value in source.state_dict().items()}
        inputs = torch.randn(80, 12) * 3
        for depth, width in ((3, 16), (5, 8), (5, 16)):
            with self.subTest(depth=depth, width=width):
                grown = grow_model(source, hidden_layers=depth, hidden_width=width)
                torch.testing.assert_close(grown(inputs), source(inputs), atol=2e-5, rtol=2e-5)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_duplicate_neurons_can_receive_different_gradients(self):
        torch.manual_seed(7)
        source = build_plain_mlp(torch, input_dim=12, output_dim=4,
                                 hidden_layers=1, hidden_width=8)
        grown = grow_model(source, hidden_layers=1, hidden_width=24)
        grown(torch.randn(80, 12)).square().mean().backward()
        found = False
        for duplicate in range(8, 24):
            for original in range(8):
                if torch.equal(grown[0].weight[duplicate], grown[0].weight[original]):
                    found |= not torch.allclose(grown[0].weight.grad[duplicate], grown[0].weight.grad[original])
        self.assertTrue(found, "equal outgoing splits would lock duplicates into symmetry")

    def test_shrinking_is_rejected(self):
        source = build_plain_mlp(torch, input_dim=12, output_dim=4,
                                 hidden_layers=3, hidden_width=8)
        for depth, width in ((2, 8), (3, 4)):
            with self.assertRaises(ValueError):
                grow_model(source, hidden_layers=depth, hidden_width=width)


if __name__ == "__main__":
    unittest.main()
