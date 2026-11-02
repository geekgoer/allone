"""Plain fully connected ReLU controller with explicit hidden-layer count."""

from __future__ import annotations


def build_plain_mlp(torch, *, input_dim: int, output_dim: int, hidden_layers: int, hidden_width: int):
    if min(input_dim, output_dim, hidden_layers, hidden_width) < 1:
        raise ValueError("all model dimensions must be positive")
    layers = []
    source = input_dim
    for _ in range(hidden_layers):
        linear = torch.nn.Linear(source, hidden_width)
        torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
        torch.nn.init.zeros_(linear.bias)
        layers.extend((linear, torch.nn.ReLU()))
        source = hidden_width
    output = torch.nn.Linear(source, output_dim)
    torch.nn.init.xavier_uniform_(output.weight)
    torch.nn.init.zeros_(output.bias)
    layers.append(output)
    return torch.nn.Sequential(*layers)
