"""Shared-weight width-elastic MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .elastic_linear import elastic_linear


@dataclass(frozen=True)
class SubnetConfig:
    widths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.widths or any(width < 1 for width in self.widths):
            raise ValueError("subnet widths must be a non-empty positive tuple")

    @property
    def label(self) -> str:
        if len(set(self.widths)) == 1:
            return f"uniform_{self.widths[0]}"
        return "layers_" + "-".join(str(width) for width in self.widths)

    @classmethod
    def uniform(cls, width: int, hidden_layers: int) -> "SubnetConfig":
        return cls((int(width),) * int(hidden_layers))


class ElasticMLP:
    """One maximum parameter set supporting prefix-sliced hidden widths."""

    def __init__(self, torch: Any, *, input_dim: int, output_dim: int, hidden_layers: int, max_width: int):
        class _Module(torch.nn.Module):
            def __init__(inner_self) -> None:
                super().__init__()
                linears = [torch.nn.Linear(input_dim, max_width)]
                linears.extend(torch.nn.Linear(max_width, max_width) for _ in range(hidden_layers - 1))
                linears.append(torch.nn.Linear(max_width, output_dim))
                inner_self.linears = torch.nn.ModuleList(linears)

            def forward(inner_self, x: Any, widths: tuple[int, ...]) -> Any:
                if len(widths) != hidden_layers:
                    raise ValueError(f"expected {hidden_layers} active hidden widths")
                value = x
                active_in = input_dim
                for index, active_out in enumerate(widths):
                    if active_out > max_width:
                        raise ValueError("subnet width exceeds max_width")
                    layer = inner_self.linears[index]
                    value = elastic_linear(
                        torch,
                        value,
                        layer.weight,
                        layer.bias,
                        active_in=active_in,
                        active_out=active_out,
                    )
                    value = torch.nn.functional.relu(value)
                    active_in = active_out
                output = inner_self.linears[-1]
                return elastic_linear(
                    torch,
                    value,
                    output.weight,
                    output.bias,
                    active_in=active_in,
                    active_out=output_dim,
                )

        self.torch = torch
        self.module = _Module()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = int(hidden_layers)
        self.max_width = int(max_width)

    @property
    def max_config(self) -> SubnetConfig:
        return SubnetConfig.uniform(self.max_width, self.hidden_layers)

    def to(self, device: str) -> "ElasticMLP":
        self.module.to(device)
        return self

    def train(self, mode: bool = True) -> "ElasticMLP":
        self.module.train(mode)
        return self

    def eval(self) -> "ElasticMLP":
        return self.train(False)

    def parameters(self) -> Iterable[Any]:
        return self.module.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        return self.module.load_state_dict(state_dict, strict=True)

    def __call__(self, x: Any, config: SubnetConfig) -> Any:
        return self.module(x, config.widths)

    @classmethod
    def from_fixed_controller(cls, fixed: Any) -> "ElasticMLP":
        torch = fixed.torch
        info = fixed.info
        result = cls(
            torch,
            input_dim=info.input_dim,
            output_dim=info.output_dim,
            hidden_layers=info.hidden_layers,
            max_width=info.width,
        )
        fixed_linears = [module for module in fixed.model.modules() if isinstance(module, torch.nn.Linear)]
        with torch.no_grad():
            for source, target in zip(fixed_linears, result.module.linears, strict=True):
                target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
        return result


class ElasticController:
    """Physical-unit wrapper sharing normalization with the fixed baseline."""

    def __init__(self, elastic: ElasticMLP, fixed: Any):
        self.elastic = elastic
        self.fixed = fixed

    def to(self, device: str) -> "ElasticController":
        self.elastic.to(device)
        self.fixed.to(device)
        return self

    def eval(self) -> "ElasticController":
        self.elastic.eval()
        return self

    def normalized_forward(self, normalized_x: Any, config: SubnetConfig) -> Any:
        return self.elastic(normalized_x, config)

    def __call__(self, physical_x: Any, config: SubnetConfig) -> Any:
        normalized_x = (physical_x - self.fixed.x_mean) / self.fixed.x_std
        normalized_u = self.normalized_forward(normalized_x, config)
        return normalized_u * self.fixed.u_std + self.fixed.u_mean

