"""Causal channel groups: later groups cannot change an earlier group's features."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


NETWORK_CONTRACT = "causal_group_residual_depth_width_v2"


class CausalSupernet(nn.Module):
    def __init__(self, input_dim=237, output_dim=4, depths=None, widths=None, group_width=128):
        super().__init__()
        self.depths = list(depths if depths is not None else range(1, 34, 2))
        self.widths = list(widths if widths is not None else range(128, 1025, 128))
        if (sorted(set(self.depths)) != self.depths or self.depths[0] < 1
                or self.widths != list(range(group_width, self.widths[-1] + 1, group_width))):
            raise ValueError("depths must increase; widths must be consecutive channel groups")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.group_width = group_width
        self.max_depth, self.max_width = self.depths[-1], self.widths[-1]
        self.residual_scale = 1.0 / math.sqrt(max(self.max_depth - 1, 1))
        self.stem = nn.Linear(input_dim, self.max_width)
        self.blocks = nn.ModuleList(nn.Linear(self.max_width, self.max_width)
                                    for _ in range(self.max_depth - 1))
        self.heads = nn.ModuleList(nn.Linear(self.max_width, output_dim) for _ in self.depths)
        groups = torch.arange(self.max_width) // group_width
        self.register_buffer("causal_mask", (groups[:, None] >= groups[None, :]).float())
        for layer in (self.stem, *self.blocks):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        self.enforce_mask()

    @torch.jit.ignore
    def enforce_mask(self):
        with torch.no_grad():
            for layer in self.blocks:
                layer.weight.mul_(self.causal_mask)

    def forward(self, x: torch.Tensor, depth: int = 0, width: int = 0) -> torch.Tensor:
        # Zero means maximum. This method executes the actual selected subgraph.
        if depth == 0:
            depth = self.max_depth
        if width == 0:
            width = self.max_width
        if depth not in self.depths or width not in self.widths:
            raise ValueError("unsupported depth/width")
        h = torch.relu(F.linear(x, self.stem.weight[:width], self.stem.bias[:width]))
        for i, block in enumerate(self.blocks):
            if i < depth - 1:
                branch = F.linear(h, block.weight[:width, :width] * self.causal_mask[:width, :width],
                                  block.bias[:width])
                h = torch.relu(h + self.residual_scale * branch)
        output = x.new_zeros((x.size(0), self.output_dim))
        for i, head in enumerate(self.heads):
            if self.depths[i] == depth:
                output = F.linear(h, head.weight[:, :width], head.bias)
        return output

    @torch.jit.ignore
    def features(self, x):
        """One maximum forward for training/evaluation; NOT a cheap small inference."""
        h = torch.relu(self.stem(x))
        result = {1: h}
        for d, block in enumerate(self.blocks, 2):
            h = torch.relu(h + self.residual_scale * F.linear(h, block.weight * self.causal_mask, block.bias))
            if d in self.depths:
                result[d] = h
        return result

    @torch.jit.ignore
    def outputs(self, features, configs):
        result = {}
        for d in sorted({d for d, _ in configs}):
            head = self.heads[self.depths.index(d)]
            h = features[d].reshape(-1, len(self.widths), self.group_width)
            weight = head.weight.reshape(self.output_dim, len(self.widths), self.group_width)
            # Bias ONCE, after the cumulative sum of channel-group contributions.
            prefixes = torch.einsum("bgi,ogi->bgo", h, weight).cumsum(dim=1) + head.bias
            for depth, width in configs:
                if depth == d:
                    result[(d, width)] = prefixes[:, width // self.group_width - 1]
        return result

    @torch.jit.ignore
    def initialize_from_teacher(self, teacher):
        if (teacher.stem.weight.shape != self.stem.weight.shape
                or len(teacher.blocks) != len(self.blocks)):
            raise ValueError("teacher and maximum supernet dimensions disagree")
        self.stem.load_state_dict(teacher.stem.state_dict())
        for dst, src in zip(self.blocks, teacher.blocks):
            dst.load_state_dict(src.state_dict())
        for head in self.heads:
            head.load_state_dict(teacher.head.state_dict())
        self.enforce_mask()

    @torch.jit.ignore
    def complexity(self, depth, width):
        m, g = width // self.group_width, self.group_width
        macs = self.input_dim * width + (depth - 1) * g * g * m * (m + 1) // 2 + width * self.output_dim
        return {"effective_macs": macs, "effective_parameters": macs + depth * width + self.output_dim,
                "allocated_parameters": sum(p.numel() for p in self.parameters()),
                "dense_mask_gpu_is_not_sparse_acceleration": True}


class PhysicalController(nn.Module):
    """Raw 237D input -> physical controls; normalize/denormalize exactly once."""
    def __init__(self, model, normalization):
        super().__init__()
        self.model = model
        for key in ("x_mean", "x_std", "u_mean", "u_std"):
            self.register_buffer(key, torch.as_tensor(normalization[key], dtype=torch.float32))

    def forward(self, raw: torch.Tensor, depth: int = 0, width: int = 0) -> torch.Tensor:
        return self.model((raw - self.x_mean) / self.x_std, depth, width) * self.u_std + self.u_mean


@dataclass
class IncrementalContext:
    """Inference-only cache bound to one input and one unchanged model instance."""
    model: CausalSupernet
    normalized_input: torch.Tensor
    sample_id: str
    model_version: str
    activations: dict = field(default_factory=dict)
    frontier: list = field(init=False)

    def __post_init__(self):
        if self.model.training:
            raise ValueError("incremental inference requires model.eval()")
        if self.normalized_input.ndim != 2 or self.normalized_input.shape[1] != self.model.input_dim:
            raise ValueError("expected a batch of normalized input vectors")
        with torch.inference_mode(False):
            self.normalized_input = self.normalized_input.detach().clone()
        self.frontier = [0] * len(self.model.widths)
        self._input_version = self.normalized_input._version
        self._versions = tuple(p._version for p in (*self.model.parameters(), *self.model.buffers()))

    @torch.inference_mode()
    def grow(self, depth, width, *, sample_id, model_version):
        if (sample_id != self.sample_id or model_version != self.model_version
                or self._input_version != self.normalized_input._version
                or self._versions != tuple(p._version for p in (*self.model.parameters(), *self.model.buffers()))):
            raise ValueError("stale context: different sample, model version, or mutated weights")
        if self.model.training or depth not in self.model.depths or width not in self.model.widths:
            raise ValueError("invalid inference configuration")
        g = self.model.group_width
        computed = []
        # Complete earlier groups first. Each later group only reads those groups.
        for group in range(width // g):
            lo, hi = group * g, (group + 1) * g
            for d in range(self.frontier[group] + 1, depth + 1):
                if d == 1:
                    h = torch.relu(F.linear(self.normalized_input, self.model.stem.weight[lo:hi],
                                           self.model.stem.bias[lo:hi]))
                else:
                    previous = torch.cat([self.activations[(d - 1, j)] for j in range(group + 1)], dim=1)
                    block = self.model.blocks[d - 2]
                    branch = F.linear(previous, block.weight[lo:hi, :hi], block.bias[lo:hi])
                    h = torch.relu(self.activations[(d - 1, group)] + self.model.residual_scale * branch)
                self.activations[(d, group)] = h
                self.frontier[group] = d
                computed.append((d, group))
        head = self.model.heads[self.model.depths.index(depth)]
        output = head.bias.unsqueeze(0).expand(self.normalized_input.size(0), -1).clone()
        for group in range(width // g):
            output.add_(F.linear(self.activations[(depth, group)], head.weight[:, group*g:(group+1)*g]))
        return output, computed

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.activations.values())
