"""Fixed-scale W8A16 QAT; one causal feature encoding for every depth/width."""

import torch
from torch import nn
from torch.nn import functional as F

from .common import grid


QUANT_CONTRACT = "causal_w8a16_pow2_rne_v1"
MIXED_QUANT_CONTRACT = "causal_w8_head16_a16_pow2_rne_v2"


def exponent_for(maximum, limit, margin=1.25):
    # Powers of two avoid a separate, potentially overflowing integer multiplier.
    return torch.ceil(torch.log2((maximum * margin / limit).clamp_min(2.**-30))).to(torch.int16)


def fake_quant(value, exponent, limit):
    scale = torch.pow(2., exponent.to(device=value.device, dtype=value.dtype))
    clipped = (value / scale).clamp(-limit, limit)
    rounded = clipped + (torch.round(clipped) - clipped).detach()
    return rounded * scale


class QuantizedSupernet(nn.Module):
    def __init__(self, model, activation_exponents, *, fold_residual=True, weight_margin=1.25,
                 head_weight_bits=8):
        super().__init__()
        if head_weight_bits not in (8, 16):
            raise ValueError("head weights must use 8 or 16 bits")
        self.head_weight_bits = head_weight_bits
        self.quantization_contract = MIXED_QUANT_CONTRACT if head_weight_bits == 16 else QUANT_CONTRACT
        self.model = model
        self.depths, self.widths = model.depths, model.widths
        self.max_depth, self.max_width = model.max_depth, model.max_width
        self.input_dim, self.group_width = model.input_dim, model.group_width
        self.original_residual_scale = model.residual_scale
        if fold_residual:
            with torch.no_grad():
                for block in model.blocks:
                    block.weight.mul_(model.residual_scale)
                    block.bias.mul_(model.residual_scale)
        self.register_buffer("activation_exponents", torch.as_tensor(activation_exponents, dtype=torch.int16))
        for name, layer in self.layers():
            self.register_buffer(name + "_weight_exponent",
                                 exponent_for(layer.weight.detach().abs().amax(1), self.weight_limit(name), weight_margin))

    def weight_limit(self, name):
        return 2**(self.head_weight_bits-1)-1 if name.startswith("head") else 127

    def layers(self):
        return [("stem", self.model.stem)] + [
            (f"layer{d}", layer) for d, layer in enumerate(self.model.blocks, 2)] + [
            (f"head{d}", head) for d, head in zip(self.depths, self.model.heads)]

    def linear(self, x, name, layer, input_exponent, width=None):
        exponent = getattr(self, name + "_weight_exponent")
        weight = layer.weight
        if name.startswith("layer"):
            weight = weight * self.model.causal_mask
        weight = fake_quant(weight, exponent[:, None], self.weight_limit(name))
        bias = fake_quant(layer.bias, exponent + input_exponent, 2**47 - 1)
        if width is not None:
            if name.startswith("head"):
                weight = weight[:, :width]
            else:
                weight, bias = weight[:width, :x.shape[1]], bias[:width]
        return F.linear(x, weight, bias)

    def features(self, x):
        exponents = self.activation_exponents
        h = fake_quant(x, exponents[0], 32767)
        h = fake_quant(torch.relu(self.linear(h, "stem", self.model.stem, exponents[0])), exponents[1], 32767)
        result = {1: h}
        for d, block in enumerate(self.model.blocks, 2):
            # Alpha is in W/b, NOT in the identity branch. Round once after addition.
            h = fake_quant(torch.relu(h + self.linear(h, f"layer{d}", block, exponents[d-1])),
                           exponents[d], 32767)
            if d in self.depths:
                result[d] = h
        return result

    def outputs(self, features, configs):
        result = {}
        for d in sorted({d for d, _ in configs}):
            head = self.model.heads[self.depths.index(d)]
            exponent = getattr(self, f"head{d}_weight_exponent")
            weight = fake_quant(head.weight, exponent[:, None], self.weight_limit(f"head{d}"))
            bias = fake_quant(head.bias, exponent + self.activation_exponents[d], 2**47 - 1)
            h = features[d].reshape(-1, len(self.widths), self.group_width)
            weight = weight.reshape(4, len(self.widths), self.group_width)
            prefixes = torch.einsum("bgi,ogi->bgo", h, weight).cumsum(1) + bias
            for depth, width in configs:
                if depth == d:
                    result[(d, width)] = prefixes[:, width // self.group_width - 1]
        return result

    def forward(self, x, depth=0, width=0):
        depth, width = depth or self.max_depth, width or self.max_width
        if depth not in self.depths or width not in self.widths:
            raise ValueError("unsupported depth/width")
        exponents = self.activation_exponents
        h = fake_quant(x, exponents[0], 32767)
        h = fake_quant(torch.relu(self.linear(h, "stem", self.model.stem, exponents[0], width)), exponents[1], 32767)
        for d, block in enumerate(self.model.blocks, 2):
            if d <= depth:
                h = fake_quant(torch.relu(h + self.linear(h, f"layer{d}", block, exponents[d-1], width)),
                               exponents[d], 32767)
        return self.linear(h, f"head{depth}", self.model.heads[self.depths.index(depth)], exponents[depth], width)


@torch.inference_mode()
def calibrate_activations(model, x, batch_size, margin=1.25):
    """TRAIN-only maximum-path calibration; freeze these scales for the whole grid."""
    device = next(model.parameters()).device
    model.eval()
    maxima = torch.zeros(model.max_depth + 1, device=device)
    for start in range(0, len(x), batch_size):
        h = x[start:start+batch_size].to(device)
        maxima[0] = torch.maximum(maxima[0], h.abs().max())
        h = torch.relu(model.stem(h))
        maxima[1] = torch.maximum(maxima[1], h.abs().max())
        for d, block in enumerate(model.blocks, 2):
            h = torch.relu(h + model.residual_scale * F.linear(h, block.weight * model.causal_mask, block.bias))
            maxima[d] = torch.maximum(maxima[d], h.abs().max())
        print(f"Calibration: {min(start+batch_size, len(x))}/{len(x)} TRAIN rows", flush=True)
    return exponent_for(maxima, 32767, margin).cpu(), maxima.cpu()


def adjacency(model):
    configs = grid(model)
    pairs = []
    for d in model.depths:
        pairs.extend(((d, a), (d, b), "width") for a, b in zip(model.widths, model.widths[1:]))
    for w in model.widths:
        pairs.extend(((a, w), (b, w), "depth") for a, b in zip(model.depths, model.depths[1:]))
    return configs, pairs


def common_function_loss(outputs, target, teacher, family_thresholds, families, model, config,
                         reference_outputs=None):
    configs, pairs = adjacency(model)
    prediction = torch.stack([outputs[key] for key in configs])
    label = (prediction - target.unsqueeze(0)).square().mean()
    kd = (prediction - teacher.unsqueeze(0)).square().mean()
    maximum = outputs[(model.max_depth, model.max_width)]
    if config.get("reference_preservation", False):
        if reference_outputs is None:
            raise ValueError("configuration preservation requires frozen FP32 outputs")
        reference = torch.stack([reference_outputs[key].detach() for key in configs])
        configuration_kd = (prediction - reference).square().mean()
        index = {key: i for i, key in enumerate(configs)}
        a, b = [index[a] for a, _, _ in pairs], [index[b] for _, b, _ in pairs]
        # Preserve legitimate FP32 differences; no moving confidence gate and no
        # demand that accurate configurations all emit the same control command.
        switch_delta = ((prediction[a] - prediction[b]) - (reference[a] - reference[b])).square().mean()
        loss = (label + config["kd_weight"] * kd
                + config["configuration_kd_weight"] * configuration_kd
                + config["switch_delta_weight"] * switch_delta
                + config["maximum_label_weight"] * (maximum - target).square().mean()
                + config["maximum_anchor_weight"] * (maximum - teacher).square().mean())
        return loss, {"label": label, "kd": kd, "configuration_kd": configuration_kd,
                      "consistency": switch_delta, "switch_delta": switch_delta}
    index = {key: i for i, key in enumerate(configs)}
    small = prediction[[index[a] for a, _, _ in pairs]]
    large = prediction[[index[b] for _, b, _ in pairs]]
    small_error = (small - target).square().mean(-1)
    large_error = (large - target).square().mean(-1)
    # Label-informed gate: do not pull a better small model towards a worse large one.
    gate = ((large_error.detach() <= small_error.detach()) &
            (large_error.detach() <= family_thresholds[families].unsqueeze(0)))
    distance = (small - large.detach()).square().mean(-1)
    consistency = (distance * gate).sum() / gate.sum().clamp_min(1)
    loss = (label + config["kd_weight"] * kd + config["consistency_weight"] * consistency
            + config["maximum_label_weight"] * (maximum - target).square().mean()
            + config["maximum_anchor_weight"] * (maximum - teacher).square().mean())
    return loss, {"label": label, "kd": kd, "consistency": consistency,
                  "consistency_gate_fraction": gate.float().mean()}
