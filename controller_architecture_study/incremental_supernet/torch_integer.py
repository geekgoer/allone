"""Independent frozen-dyadic evaluator: FP64 exact MAC, INT64 scaling/RNE.

Derives codes from checkpoint parameters, not the NumPy packed weight arrays.
FP64 MAC is exact because products, partial sums and biases are bounded to 48
bits; all subsequent multiplier/alignment operations use checked INT64.
"""

import torch
from torch.nn import functional as F

from .lsq_model import DYADIC_QUANT_CONTRACT


def check_bits(value, bits, name):
    if bool(torch.any(value < -(2**(bits-1)))) or bool(torch.any(value > 2**(bits-1)-1)):
        raise OverflowError(f"{name} exceeds signed {bits} bits")


def multiply(value, multiplier):
    multiplier = multiplier.to(torch.int64)
    if bool(torch.any(multiplier < 1)) or bool(torch.any(multiplier > 65535)):
        raise ValueError("invalid dyadic multiplier")
    check_bits(value, 63, "dyadic input")
    if bool(torch.any(value.abs() > torch.div(2**62-1, multiplier, rounding_mode="floor"))):
        raise OverflowError("dyadic product exceeds signed 63-bit working range")
    return value * multiplier


def rescale(value, source, target):
    shift = torch.as_tensor(source, device=value.device, dtype=torch.int64) - torch.as_tensor(
        target, device=value.device, dtype=torch.int64)
    if bool(torch.any(shift.abs() > 62)):
        raise OverflowError("integer alignment shift exceeds 62")
    left, right = shift.clamp_min(0), (-shift).clamp_min(0)
    check_bits(value, 63, "alignment input")
    bound = torch.bitwise_right_shift(value.new_tensor(2**62-1), left)
    if bool(torch.any(value.abs() > bound)):
        raise OverflowError("integer alignment exceeds signed 63-bit working range")
    quotient = torch.bitwise_right_shift(value, right)
    remainder = value - torch.bitwise_left_shift(quotient, right)
    half = torch.bitwise_left_shift(value.new_tensor(1), (right-1).clamp_min(0))
    increment = (right > 0) & ((remainder > half) | ((remainder == half) & ((quotient & 1) != 0)))
    return torch.bitwise_left_shift(quotient + increment, left)


def codes(model, name, layer, input_exponent):
    multiplier = getattr(model, name + "_weight_multiplier").to(torch.int64)
    base = getattr(model, name + "_weight_exponent").to(torch.int64)
    scale = torch.ldexp(multiplier.double(), base.to(torch.int32))
    weight = layer.weight.detach().double()
    if name.startswith("layer"):
        weight = weight * model.model.causal_mask.double()
    limit = model.weight_limit(name)
    weight = torch.round(weight / scale[:, None]).clamp(-limit, limit).to(torch.int64)
    bias_scale = torch.ldexp(scale, input_exponent.to(torch.int32))
    bias_float = torch.round(layer.bias.detach().double() / bias_scale)
    if not bool(torch.isfinite(bias_float).all()):
        raise ValueError("nonfinite bias codes")
    check_bits(bias_float, 48, "bias")
    return weight, bias_float.to(torch.int64), multiplier, base


def mac(value, weight, bias, weight_bits):
    bound = value.shape[1] * 32767 * (2**(weight_bits-1)-1) + int(bias.abs().max())
    if bound > 2**47-1:
        raise OverflowError("exact FP64 MAC requires a signed 48-bit conservative bound")
    return F.linear(value.double(), weight.double(), bias.double()).to(torch.int64)


@torch.inference_mode()
def evaluate(model, normalized, configurations):
    if model.quantization_contract != DYADIC_QUANT_CONTRACT or not bool(model.scales_frozen):
        raise ValueError("independent integer check requires a frozen v3 checkpoint")
    if normalized.ndim != 2 or normalized.shape[1] != model.input_dim or not bool(torch.isfinite(normalized).all()):
        raise ValueError("expected finite normalized observations")
    if not configurations or any(d not in model.depths or w not in model.widths for d, w in configurations):
        raise ValueError("unsupported integer configurations")
    depth, width = max(d for d, _ in configurations), max(w for _, w in configurations)
    exps = model.activation_exponents.to(torch.int64)
    h = torch.round(torch.ldexp(normalized.float().double(), (-exps[0]).to(torch.int32))).clamp(-32767, 32767).to(torch.int64)
    features = {}
    for d in range(1, depth+1):
        name, layer = ("stem", model.model.stem) if d == 1 else (f"layer{d}", model.model.blocks[d-2])
        weight, bias, multiplier, base = codes(model, name, layer, exps[d-1])
        branch = mac(h, weight[:width, :h.shape[1]], bias[:width], 8)
        branch = multiply(branch, multiplier[:width])
        source = exps[d-1] + base[:width]
        if d > 1:
            common = torch.minimum(source, exps[d-1])
            branch = rescale(branch, source, common) + rescale(h, exps[d-1], common)
            check_bits(branch, 63, "residual")
            source = common
        h = rescale(branch.clamp_min(0), source, exps[d]).clamp(0, 32767)
        if d in model.depths:
            features[d] = h
    outputs, accumulators = {}, {}
    g = model.group_width
    for d in sorted({d for d, _ in configurations}):
        head = model.model.heads[model.depths.index(d)]
        weight, bias, multiplier, base = codes(model, f"head{d}", head, exps[d])
        h = features[d]
        # Exact integer group MACs; sum in INT64, add bias once, multiply once.
        contributions = torch.einsum("bgi,ogi->bgo", h.double().reshape(-1, width//g, g),
                                     weight[:, :width].double().reshape(4, width//g, g)).to(torch.int64)
        prefix = contributions.cumsum(1) + bias
        check_bits(prefix, 48, "head prefix")
        scaled = multiply(prefix, multiplier)
        normalized_output = torch.ldexp(scaled.double(), (exps[d]+base).to(torch.int32)).float()
        for key in configurations:
            if key[0] == d:
                outputs[key] = normalized_output[:, key[1]//g-1]
                accumulators[key] = scaled[:, key[1]//g-1]
    return outputs, accumulators
