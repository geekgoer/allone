"""LSQ-style shared row scales, with deployed 16-bit dyadic multipliers in QAT."""

import torch
from torch import nn
from torch.nn import functional as F

from .qat_model import QuantizedSupernet


DYADIC_QUANT_CONTRACT = "causal_w8_head16_a16_lsq_dyadic_v3"


def round_ste(value):
    return value + (torch.round(value) - value).detach()


class LearnedScaleSupernet(QuantizedSupernet):
    multiplier_bits = 16

    def __init__(self, model, activation_exponents, *, fold_residual=True, weight_margin=1.,
                 head_weight_bits=16):
        if head_weight_bits != 16:
            raise ValueError("dyadic v3 requires W16 heads")
        super().__init__(model, activation_exponents, fold_residual=fold_residual,
                         weight_margin=weight_margin, head_weight_bits=head_weight_bits)
        self.quantization_contract = DYADIC_QUANT_CONTRACT
        self.log_steps = nn.ParameterDict()
        self.register_buffer("scales_frozen", torch.tensor(False))
        for name, layer in self.layers():
            step = (layer.weight.detach().abs().amax(1)*weight_margin/self.weight_limit(name)).clamp_min(2.**-30)
            # Initial multiplier in [16384,32768), leaving room for scale updates.
            base = torch.floor(torch.log2(step)).to(torch.int16) - (self.multiplier_bits-2)
            getattr(self, name + "_weight_exponent").copy_(base)
            self.log_steps[name] = nn.Parameter(torch.log(step))
            self.register_buffer(name + "_weight_multiplier", torch.zeros(len(step), dtype=torch.int32))

    def freeze_scales(self):
        if not bool(self.scales_frozen):
            for name, _ in self.layers():
                # Canonical CPU FP64 encoding; CUDA exp rounding cannot alter an
                # exported multiplier once this integer buffer has been frozen.
                step = self.log_steps[name].detach().cpu().double().exp()
                base = getattr(self, name + "_weight_exponent").detach().cpu().double()
                multiplier = torch.round(step / torch.pow(2., base)).clamp(1, 65535).to(torch.int32)
                getattr(self, name + "_weight_multiplier").copy_(multiplier)
            self.scales_frozen.fill_(True)
        self.log_steps.requires_grad_(False)

    def thaw_scales(self):
        self.scales_frozen.fill_(False)
        self.log_steps.requires_grad_(True)

    def weight_scale(self, name):
        base = getattr(self, name + "_weight_exponent").float()
        unit = torch.pow(2., base)
        if bool(self.scales_frozen):
            return getattr(self, name + "_weight_multiplier").float() * unit
        step = self.log_steps[name].exp()
        layer = dict(self.layers())[name]
        if name.startswith("layer"):
            count = self.model.causal_mask.sum(1)
        else:
            count = step.new_full(step.shape, layer.weight.shape[1])
        gradient_factor = torch.rsqrt(count * self.weight_limit(name))
        step = step.detach() + (step - step.detach()) * gradient_factor
        # Quantize the step itself throughout QAT, rather than projecting it only
        # after training. One encoding is shared by all width/depth configurations.
        return round_ste((step / unit).clamp(1, 65535)) * unit

    def codes(self, name, layer, input_exponent):
        scale = self.weight_scale(name)
        weight = layer.weight * self.model.causal_mask if name.startswith("layer") else layer.weight
        limit = self.weight_limit(name)
        code = round_ste((weight / scale[:, None]).clamp(-limit, limit))
        bias_scale = scale.double() * 2.**int(input_exponent)
        bias_code = round_ste((layer.bias.double() / bias_scale).clamp(-(2**47-1), 2**47-1))
        return code, bias_code, scale

    def linear(self, x, name, layer, input_exponent, width=None):
        code, bias_code, scale = self.codes(name, layer, input_exponent)
        head = name.startswith("head")
        weight = code.double() * scale.double()[:, None] if head else code * scale[:, None]
        bias = bias_code * scale.double() * 2.**int(input_exponent)
        if not head:
            bias = bias.to(x.dtype)
        if width is not None:
            if name.startswith("head"):
                weight = weight[:, :width]
            else:
                weight, bias = weight[:width, :x.shape[1]], bias[:width]
        return F.linear(x.double() if head else x, weight, bias).to(x.dtype)

    def outputs(self, features, configs):
        result = {}
        for d in sorted({d for d, _ in configs}):
            head = self.model.heads[self.depths.index(d)]
            code, bias_code, scale = self.codes(f"head{d}", head, self.activation_exponents[d])
            # W16 times a 16-bit scale mantissa need more than FP32's 24
            # significant bits; the small readout uses FP64 accumulation.
            weight = code.double() * scale.double()[:, None]
            bias = bias_code * scale.double() * 2.**int(self.activation_exponents[d])
            h = features[d].double().reshape(-1, len(self.widths), self.group_width)
            weight = weight.reshape(4, len(self.widths), self.group_width)
            prefixes = torch.einsum("bgi,ogi->bgo", h, weight).cumsum(1) + bias
            for depth, width in configs:
                if depth == d:
                    result[(d, width)] = prefixes[:, width // self.group_width - 1].float()
        return result
