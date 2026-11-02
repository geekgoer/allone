"""NumPy-only bit-exact reference for the shared W8A16 deployment package."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


QUANT_CONTRACT = "causal_w8a16_pow2_rne_v1"
MIXED_QUANT_CONTRACT = "causal_w8_head16_a16_pow2_rne_v2"
DYADIC_QUANT_CONTRACT = "causal_w8_head16_a16_lsq_dyadic_v3"


def multiply_checked(values, multiplier):
    values = np.asarray(values, dtype=np.int64)
    multiplier = np.asarray(multiplier, dtype=np.int64)
    if np.any(multiplier < 1) or np.any(multiplier > 65535):
        raise ValueError("dyadic multiplier must be in [1,65535]")
    check_range(values, 63, "dyadic multiplication input")
    if np.any(np.abs(values) > (2**62-1)//multiplier):
        raise OverflowError("dyadic product exceeds signed 63-bit working range")
    return values * multiplier


def check_range(values, bits, name):
    if np.any(values < -(2**(bits-1))) or np.any(values > 2**(bits-1)-1):
        raise OverflowError(f"{name} exceeds signed {bits}-bit range")


def rescale(values, source_exponent, target_exponent):
    """Power-of-two scaling, round-to-nearest ties-to-even, checked INT64 intermediates."""
    values = np.asarray(values, dtype=np.int64)
    shifts = np.asarray(source_exponent, dtype=np.int64) - np.asarray(target_exponent, dtype=np.int64)
    if np.any(np.abs(shifts) > 62):
        raise OverflowError("requantization shift exceeds the INT64 reference contract")
    left, right = np.maximum(shifts, 0), np.maximum(-shifts, 0)
    # Check BEFORE shifting; a check after overflow cannot recover lost information.
    check_range(values, 63, "requantization input")
    bounds = np.right_shift(np.int64(2**62-1), left)
    if np.any(np.abs(values) > bounds):
        raise OverflowError("requantization alignment exceeds signed 63-bit working range")
    quotient = np.right_shift(values, right)  # arithmetic shift: floor, including negatives
    remainder = values - np.left_shift(quotient, right)
    half = np.left_shift(np.int64(1), np.maximum(right-1, 0))
    increment = (right > 0) & ((remainder > half) | ((remainder == half) & ((quotient & 1) != 0)))
    return np.left_shift(quotient + increment, left)


def mac(x, weight, bias, weight_bits=8):
    x, weight, bias = np.asarray(x), np.asarray(weight), np.asarray(bias, dtype=np.int64)
    if weight_bits not in (8, 16):
        raise ValueError("unsupported MAC weight precision")
    limit = 2**(weight_bits-1)-1
    if np.any(np.abs(x.astype(np.int64)) > 32767) or np.any(np.abs(weight.astype(np.int64)) > limit):
        raise ValueError(f"MAC requires symmetric A16/W{weight_bits} values")
    check_range(bias, 48, "bias")
    bound = x.shape[1] * 32767 * limit + int(np.max(np.abs(bias), initial=0))
    if bound > 2**47-1:
        raise OverflowError("MAC worst-case bound exceeds 48 bits")
    result = x.astype(np.int64) @ weight.astype(np.int64).T + bias
    check_range(result, 48, "MAC accumulator")
    return result


class IntegerExecutor:
    def __init__(self, arrays, spec):
        self.spec = json.loads(json.dumps(spec))
        if spec["quantization_contract"] not in (QUANT_CONTRACT, MIXED_QUANT_CONTRACT, DYADIC_QUANT_CONTRACT):
            raise ValueError("unsupported integer contract")
        self.head_weight_bits = 8 if spec["quantization_contract"] == QUANT_CONTRACT else 16
        self.dyadic = spec["quantization_contract"] == DYADIC_QUANT_CONTRACT
        if spec.get("head_weight_bits", self.head_weight_bits) != self.head_weight_bits:
            raise ValueError("head precision disagrees with integer contract")
        self.arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
        for value in self.arrays.values():
            value.setflags(write=False)
        self.depths, self.widths = spec["depth_choices"], spec["width_choices"]
        self.max_depth, self.max_width = self.depths[-1], self.widths[-1]
        self.group_width, self.input_dim = spec["group_width"], spec["input_dim"]
        self.version = spec["package_version"]
        if not self.dyadic and any(name.endswith("_weight_multiplier") for name in self.arrays):
            raise ValueError("dyadic multiplier arrays require the v3 integer contract")
        for name, value in self.arrays.items():
            if name.endswith("_weight") or "_from" in name:
                expected = np.int16 if name.startswith("head") and self.head_weight_bits == 16 else np.int8
                if value.dtype != expected:
                    raise ValueError(f"{name} dtype disagrees with integer contract")
        self.clipped, self.activation_values = 0, 0
        if self.dyadic:
            self.validate_dyadic_bounds()

    def validate_dyadic_bounds(self):
        # Validate all A16 inputs, not just the observed golden-vector values.
        for name in ("stem", *[f"layer{d}" for d in range(2, self.max_depth+1)],
                     *[f"head{d}" for d in self.depths]):
            head = name.startswith("head")
            d = int(name[4:]) if head else int(name[5:]) if name.startswith("layer") else 1
            bias = self.arrays[name + "_bias"]
            mult = self.arrays[name + "_weight_multiplier"]
            base = self.arrays[name + "_weight_exponent"]
            if mult.dtype != np.uint16 or mult.shape != bias.shape or np.any(mult == 0):
                raise ValueError(f"invalid {name} shared multipliers")
            for row in range(len(bias)):
                columns = self.max_width if head else self.input_dim if name == "stem" else (row//self.group_width+1)*self.group_width
                bound = columns*32767*(32767 if head else 127) + abs(int(bias[row]))
                if bound > 2**47-1:
                    raise OverflowError(f"{name} MAC bound exceeds 48 bits")
                bound *= int(mult[row])
                if not head and d > 1:
                    # Branch base is input activation exponent + weight base.
                    # The identity has input activation exponent.
                    offset = int(base[row])
                    bound = (bound << max(offset, 0)) + (32767 << max(-offset, 0))
                if bound > 2**62-1:
                    raise OverflowError(f"{name} product/residual bound exceeds signed 63 bits")

    def scaled_accumulator(self, values, name, lo=None, hi=None):
        if not self.dyadic:
            return values
        multiplier = self.arrays[name + "_weight_multiplier"]
        return multiply_checked(values, multiplier if lo is None else multiplier[lo:hi])

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        spec = json.loads((directory / "quantization_spec.json").read_text(encoding="utf-8"))
        for name, expected in spec["package_file_sha256"].items():
            with (directory / name).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError(f"deployment package hash mismatch: {name}")
        with np.load(directory / spec.get("weights_archive", "weights_int8.npz"), allow_pickle=False) as archive:
            return cls(dict(archive), spec)

    def encode_normalized(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError("expected finite normalized input vectors")
        q = np.rint(np.ldexp(x.astype(np.float64), -int(self.arrays["activation_exponents"][0])))
        self.clipped += int(np.count_nonzero(np.abs(q) > 32767))
        self.activation_values += q.size
        return q.clip(-32767, 32767).astype(np.int16)

    def encode_raw(self, raw, norm):
        return self.encode_normalized((np.asarray(raw, dtype=np.float32) - np.asarray(norm["x_mean"], dtype=np.float32))
                                      / np.asarray(norm["x_std"], dtype=np.float32))

    def activation(self, value, source_exp, d):
        quantized = rescale(np.maximum(value, 0), source_exp, self.arrays["activation_exponents"][d])
        self.clipped += int(np.count_nonzero(quantized > 32767))
        self.activation_values += quantized.size
        result = quantized.clip(0, 32767).astype(np.int16)
        result.setflags(write=False)
        return result

    def context(self, q_input, sample_id):
        return IntegerContext(self, q_input, sample_id, self.version)

    def all_outputs(self, normalized_input):
        context = self.context(self.encode_normalized(normalized_input), "offline")
        context.grow(self.max_depth, self.max_width, sample_id="offline", model_version=self.version)
        return {(d, w): context.readout(d, w)[0] for d in self.depths for w in self.widths}


@dataclass
class IntegerContext:
    executor: IntegerExecutor
    q_input: np.ndarray
    sample_id: str
    model_version: str
    activations: dict = field(default_factory=dict)
    head_contributions: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.q_input.dtype != np.int16 or self.q_input.ndim != 2
                or self.q_input.shape[1] != self.executor.input_dim):
            raise ValueError("context requires A16 encoded inputs")
        self.q_input = self.q_input.copy()
        self.q_input.setflags(write=False)
        self.frontier = [0] * len(self.executor.widths)

    def grow(self, depth, width, *, sample_id, model_version):
        net, arrays = self.executor, self.executor.arrays
        if sample_id != self.sample_id or model_version != self.model_version or model_version != net.version:
            raise ValueError("stale context: another observation or model/quantization version")
        if depth not in net.depths or width not in net.widths:
            raise ValueError("unsupported depth/width")
        exps, g = arrays["activation_exponents"], net.group_width
        computed = []
        for group in range(width // g):
            lo, hi = group*g, (group+1)*g
            for d in range(self.frontier[group]+1, depth+1):
                if d == 1:
                    value = mac(self.q_input, arrays["stem_weight"][lo:hi], arrays["stem_bias"][lo:hi])
                    value = net.scaled_accumulator(value, "stem", lo, hi)
                    source_exp = int(exps[0]) + arrays["stem_weight_exponent"][lo:hi].astype(np.int64)
                else:
                    previous = np.concatenate([self.activations[(d-1, j)] for j in range(group+1)], axis=1)
                    weight = np.concatenate([arrays[f"layer{d}_g{group}_from{j}"] for j in range(group+1)], axis=1)
                    branch = mac(previous, weight, arrays[f"layer{d}_bias"][lo:hi])
                    branch = net.scaled_accumulator(branch, f"layer{d}", lo, hi)
                    branch_exp = int(exps[d-1]) + arrays[f"layer{d}_weight_exponent"][lo:hi].astype(np.int64)
                    source_exp = np.minimum(branch_exp, int(exps[d-1]))
                    value = (rescale(branch, branch_exp, source_exp)
                             + rescale(self.activations[(d-1, group)], int(exps[d-1]), source_exp))
                    check_range(value, 63, "residual sum")
                self.activations[(d, group)] = net.activation(value, source_exp, d)
                self.frontier[group] = d
                computed.append((d, group))
        return self.readout(depth, width), computed

    def readout(self, depth, width):
        net, arrays = self.executor, self.executor.arrays
        if depth not in net.depths or width not in net.widths or any(d < depth for d in self.frontier[:width//net.group_width]):
            raise ValueError("requested output is not ready")
        total = np.broadcast_to(arrays[f"head{depth}_bias"], (len(self.q_input), 4)).astype(np.int64).copy()
        g = net.group_width
        for group in range(width//g):
            key = (depth, group)
            if key not in self.head_contributions:
                self.head_contributions[key] = mac(self.activations[key],
                    arrays[f"head{depth}_weight"][:, group*g:(group+1)*g], np.zeros(4, dtype=np.int64),
                    weight_bits=net.head_weight_bits)
            total += self.head_contributions[key]
            check_range(total, 48, "head prefix accumulator")
        exponent = int(arrays["activation_exponents"][depth]) + arrays[f"head{depth}_weight_exponent"].astype(np.int64)
        total = net.scaled_accumulator(total, f"head{depth}")
        normalized = np.ldexp(total.astype(np.float64), exponent.astype(np.int32)).astype(np.float32)
        return normalized, total

    def cache_bytes(self):
        return sum(value.nbytes for value in self.activations.values()) + sum(value.nbytes for value in self.head_contributions.values())
