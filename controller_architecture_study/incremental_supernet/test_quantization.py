"""Short CPU numerical, loss-direction and interruption checks; no formal training."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .common import ROOT, atomic_torch, grid, sha256
from .integer_reference import IntegerExecutor, mac, rescale, multiply_checked
from .lsq_model import LearnedScaleSupernet
from .torch_integer import evaluate as evaluate_torch_integer, codes as torch_integer_codes, rescale as torch_integer_rescale, multiply as torch_integer_multiply
from .model import CausalSupernet, NETWORK_CONTRACT
from .qat_model import QUANT_CONTRACT, QuantizedSupernet, calibrate_activations, common_function_loss
from .quantize import DEFAULT_CONFIG, export, packed_arrays, stratified_rows, training, write_binary


class QuantizationTests(unittest.TestCase):
    head_weight_bits = 8
    learned_weight_scales = False

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        self.model = CausalSupernet(237, 4, [1, 3, 5], [4, 8, 12], 4).eval()
        self.x = torch.randn(4, 237)

    def quantized(self):
        with contextlib.redirect_stdout(io.StringIO()):
            exponents, _ = calibrate_activations(self.model, self.x, 2)
        cls = LearnedScaleSupernet if self.learned_weight_scales else QuantizedSupernet
        quantized = cls(self.model, exponents, head_weight_bits=self.head_weight_bits).eval()
        if self.learned_weight_scales:
            quantized.freeze_scales()
        arrays, _ = packed_arrays(quantized)
        spec = {"quantization_contract": quantized.quantization_contract, "depth_choices": [1, 3, 5],
                "width_choices": [4, 8, 12], "group_width": 4, "input_dim": 237, "package_version": "fixture"}
        return quantized, IntegerExecutor(arrays, spec)

    def test_negative_ties_and_overflow(self):
        np.testing.assert_array_equal(rescale(np.arange(-7, 8), 0, 1), np.rint(np.arange(-7, 8)/2).astype(np.int64))
        self.assertEqual(int(rescale(np.array([3]), 2, 0)[0]), 12)
        with self.assertRaises(OverflowError):
            rescale(np.asarray([2**50], dtype=np.int64), 20, 0)
        with self.assertRaises(OverflowError):
            mac(np.full((1, 1024), 32767, dtype=np.int16), np.full((1, 1024), 127, dtype=np.int8), np.asarray([2**47-1]))
        # A16 MAC legitimately exceeds INT32; it must NOT overflow.
        output = mac(np.full((1, 1024), 32767, dtype=np.int16), np.full((1, 1024), 127, dtype=np.int8), np.zeros(1))
        self.assertEqual(int(output[0, 0]), 4261282816)

    def test_residual_fold_preserves_float_function(self):
        with torch.inference_mode():
            original = self.model.features(self.x)
            folded = QuantizedSupernet(copy.deepcopy(self.model), [0]*6)
            h = torch.relu(folded.model.stem(self.x))
            for d, block in enumerate(folded.model.blocks, 2):
                h = torch.relu(h + torch.nn.functional.linear(h, block.weight * folded.model.causal_mask, block.bias))
                if d in original:
                    torch.testing.assert_close(h, original[d], atol=3e-6, rtol=3e-6)

    def test_causal_growth_and_fake_quant(self):
        quantized, integer = self.quantized()
        actual = integer.all_outputs(self.x.numpy())
        with torch.inference_mode():
            expected = quantized.outputs(quantized.features(self.x), grid(quantized))
            for key in grid(quantized):
                np.testing.assert_allclose(actual[key], expected[key].numpy(), atol=3e-5, rtol=2e-5)
                torch.testing.assert_close(expected[key], quantized(self.x, *key), atol=3e-5, rtol=2e-5)
        q_input = integer.encode_normalized(self.x.numpy())
        for order in (((3, 4), (5, 4), (5, 12)), ((1, 12), (3, 12), (5, 12))):
            cache = integer.context(q_input, "one")
            old = {}
            for d, w in order:
                (value, codes), computed = cache.grow(d, w, sample_id="one", model_version="fixture")
                np.testing.assert_array_equal(value, actual[(d, w)])
                self.assertTrue(all(not np.any(old[key] != cache.activations[key]) for key in old))
                self.assertTrue(all(key not in old for key in computed))
                old = {key: value.copy() for key, value in cache.activations.items()}
            (_, repeat), computed = cache.grow(5, 12, sample_id="one", model_version="fixture")
            np.testing.assert_array_equal(repeat, codes)
            self.assertFalse(computed)
            with self.assertRaises(ValueError):
                cache.grow(5, 12, sample_id="new", model_version="fixture")
            with self.assertRaises(ValueError):
                cache.grow(5, 12, sample_id="one", model_version="other")
        with self.assertRaises(ValueError):
            cache.readout(4, 12)

    def test_consistency_gradient_direction_and_common_teacher(self):
        model = CausalSupernet(237, 4, [1, 3], [4, 8], 4)
        outputs = {key: torch.full((2, 4), 1., requires_grad=True) for key in grid(model)}
        # Smaller is wrong, largest is correct: the consistency gradient moves only
        # the smaller output toward the accurate larger output.
        outputs[(3, 8)] = torch.zeros((2, 4), requires_grad=True)
        config = {"kd_weight": 0., "consistency_weight": 1., "maximum_label_weight": 0., "maximum_anchor_weight": 0.}
        target = torch.zeros(2, 4)
        _, terms = common_function_loss(outputs, target, target, torch.ones(1), torch.zeros(2, dtype=torch.long), model, config)
        gradients = torch.autograd.grad(terms["consistency"], (outputs[(3, 4)], outputs[(3, 8)]), allow_unused=True)
        self.assertTrue(torch.all(gradients[0] > 0))
        self.assertTrue(gradients[1] is None or torch.count_nonzero(gradients[1]) == 0)
        # An inaccurate larger path must not pull an accurate smaller path away.
        for key in outputs:
            outputs[key] = torch.zeros((2, 4), requires_grad=True)
        outputs[(3, 8)] = torch.ones((2, 4), requires_grad=True)
        _, terms = common_function_loss(outputs, target, target, torch.ones(1)*2, torch.zeros(2, dtype=torch.long), model, config)
        self.assertEqual(float(terms["consistency"]), 0.)
        selected = stratified_rows(np.asarray([0]*10+[1]*3), 8, 4)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(np.unique(selected)), 8)
        self.assertTrue(np.any(selected >= 10))

    def test_training_resume_export_and_source_protection(self):
        parent = ROOT / "controller_architecture_study/results/incremental_supernet_checks"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="qat_", dir=parent) as directory:
            base = Path(directory)
            source_dir, assets = base/"source", base/"assets"
            source_dir.mkdir()
            assets.mkdir()
            teacher_path = assets/"teacher.pt"
            atomic_torch(teacher_path, {"fixture": True})
            for split, count in (("train", 8), ("validation", 4)):
                x = torch.randn(count, 237)
                with torch.no_grad():
                    y = self.model(x).numpy()
                np.savez(assets/f"{split}.npz", X=x.numpy(), U=y, family_id=np.arange(count)%2)
            (assets/"normalization.json").write_text(json.dumps({"input_contract": "decision_convex_local_relative_pv_237d_v1",
                "x_mean": [0.]*237, "x_std": [1.]*237, "u_mean": [0.]*4, "u_std": [1.]*4}), encoding="utf-8")
            source_config = {"input_dim": 237, "input_contract": "decision_convex_local_relative_pv_237d_v1",
                "depth_choices": [1, 3, 5], "width_choices": [4, 8, 12], "group_width": 4,
                "data_dir": str(assets), "sample_counts": {"train": 8, "validation": 4},
                "teacher_checkpoint": str(teacher_path), "teacher_sha256": sha256(teacher_path),
                "dataset_file_sha256": {name: sha256(assets/name) for name in ("train.npz", "validation.npz", "normalization.json")}}
            source_path = source_dir/"best_model.pt"
            atomic_torch(source_path, {"network_contract": NETWORK_CONTRACT, "config": source_config,
                                      "model_state": self.model.state_dict(), "epoch": 1})
            config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            config.update(source_checkpoint=str(source_path), source_sha256=sha256(source_path), epochs=2,
                          batch_size=4, evaluation_batch_size=4, cpu_threads=1, calibration_samples=8,
                          fake_integer_physical_tolerance=1e-4)
            if self.head_weight_bits == 16:
                config.update(head_weight_bits=16, reference_preservation=True,
                              configuration_kd_weight=2., switch_delta_weight=1., weight_margin=1., weight_decay=0.)
            if self.learned_weight_scales:
                config.update(learned_weight_scales=True, scale_learning_epochs=1, scale_lr=3e-4, epochs=3)
                with np.load(assets/'train.npz', allow_pickle=False) as archive:
                    calibration = torch.from_numpy(archive['X'].copy())
                with contextlib.redirect_stdout(io.StringIO()):
                    exponents, _ = calibrate_activations(self.model, calibration, 4)
                warm = QuantizedSupernet(copy.deepcopy(self.model), exponents, head_weight_bits=16)
                warm_path = assets/'warm_v2.pt'
                atomic_torch(warm_path, dict(model_state=warm.state_dict(), activation_exponents=exponents.tolist(),
                    source_config=source_config, config=dict(source_sha256=sha256(source_path)),
                    quantization_contract=warm.quantization_contract))
                config.update(initial_qat_checkpoint=str(warm_path), initial_qat_sha256=sha256(warm_path))
            uninterrupted, interrupted = base/"full", base/"resume"
            source_hash = sha256(source_path)
            interruption_epoch = 1
            def interrupt(path, payload):
                atomic_torch(path, payload)
                if Path(path).name == "latest_checkpoint.pt" and payload["epoch"] == interruption_epoch:
                    raise KeyboardInterrupt
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    training(config, source_dir/"nested", device="cpu", resume=True)
                training(copy.deepcopy(config), uninterrupted, device="cpu", resume=True)
                with patch("controller_architecture_study.incremental_supernet.quantize.atomic_torch", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        training(copy.deepcopy(config), interrupted, device="cpu", resume=True)
                training(copy.deepcopy(config), interrupted, device="cpu", resume=True)
                if self.learned_weight_scales:
                    interruption_epoch = 2
                    frozen_resume = base/'resume_frozen'
                    with patch("controller_architecture_study.incremental_supernet.quantize.atomic_torch", side_effect=interrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            training(copy.deepcopy(config), frozen_resume, device="cpu", resume=True)
                    training(copy.deepcopy(config), frozen_resume, device="cpu", resume=True)
                report = export(interrupted/"best_model.pt", interrupted/"export", golden_count=3, integer_batch_size=2)
            a = torch.load(uninterrupted/"last_model.pt", weights_only=False)
            b = torch.load(interrupted/"last_model.pt", weights_only=False)
            for key in a["model_state"]:
                self.assertTrue(torch.equal(a["model_state"][key], b["model_state"][key]), key)
            if self.learned_weight_scales:
                c = torch.load(frozen_resume/'last_model.pt', weights_only=False)
                for key in a['model_state']:
                    self.assertTrue(torch.equal(a['model_state'][key], c['model_state'][key]), key)
                self.assertTrue(bool(a['model_state']['scales_frozen']))
                self.assertEqual(sha256(warm_path), config['initial_qat_sha256'])
            self.assertEqual(sha256(source_path), source_hash)
            self.assertTrue(report["integer_direct_incremental_bit_exact"])
            self.assertEqual(report["configurations"], 9)
            self.assertFalse(report["test_read"])
            restored = IntegerExecutor.load(interrupted/"export")
            self.assertEqual(restored.max_width, 12)
            name = "weights_mixed" if self.head_weight_bits == 16 else "weights_int8"
            with np.load(interrupted/"export"/(name + ".npz"), allow_pickle=False) as arrays:
                spec = json.loads((interrupted/"export"/"quantization_spec.json").read_text(encoding="utf-8"))
                binary = (interrupted/"export"/(name + ".bin")).read_bytes()
                for name, descriptor in spec["array_layout"].items():
                    actual = np.frombuffer(binary, dtype=descriptor["dtype"], count=arrays[name].size,
                                           offset=descriptor["offset_bytes"]).reshape(descriptor["shape"])
                    np.testing.assert_array_equal(actual, arrays[name])
                    self.assertEqual(descriptor["offset_bytes"] % 64, 0)


class MixedQuantizationTests(QuantizationTests):
    head_weight_bits = 16

    def test_w16_mac_range_and_package_precision(self):
        x = np.full((1, 1024), 32767, dtype=np.int16)
        w = np.full((1, 1024), 32767, dtype=np.int16)
        self.assertEqual(int(mac(x, w, np.zeros(1), weight_bits=16)[0, 0]), 1024*32767**2)
        with self.assertRaises(ValueError):
            mac(x, w, np.zeros(1))
        with self.assertRaises(OverflowError):
            mac(x, w, np.asarray([2**47-1]), weight_bits=16)
        _, integer = self.quantized()
        self.assertEqual(integer.arrays['head5_weight'].dtype, np.int16)
        self.assertEqual(integer.arrays['stem_weight'].dtype, np.int8)
        incompatible = dict(integer.spec, quantization_contract=QUANT_CONTRACT)
        with self.assertRaises(ValueError):
            IntegerExecutor(integer.arrays, incompatible)

    def test_preserve_configuration_and_switch_differences(self):
        model = CausalSupernet(237, 4, [1, 3], [4, 8], 4)
        reference = {key: torch.full((2, 4), float(i), requires_grad=True) for i, key in enumerate(grid(model))}
        outputs = {key: value.detach().clone().requires_grad_(True) for key, value in reference.items()}
        config = dict(reference_preservation=True, configuration_kd_weight=2., switch_delta_weight=1.,
                      kd_weight=0., maximum_label_weight=0., maximum_anchor_weight=0.)
        def terms_for(values):
            return common_function_loss(values, torch.zeros(2, 4), torch.zeros(2, 4),
                                        torch.zeros(1), torch.zeros(2, dtype=torch.long), model, config, reference)[1]
        terms = terms_for(outputs)
        self.assertEqual(float(terms['configuration_kd']), 0.)
        self.assertEqual(float(terms['switch_delta']), 0.)  # different FP32 outputs are legitimate
        outputs[(1, 4)] = (outputs[(1, 4)].detach() + 1).requires_grad_(True)
        terms = terms_for(outputs)
        gradients = torch.autograd.grad(terms['switch_delta'],
                                       (outputs[(1, 4)], outputs[(1, 8)], reference[(1, 4)]), allow_unused=True)
        self.assertTrue(torch.all(gradients[0] > 0))
        self.assertTrue(torch.all(gradients[1] < 0))
        self.assertIsNone(gradients[2])
        shifted = {key: value.detach() + 1 for key, value in reference.items()}
        terms = terms_for(shifted)
        self.assertEqual(float(terms['switch_delta']), 0.)
        self.assertGreater(float(terms['configuration_kd']), 0.)


class LearnedScaleTests(MixedQuantizationTests):
    learned_weight_scales = True

    def test_step_gradient_freeze_and_export_gate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            exponents, _ = calibrate_activations(self.model, self.x, 2)
        q = LearnedScaleSupernet(self.model, exponents)
        loss = q(self.x).square().mean()
        grad = torch.autograd.grad(loss, q.log_steps['stem'])[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            packed_arrays(q)
        q.freeze_scales()
        step = q.weight_scale('stem').clone()
        self.assertFalse(q.log_steps['stem'].requires_grad)
        with torch.no_grad():
            q.log_steps['stem'].add_(0.01)
        torch.testing.assert_close(q.weight_scale('stem'), step, rtol=0, atol=0)
        arrays, _ = packed_arrays(q)
        self.assertEqual(arrays['stem_weight_multiplier'].dtype, np.uint16)
        self.assertTrue(np.all(arrays['stem_weight_multiplier'] > 0))

    def test_dyadic_product_checks_before_overflow(self):
        np.testing.assert_array_equal(multiply_checked(np.asarray([-3, 2, 7]), np.asarray([7, 9, 3])), [-21, 18, 21])
        with self.assertRaises(OverflowError):
            multiply_checked(np.asarray([2**50]), np.asarray([65535]))
        with self.assertRaises(ValueError):
            multiply_checked(np.asarray([1]), np.asarray([0]))
        _, integer = self.quantized()
        arrays = dict(integer.arrays)
        arrays['layer2_weight_exponent'] = np.full(12, -62, dtype=np.int16)
        with self.assertRaises(OverflowError):
            IntegerExecutor(arrays, integer.spec)

    def test_independent_torch_integer_and_fp64_fake_quant(self):
        q, integer = self.quantized()
        keys = grid(q)
        output, codes = evaluate_torch_integer(q, self.x, keys)
        ctx = integer.context(integer.encode_normalized(self.x.numpy()), 'check')
        ctx.grow(q.max_depth, q.max_width, sample_id='check', model_version=integer.version)
        for key in keys:
            normalized, expected = ctx.readout(*key)
            np.testing.assert_array_equal(codes[key].numpy(), expected)
            np.testing.assert_array_equal(output[key].numpy(), normalized)
        q.double()
        with torch.no_grad():
            precise = q.outputs(q.features(self.x.double()), keys)
        for key in keys:
            np.testing.assert_array_equal(precise[key].numpy(), output[key].numpy())
        values = torch.arange(-7, 8, dtype=torch.int64)
        np.testing.assert_array_equal(torch_integer_rescale(values, 0, 1).numpy(), np.rint(values.numpy()/2))
        with self.assertRaises(OverflowError):
            torch_integer_multiply(torch.tensor([2**50]), torch.tensor([65535]))

    def test_weight_division_at_rounding_boundary(self):
        q, _ = self.quantized()
        head = q.model.heads[0]
        with torch.no_grad():
            head.weight[0, 0] = 0.08848472684621811
            q.head1_weight_multiplier[0] = 19001
            q.head1_weight_exponent[0] = -31
        approximate, _, _ = q.codes('head1', head, q.activation_exponents[1])
        canonical, _, _, _ = torch_integer_codes(q, 'head1', head, q.activation_exponents[1])
        self.assertEqual(int(approximate[0, 0]), 10000)
        self.assertEqual(int(canonical[0, 0]), 10001)


if __name__ == "__main__":
    unittest.main()
