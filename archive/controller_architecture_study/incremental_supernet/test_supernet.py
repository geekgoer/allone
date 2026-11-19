"""Short CPU checks: causal prefixes, resume and export; NOT a flight benchmark."""

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

from ..models.residual_mlp import ResidualMLP
from .common import ROOT, atomic_torch, grid, sha256
from .export import export
from .model import CausalSupernet, IncrementalContext, PhysicalController
from .train import continuation_state, loss_for, optimizer_for, sample_configs, selection_key, train


class SupernetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = CausalSupernet(237, 4, [1,3,5], [4,8,12], 4).eval()
        self.x = torch.randn(3, 237)

    def test_prefix_and_gradients(self):
        full = self.model.features(self.x)
        for width in self.model.widths:
            h = torch.relu(torch.nn.functional.linear(self.x, self.model.stem.weight[:width], self.model.stem.bias[:width]))
            torch.testing.assert_close(h, full[1][:, :width], atol=2e-6, rtol=2e-6)
            for d, block in enumerate(self.model.blocks, 2):
                h = torch.relu(h + self.model.residual_scale * torch.nn.functional.linear(h,
                    block.weight[:width,:width] * self.model.causal_mask[:width,:width], block.bias[:width]))
                if d in full:
                    torch.testing.assert_close(h, full[d][:,:width], atol=3e-6, rtol=3e-6)
        outputs = self.model.outputs(full, grid(self.model))
        for key, value in outputs.items():
            torch.testing.assert_close(value, self.model(self.x, *key), atol=3e-6, rtol=3e-6)
        sum(v.square().mean() for v in outputs.values()).backward()
        for block in self.model.blocks:
            self.assertEqual(float(block.weight.grad[self.model.causal_mask == 0].abs().sum()), 0)

    def test_incremental_orders_and_stale_rejection(self):
        cache = IncrementalContext(self.model, self.x, "sample", "version")
        cache.grow(3,4,sample_id="sample",model_version="version")
        old = {key: value.clone() for key, value in cache.activations.items()}
        _, nodes = cache.grow(5,4,sample_id="sample",model_version="version")
        self.assertEqual(nodes, [(4,0),(5,0)])
        _, nodes = cache.grow(5,8,sample_id="sample",model_version="version")
        self.assertEqual(nodes, [(d,1) for d in range(1,6)])
        for key, value in old.items():
            self.assertTrue(torch.equal(value, cache.activations[key]))
        a, _ = cache.grow(5,12,sample_id="sample",model_version="version")
        other = IncrementalContext(self.model,self.x,"sample","version")
        other.grow(1,12,sample_id="sample",model_version="version")
        b, _ = other.grow(5,12,sample_id="sample",model_version="version")
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(a,self.model(self.x),atol=3e-6,rtol=3e-6)
        self.assertEqual(cache.cache_bytes(), 3*5*12*4)
        with self.assertRaises(ValueError):
            cache.grow(5,12,sample_id="new_sample",model_version="version")
        changed_input = IncrementalContext(self.model,self.x,"sample","version")
        changed_input.normalized_input.add_(0.1)
        with self.assertRaises(ValueError):
            changed_input.grow(5,12,sample_id="sample",model_version="version")
        with torch.no_grad():
            self.model.heads[-1].bias.add_(0.1)
        with self.assertRaises(ValueError):
            cache.grow(5,12,sample_id="sample",model_version="version")

    def test_scripted_dynamic_grid_and_normalization(self):
        norm = {"x_mean": np.ones(237), "x_std": np.full(237,2),
                "u_mean": np.arange(4), "u_std": np.arange(1,5)}
        wrapper = PhysicalController(self.model,norm).eval()
        script = torch.jit.script(wrapper)
        for d,w in grid(self.model):
            torch.testing.assert_close(script(self.x,d,w),wrapper(self.x,d,w))

    def test_warm_start_is_masked_not_equal_dense_teacher(self):
        teacher = ResidualMLP(237,4,5,12)
        self.model.initialize_from_teacher(teacher)
        self.assertTrue(torch.equal(self.model.stem.weight,teacher.stem.weight))
        self.assertTrue(torch.equal(self.model.heads[-1].weight,teacher.head.weight))
        for block in self.model.blocks:
            self.assertEqual(float(block.weight[self.model.causal_mask == 0].abs().sum()),0.)
        self.assertFalse(torch.allclose(self.model(self.x),teacher(self.x)))

    def test_sampler_covers_grid_and_gating(self):
        import random
        sampler, seen = random.Random(123), set()
        for _ in range(400):
            configs, pairs = sample_configs(self.model,"joint",sampler)
            self.assertLessEqual(len(configs),6)
            seen.update(configs)
        self.assertEqual(seen,set(grid(self.model)))
        a = torch.zeros(2,4,requires_grad=True)
        b = torch.ones(2,4,requires_grad=True)
        cfg = {"kd_weight":0.,"max_anchor_weight":0.,"consistency_weight":1.}
        loss,_ = loss_for({(1,4):a,(3,4):b},[(1,4),(3,4)],[((1,4),(3,4))],
                         torch.zeros(2,4),torch.zeros(2,4),cfg,0.1,(5,12))
        self.assertEqual(float(loss),0.5)  # Inaccurate larger output: no consistency penalty.

    def test_recovery_head_sampling_and_label_anchor(self):
        import random
        seen = set()
        sampler = random.Random(321)
        for _ in range(400):
            configs, _ = sample_configs(self.model, "heads", sampler, include_maximum_head=True)
            self.assertIn((5,12), configs)
            seen.update(configs)
        self.assertEqual(seen, set(grid(self.model)))
        optimizer_for(self.model, {"name":"heads", "head_lr":1e-4, "maximum_head_lr":1e-5},
                      {"weight_decay":1e-5})
        self.assertFalse(self.model.stem.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in self.model.heads.parameters()))
        a, b = torch.zeros(2,4), torch.ones(2,4)
        cfg = {"kd_weight":0., "max_anchor_weight":0., "maximum_label_weight":4.}
        loss, label = loss_for({(1,4):a,(5,12):b}, [(1,4),(5,12)], [], a, a, cfg, None, (5,12))
        self.assertEqual(float(label), .5)
        self.assertEqual(float(loss), 4.5)

    def test_full_grid_loss_and_guard_selection(self):
        keys = [(1,4),(3,8),(5,12)]
        outputs = {key:torch.randn(2,4,requires_grad=True) for key in keys}
        target, teacher = torch.randn(2,4), torch.randn(2,4)
        cfg = {"kd_weight":.1, "max_anchor_weight":.25, "maximum_label_weight":1.,
               "maximum_validation_degradation":1.05}
        reference, label = loss_for(outputs, keys, [], target, teacher, cfg, None, (5,12))
        vectorized, vector_label = loss_for(outputs, keys, [], target, teacher,
                                           {**cfg,"full_grid_training":True}, None, (5,12))
        torch.testing.assert_close(reference, vectorized)
        torch.testing.assert_close(label, vector_label)
        a = torch.autograd.grad(reference, list(outputs.values()), retain_graph=True)
        b = torch.autograd.grad(vectorized, list(outputs.values()))
        for lhs,rhs in zip(a,b):
            torch.testing.assert_close(lhs,rhs)
        baseline = selection_key(cfg,.2,.5,.1,.22)
        better_grid = selection_key(cfg,.21,.4,.1,.22)
        over_guard = selection_key(cfg,.23,.1,.1,.22)
        self.assertLess(better_grid, baseline)
        self.assertGreater(over_guard, baseline)

    def test_epoch_resume_and_packed_export(self):
        parent = ROOT / "controller_architecture_study/results/incremental_supernet_checks"
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fixture_",dir=parent) as directory:
            base = Path(directory)
            assets = base / "assets"
            assets.mkdir()
            teacher = ResidualMLP(237,4,5,12)
            source = assets / "teacher.pt"
            atomic_torch(source,{"model_type":"residual_mlp","input_dim":237,"hidden_layers":5,
                                 "hidden_width":12,"model_state":teacher.state_dict()})
            rng = np.random.default_rng(31)
            for split,count in (("train",12),("validation",8)):
                x = rng.normal(size=(count,237)).astype(np.float32)
                with torch.no_grad():
                    u = teacher(torch.from_numpy(x)).numpy()
                np.savez(assets/f"{split}.npz",X=x,U=u,family_id=np.arange(count)%2)
            norm = {"input_contract":"decision_convex_local_relative_pv_237d_v1","x_mean":[0.]*237,
                    "x_std":[1.]*237,"u_mean":[0.]*4,"u_std":[1.]*4}
            (assets/"normalization.json").write_text(json.dumps(norm),encoding="utf-8")
            config = json.loads((ROOT/"controller_architecture_study/configs/incremental_supernet_237d_v2_causal_width.json").read_text())
            config.update(depth_choices=[1,3,5],width_choices=[4,8,12],group_width=4,batch_size=4,
                          evaluation_batch_size=4,cpu_threads=1,sample_counts={"train":12,"validation":8},
                          consistency_calibration_samples=8,data_dir=str(assets),teacher_checkpoint=str(source),
                          teacher_sha256=sha256(source),dataset_file_sha256={n:sha256(assets/n) for n in
                              ("train.npz","validation.npz","normalization.json")})
            for stage in config["stages"]:
                stage["epochs"] = 2
            full, resumed = base/"full",base/"resumed"
            with contextlib.redirect_stdout(io.StringIO()):
                train(copy.deepcopy(config),full,device="cpu",resume=True)
                def interrupt(path,payload):
                    atomic_torch(path,payload)
                    if Path(path).name == "latest_checkpoint.pt" and payload["epoch"] == 3:
                        raise KeyboardInterrupt
                with patch("controller_architecture_study.incremental_supernet.train.atomic_torch",side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        train(copy.deepcopy(config),resumed,device="cpu",resume=True)
                train(copy.deepcopy(config),resumed,device="cpu",resume=True)
                report = export(resumed/"best_model.pt",resumed/"export",device="cpu",golden_count=3)
            a = torch.load(full/"last_model.pt",weights_only=False)
            b = torch.load(resumed/"last_model.pt",weights_only=False)
            for key in a["model_state"]:
                self.assertTrue(torch.equal(a["model_state"][key],b["model_state"][key]),key)
            self.assertEqual(report["status"],"PASS")
            self.assertEqual(report["configurations"],9)
            self.assertFalse(report["test_read"])
            continuation = copy.deepcopy(config)
            continuation.update(initial_supernet_checkpoint=str(full/"best_model.pt"),
                                initial_supernet_sha256=sha256(full/"best_model.pt"),
                                maximum_label_weight=4., max_anchor_weight=1., maximum_first_selection=True)
            continuation["stages"][1]["maximum_head_lr"] = 1e-5
            destination = base/"continuation"
            with self.assertRaises(ValueError):
                continuation_state(continuation, full/"nested")
            wrong_hash = {**continuation, "initial_supernet_sha256":"0"*64}
            with self.assertRaises(ValueError):
                continuation_state(wrong_hash, destination)
            wrong_contract = {**continuation, "group_width":3}
            with self.assertRaises(ValueError):
                continuation_state(wrong_contract, destination)
            source_hash = sha256(full/"best_model.pt")
            def stop_at_initial(path,payload):
                atomic_torch(path,payload)
                if Path(path).name == "latest_checkpoint.pt" and payload["epoch"] == 0:
                    raise KeyboardInterrupt
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("controller_architecture_study.incremental_supernet.train.atomic_torch", side_effect=stop_at_initial):
                    with self.assertRaises(KeyboardInterrupt):
                        train(continuation, destination, device="cpu", resume=True)
                initialized = torch.load(destination/"latest_checkpoint.pt",weights_only=False)
                source_weights = torch.load(full/"best_model.pt",weights_only=False)["model_state"]
                for key in source_weights:
                    self.assertTrue(torch.equal(initialized["model_state"][key],source_weights[key]),key)
                train(continuation, destination, device="cpu", resume=True)
                recovery_report = export(destination/"best_model.pt", destination/"export", device="cpu", golden_count=3)
            self.assertEqual(sha256(full/"best_model.pt"), source_hash)
            self.assertEqual(recovery_report["status"], "PASS")
            self.assertEqual(torch.load(destination/"last_model.pt",weights_only=False)["epoch"], 6)
            grid_config = copy.deepcopy(continuation)
            grid_config.update(full_grid_training=True, maximum_first_selection=False,
                               grid_first_maximum_guard_factor=1.1, maximum_label_weight=1., max_anchor_weight=.25)
            grid_config["stages"] = grid_config["stages"][1:]
            uninterrupted, grid_resumed = base/"grid_full", base/"grid_resumed"
            with contextlib.redirect_stdout(io.StringIO()):
                train(grid_config, uninterrupted, device="cpu", resume=True)
                with patch("controller_architecture_study.incremental_supernet.train.atomic_torch",side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        train(grid_config, grid_resumed, device="cpu", resume=True)
                train(grid_config, grid_resumed, device="cpu", resume=True)
                grid_report = export(grid_resumed/"best_model.pt", grid_resumed/"export", device="cpu", golden_count=3)
            a = torch.load(uninterrupted/"last_model.pt",weights_only=False)
            b = torch.load(grid_resumed/"last_model.pt",weights_only=False)
            for key in a["model_state"]:
                self.assertTrue(torch.equal(a["model_state"][key],b["model_state"][key]),key)
            summary = json.loads((grid_resumed/"completion.json").read_text())
            grid_history = json.loads((grid_resumed/"history.json").read_text())
            self.assertTrue(all(row["configurations_per_batch"] == 9 for row in grid_history))
            self.assertLessEqual(summary["best_maximum_validation_mse"],summary["maximum_guard_mse"])
            self.assertTrue((grid_resumed/"maximum_best_model.pt").exists())
            self.assertEqual(grid_report["status"], "PASS")
            self.assertEqual(sha256(full/"best_model.pt"), source_hash)


if __name__ == "__main__":
    unittest.main()
