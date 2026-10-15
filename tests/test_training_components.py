"""Config-based model, optimizer and explicit-path loader behavior."""

from __future__ import annotations

import copy
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from config import AdamWConfig, ModelConfig, OptimizerConfig, PretrainingConfig, ScheduleConfig
from src.model.model import (
    CanonAttentionLayer, CanonFFNLayer, CanonLayer, CausalSelfAttention,
    DecoderTransformer, RoPE, SwiGLUFFN, TransformerBlock,
)
from components.data import TokenBlockDataset, TrainDataLoader, ValidationDataLoader
from components.optimizer import Optimizer


def model_config(**overrides):
    return ModelConfig(
        vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4, ffn_dim=32,
        seq_len=8, **overrides,
    )


class ModelConfigConstructorTests(unittest.TestCase):
    def test_model_and_layers_accept_config_and_preserve_shapes(self):
        config = model_config(qk_norm=True, attn_bias=True, canon_layers=True, canon_kernel_size=5)
        inputs = torch.randn(2, 8, 16)
        for layer in (CanonLayer(config), CausalSelfAttention(config), SwiGLUFFN(config), TransformerBlock(config)):
            self.assertEqual(layer(inputs).shape, inputs.shape)
        self.assertEqual(CanonAttentionLayer(config)(torch.randn(2, 8, 48)).shape, (2, 8, 48))
        self.assertEqual(CanonFFNLayer(config)(torch.randn(2, 8, 64)).shape, (2, 8, 64))
        rope = RoPE(config)
        self.assertEqual(rope(torch.randn(2, 4, 8, 4)).shape, (2, 4, 8, 4))
        with self.assertRaisesRegex(ValueError, "exceeds RoPE limit"):
            rope(torch.randn(2, 4, 9, 4))
        model = DecoderTransformer(config)
        self.assertEqual(model(torch.randint(0, 32, (2, 8))).shape, (2, 8, 32))
        self.assertEqual(model.layers[0].attn.c_attn.bias.shape, (48,))
        self.assertEqual(model.layers[0].ca.weight.shape, (16, 1, 5))

    def test_state_dict_layouts_and_initialization_are_reproducible(self):
        for layout in (False, True):
            for recipe in ("pytorch_default", "gpt_scaled", "olmo"):
                with self.subTest(layout=layout, recipe=recipe):
                    config = model_config(split_qkv_projections=layout, initialization=recipe)
                    torch.manual_seed(17)
                    model = DecoderTransformer(config)
                    torch.manual_seed(17)
                    same = DecoderTransformer(config)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, same.state_dict()[name], rtol=0, atol=0)
                    same.load_state_dict(model.state_dict(), strict=True)
                    projection = "q_proj" if layout else "c_attn"
                    self.assertIn(f"layers.0.attn.{projection}.weight", model.state_dict())
                    self.assertFalse(any("rope_cos" in name for name in model.state_dict()))
                    inputs = torch.randint(0, 32, (2, 8))
                    torch.testing.assert_close(model(inputs), same(inputs), rtol=0, atol=0)


class OptimizerTests(unittest.TestCase):
    def config(self, kind="adamw", schedule_kind="cosine", **overrides):
        return OptimizerConfig(
            kind=kind, lr=0.02, norm_lr_multiplier=0.25,
            adamw=AdamWConfig(fused=False, weight_decay=0.1),
            schedule=ScheduleConfig(kind=schedule_kind, warmup_steps=2, min_lr=0.002),
            **overrides,
        )

    def test_partition_covers_each_trainable_parameter_once(self):
        for kind in ("adamw", "muon_adamw"):
            with self.subTest(kind=kind):
                model = DecoderTransformer(model_config(qk_norm=True, canon_layers=True))
                model.layers[0].ffn.w1.bias.requires_grad_(False)
                wrapper = Optimizer(self.config(kind), model, total_steps=8)
                names = {id(p): name for name, p in model.named_parameters()}
                groups = wrapper.optimizers["adamw"].param_groups
                self.assertTrue(any(p is model.embeddings.weight for p in groups[1]["params"]))
                self.assertEqual(groups[1]["weight_decay"], 0.0)
                self.assertEqual(groups[2]["weight_decay"], 0.0)
                self.assertAlmostEqual(groups[2]["lr"] / groups[0]["lr"], 0.25)
                self.assertTrue(any(p is model.layers[0].ca.weight for p in groups[0]["params"]))
                assigned = [id(p) for opt in wrapper.optimizers.values()
                            for group in opt.param_groups for p in group["params"]]
                self.assertEqual(len(assigned), len(set(assigned)))
                self.assertEqual(set(assigned), {id(p) for p in model.parameters() if p.requires_grad})
                if kind == "muon_adamw":
                    muon = wrapper.optimizers["muon"]
                    for group in muon.param_groups:
                        for p in group["params"]:
                            self.assertEqual(p.ndim, 2)
                            self.assertNotEqual(names[id(p)], "embeddings.weight")
                    self.assertAlmostEqual(muon.param_groups[0]["lr"], groups[0]["lr"])

    def test_step_methods_are_separate_and_all_optimizers_update(self):
        for kind in ("adamw", "muon_adamw"):
            with self.subTest(kind=kind):
                model = DecoderTransformer(model_config())
                wrapper = Optimizer(self.config(kind), model, total_steps=8)
                before = copy.deepcopy(model.state_dict())
                model(torch.randint(0, 32, (2, 8))).square().mean().backward()
                counters = {name: scheduler.last_epoch for name, scheduler in wrapper.schedulers.items()}
                wrapper.step_optimizer()
                self.assertEqual(counters, {name: s.last_epoch for name, s in wrapper.schedulers.items()})
                for optimizer in wrapper.optimizers.values():
                    self.assertTrue(optimizer.state)
                    self.assertTrue(any(not torch.equal(p, before[name])
                                        for name, p in model.named_parameters()
                                        if p in optimizer.state))
                after_update = copy.deepcopy(model.state_dict())
                wrapper.step_schedule()
                for name, scheduler in wrapper.schedulers.items():
                    self.assertEqual(scheduler.last_epoch, counters[name] + 1)
                for name, p in model.state_dict().items():
                    torch.testing.assert_close(p, after_update[name], rtol=0, atol=0)

    def test_lr_sequences_preserve_warmup_decay_and_group_ratios(self):
        for kind in ("adamw", "muon_adamw"):
            for schedule in ("cosine", "wsd"):
                with self.subTest(kind=kind, schedule=schedule):
                    config = self.config(kind, schedule)
                    config.schedule.decay_fraction = 0.5
                    config.muon.lr_multiplier = 0.4
                    wrapper = Optimizer(config, DecoderTransformer(model_config()), total_steps=8)
                    expected = [0.5, 1.0]
                    if schedule == "cosine":
                        expected += [0.1 + 0.9 * (1 + math.cos(math.pi * i / 5)) / 2 for i in range(6)]
                    else:
                        expected += [1.0, 1.0, 1.0, 1.0, 0.55, 0.1]
                    for ratio in expected:
                        groups = wrapper.optimizers["adamw"].param_groups
                        self.assertAlmostEqual(groups[0]["lr"], config.lr * ratio)
                        self.assertAlmostEqual(groups[2]["lr"], config.lr * ratio * 0.25)
                        if kind == "muon_adamw":
                            self.assertAlmostEqual(wrapper.optimizers["muon"].param_groups[0]["lr"], config.lr * ratio * 0.4)
                        wrapper.step_optimizer()
                        wrapper.step_schedule()
                    self.assertAlmostEqual(groups[0]["lr"], config.schedule.min_lr)

    def test_schedule_snapshot_and_restoration(self):
        config = self.config("muon_adamw")
        model = DecoderTransformer(model_config())
        wrapper = Optimizer(config, model, total_steps=8)
        for _ in range(3):
            model.zero_grad()
            model(torch.randint(0, 32, (2, 8))).square().mean().backward()
            wrapper.step_optimizer()
            wrapper.step_schedule()
        checkpoint = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizers": {name: copy.deepcopy(opt.state_dict()) for name, opt in wrapper.optimizers.items()},
            "schedulers": {name: copy.deepcopy(s.state_dict()) for name, s in wrapper.schedulers.items()},
        }
        restored_model = DecoderTransformer(model_config())
        restored_model.load_state_dict(checkpoint["model"])
        restored = Optimizer(config, restored_model, total_steps=8)
        for name, opt in restored.optimizers.items():
            opt.load_state_dict(checkpoint["optimizers"][name])
            restored.schedulers[name].load_state_dict(checkpoint["schedulers"][name])
        inputs = torch.randint(0, 32, (2, 8))
        for active_model, active in ((model, wrapper), (restored_model, restored)):
            active_model.zero_grad()
            active_model(inputs).square().mean().backward()
            active.step_optimizer()
            active.step_schedule()
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, restored_model.state_dict()[name], rtol=0, atol=0)
        config.lr = 0.5
        self.assertEqual(wrapper.config.lr, 0.02)
        self.assertEqual(wrapper.total_steps, 8)

    def test_invalid_horizon_and_missing_trainable_parameters_fail(self):
        model = DecoderTransformer(model_config())
        with self.assertRaises(TypeError):
            Optimizer(OptimizerConfig(), model)
        for steps in (0, -1, True, 8.0):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "positive integer"):
                Optimizer(self.config(), model, total_steps=steps)
        with self.assertRaises(ValidationError):
            OptimizerConfig(schedule=ScheduleConfig(min_lr=0.1))
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            Optimizer(self.config(), model, total_steps=2)
        for p in model.parameters():
            p.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "no trainable"):
            Optimizer(self.config(), model, total_steps=8)
        with self.assertRaisesRegex(ValueError, "eligible hidden"):
            Optimizer(self.config("muon_adamw"), torch.nn.Embedding(32, 16), total_steps=8)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "requires BF16 CUDA")
class OptimizerCudaTests(unittest.TestCase):
    def test_fused_adamw_and_muon_update_fp32_parameters_under_autocast(self):
        for kind in ("adamw", "muon_adamw"):
            with self.subTest(kind=kind):
                model = DecoderTransformer(model_config(canon_layers=True, split_qkv_projections=True)).cuda()
                config = OptimizerConfig(
                    kind=kind, schedule=ScheduleConfig(warmup_steps=1),
                )
                optimizer = Optimizer(config, model, total_steps=4)
                before = model.embeddings.weight.detach().clone()
                inputs = torch.randint(0, 32, (2, 8), device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), inputs.flatten())
                loss.backward()
                optimizer.step_optimizer()
                optimizer.step_schedule()
                self.assertFalse(torch.equal(before, model.embeddings.weight))
                for parameter in model.parameters():
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(parameter).all())
                    self.assertTrue(torch.isfinite(parameter.grad).all())


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = PretrainingConfig.from_dict({
            "name": "loaders", "model": model_config().to_dict(),
            "training": {"batch_size": 3},
            "runtime": {"num_workers": 0, "pin_memory": False, "dataset_dir": "/unused"},
            "train_dataset": {"path": "unused"}, "validation_dataset": {"path": "unused"},
        })

    def tearDown(self):
        self.directory.cleanup()

    def artifact(self, name, token_count=81, sample=None):
        path = self.root / name
        path.mkdir()
        shards = []
        for index, start in enumerate(range(0, token_count, 7)):
            tokens = list(range(start, min(start + 7, token_count)))
            filename = f"tokens-{index:04d}.bin"
            (path / filename).write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
            shards.append({"file": filename, "tokens": len(tokens)})
        metadata = {"storage": {"dtype": "uint16", "endianness": "little", "layout": "flat_token_stream"},
                    "token_count": token_count, "shards": shards}
        if sample is not None:
            metadata["fixed_evaluation_sample"] = {
                "sequence_length": 8, "block_count": len(sample),
                "blocks": [{"start_token": start} for start in sample],
            }
        (path / "metadata.json").write_text(json.dumps(metadata))
        return path

    def test_explicit_paths_shifted_blocks_and_validation_tail(self):
        path = self.artifact("arbitrary-name")
        validation = ValidationDataLoader(path, self.config)
        self.assertEqual([len(x) for x, _ in validation], [3, 3, 3, 1])
        blocks = torch.cat([x for x, _ in validation])
        targets = torch.cat([y for _, y in validation])
        self.assertEqual(blocks[1].tolist(), list(range(8, 16)))
        self.assertTrue(torch.equal(targets, blocks + 1))
        self.assertEqual(validation.dataset.shard_paths[0].parent, path)
        training = TrainDataLoader(path, self.config)
        self.assertEqual([len(x) for x, _ in training], [3, 3, 3])
        self.assertFalse(training.persistent_workers)
        self.assertIsNone(training.prefetch_factor)

    def test_epoch_shuffle_and_consumed_cursor_leave_model_rng_alone(self):
        loader = TrainDataLoader(self.artifact("train"), self.config)
        rng = torch.get_rng_state().clone()
        batches = [x.clone() for x, _ in loader]
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        loader.set_epoch(0)
        repeated = [x for x, _ in loader]
        for first, second in zip(batches, repeated):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        loader.set_epoch(0, start_batch=1)
        remaining = [x for x, _ in loader]
        self.assertEqual(len(remaining), 2)
        for first, second in zip(batches[1:], remaining):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        loader.set_epoch(1)
        self.assertFalse(torch.equal(batches[0], next(iter(loader))[0]))

    def test_fixed_validation_sample_is_complete_and_repeatable(self):
        starts = [0, 11, 25, 38, 50]
        path = self.artifact("fixed", sample=starts)
        loader = ValidationDataLoader(path, self.config)
        batches = [x for x, _ in loader]
        self.assertEqual([len(x) for x in batches], [3, 2])
        self.assertEqual(torch.cat(batches)[:, 0].tolist(), starts)
        self.assertEqual(torch.cat([x for x, _ in loader])[:, 0].tolist(), starts)
        config = self.config.with_overrides({"evaluation.steps": 1})
        with self.assertRaisesRegex(ValueError, "complete fixed"):
            ValidationDataLoader(path, config)
        with self.assertRaisesRegex(ValueError, "must not be used as training"):
            TrainDataLoader(path, self.config)
        with self.assertRaisesRegex(ValueError, "model.seq_len"):
            TokenBlockDataset(path, 4)

    def test_worker_prefetch_does_not_advance_resume_cursor(self):
        config = self.config.with_overrides({
            "runtime.num_workers": 1, "runtime.persistent_workers": False,
            "runtime.prefetch_factor": 3,
        })
        loader = TrainDataLoader(self.artifact("prefetch"), config)
        expected = list(loader.batch_sampler)
        iterator = iter(loader)
        next(iterator)
        self.assertEqual(loader.batch_sampler.start_batch, 0)
        del iterator
        loader.set_epoch(0, start_batch=1)
        self.assertEqual(list(loader.batch_sampler), expected[1:])

    def test_corrupt_shards_and_small_training_data_fail(self):
        small = self.artifact("small", token_count=17)
        with self.assertRaisesRegex(ValueError, "too small"):
            TrainDataLoader(small, self.config)
        path = self.artifact("corrupt")
        (path / "tokens-0000.bin").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "bytes, expected"):
            ValidationDataLoader(path, self.config)


if __name__ == "__main__":
    unittest.main()
