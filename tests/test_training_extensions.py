"""Tests for embedding weight decay, constant schedule, pre-decay saves, and EMA."""

from __future__ import annotations

import copy
import unittest

import torch

from components.optimizer import Optimizer, wsd_decay_start_step
from config import ModelConfig, OptimizerConfig, ScheduleConfig
from model import DecoderTransformer
from pavullmo import generate
import test_trainer as fixtures


class TrainingExtensionTests(unittest.TestCase):
    def test_embedding_weight_decay_routing(self):
        model = DecoderTransformer(ModelConfig(
            vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4,
            num_kv_heads=2, ffn_dim=24, seq_len=8,
        ))
        without = Optimizer(
            OptimizerConfig(
                kind="muon_adamw",
                adamw={"weight_decay": 0.05, "embedding_weight_decay": False, "fused": False},
            ),
            model,
            total_steps=20,
        )
        self.assertEqual(len(without.parameter_groups["adamw"][0]["params"]), 0)
        self.assertIn(
            id(model.embeddings.weight),
            {id(p) for p in without.parameter_groups["adamw"][1]["params"]},
        )

        with_decay = Optimizer(
            OptimizerConfig(
                kind="muon_adamw",
                adamw={"weight_decay": 0.05, "embedding_weight_decay": True, "fused": False},
            ),
            model,
            total_steps=20,
        )
        self.assertEqual(with_decay.parameter_groups["adamw"][0]["weight_decay"], 0.05)
        self.assertIn(
            id(model.embeddings.weight),
            {id(p) for p in with_decay.parameter_groups["adamw"][0]["params"]},
        )
        self.assertNotIn(
            id(model.embeddings.weight),
            {id(p) for p in with_decay.parameter_groups["adamw"][1]["params"]},
        )

    def test_constant_schedule_and_wsd_decay_start(self):
        model = DecoderTransformer(ModelConfig(
            vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4,
            num_kv_heads=2, ffn_dim=24, seq_len=8,
        ))
        opt = Optimizer(
            OptimizerConfig(
                kind="adamw",
                lr=0.01,
                adamw={"fused": False},
                schedule={"kind": "constant", "warmup_steps": 2},
            ),
            model,
            total_steps=10,
        )
        self.assertIsNone(opt.decay_start_step)
        lrs = []
        for _ in range(10):
            lrs.append(opt.optimizers["adamw"].param_groups[0]["lr"])
            opt.step_schedule()
        self.assertAlmostEqual(lrs[0], 0.005)
        for lr in lrs[1:]:
            self.assertAlmostEqual(lr, 0.01)

        wsd = ScheduleConfig(kind="wsd", warmup_steps=2, decay_fraction=0.4)
        self.assertEqual(wsd_decay_start_step(wsd, 12), 8)

    def test_ema_tracking_loading_and_pre_decay_checkpoint_protection(self):
        case = fixtures.TrainerTests()
        case.setUp()
        try:
            config = case.config(optimizer="muon_adamw", schedule="wsd").with_overrides({
                "optimizer.adamw.embedding_weight_decay": True,
                "optimizer.adamw.weight_decay": 0.05,
                "optimizer.schedule.warmup_steps": 1,
                "optimizer.schedule.decay_fraction": 0.5,
                "training.max_steps": 5,
                "training.ema_decay": 0.9,
                "training.ema_warmup": True,
                "checkpoint.keep_last": 1,
                "checkpoint.save_pre_decay": True,
            })
            baseline = case.trainer(config)
            baseline.train()
            expected = copy.deepcopy(baseline.state_dict())

            # Pre-decay checkpoint at step 3 must survive keep_last=1 alongside final step 5.
            ckpts = sorted(p.name for p in (baseline.run_dir / "checkpoints").glob("step_*.pt"))
            self.assertEqual(ckpts, ["step_00000003.pt", "step_00000005.pt"])
            self.assertIsNotNone(baseline.ema_validation_loss)

            raw_model, _, _ = generate.load_model(baseline.run_dir, torch.device("cpu"), use_ema=False)
            ema_model, _, _ = generate.load_model(baseline.run_dir, torch.device("cpu"), use_ema=True)
            diffs = [
                (raw_p - ema_p).abs().max().item()
                for raw_p, ema_p in zip(raw_model.parameters(), ema_model.parameters())
            ]
            self.assertGreater(max(diffs), 0.0)

            # Verify exact resume trajectory including EMA state.
            interrupted = case.trainer(config.with_overrides({"name": "resumed"}))
            with case.stop_at(interrupted, 2):
                interrupted.train()
            restored = case.resume(interrupted.checkpoints.latest_checkpoint)
            restored.train()
            case.assert_state_equal(expected, restored.state_dict())
        finally:
            case.tearDown()


if __name__ == "__main__":
    unittest.main()
