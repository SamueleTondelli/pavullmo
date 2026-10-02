"""GQA retained while the evaluation branch adopts ModelConfig/trainer APIs."""

from __future__ import annotations

import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from pydantic import ValidationError

from config import ModelConfig, PretrainingConfig
from model import DecoderTransformer
from pavullmo.generate import load_model
from pavullmo.migrate_checkpoint_config import migrate_checkpoint_config
import test_trainer as fixtures


def model_config(**kwargs):
    return ModelConfig(vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4,
                       ffn_dim=24, seq_len=8, **kwargs)


class GQAConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_head_validation_and_gqa_matches_explicit_kv_repetition(self):
        for heads in (-1, 3, 6):
            with self.subTest(heads=heads), self.assertRaises(ValidationError):
                model_config(num_kv_heads=heads)
        sdpa = F.scaled_dot_product_attention

        def repeated_kv(query, key, value, **kwargs):
            if kwargs.pop('enable_gqa', False):
                repeats = query.shape[-3] // key.shape[-3]
                key = key.repeat_interleave(repeats, dim=-3)
                value = value.repeat_interleave(repeats, dim=-3)
            return sdpa(query, key, value, **kwargs)

        for heads, split, canon in itertools.product((0, 1, 2, 4), (False, True), (False, True)):
            with self.subTest(heads=heads, split=split, canon=canon):
                model = DecoderTransformer(model_config(num_kv_heads=heads,
                    split_qkv_projections=split, canon_layers=canon, qk_norm=True)).eval()
                inputs = torch.tensor([[1, 4, 7, 2]])
                with torch.inference_mode():
                    actual = model(inputs)
                    with patch('model.model.F.scaled_dot_product_attention', side_effect=repeated_kv):
                        expected = model(inputs)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(model.layers[0].attn.num_kv_heads, heads or 4)

    def test_legacy_gqa_migration_preserves_logits_and_infers_missing_heads(self):
        for heads, split, metadata in itertools.product(
            (1, 2, 4), (False, True), ('recorded', 'kv_heads', 'missing', 'none')
        ):
            with self.subTest(heads=heads, split=split, metadata=metadata), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = root / 'runs' / 'legacy'
                models_dir = root / 'models'
                run_dir.mkdir(parents=True)
                models_dir.mkdir(parents=True)
                config = model_config(num_kv_heads=heads, split_qkv_projections=split,
                                      canon_layers=True, qk_norm=True)
                model = DecoderTransformer(config).eval()
                values = {name.upper(): value for name, value in config.to_dict().items()}
                if metadata == 'kv_heads':
                    values['KV_HEADS'] = values.pop('NUM_KV_HEADS')
                elif metadata == 'missing':
                    values.pop('NUM_KV_HEADS')
                elif metadata == 'none':
                    values['NUM_KV_HEADS'] = None
                torch.save(
                    {'global_step': 0, 'hyperparameters': values, 'model_state_dict': model.state_dict()},
                    models_dir / 'legacy.pt',
                )
                migrate_checkpoint_config(run_dir, models_dir=models_dir)
                loaded, _, _ = load_model(run_dir, torch.device('cpu'))
                inputs = torch.tensor([[1, 4, 7, 2]])
                with torch.inference_mode():
                    torch.testing.assert_close(loaded(inputs), model(inputs), rtol=0, atol=0)
                self.assertEqual(loaded.layers[0].attn.num_kv_heads, heads)

    def test_gqa_pretraining_resume_preserves_trajectory(self):
        case = fixtures.TrainerTests()
        case.setUp()
        try:
            config = case.config().with_overrides({'model.num_kv_heads': 2, 'training.max_steps': 2})
            baseline = case.trainer(config)
            baseline.train()
            expected = copy.deepcopy(baseline.state_dict())
            interrupted = case.trainer(config)
            with case.stop_at(interrupted, 1):
                interrupted.train()
            restored = case.resume(interrupted.checkpoints.latest_checkpoint)
            restored.train()
            case.assert_state_equal(expected, restored.state_dict())
        finally:
            case.tearDown()

    def test_default_model_deepened_with_gqa_and_35m_preset_uses_mqa(self):
        defaults = ModelConfig()
        self.assertEqual(defaults.n_blocks, 13)
        self.assertEqual(defaults.num_kv_heads, 2)
        preset = PretrainingConfig.load(
            Path(__file__).resolve().parents[1] / "src/pavullmo/configs/35m_config.toml",
            overrides={"name": "test"},
        )
        self.assertEqual(preset.model.attn_heads, 7)
        self.assertEqual(preset.model.num_kv_heads, 1)

    def test_mha_signature_from_before_gqa_config_is_accepted(self):
        case = fixtures.TrainerTests()
        case.setUp()
        try:
            trainer = case.trainer(case.config().with_overrides({'model.num_kv_heads': 0}))
            state = trainer.state_dict()
            state['training_signature']['model'].pop('num_kv_heads')
            trainer.load_state_dict(state)
            self.assertEqual(trainer.global_step, 0)
        finally:
            case.tearDown()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GroupedQueryAttentionCudaTests(unittest.TestCase):
    def test_canon_grouped_attention_compiles_with_bf16_gradients(self):
        for split in (False, True):
            with self.subTest(split=split):
                torch.manual_seed(0)
                model = DecoderTransformer(model_config(
                    num_kv_heads=2,
                    qk_norm=True,
                    split_qkv_projections=split,
                    canon_layers=True,
                )).cuda()
                tokens = torch.randint(0, 32, (2, 8), device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = model(tokens)
                expected.float().square().mean().backward()
                expected_grads = {
                    name: parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                }
                model.zero_grad(set_to_none=True)
                compiled = torch.compile(model, fullgraph=True, dynamic=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = compiled(tokens)
                actual.float().square().mean().backward()
                torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.03)
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    torch.testing.assert_close(
                        parameter.grad, expected_grads[name], atol=0.01, rtol=0.05
                    )


if __name__ == '__main__':
    unittest.main()
