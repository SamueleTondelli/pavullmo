"""Run with .venv/bin/python -m unittest discover -s tests -v."""

from __future__ import annotations

import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for directory in ("model", "pavullmo", "evaluation", "scaling"):
    sys.path.insert(0, str(ROOT / "src" / directory))

from model import DecoderTransformer
from model_checkpoint import load_checkpoint_model
import evaluate_base
import evaluate_loss
import generate
import inspect_logits
import score_synthetic_benchmark


class CheckpointLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "model.pt"
        self.hyperparameters = {
            "VOCAB_SIZE": 32,
            "N_BLOCKS": 2,
            "EMBED_DIM": 16,
            "ATTN_HEADS": 4,
            "FFN_DIM": 24,
            "SEQ_LEN": 8,
            "ROPE_BASE": 10000.0,
            "DROPOUT": 0.0,
        }

    def save_checkpoint(self, hyperparameters, state_dict) -> None:
        torch.save(
            {"hyperparameters": hyperparameters, "model_state_dict": state_dict},
            self.path,
        )

    def test_evaluation_loaders_preserve_logits(self) -> None:
        # Cover legacy MHA, GQA and MQA, both QKV layouts, and Canon/QK norm.
        loaders = (
            evaluate_loss.load_model,
            evaluate_base.load_model,
            generate.load_model,
            score_synthetic_benchmark.load_model,
            inspect_logits.load_model,
        )
        input_ids = torch.tensor([[1, 4, 7, 2], [1, 6, 5, 2]])
        variants = itertools.product((4, 2, 1), (False, True))
        for kv_heads, split_qkv in variants:
            for canon, qk_norm, recorded in itertools.product((False, True), repeat=3):
                with self.subTest(
                    kv_heads=kv_heads, split_qkv=split_qkv,
                    canon=canon, qk_norm=qk_norm, recorded=recorded,
                ):
                    original = DecoderTransformer(
                        vocab_size=32, n_blocks=2, embed_dim=16, attn_heads=4,
                        num_kv_heads=kv_heads, ffn_dim=24, dropout=0.0,
                        seq_len=8, qk_norm=qk_norm,
                        split_qkv_projections=split_qkv, canon_layers=canon,
                    ).eval()
                    hyperparameters = dict(self.hyperparameters)
                    if recorded:
                        hyperparameters.update(
                            NUM_KV_HEADS=kv_heads,
                            SPLIT_QKV_PROJECTIONS=split_qkv,
                            CANON_LAYERS=canon,
                            QK_NORM=qk_norm,
                        )
                    self.save_checkpoint(hyperparameters, original.state_dict())
                    with torch.inference_mode():
                        expected_logits = original(input_ids)
                    for loader in loaders:
                        with self.subTest(loader=loader.__module__):
                            loaded, *metadata = loader(self.path, torch.device("cpu"))
                            self.assertFalse(loaded.training)
                            self.assertEqual(loaded.layers[0].attn.num_kv_heads, kv_heads)
                            if loader in (evaluate_loss.load_model, evaluate_base.load_model):
                                self.assertEqual(metadata, [hyperparameters])
                            else:
                                self.assertEqual(metadata, [32, 8])
                            with torch.inference_mode():
                                torch.testing.assert_close(
                                    loaded(input_ids), expected_logits, rtol=0, atol=0,
                                )

    def test_none_kv_heads_metadata_infers_projection_width(self) -> None:
        original = DecoderTransformer(
            vocab_size=32, n_blocks=2, embed_dim=16, attn_heads=4,
            num_kv_heads=2, ffn_dim=24, dropout=0.0, seq_len=8,
        )
        self.hyperparameters["NUM_KV_HEADS"] = None
        self.save_checkpoint(self.hyperparameters, original.state_dict())
        loaded, _ = load_checkpoint_model(self.path, torch.device("cpu"))
        self.assertEqual(loaded.layers[0].attn.num_kv_heads, 2)

    def test_inconsistent_metadata_is_rejected(self) -> None:
        original = DecoderTransformer(
            vocab_size=32, n_blocks=2, embed_dim=16, attn_heads=4,
            num_kv_heads=2, ffn_dim=24, dropout=0.0, seq_len=8,
        )
        self.hyperparameters["NUM_KV_HEADS"] = 4
        self.save_checkpoint(self.hyperparameters, original.state_dict())
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            load_checkpoint_model(self.path, torch.device("cpu"))

    def test_invalid_projection_width_is_rejected(self) -> None:
        original = DecoderTransformer(
            vocab_size=32, n_blocks=2, embed_dim=16, attn_heads=4,
            num_kv_heads=2, ffn_dim=24, dropout=0.0, seq_len=8,
        )
        state_dict = original.state_dict()
        for rows, message in ((31, "equal K and V widths"), (28, "incompatible")):
            with self.subTest(rows=rows):
                state_dict["layers.0.attn.c_attn.weight"] = torch.zeros(rows, 16)
                self.save_checkpoint(self.hyperparameters, state_dict)
                with self.assertRaisesRegex(ValueError, message):
                    load_checkpoint_model(self.path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
