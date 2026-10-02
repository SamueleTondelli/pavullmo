"""Grouped-query attention keeps causality and supports both projection layouts."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model.model import CausalSelfAttention, DecoderTransformer
from src.evaluation.evaluate_loss import load_model as load_loss_model
from src.pavullmo.generate import load_model as load_generation_model
from src.scaling.evaluate_base import load_model as load_scaling_model


class GroupedQueryAttentionTests(unittest.TestCase):
    def test_grouped_attention_matches_explicit_kv_replication(self):
        torch.manual_seed(0)
        for split in (False, True):
            with self.subTest(split=split):
                attention = CausalSelfAttention(
                    num_heads=4, num_kv_heads=2, embed_dimension=16,
                    seq_len=8, qk_norm=True, split_qkv_projections=split,
                ).eval()
                inputs = torch.randn(2, 6, 16, requires_grad=True)
                projected = (
                    F.linear(inputs, torch.cat([
                        attention.q_proj.weight, attention.k_proj.weight,
                        attention.v_proj.weight,
                    ])) if split else attention.c_attn(inputs)
                )
                q, k, v = projected.split((16, 8, 8), dim=-1)
                q = attention.rope(attention.q_norm(q.view(2, 6, 4, 4).transpose(1, 2)))
                k = attention.rope(attention.k_norm(k.view(2, 6, 2, 4).transpose(1, 2)))
                v = v.view(2, 6, 2, 4).transpose(1, 2)
                expected = F.scaled_dot_product_attention(
                    q, k.repeat_interleave(2, dim=1),
                    v.repeat_interleave(2, dim=1), is_causal=True,
                )
                expected = attention.c_proj(expected.transpose(1, 2).reshape(2, 6, 16))
                actual = attention(inputs)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                actual.sum().backward()
                self.assertIsNotNone(inputs.grad)

    def test_model_with_canon_is_causal(self):
        for split in (False, True):
            with self.subTest(split=split):
                model = DecoderTransformer(
                    vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4,
                    num_kv_heads=2, ffn_dim=32, dropout=0, seq_len=8,
                    split_qkv_projections=split, canon_layers=True,
                ).eval()
                self.assertEqual(tuple(model.layers[0].attn.cb.weight.shape), (32, 1, 4))
                tokens = torch.randint(0, 32, (1, 8))
                changed = tokens.clone()
                changed[:, 5:] = (changed[:, 5:] + 1) % 32
                torch.testing.assert_close(model(tokens)[:, :5], model(changed)[:, :5])

    def test_invalid_kv_head_count(self):
        for count in (0, 3, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                CausalSelfAttention(4, 16, num_kv_heads=count)

    def test_legacy_default_retains_full_attention_shapes(self):
        attention = CausalSelfAttention(4, 16)
        self.assertEqual(attention.num_kv_heads, 4)
        self.assertEqual(tuple(attention.c_attn.weight.shape), (48, 16))

    def test_generation_loads_legacy_and_grouped_checkpoints(self):
        for kv_heads in (4, 2):
            with self.subTest(kv_heads=kv_heads), tempfile.TemporaryDirectory() as folder:
                model = DecoderTransformer(
                    vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4,
                    num_kv_heads=kv_heads, ffn_dim=32, dropout=0, seq_len=8,
                )
                settings = {
                    "VOCAB_SIZE": 32, "N_BLOCKS": 1, "EMBED_DIM": 16,
                    "ATTN_HEADS": 4, "FFN_DIM": 32, "DROPOUT": 0,
                    "SEQ_LEN": 8, "ROPE_BASE": 10000.0, "QK_NORM": False,
                }
                if kv_heads != 4:
                    settings["KV_HEADS"] = kv_heads
                checkpoint = Path(folder) / "model.pt"
                torch.save({"hyperparameters": settings,
                            "model_state_dict": model.state_dict()}, checkpoint)
                for loader in (load_generation_model, load_loss_model, load_scaling_model):
                    restored = loader(checkpoint, torch.device("cpu"))[0]
                    self.assertEqual(restored.layers[0].attn.num_kv_heads, kv_heads)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GroupedQueryAttentionCudaTests(unittest.TestCase):
    def test_canon_grouped_attention_compiles_with_bf16_gradients(self):
        for split in (False, True):
            with self.subTest(split=split):
                torch.manual_seed(0)
                model = DecoderTransformer(
                    vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4,
                    num_kv_heads=2, ffn_dim=32, dropout=0, seq_len=8,
                    qk_norm=True, split_qkv_projections=split,
                    canon_layers=True,
                ).cuda()
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


if __name__ == "__main__":
    unittest.main()
