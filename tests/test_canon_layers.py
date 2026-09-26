"""Regression tests for the Canon-ABCD Transformer integration."""

import unittest

import torch

from src.model.model import CanonLayer, DecoderTransformer


class CanonLayerTests(unittest.TestCase):
    def test_depthwise_convolution_is_causal_and_residual(self):
        layer = CanonLayer(1, kernel_size=4)
        with torch.no_grad():
            layer.weight.fill_(1.0)

        inputs = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
        outputs = layer(inputs)

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertEqual(outputs.flatten().tolist(), [2.0, 5.0, 9.0, 14.0, 19.0])

        changed_future = inputs.clone()
        changed_future[:, 3:] = -100.0
        changed_outputs = layer(changed_future)
        torch.testing.assert_close(outputs[:, :3], changed_outputs[:, :3])

    def test_canon_abcd_shapes_and_forward_for_both_qkv_layouts(self):
        for split_qkv in (False, True):
            with self.subTest(split_qkv=split_qkv):
                model = DecoderTransformer(
                    vocab_size=32,
                    n_blocks=1,
                    embed_dim=16,
                    attn_heads=4,
                    ffn_dim=32,
                    dropout=0.0,
                    seq_len=8,
                    qk_norm=True,
                    split_qkv_projections=split_qkv,
                    canon_layers=True,
                )
                layer = model.layers[0]
                self.assertEqual(tuple(layer.ca.weight.shape), (16, 1, 4))
                self.assertEqual(tuple(layer.attn.cb.weight.shape), (48, 1, 4))
                self.assertEqual(tuple(layer.cc.weight.shape), (16, 1, 4))
                self.assertEqual(tuple(layer.ffn.cd.weight.shape), (64, 1, 4))

                token_ids = torch.randint(0, 32, (2, 8))
                logits = model(token_ids)
                self.assertEqual(tuple(logits.shape), (2, 8, 32))
                logits.sum().backward()
                for canon in (layer.ca, layer.attn.cb, layer.cc, layer.ffn.cd):
                    self.assertIsNotNone(canon.weight.grad)

    def test_full_model_does_not_leak_future_tokens(self):
        model = DecoderTransformer(
            vocab_size=32,
            n_blocks=2,
            embed_dim=16,
            attn_heads=4,
            ffn_dim=32,
            dropout=0.0,
            seq_len=8,
            canon_layers=True,
        ).eval()
        token_ids = torch.randint(0, 32, (1, 8))
        changed_future = token_ids.clone()
        changed_future[:, 5:] = (changed_future[:, 5:] + 1) % 32

        with torch.no_grad():
            logits = model(token_ids)
            changed_logits = model(changed_future)

        torch.testing.assert_close(logits[:, :5], changed_logits[:, :5])


if __name__ == "__main__":
    unittest.main()
