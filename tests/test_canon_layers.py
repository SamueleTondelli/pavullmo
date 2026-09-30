"""Regression tests for the Canon-ABCD Transformer integration."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CanonLayerCudaTests(unittest.TestCase):
    def test_compiled_model_trains_with_bf16_and_both_qkv_layouts(self):
        for split_qkv in (False, True):
            with self.subTest(split_qkv=split_qkv):
                torch.manual_seed(0)
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
                ).cuda()
                token_ids = torch.randint(0, 32, (2, 8), device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = model(token_ids)
                expected.float().square().mean().backward()
                expected_grads = {
                    name: param.grad.clone() for name, param in model.named_parameters()
                }
                model.zero_grad(set_to_none=True)
                compiled = torch.compile(model, fullgraph=True, dynamic=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = compiled(token_ids)
                actual.float().square().mean().backward()
                torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.03)
                for name, param in model.named_parameters():
                    self.assertEqual(param.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(param.grad).all())
                    torch.testing.assert_close(
                        param.grad, expected_grads[name], atol=0.01, rtol=0.05
                    )
                layer = model.layers[0]
                for canon in (layer.ca, layer.attn.cb, layer.cc, layer.ffn.cd):
                    self.assertGreater(canon.weight.grad.abs().sum().item(), 0)

    def test_kernel_matches_conv1d_outputs_and_gradients(self):
        from causal_conv1d import causal_conv1d_fn

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for channels in (3, 16):
                for width in (2, 3, 4):
                    for layout in ("contiguous", "sequence_slice", "channel_slice"):
                        with self.subTest(
                            dtype=dtype, channels=channels, width=width, layout=layout
                        ):
                            layer = CanonLayer(channels, width).cuda()
                            if layout == "sequence_slice":
                                x = torch.randn(
                                    2, 34, channels, device="cuda", dtype=dtype
                                )[:, ::2]
                            elif layout == "channel_slice":
                                x = torch.randn(
                                    2, 17, channels * 2, device="cuda", dtype=dtype
                                )[..., ::2]
                            else:
                                x = torch.randn(
                                    2, 17, channels, device="cuda", dtype=dtype
                                )
                            x = x.detach().requires_grad_()
                            x_ref = x.detach().clone().requires_grad_()
                            weight_ref = layer.weight.detach().clone().requires_grad_()
                            with patch(
                                "causal_conv1d.causal_conv1d_fn", wraps=causal_conv1d_fn
                            ) as kernel:
                                actual = layer(x)
                            kernel.assert_called_once()
                            expected = x_ref + F.conv1d(
                                x_ref.float().transpose(1, 2),
                                weight_ref,
                                padding=width - 1,
                                groups=channels,
                            )[..., :17].transpose(1, 2).to(dtype)
                            tol = {
                                torch.float32: 1e-5,
                                torch.float16: 0.005,
                                torch.bfloat16: 0.03,
                            }[dtype]
                            torch.testing.assert_close(
                                actual, expected, atol=tol, rtol=tol
                            )
                            grad = torch.randn_like(actual)
                            actual.backward(grad)
                            expected.backward(grad)
                            torch.testing.assert_close(
                                x.grad, x_ref.grad, atol=tol, rtol=tol
                            )
                            torch.testing.assert_close(
                                layer.weight.grad, weight_ref.grad, atol=tol, rtol=tol
                            )

    def test_autocast_and_checkpoint_compatibility(self):
        from causal_conv1d import causal_conv1d_fn

        original = torch.nn.Conv1d(16, 16, 4, groups=16, padding=3, bias=False)
        layer = CanonLayer(16).cuda()
        layer.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(tuple(layer.weight.shape), (16, 1, 4))
        x = torch.randn(2, 17, 16, device="cuda", requires_grad=True)
        with patch("causal_conv1d.causal_conv1d_fn", wraps=causal_conv1d_fn) as kernel:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual = layer(x)
                expected = x + F.conv1d(
                    x.transpose(1, 2), layer.weight, padding=3, groups=16
                )[..., :17].transpose(1, 2)
        self.assertEqual(kernel.call_args.args[0].dtype, torch.bfloat16)
        self.assertEqual(kernel.call_args.args[1].dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        actual.square().mean().backward()
        self.assertEqual(layer.weight.grad.dtype, torch.float32)

    def test_kernel_preserves_causality_and_unsupported_widths(self):
        for width in (1, 4, 5):
            with self.subTest(width=width):
                layer = CanonLayer(16, width).cuda()
                x = torch.randn(2, 17, 16, device="cuda")
                changed = x.clone()
                changed[:, 8:] = -100
                actual = layer(x)
                torch.testing.assert_close(actual[:, :8], layer(changed)[:, :8])
                expected = x + F.conv1d(
                    x.transpose(1, 2), layer.weight, padding=width - 1, groups=16
                )[..., :17].transpose(1, 2)
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
