from __future__ import annotations

import platform

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    print(f"Python:          {platform.python_version()}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"PyTorch CUDA:    {torch.version.cuda}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU:             {properties.name}")
    print(f"Compute cap.:    {properties.major}.{properties.minor}")
    print(f"VRAM:            {properties.total_memory / 2**30:.1f} GiB")
    print(f"BF16 supported:  {torch.cuda.is_bf16_supported()}")

    # A tiny CUDA operation catches broken driver/runtime combinations.
    x = torch.randn(256, 256, device=device, dtype=torch.float16)
    torch.mm(x, x)
    torch.cuda.synchronize()
    print("CUDA smoke test: passed")

    query, key, value = (
        torch.randn(
            1, 4, 128, 64, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        output.float().square().mean().backward()
    torch.cuda.synchronize()
    print("PyTorch FA2 test: passed (forward and backward)")
