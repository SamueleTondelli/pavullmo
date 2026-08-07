from __future__ import annotations

import platform

import torch


def main() -> None:
    print(f"Python:          {platform.python_version()}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"PyTorch CUDA:    {torch.version.cuda}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except ImportError:
        flash_attn_func = None
        print("FlashAttention:  not installed (supported only on Linux x86_64)")
    else:
        print(f"FlashAttention:  {flash_attn.__version__}")

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

    if flash_attn_func is not None:
        qkv = torch.randn(
            1, 128, 3, 4, 64, device=device, dtype=torch.float16
        )
        query, key, value = qkv.unbind(dim=2)
        flash_attn_func(query, key, value, causal=True)
        torch.cuda.synchronize()
        print("FlashAttn test:  passed")
