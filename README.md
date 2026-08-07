# pavullmo

A small (~50M parameter) Italian language model project.

## Training environment

The environment is pinned to one stack that works on both development and
training GPUs:

| Component | Version | Notes |
| --- | --- | --- |
| Python | 3.12 | Required by the prebuilt FlashAttention wheel |
| PyTorch | 2.9.1 | Installed from the official CUDA 12.8 index |
| CUDA runtime | 12.8 | Bundled with the PyTorch wheel; no system toolkit required |
| FlashAttention | 2.8.3 | Official CUDA 12 / PyTorch 2.9 / CPython 3.12 wheel |
| Monitoring | TensorBoard 2.21.0 | Free and local; writes portable event files |

CUDA 12.8 supports both the local RTX 3070 (Ampere, compute capability 8.6)
and Modal's L4 (Ada, compute capability 8.9). The NVIDIA driver is supplied by
the host machine or Modal and may report a newer maximum CUDA version; it does
not need to equal 12.8.

Create the virtual environment and verify the GPU stack:

```bash
uv sync
uv run pavullmo-check
```

Install the optional Modal client when cloud execution is added:

```bash
uv sync --extra cloud
uv run modal setup
```

The FlashAttention dependency is a prebuilt Linux x86-64 wheel, so `uv sync`
does not need `nvcc`, Ninja, or a local CUDA toolkit. If a different Python or
PyTorch version is selected later, the FlashAttention wheel must be changed to
an exactly matching build.

For a conventional decoder-only transformer, prefer PyTorch's
`torch.nn.functional.scaled_dot_product_attention` first. PyTorch can dispatch
it to a fused flash-attention kernel without importing `flash_attn`; use the
third-party package when its varlen or packed-sequence APIs are actually
needed.

## TensorBoard

Training code can log without a hosted account:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/pretrain")
writer.add_scalar("train/loss", loss.item(), step)
writer.add_scalar("train/learning_rate", learning_rate, step)
writer.add_scalar("train/tokens_per_second", tokens_per_second, step)
```

Start the UI locally:

```bash
uv run tensorboard --logdir runs --port 6006
```

On Modal, place `runs/` on a persistent `modal.Volume`. The TensorBoard event
files can then be downloaded and viewed locally, with no third-party tracking
service or subscription.

When a Modal entrypoint is added, build its image from the committed lockfile
with `modal.Image.debian_slim(python_version="3.12").uv_sync(frozen=True)` so
local and cloud runs use the same packages.
