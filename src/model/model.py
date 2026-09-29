import torch
import torch.nn as nn
import torch.nn.functional as F


INITIALIZATION_RECIPES = frozenset({"pytorch_default", "olmo", "gpt_scaled"})
BASE_INITIALIZATION_STD = 0.02
CANON_KERNEL_SIZE = 4


class CanonLayer(nn.Conv1d):
    """Causal depthwise convolution with an explicit residual connection."""

    def __init__(self, channels: int, kernel_size: int = CANON_KERNEL_SIZE):
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=kernel_size - 1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Canon input must have shape [batch, sequence, channels]")
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {x.size(-1)}"
            )

        sequence_length = x.size(1)
        convolved = super().forward(x.transpose(1, 2))
        # Conv1d pads both sides. Keeping the first sequence_length outputs makes
        # the receptive field causal: output t only sees inputs t-k+1 through t.
        convolved = convolved[..., :sequence_length].transpose(1, 2)
        return x + convolved


class RoPE(nn.Module):
    def __init__(self, seq_len: int, head_dim: int, base: float = 10000.0):
        super().__init__()

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive, even number")
        if base <= 0:
            raise ValueError("base must be positive")

        self.seq_len = seq_len
        self.head_dim = head_dim

        # pre-compute RoPE frequencies
        inv_freq = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )

        positions = torch.arange(seq_len, dtype=torch.float32)
        # [seq_len, head_dim / 2]
        angles = torch.outer(positions, inv_freq)

        # Repeat each angle for the two dimensions in a pair.
        # [seq_len, head_dim]
        angles = torch.repeat_interleave(angles, 2, dim=-1)

        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("RoPE input must have at least two dimensions")
        if x.size(-1) != self.head_dim:
            raise ValueError(
                f"expected head dimension {self.head_dim}, got {x.size(-1)}"
            )

        sequence_length = x.size(-2)
        if sequence_length > self.seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds RoPE limit {self.seq_len}"
            )

        # Add leading singleton dimensions so the cache broadcasts over batches
        # and attention heads. Casting avoids promoting fp16/bf16 activations to fp32.
        cos = self.rope_cos[:sequence_length].to(dtype=x.dtype)
        sin = self.rope_sin[:sequence_length].to(dtype=x.dtype)
        for _ in range(x.ndim - 2):
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return x * cos + self._rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dimension: int,
        bias: bool = False,
        dropout: float = 0.0,
        seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        split_qkv_projections: bool = False,
        canon_layers: bool = False,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dimension % num_heads != 0:
            raise ValueError("embed_dimension must be divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        head_dim = embed_dimension // num_heads
        self.rope = RoPE(seq_len, head_dim, base=rope_base)
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.split_qkv_projections = split_qkv_projections
        if split_qkv_projections:
            self.q_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
            self.k_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
            self.v_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        else:
            # Preserve the original parameter layout and checkpoint keys.
            self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension

        self.cb = CanonLayer(3 * embed_dimension) if canon_layers else None

    def forward(self, x):
        if self.split_qkv_projections:
            qkv_weight = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
            )
            qkv_bias = (
                None
                if self.q_proj.bias is None
                else torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
                )
            )
            query_projected = F.linear(x, qkv_weight, qkv_bias)
        else:
            query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        sequence_length = query_projected.size(1)
        head_dim = self.embed_dimension // self.num_heads

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(
            batch_size, sequence_length, self.num_heads, head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.num_heads, head_dim
        ).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        if self.cb is not None:
            # Canon-B acts jointly on the normalized Q, K, and V projections.
            # Move Q/K back to dimension-last before applying the sequence mixer.
            projected = torch.cat(
                [
                    query.transpose(1, 2).reshape(
                        batch_size, sequence_length, self.embed_dimension
                    ),
                    key.transpose(1, 2).reshape(
                        batch_size, sequence_length, self.embed_dimension
                    ),
                    value,
                ],
                dim=-1,
            )
            query, key, value = self.cb(projected).chunk(3, dim=-1)
            query = query.view(
                batch_size, sequence_length, self.num_heads, head_dim
            ).transpose(1, 2)
            key = key.view(
                batch_size, sequence_length, self.num_heads, head_dim
            ).transpose(1, 2)

        value = value.view(
            batch_size, sequence_length, self.num_heads, head_dim
        ).transpose(1, 2)
        query = self.rope(query)
        key = self.rope(key)

        if self.training:
            dropout = self.dropout
        else:
            dropout = 0.0

        y = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(batch_size, -1, self.embed_dimension)
        return self.c_proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, canon_layers: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.cd = CanonLayer(hidden_dim * 2) if canon_layers else None

    def forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        if self.cd is not None:
            cat = torch.cat([gate, up], dim=-1)
            gate, up = self.cd(cat).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_heads: int,
        ffn_dim: int,
        dropout: float,
        seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        split_qkv_projections: bool = False,
        canon_layers: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_norm = nn.RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(
            attn_heads,
            embed_dim,
            bias=False,
            dropout=dropout,
            seq_len=seq_len,
            rope_base=rope_base,
            qk_norm=qk_norm,
            split_qkv_projections=split_qkv_projections,
            canon_layers=canon_layers,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.RMSNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, ffn_dim, canon_layers=canon_layers)
        self.ffn_dropout = nn.Dropout(dropout)

        self.canon_layers = canon_layers
        if canon_layers:
            self.ca = CanonLayer(embed_dim)
            self.cc = CanonLayer(embed_dim)

    def forward(self, x):
        pre_attn = x
        x = self.attn_norm(x)
        if self.canon_layers:
            x = self.ca(x)
        x = self.attn(x)
        x = self.attn_dropout(x)
        x = pre_attn + x

        pre_ffn = x
        x = self.ffn_norm(x)
        if self.canon_layers:
            x = self.cc(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        return pre_ffn + x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_blocks: int,
        embed_dim: int,
        attn_heads: int,
        ffn_dim: int,
        dropout: float,
        seq_len: int = 2048,
        rope_base: float = 10000.0,
        initialization: str = "pytorch_default",
        initialization_std: float = BASE_INITIALIZATION_STD,
        qk_norm: bool = False,
        split_qkv_projections: bool = False,
        canon_layers: bool = False,
    ):
        super().__init__()
        initialization = initialization.strip().lower()
        if initialization not in INITIALIZATION_RECIPES:
            choices = ", ".join(sorted(INITIALIZATION_RECIPES))
            raise ValueError(
                f"unknown initialization recipe {initialization!r}; choose from {choices}"
            )
        if initialization_std <= 0.0:
            raise ValueError("initialization_std must be positive")

        self.initialization = initialization
        self.initialization_std = initialization_std
        self.n_blocks = n_blocks
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            TransformerBlock(
                embed_dim,
                attn_heads,
                ffn_dim,
                dropout,
                seq_len=seq_len,
                rope_base=rope_base,
                qk_norm=qk_norm,
                split_qkv_projections=split_qkv_projections,
                canon_layers=canon_layers,
            )
            for _ in range(n_blocks)
        )
        self.norm = nn.RMSNorm(embed_dim)

        if initialization != "pytorch_default":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Apply the configured transformer initialization recipe."""

        if self.initialization == "pytorch_default":
            for module in self.modules():
                if module is not self and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            return

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.initialization_std,
                )
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                module.reset_parameters()

        # Canon layers intentionally retain Conv1d's default fan-in
        # initialization instead of the transformer's linear-layer std.

        if self.initialization == "gpt_scaled":
            residual_std = self.initialization_std / (2 * self.n_blocks) ** 0.5
            for layer in self.layers:
                nn.init.normal_(layer.attn.c_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(layer.ffn.w2.weight, mean=0.0, std=residual_std)

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return F.linear(x, self.embeddings.weight, bias=None)
