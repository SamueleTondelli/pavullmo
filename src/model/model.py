import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dimension: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

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
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)
        return y


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
    ):
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, attn_heads: int, ffn_dim: int, dropout: float):
        self.embed_dim = embed_dim
        self.attn_norm = nn.RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(
            attn_heads, embed_dim, bias=False, dropout=dropout
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = SwiGLUFFN(embed_dim, ffn_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        pre_attn = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.attn_dropout(x)
        x = pre_attn + x

        pre_ffn = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        return pre_ffn + x
