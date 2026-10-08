import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight).to(x.dtype)


class CausalAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.n_heads: int = n_heads
        self.head_dim: int = hidden_dim // n_heads

        self.qkv = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False),
        )
        self.q_ln = RMSNorm(self.head_dim)
        self.k_ln = RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # setting RoPE
        base = 10000.0
        max_length = 512  # Most SMILES sequences are shorter than this.
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        angles = torch.outer(torch.arange(max_length, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-2]
        cos, sin = self.cos[:length].to(x.dtype), self.sin[:length].to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.n_heads

        # Compute Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape and transpose for multi-head attention
        q, k, v = map(
            lambda t: t.view(B, L, H, D // H).transpose(1, 2),
            (q, k, v),
        )  # [B, H, L, Dh]

        # Normalize each head before applying RoPE.
        q, k = self.q_ln(q), self.k_ln(k)

        # Apply RoPE to q and k
        q, k = map(self.rotate, (q, k))

        # Compute attention
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, L, Dh]
        a = a.transpose(1, 2).reshape(B, L, D)  # [B, L, D]

        return self.proj(a)  # [B, L, D]


class SwiGLU(torch.nn.Module):
    """SwiGLU activation function as an nn.Module"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2


class TransformerBlock(torch.nn.Module):
    """A transformer block

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    expansion_ratio : float
        SwiGLU hidden width relative to d_model, rounded up to a multiple of 64.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: float = 8 / 3,
    ) -> None:
        super().__init__()
        hidden_dim = math.ceil(expansion_ratio * d_model / 64) * 64
        self.attn = CausalAttention(d_model, n_heads)
        self.ffn = torch.nn.Sequential(
            RMSNorm(d_model),
            torch.nn.Linear(d_model, hidden_dim * 2, bias=False),
            SwiGLU(),
            torch.nn.Linear(hidden_dim, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, L, D) -> (B, L, D)
        """
        r1 = self.attn(x)
        x = x + r1
        r2 = self.ffn(x)
        x = x + r2
        return x


class TransformerStack(torch.nn.Module):
    """
    A stack of causal pre-RMSNorm blocks with head-wise QK normalization.

    Parameters
    ----------
    hidden_dim: int
        The dimensionality of the input and output feature vectors.
    n_heads: int
        The number of attention heads.
    n_layers: int
        The number of transformer blocks in the stack.
    expansion_ratio: float
        SwiGLU hidden width relative to hidden_dim.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        expansion_ratio: float = 8 / 3,
    ) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(hidden_dim, n_heads, expansion_ratio)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the TransformerStack.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        out: torch.Tensor
            The output tensor of shape (batch_size, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x)
        out = self.norm(x)
        return out
