import sys
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LinearAttention(nn.Module):
    """
    Linear Attention using kernel feature maps.
    Complexity: O(N * d^2) instead of O(N^2 * d)
    """

    def __init__(self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64, dropout=0., eps=1e-6):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        out_dim = default(out_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        Args:
            x: [batch, seq_len, dim]
            context: [batch, seq_len_ctx, dim] or None
            mask: [batch, seq_len] or None
        """
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape to [batch * heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Apply feature map (elu + 1 for non-negativity)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Handle mask if provided
        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k * mask
            v = v * mask

        # Compute normalization: Z = Q @ K^T @ 1
        # k_sum: [b, h, 1, d]
        k_sum = k.sum(dim=2, keepdim=True)
        # z: [b, h, n, 1]
        z = einsum('b h n d, b h m d -> b h n m', q, k_sum)
        z = z + self.eps

        # Compute attention: KV = K^T @ V
        # kv: [b, h, d, d]
        kv = einsum('b h n d, b h n e -> b h d e', k, v)

        # Compute output: out = Q @ KV
        # out: [b, h, n, d]
        out = einsum('b h n d, b h d e -> b h n e', q, kv)

        # Normalize
        out = out / z

        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CosineLinearAttention(nn.Module):
    """
    Linear Attention using cosine similarity kernel.
    More stable than ELU-based version.
    """

    def __init__(self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64, dropout=0., eps=1e-6):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        out_dim = default(out_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # L2 normalize for cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k * mask
            v = v * mask

        # KV: [b, h, d, d]
        kv = einsum('b h n d, b h n e -> b h d e', k, v)

        # out: [b, h, n, d]
        out = einsum('b h n d, b h d e -> b h n e', q, kv)

        # Scale
        out = out * self.scale

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class BasicLinearTransformerBlock(nn.Module):
    def __init__(self, dim, out_dim, n_heads=8, d_head=64, dropout=0., use_cosine=False):
        super().__init__()

        if use_cosine:
            self.attn = CosineLinearAttention(
                query_dim=dim,
                out_dim=out_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout
            )
        else:
            self.attn = LinearAttention(
                query_dim=dim,
                out_dim=out_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm(x))
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LinearAdaptor(nn.Module):
    """
    Adaptor using Linear Attention for efficient processing.
    """

    def __init__(self, inplanes=1024, outplanes=None, use_cosine=False):
        super(LinearAdaptor, self).__init__()
        outplanes = default(outplanes, inplanes)
        self.attention = BasicLinearTransformerBlock(
            dim=inplanes,
            out_dim=outplanes,
            use_cosine=use_cosine
        )

    def forward(self, img_token):
        return self.attention(img_token)

