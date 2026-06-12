"""
Graph Attention Network (GAT) layer.
Each sensor node attends to its neighbours and weights their features
by learned attention coefficients — so an upwind node gets more attention
than a crosswind neighbour when wind data signals fire approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single-head Graph Attention Layer (Veličković et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        # Split attention: a_src scores source nodes, a_dst scores destination nodes
        # e_ij = leakyrelu(a_src_i + a_dst_j) — avoids (B,N,N,2*out) tensor
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.weight)
        nn.init.xavier_uniform_(self.a_dst.weight)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h:   (B, N, in_features) or (N, in_features)
        adj: (N, N)
        Returns: same leading dims as h, with last dim = out_features
        """
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(0)

        B, N, _ = h.shape
        Wh = self.W(h)                            # (B, N, out)

        # (B, N, 1) + (B, 1, N) → (B, N, N) — no large intermediate tensor
        e_src = self.a_src(Wh)                    # (B, N, 1)
        e_dst = self.a_dst(Wh)                    # (B, N, 1)
        e = self.leaky_relu(e_src + e_dst.transpose(1, 2))  # (B, N, N)

        mask = (adj == 0).float() * -1e9          # (N, N) broadcasts to (B, N, N)
        e = e + mask
        attention = F.softmax(e, dim=-1)          # (B, N, N)
        attention = self.dropout(attention)

        out = F.elu(torch.bmm(attention, Wh))     # (B, N, out)
        return out.squeeze(0) if squeeze else out


class MultiHeadGAT(nn.Module):
    """Multi-head GAT: concatenates K attention heads."""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            GATLayer(in_features, out_features // n_heads, dropout=dropout)
            for _ in range(n_heads)
        ])
        self.out_features = out_features

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(h, adj) for head in self.heads]
        return torch.cat(head_outputs, dim=-1)  # (B, N, out) or (N, out)
