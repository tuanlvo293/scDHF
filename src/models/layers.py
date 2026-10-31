"""Hypergraph convolution layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.residual = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        propagated = G @ X
        out = self.linear(propagated)
        out = out + self.residual(X)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out
