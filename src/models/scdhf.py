"""Dual cell and gene hypergraph model for scDHF."""

import torch
import torch.nn as nn

from .layers import HypergraphConv


class DualHyperRNA(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        gene_init: torch.Tensor,
        hidden_dim: int = 128,
        gene_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Cell branch: nodes are cells and input features are genes.
        self.cell_hgc1 = HypergraphConv(n_genes, hidden_dim, dropout)
        self.cell_hgc2 = HypergraphConv(hidden_dim, hidden_dim, dropout)

        # Gene branch: nodes are genes and features are SVD gene embeddings.
        self.gene_embedding = nn.Parameter(gene_init.float())
        self.gene_hgc1 = HypergraphConv(gene_dim, gene_dim, dropout)
        self.gene_hgc2 = HypergraphConv(gene_dim, gene_dim, dropout)

        # Project the gene-neighborhood-informed cell representation.
        self.gene_to_cell_proj = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        X: torch.Tensor,
        G_cell: torch.Tensor,
        G_gene: torch.Tensor,
    ):
        # Cell-neighborhood branch.
        Zc = self.cell_hgc1(X, G_cell)
        Zc = self.cell_hgc2(Zc, G_cell)

        # Gene-neighborhood branch.
        Zg = self.gene_hgc1(self.gene_embedding, G_gene)
        Zg = self.gene_hgc2(Zg, G_gene)

        # Expression-weighted projection from gene space back to cell space.
        Xg = X @ Zg
        Xg = self.gene_to_cell_proj(Xg)

        # Fuse the two cell representations.
        Z = torch.cat([Zc, Xg], dim=1)
        Z = self.fusion(Z)
        logits = self.classifier(Z)

        return logits, Zc, Zg, Xg, Z
