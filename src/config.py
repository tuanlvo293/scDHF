"""Command-line options and repository paths for scDHF."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "scDHF: dual overlapping cell-kNN and gene-kNN hypergraphs "
            "for transductive scRNA-seq cell-type classification."
        )
    )
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, with or without the .csv extension.")
    parser.add_argument("--results_dir", type=str, default=str(REPO_ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_neighbors", type=int, default=20,
                        help="Number of nearest neighbors in each cell-centered hyperedge.")
    parser.add_argument("--gene_neighbors",type=int, default=20,
                        help="Number of nearest neighbors in each gene-centered hyperedge.")
    parser.add_argument("--gene_svd_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gene_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_hvg", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.2)

    return parser
