"""Construct cell and gene hypergraph operators."""

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def normalize_hypergraph(H: torch.Tensor) -> torch.Tensor:
    """
    Compute D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}.

    H has shape [number of nodes, number of hyperedges].
    """
    H = H.float()
    node_degree = torch.sum(H, dim=1)
    edge_degree = torch.sum(H, dim=0)

    node_inv_sqrt = torch.pow(node_degree + 1e-8, -0.5)
    edge_inv = torch.pow(edge_degree + 1e-8, -1.0)

    # Equivalent to explicit diagonal-matrix multiplication, but avoids
    # materializing two additional dense diagonal matrices.
    H_left = node_inv_sqrt[:, None] * H
    H_scaled = H_left * edge_inv[None, :]
    G = H_scaled @ H_left.t()
    return G


def build_knn_incidence(
    features: np.ndarray,
    n_neighbors: int,
    metric: str,
    node_name: str,
) -> torch.Tensor:
    """
    Build one centered kNN hyperedge per node.

    For node i, column i of H represents
        e_i = {i} union N_k(i).

    Because a node can be a neighbor of several centers, hyperedges overlap.
    """
    features = np.asarray(features, dtype=np.float32)
    n_nodes = features.shape[0]

    if n_nodes < 2:
        raise ValueError(
            f"At least two {node_name} nodes are required to build a kNN hypergraph."
        )
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    k = min(n_neighbors, n_nodes - 1)
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="brute" if metric == "cosine" else "auto",
        n_jobs=-1,
    )
    nbrs.fit(features)
    indices = nbrs.kneighbors(features, return_distance=False)

    H = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for center in range(n_nodes):
        # sklearn normally returns the query point itself. Explicitly adding the
        # center makes the definition robust to ties and numerical edge cases.
        members = np.unique(np.concatenate(([center], indices[center])))
        H[members, center] = 1.0

    edge_sizes = H.sum(axis=0)
    node_memberships = H.sum(axis=1)
    overlap_fraction = float(np.mean(node_memberships > 1))

    print(
        f"{node_name.capitalize()} kNN hypergraph | "
        f"nodes={n_nodes}, hyperedges={H.shape[1]}, requested_k={n_neighbors}, "
        f"effective_k={k}, mean_edge_size={edge_sizes.mean():.2f}, "
        f"mean_memberships={node_memberships.mean():.2f}, "
        f"overlap_fraction={overlap_fraction:.4f}"
    )

    return torch.from_numpy(H)


def build_cell_hypergraph(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "cosine",
) -> torch.Tensor:
    """Construct overlapping cell-centered kNN hyperedges."""
    return build_knn_incidence(
        features=X,
        n_neighbors=n_neighbors,
        metric=metric,
        node_name="cell",
    )


def reduce_gene_features(
    X: np.ndarray,
    max_svd_dim: int,
    random_state: int,
) -> np.ndarray:
    """
    Represent each gene by a reduced version of its expression profile across cells.

    Input X has shape [N, G]; the returned matrix has shape [G, d_svd].
    """
    n_cells, n_genes = X.shape
    gene_features = X.T.astype(np.float32)

    max_valid_dim = min(n_cells - 1, n_genes - 1)
    if max_valid_dim < 1:
        raise ValueError(
            "TruncatedSVD for gene features requires at least two cells and two genes."
        )

    svd_dim = min(max_svd_dim, max_valid_dim)
    print(
        "Reducing gene features with TruncatedSVD: "
        f"{gene_features.shape} -> ({n_genes}, {svd_dim})"
    )

    svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
    gene_features_low = svd.fit_transform(gene_features).astype(np.float32)
    return np.nan_to_num(gene_features_low, nan=0.0, posinf=0.0, neginf=0.0)


def build_gene_hypergraph(
    gene_features_low: np.ndarray,
    n_neighbors: int,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Construct overlapping gene-centered kNN hyperedges.

    Each gene g defines
        e_g^gene = {g} union N_k^gene(g),
    so H_gene has shape [G, G]. A gene may belong to multiple columns.
    """
    return build_knn_incidence(
        features=gene_features_low,
        n_neighbors=n_neighbors,
        metric=metric,
        node_name="gene",
    )


def prepare_gene_initialization(
    gene_features_low: np.ndarray,
    gene_dim: int,
) -> torch.Tensor:
    """Pad or truncate reduced gene features to the model's gene dimension."""
    n_genes, current_dim = gene_features_low.shape

    if current_dim < gene_dim:
        padding = np.zeros((n_genes, gene_dim - current_dim), dtype=np.float32)
        gene_init_np = np.concatenate([gene_features_low, padding], axis=1)
    else:
        gene_init_np = gene_features_low[:, :gene_dim]

    return torch.from_numpy(gene_init_np.astype(np.float32))
