"""Load and preprocess labeled gene-expression CSV datasets."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_scrna_csv(path: str, n_hvg: int = 2048):
    """Load a CSV whose first column is the cell-type label."""
    df = pd.read_csv(path)
    y_raw = df.iloc[:, 0].values
    X_df = df.iloc[:, 1:].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    gene_names = np.asarray(X_df.columns)

    # Remove genes that are zero in every cell.
    nonzero_mask = np.sum(X != 0, axis=0) > 0
    X = X[:, nonzero_mask]
    gene_names = gene_names[nonzero_mask]
    print(f"After removing all-zero genes: {X.shape}")

    # Apply log1p to count-like, nonnegative data.
    if X.size > 0 and np.nanmin(X) >= 0 and np.nanmax(X) > 20:
        X = np.log1p(X)

    # Select highly variable genes using dispersion.
    if X.shape[1] > n_hvg:
        means = np.mean(X, axis=0)
        variances = np.var(X, axis=0)
        dispersion = variances / (means + 1e-8)
        dispersion = np.nan_to_num(
            dispersion, nan=0.0, posinf=0.0, neginf=0.0
        )
        top_idx = np.argsort(dispersion)[-n_hvg:]
        X = X[:, top_idx]
        gene_names = gene_names[top_idx]
        print(f"After HVG selection: {X.shape}")
    else:
        print(f"Skip HVG selection because genes <= {n_hvg}: {X.shape}")

    if X.shape[1] == 0:
        raise ValueError("No expressed genes remain after preprocessing.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    return X.astype(np.float32), y, label_encoder, gene_names.tolist()
