"""Train and evaluate scDHF with stratified cross-validation."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ..evaluation.metrics import compute_test_metrics
from ..graphs.hypergraph import (
    build_cell_hypergraph,
    build_gene_hypergraph,
    normalize_hypergraph,
    prepare_gene_initialization,
    reduce_gene_features,
)
from ..models.scdhf import DualHyperRNA
from ..utils.reproducibility import set_seed


def train_model_5fold(
    X,
    y,
    hidden_dim=128,
    gene_dim=64,
    n_neighbors=20,
    gene_neighbors=20,
    gene_svd_dim=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-3,
    seed=42,
    patience=20,
    lambda_cons=0.1,
    dropout=0.2,
    n_splits=5,
    val_size=0.2,
):
    """Train and evaluate scDHF using stratified cross-validation."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_cells, n_genes = X.shape
    n_classes = len(np.unique(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    fold_predictions = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}/{n_splits}")
        print("=" * 80)

        fold_seed = seed + fold
        set_seed(fold_seed)

        train_idx, val_idx = train_test_split( train_val_idx, test_size=val_size, random_state=fold_seed , stratify=y[train_val_idx],)

        print(f"Train size: {len(train_idx)}")
        print(f"Val size:   {len(val_idx)}")
        print(f"Test size:  {len(test_idx)}")

        # Fit preprocessing statistics only on training cells, then transform all
        # cells in the fold. Hypergraph construction remains transductive.
        scaler = StandardScaler()
        scaler.fit(X[train_idx])
        X_scaled = scaler.transform(X).astype(np.float32)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        n_cells, n_genes = X_scaled.shape
        X_tensor = torch.from_numpy(X_scaled).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)

        print("Building overlapping cell kNN hypergraph...")
        H_cell = build_cell_hypergraph(X_scaled, n_neighbors=n_neighbors)
        G_cell = normalize_hypergraph(H_cell).to(device)

        # One SVD representation is shared by gene-kNN construction and gene
        # embedding initialization, ensuring that both use the same gene geometry.
        gene_features_low = reduce_gene_features(
            X_scaled,
            max_svd_dim=max(gene_svd_dim, gene_dim),
            random_state=fold_seed,
        )

        print("Building overlapping gene kNN hypergraph...")
        H_gene = build_gene_hypergraph(
            gene_features_low,
            n_neighbors=gene_neighbors,
        )
        G_gene = normalize_hypergraph(H_gene).to(device)

        print("Initializing gene embeddings from the reduced gene features...")
        gene_init = prepare_gene_initialization(
            gene_features_low=gene_features_low,
            gene_dim=gene_dim,
        ).to(device)

        model = DualHyperRNA(
            n_genes=n_genes,
            n_classes=n_classes,
            gene_init=gene_init,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

        best_val_f1 = -1.0
        best_val_acc = 0.0
        best_state = None
        counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            logits, Zc, _, Xg, _ = model(X_tensor, G_cell, G_gene)

            loss_cls = F.cross_entropy(
                logits[train_idx_t],
                y_tensor[train_idx_t],
            )

            loss_consistency = F.mse_loss(
                F.normalize(Zc, dim=1),
                F.normalize(Xg, dim=1),
            )

            loss = loss_cls + lambda_cons * loss_consistency

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                eval_logits, _, _, _, _ = model(X_tensor, G_cell, G_gene)
                pred = eval_logits.argmax(dim=1)

                train_acc = accuracy_score(
                    y_tensor[train_idx_t].cpu().numpy(),
                    pred[train_idx_t].cpu().numpy(),
                )
                val_acc = accuracy_score(
                    y_tensor[val_idx_t].cpu().numpy(),
                    pred[val_idx_t].cpu().numpy(),
                )
                val_f1 = f1_score(
                    y_tensor[val_idx_t].cpu().numpy(),
                    pred[val_idx_t].cpu().numpy(),
                    average="macro",
                    zero_division=0,
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                counter = 0
            else:
                counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Fold {fold} | Epoch {epoch:03d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Cls {loss_cls.item():.4f} | "
                    f"Cons {loss_consistency.item():.4f} | "
                    f"Train Acc {train_acc:.4f} | "
                    f"Val Acc {val_acc:.4f} | "
                    f"Val Macro-F1 {val_f1:.4f}"
                )

            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is None:
            raise RuntimeError("No valid model state was selected during training.")

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            logits, _, _, _, _ = model(X_tensor, G_cell, G_gene)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()

        y_test = y[test_idx]
        y_pred = predictions[test_idx]
        y_prob = probabilities[test_idx]

        # Save test-set predictions for this fold.
        prediction_df = pd.DataFrame({
            "sample_index": test_idx,
            "y_true": y_test,
            "y_pred": y_pred,
            "correct": (y_test == y_pred).astype(int),
        })
        for class_idx in range(n_classes):
            prediction_df[f"prob_class_{class_idx}"] = y_prob[:, class_idx]
        fold_predictions.append(prediction_df)

        test_metrics = compute_test_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            n_classes=n_classes,
        )

        print("\nFold result:")
        print("Best Validation Accuracy:", best_val_acc)
        print("Best Validation Macro-F1:", best_val_f1)
        print("\nTest metrics:")
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (float, np.floating)):
                print(f"{metric_name}: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value}")

        print("\nClassification report on test fold:")
        print(classification_report(y_test, y_pred, zero_division=0))

        fold_row = {
            "fold": fold,
            "val_acc": best_val_acc,
            "val_macro_f1": best_val_f1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "cell_neighbors": min(n_neighbors, n_cells - 1),
            "gene_neighbors": min(gene_neighbors, n_genes - 1),
        }
        fold_row.update(test_metrics)
        fold_results.append(fold_row)

        del (
            model,
            optimizer,
            X_tensor,
            y_tensor,
            G_cell,
            G_gene,
            H_cell,
            H_gene,
            gene_init,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(fold_results)

    print("\n" + "=" * 80)
    print("5-Fold Cross Validation Summary")
    print("=" * 80)
    print(results_df)

    test_metric_cols = [c for c in results_df.columns if c.startswith("test_")]
    print("\nMean ± Std:")
    print(results_df[test_metric_cols].agg(["mean", "std"]))

    return results_df, fold_predictions
