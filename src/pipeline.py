"""Run one dataset and save its cross-validation outputs."""

import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.preprocessing import load_scrna_csv
from .training.cross_validation import train_model_5fold


def run(args):
    data_dir = Path(os.path.expanduser(args.data_dir))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("scDHF | Overlapping Cell-kNN + Gene-kNN Hypergraphs | 5-Fold CV")
    print("\nConfig:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("*" * 80)

    dataset_name = args.dataset
    if dataset_name.lower().endswith(".csv"):
        dataset_name = dataset_name[:-4]

    data_path = data_dir / f"{dataset_name}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset_paths = [data_path]

    all_summary = []

    for data_path in dataset_paths:
        dataset_name = data_path.name
        dataset_stem = data_path.stem

        print("\n" + "#" * 100)
        print(f"DATASET: {dataset_name}")
        print("#" * 100)

        try:
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset does not exist: {data_path}")

            X, y, _, _ = load_scrna_csv(str(data_path), n_hvg=args.n_hvg)
            print("X shape:", X.shape)
            print("Number of classes:", len(np.unique(y)))

            results_df, fold_predictions = train_model_5fold(
                X=X,
                y=y,
                hidden_dim=args.hidden_dim,
                gene_dim=args.gene_dim,
                n_neighbors=args.n_neighbors,
                gene_neighbors=args.gene_neighbors,
                gene_svd_dim=args.gene_svd_dim,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
                patience=args.patience,
                lambda_cons=args.lambda_cons,
                dropout=args.dropout,
                n_splits=args.n_splits,
                val_size=args.val_size,
            )

            results_df.insert(0, "dataset", dataset_name)
            out_path = results_dir / f"{dataset_stem}_scDHF_5fold_results.csv"
            results_df.to_csv(out_path, index=False)

            prediction_path = (
                results_dir / f"{dataset_stem}_scDHF_5fold_predictions.xlsx"
            )
            with pd.ExcelWriter(prediction_path, engine="openpyxl") as writer:
                for fold_idx, prediction_df in enumerate(
                    fold_predictions, start=1
                ):
                    prediction_df.to_excel(
                        writer,
                        sheet_name=f"Fold_{fold_idx}",
                        index=False,
                    )

            print(f"Saved fold predictions to: {prediction_path}")

            test_metric_cols = [
                c for c in results_df.columns if c.startswith("test_")
            ]
            summary_row = {
                "dataset": dataset_name,
                "n_cells": X.shape[0],
                "n_genes": X.shape[1],
                "n_classes": len(np.unique(y)),
                "cell_neighbors": args.n_neighbors,
                "gene_neighbors": args.gene_neighbors,
                "status": "done",
            }

            for col in test_metric_cols:
                summary_row[f"{col}_mean"] = results_df[col].mean()
                summary_row[f"{col}_std"] = results_df[col].std()

            all_summary.append(summary_row)

            print(f"\nSaved per-fold results to: {out_path}")
            print(
                "Summary | "
                f"Acc: {results_df['test_acc'].mean():.4f} "
                f"± {results_df['test_acc'].std():.4f} | "
                f"Macro-F1: {results_df['test_macro_f1'].mean():.4f} "
                f"± {results_df['test_macro_f1'].std():.4f} | "
                f"Balanced Acc: {results_df['test_balanced_acc'].mean():.4f} "
                f"± {results_df['test_balanced_acc'].std():.4f}"
            )

        except Exception as error:
            print(f"FAILED: {dataset_name} | {error}")
            traceback.print_exc()
            all_summary.append(
                {
                    "dataset": dataset_name,
                    "n_cells": None,
                    "n_genes": None,
                    "n_classes": None,
                    "cell_neighbors": args.n_neighbors,
                    "gene_neighbors": args.gene_neighbors,
                    "status": f"failed: {error}",
                }
            )

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_summary)
    summary_path = results_dir / f"{dataset_stem}_scDHF_5fold_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("DATASET SUMMARY")
    print(summary_df)
    print(f"\nSaved summary to: {summary_path}")
