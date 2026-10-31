"""Classification metrics for held-out test cells."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


def compute_test_metrics(y_true, y_pred, y_prob=None, n_classes=None):
    metrics = {
        "test_acc": accuracy_score(y_true, y_pred),
        "test_balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "test_precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "test_macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "test_precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "test_recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "test_weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "test_micro_f1": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "test_mcc": matthews_corrcoef(y_true, y_pred),
        "test_cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_prob is not None and n_classes is not None:
        labels = np.arange(n_classes)

        try:
            metrics["test_log_loss"] = log_loss(y_true, y_prob, labels=labels)
        except Exception:
            metrics["test_log_loss"] = np.nan

        # top-k accuracy is only informative when k < number of classes.
        if n_classes > 2:
            try:
                metrics["test_top2_acc"] = top_k_accuracy_score(
                    y_true, y_prob, k=2, labels=labels
                )
            except Exception:
                metrics["test_top2_acc"] = np.nan
        else:
            metrics["test_top2_acc"] = np.nan

        if n_classes > 3:
            try:
                metrics["test_top3_acc"] = top_k_accuracy_score(
                    y_true, y_prob, k=3, labels=labels
                )
            except Exception:
                metrics["test_top3_acc"] = np.nan
        else:
            metrics["test_top3_acc"] = np.nan

    return metrics
