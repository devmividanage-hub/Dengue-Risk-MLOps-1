"""Calculated classification metrics and reproducible evaluation artifacts."""

import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.helpers import RISK_CLASSES, ROOT, resolve_path,save_json

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
matplotlib.use("Agg")
from matplotlib import pyplot as plt # noqa: E402

def calculate_metrics(
    true_labels: Any,
    predicted_labels: Any,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute metrics with an explicit class order, including HIGH-risk recall."""
    truth = np.asarray(true_labels, dtype=str)
    prediction = np.asarray(predicted_labels, dtype=str)
    if len(truth) == 0 or len(truth) != len(prediction):
        raise ValueError("Nonempty aligned true and predicted labels are required")
    if not set(truth).union(prediction).issubset(RISK_CLASSES):
        raise ValueError("Unknown risk class in evaluation labels")
    report = classification_report(
        truth, prediction, labels=list(RISK_CLASSES), output_dict=True, zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_macro": float(
            precision_score(
                truth, prediction, labels=list(RISK_CLASSES), average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                truth, prediction, labels=list(RISK_CLASSES), average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(truth, prediction, labels=list(RISK_CLASSES), average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "high_risk_recall": float(report["HIGH"]["recall"]),
        "per_class": {label: report[label] for label in RISK_CLASSES},
        "confusion_matrix": confusion_matrix(truth, prediction, labels=list(RISK_CLASSES)).tolist(),
        "class_order": list(RISK_CLASSES),
        "roc_auc_ovr_macro": None,
        "classification_report": report,
    }
    if probabilities is not None and len(set(truth)) == 3:
        encoded = np.asarray([RISK_CLASSES.index(label) for label in truth])
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                encoded, probabilities, multi_class="ovr", average="macro", labels=[0, 1, 2]
            )
        )
    elif probabilities is not None:
        metrics["roc_auc_reason"] = "Undefined: this evaluation period lacks one or more classes"
    return metrics


def save_evaluation(metrics: dict[str, Any], directory: str | Path, prefix: str) -> list[Path]:
    """Save JSON, readable report, and confusion matrix underreports/."""
    root = resolve_path(directory)
    metrics_path = root / "metrics" / f"{prefix}.json"
    report_path = root / "metrics" / f"{prefix}_classification_report.json"
    figure_path = root / "figures" / f"{prefix}_confusion_matrix.png"
    save_json(metrics_path, metrics)
    save_json(report_path, metrics["classification_report"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        np.asarray(metrics["confusion_matrix"]), display_labels=list(RISK_CLASSES)
    )
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title(prefix.replace("_", " "))
    display.figure_.tight_layout()
    display.figure_.savefig(figure_path, dpi=140)
    plt.close(display.figure_)
    return [metrics_path, report_path, figure_path]