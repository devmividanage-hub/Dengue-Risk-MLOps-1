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

