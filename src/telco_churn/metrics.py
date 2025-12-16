from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class MetricReport:
    roc_auc: float
    pr_auc: float
    brier: float


def classification_report_proba(y_true, y_proba) -> MetricReport:
    """
    Metrics for probabilistic classifiers.
    y_proba should be P(y=1).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    return MetricReport(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        brier=float(brier_score_loss(y_true, y_proba)),
    )


def pick_threshold_by_cost(
    y_true,
    y_proba,
    contact_cost: float = 1.0,
    save_benefit: float = 10.0,
) -> Dict[str, float]:
    """
    Simple business framing:
      - Contacting a customer costs `contact_cost`.
      - Successfully retaining a would-be churner yields `save_benefit`.
    Choose the probability threshold that maximizes expected utility
    over a grid of thresholds.

    Returns: {"threshold": ..., "utility": ...}
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_u = float("-inf")

    for t in thresholds:
        contacted = y_proba >= t
        tp = (contacted & (y_true == 1)).sum()
        n_contact = contacted.sum()
        utility = tp * save_benefit - n_contact * contact_cost

        if utility > best_u:
            best_u = float(utility)
            best_t = float(t)

    return {"threshold": best_t, "utility": best_u}
