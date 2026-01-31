# metrics.py
# -*- coding: utf-8 -*-
"""
Evaluation metrics for multilingual multi-label emotion classification.

Implements (masked / partial-label aware):
- Hamming Loss (HL ↓)
- Ranking Loss (RL ↓)
- Micro-F1, Macro-F1 (↑)
- Average Precision (AP ↑):
    * mAP (mean per-label AP)
    * microAP (flattened masked positions)
- LRAP, Coverage Error
- Jaccard (samples)
- Shared-label subset evaluation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Iterable

import numpy as np
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    average_precision_score
)


# ----------------------------
# Masked helpers
# ----------------------------
def masked_hamming_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray
) -> float:
    m = mask.astype(bool)
    denom = m.sum()
    if denom == 0:
        return 0.0
    return float((y_true[m] != y_pred[m]).sum() / denom)


def masked_label_ranking_loss(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Ranking loss computed per sample over observed labels only.
    """
    N, K = y_true.shape
    losses = []

    for i in range(N):
        idx = np.where(mask[i] > 0.5)[0]
        if idx.size == 0:
            continue

        yt = y_true[i, idx].astype(int)
        sc = y_score[i, idx].astype(float)

        pos = np.where(yt == 1)[0]
        neg = np.where(yt == 0)[0]

        if pos.size == 0 or neg.size == 0:
            losses.append(0.0)
            continue

        wrong = 0
        total = pos.size * neg.size
        for p in pos:
            for n in neg:
                if sc[p] <= sc[n]:
                    wrong += 1
        losses.append(wrong / total)

    return float(np.mean(losses)) if losses else 0.0


def masked_lrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Label Ranking Average Precision (masked).
    """
    N, K = y_true.shape
    scores = []

    for i in range(N):
        idx = np.where(mask[i] > 0.5)[0]
        if idx.size == 0:
            continue

        yt = y_true[i, idx].astype(int)
        sc = y_score[i, idx].astype(float)

        if yt.sum() == 0:
            scores.append(0.0)
            continue

        order = np.argsort(-sc)
        yt_sorted = yt[order]
        cumsum_pos = np.cumsum(yt_sorted)
        pos_pos = np.where(yt_sorted == 1)[0]

        precisions = cumsum_pos[pos_pos] / (pos_pos + 1)
        scores.append(float(np.mean(precisions)))

    return float(np.mean(scores)) if scores else 0.0


def masked_coverage_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Coverage error over observed labels.
    """
    N, K = y_true.shape
    cov = []

    for i in range(N):
        idx = np.where(mask[i] > 0.5)[0]
        if idx.size == 0:
            continue

        yt = y_true[i, idx].astype(int)
        sc = y_score[i, idx].astype(float)

        pos = np.where(yt == 1)[0]
        if pos.size == 0:
            cov.append(0.0)
            continue

        order = np.argsort(-sc)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.size)

        cov.append(float(ranks[pos].max() + 1))

    return float(np.mean(cov)) if cov else 0.0


# ----------------------------
# Threshold tuning
# ----------------------------
def tune_thresholds_per_label(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    grid: int = 101
) -> np.ndarray:
    """
    Tune per-label thresholds to maximize F1.
    """
    K = y_true.shape[1]
    thresholds = np.full(K, 0.5, dtype=np.float32)
    cand = np.linspace(0.0, 1.0, grid)

    for k in range(K):
        m_k = mask[:, k].astype(bool)
        if m_k.sum() == 0:
            continue

        yt = y_true[m_k, k].astype(int)
        pr = y_prob[m_k, k].astype(float)

        best_f1, best_t = -1.0, 0.5
        for t in cand:
            yp = (pr >= t).astype(int)
            f1 = f1_score(yt, yp, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        thresholds[k] = float(best_t)

    return thresholds


# ----------------------------
# Main metric computation
# ----------------------------
def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    subset: Optional[Iterable[int]] = None,
) -> Dict:
    """
    Compute all metrics required for reporting.

    Args:
        y_true: [N,K]
        y_prob: [N,K]
        mask:   [N,K]
        thresholds: [K] or None (defaults to 0.5)
        subset: iterable of label indices to evaluate (optional)

    Returns:
        dict with metrics
    """
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)

    if subset is not None:
        subset = list(subset)
        y_true = y_true[:, subset]
        y_prob = y_prob[:, subset]
        mask = mask[:, subset]
        thresholds = thresholds[subset]
        if label_names is not None:
            label_names = [label_names[i] for i in subset]

    y_pred = (y_prob >= thresholds[None, :]).astype(int)

    # Flattened masked positions
    mbool = mask.astype(bool)
    yt_flat = y_true[mbool].astype(int)
    yp_flat = y_pred[mbool].astype(int)
    pr_flat = y_prob[mbool].astype(float)

    # Core metrics
    micro_f1 = float(f1_score(yt_flat, yp_flat, average="micro", zero_division=0))
    per_label_f1 = {}
    per_label_ap = {}

    for k in range(y_true.shape[1]):
        m_k = mask[:, k].astype(bool)
        if m_k.sum() == 0:
            per_label_f1[k] = None
            per_label_ap[k] = None
            continue

        yt = y_true[m_k, k].astype(int)
        yp = y_pred[m_k, k].astype(int)
        pr = y_prob[m_k, k].astype(float)

        per_label_f1[k] = float(f1_score(yt, yp, zero_division=0))
        try:
            per_label_ap[k] = float(average_precision_score(yt, pr))
        except Exception:
            per_label_ap[k] = None

    macro_f1 = float(np.mean([v for v in per_label_f1.values() if v is not None]))

    # Average Precision
    ap_vals = [v for v in per_label_ap.values() if v is not None]
    mAP = float(np.mean(ap_vals)) if ap_vals else 0.0

    try:
        microAP = float(average_precision_score(yt_flat, pr_flat))
    except Exception:
        microAP = None

    metrics = {
        "HL": masked_hamming_loss(y_true, y_pred, mask),
        "RL": masked_label_ranking_loss(y_true, y_prob, mask),
        "miF1": micro_f1,
        "maF1": macro_f1,
        "AP": mAP,
        "microAP": microAP,
        "LRAP": masked_lrap(y_true, y_prob, mask),
        "coverage": masked_coverage_error(y_true, y_prob, mask),
        "jaccard_samples": float(
            jaccard_score(y_true.astype(int), y_pred.astype(int),
                          average="samples", zero_division=0)
        ),
        "per_label_f1": per_label_f1,
        "per_label_ap": per_label_ap,
    }

    # Optional label-name remap
    if label_names is not None:
        metrics["per_label_f1"] = {
            label_names[k]: v for k, v in per_label_f1.items()
        }
        metrics["per_label_ap"] = {
            label_names[k]: v for k, v in per_label_ap.items()
        }

    return metrics
