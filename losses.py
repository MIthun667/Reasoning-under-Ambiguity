# losses.py
# -*- coding: utf-8 -*-
"""
Loss functions for multilingual multi-label emotion classification.

Includes:
- Masked BCE loss (partial-label supervision)
- Ambiguity-weighted loss (entropy-based down-weighting)
- Evidential loss (Beta evidential learning)
- PU-style weak negative constraint for missing labels (addresses masking weakness)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ----------------------------
# Core masked BCE
# ----------------------------
def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Binary cross-entropy over observed labels only.

    Args:
        logits: [B, K]
        targets: [B, K]
        mask: [B, K] (1 = observed, 0 = missing)
        pos_weight: optional [K]

    Returns:
        scalar loss
    """
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
    else:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

    masked = bce * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


# ----------------------------
# Ambiguity weighting (entropy-based)
# ----------------------------
def ambiguity_weight_from_probs(
    probs: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 2.0,
) -> torch.Tensor:
    """
    Compute per-sample ambiguity weight using entropy.

    High-entropy (ambiguous) samples receive lower weight.

    Args:
        probs: sigmoid probabilities [B, K]
        mask: label mask [B, K]
        tau: temperature (>0)

    Returns:
        weights: [B]
    """
    eps = 1e-8
    entropy = -(
        probs * torch.log(probs + eps) +
        (1.0 - probs) * torch.log(1.0 - probs + eps)
    )

    entropy = entropy * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    entropy_mean = entropy.sum(dim=1) / denom

    weights = torch.exp(-tau * entropy_mean)
    return weights.clamp(0.05, 1.0)


def ambiguity_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    tau: float,
) -> torch.Tensor:
    """
    Ambiguity-weighted masked BCE loss.
    """
    probs = torch.sigmoid(logits)
    weights = ambiguity_weight_from_probs(probs.detach(), mask, tau=tau)

    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
    else:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

    bce = bce * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_sample = bce.sum(dim=1) / denom

    return (weights * per_sample).mean()


# ----------------------------
# Evidential loss (Beta)
# ----------------------------
def evidential_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    lam_reg: float = 0.1,
) -> torch.Tensor:
    """
    Evidential loss for Bernoulli outputs.

    Encourages:
    - accurate prediction where labels exist
    - low evidence for incorrect predictions
    """
    eps = 1e-8
    probs = alpha / (alpha + beta + eps)

    # Prediction loss
    bce = F.binary_cross_entropy(probs, targets, reduction="none")
    pred_loss = (bce * mask).sum() / mask.sum().clamp_min(1.0)

    # Regularization: discourage confident wrong evidence
    reg = targets * torch.log1p(beta) + (1.0 - targets) * torch.log1p(alpha)
    reg_loss = (reg * mask).sum() / mask.sum().clamp_min(1.0)

    return pred_loss + lam_reg * reg_loss

def build_pos_weight(examples, device):
    """
    Compute per-label pos_weight for BCEWithLogits,
    using ONLY observed labels (mask == 1).

    pos_weight_k = (#negative_k) / (#positive_k)

    Args:
        examples: list of dicts with keys ["y", "mask"]
        device: torch device

    Returns:
        torch.Tensor of shape [K]
    """
    # stack labels and masks
    y = torch.tensor(
        [ex["y"] for ex in examples],
        dtype=torch.float32
    )
    m = torch.tensor(
        [ex["mask"] for ex in examples],
        dtype=torch.float32
    )

    # count positives and negatives only where mask==1
    pos = (y * m).sum(dim=0)
    neg = ((1.0 - y) * m).sum(dim=0)

    # avoid division by zero
    pos = pos.clamp_min(1.0)

    pos_weight = neg / pos
    return pos_weight.to(device)

# ----------------------------
# PU-style weak negative constraint (IMPORTANT FIX)
# ----------------------------
def pu_missing_negative_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.05,
) -> torch.Tensor:
    """
    Weakly encourage missing labels to be negative (PU learning).

    Args:
        logits: [B, K]
        mask: [B, K] (1 = observed, 0 = missing)
        weight: small coefficient (e.g., 0.05)

    Returns:
        scalar loss
    """
    if weight <= 0.0:
        return logits.new_tensor(0.0)

    missing_mask = (1.0 - mask)
    if missing_mask.sum() == 0:
        return logits.new_tensor(0.0)

    # treat missing labels as weak negatives
    weak_targets = torch.zeros_like(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits, weak_targets, reduction="none"
    )

    loss = (bce * missing_mask).sum() / missing_mask.sum().clamp_min(1.0)
    return weight * loss


# ----------------------------
# Combined loss wrapper
# ----------------------------
def compute_total_loss(
    outputs: dict,
    targets: torch.Tensor,
    mask: torch.Tensor,
    uncertainty_mode: str,
    pos_weight: Optional[torch.Tensor] = None,
    tau: float = 2.0,
    lam_reg: float = 0.1,
    pu_weight: float = 0.0,
) -> torch.Tensor:
    """
    Unified loss interface used by train.py.
    """
    if uncertainty_mode in ["baseline", "ambiguity_weight"]:
        logits = outputs["logits"]

        if uncertainty_mode == "baseline":
            loss = masked_bce_with_logits(
                logits, targets, mask, pos_weight
            )
        else:
            loss = ambiguity_weighted_loss(
                logits, targets, mask, pos_weight, tau
            )

        # Optional PU constraint
        loss = loss + pu_missing_negative_loss(
            logits, mask, weight=pu_weight
        )
        return loss

    # evidential
    loss = evidential_loss(
        outputs["alpha"],
        outputs["beta"],
        targets,
        mask,
        lam_reg=lam_reg
    )
    return loss
