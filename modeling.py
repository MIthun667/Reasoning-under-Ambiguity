# modeling.py
# -*- coding: utf-8 -*-
"""
Model architectures for multilingual multi-label emotion classification.

Supports:
- Backbone encoders: XLM-R, mDeBERTa-v3
- Uncertainty modes:
    - baseline (independent sigmoid)
    - ambiguity_weight (same head, entropy used in loss)
    - evidential (Beta-style evidential head)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ----------------------------
# Classification Heads
# ----------------------------
class MultiLabelHead(nn.Module):
    """
    Multi-label prediction head with uncertainty options.

    Modes:
      - baseline / ambiguity_weight:
          logits -> sigmoid probabilities
      - evidential:
          alpha, beta -> predictive mean + uncertainty
    """
    def __init__(self, hidden_size: int, num_labels: int, uncertainty_mode: str):
        super().__init__()
        self.num_labels = num_labels
        self.uncertainty_mode = uncertainty_mode

        if uncertainty_mode in ["baseline", "ambiguity_weight"]:
            self.classifier = nn.Linear(hidden_size, num_labels)

        elif uncertainty_mode == "evidential":
            self.alpha_layer = nn.Linear(hidden_size, num_labels)
            self.beta_layer = nn.Linear(hidden_size, num_labels)

        else:
            raise ValueError(f"Unknown uncertainty_mode: {uncertainty_mode}")

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, H]

        Returns:
            Dict with keys depending on uncertainty_mode
        """
        if self.uncertainty_mode in ["baseline", "ambiguity_weight"]:
            logits = self.classifier(features)
            probs = torch.sigmoid(logits)
            return {
                "logits": logits,
                "probs": probs
            }

        # evidential
        alpha_evidence = F.softplus(self.alpha_layer(features))
        beta_evidence = F.softplus(self.beta_layer(features))

        alpha = alpha_evidence + 1.0
        beta = beta_evidence + 1.0

        probs = alpha / (alpha + beta)
        uncertainty = 1.0 / (alpha + beta)

        return {
            "alpha": alpha,
            "beta": beta,
            "probs": probs,
            "uncertainty": uncertainty
        }


# ----------------------------
# Full Emotion Model
# ----------------------------
class EmotionModel(nn.Module):
    """
    Transformer encoder + uncertainty-aware multi-label head.
    """

    def __init__(
        self,
        backbone_name: str,
        num_labels: int,
        uncertainty_mode: str,
        dropout: float = 0.1
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.uncertainty_mode = uncertainty_mode

        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.head = MultiLabelHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            uncertainty_mode=uncertainty_mode
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # CLS / first token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        return self.head(pooled)

    def forward_with_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass used for interpretability (Gradient × Input).
        """
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        return self.head(pooled)

    def get_input_embeddings(self):
        """
        Expose embedding layer for interpretability.
        """
        return self.encoder.get_input_embeddings()
