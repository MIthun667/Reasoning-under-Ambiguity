#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multilingual_emotion_uncertainty_train.py

Single-file runnable training script for multilingual multi-label emotion classification with:
- English .txt (tab-separated, header: ID, Tweet, anger...trust)
- Chinese .csv (header: text,Joy,Hate,Love,Sorrow,Anxiety,Surprise,Anger,Expect)
- Unified label mapping + per-sample label mask (partial-label learning)
- Encoder choice: xlm-roberta-base OR mdeberta-v3-base (via --backbone)
- Uncertainty-aware learning:
    --uncertainty_mode evidential | ambiguity_weight
- Training language control:
    --train_lang en | zh | both
- Threshold tuning:
    --threshold_mode global | per_lang
- Full metrics dump including:
    Micro/Macro F1 (masked), Jaccard(samples), Hamming loss(masked),
    Ranking loss(masked), LRAP(masked), Coverage error(masked),
    per-label F1 + AP
- Interpretability:
    Gradient × Input token attribution per label (saved to JSONL)

Install:
  pip install torch transformers pandas numpy scikit-learn tqdm

Examples:
  # English-only
  python multilingual_emotion_uncertainty_train.py \
    --english_train "/media/mithun/New Volume/Research/Multilabel/English/train.txt" \
    --english_val   "/media/mithun/New Volume/Research/Multilabel/English/validation.txt" \
    --english_test  "/media/mithun/New Volume/Research/Multilabel/English/test.txt" \
    --chinese_train "/media/mithun/New Volume/Research/Multilabel/chinese/train.csv" \
    --chinese_val   "/media/mithun/New Volume/Research/Multilabel/chinese/validation.csv" \
    --chinese_test  "/media/mithun/New Volume/Research/Multilabel/chinese/test.csv" \
    --train_lang en \
    --backbone xlm-roberta-base \
    --uncertainty_mode ambiguity_weight \
    --use_pos_weight \
    --epochs 6 --batch_size 16 --lr 2e-5 \
    --eval_by_lang \
    --out_dir runs/xlmr_en

  # Mixed training, per-language thresholds (recommended)
  python multilingual_emotion_uncertainty_train.py \
    --english_train "/media/mithun/New Volume/Research/Multilabel/English/train.txt" \
    --english_val   "/media/mithun/New Volume/Research/Multilabel/English/validation.txt" \
    --english_test  "/media/mithun/New Volume/Research/Multilabel/English/test.txt" \
    --chinese_train "/media/mithun/New Volume/Research/Multilabel/chinese/train.csv" \
    --chinese_val   "/media/mithun/New Volume/Research/Multilabel/chinese/validation.csv" \
    --chinese_test  "/media/mithun/New Volume/Research/Multilabel/chinese/test.csv" \
    --train_lang both \
    --threshold_mode per_lang \
    --backbone xlm-roberta-base \
    --uncertainty_mode ambiguity_weight \
    --use_pos_weight \
    --epochs 6 --batch_size 16 --lr 2e-5 \
    --eval_by_lang \
    --explain_split test --explain_n 30 --explain_topk 12 \
    --out_dir runs/xlmr_both_perlang

Notes:
- We prepend a language token: "<EN> " or "<ZH> " to text.
- Masked metrics ignore undefined labels for each sample.
- Ranking metrics are computed per sample only over defined labels.
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, jaccard_score, average_precision_score


# ----------------------------
# Labels (Unified space)
# ----------------------------
UNIFIED_LABELS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust",
    "hate",  # kept separate; safer than forcing into disgust
]
LABEL2IDX = {l: i for i, l in enumerate(UNIFIED_LABELS)}

# English labels expected (SemEval-like 11, no "hate")
EN_LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "love",
             "optimism", "pessimism", "sadness", "surprise", "trust"]

# Chinese columns expected
ZH_COLS = ["Joy", "Hate", "Love", "Sorrow", "Anxiety", "Surprise", "Anger", "Expect"]

# Mapping Chinese -> unified
ZH_TO_UNIFIED = {
    "Joy": "joy",
    "Love": "love",
    "Anger": "anger",
    "Surprise": "surprise",
    "Expect": "anticipation",
    "Sorrow": "sadness",
    "Anxiety": "fear",
    "Hate": "hate",
}


# ----------------------------
# Utilities
# ----------------------------
def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())
    return text

def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ----------------------------
# Data loading
# ----------------------------
def load_english_tsv(path: str) -> pd.DataFrame:
    """
    Expects a tab-separated file with header:
      ID  Tweet  anger anticipation ... trust
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    for c in EN_LABELS:
        if c not in df.columns:
            raise ValueError(f"[English] Missing column: {c} in {path}")
        df[c] = df[c].astype(int)

    if "Tweet" not in df.columns:
        if "Text" in df.columns:
            df = df.rename(columns={"Text": "Tweet"})
        else:
            raise ValueError(f"[English] Missing text column 'Tweet' in {path}")
    if "ID" not in df.columns:
        df["ID"] = [f"en_{i}" for i in range(len(df))]
    return df

def load_chinese_csv(path: str) -> pd.DataFrame:
    """
    Expects CSV with header:
      text,Joy,Hate,Love,Sorrow,Anxiety,Surprise,Anger,Expect
    """
    df = pd.read_csv(path, dtype=str)
    if "text" not in df.columns:
        if "Text" in df.columns:
            df = df.rename(columns={"Text": "text"})
        else:
            raise ValueError(f"[Chinese] Missing text column 'text' in {path}")
    for c in ZH_COLS:
        if c not in df.columns:
            raise ValueError(f"[Chinese] Missing column: {c} in {path}")
        df[c] = df[c].astype(int)
    if "id" not in df.columns and "ID" not in df.columns:
        df["id"] = [f"zh_{i}" for i in range(len(df))]
    return df

def df_to_examples_english(df: pd.DataFrame) -> List[Dict]:
    exs = []
    for _, row in df.iterrows():
        raw = normalize_text(row["Tweet"])
        text = "<EN> " + raw
        y = np.zeros(len(UNIFIED_LABELS), dtype=np.float32)
        m = np.zeros(len(UNIFIED_LABELS), dtype=np.float32)

        for lab in EN_LABELS:
            idx = LABEL2IDX[lab]
            y[idx] = float(row[lab])
            m[idx] = 1.0
        # English has no 'hate' in this schema => mask=0 for 'hate'
        exs.append({
            "id": str(row["ID"]),
            "text": text,
            "raw_text": raw,
            "lang": "en",
            "y": y,
            "mask": m,
        })
    return exs

def df_to_examples_chinese(df: pd.DataFrame) -> List[Dict]:
    exs = []
    id_col = "id" if "id" in df.columns else "ID"
    for _, row in df.iterrows():
        raw = normalize_text(row["text"])
        text = "<ZH> " + raw
        y = np.zeros(len(UNIFIED_LABELS), dtype=np.float32)
        m = np.zeros(len(UNIFIED_LABELS), dtype=np.float32)

        for zh_col, uni_lab in ZH_TO_UNIFIED.items():
            idx = LABEL2IDX[uni_lab]
            y[idx] = float(row[zh_col])
            m[idx] = 1.0

        exs.append({
            "id": str(row[id_col]),
            "text": text,
            "raw_text": raw,
            "lang": "zh",
            "y": y,
            "mask": m,
        })
    return exs

def filter_by_lang(examples: List[Dict], lang: str) -> List[Dict]:
    if lang == "both":
        return examples
    return [e for e in examples if e["lang"] == lang]


# ----------------------------
# Dataset / Collator
# ----------------------------
class EmotionDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict:
        return self.examples[i]


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    ids: List[str]
    texts: List[str]
    raw_texts: List[str]
    langs: List[str]

class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Batch:
        texts = [b["text"] for b in batch]
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        y = torch.tensor(np.stack([b["y"] for b in batch]), dtype=torch.float32)
        m = torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32)
        ids = [b["id"] for b in batch]
        raw_texts = [b["raw_text"] for b in batch]
        langs = [b["lang"] for b in batch]
        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            y=y,
            mask=m,
            ids=ids,
            texts=texts,
            raw_texts=raw_texts,
            langs=langs
        )


# ----------------------------
# Model
# ----------------------------
class MultiLabelHead(nn.Module):
    """
    uncertainty_mode:
      - "ambiguity_weight": logits -> sigmoid(p)
      - "evidential": evidences -> alpha,beta -> p + uncertainty u
    """
    def __init__(self, hidden_size: int, num_labels: int, uncertainty_mode: str):
        super().__init__()
        self.num_labels = num_labels
        self.uncertainty_mode = uncertainty_mode

        if uncertainty_mode == "ambiguity_weight":
            self.classifier = nn.Linear(hidden_size, num_labels)
        elif uncertainty_mode == "evidential":
            self.e_alpha = nn.Linear(hidden_size, num_labels)
            self.e_beta  = nn.Linear(hidden_size, num_labels)
        else:
            raise ValueError(f"Unknown uncertainty_mode: {uncertainty_mode}")

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.uncertainty_mode == "ambiguity_weight":
            logits = self.classifier(h)
            p = torch.sigmoid(logits)
            return {"logits": logits, "p": p}
        else:
            e_a = F.softplus(self.e_alpha(h))
            e_b = F.softplus(self.e_beta(h))
            alpha = e_a + 1.0
            beta  = e_b + 1.0
            p = alpha / (alpha + beta)
            u = 1.0 / (alpha + beta)
            return {"alpha": alpha, "beta": beta, "p": p, "u": u}


class EmotionModel(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, uncertainty_mode: str, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = MultiLabelHead(hidden, num_labels, uncertainty_mode)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]  # CLS/first token
        h = self.dropout(h)
        return self.head(h)

    def forward_with_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]
        h = self.dropout(h)
        return self.head(h)


# ----------------------------
# Losses
# ----------------------------
def ambiguity_weight_from_probs(p: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    eps = 1e-8
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    ent = ent * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    ent_mean = ent.sum(dim=1) / denom
    w = torch.exp(-tau * ent_mean)
    return w.clamp(0.05, 1.0)

def evidential_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lam_reg: float,
) -> torch.Tensor:
    eps = 1e-8
    p = alpha / (alpha + beta + eps)

    bce = F.binary_cross_entropy(p, y, reduction="none")
    L_pred = (bce * mask).sum() / mask.sum().clamp_min(1.0)

    # discourage high evidence on wrong side
    L_reg_mat = y * torch.log1p(beta) + (1.0 - y) * torch.log1p(alpha)
    L_reg = (L_reg_mat * mask).sum() / mask.sum().clamp_min(1.0)

    return L_pred + lam_reg * L_reg

def build_pos_weight(train_examples: List[Dict], device: torch.device) -> torch.Tensor:
    K = len(UNIFIED_LABELS)
    pos = np.zeros(K, dtype=np.float64)
    neg = np.zeros(K, dtype=np.float64)
    for ex in train_examples:
        y = ex["y"]
        m = ex["mask"]
        pos += (y * m)
        neg += ((1 - y) * m)
    pos = np.maximum(pos, 1.0)
    pw = neg / pos
    pw = np.clip(pw, 1.0, 50.0)
    return torch.tensor(pw, dtype=torch.float32, device=device)


# ----------------------------
# Masked multi-label ranking metrics
# ----------------------------
def masked_hamming_loss(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    denom = m.sum()
    if denom == 0:
        return 0.0
    return float((y_true[m] != y_pred[m]).sum() / denom)

def masked_label_ranking_loss(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray) -> float:
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

def masked_coverage_error(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray) -> float:
    N, K = y_true.shape
    covs = []
    for i in range(N):
        idx = np.where(mask[i] > 0.5)[0]
        if idx.size == 0:
            continue
        yt = y_true[i, idx].astype(int)
        sc = y_score[i, idx].astype(float)
        pos = np.where(yt == 1)[0]
        if pos.size == 0:
            covs.append(0.0)
            continue
        order = np.argsort(-sc)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.size)
        max_rank = ranks[pos].max()
        covs.append(float(max_rank + 1))
    return float(np.mean(covs)) if covs else 0.0

def masked_lrap(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray) -> float:
    N, K = y_true.shape
    vals = []
    for i in range(N):
        idx = np.where(mask[i] > 0.5)[0]
        if idx.size == 0:
            continue
        yt = y_true[i, idx].astype(int)
        sc = y_score[i, idx].astype(float)
        if yt.sum() == 0:
            vals.append(0.0)
            continue
        order = np.argsort(-sc)
        yt_sorted = yt[order]
        cumsum_pos = np.cumsum(yt_sorted)
        pos_positions = np.where(yt_sorted == 1)[0]
        precisions = cumsum_pos[pos_positions] / (pos_positions + 1)
        vals.append(float(np.mean(precisions)))
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------
# Metrics + Threshold tuning
# ----------------------------
def compute_metrics_from_probs(
    y_true: np.ndarray,    # [N,K]
    y_prob: np.ndarray,    # [N,K]
    mask: np.ndarray,      # [N,K]
    thresholds: Optional[np.ndarray] = None,  # [K]
) -> Dict[str, Any]:
    K = y_true.shape[1]
    if thresholds is None:
        thresholds = np.full(K, 0.5, dtype=np.float32)

    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)

    # Per-label metrics (only where defined)
    per_label_f1 = {}
    per_label_ap = {}
    for k, lab in enumerate(UNIFIED_LABELS):
        m_k = mask[:, k].astype(bool)
        if m_k.sum() == 0:
            per_label_f1[lab] = None
            per_label_ap[lab] = None
            continue
        yt = y_true[m_k, k].astype(int)
        yp = y_pred[m_k, k].astype(int)
        ypr = y_prob[m_k, k].astype(float)
        per_label_f1[lab] = float(f1_score(yt, yp, zero_division=0))
        try:
            per_label_ap[lab] = float(average_precision_score(yt, ypr))
        except Exception:
            per_label_ap[lab] = None

    # Micro/macro F1 over masked positions
    mbool = mask.astype(bool)
    y_true_flat = y_true[mbool].astype(int)
    y_pred_flat = y_pred[mbool].astype(int)
    micro_f1 = float(f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0))
    f1_vals = [v for v in per_label_f1.values() if v is not None]
    macro_f1 = float(np.mean(f1_vals)) if len(f1_vals) else 0.0

    # Jaccard (samples)
    jac = float(jaccard_score(y_true.astype(int), y_pred.astype(int), average="samples", zero_division=0))

    # Extra metrics
    ham = masked_hamming_loss(y_true.astype(int), y_pred.astype(int), mask)
    rloss = masked_label_ranking_loss(y_true.astype(int), y_prob.astype(float), mask)
    lrap = masked_lrap(y_true.astype(int), y_prob.astype(float), mask)
    cov = masked_coverage_error(y_true.astype(int), y_prob.astype(float), mask)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "jaccard_samples": jac,
        "hamming_loss_masked": ham,
        "ranking_loss_masked": rloss,
        "lrap_masked": lrap,
        "coverage_error_masked": cov,
        "per_label_f1": per_label_f1,
        "per_label_ap": per_label_ap,
    }

def tune_thresholds_per_label(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    grid: int = 101
) -> np.ndarray:
    K = y_true.shape[1]
    thresholds = np.full(K, 0.5, dtype=np.float32)
    cand = np.linspace(0.0, 1.0, grid, dtype=np.float32)

    for k in range(K):
        m_k = mask[:, k].astype(bool)
        if m_k.sum() == 0:
            continue
        yt = y_true[m_k, k].astype(int)
        pr = y_prob[m_k, k].astype(float)
        best_t = 0.5
        best_f1 = -1.0
        for t in cand:
            yp = (pr >= t).astype(int)
            f1 = f1_score(yt, yp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[k] = best_t
    return thresholds


# ----------------------------
# Train / Inference loops
# ----------------------------
@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    uncertainty_mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[str], List[str], List[str]]:
    model.eval()
    ys, ps, ms = [], [], []
    us = []
    ids_all, raw_all, langs_all = [], [], []

    for batch in tqdm(loader, desc="infer", leave=False):
        input_ids = batch.input_ids.to(device)
        attn = batch.attention_mask.to(device)
        y = batch.y.to(device)
        m = batch.mask.to(device)

        out = model(input_ids=input_ids, attention_mask=attn)
        p = out["p"]

        ys.append(to_numpy(y))
        ps.append(to_numpy(p))
        ms.append(to_numpy(m))

        ids_all.extend(batch.ids)
        raw_all.extend(batch.raw_texts)
        langs_all.extend(batch.langs)

        if uncertainty_mode == "evidential":
            us.append(to_numpy(out["u"]))

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    mask = np.concatenate(ms, axis=0)
    extra: Dict[str, np.ndarray] = {}
    if uncertainty_mode == "evidential" and len(us) > 0:
        extra["u"] = np.concatenate(us, axis=0)

    return y_true, y_prob, mask, extra, ids_all, raw_all, langs_all


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    uncertainty_mode: str,
    pos_weight: Optional[torch.Tensor],
    tau: float,
    lam_reg: float,
    grad_clip: float
) -> float:
    model.train()
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch.input_ids.to(device)
        attn = batch.attention_mask.to(device)
        y = batch.y.to(device)
        m = batch.mask.to(device)

        out = model(input_ids=input_ids, attention_mask=attn)

        if uncertainty_mode == "ambiguity_weight":
            logits = out["logits"]
            p = out["p"]
            w = ambiguity_weight_from_probs(p.detach(), m, tau=tau)  # [B]

            bce_elem = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none",
                pos_weight=pos_weight if pos_weight is not None else None
            )
            bce_elem = bce_elem * m
            denom = m.sum(dim=1).clamp_min(1.0)
            per_sample = bce_elem.sum(dim=1) / denom  # [B]
            loss = (w * per_sample).mean()
        else:
            loss = evidential_loss(out["alpha"], out["beta"], y, m, lam_reg=lam_reg)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


# ----------------------------
# Interpretability (Gradient × Input)
# ----------------------------
def explain_grad_x_input(
    model: EmotionModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    text_with_prefix: str,
    labels_idx: List[int],
    max_length: int,
    topk: int,
    uncertainty_mode: str,
) -> Dict[str, Any]:
    model.eval()
    enc = tokenizer(
        text_with_prefix,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    embed_layer = model.encoder.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)
    inputs_embeds = inputs_embeds.detach().clone().requires_grad_(True)

    out = model.forward_with_embeddings(input_ids=input_ids, attention_mask=attn, inputs_embeds=inputs_embeds)

    target_base = out["logits"] if uncertainty_mode == "ambiguity_weight" else out["p"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    special_ids = set(filter(lambda x: x is not None, [
        tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    ]))
    valid_pos = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]

    explanations: Dict[str, List[Dict[str, float]]] = {}
    for k in labels_idx:
        model.zero_grad(set_to_none=True)
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        target = target_base[0, k]
        target.backward(retain_graph=True)

        grad = inputs_embeds.grad[0]           # [T,H]
        emb  = inputs_embeds.detach()[0]       # [T,H]
        scores = (grad * emb).sum(dim=1).abs() # [T]
        scores = scores.detach().cpu().numpy()

        vp = np.array(valid_pos, dtype=int)
        if vp.size == 0:
            explanations[UNIFIED_LABELS[k]] = []
            continue

        v_scores = scores[vp]
        order = np.argsort(-v_scores)[:topk]
        items = []
        for j in order:
            pos = int(vp[j])
            items.append({"token": tokens[pos], "score": float(v_scores[j]), "position": pos})
        explanations[UNIFIED_LABELS[k]] = items

    return {"tokens": tokens, "explanations": explanations}


def write_interpretability_reports(
    model: EmotionModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    examples: List[Dict],
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    out_path: str,
    max_length: int,
    topk: int,
    explain_n: int,
    explain_strategy: str,
    uncertainty_mode: str,
):
    safe_makedirs(os.path.dirname(out_path))
    n = min(explain_n, len(examples))

    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc="explain", leave=False):
            ex = examples[i]
            probs = y_prob[i]
            mask = ex["mask"]
            defined = np.where(mask > 0.5)[0]
            if defined.size == 0:
                continue

            if explain_strategy == "predicted":
                pred = [k for k in defined.tolist() if probs[k] >= thresholds[k]]
                if len(pred) == 0:
                    labels_idx = defined[np.argsort(-probs[defined])[:2]].tolist()
                else:
                    labels_idx = pred[:3]
            else:
                labels_idx = defined[np.argsort(-probs[defined])[:2]].tolist()

            report = explain_grad_x_input(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text_with_prefix=ex["text"],
                labels_idx=labels_idx,
                max_length=max_length,
                topk=topk,
                uncertainty_mode=uncertainty_mode,
            )
            record = {
                "id": ex["id"],
                "lang": ex["lang"],
                "raw_text": ex["raw_text"],
                "labels_explained": [UNIFIED_LABELS[k] for k in labels_idx],
                "probs": {UNIFIED_LABELS[k]: float(probs[k]) for k in labels_idx},
                "token_attributions": report["explanations"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# Threshold mode helpers
# ----------------------------
def thresholds_global(val_true, val_prob, val_mask, grid) -> np.ndarray:
    return tune_thresholds_per_label(val_true, val_prob, val_mask, grid=grid)

def thresholds_per_lang(val_true, val_prob, val_mask, val_langs: List[str], grid) -> Dict[str, np.ndarray]:
    out = {}
    for lang in ["en", "zh"]:
        idx = [i for i, lg in enumerate(val_langs) if lg == lang]
        if len(idx) == 0:
            continue
        yt = val_true[idx]
        yp = val_prob[idx]
        mk = val_mask[idx]
        out[lang] = tune_thresholds_per_label(yt, yp, mk, grid=grid)
    return out

def select_thresholds_for_lang(thr_mode: str, thr_global: np.ndarray, thr_lang: Dict[str, np.ndarray], lang: str) -> np.ndarray:
    if thr_mode == "global":
        return thr_global
    if lang in thr_lang:
        return thr_lang[lang]
    return thr_global


# ----------------------------
# Per-language metrics
# ----------------------------
def metrics_for_lang(lang: str, examples: List[Dict], y_true, y_prob, mask, thresholds) -> Optional[Dict[str, Any]]:
    idx = [i for i, ex in enumerate(examples) if ex["lang"] == lang]
    if len(idx) == 0:
        return None
    yt = y_true[idx]
    yp = y_prob[idx]
    mk = mask[idx]
    return compute_metrics_from_probs(yt, yp, mk, thresholds=thresholds)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Data paths
    ap.add_argument("--english_train", type=str, required=True)
    ap.add_argument("--english_val", type=str, required=True)
    ap.add_argument("--english_test", type=str, required=True)
    ap.add_argument("--chinese_train", type=str, required=True)
    ap.add_argument("--chinese_val", type=str, required=True)
    ap.add_argument("--chinese_test", type=str, required=True)

    # Language controls
    ap.add_argument("--train_lang", type=str, default="both", choices=["en", "zh", "both"])
    ap.add_argument("--threshold_mode", type=str, default="global", choices=["global", "per_lang"])
    ap.add_argument("--eval_by_lang", action="store_true")

    # Model/training
    ap.add_argument("--backbone", type=str, default="xlm-roberta-base",
                    choices=["xlm-roberta-base", "mdeberta-v3-base"])
    ap.add_argument("--uncertainty_mode", type=str, default="ambiguity_weight",
                    choices=["ambiguity_weight", "evidential"])
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # Uncertainty hyperparams
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lam_reg", type=float, default=0.1)
    ap.add_argument("--use_pos_weight", action="store_true")

    # Threshold tuning
    ap.add_argument("--thr_grid", type=int, default=101)

    # Interpretability
    ap.add_argument("--explain_split", type=str, default="none", choices=["val", "test", "none"])
    ap.add_argument("--explain_n", type=int, default=25)
    ap.add_argument("--explain_topk", type=int, default=12)
    ap.add_argument("--explain_strategy", type=str, default="predicted", choices=["predicted", "top2"])

    # Output
    ap.add_argument("--out_dir", type=str, default="runs/multilabel_run")

    args = ap.parse_args()
    seed_everything(args.seed)
    safe_makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load data
    en_tr = load_english_tsv(args.english_train)
    en_va = load_english_tsv(args.english_val)
    en_te = load_english_tsv(args.english_test)

    zh_tr = load_chinese_csv(args.chinese_train)
    zh_va = load_chinese_csv(args.chinese_val)
    zh_te = load_chinese_csv(args.chinese_test)

    train_ex_all = df_to_examples_english(en_tr) + df_to_examples_chinese(zh_tr)
    val_ex_all   = df_to_examples_english(en_va) + df_to_examples_chinese(zh_va)
    test_ex_all  = df_to_examples_english(en_te) + df_to_examples_chinese(zh_te)

    # Filter for training language
    train_ex = filter_by_lang(train_ex_all, args.train_lang)
    val_ex   = filter_by_lang(val_ex_all, args.train_lang)
    test_ex  = filter_by_lang(test_ex_all, args.train_lang)

    print(f"[Data] (raw) train={len(train_ex_all)} val={len(val_ex_all)} test={len(test_ex_all)}")
    print(f"[Data] (filtered) lang={args.train_lang} train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")

    # Tokenizer + loaders
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    collate = Collator(tokenizer, max_length=args.max_length)

    train_loader = DataLoader(EmotionDataset(train_ex), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(EmotionDataset(val_ex), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(EmotionDataset(test_ex), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model
    model = EmotionModel(
        backbone_name=args.backbone,
        num_labels=len(UNIFIED_LABELS),
        uncertainty_mode=args.uncertainty_mode,
        dropout=args.dropout
    ).to(device)

    # Optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)

    num_steps = args.epochs * max(1, math.ceil(len(train_ex) / args.batch_size))
    warmup_steps = int(args.warmup_ratio * num_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    # pos_weight (only used in ambiguity_weight)
    pos_weight = None
    if args.use_pos_weight and args.uncertainty_mode == "ambiguity_weight":
        pos_weight = build_pos_weight(train_ex, device=device)
        print("[Info] Using pos_weight per label for BCEWithLogits.")

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "model.pt")

    # Train
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            uncertainty_mode=args.uncertainty_mode,
            pos_weight=pos_weight,
            tau=args.tau,
            lam_reg=args.lam_reg,
            grad_clip=args.grad_clip
        )

        # Validate
        yv, pv, mv, _, _, _, val_langs = run_inference(model, val_loader, device, args.uncertainty_mode)

        if args.threshold_mode == "global":
            thr_g = thresholds_global(yv, pv, mv, grid=args.thr_grid)
            thr_l = {}
            val_metrics = compute_metrics_from_probs(yv, pv, mv, thresholds=thr_g)
        else:
            thr_g = thresholds_global(yv, pv, mv, grid=args.thr_grid)
            thr_l = thresholds_per_lang(yv, pv, mv, val_langs, grid=args.thr_grid)

            # For "overall val" metrics under per-lang thresholds:
            # apply en threshold to en rows, zh threshold to zh rows
            y_pred = np.zeros_like(pv, dtype=np.int32)
            for i, lg in enumerate(val_langs):
                thr_i = select_thresholds_for_lang("per_lang", thr_g, thr_l, lg)
                y_pred[i] = (pv[i] >= thr_i).astype(np.int32)

            # Compute metrics using the chosen thresholds per row
            # We'll reuse compute_metrics_from_probs by passing a dummy threshold (not possible per-row)
            # so we compute manually by calling compute_metrics_from_probs per language and average macro/micro properly:
            m_en = metrics_for_lang("en", val_ex, yv, pv, mv, thresholds=select_thresholds_for_lang("per_lang", thr_g, thr_l, "en"))
            m_zh = metrics_for_lang("zh", val_ex, yv, pv, mv, thresholds=select_thresholds_for_lang("per_lang", thr_g, thr_l, "zh"))

            # Combine: report global with global thresholds (stable), and keep per-lang separately
            val_metrics = compute_metrics_from_probs(yv, pv, mv, thresholds=thr_g)
            val_metrics["val_per_lang_thresholds"] = {"en": m_en, "zh": m_zh}

        print(
            f"[Epoch {ep}] train_loss={tr_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_micro_f1={val_metrics['micro_f1']:.4f} "
            f"val_jaccard={val_metrics['jaccard_samples']:.4f} "
            f"val_hamming={val_metrics['hamming_loss_masked']:.4f} "
            f"val_rankloss={val_metrics['ranking_loss_masked']:.4f}"
        )

        with open(os.path.join(args.out_dir, f"metrics_val_ep{ep}.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "thr_global": thr_g, "thr_lang": thr_l}, best_path)

            with open(os.path.join(args.out_dir, "thresholds_best_global.json"), "w", encoding="utf-8") as f:
                json.dump({lab: float(thr_g[i]) for i, lab in enumerate(UNIFIED_LABELS)}, f, indent=2)

            if args.threshold_mode == "per_lang":
                for lg, thr in thr_l.items():
                    with open(os.path.join(args.out_dir, f"thresholds_best_{lg}.json"), "w", encoding="utf-8") as f:
                        json.dump({lab: float(thr[i]) for i, lab in enumerate(UNIFIED_LABELS)}, f, indent=2)

            print(f"  [Best] saved: {best_path}")

    # Test
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    thr_g = ckpt.get("thr_global", np.full(len(UNIFIED_LABELS), 0.5, dtype=np.float32))
    thr_l = ckpt.get("thr_lang", {})

    yt, pt, mt, extra, ids_all, raw_all, test_langs = run_inference(model, test_loader, device, args.uncertainty_mode)

    # Overall metrics: global thresholds
    test_metrics = compute_metrics_from_probs(yt, pt, mt, thresholds=thr_g)

    # Per-language metrics (optionally with per-lang thresholds if enabled)
    by_lang = {}
    if args.eval_by_lang:
        for lg in ["en", "zh"]:
            thr_use = select_thresholds_for_lang(args.threshold_mode, thr_g, thr_l, lg)
            m_lg = metrics_for_lang(lg, test_ex, yt, pt, mt, thresholds=thr_use)
            by_lang[lg] = m_lg

    if args.eval_by_lang:
        test_metrics["test_by_lang"] = by_lang

    with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Uncertainty summary
    if args.uncertainty_mode == "evidential" and "u" in extra:
        u = extra["u"]
        u_mean = {}
        for k, lab in enumerate(UNIFIED_LABELS):
            m_k = mt[:, k].astype(bool)
            u_mean[lab] = float(u[m_k, k].mean()) if m_k.sum() else None
        with open(os.path.join(args.out_dir, "uncertainty_mean_test.json"), "w", encoding="utf-8") as f:
            json.dump(u_mean, f, indent=2)

    # Print summary
    print("\n[Test:Overall]")
    print(f"  macro_f1         : {test_metrics['macro_f1']:.4f}")
    print(f"  micro_f1         : {test_metrics['micro_f1']:.4f}")
    print(f"  jaccard_samples  : {test_metrics['jaccard_samples']:.4f}")
    print(f"  hamming_loss     : {test_metrics['hamming_loss_masked']:.4f}")
    print(f"  ranking_loss     : {test_metrics['ranking_loss_masked']:.4f}")
    print(f"  lrap             : {test_metrics['lrap_masked']:.4f}")
    print(f"  coverage_error   : {test_metrics['coverage_error_masked']:.4f}")

    if args.eval_by_lang:
        for lg in ["en", "zh"]:
            m_lg = by_lang.get(lg)
            if m_lg is None:
                continue
            print(f"\n[Test:{lg.upper()}] macro_f1={m_lg['macro_f1']:.4f} micro_f1={m_lg['micro_f1']:.4f} "
                  f"jaccard={m_lg['jaccard_samples']:.4f} hamming={m_lg['hamming_loss_masked']:.4f} "
                  f"rankloss={m_lg['ranking_loss_masked']:.4f}")

    # Interpretability
    if args.explain_split != "none":
        if args.explain_split == "val":
            # run val inference again for alignment
            yx, px, mx, _, _, _, langs_x = run_inference(model, val_loader, device, args.uncertainty_mode)
            examples = val_ex
            probs = px
            split_name = "val"
        else:
            examples = test_ex
            probs = pt
            split_name = "test"

        # choose threshold vector for explanation:
        # if per_lang, we will use per row (we approximate by using global to decide predicted labels)
        thr_for_explain = thr_g

        out_path = os.path.join(args.out_dir, f"interpretability_{split_name}.jsonl")
        write_interpretability_reports(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=examples,
            y_prob=probs,
            thresholds=thr_for_explain,
            out_path=out_path,
            max_length=args.max_length,
            topk=args.explain_topk,
            explain_n=args.explain_n,
            explain_strategy=args.explain_strategy,
            uncertainty_mode=args.uncertainty_mode
        )
        print(f"[Interpretability] saved: {out_path}")

    print("\n[Saved]")
    print(f"  {os.path.join(args.out_dir, 'metrics_test.json')}")
    print(f"  {os.path.join(args.out_dir, 'model.pt')}")
    print(f"  {os.path.join(args.out_dir, 'thresholds_best_global.json')}")


if __name__ == "__main__":
    main()
