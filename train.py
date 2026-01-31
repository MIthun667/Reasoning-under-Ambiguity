# -*- coding: utf-8 -*-
"""
Main training / evaluation entry point.

Supports:
- Single experiment run
- Internal ablation grid (--ablation_grid)
- Multilingual training (en | es | ar | both)
- Uncertainty-aware learning
- Safe checkpointing (PyTorch 2.6+ compatible)

Expected data format (TSV .txt) for all languages:
  ID, Tweet, anger, anticipation, disgust, fear, joy, love,
  optimism, pessimism, sadness, surprise, trust
"""

from __future__ import annotations

import os
import json
import argparse
import random
from typing import Dict, Any

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Local modules
import data_io
import modeling
import losses
import metrics
import ablation


# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Core experiment runner
# ----------------------------
def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs ONE experiment (train + val + test).

    Returns:
        metrics dict (HL, RL, miF1, maF1, AP, microAP)
    """
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_makedirs(cfg["out_dir"])

    # ----------------------------
    # Load data (EN/ES/AR all same format)
    # ----------------------------
    en_tr = data_io.load_multilingual_tsv(cfg["english_train"], "en")
    en_va = data_io.load_multilingual_tsv(cfg["english_val"], "en")
    en_te = data_io.load_multilingual_tsv(cfg["english_test"], "en")

    es_tr = data_io.load_multilingual_tsv(cfg["spanish_train"], "es")
    es_va = data_io.load_multilingual_tsv(cfg["spanish_val"], "es")
    es_te = data_io.load_multilingual_tsv(cfg["spanish_test"], "es")

    ar_tr = data_io.load_multilingual_tsv(cfg["arabic_train"], "ar")
    ar_va = data_io.load_multilingual_tsv(cfg["arabic_val"], "ar")
    ar_te = data_io.load_multilingual_tsv(cfg["arabic_test"], "ar")

    train_all = data_io.build_examples(en_tr, es_tr, ar_tr)
    val_all   = data_io.build_examples(en_va, es_va, ar_va)
    test_all  = data_io.build_examples(en_te, es_te, ar_te)

    train_ex = data_io.filter_by_lang(train_all, cfg["train_lang"])
    val_ex   = data_io.filter_by_lang(val_all, cfg["train_lang"])
    test_ex  = data_io.filter_by_lang(test_all, cfg["train_lang"])

    # ----------------------------
    # Tokenizer + loaders
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)

    train_loader, val_loader, test_loader = data_io.make_loaders(
        train_ex, val_ex, test_ex,
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = modeling.EmotionModel(
        backbone_name=cfg["backbone"],
        num_labels=len(data_io.UNIFIED_LABELS),
        uncertainty_mode=cfg["uncertainty_mode"],
        dropout=cfg["dropout"],
    ).to(device)

    # ----------------------------
    # Optimizer + scheduler
    # ----------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = cfg["epochs"] * max(1, len(train_loader))
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # pos_weight
    pos_weight = None
    if cfg.get("use_pos_weight", False):
        pos_weight = losses.build_pos_weight(train_ex, device)

    best_val = -1.0
    best_ckpt = None

    # ----------------------------
    # Training loop
    # ----------------------------
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = []

        for batch in tqdm(train_loader, desc=f"train ep{ep}", leave=False):
            optimizer.zero_grad(set_to_none=True)

            inputs = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            targets = batch.y.to(device)
            mask = batch.mask.to(device)

            outputs = model(inputs, attn)

            loss = losses.compute_total_loss(
                outputs=outputs,
                targets=targets,
                mask=mask,
                uncertainty_mode=cfg["uncertainty_mode"],
                pos_weight=pos_weight,
                tau=cfg["tau"],
                lam_reg=cfg["lam_reg"],
                pu_weight=cfg.get("pu_weight", 0.0),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss.append(loss.item())

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        y_true, y_prob, m_all = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch.input_ids.to(device)
                attn = batch.attention_mask.to(device)

                out = model(inputs, attn)
                probs = out["probs"]

                y_true.append(batch.y.numpy())
                y_prob.append(probs.cpu().numpy())
                m_all.append(batch.mask.numpy())

        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        mask   = np.concatenate(m_all)

        thresholds = metrics.tune_thresholds_per_label(
            y_true, y_prob, mask, grid=cfg["thr_grid"]
        )

        val_metrics = metrics.compute_all_metrics(
            y_true=y_true,
            y_prob=y_prob,
            mask=mask,
            thresholds=thresholds,
            label_names=data_io.UNIFIED_LABELS,
        )

        if val_metrics["maF1"] > best_val:
            best_val = val_metrics["maF1"]
            best_ckpt = {
                "model": model.state_dict(),
                "thresholds": thresholds.tolist(),  # SAFE to serialize
            }

    # ----------------------------
    # Test (best model)
    # ----------------------------
    model.load_state_dict(best_ckpt["model"])
    thresholds = np.array(best_ckpt["thresholds"], dtype=np.float32)

    model.eval()
    y_true, y_prob, m_all = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)

            out = model(inputs, attn)
            probs = out["probs"]

            y_true.append(batch.y.numpy())
            y_prob.append(probs.cpu().numpy())
            m_all.append(batch.mask.numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    mask   = np.concatenate(m_all)

    test_metrics = metrics.compute_all_metrics(
        y_true=y_true,
        y_prob=y_prob,
        mask=mask,
        thresholds=thresholds,
        label_names=data_io.UNIFIED_LABELS,
    )

    # Save outputs
    with open(os.path.join(cfg["out_dir"], "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    torch.save(
        {"model": best_ckpt["model"], "thresholds": best_ckpt["thresholds"]},
        os.path.join(cfg["out_dir"], "model.pt")
    )

    return {
        "HL": test_metrics["HL"],
        "RL": test_metrics["RL"],
        "miF1": test_metrics["miF1"],
        "maF1": test_metrics["maF1"],
        "AP": test_metrics["AP"],
        "microAP": test_metrics["microAP"],
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Data (EN)
    ap.add_argument("--english_train", required=True)
    ap.add_argument("--english_val", required=True)
    ap.add_argument("--english_test", required=True)

    # Data (ES)
    ap.add_argument("--spanish_train", required=True)
    ap.add_argument("--spanish_val", required=True)
    ap.add_argument("--spanish_test", required=True)

    # Data (AR)
    ap.add_argument("--arabic_train", required=True)
    ap.add_argument("--arabic_val", required=True)
    ap.add_argument("--arabic_test", required=True)

    # Model
    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument(
        "--uncertainty_mode",
        default="baseline",
        choices=["baseline", "ambiguity_weight", "evidential"]
    )

    # Training
    ap.add_argument("--train_lang", default="both", choices=["en", "es", "ar", "both"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # Loss extras
    ap.add_argument("--use_pos_weight", action="store_true")
    ap.add_argument("--pu_weight", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lam_reg", type=float, default=0.1)

    # Thresholds
    ap.add_argument("--thr_grid", type=int, default=101)

    # Misc
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out_dir", default="runs/exp")

    # Ablation
    ap.add_argument("--ablation_grid", action="store_true")
    ap.add_argument("--grid_name", default="default", choices=["default", "extended"])
    ap.add_argument("--grid_max_runs", type=int, default=None)

    args = ap.parse_args()

    if args.ablation_grid:
        ablation.run_ablation_grid(args, run_experiment)
    else:
        run_experiment(vars(args))


if __name__ == "__main__":
    main()
