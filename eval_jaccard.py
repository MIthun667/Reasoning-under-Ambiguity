#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import data_io
import modeling


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        thresholds = ckpt.get("thresholds", None)
    else:
        model.load_state_dict(ckpt, strict=True)
        thresholds = None
    model.eval()
    return thresholds


@torch.no_grad()
def predict_probs(model, tokenizer, texts, device, batch_size=16, max_length=192):
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(enc["input_ids"], enc["attention_mask"])
        probs_all.append(out["probs"].cpu().numpy())
    return np.vstack(probs_all)


def jaccard_score(y_true, y_pred):
    """
    y_true, y_pred: binary vectors [K]
    """
    inter = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()
    if union == 0:
        return 1.0
    return inter / union


def evaluate_jaccard(
    model,
    tokenizer,
    df,
    label_cols,
    lang_tag,
    thresholds,
    device,
    batch_size=16,
    max_length=192,
):
    texts = [f"<{lang_tag}> {t}" for t in df["text"].astype(str).tolist()]
    probs = predict_probs(model, tokenizer, texts, device, batch_size, max_length)

    # thresholds
    if thresholds is None:
        thr = np.full(len(label_cols), 0.5)
    else:
        thr = np.asarray(thresholds)

    y_true = df[label_cols].values.astype(int)
    y_pred = (probs >= thr).astype(int)

    scores = [jaccard_score(y_true[i], y_pred[i]) for i in range(len(df))]
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "es", "ar", "both"])
    ap.add_argument("--uncertainty_mode", default="baseline",
                    choices=["baseline", "ambiguity_weight", "evidential"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if args.lang == "both":
        # expects merged multilingual test set
        df = data_io.load_multilingual_test_all(args.test_path)
        lang_tags = df["lang"].tolist()
    else:
        df = data_io.load_multilingual_tsv(args.test_path, args.lang)
        lang_tags = [args.lang.upper()] * len(df)

    if "text" not in df.columns and "Tweet" in df.columns:
        df = df.rename(columns={"Tweet": "text"})

    label_cols = list(data_io.UNIFIED_LABELS)

    # Ensure missing labels exist (important!)
    for lab in label_cols:
        if lab not in df.columns:
            df[lab] = 0

    # Model
    model = modeling.EmotionModel(
        backbone_name=args.backbone,
        num_labels=len(label_cols),
        uncertainty_mode=args.uncertainty_mode,
        dropout=0.0,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)

    thresholds = load_checkpoint(model, args.ckpt, device)

    # Evaluate (handle multi-lang tags)
    if args.lang == "both":
        scores = []
        for lang in ["EN", "ES", "AR"]:
            sub = df[df["lang"] == lang.lower()]
            if len(sub) == 0:
                continue
            s = evaluate_jaccard(
                model, tokenizer, sub, label_cols, lang,
                thresholds, device,
                args.batch_size, args.max_length
            )
            scores.append(s)
        jaccard = float(np.mean(scores))
    else:
        jaccard = evaluate_jaccard(
            model, tokenizer, df, label_cols, args.lang.upper(),
            thresholds, device,
            args.batch_size, args.max_length
        )

    print("=" * 50)
    print(f"Checkpoint : {args.ckpt}")
    print(f"Language   : {args.lang}")
    print(f"Jaccard    : {jaccard:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()



"""
python eval_jaccard.py \
  --ckpt "/media/mithun/New Volume/Research/Multilabel/runs/en/model.pt" \
  --test_path "/media/mithun/New Volume/Research/Multilabel/Spanish/Spanish-E-c/test.txt" \
  --lang en \
  --uncertainty_mode ambiguity_weight





"""