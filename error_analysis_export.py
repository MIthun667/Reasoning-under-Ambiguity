#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

import data_io
import modeling

def load_ckpt(model, ckpt, device):
    obj = torch.load(ckpt, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"], strict=True)
        thresholds = obj.get("thresholds", None)
    else:
        model.load_state_dict(obj, strict=True)
        thresholds = None
    model.eval()
    return thresholds

@torch.no_grad()
def probs_batch(model, tokenizer, texts, device, max_len):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return out["probs"].detach().cpu().numpy()

def bern_entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def get_text_col(df):
    if "Tweet" in df.columns: return "Tweet"
    if "text" in df.columns: return "text"
    raise ValueError("No Tweet/text column found")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument("--uncertainty_mode", default="ambiguity_weight", choices=["baseline","ambiguity_weight","evidential"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--lang", required=True, choices=["en","es","ar"])
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--topn", type=int, default=80, help="how many errors to export")
    ap.add_argument("--out_csv", default="error_analysis.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = data_io.load_multilingual_tsv(args.data_path, args.lang)

    text_col = get_text_col(df)
    lab_cols = [c for c in data_io.EN_LABELS if c in df.columns]
    if len(lab_cols) == 0:
        raise ValueError("No label columns found (use validation.txt with labels).")

    texts = [f"<{args.lang.upper()}> {t}" for t in df[text_col].astype(str).tolist()]
    y_true = df[lab_cols].astype(int).to_numpy()

    model = modeling.EmotionModel(
        backbone_name=args.backbone,
        num_labels=len(data_io.UNIFIED_LABELS),
        uncertainty_mode=args.uncertainty_mode,
        dropout=0.0,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    thresholds = load_ckpt(model, args.ckpt, device)

    probs_full = probs_batch(model, tok, texts, device, args.max_len)
    idxs = [data_io.LABEL2IDX[c] for c in lab_cols]
    y_prob = probs_full[:, idxs]

    if thresholds is not None:
        thr = np.array([float(thresholds[i]) for i in idxs])
    else:
        thr = np.full((len(idxs),), 0.5, dtype=np.float32)
    y_pred = (y_prob >= thr[None, :]).astype(int)

    # per-instance micro-F1 proxy: F1 over labels for each row
    f1s = np.array([f1_score(y_true[i], y_pred[i], average="binary", zero_division=0) for i in range(len(y_true))])
    H = bern_entropy(y_prob).mean(axis=1)

    # pick worst (lowest f1), tie-break by high entropy
    order = np.lexsort((-H, f1s))  # f1 asc, H desc
    pick = order[: min(args.topn, len(order))]

    def labels_from_row(row):
        return [lab_cols[j] for j in range(len(lab_cols)) if row[j] == 1]

    rows = []
    for i in pick:
        gold = labels_from_row(y_true[i])
        pred = labels_from_row(y_pred[i])
        rows.append({
            "id": str(df.loc[i, "ID"]) if "ID" in df.columns else str(i),
            "lang": args.lang,
            "text": str(df.loc[i, text_col]),
            "entropy_H": float(H[i]),
            "gold": ",".join(gold),
            "pred": ",".join(pred),
            "probs": ",".join([f"{lab_cols[j]}:{y_prob[i,j]:.3f}" for j in range(len(lab_cols))]),
            "error_type": ""  # you fill manually: sarcasm / missing label / negation / domain shift / etc.
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()
