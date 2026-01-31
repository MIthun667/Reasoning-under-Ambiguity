#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import hamming_loss, label_ranking_loss, f1_score, average_precision_score

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

def get_label_cols(df):
    # Prefer shared 11 labels if present
    cols = [c for c in data_io.EN_LABELS if c in df.columns]
    if len(cols) >= 3:
        return cols
    # fallback: any binary-ish columns excluding ID/Tweet
    bad = {"ID", "Tweet"}
    cand = [c for c in df.columns if c not in bad]
    return cand

def get_text_col(df):
    if "Tweet" in df.columns: return "Tweet"
    if "text" in df.columns: return "text"
    raise ValueError("No Tweet/text column found")

def metrics(y_true, y_prob, y_pred):
    HL = hamming_loss(y_true, y_pred)
    RL = label_ranking_loss(y_true, y_prob)
    miF1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    maF1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    AP = average_precision_score(y_true, y_prob, average="macro")
    return {"HL": HL, "RL": RL, "miF1": miF1, "maF1": maF1, "AP": AP}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument("--uncertainty_mode", default="ambiguity_weight", choices=["baseline","ambiguity_weight","evidential"])
    ap.add_argument("--data_path", required=True, help="validation.txt (preferred) or test.txt if labeled")
    ap.add_argument("--lang", required=True, choices=["en","es","ar"])
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--bins", type=int, default=5, help="entropy buckets")
    ap.add_argument("--out_dir", default="analysis_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = data_io.load_multilingual_tsv(args.data_path, args.lang)
    text_col = get_text_col(df)
    lab_cols = get_label_cols(df)
    if len(lab_cols) == 0:
        raise ValueError("No label columns found. Use validation.txt (must contain labels).")

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

    # probs for only those label columns
    probs_full = probs_batch(model, tok, texts, device, args.max_len)  # [N, K]
    lab_idxs = [data_io.LABEL2IDX[c] for c in lab_cols]
    y_prob = probs_full[:, lab_idxs]

    # entropy over shared labels present
    H = bern_entropy(y_prob).mean(axis=1)  # [N]
    w = np.exp(-args.tau * H)

    # predictions
    if thresholds is not None:
        thr = np.array([float(thresholds[i]) for i in lab_idxs])
    else:
        thr = np.full((len(lab_idxs),), 0.5, dtype=np.float32)
    y_pred = (y_prob >= thr[None, :]).astype(int)

    # bucket by quantiles
    q = np.quantile(H, np.linspace(0, 1, args.bins+1))
    # ensure strictly increasing edges
    for i in range(1, len(q)):
        if q[i] <= q[i-1]:
            q[i] = q[i-1] + 1e-8

    rows = []
    for b in range(args.bins):
        lo, hi = q[b], q[b+1]
        idx = np.where((H >= lo) & (H <= hi if b == args.bins-1 else H < hi))[0]
        if len(idx) == 0:
            continue
        m = metrics(y_true[idx], y_prob[idx], y_pred[idx])
        rows.append({
            "bin": b+1,
            "H_range": f"[{lo:.4f}, {hi:.4f}]",
            "n": int(len(idx)),
            "H_mean": float(H[idx].mean()),
            "w_mean": float(w[idx].mean()),
            **m
        })

    out_csv = os.path.join(args.out_dir, f"ambiguity_vs_perf_{args.lang}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_all = os.path.join(args.out_dir, f"instance_scores_{args.lang}.csv")
    pd.DataFrame({
        "ID": df["ID"].astype(str) if "ID" in df.columns else np.arange(len(df)).astype(str),
        "H": H,
        "w": w
    }).to_csv(out_all, index=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_all}")

if __name__ == "__main__":
    main()
