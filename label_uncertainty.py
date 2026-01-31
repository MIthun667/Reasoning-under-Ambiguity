#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

import data_io
import modeling

def load_ckpt(model, ckpt, device):
    obj = torch.load(ckpt, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"], strict=True)
    else:
        model.load_state_dict(obj, strict=True)
    model.eval()

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
    ap.add_argument("--out_dir", default="analysis_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = data_io.load_multilingual_tsv(args.data_path, args.lang)
    text_col = get_text_col(df)

    texts = [f"<{args.lang.upper()}> {t}" for t in df[text_col].astype(str).tolist()]

    model = modeling.EmotionModel(
        backbone_name=args.backbone,
        num_labels=len(data_io.UNIFIED_LABELS),
        uncertainty_mode=args.uncertainty_mode,
        dropout=0.0,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    load_ckpt(model, args.ckpt, device)

    probs = probs_batch(model, tok, texts, device, args.max_len)  # [N, K]

    # Focus on shared 11 labels
    labels = data_io.EN_LABELS
    idxs = [data_io.LABEL2IDX[l] for l in labels]
    p = probs[:, idxs]
    H = bern_entropy(p)  # [N, 11]

    rows = []
    for j, lab in enumerate(labels):
        rows.append({
            "label": lab,
            "entropy_mean": float(H[:, j].mean()),
            "entropy_std": float(H[:, j].std()),
            "prob_mean": float(p[:, j].mean()),
            "prob_std": float(p[:, j].std()),
        })

    out_csv = os.path.join(args.out_dir, f"label_uncertainty_{args.lang}.csv")
    pd.DataFrame(rows).sort_values("entropy_mean", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
