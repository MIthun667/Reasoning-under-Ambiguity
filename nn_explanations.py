#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

import data_io


def get_text_col(df):
    if "Tweet" in df.columns:
        return "Tweet"
    if "text" in df.columns:
        return "text"
    raise ValueError("No Tweet/text column found.")


def get_label_cols(df):
    cols = [c for c in data_io.EN_LABELS if c in df.columns]
    if cols:
        return cols
    # fallback: anything that looks like labels (exclude ID/text)
    bad = {"ID", "Tweet", "text"}
    return [c for c in df.columns if c not in bad]


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, max_len=192, batch_size=32):
    """
    Returns L2-normalized CLS embeddings: [N, H]
    """
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)

        # CLS embedding:
        # For XLM-R / RoBERTa style, use last_hidden_state[:,0,:]
        vec = out.last_hidden_state[:, 0, :]  # [B,H]

        # L2 normalize for cosine sim via dot product
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        all_vecs.append(vec.cpu().numpy())

    return np.vstack(all_vecs)


def labels_to_str(row, label_cols):
    labs = [c for c in label_cols if int(row[c]) == 1]
    return ",".join(labs) if labs else "none"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--dev_path", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "es", "ar"])
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--topk", type=int, default=1, help="retrieve top-k nearest neighbors")
    ap.add_argument("--out_csv", default="nn_explanations.csv")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (needs labels in TRAIN; DEV may have labels but not required)
    df_tr = data_io.load_multilingual_tsv(args.train_path, args.lang)
    df_dv = data_io.load_multilingual_tsv(args.dev_path, args.lang)

    tcol_tr = get_text_col(df_tr)
    tcol_dv = get_text_col(df_dv)
    label_cols = get_label_cols(df_tr)
    if not label_cols:
        raise ValueError("No label columns found in train file (needed for reporting neighbor labels).")

    # Prepare texts with language tag (must match training style)
    tr_texts = [f"<{args.lang.upper()}> {t}" for t in df_tr[tcol_tr].astype(str).tolist()]
    dv_texts = [f"<{args.lang.upper()}> {t}" for t in df_dv[tcol_dv].astype(str).tolist()]

    # Model = encoder backbone (embedding space)
    tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    enc_model = AutoModel.from_pretrained(args.backbone).to(device)
    enc_model.eval()

    # Encode
    print("Encoding TRAIN...")
    tr_vecs = encode_texts(enc_model, tok, tr_texts, device, args.max_len, args.batch_size)  # [N,H]
    print("Encoding DEV...")
    dv_vecs = encode_texts(enc_model, tok, dv_texts, device, args.max_len, args.batch_size)  # [M,H]

    # Cosine similarity by dot product because normalized
    # For each dev vector, find top-k train vectors
    print("Retrieving nearest neighbors...")
    sims = dv_vecs @ tr_vecs.T  # [M,N]

    topk = int(args.topk)
    top_idx = np.argpartition(-sims, kth=min(topk, sims.shape[1]-1), axis=1)[:, :topk]

    # Sort the top-k properly
    rows = []
    for i in range(sims.shape[0]):
        cand = top_idx[i]
        cand = cand[np.argsort(-sims[i, cand])]  # sorted desc

        dev_id = str(df_dv.loc[i, "ID"]) if "ID" in df_dv.columns else f"dev_{i}"
        dev_text_raw = str(df_dv.loc[i, tcol_dv])

        for rank, j in enumerate(cand, start=1):
            tr_id = str(df_tr.loc[j, "ID"]) if "ID" in df_tr.columns else f"train_{j}"
            tr_text_raw = str(df_tr.loc[j, tcol_tr])
            tr_labels = labels_to_str(df_tr.loc[j], label_cols)
            sim = float(sims[i, j])

            rows.append({
                "dev_id": dev_id,
                "train_nn_rank": rank,
                "train_id": tr_id,
                "cosine_sim": sim,
                "dev_text": dev_text_raw,
                "train_text": tr_text_raw,
                "train_labels": tr_labels,
                # optional: fill manually for anonymization/paraphrase
                "dev_paraphrase": "",
                "train_paraphrase": "",
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print("Tip: fill dev_paraphrase/train_paraphrase manually for anonymization, then use it in the paper table.")


if __name__ == "__main__":
    main()
