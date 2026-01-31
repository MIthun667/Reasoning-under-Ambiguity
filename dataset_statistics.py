#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
from collections import Counter
import data_io


def tokenize_len(text):
    # simple whitespace tokenization (paper-friendly & reproducible)
    return len(str(text).split())


def load_split(path, lang):
    df = data_io.load_multilingual_tsv(path, lang)
    if "text" not in df.columns and "Tweet" in df.columns:
        df = df.rename(columns={"Tweet": "text"})
    return df


def get_label_cols(df):
    return [c for c in data_io.UNIFIED_LABELS if c in df.columns]


def compute_split_stats(df, label_cols):
    n = len(df)
    text_lens = df["text"].astype(str).map(tokenize_len)

    label_mat = df[label_cols].values
    label_counts = label_mat.sum(axis=1)

    stats = {
        "num_instances": n,
        "avg_tokens": text_lens.mean(),
        "median_tokens": text_lens.median(),
        "num_labels": len(label_cols),
        "avg_labels_per_instance": label_counts.mean(),
        "pct_multilabel": (label_counts >= 2).mean() * 100.0,
    }
    return stats, label_mat


def label_statistics(label_mat, label_cols):
    freq = label_mat.sum(axis=0)
    return pd.DataFrame({
        "label": label_cols,
        "count": freq,
        "percent": freq / label_mat.shape[0] * 100.0
    }).sort_values("count", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en_root", required=True)
    ap.add_argument("--es_root", required=True)
    ap.add_argument("--ar_root", required=True)
    ap.add_argument("--out_dir", default="analysis_out/dataset_stats")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    languages = {
        "EN": (args.en_root, "en"),
        "ES": (args.es_root, "es"),
        "AR": (args.ar_root, "ar"),
    }

    rows = []

    for lang_name, (root, lang_code) in languages.items():
        for split in ["train", "validation", "test"]:
            path = os.path.join(root, f"{split}.txt")
            if not os.path.exists(path):
                continue

            df = load_split(path, lang_code)
            label_cols = get_label_cols(df)

            stats, label_mat = compute_split_stats(df, label_cols)
            row = {
                "language": lang_name,
                "split": split,
                **stats
            }
            rows.append(row)

            # Save label distribution (train only usually shown)
            if split == "train":
                lab_df = label_statistics(label_mat, label_cols)
                lab_df.to_csv(
                    os.path.join(args.out_dir, f"dataset_stats_labels_{lang_name.lower()}.csv"),
                    index=False
                )

    overall = pd.DataFrame(rows)
    overall.to_csv(os.path.join(args.out_dir, "dataset_stats_overall.csv"), index=False)

    print("Saved:")
    print(" - dataset_stats_overall.csv")
    print(" - dataset_stats_labels_<lang>.csv")


if __name__ == "__main__":
    main()
