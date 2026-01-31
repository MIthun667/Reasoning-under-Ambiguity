#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import os

def tex_escape(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return (s.replace("&", "\\&")
             .replace("%", "\\%")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("$", "\\$")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))

def pick_text(row, paraphrase_col, raw_col):
    """
    Use paraphrase if it is non-empty and non-NaN, else fall back to raw text.
    """
    v = row.get(paraphrase_col, "")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = ""
    if isinstance(v, str) and v.strip() != "":
        return v.strip()

    raw = row.get(raw_col, "")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        raw = ""
    return str(raw).strip()

def load_rows(csv_path, k, select, seed):
    df = pd.read_csv(csv_path)

    # Keep only top-1 neighbors (rank 1) to avoid multiple ranks per dev
    if "train_nn_rank" in df.columns:
        df = df[df["train_nn_rank"] == 1].copy()

    if len(df) == 0:
        raise ValueError(f"No usable rows in {csv_path}")

    # Selection strategy
    if select == "top_sim":
        df = df.sort_values("cosine_sim", ascending=False).head(k)
    elif select == "random":
        df = df.sample(n=min(k, len(df)), random_state=seed)
    else:
        # first_k
        df = df.head(k)

    return df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn_en", required=True)
    ap.add_argument("--nn_es", required=True)
    ap.add_argument("--nn_ar", required=True)
    ap.add_argument("--k_per_lang", type=int, default=2, help="rows per language in table")
    ap.add_argument("--select", choices=["top_sim", "random", "first_k"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tex", default="nn_examples_multi.tex")

    # Table width settings
    ap.add_argument("--dev_w", default="5.0cm")
    ap.add_argument("--train_w", default="5.0cm")
    ap.add_argument("--label_w", default="3.3cm")
    args = ap.parse_args()

    blocks = [
        ("EN", args.nn_en),
        ("ES", args.nn_es),
        ("AR", args.nn_ar),
    ]

    rows = []
    for lang, path in blocks:
        sub = load_rows(path, args.k_per_lang, args.select, args.seed)
        for _, r in sub.iterrows():
            dev_text = pick_text(r, "dev_paraphrase", "dev_text")
            tr_text  = pick_text(r, "train_paraphrase", "train_text")
            sim = float(r["cosine_sim"]) if "cosine_sim" in r else None
            labs = str(r.get("train_labels", ""))

            rows.append({
                "Lang": lang,
                "Dev": tex_escape(dev_text),
                "Train": tex_escape(tr_text),
                "Sim": f"{sim:.2f}" if sim is not None else "--",
                "Labels": tex_escape(labs),
            })

    # Build LaTeX
    dev_w = args.dev_w
    train_w = args.train_w
    label_w = args.label_w

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{5pt}")
    latex.append(r"\renewcommand{\arraystretch}{1.15}")
    latex.append(rf"\begin{{tabular}}{{l p{{{dev_w}}} p{{{train_w}}} c p{{{label_w}}}}}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Lang} & \textbf{Dev Query (paraphrased)} & \textbf{Nearest Train Instance (paraphrased)} & \textbf{Sim.} & \textbf{Train Emotion Labels} \\")
    latex.append(r"\midrule")

    # Group by language with a subtle separator
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["Lang"], []).append(r)

    for li, lang in enumerate(["EN", "ES", "AR"]):
        if lang not in by_lang:
            continue
        for r in by_lang[lang]:
            latex.append(f"{r['Lang']} & {r['Dev']} & {r['Train']} & {r['Sim']} & {r['Labels']} \\\\")
        if lang != "AR":
            latex.append(r"\addlinespace[2pt]")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{")
    latex.append(r"Nearest-neighbor explanations using cosine similarity in the learned embedding space. ")
    latex.append(r"For each development-set query, we report the most similar training instance along with the corresponding emotion labels. ")
    latex.append(r"Texts are lightly paraphrased for clarity and anonymization.")
    latex.append(r"}")
    latex.append(r"\label{tab:nn_examples}")
    latex.append(r"\end{table*}")

    out = "\n".join(latex)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"Saved: {args.out_tex}")

if __name__ == "__main__":
    main()


"""

python nn_csv_to_latex.py \
  --nn_en "/media/mithun/New Volume/Research/Multilabel/analysis_out/nn_en.csv" \
  --nn_es "/media/mithun/New Volume/Research/Multilabel/analysis_out/nn_es.csv" \
  --nn_ar "/media/mithun/New Volume/Research/Multilabel/analysis_out/nn_ar.csv" \
  --k_per_lang 2 \
  --select random \
  --seed 42 \
  --out_tex "/media/mithun/New Volume/Research/Multilabel/analysis_out/nn_examples_multi.tex"
"""