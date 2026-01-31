#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_tex(path, s):
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def latex_table(df, caption, label, col_format=None, notes=None):
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=col_format if col_format else None
    )
    if notes:
        # insert notes right after caption/label block
        # simplest: append as a small line below table
        latex = latex.replace(r"\end{tabular}", r"\end{tabular}" + "\n" + notes)
    return latex

def fmt_pm(mean, std, digits=4):
    if pd.isna(mean) or pd.isna(std):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"

def build_ambiguity_vs_perf(in_csv, out_tex):
    df = pd.read_csv(in_csv)

    # Keep minimal, paper-friendly columns
    keep = ["bin", "H_range", "n", "H_mean", "w_mean", "miF1", "AP"]
    for c in keep:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {in_csv}")

    out = df[keep].copy()
    out = out.rename(columns={
        "bin": "Bin",
        "H_range": "$H$ range",
        "n": "$\\#Inst$",
        "H_mean": "$H$ mean",
        "w_mean": "$w$ mean",
        "miF1": "miF1 $\\uparrow$",
        "AP": "AP $\\uparrow$",
    })

    # round numeric columns
    for c in ["$H$ mean", "$w$ mean", "miF1 $\\uparrow$", "AP $\\uparrow$"]:
        out[c] = out[c].astype(float).map(lambda x: f"{x:.4f}")

    col_format = "c l r c c c c"
    caption = "Performance stratified by ambiguity level (entropy $H$) on Spanish validation/test instances."
    label = "tab:ambig_perf_es"
    notes = r"\vspace{2pt}\small{$w=\exp(-\tau H)$; higher $H$ indicates more ambiguous instances.}"
    tex = latex_table(out, caption, label, col_format=col_format, notes=notes)
    write_tex(out_tex, tex)

def build_label_uncertainty(in_csv, out_tex):
    df = pd.read_csv(in_csv)

    need = ["label", "entropy_mean", "entropy_std", "prob_mean", "prob_std"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {in_csv}")

    # keep top-K most uncertain to save space
    df = df.sort_values("entropy_mean", ascending=False).head(11).copy()

    out = pd.DataFrame({
        "Emotion": df["label"],
        "$\\mathbb{E}[H_k]$": df["entropy_mean"].map(lambda x: f"{float(x):.4f}"),
        "$\\mathrm{Std}(H_k)$": df["entropy_std"].map(lambda x: f"{float(x):.4f}"),
        "$\\mathbb{E}[p_k]$": df["prob_mean"].map(lambda x: f"{float(x):.4f}"),
        "$\\mathrm{Std}(p_k)$": df["prob_std"].map(lambda x: f"{float(x):.4f}"),
    })

    col_format = "l c c c c"
    caption = "Label-wise uncertainty on Arabic split. Higher entropy indicates more ambiguous emotion labels."
    label = "tab:label_uncertainty_ar"
    tex = latex_table(out, caption, label, col_format=col_format)
    write_tex(out_tex, tex)

def build_stability_ablation(in_csv, out_tex):
    df = pd.read_csv(in_csv)

    # expected columns
    # train_lang, uncertainty_mode, n_runs, HL_mean, HL_std, ...
    required = ["train_lang", "uncertainty_mode", "n_runs"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {in_csv}")

    # only show two key metrics to keep table small
    for base in ["miF1", "AP"]:
        if f"{base}_mean" not in df.columns or f"{base}_std" not in df.columns:
            raise ValueError(f"Missing {base}_mean/std in {in_csv}")

    out = pd.DataFrame({
        "Train": df["train_lang"].str.upper(),
        "Mode": df["uncertainty_mode"],
        "$n$": df["n_runs"].astype(int),
        "miF1 $\\uparrow$": [fmt_pm(m, s, 4) for m, s in zip(df["miF1_mean"], df["miF1_std"])],
        "AP $\\uparrow$": [fmt_pm(m, s, 4) for m, s in zip(df["AP_mean"], df["AP_std"])],
    })

    # nice ordering
    order_lang = {"en": 0, "es": 1, "ar": 2, "both": 3}
    out["_k"] = df["train_lang"].map(lambda x: order_lang.get(str(x).lower(), 99))
    out = out.sort_values(["_k", "Mode"]).drop(columns=["_k"])

    col_format = "l l r c c"
    caption = "Training stability across seeds (mean $\\pm$ std) for different uncertainty modes."
    label = "tab:stability"
    tex = latex_table(out, caption, label, col_format=col_format)
    write_tex(out_tex, tex)

def build_error_types(in_csv, out_tex):
    df = pd.read_csv(in_csv)

    if "error_type" not in df.columns:
        print(f"[SKIP] {in_csv}: no 'error_type' column found.")
        return

    # if not filled, table will be empty-ish
    counts = df["error_type"].fillna("").astype(str)
    counts = counts[counts.str.strip() != ""]
    if len(counts) == 0:
        print(f"[SKIP] {in_csv}: 'error_type' is empty. Fill it manually then rerun.")
        return

    tab = counts.value_counts().reset_index()
    tab.columns = ["Error Type", "Count"]
    tab["Percent"] = (tab["Count"] / tab["Count"].sum() * 100).map(lambda x: f"{x:.1f}\\%")

    col_format = "l r c"
    caption = "Manual error type distribution on English (sampled errors)."
    label = "tab:error_types_en"
    tex = latex_table(tab, caption, label, col_format=col_format)
    write_tex(out_tex, tex)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/media/mithun/New Volume/Research/Multilabel/analysis_out/latex_tables")
    ap.add_argument("--ambiguity_vs_perf_es", default="/media/mithun/New Volume/Research/Multilabel/analysis_out/ambiguity_vs_perf_es.csv")
    ap.add_argument("--label_uncertainty_ar", default="/media/mithun/New Volume/Research/Multilabel/analysis_out/label_uncertainty_ar.csv")
    ap.add_argument("--stability_ablation", default="/media/mithun/New Volume/Research/Multilabel/analysis_out/stability_ablation.csv")
    ap.add_argument("--errors_en", default="/media/mithun/New Volume/Research/Multilabel/analysis_out/errors_en.csv")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    build_ambiguity_vs_perf(
        args.ambiguity_vs_perf_es,
        os.path.join(args.out_dir, "tab_ambiguity_vs_perf_es.tex")
    )
    build_label_uncertainty(
        args.label_uncertainty_ar,
        os.path.join(args.out_dir, "tab_label_uncertainty_ar.tex")
    )
    build_stability_ablation(
        args.stability_ablation,
        os.path.join(args.out_dir, "tab_stability_ablation.tex")
    )
    build_error_types(
        args.errors_en,
        os.path.join(args.out_dir, "tab_error_types_en.tex")
    )

    print("\nDone. LaTeX tables written to:", args.out_dir)
    print(" - tab_ambiguity_vs_perf_es.tex")
    print(" - tab_label_uncertainty_ar.tex")
    print(" - tab_stability_ablation.tex")
    print(" - tab_error_types_en.tex (only if error_type is filled)")

if __name__ == "__main__":
    main()
