#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

METRICS = ["HL", "RL", "miF1", "maF1", "AP"]

def fmt(m, s):
    if pd.isna(m) or pd.isna(s):
        return "--"
    return f"{m:.4f} $\\pm$ {s:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_tex", default="analysis_out/stability_ablation.tex")
    ap.add_argument("--caption", default="Training stability across seeds (mean $\\pm$ std).")
    ap.add_argument("--label", default="tab:stability")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    out_rows = []
    for _, r in df.iterrows():
        row = {
            "Train": str(r["train_lang"]).upper(),
            "Mode": str(r["uncertainty_mode"]),
        }
        for k in METRICS:
            row[k] = fmt(r[f"{k}_mean"], r[f"{k}_std"])
        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    latex = out.to_latex(
        index=False,
        escape=False,
        caption=args.caption,
        label=args.label,
        column_format="l" + "c" * (len(out.columns) - 1)
    )

    with open(args.out_tex, "w") as f:
        f.write(latex)

    print(f"Saved: {args.out_tex}")

if __name__ == "__main__":
    main()
