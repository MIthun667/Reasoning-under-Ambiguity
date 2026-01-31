#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import re
import ast


def norm(s: str) -> str:
    return s.strip().lower()


def find_label_columns(df: pd.DataFrame, labels: list[str]):
    """
    Tries to find per-label columns, allowing:
      joy / Joy / label_joy / y_joy / joy_label
    Returns mapping: label -> column name (if found).
    """
    cols = list(df.columns)
    cols_norm = {norm(c): c for c in cols}

    mapping = {}
    for lab in labels:
        labn = norm(lab)

        candidates = [
            labn,
            f"label_{labn}",
            f"y_{labn}",
            f"{labn}_label",
            f"{labn}_y",
        ]

        found = None
        for cand in candidates:
            if cand in cols_norm:
                found = cols_norm[cand]
                break

        mapping[labn] = found

    found_cols = [mapping[norm(l)] for l in labels if mapping[norm(l)] is not None]
    return mapping, found_cols


def parse_label_list_cell(x):
    """
    Parse a cell containing labels. Supports:
      - list literal: "['joy','anger']"
      - comma-separated: "joy, anger"
      - pipe-separated: "joy|anger"
      - empty/NaN
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    # Try python literal list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [norm(str(t)) for t in v]
        except Exception:
            pass

    # Split by common separators
    parts = re.split(r"[,\|;]\s*", s)
    return [norm(p) for p in parts if p.strip()]


def compute_stats_from_label_cols(df: pd.DataFrame, label_cols: list[str]):
    y = df[label_cols]

    observed = ~y.isna()
    n = len(df)
    k = len(label_cols)

    # tolerate '0'/'1' strings
    y_num = y.apply(pd.to_numeric, errors="coerce")

    pos = (y_num == 1).sum().sum()
    neg = (y_num == 0).sum().sum()
    miss = y_num.isna().sum().sum()

    avg_pos = (y_num == 1).sum(axis=1).mean()
    avg_obs = (~y_num.isna()).sum(axis=1).mean()
    obs_rate = float(avg_obs / k) if k > 0 else 0.0

    return {
        "N": int(n),
        "K": int(k),
        "Pos": int(pos),
        "Neg": int(neg),
        "Miss": int(miss),
        "AvgPos": float(avg_pos),
        "AvgObs": float(avg_obs),
        "ObsRate": float(obs_rate),
    }


def compute_stats_from_labels_column(df: pd.DataFrame, labels: list[str], labels_col: str):
    label_set = {norm(x) for x in labels}
    n = len(df)
    k = len(labels)

    # build binary matrix only for positives; missing supervision not representable here
    pos_counts = {lab: 0 for lab in label_set}
    pos_per_inst = []

    for x in df[labels_col]:
        labs = set(parse_label_list_cell(x))
        labs = labs & label_set
        for lab in labs:
            pos_counts[lab] += 1
        pos_per_inst.append(len(labs))

    # In this format we cannot know "missing labels" vs "true negatives"
    # so we report Miss as 0 and Obs% as 100, but we add a warning in stdout.
    pos_total = sum(pos_counts.values())
    avg_pos = float(pd.Series(pos_per_inst).mean()) if n > 0 else 0.0

    return {
        "N": int(n),
        "K": int(k),
        "Pos": int(pos_total),
        "Neg": 0,
        "Miss": 0,
        "AvgPos": float(avg_pos),
        "AvgObs": float(k),
        "ObsRate": 1.0,
        "_pos_counts": pos_counts,
        "_warning": f"Used '{labels_col}' column. Missing labels cannot be estimated from this format."
    }


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
    )


def write_latex(rows, caption: str, label: str, out_tex: Path):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{\\small {caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\begin{tabular}{l r r r r r r}")
    lines.append("\\toprule")
    lines.append("Lang & $N$ & $K$ & Obs.\\% & Avg. pos. & Avg. obs. & Missing \\\\")
    lines.append("\\midrule")

    for r in rows:
        lang = latex_escape(r["lang"])
        st = r["stats"]
        lines.append(
            f"{lang} & {st['N']} & {st['K']} & {100.0*st['ObsRate']:.1f} & "
            f"{st['AvgPos']:.2f} & {st['AvgObs']:.2f} & {st['Miss']} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--langs", type=str, required=True, help="Comma-separated language ids (en,es,ar,...)")
    ap.add_argument("--pattern", type=str, required=True, help="Filename pattern, e.g., dataset_stats_labels_{lang}.csv")
    ap.add_argument("--labels", type=str, required=True, help="Comma-separated unified label names")
    ap.add_argument("--labels_col", type=str, default="labels", help="Name of column that stores label lists, if used")
    ap.add_argument("--caption", type=str, default="Dataset statistics under partial supervision.")
    ap.add_argument("--label", type=str, default="tab:dataset_stats")
    ap.add_argument("--out_tex", type=str, default="dataset_stats_table.tex")
    args = ap.parse_args()

    root = Path(args.root)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    rows = []
    for lang in langs:
        p = root / args.pattern.format(lang=lang)
        if not p.exists():
            print(f"[skip] missing file: {p}")
            continue

        df = pd.read_csv(p)

        mapping, found_cols = find_label_columns(df, labels)

        if len(found_cols) == len(labels):
            st = compute_stats_from_label_cols(df, found_cols)
        elif args.labels_col in df.columns:
            st = compute_stats_from_labels_column(df, labels, args.labels_col)
            print(f"[warn] {lang}: {st['_warning']}")
        else:
            raise SystemExit(
                f"{p} does not contain the requested label columns and also has no '{args.labels_col}' column.\n"
                f"Columns found: {list(df.columns)}\n"
                f"Try adjusting label names or using --labels_col."
            )

        rows.append({"lang": lang.upper(), "stats": st})

    if not rows:
        raise SystemExit("No files processed. Check --root/--langs/--pattern.")

    out_tex = Path(args.out_tex)
    write_latex(rows, args.caption, args.label, out_tex)
    print(f"Wrote LaTeX table to: {out_tex.resolve()}")


if __name__ == "__main__":
    main()



"""
python dataset_stats_to_latex.py \
  --root "/media/mithun/New Volume/Research/Multilabel/analysis_out/dataset_stats" \
  --langs en,es,ar \
  --pattern "dataset_stats_labels_{lang}.csv" \
  --labels joy,anger,sadness,fear,disgust,trust,anticipation,optimism,surprise,pessimism,love \
  --out_tex dataset_stats_table.tex

"""