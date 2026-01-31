#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

METRICS = ["HL", "RL", "miF1", "maF1", "AP"]

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_dir", required=True,
                    help="e.g., /media/mithun/New Volume/Research/Multilabel/runs/ablation_en_es_ar")
    ap.add_argument("--out_csv", default="analysis_out/stability_ablation.csv")
    ap.add_argument("--use", default="result_summary",
                    choices=["result_summary", "metrics_test"],
                    help="Prefer result_summary.json. If missing, can try metrics_test.json.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    groups = defaultdict(list)

    exp_dirs = sorted([d for d in os.listdir(args.ablation_dir) if d.startswith("exp_")])
    if not exp_dirs:
        raise RuntimeError(f"No exp_* folders found in: {args.ablation_dir}")

    n_loaded = 0
    for exp in exp_dirs:
        exp_dir = os.path.join(args.ablation_dir, exp)

        if args.use == "result_summary":
            p = os.path.join(exp_dir, "result_summary.json")
            if not os.path.exists(p):
                continue
            r = load_json(p)
        else:
            p = os.path.join(exp_dir, "metrics_test.json")
            if not os.path.exists(p):
                continue
            m = load_json(p)
            # metrics_test.json usually doesn't store lang/mode; try infer from folder name if needed
            # so result_summary is strongly recommended
            r = m

        # required keys
        if "train_lang" not in r or "uncertainty_mode" not in r:
            # skip if missing metadata
            continue

        lang = str(r["train_lang"]).lower()
        mode = str(r["uncertainty_mode"]).lower()

        item = {k: safe_float(r.get(k)) for k in METRICS}
        item["_exp"] = exp
        groups[(lang, mode)].append(item)
        n_loaded += 1

    if n_loaded == 0:
        raise RuntimeError("No usable result_summary.json found (missing train_lang/uncertainty_mode?).")

    rows = []
    for (lang, mode), runs in sorted(groups.items()):
        row = {"train_lang": lang, "uncertainty_mode": mode, "n_runs": len(runs)}
        for k in METRICS:
            vals = [x[k] for x in runs if x[k] is not None]
            if len(vals) == 0:
                row[f"{k}_mean"] = None
                row[f"{k}_std"] = None
            else:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print(df)

if __name__ == "__main__":
    main()
