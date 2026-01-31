import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# ==============================
# Paths (EDIT ONLY THIS PART)
# ==============================
BASE = "/media/mithun/New Volume/Research/Multilabel/runs"

# Update these to your actual run folders (single best runs you want in Table 1)
SINGLE_RUNS = {
    "EN + Ambiguity": f"{BASE}/en/metrics_test.json",
    "ES + Ambiguity": f"{BASE}/es/metrics_test.json",
    "AR + Ambiguity": f"{BASE}/ar/metrics_test.json",
    "BOTH + Ambiguity": f"{BASE}/en_es_ar_evidential/metrics_test.json",
    # optional if you ran PU explicitly as a separate single run
    "BOTH + Ambiguity + PU": f"{BASE}/en_es_ar_evidential/metrics_test.json",
}

# Your updated ablation output dir (the one you used with --ablation_grid)
ABLATION_DIR = f"{BASE}/ablation_en_es_ar"

METRICS = ["HL", "RL", "miF1", "maF1", "AP"]

# ==============================
# Helpers
# ==============================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def fmt(mean, std):
    return f"{mean:.4f} $\\pm$ {std:.4f}"

def safe_mean_std(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) == 0:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))

def mode_pretty(mode: str) -> str:
    # nicer naming in table
    mapping = {
        "baseline": "Baseline",
        "ambiguity_weight": "Ambiguity",
        "evidential": "Evidential",
    }
    return mapping.get(mode, mode)

def lang_pretty(lang: str) -> str:
    mapping = {
        "en": "EN",
        "es": "ES",
        "ar": "AR",
        "both": "BOTH",
    }
    return mapping.get(lang, lang.upper())

# ==============================
# Table 1: Main Results
# ==============================
rows = []
for name, path in SINGLE_RUNS.items():
    if not os.path.exists(path):
        print(f"[WARN] Missing single-run metrics: {path}")
        continue

    m = load_json(path)
    rows.append({
        "Model": name,
        "HL ↓": m.get("HL"),
        "RL ↓": m.get("RL"),
        "miF1 ↑": m.get("miF1"),
        "maF1 ↑": m.get("maF1"),
        "AP ↑": m.get("AP"),
    })

df_main = pd.DataFrame(rows)
df_main.to_csv("table_main_results.csv", index=False)

print("\n=== Main Results (CSV saved) ===")
print(df_main)

# ==============================
# Table 2: Ablation (mean ± std)
# Preferred source: ablation_summary.csv
# Fallback: exp_*/result_summary.json
# ==============================
summary_csv = os.path.join(ABLATION_DIR, "ablation_summary.csv")

rows = []

if os.path.exists(summary_csv):
    # ---- Preferred: read summary CSV produced by ablation.py
    df = pd.read_csv(summary_csv)

    # expecting columns like:
    # train_lang, uncertainty_mode, HL_mean, HL_std, ...
    for _, r in df.iterrows():
        row = {
            "Train": lang_pretty(str(r["train_lang"])),
            "Mode": mode_pretty(str(r["uncertainty_mode"])),
        }
        for m in METRICS:
            mean_key = f"{m}_mean"
            std_key  = f"{m}_std"
            mean = float(r[mean_key]) if mean_key in df.columns and pd.notna(r[mean_key]) else None
            std  = float(r[std_key])  if std_key  in df.columns and pd.notna(r[std_key])  else None
            row[m] = fmt(mean, std) if (mean is not None and std is not None) else "-"
        rows.append(row)

else:
    # ---- Fallback: parse exp_ folders
    groups = defaultdict(list)

    if not os.path.isdir(ABLATION_DIR):
        raise FileNotFoundError(f"Ablation dir not found: {ABLATION_DIR}")

    for exp in sorted(os.listdir(ABLATION_DIR)):
        exp_dir = os.path.join(ABLATION_DIR, exp)
        if not exp.startswith("exp_"):
            continue

        path = os.path.join(exp_dir, "result_summary.json")
        if not os.path.exists(path):
            continue

        r = load_json(path)
        key = (r.get("train_lang"), r.get("uncertainty_mode"))
        groups[key].append(r)

    for (lang, mode), runs in groups.items():
        row = {
            "Train": lang_pretty(str(lang)),
            "Mode": mode_pretty(str(mode)),
        }
        for m in METRICS:
            vals = [rr.get(m) for rr in runs]
            mean, std = safe_mean_std(vals)
            row[m] = fmt(mean, std) if mean is not None else "-"
        rows.append(row)

# sort rows nicely: EN/ES/AR/BOTH then Baseline/Ambiguity/Evidential
lang_order = {"EN": 0, "ES": 1, "AR": 2, "BOTH": 3}
mode_order = {"Baseline": 0, "Ambiguity": 1, "Evidential": 2}

df_ablation = pd.DataFrame(rows)
df_ablation["__lang_ord"] = df_ablation["Train"].map(lambda x: lang_order.get(x, 99))
df_ablation["__mode_ord"] = df_ablation["Mode"].map(lambda x: mode_order.get(x, 99))
df_ablation = df_ablation.sort_values(["__lang_ord", "__mode_ord"]).drop(columns=["__lang_ord", "__mode_ord"])

df_ablation.to_csv("table_ablation_results.csv", index=False)

print("\n=== Ablation Results (CSV saved) ===")
print(df_ablation)

# ==============================
# LaTeX Tables
# ==============================
def to_latex(df, caption, label):
    return df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * (len(df.columns) - 1)
    )

latex_main = to_latex(
    df_main,
    "Performance comparison of proposed models under different training settings.",
    "tab:main_results"
)

latex_ablation = to_latex(
    df_ablation,
    "Ablation study (mean $\\pm$ std over three seeds).",
    "tab:ablation"
)

with open("table_main_results.tex", "w") as f:
    f.write(latex_main)

with open("table_ablation_results.tex", "w") as f:
    f.write(latex_ablation)

print("\nLaTeX tables written:")
print(" - table_main_results.tex")
print(" - table_ablation_results.tex")
