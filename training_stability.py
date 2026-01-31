
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def read_history(run_dir):
    # Try common formats
    p = os.path.join(run_dir, "train_log.jsonl")
    if os.path.exists(p):
        df = read_jsonl(p)
        return df

    p = os.path.join(run_dir, "metrics_history.json")
    if os.path.exists(p):
        obj = json.load(open(p))
        return pd.DataFrame(obj)

    p = os.path.join(run_dir, "history.json")
    if os.path.exists(p):
        obj = json.load(open(p))
        return pd.DataFrame(obj)

    p = os.path.join(run_dir, "log.csv")
    if os.path.exists(p):
        return pd.read_csv(p)

    return None

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True,
                    help="Folder containing multiple seed runs, e.g., runs/stability/en_baseline_seed*/")
    ap.add_argument("--out_png", default="training_stability.png")
    ap.add_argument("--title", default="Training Stability (mean ± std)")
    args = ap.parse_args()

    run_dirs = [os.path.join(args.runs_root, d) for d in os.listdir(args.runs_root)]
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]

    histories = []
    for rd in sorted(run_dirs):
        df = read_history(rd)
        if df is None or len(df) == 0:
            continue

        step_col = pick_col(df, ["epoch", "Epoch", "step"])
        if step_col is None:
            continue

        train_loss_col = pick_col(df, ["train_loss", "loss", "train/loss"])
        val_f1_col = pick_col(df, ["val_miF1", "dev_miF1", "valid_miF1", "miF1_dev", "miF1"])
        val_loss_col = pick_col(df, ["val_loss", "dev_loss", "valid_loss"])

        keep = [step_col]
        if train_loss_col: keep.append(train_loss_col)
        if val_loss_col: keep.append(val_loss_col)
        if val_f1_col: keep.append(val_f1_col)

        sub = df[keep].copy()
        sub.columns = ["step"] + [c for c in ["train_loss","val_loss","val_miF1"] if c in sub.columns or True]
        # Fix column names robustly
        sub = df[keep].copy()
        sub = sub.rename(columns={
            step_col: "step",
            train_loss_col: "train_loss" if train_loss_col else train_loss_col,
            val_loss_col: "val_loss" if val_loss_col else val_loss_col,
            val_f1_col: "val_miF1" if val_f1_col else val_f1_col,
        })
        histories.append(sub)

    if len(histories) < 2:
        raise RuntimeError("Not enough readable run logs found. Make sure your runs save train_log.jsonl/history.json/log.csv.")

    # Align by step
    all_steps = sorted(set(np.concatenate([h["step"].values for h in histories])))
    def stack_metric(name):
        mats = []
        for h in histories:
            if name not in h.columns: 
                continue
            tmp = h.set_index("step")[name].reindex(all_steps).interpolate().ffill().bfill().values
            mats.append(tmp)
        if len(mats) == 0:
            return None
        return np.vstack(mats)

    m_train = stack_metric("train_loss")
    m_val_loss = stack_metric("val_loss")
    m_val_f1 = stack_metric("val_miF1")

    plt.figure(figsize=(10,4))
    ax = plt.gca()

    if m_train is not None:
        ax.plot(all_steps, m_train.mean(axis=0), label="train_loss")
        ax.fill_between(all_steps, m_train.mean(axis=0)-m_train.std(axis=0), m_train.mean(axis=0)+m_train.std(axis=0), alpha=0.2)

    if m_val_loss is not None:
        ax.plot(all_steps, m_val_loss.mean(axis=0), label="val_loss")
        ax.fill_between(all_steps, m_val_loss.mean(axis=0)-m_val_loss.std(axis=0), m_val_loss.mean(axis=0)+m_val_loss.std(axis=0), alpha=0.2)

    if m_val_f1 is not None:
        ax2 = ax.twinx()
        ax2.plot(all_steps, m_val_f1.mean(axis=0), label="val_miF1", linestyle="--")
        ax2.fill_between(all_steps, m_val_f1.mean(axis=0)-m_val_f1.std(axis=0), m_val_f1.mean(axis=0)+m_val_f1.std(axis=0), alpha=0.2)
        ax2.set_ylabel("miF1")

    ax.set_title(args.title)
    ax.set_xlabel("epoch/step")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.close()
    print(f"Saved: {args.out_png}")

if __name__ == "__main__":
    main()
