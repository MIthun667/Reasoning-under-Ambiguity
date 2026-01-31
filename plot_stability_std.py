#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--metric", default="miF1", choices=["HL","RL","miF1","maF1","AP"])
    ap.add_argument("--out_png", default="analysis_out/stability_std.png")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df["setting"] = df["train_lang"].str.upper() + " | " + df["uncertainty_mode"]
    y = df[f"{args.metric}_std"].fillna(0.0).values

    plt.figure(figsize=(10, 4))
    plt.bar(df["setting"].values, y)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(f"{args.metric} std (across seeds)")
    plt.title(f"Training stability: std of {args.metric} across seeds")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.close()

    print(f"Saved: {args.out_png}")

if __name__ == "__main__":
    main()
