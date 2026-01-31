# -*- coding: utf-8 -*-
"""
Internal ablation runner.

This module:
- Defines ablation grids (paper-ready defaults)
- Runs multiple experiments internally (no bash needed)
- Aggregates metrics across seeds
- Writes ablation_summary.csv and ablation_summary.json

It expects train.py to expose:
    run_experiment(cfg: dict) -> dict
where the returned dict includes metrics like:
    HL, RL, miF1, maF1, AP, microAP
"""

from __future__ import annotations

import os
import json
import copy
import csv
import itertools
from typing import Dict, List, Any

import numpy as np


# ----------------------------
# Grid definitions
# ----------------------------
def build_default_grid(args) -> List[Dict[str, Any]]:
    """
    Core paper grid (recommended, not insane):
      - train_lang: en, es, ar, both (en+es+ar)
      - uncertainty_mode: baseline, ambiguity_weight, evidential
      - seed: 13, 42, 2026

    Total runs = 4 × 3 × 3 = 36
    """
    grid = {
        "train_lang": ["en", "es", "ar", "both"],
        "uncertainty_mode": ["baseline", "ambiguity_weight", "evidential"],
        "seed": [13, 42, 2026],
    }
    return expand_grid(args, grid)


def build_extended_grid(args) -> List[Dict[str, Any]]:
    """
    Extended grid (stronger ablation, more runs):
      - adds pos_weight and PU loss
    """
    grid = {
        "train_lang": ["en", "es", "ar", "both"],
        "uncertainty_mode": ["baseline", "ambiguity_weight", "evidential"],
        "use_pos_weight": [False, True],
        "pu_weight": [0.0, 0.05],
        "seed": [13, 42, 2026],
    }
    return expand_grid(args, grid)


def expand_grid(args, grid_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Turn a grid dict into a list of full experiment configs,
    starting from CLI args as the base.
    """
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())

    runs = []
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(vars(args))
        for k, v in zip(keys, combo):
            cfg[k] = v
        runs.append(cfg)
    return runs


# ----------------------------
# Aggregation helpers
# ----------------------------
def summarize_runs(results: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, Any]:
    """
    Compute mean ± std for selected metrics.
    """
    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in results if r.get(key) is not None]
        if len(vals) == 0:
            summary[key] = None
        else:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }
    return summary


def group_by_config(results: List[Dict[str, Any]], group_keys: List[str]) -> Dict[str, List[Dict]]:
    """
    Group run results by configuration keys (excluding seed).
    """
    groups = {}
    for r in results:
        key = tuple((k, r[k]) for k in group_keys)
        groups.setdefault(key, []).append(r)
    return groups


# ----------------------------
# Main ablation runner
# ----------------------------
def run_ablation_grid(
    args,
    run_experiment_fn,
):
    """
    Entry point used by train.py when --ablation_grid is enabled.

    Args:
        args: argparse.Namespace
        run_experiment_fn: function(cfg_dict) -> metrics dict
    """
    if args.grid_name == "extended":
        grid = build_extended_grid(args)
    else:
        grid = build_default_grid(args)

    if args.grid_max_runs is not None:
        grid = grid[: args.grid_max_runs]

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    print(f"[Ablation] Running {len(grid)} experiments")

    for i, cfg in enumerate(grid):
        run_id = f"exp_{i:04d}"
        run_dir = os.path.join(out_root, run_id)
        cfg["out_dir"] = run_dir
        os.makedirs(run_dir, exist_ok=True)

        print(
            f"\n[Ablation] {run_id} | "
            f"lang={cfg['train_lang']} "
            f"uncertainty={cfg['uncertainty_mode']} "
            f"seed={cfg['seed']}"
        )

        # Run single experiment
        metrics_out = run_experiment_fn(cfg)

        # Record configuration + results
        record = {
            "run_id": run_id,
            "train_lang": cfg["train_lang"],
            "uncertainty_mode": cfg["uncertainty_mode"],
            "seed": cfg["seed"],
            "HL": metrics_out.get("HL"),
            "RL": metrics_out.get("RL"),
            "miF1": metrics_out.get("miF1"),
            "maF1": metrics_out.get("maF1"),
            "AP": metrics_out.get("AP"),
            "microAP": metrics_out.get("microAP"),
        }
        all_results.append(record)

        with open(os.path.join(run_dir, "result_summary.json"), "w") as f:
            json.dump(record, f, indent=2)

    # ----------------------------
    # Aggregate (mean ± std)
    # ----------------------------
    group_keys = ["train_lang", "uncertainty_mode"]
    groups = group_by_config(all_results, group_keys)

    summary_rows = []
    summary_json = {}

    metric_keys = ["HL", "RL", "miF1", "maF1", "AP"]

    for gkey, runs in groups.items():
        cfg_dict = dict(gkey)
        summary = summarize_runs(runs, metric_keys)

        row = {
            "train_lang": cfg_dict["train_lang"],
            "uncertainty_mode": cfg_dict["uncertainty_mode"],
        }
        for m in metric_keys:
            if summary[m] is None:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
            else:
                row[f"{m}_mean"] = summary[m]["mean"]
                row[f"{m}_std"] = summary[m]["std"]

        summary_rows.append(row)
        summary_json[f"{cfg_dict['train_lang']}_{cfg_dict['uncertainty_mode']}"] = summary

    # Write CSV
    csv_path = os.path.join(out_root, "ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # Write JSON
    json_path = os.path.join(out_root, "ablation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n[Ablation] Finished")
    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")
