#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

import data_io
import modeling


# ----------------------------
# Checkpoint
# ----------------------------
def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        thresholds = ckpt.get("thresholds", None)
    else:
        model.load_state_dict(ckpt, strict=True)
        thresholds = None
    model.eval()
    return thresholds


# ----------------------------
# Forward probs
# ----------------------------
@torch.no_grad()
def model_probs(model, tokenizer, text, device, max_length=192):
    enc = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return out["probs"][0].detach().cpu().numpy()


# ----------------------------
# Entropy + weight (ambiguity explanation)
# ----------------------------
def bernoulli_entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def ambiguity_entropy_and_weight(probs, tau):
    """
    Use the 11 shared labels (EN_LABELS) for ambiguity.
    Returns:
        H: scalar normalized entropy
        w: exp(-tau * H)
        per_label_H: dict(label -> entropy contribution)
    """
    idxs = [data_io.LABEL2IDX[l] for l in data_io.EN_LABELS]
    p = probs[idxs]
    H_vec = bernoulli_entropy(p)  # per-label
    H = float(np.mean(H_vec))
    w = float(np.exp(-tau * H))
    per_label = {data_io.EN_LABELS[i]: float(H_vec[i]) for i in range(len(idxs))}
    return H, w, per_label


# ----------------------------
# Gradient × Input attribution
# ----------------------------
def grad_x_input_attribution(model, tokenizer, text, label_idx, device, max_length=192):
    model.eval()
    enc = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    emb_layer = model.get_input_embeddings()
    embeds = emb_layer(input_ids).detach().clone().requires_grad_(True)

    out = model.forward_with_embeddings(
        input_ids=input_ids,
        attention_mask=attn,
        inputs_embeds=embeds
    )
    prob = out["probs"][0, label_idx]

    model.zero_grad(set_to_none=True)
    if embeds.grad is not None:
        embeds.grad.zero_()
    prob.backward()

    scores = (embeds.grad * embeds).sum(dim=-1).squeeze(0)  # [T]
    scores = scores.abs() * attn.squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    scores = scores.detach().cpu().numpy()
    scores = scores / (scores.max() + 1e-12)  # normalize [0,1]

    return tokens, scores, float(prob.detach().cpu().item())


def merge_sentencepiece(tokens, scores):
    """
    Merge XLM-R sentencepiece tokens:
      '▁word' starts new word.
    Word score = max of subpieces.
    """
    words, w_scores = [], []
    cur, cur_scores = "", []
    for t, s in zip(tokens, scores):
        if t in ["<s>", "</s>", "<pad>"]:
            continue
        if t.startswith("▁"):
            if cur:
                words.append(cur)
                w_scores.append(float(max(cur_scores) if cur_scores else 0.0))
            cur = t[1:]
            cur_scores = [float(s)]
        else:
            cur += t
            cur_scores.append(float(s))
    if cur:
        words.append(cur)
        w_scores.append(float(max(cur_scores) if cur_scores else 0.0))
    return words, np.array(w_scores, dtype=np.float32)


# ----------------------------
# HTML helpers
# ----------------------------
def html_escape(s):
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))

def render_heatmap(words, scores):
    # red highlight alpha = score
    spans = []
    for w, sc in zip(words, scores):
        a = float(np.clip(sc, 0.0, 1.0))
        spans.append(
            f'<span class="tok" style="background: rgba(255,0,0,{a:.3f});">{html_escape(w)}</span>'
        )
    return " ".join(spans)

def render_label_chips(pred_labels):
    # pred_labels: list of dict {label, prob, pred_bool}
    chips = []
    for d in pred_labels:
        tag = "pos" if d.get("pred", False) else "neg"
        chips.append(
            f'<span class="chip {tag}">{html_escape(d["label"])} '
            f'<span class="chipval">{d["prob"]:.3f}</span></span>'
        )
    return " ".join(chips)


def build_single_html_report(records, out_html, title):
    """
    records: list of dict with keys:
      id, text, probs, pred_labels, H, w, per_label_H, explain_heatmaps (list per label)
    """
    parts = []
    parts.append(f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>{html_escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 18px; }}
h2 {{ margin: 0 0 12px 0; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; margin: 14px 0; }}
.meta {{ color:#333; font-size: 13px; margin-bottom: 8px; }}
.text {{ font-size: 15px; margin: 10px 0; }}
.chip {{ display:inline-block; padding: 4px 8px; border-radius: 999px; margin: 2px 4px 2px 0; font-size: 12px; }}
.chip.pos {{ background: #e6f7e6; border: 1px solid #9dd49d; }}
.chip.neg {{ background: #f4f4f4; border: 1px solid #d0d0d0; }}
.chipval {{ opacity: 0.75; margin-left: 6px; }}
.tok {{ padding: 2px 4px; margin: 1px; border-radius: 4px; display:inline-block; }}
.small {{ font-size: 12px; color:#444; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
.hm {{ border: 1px dashed #ddd; border-radius: 10px; padding: 10px; }}
.hm_title {{ font-size: 13px; margin-bottom: 6px; color:#222; }}
kbd {{ background:#f2f2f2; padding:2px 6px; border-radius:6px; border:1px solid #ddd; }}
</style></head>
<body>
<h2>{html_escape(title)}</h2>
<div class="small">
This report shows <b>multi-label predictions</b> and <b>ambiguity explanations</b>.
Ambiguity is measured by normalized entropy <kbd>H</kbd> over the 11 shared labels and converted to a training influence weight <kbd>w = exp(-τH)</kbd>.
Token heatmaps are computed with <b>Gradient × Input</b> for each predicted label.
</div>
""")

    for r in records:
        meta = (f'ID={html_escape(r["id"])} | '
                f'Lang={html_escape(r["lang"].upper())} | '
                f'H={r["H"]:.4f} | w={r["w"]:.4f}')
        chips = render_label_chips(r["pred_labels"])

        # show top ambiguous labels (highest per-label entropy)
        perH = r["per_label_H"]
        top_amb = sorted(perH.items(), key=lambda x: x[1], reverse=True)[:5]
        top_amb_str = ", ".join([f"{k}:{v:.3f}" for k, v in top_amb])

        parts.append(f'<div class="card">')
        parts.append(f'<div class="meta">{meta}</div>')
        parts.append(f'<div class="text">{html_escape(r["raw_text"])}</div>')
        parts.append(f'<div class="meta"><b>Predicted labels</b>: {chips}</div>')
        parts.append(f'<div class="small"><b>Most ambiguous labels</b> (per-label entropy): {html_escape(top_amb_str)}</div>')

        # heatmaps per label
        parts.append('<div class="grid">')
        for hm in r["heatmaps"]:
            parts.append('<div class="hm">')
            parts.append(f'<div class="hm_title"><b>{html_escape(hm["label"])}</b> '
                         f'(prob={hm["prob"]:.3f})</div>')
            parts.append(hm["html"])
            parts.append('</div>')
        parts.append('</div>')

        parts.append('</div>')  # card

    parts.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ----------------------------
# Multi-label selection
# ----------------------------
def select_multilabels(probs, thresholds, mode, topn):
    """
    mode:
      - "threshold": labels where prob >= threshold (requires thresholds)
      - "topn": take topn among shared labels
      - "threshold_or_topn": threshold if any, else topn
    Returns list of (label_idx, label_name, prob, pred_bool)
    """
    shared = data_io.EN_LABELS  # 11 labels
    shared_idxs = [data_io.LABEL2IDX[l] for l in shared]

    cand = [(idx, data_io.UNIFIED_LABELS[idx], float(probs[idx])) for idx in shared_idxs]

    if mode == "topn":
        cand = sorted(cand, key=lambda x: x[2], reverse=True)[:topn]
        return [(i, n, p, True) for (i, n, p) in cand]

    if thresholds is None:
        if mode == "threshold":
            raise ValueError("threshold mode requires thresholds in checkpoint")
        # fallback
        cand = sorted(cand, key=lambda x: x[2], reverse=True)[:topn]
        return [(i, n, p, True) for (i, n, p) in cand]

    # threshold-based
    picked = []
    for i, n, p in cand:
        t = float(thresholds[i])
        pred = (p >= t)
        if pred:
            picked.append((i, n, p, pred))

    if mode == "threshold_or_topn" and len(picked) == 0:
        cand = sorted(cand, key=lambda x: x[2], reverse=True)[:topn]
        picked = [(i, n, p, True) for (i, n, p) in cand]

    return picked


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument("--uncertainty_mode", default="ambiguity_weight",
                    choices=["baseline", "ambiguity_weight", "evidential"])

    ap.add_argument("--test_path", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "es", "ar"])

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=192)

    ap.add_argument("--strategy", default="random", choices=["random", "top_uncertain"])
    ap.add_argument("--tau", type=float, default=2.0, help="same tau used in training for w=exp(-tau H)")

    ap.add_argument("--label_select", default="threshold_or_topn",
                    choices=["threshold", "topn", "threshold_or_topn"])
    ap.add_argument("--topn", type=int, default=3, help="used for topn selection or fallback")
    ap.add_argument("--out_html", default="interpret_report.html")

    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model = modeling.EmotionModel(
        backbone_name=args.backbone,
        num_labels=len(data_io.UNIFIED_LABELS),
        uncertainty_mode=args.uncertainty_mode,
        dropout=0.0,
    ).to(device)

    thresholds = load_checkpoint(model, args.ckpt, device)

    # load test
    df = data_io.load_multilingual_tsv(args.test_path, args.lang)
    if "Tweet" not in df.columns:
        raise ValueError("Expected column 'Tweet' in test file")
    if "ID" not in df.columns:
        df["ID"] = [f"{args.lang}_{i}" for i in range(len(df))]

    # select samples
    if args.strategy == "random":
        idxs = rng.choice(len(df), size=min(args.n, len(df)), replace=False).tolist()
    else:
        # top_uncertain by entropy over 11 labels
        ent = []
        for i in range(len(df)):
            raw = str(df.loc[i, "Tweet"])
            text = f"<{args.lang.upper()}> {raw}"
            probs = model_probs(model, tokenizer, text, device, max_length=args.max_length)
            H, _, _ = ambiguity_entropy_and_weight(probs, tau=args.tau)
            ent.append(H)
        idxs = np.argsort(-np.array(ent))[: min(args.n, len(df))].tolist()

    records = []
    for idx in idxs:
        ex_id = str(df.loc[idx, "ID"])
        raw = str(df.loc[idx, "Tweet"])
        text = f"<{args.lang.upper()}> {raw}"

        probs = model_probs(model, tokenizer, text, device, max_length=args.max_length)
        H, w, perH = ambiguity_entropy_and_weight(probs, tau=args.tau)

        picks = select_multilabels(probs, thresholds, mode=args.label_select, topn=args.topn)

        pred_labels = [{"label": n, "prob": p, "pred": bool(pred)} for (i, n, p, pred) in picks]

        heatmaps = []
        for (lab_idx, lab_name, lab_prob, _) in picks:
            toks, tok_scores, _ = grad_x_input_attribution(
                model, tokenizer, text, lab_idx, device, max_length=args.max_length
            )
            words, word_scores = merge_sentencepiece(toks, tok_scores)
            heatmaps.append({
                "label": lab_name,
                "prob": float(lab_prob),
                "html": render_heatmap(words, word_scores),
            })

        records.append({
            "id": ex_id,
            "lang": args.lang,
            "raw_text": raw,
            "H": float(H),
            "w": float(w),
            "per_label_H": perH,
            "pred_labels": pred_labels,
            "heatmaps": heatmaps,
        })

    title = f"Interpretability Report ({args.lang.upper()}) | multi-label + ambiguity"
    build_single_html_report(records, args.out_html, title)
    print(f"\nSaved ONE HTML report: {args.out_html}")
    print("Tip: open it in your browser and screenshot a few cards for the paper.")


if __name__ == "__main__":
    main()




"""
python interpret.py \
  --ckpt runs/en_es_ar_ambiguity/model.pt \
  --uncertainty_mode ambiguity_weight \
  --lang en \
  --test_path "/media/mithun/New Volume/Research/Multilabel/English/test.txt" \
  --n 12 \
  --label_select threshold_or_topn \
  --topn 3 \
  --out_html interpret_en.html

python interpret.py \
  --ckpt runs/en_es_ar_ambiguity/model.pt \
  --uncertainty_mode ambiguity_weight \
  --lang ar \
  --test_path "/media/mithun/New Volume/Research/Multilabel/Arabic/E-c/test.txt" \
  --n 12 \
  --strategy top_uncertain \
  --topn 3 \
  --out_html interpret_ar_uncertain.html

python interpret.py \
  --ckpt runs/en_es_ar_ambiguity/model.pt \
  --uncertainty_mode ambiguity_weight \
  --lang es \
  --test_path "/media/mithun/New Volume/Research/Multilabel/Spanish/Spanish-E-c/test.txt" \
  --n 12 \
  --strategy top_uncertain \
  --topn 3 \
  --out_html interpret_es_uncertain.html

"""