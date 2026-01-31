# -*- coding: utf-8 -*-
"""
Data loading + preprocessing for multilingual multi-label emotion classification.

Supports (all TSV .txt, tab-separated) with header:
  ID, Tweet, anger, anticipation, disgust, fear, joy, love,
  optimism, pessimism, sadness, surprise, trust

Languages supported:
- English (en)
- Spanish (es)
- Arabic (ar)

Outputs unified label space with per-sample masks (partial-label learning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Unified label space
# ----------------------------
UNIFIED_LABELS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust",
    "hate",  # kept for compatibility (unused in EN/ES/AR)
]
LABEL2IDX = {l: i for i, l in enumerate(UNIFIED_LABELS)}

# SemEval-ish 11 emotions (shared across EN/ES/AR in your dataset)
EN_LABELS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust"
]


# ----------------------------
# Helpers
# ----------------------------
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())
    return text


def load_multilingual_tsv(path: str, lang: str) -> pd.DataFrame:
    """
    Reads a tab-separated file with header:
      ID  Tweet  anger anticipation ... trust

    Args:
        path: file path
        lang: 'en' | 'es' | 'ar'

    Returns:
        pandas DataFrame with validated label columns.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)

    if "Tweet" not in df.columns:
        if "Text" in df.columns:
            df = df.rename(columns={"Text": "Tweet"})
        else:
            raise ValueError(f"[{lang}] Missing text column 'Tweet' (or 'Text') in {path}")

    # If ID missing, auto-create
    if "ID" not in df.columns:
        df["ID"] = [f"{lang}_{i}" for i in range(len(df))]

    # Validate + cast labels
    for c in EN_LABELS:
        if c not in df.columns:
            raise ValueError(f"[{lang}] Missing label column: {c} in {path}")
        df[c] = df[c].astype(int)

    df["lang"] = lang
    return df


def df_to_examples_generic(df: pd.DataFrame, lang: str) -> List[Dict]:
    """
    Convert a dataframe to examples with unified label vector + mask.

    Observed labels: the 11 EN_LABELS
    Missing labels: everything else in UNIFIED_LABELS (e.g., 'hate') will be mask=0
    """
    exs: List[Dict] = []

    for _, row in df.iterrows():
        raw = normalize_text(row["Tweet"])
        text = f"<{lang.upper()}> " + raw

        y = np.zeros(len(UNIFIED_LABELS), dtype=np.float32)
        m = np.zeros(len(UNIFIED_LABELS), dtype=np.float32)

        for lab in EN_LABELS:
            idx = LABEL2IDX[lab]
            y[idx] = float(row[lab])
            m[idx] = 1.0

        exs.append({
            "id": str(row["ID"]),
            "lang": lang,
            "text": text,
            "raw_text": raw,
            "y": y,
            "mask": m,
        })

    return exs


def build_examples(en_df: pd.DataFrame, es_df: pd.DataFrame, ar_df: pd.DataFrame) -> List[Dict]:
    """
    Merge all languages into one pool of examples.
    """
    return (
        df_to_examples_generic(en_df, "en")
        + df_to_examples_generic(es_df, "es")
        + df_to_examples_generic(ar_df, "ar")
    )


def filter_by_lang(examples: List[Dict], train_lang: str) -> List[Dict]:
    """
    train_lang: 'en' | 'es' | 'ar' | 'both'
    where 'both' means all languages.
    """
    if train_lang == "both":
        return examples
    return [e for e in examples if e["lang"] == train_lang]


# ----------------------------
# Torch dataset + collator
# ----------------------------
class EmotionDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    ids: List[str]
    langs: List[str]
    texts: List[str]
    raw_texts: List[str]


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Batch:
        texts = [b["text"] for b in batch]
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        y = torch.tensor(np.stack([b["y"] for b in batch]), dtype=torch.float32)
        m = torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32)
        ids = [b["id"] for b in batch]
        langs = [b["lang"] for b in batch]
        raw_texts = [b["raw_text"] for b in batch]

        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            y=y,
            mask=m,
            ids=ids,
            langs=langs,
            texts=texts,
            raw_texts=raw_texts
        )


def make_loaders(
    train_examples: List[Dict],
    val_examples: List[Dict],
    test_examples: List[Dict],
    tokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    collate = Collator(tokenizer, max_length=max_length)

    train_loader = DataLoader(
        EmotionDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        EmotionDataset(val_examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        EmotionDataset(test_examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
