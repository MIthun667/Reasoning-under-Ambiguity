# Reasoning under Ambiguity  
### Uncertainty-Aware Multilingual Emotion Classification under Partial Supervision

This repository contains the official implementation of **Reasoning under Ambiguity**, a framework for multilingual multi-label emotion classification that explicitly models annotation ambiguity under partial supervision.  
The method addresses the common but often overlooked assumption that missing emotion labels indicate negative evidence, which is not valid for real-world emotion datasets.

---

## 📌 Key Idea

Emotion annotations are inherently **ambiguous**, **overlapping**, and often **incomplete**.  
Rather than treating unannotated labels as negatives, this work:

- Quantifies **annotation ambiguity** using entropy over observed labels  
- Applies **instance-level ambiguity weighting** during training  
- Supports **partial supervision** via masked losses  
- Achieves **stable and robust learning** across languages  

---

## ✨ Contributions

- **Ambiguity-aware learning objective** using entropy-based instance weighting  
- **Masked multi-label loss** that avoids penalizing missing annotations  
- Optional **positive–unlabeled (PU) regularization**  
- Extensive evaluation on **SemEval-2018 Task 1 (E-c)**  
- Analyses on stability, label-wise uncertainty, ambiguity stratification, and interpretability  

---

## 🌍 Datasets

We evaluate on **SemEval-2018 Task 1: Affect in Tweets (Emotion Classification)**:

| Language | Split | Instances |
|--------|------|-----------|
| English | train / dev / test | ✓ |
| Spanish | train / dev / test | ✓ |
| Arabic | train / dev / test | ✓ |

Each instance may contain **multiple emotion labels**, drawn from a shared inventory of 11 emotions.

> Missing labels indicate *unknown*, not *negative* supervision.

---

## 🧠 Method Overview

For each training instance:

1. Encode text using a multilingual transformer (e.g., XLM-R)
2. Predict label probabilities
3. Compute entropy over **observed labels only**
4. Convert entropy into an **ambiguity weight**
5. Optimize a **masked, ambiguity-weighted BCE loss**
6. Optionally apply PU regularization to unobserved labels

See Algorithm 1 in the paper for full details.

## 📂 Repository Structure
```
├── train.py # Main training script
├── multilingual_emotion_uncertainty_train.py
├── modeling.py # Model definitions
├── losses.py # Ambiguity-weighted + PU losses
├── metrics.py # Evaluation metrics
├── data_io.py # Dataset loading and masking
│
├── ablation.py # Ablation experiments
├── training_stability.py # Stability across seeds
├── eval_jaccard.py # Jaccard similarity evaluation
│
├── analysis_out/ # Analysis artifacts
│ ├── dataset_stats/
│ ├── label_uncertainty_.csv
│ ├── ambiguity_vs_perf_.csv
│ ├── nn_examples*.tex
│
├── runs/ # Saved checkpoints and metrics
├── README.md
```
## 🚀 Training

### Example: English with ambiguity weighting

```
python train.py \
  --lang en \
  --uncertainty_mode ambiguity_weight \
  --encoder xlm-roberta-base \
  --epochs 10 \
  --batch_size 16
```
Multilingual training
```
python train.py \
  --lang both \
  --uncertainty_mode ambiguity_weight \
  --encoder xlm-roberta-base
```
📊 Evaluation
Standard metrics
  1. Hamming Loss (HL)
  2. Ranking Loss (RL)
  3. Micro-F1
  4. Macro-F1
  5. Average Precision (AP)
  6. Jaccard Similarity
```
python eval_jaccard.py \
  --ckpt runs/en_es_ar_ambiguity/model.pt \
  --test_path Spanish/Spanish-E-c/test.txt \
  --lang es \
  --uncertainty_mode ambiguity_weight
```
🔍 Analysis and Interpretability
The repository includes scripts to reproduce:
  1. Ablation studies (baseline vs ambiguity vs evidential)
  2. Training stability across seeds
  3. Label-wise uncertainty analysis
  4. Ambiguity-stratified performance
  5. Nearest-neighbor explanations in embedding space
  6. Generated tables are exported directly to LaTeX for paper use.

📈 Main Findings
  a. Ambiguity-weighted learning consistently improves macro-F1 and AP
  b. Training is more stable compared to evidential uncertainty methods
  c. Performance degrades smoothly with increasing ambiguity
  d. Learned embeddings support interpretable similarity-based explanations

📄 Citation
If you use this work, please cite:
```
@inproceedings{mohammad2018semeval,
  title     = {SemEval-2018 Task 1: Affect in Tweets},
  author    = {Mohammad, Saif M. and Bravo-Marquez, Felipe and Salameh, Mohammad and Kiritchenko, Svetlana},
  booktitle = {Proceedings of the 12th International Workshop on Semantic Evaluation},
  year      = {2018}
}
```
⚠️ Notes
  This code assumes partial supervision by design
  Missing labels are never treated as negatives
  Results may differ if labels are artificially completed

📬 Contact
Md. Mithun Hossain
Research Assistant, BUBT Research Graduate School
📧 mhosen751@gmail.com

⭐ Acknowledgements
This work builds on prior research in multi-label learning, emotion analysis, and uncertainty-aware modeling. We thank the SemEval organizers for providing high-quality multilingual benchmarks.
