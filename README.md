# Cross-Platform Fairness Evaluation of NLP Models for Mental Health Detection

<div align="center">

**A Peer-Reviewed Research Codebase — JBI Submission**

*Rajveer Singh Pall · Sameer Yadav*
*Gyan Ganga Institute of Technology and Sciences, Jabalpur, India*

---

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)
[![Paper: JBI](https://img.shields.io/badge/Journal-JBI%20%28under%20review%29-6366F1)](https://www.sciencedirect.com/journal/journal-of-biomedical-informatics)

</div>

---

## Abstract

Transformer-based language models trained on curated social media corpora routinely achieve near-perfect within-distribution accuracy for mental health text classification, yet their cross-platform generalization and algorithmic fairness remain poorly characterized. This work introduces the **Cross-Platform Fairness Evaluation (CPFE) framework**, a systematic audit pipeline applied to four pre-trained language models — BERT, RoBERTa, DistilRoBERTa, and SamLowe-RoBERTa — trained on a Kaggle mental health corpus and evaluated on unseen Reddit (GoEmotions) and Twitter (dair-ai/emotion) data. We observe a consistent **29–40% macro-AUC degradation** across all models and both target platforms, with calibration error rising from ≈ 0.058 (within-platform) to ≈ 0.520 (Twitter). Fairness audits using symmetric Disparate Impact (DI) and Equalized Odds Difference (EOD) reveal systematic bias in minority class detection (anxiety, stress). Gradient-based token saliency and cross-platform Jaccard stability analysis confirm that models anchor on platform-specific linguistic artifacts rather than clinical signal. Sensitivity analyses across three independent label-mapping schemas confirm the robustness of the degradation pattern.

---

## Table of Contents

1. [Research Problem and Contributions](#1-research-problem-and-contributions)
2. [Repository Structure](#2-repository-structure)
3. [Data and Label Schema](#3-data-and-label-schema)
4. [Models](#4-models)
5. [Key Results](#5-key-results)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Setup and Installation](#7-setup-and-installation)
8. [Reproducing the Full Pipeline](#8-reproducing-the-full-pipeline)
9. [Extended Analyses](#9-extended-analyses-appendix-scripts)
10. [Output Artifacts](#10-output-artifacts)
11. [Design Principles](#11-design-principles)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. Research Problem and Contributions

### The Problem

Mental health NLP models are trained and benchmarked on single-platform corpora. When deployed across different social media platforms — Reddit, Twitter, clinical forums — they encounter vocabulary shifts, demographic differences, and labelling inconsistencies that cause silent performance collapse. These failures are especially dangerous in clinical-adjacent applications where a misclassification carries real cost.

### Contributions

| # | Contribution |
|---|---|
| 1 | **CPFE Framework** — end-to-end pipeline for cross-platform fairness auditing of mental health classifiers, covering data harmonization, training, evaluation, statistical auditing, feature attribution, and sensitivity analysis |
| 2 | **Empirical evidence of cross-platform collapse** — all four models drop 29–40% macro-AUC on unseen platforms; calibration degrades catastrophically (ECE → 0.52 on Twitter) |
| 3 | **Clinical-grade statistical methodology** — per-class AUC with DeLong 95% CIs, Bonferroni-corrected multi-test comparisons, symmetric Disparate Impact, Equalized Odds Difference |
| 4 | **Feature introspection** — gradient-based token saliency maps (Eqs. 8–9) and cross-platform Jaccard feature stability scores reveal platform-specific lexical anchoring |
| 5 | **Label-mapping robustness** — stable degradation pattern confirmed across 4-class, binary, and 3-class schemas, ruling out artefact of label design |
| 6 | **Fully reproducible** — config-driven pipeline (`configs/config.yaml`), no hardcoded paths, no magic numbers, seed-controlled randomness |

---

## 2. Repository Structure

```
mental-health-fairness-nlp/
│
├── configs/
│   └── config.yaml                        # Single source of truth: all paths, hyperparameters,
│                                          # fairness thresholds, seeds
│
├── data/
│   ├── raw/
│   │   ├── kaggle_mental_health/
│   │   │   └── Combined Data.csv          # 53 K samples, 7 original labels
│   │   ├── reddit_goemotions/             # HuggingFace cache (28-class GoEmotions)
│   │   └── twitter_emotion/              # HuggingFace cache (dair-ai/emotion, 6-class)
│   └── splits/
│       └── cross_platform/
│           ├── train.csv                  # Kaggle train split (stratified)
│           ├── val.csv                    # Kaggle validation split
│           ├── test_kaggle.csv            # Within-distribution test
│           ├── test_reddit.csv            # Cross-platform test 1
│           └── test_twitter.csv           # Cross-platform test 2
│
├── src/
│   ├── utils.py                           # Canonical constants + 8 shared utility functions
│   ├── preprocess.py                      # Data loading, text cleaning, label remapping, splits
│   ├── train.py                           # Fine-tuning loop with early stopping + checkpointing
│   ├── evaluate.py                        # Cross-platform inference → per-sample prediction CSVs
│   ├── fairness_audit.py                  # DeLong AUC, ECE, Bonferroni, DI, EOD, calibration curves
│   ├── shap_analysis.py                   # Gradient saliency maps + Jaccard cross-platform stability
│   ├── sensitivity_analysis.py            # Label-mapping robustness (4-class / binary / 3-class)
│   ├── gpt_eval.py                        # GPT-4 evaluation stub (future work)
│   │
│   │   ── Appendix / Extended analyses ──────────────────────────────────────────────────────
│   ├── code_A1_di_eod_analysis.py         # Symmetric DI and EOD heatmaps (Eq. 6)
│   ├── code_A2_stress_attribution.py      # Stress-class gradient attribution across platforms
│   ├── code_A3_A4_A6_ece_jaccard.py       # ECE bootstrap CIs + binning sensitivity + Jaccard
│   ├── code_A4_patch_attribution.py       # Per-class patch-level token attribution
│   ├── code_A5_temperature_scaling.py     # Post-hoc temperature calibration
│   ├── fix_di_symmetric.py                # [DEPRECATED] DI formula fix — backported into A1
│   ├── jaccard_full_analysis.py           # Full cross-platform Jaccard stability (RoBERTa)
│   ├── label_sensitivity_mappings_DE.py   # Label sensitivity mappings D and E
│   ├── perclass_ece_analysis.py           # Per-class calibration error breakdown
│   └── truncation_audit.py               # Token-length truncation vs. prediction error analysis
│
├── outputs/
│   ├── models/                            # Fine-tuned checkpoints (bert/, roberta/, ...)
│   ├── results/
│   │   ├── master_results.csv             # Per-model × per-platform aggregate metrics
│   │   ├── {model}_eval.json              # Per-model evaluation details
│   │   ├── fairness/                      # Fairness audit tables and bootstrap CIs
│   │   └── sensitivity/                   # Label-mapping robustness results
│   └── figures/                           # All paper figures (PNG, 300 dpi)
│       ├── figure1_forest_plot.png
│       ├── figure2_platform_degradation.png
│       ├── figure3_calibration_curves.png
│       ├── figure4_f1_heatmap.png
│       ├── figure5_*_shap_cross_platform.png
│       ├── figure6_feature_stability.png
│       ├── figure7_sensitivity_analysis.png
│       └── figure8_sensitivity_heatmap.png
│
├── requirements.txt
└── README.md                              # This file
```

---

## 3. Data and Label Schema

### 3.1 Source Datasets

| Platform | Dataset | Split Role | Samples | Original Labels |
|---|---|---|---|---|
| Kaggle | [Mental Health Corpus](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) | Train + within-test | ~53 K | 7 (normal, depression, suicidal, anxiety, stress, bipolar, personality disorder) |
| Reddit | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Cross-platform test 1 | ~54 K | 28 fine-grained emotions |
| Twitter | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | Cross-platform test 2 | ~20 K | 6 basic emotions |

### 3.2 Unified 4-Class Schema

All three datasets are harmonized into a single label space:

| ID | Class | Clinical Interpretation | Kaggle Source Labels |
|---|---|---|---|
| 0 | `normal` | No clinical signal | normal |
| 1 | `depression` | Depressive disorders | depression, suicidal, bipolar |
| 2 | `anxiety` | Anxiety-spectrum | anxiety |
| 3 | `stress` | Stress-related | stress, personality disorder |

> **Class imbalance**: depression ≈ 56.6%, normal ≈ 29%, anxiety ≈ 7%, stress ≈ 7%. Imbalance is not artificially corrected — it reflects the real-world distribution of the training corpus.

Remapping logic lives entirely in `src/preprocess.py` (`KAGGLE_TO_UNIFIED`, `GOEMO_TO_UNIFIED`, `DAIREMO_TO_UNIFIED`). It is not duplicated elsewhere.

### 3.3 Sensitivity Label Schemas (Appendix)

| Schema | Description | Script |
|---|---|---|
| A (primary) | 4-class as above | `sensitivity_analysis.py` |
| B | Binary: normal vs. any mental health | `sensitivity_analysis.py` |
| C | 3-class: normal / depression / distress (anxiety+stress) | `sensitivity_analysis.py` |
| D | Alternative 4-class remapping | `label_sensitivity_mappings_DE.py` |
| E | Alternative 4-class remapping | `label_sensitivity_mappings_DE.py` |

---

## 4. Models

| Key | HuggingFace Model ID | Paper Alias | Pre-training Domain |
|---|---|---|---|
| `bert` | `bert-base-uncased` | BERT | General English (BookCorpus + Wikipedia) |
| `roberta` | `roberta-base` | RoBERTa | General English (160 GB corpus) |
| `mentalbert` | `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa | Emotion corpora (distilled) |
| `mentalroberta` | `SamLowe/roberta-base-go_emotions` | SamLowe-RoBERTa | GoEmotions (27-class fine-tuned) |

All models are fine-tuned with identical hyperparameters for a controlled comparison:

| Hyperparameter | Value |
|---|---|
| `max_length` | 64 tokens |
| `batch_size` | 16 |
| `learning_rate` | 2 × 10⁻⁵ |
| `epochs` | 5 |
| `seed` | 42 |
| Optimizer | AdamW |

---

## 5. Key Results

### 5.1 Macro-AUC (with 95% DeLong CI)

| Model | Kaggle (within-dist.) | Reddit (cross-platform) | Twitter (cross-platform) | Drop: Kaggle → Twitter |
|---|---|---|---|---|
| BERT | 0.983 | 0.699 | 0.605 | **−37.7%** |
| RoBERTa | 0.987 | 0.689 | 0.601 | **−39.1%** |
| DistilRoBERTa | 0.985 | 0.637 | 0.598 | **−39.3%** |
| SamLowe-RoBERTa | 0.984 | 0.629 | 0.596 | **−39.4%** |

> Clinical threshold: cross-platform AUC drop > 20% is concerning; > 35% is unacceptable for deployment in clinical-adjacent applications.

### 5.2 Expected Calibration Error (ECE ↓ lower is better)

| Model | Kaggle | Reddit | Twitter |
|---|---|---|---|
| BERT | 0.056 | 0.218 | 0.499 |
| RoBERTa | 0.058 | 0.231 | 0.521 |
| DistilRoBERTa | 0.060 | 0.244 | 0.537 |
| SamLowe-RoBERTa | 0.059 | 0.238 | 0.542 |

> ECE = 0.0 is perfect calibration; ECE ≈ 0.5 is equivalent to random confidence assignment.

### 5.3 Fairness Metrics (Symmetric Disparate Impact)

- Symmetric DI < 0.80 for minority classes (anxiety, stress) across all models and platforms — violates the 80% rule.
- EOD > 0.15 for stress class on Twitter for all models.
- Post-hoc temperature scaling reduces ECE by ≈ 40% on Reddit but does not close the cross-platform AUC gap.

### 5.4 Feature Stability (Jaccard)

| Comparison | Jaccard (top-20 tokens) |
|---|---|
| Within-platform (Kaggle train → test) | 0.62–0.71 |
| Cross-platform (Kaggle → Reddit) | 0.18–0.26 |
| Cross-platform (Kaggle → Twitter) | **0.08–0.14** |

Near-zero cross-platform token overlap confirms that models anchor on platform-specific vocabulary and slang, not on clinically meaningful terms.

---

## 6. Evaluation Metrics

| Metric | Symbol | Reference | Interpretation |
|---|---|---|---|
| Macro-AUC | AUC | Eq. 1 | Discriminative power across all classes; 1.0 = perfect, 0.5 = random |
| DeLong Confidence Interval | CI | Eq. 2 | Non-parametric 95% CI around AUC for statistical significance testing |
| Expected Calibration Error | ECE | Eq. 5 | Probability reliability; 0.0 = perfectly calibrated, 0.5 = random |
| Disparate Impact (symmetric) | DI | Eq. 6 | `min(r, 1/r)` bounded in (0, 1]; < 0.80 indicates disparate treatment |
| Equalized Odds Difference | EOD | Eq. 7 | True positive rate parity across demographic proxy groups |
| Gradient Token Importance | — | Eqs. 8–9 | L2-norm of `∂ŷ/∂emb`, normalized per text; identifies salient tokens |
| Jaccard Feature Stability | J | — | `\|A ∩ B\| / \|A ∪ B\|` for top-K token sets compared across platforms |

> **Correctness note on DI**: The common asymmetric formula `rate_tgt / rate_ref` can exceed 1.0 and is direction-dependent, making it uninterpretable as a parity measure. This codebase exclusively uses the symmetric form `min(r, 1/r)` which constrains DI to (0, 1] regardless of group assignment order.

---

## 7. Setup and Installation

### 7.1 Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU recommended (CPU is very slow for attribution scripts)
- ~10 GB disk space for model checkpoints

### 7.2 Clone and Install

```bash
git clone https://github.com/Rajveer-code/mental-health-fairness-nlp.git
cd mental-health-fairness-nlp

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# OR
.venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 7.3 Data Preparation

Place the Kaggle dataset at:

```
data/raw/kaggle_mental_health/Combined Data.csv
```

The Reddit (GoEmotions) and Twitter (dair-ai/emotion) datasets are downloaded automatically from HuggingFace Hub on the first run of `preprocess.py`.

---

## 8. Reproducing the Full Pipeline

Run steps **in order**. Each step reads from the output of the previous one.

```bash
# Step 1 — Preprocessing
# Cleans text, remaps labels across all three corpora, creates stratified splits
python src/preprocess.py

# Step 2 — Training  (≈2–4 hours per model on a single GPU)
# Fine-tunes all 4 models; saves best checkpoints to outputs/models/
python src/train.py

# Step 3 — Cross-platform evaluation
# Runs inference on Kaggle / Reddit / Twitter test sets for all models
# Writes per-sample prediction CSVs to outputs/results/
python src/evaluate.py

# Step 4 — Fairness audit  (produces Figures 1–4)
# DeLong AUC with CIs, ECE calibration curves, Bonferroni tests, DI, EOD
python src/fairness_audit.py

# Step 5 — Feature attribution  (produces Figures 5–6)
# Gradient-based token saliency maps + cross-platform Jaccard stability
python src/shap_analysis.py

# Step 6 — Label-mapping sensitivity  (produces Figures 7–8)
# Validates degradation pattern across 4-class / binary / 3-class label schemas
python src/sensitivity_analysis.py
```

All file paths are resolved from `configs/config.yaml`. No script contains hardcoded paths or magic numbers.

---

## 9. Extended Analyses (Appendix Scripts)

These scripts reproduce appendix results and supplementary figures. They are independent of each other and can be run after Step 3.

| Script | Output | Appendix Section |
|---|---|---|
| `src/code_A1_di_eod_analysis.py` | Symmetric DI and EOD heatmaps across all models × platforms × classes | A.1 |
| `src/code_A2_stress_attribution.py` | Stress-class gradient attribution comparison (all 4 models) | A.2 |
| `src/code_A3_A4_A6_ece_jaccard.py` | ECE bootstrap CIs, binning sensitivity, extended Jaccard | A.3, A.4, A.6 |
| `src/code_A4_patch_attribution.py` | Per-class patch-level token attribution (gradient × embedding) | A.4 |
| `src/code_A5_temperature_scaling.py` | Post-hoc temperature scaling; before/after ECE comparison | A.5 |
| `src/perclass_ece_analysis.py` | Per-class calibration breakdown (normal / depression / anxiety / stress) | A.7 |
| `src/truncation_audit.py` | AUC vs. token-length quartile; truncation error correlations | A.8 |
| `src/jaccard_full_analysis.py` | Full pairwise Jaccard stability analysis (RoBERTa, all platform pairs) | A.6 |
| `src/label_sensitivity_mappings_DE.py` | Alternative label remapping schemas D and E | A.9 |

---

## 10. Output Artifacts

### Figures (`outputs/figures/`)

| File | Paper Reference | Description |
|---|---|---|
| `figure1_forest_plot.png` | Figure 1 | Per-class AUC forest plot with 95% DeLong CIs |
| `figure2_platform_degradation.png` | Figure 2 | Cross-platform macro-AUC degradation (all 4 models) |
| `figure3_calibration_curves.png` | Figure 3 | Reliability diagrams: within vs. cross-platform calibration |
| `figure4_f1_heatmap.png` | Figure 4 | Per-class F1 heatmap: model × platform × class |
| `figure5_*_shap_cross_platform.png` | Figure 5 | Per-model gradient saliency word clouds (cross-platform) |
| `figure5_stress_combined_comparison.png` | Figure 5 | Stress-class attribution comparison across all 4 models |
| `figure6_feature_stability.png` | Figure 6 | Cross-platform Jaccard feature stability |
| `figure7_sensitivity_analysis.png` | Figure 7 | Label-mapping robustness: AUC across 3 schemas |
| `figure8_sensitivity_heatmap.png` | Figure 8 | Sensitivity heatmap: model × platform × schema |
| `figure_di_heatmap_symmetric.png` | Appendix A.1 | Symmetric Disparate Impact heatmap |
| `figure_eod_heatmap.png` | Appendix A.1 | Equalized Odds Difference heatmap |
| `figure_ece_bootstrap_cis.png` | Appendix A.3 | ECE with 95% bootstrap CI error bars |
| `figure_recalibration.png` | Appendix A.5 | Temperature scaling before/after calibration curves |
| `figure_perclass_ece_heatmap.png` | Appendix A.7 | Per-class ECE breakdown heatmap |
| `figure_clinical_vocabulary_heatmap.png` | Appendix A.8 | Clinical term retention rate across platforms |

### Tables (`outputs/results/`)

| File | Description |
|---|---|
| `master_results.csv` | Primary results: model × platform × all metrics |
| `fairness/fairness_audit_full.csv` | Complete fairness audit with DeLong CIs |
| `fairness/di_eod_table_symmetric.csv` | Symmetric DI and EOD per class × platform |
| `fairness/ece_bootstrap_cis.csv` | ECE with 95% bootstrap CIs |
| `fairness/temperature_scaling_results.csv` | Pre/post temperature scaling ECE |
| `fairness/perclass_ece.csv` | Per-class calibration error breakdown |
| `fairness/jaccard_full_analysis.csv` | Pairwise platform Jaccard scores per model |
| `sensitivity/sensitivity_full_results.csv` | Mapping robustness: AUC per schema × model |
| `sensitivity/sensitivity_drops_all_mappings.csv` | AUC drop per model per label schema |

---

## 11. Design Principles

This codebase is held to the standards of a Q1 biomedical informatics submission:

| Principle | Implementation |
|---|---|
| **Single source of truth** | All constants (`MODELS`, `PLATFORMS`, `CLASSES`, `PROB_COLS`, etc.) defined once in `src/utils.py`; no script redefines them |
| **Config-driven** | All file paths and hyperparameters live in `configs/config.yaml`; no script has hardcoded paths |
| **Reproducibility** | Fixed `seed = 42` throughout; deterministic stratified splits; version-pinned `requirements.txt` |
| **Statistical rigour** | DeLong CIs (non-parametric), Bonferroni correction, bootstrap CIs for ECE — no uncorrected p-values |
| **Correct fairness math** | Symmetric DI `min(r, 1/r)` — the asymmetric `rate_tgt / rate_ref` is mathematically incorrect and explicitly rejected |
| **No deprecated APIs** | No `pd.DataFrame.append()`, no deprecated sklearn patterns; all functions have NumPy-style docstrings |
| **PEP 8 compliance** | Enforced throughout; 88-character line limit (Black-compatible) |

---

## 12. Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{pall2025cpfe,
  title   = {Cross-Platform Fairness Evaluation of Transformer-Based
             Mental Health Classifiers: A Multi-Model Audit},
  author  = {Pall, Rajveer Singh and Yadav, Sameer},
  journal = {Journal of Biomedical Informatics},
  year    = {2025},
  note    = {Under review},
  url     = {https://github.com/Rajveer-code/mental-health-fairness-nlp}
}
```

---

## 13. License

This project is released under the [MIT License](LICENSE).

The datasets used are subject to their own licenses:

| Dataset | License |
|---|---|
| Kaggle Mental Health Corpus | Public domain / CC0 |
| GoEmotions (Google Research) | [Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE) |
| dair-ai/emotion | [MIT](https://huggingface.co/datasets/dair-ai/emotion) |

---

<div align="center">
<sub>Submitted to <em>Journal of Biomedical Informatics</em> (JBI) · 2025</sub>
</div>
