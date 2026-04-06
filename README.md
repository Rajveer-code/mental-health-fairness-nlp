# Cross-Platform Generalization Failure in Mental Health NLP: A Systematic Fairness Audit of Transformer Models on Social Media

<div align="center">

**Cross-Platform Fairness Evaluation (CPFE) Framework**

*Rajveer Singh Pall · Sameer Yadav*
*Gyan Ganga Institute of Technology and Sciences, Jabalpur, India*

*Submitted to the Journal of Biomedical Informatics (JBI)*

---

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

</div>

---

## Abstract

Mental health classification models trained on labelled social media corpora are increasingly deployed across platforms without systematic evaluation of cross-platform generalization, calibration, or fairness. This repository implements the **Cross-Platform Fairness Evaluation (CPFE)** framework and applies it to four transformer architectures — BERT, RoBERTa, Emotion-DistilRoBERTa, and GoEmotions-RoBERTa — trained on a Kaggle mental health corpus (n = 35,556) and evaluated on Reddit (n = 6,257) and Twitter (n = 2,883) test sets using a unified four-class clinical label schema.

**Key finding:** All four models exhibit macro AUC degradation of 28.6–39.5% under platform shift. Expected Calibration Error rises from 0.056–0.060 in-domain to 0.499–0.542 on Twitter. Gradient-based attribution overlap collapses to near zero (Jaccard J ≈ 0 in 14/16 model–class pairs). Disparate impact falls below 0.17 for all clinical classes on Reddit, far below the four-fifths rule threshold. These failures are robust across five label-mapping sensitivity schemes and all 12 pairwise model comparisons under Bonferroni-corrected bootstrap Z-tests.

---

## Table of Contents

1. [Scientific Contributions](#1-scientific-contributions)
2. [Repository Structure](#2-repository-structure)
3. [Datasets and Label Schema](#3-datasets-and-label-schema)
4. [Models](#4-models)
5. [Key Results](#5-key-results)
6. [Installation](#6-installation)
7. [Reproducible Pipeline](#7-reproducible-pipeline)
8. [Output Reference](#8-output-reference)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [CPFE Thresholds](#10-cpfe-thresholds)
11. [Notes for Reviewers](#11-notes-for-reviewers)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. Scientific Contributions

This work makes the following principal contributions:

1. **CPFE framework** — The first systematic audit combining discriminative performance, calibration, statistical significance, platform-group fairness metrics, and attribution stability in a unified five-axis evaluation protocol for mental health NLP.

2. **Empirical characterisation of cross-platform failure** — All four architecturally diverse transformer models exhibit macro AUC degradation of 28.6–39.5% and minimal gradient-based attribution overlap (Jaccard J ≈ 0 in 14/16 model–class pairs under gradient saliency) when transferred across social media platforms without domain adaptation.

3. **Calibration vs. discriminative performance separation** — Platform-specific temperature scaling recovers calibration (88.0% mean ECE reduction, |ΔAUC| < 0.01) but cannot restore discriminative performance, demonstrating that calibration alone is insufficient as a deployment remedy.

4. **Clinical-grade fairness measurement** — Severe disparate impact violations (DI < 0.17) for all clinical classes on Reddit, with equalized odds differences exceeding 0.75 for anxiety and stress, are confirmed robust to prior-shift-adjusted DI calculations.

5. **Construct validity** — Sensitivity analysis across five label-mapping schemes (A–E, spanning 4-class, binary, 3-class, and distress-superclass formulations) confirms that cross-platform degradation is not an artefact of mapping subjectivity.

6. **Provisional CPFE deployment thresholds** — Data-driven deployment readiness criteria (ΔAUC < 15% acceptable; > 30% severe) grounded in observed degradation patterns, proposed as starting points for community standardisation pending external validation.

---

## 2. Repository Structure

```
mental-health-fairness-nlp/
│
├── configs/
│   └── config.yaml                    ← single source of truth for all paths and hyperparameters
│
├── data/
│   ├── raw/
│   │   ├── kaggle_mental_health/
│   │   │   └── Combined Data.csv      ← 53,470 samples, 7 original labels
│   │   ├── reddit_goemotions/         ← HuggingFace GoEmotions cache (58,009 comments)
│   │   └── twitter_emotion/           ← HuggingFace dair-ai/emotion cache (20,000 tweets)
│   └── splits/
│       └── cross_platform/
│           ├── train.csv              ← Kaggle training set (n = 35,556)
│           ├── val.csv                ← Kaggle validation set (n = 7,620)
│           ├── test_kaggle.csv        ← Within-platform test (n = 7,620)
│           ├── test_reddit.csv        ← Cross-platform test 1 (n = 6,257)
│           └── test_twitter.csv       ← Cross-platform test 2 (n = 2,883)
│
├── src/
│   ├── utils.py                       ← CANONICAL: all shared constants, loaders, metrics
│   ├── preprocess.py                  ← Data loading, cleaning, label mapping, splitting
│   ├── train.py                       ← Fine-tuning loop; supports --seeds for multi-seed runs
│   ├── evaluate.py                    ← Cross-platform inference, prediction CSVs, master results
│   ├── fairness_audit.py              ← AUC/ECE/DI/EOD audit; between-model significance tests
│   ├── shap_analysis.py               ← Gradient saliency attribution; cross-platform comparison
│   ├── sensitivity_analysis.py        ← Label-mapping robustness (A–E); integrates D/E mappings
│   ├── truncation_audit.py            ← Token-length diagnostics; exact tokenizer truncation rates
│   ├── jaccard_full_analysis.py       ← Feature stability; within-platform split-half baseline
│   ├── perclass_ece_analysis.py       ← Per-class ECE (one-vs-rest, Eq. 5b)
│   ├── code_A1_di_eod_analysis.py     ← DI/EOD heatmaps; prior-shift-adjusted DI
│   ├── code_A2_stress_attribution.py  ← Stress-class gradient attribution figures
│   ├── code_A3_A4_A6_ece_jaccard.py   ← ECE bootstrap CIs; Jaccard K-sensitivity; ECE bin-sensitivity
│   ├── code_A4_patch_attribution.py   ← Patch-level (phrase) gradient attribution
│   ├── code_A5_temperature_scaling.py ← Temperature scaling recalibration (bounds 0.1–20.0)
│   ├── fix_di_symmetric.py            ← (deprecated patch) symmetric DI formula; backported to A1
│   ├── gpt_eval.py                    ← Stub for GPT-4 qualitative label evaluation
│   └── label_sensitivity_mappings_DE.py ← Mappings D (distress superclass) and E (alternative)
│
├── outputs/
│   ├── models/
│   │   ├── bert/                      ← Fine-tuned BERT checkpoint
│   │   ├── roberta/                   ← Fine-tuned RoBERTa checkpoint
│   │   ├── mentalbert/                ← Fine-tuned Emotion-DistilRoBERTa checkpoint
│   │   └── mentalroberta/             ← Fine-tuned GoEmotions-RoBERTa checkpoint
│   ├── results/
│   │   ├── master_results.csv
│   │   ├── {model}_eval.json          ← 4 files
│   │   ├── {model}_{platform}_predictions.csv  ← 12 files (columns: label, pred, prob_*, correct)
│   │   ├── fairness/
│   │   │   ├── fairness_audit_full.csv
│   │   │   ├── pairwise_auc_comparisons.csv
│   │   │   ├── between_model_auc_reddit.csv
│   │   │   ├── between_model_auc_twitter.csv
│   │   │   ├── di_eod_table.csv
│   │   │   ├── perclass_ece.csv
│   │   │   ├── jaccard_full_analysis.csv
│   │   │   ├── temperature_scaling_results.csv
│   │   │   ├── clinical_signal_retention.csv
│   │   │   ├── truncation_exact_rates.csv
│   │   │   ├── ece_bootstrap_cis.csv
│   │   │   ├── ece_binning_sensitivity.csv
│   │   │   └── jaccard_k_sensitivity.csv
│   │   └── sensitivity/
│   │       ├── sensitivity_full_results.csv
│   │       └── sensitivity_drops.csv
│   ├── figures/                       ← 28+ publication-quality PNG figures
│   └── CPFE_Manuscript_Revised.docx   ← Revised manuscript (post peer-review toning)
│
├── requirements.txt
├── README.md                          ← this file
└── LICENSE
```

---

## 3. Datasets and Label Schema

### 3.1 Source Datasets

| Platform | Dataset | Source | Split Used | n (after mapping) |
|----------|---------|--------|------------|-------------------|
| Kaggle | Mental Health Corpus | [Kaggle](https://www.kaggle.com/) | Train + within-test | 35,556 train / 7,620 test |
| Reddit | GoEmotions | [Demszky et al., 2020](https://github.com/google-research/google-research/tree/master/goemotions) | Test only | 6,257 |
| Twitter | dair-ai/emotion | [HuggingFace](https://huggingface.co/datasets/dair-ai/emotion) | Test only | 2,883 |

> **Contamination note:** GoEmotions-RoBERTa and Emotion-DistilRoBERTa have pretraining exposure to Reddit/Twitter data; their cross-platform figures represent conservative upper bounds rather than true out-of-domain benchmarks. See Limitations in the manuscript.

### 3.2 Unified Four-Class Schema

| Class ID | Label | Clinical meaning |
|----------|-------|-----------------|
| 0 | normal | No clinical signal |
| 1 | depression | Primary affective distress class |
| 2 | anxiety | Anxiety-related distress |
| 3 | stress | Stress/anger-related distress |

### 3.3 Platform-Specific Label Mappings

**Kaggle → unified** (`KAGGLE_TO_UNIFIED` in `preprocess.py`):

| Original label | Unified | Notes |
|----------------|---------|-------|
| Normal | normal | |
| Depression | depression | |
| Suicidal | depression | Approximation: suicidal ideation often co-occurs with depressive episodes, though clinically distinct under DSM-5 |
| Anxiety | anxiety | |
| Bipolar | depression | Mapped to affective distress category |
| Stress | stress | |
| Personality disorder | stress | Approximation |

**Reddit (GoEmotions) → unified** (multi-label, ambiguous examples removed):

| GoEmotions labels | Unified |
|-------------------|---------|
| sadness, grief, remorse, disappointment | depression |
| nervousness, fear, anxiety | anxiety |
| anger, annoyance, frustration | stress |
| all remaining (27 categories) | normal |

**Twitter (dair-ai/emotion) → unified**:

| Twitter labels | Unified |
|----------------|---------|
| sadness | depression |
| fear | anxiety |
| anger | stress |
| joy, surprise, love | normal |

### 3.4 Label Sensitivity Mappings (A–E)

Five mapping schemes are tested to confirm robustness to construct validity concerns:

| Scheme | Description |
|--------|-------------|
| A | Primary 4-class schema (above) |
| B | Binary: normal vs. any mental health signal |
| C | 3-class: normal / depression / distress (anxiety + stress merged) |
| D | Distress superclass: collapses anxiety + stress into a single distress category, removing the most clinically ambiguous distinction |
| E | Alternative conservative mapping: narrower clinical vocabulary, stricter normal criteria |

---

## 4. Models

| Key | HuggingFace ID | Display Name | Parameters | Pre-training Domain |
|-----|----------------|--------------|------------|---------------------|
| `bert` | `bert-base-uncased` | BERT | 110 M | General English |
| `roberta` | `roberta-base` | RoBERTa | 125 M | General English |
| `mentalbert` | `j-hartmann/emotion-english-distilroberta-base` | Emotion-DistilRoBERTa | 82 M | Emotion data (inc. Twitter) |
| `mentalroberta` | `SamLowe/roberta-base-go_emotions` | GoEmotions-RoBERTa | 125 M | GoEmotions Reddit |

**Training hyperparameters** (from `configs/config.yaml`):

| Hyperparameter | Value |
|----------------|-------|
| `max_length` | 64 tokens |
| `batch_size` | 16 |
| `learning_rate` | 2 × 10⁻⁵ |
| `epochs` | 5 |
| `seed` | 42 (default) |
| `optimizer` | AdamW |
| `loss` | Cross-entropy |

---

## 5. Key Results

### 5.1 Discriminative Performance (Macro AUC)

| Model | Kaggle (in-domain) | Reddit (cross-platform) | Twitter (cross-platform) | Max ΔAUC |
|-------|--------------------|------------------------|--------------------------|----------|
| BERT | 0.983 | 0.699 | 0.605 | 37.8% |
| RoBERTa | 0.985 | 0.629 | 0.596 | 38.9% |
| Emotion-DistilRoBERTa | 0.986 | 0.693 | 0.596 | 39.0% |
| GoEmotions-RoBERTa | 0.987 | 0.699 | 0.596 | 39.1% |

### 5.2 Calibration (Expected Calibration Error, M = 10)

| Model | Kaggle | Reddit | Twitter |
|-------|--------|--------|---------|
| BERT | 0.059 | 0.271 | 0.542 |
| RoBERTa | 0.060 | 0.264 | 0.499 |
| Emotion-DistilRoBERTa | 0.056 | 0.275 | 0.536 |
| GoEmotions-RoBERTa | 0.057 | 0.268 | 0.531 |

After platform-specific temperature scaling: mean ECE reduction of **88.0%**, |ΔAUC| < 0.01.

### 5.3 Fairness (Symmetric Disparate Impact, Reddit)

DI < 0.17 for all clinical classes across all four models. Maximum EOD for depression: 0.830 (RoBERTa). These violations persist after prior-shift adjustment.

### 5.4 Attribution Stability (Jaccard Similarity, K = 10)

Jaccard J ≈ 0 in **14/16** model–class pairs on Kaggle-to-Twitter comparisons under gradient saliency. Within-platform split-half baseline: J ≈ 0.30–0.50.

### 5.5 Statistical Significance

All 12 pairwise model AUC comparisons (Reddit, Twitter) reach statistical significance under Bonferroni-corrected bootstrap Z-tests (α/6 ≈ 0.0083).

---

## 6. Installation

### Requirements

- Python 3.10 or later
- CUDA-capable GPU recommended for training (CPU inference supported)
- ~10 GB disk space for model checkpoints

### Setup

```bash
git clone https://github.com/Rajveer-code/mental-health-fairness-nlp.git
cd mental-health-fairness-nlp

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Data Download

Kaggle data must be placed manually at `data/raw/kaggle_mental_health/Combined Data.csv` (available at the Kaggle dataset link in the manuscript references).

Reddit and Twitter datasets are downloaded automatically by `preprocess.py` via HuggingFace Datasets on first run:
```bash
python src/preprocess.py   # downloads GoEmotions and dair-ai/emotion on first run
```

---

## 7. Reproducible Pipeline

All scripts are run from the repository root. All paths and hyperparameters are controlled by `configs/config.yaml`.

### 7.1 Stage 1 — Data Preparation

```bash
python src/preprocess.py
```
- Downloads and caches Reddit/Twitter datasets via HuggingFace
- Applies platform-specific label mappings and `clean_text()` preprocessing
- Writes `data/splits/cross_platform/{train,val,test_kaggle,test_reddit,test_twitter}.csv`

### 7.2 Stage 2 — Model Training

```bash
# Train all four models (default seed = 42)
python src/train.py --model all

# Optional: multi-seed training for variance estimation
python src/train.py --model all --seeds 42 43 44
```
- Saves best checkpoints (by validation AUC) to `outputs/models/{model}/`
- Writes `{model}_training_history.csv` with epoch-level metrics

### 7.3 Stage 3 — Cross-Platform Evaluation

```bash
python src/evaluate.py
```
- Runs inference on all three test sets for all four models
- Outputs `outputs/results/{model}_{platform}_predictions.csv`
  (columns: `label, pred, prob_normal, prob_depression, prob_anxiety, prob_stress, correct`)
- Writes `outputs/results/master_results.csv` (summary AUC / F1 / ECE)

### 7.4 Stage 4 — Core Fairness Audit

```bash
python src/fairness_audit.py
```
- Macro AUC with 95% bootstrap confidence intervals (B = 2000)
- Expected Calibration Error (10 equal-width bins)
- Bonferroni-corrected pairwise AUC comparisons
- Between-model AUC significance tests on Reddit and Twitter
- Outputs to `outputs/results/fairness/`

### 7.5 Stage 5 — Extended Analyses

Run these in any order after Stage 4:

```bash
# Disparate Impact and Equalized Odds Difference (with prior-shift-adjusted DI)
python src/code_A1_di_eod_analysis.py

# Gradient saliency attribution and cross-platform feature comparison
python src/shap_analysis.py

# Feature stability analysis (Jaccard with within-platform split-half baseline)
python src/jaccard_full_analysis.py

# Label-mapping robustness (Mappings A–E, integrates D/E automatically)
python src/sensitivity_analysis.py

# Token-length truncation analysis (exact tokenizer-based)
python src/truncation_audit.py

# Per-class calibration breakdown (one-vs-rest ECE)
python src/perclass_ece_analysis.py

# Temperature scaling recalibration
python src/code_A5_temperature_scaling.py

# Stress-class attribution figures
python src/code_A2_stress_attribution.py

# ECE bootstrap CIs, Jaccard K-sensitivity, ECE binning sensitivity
python src/code_A3_A4_A6_ece_jaccard.py
```

### 7.6 Complete One-Shot Command Sequence

```bash
python src/preprocess.py && \
python src/train.py --model all && \
python src/evaluate.py && \
python src/fairness_audit.py && \
python src/code_A1_di_eod_analysis.py && \
python src/shap_analysis.py && \
python src/jaccard_full_analysis.py && \
python src/sensitivity_analysis.py && \
python src/truncation_audit.py && \
python src/perclass_ece_analysis.py && \
python src/code_A5_temperature_scaling.py && \
python src/code_A2_stress_attribution.py && \
python src/code_A3_A4_A6_ece_jaccard.py
```

---

## 8. Output Reference

### 8.1 Primary Results

| File | Description |
|------|-------------|
| `outputs/results/master_results.csv` | AUC, F1, ECE for all model × platform combinations |
| `outputs/results/{model}_eval.json` | Per-model evaluation summary |
| `outputs/results/{model}_{platform}_predictions.csv` | Per-sample predictions (12 files) |

### 8.2 Fairness Audit Outputs

| File | Description |
|------|-------------|
| `fairness/fairness_audit_full.csv` | AUC CIs, ECE, all models × platforms |
| `fairness/pairwise_auc_comparisons.csv` | Within-model cross-platform Z-tests |
| `fairness/between_model_auc_reddit.csv` | Between-model AUC tests on Reddit (6 pairs) |
| `fairness/between_model_auc_twitter.csv` | Between-model AUC tests on Twitter (6 pairs) |
| `fairness/di_eod_table.csv` | Symmetric DI and EOD, raw + prior-shift-adjusted |
| `fairness/perclass_ece.csv` | Per-class calibration (one-vs-rest) |
| `fairness/jaccard_full_analysis.csv` | Jaccard J for all model–class–platform pairs + within-platform baseline |
| `fairness/temperature_scaling_results.csv` | Optimal temperature, pre/post ECE |
| `fairness/truncation_exact_rates.csv` | Exact tokenizer-based truncation rates |
| `fairness/clinical_signal_retention.csv` | Clinical vocabulary retention in top-10 attribution tokens |
| `fairness/ece_bootstrap_cis.csv` | ECE bootstrap 95% CIs |
| `fairness/ece_binning_sensitivity.csv` | ECE across M = 5, 10, 15, 20 bins |
| `fairness/jaccard_k_sensitivity.csv` | Jaccard across K = 5, 10, 15, 20 |

### 8.3 Sensitivity Analysis Outputs

| File | Description |
|------|-------------|
| `sensitivity/sensitivity_full_results.csv` | AUC × mapping scheme (A–E) × model × platform |
| `sensitivity/sensitivity_drops.csv` | Cross-platform ΔAUC per mapping scheme |

### 8.4 Figures

All figures are saved to `outputs/figures/` at ≥ 150 DPI.

| Figure | Description |
|--------|-------------|
| `figure1_forest_plot.png` | Per-class AUC forest plot with 95% CIs |
| `figure2_platform_degradation.png` | AUC degradation across platforms |
| `figure3_calibration_curves.png` | Reliability diagrams before/after temperature scaling |
| `figure4_f1_heatmap.png` | Per-class F1 heatmap (all models × platforms) |
| `figure5_{model}_gradient_cross_platform.png` | Gradient saliency cross-platform comparison (4 files) |
| `figure6_jaccard_with_baseline.png` | Jaccard stability heatmap with within-platform baseline |
| `figure7_sensitivity_analysis.png` | Cross-platform ΔAUC across mapping schemes A–E |
| `figure8_sensitivity_heatmap.png` | AUC sensitivity heatmap |
| `figure_di_eod_heatmap.png` | Disparate impact and EOD heatmap |
| `figure_perclass_ece_heatmap.png` | Per-class ECE heatmap |
| `figure_recalibration.png` | ECE before and after temperature scaling |

---

## 9. Evaluation Metrics

| Metric | Definition | Threshold (CPFE) |
|--------|-----------|-----------------|
| **Macro AUC** | Average one-vs-rest AUC across four classes; insensitive to class imbalance | ΔAUC > 30% = severe |
| **ECE** | Expected Calibration Error, M = 10 equal-width bins; measures probability reliability | — |
| **Symmetric DI** | min(P(ŷ=c\|G=A) / P(ŷ=c\|G=B), reciprocal); range (0,1] | < 0.80 = violation; < 0.50 = severe |
| **EOD** | \|TPR_c(ref) − TPR_c(target)\|; measures equalized odds gap | > 0.20 = notable |
| **Jaccard J** | \|top-K features platform A ∩ top-K features platform B\| / \|union\|; K = 10 | < 0.20 = unstable |
| **Bootstrap Z-test** | Z = (AUC₁ − AUC₂) / √(SE₁² + SE₂²); Bonferroni-corrected at α/n_comparisons | p < 0.0083 (k=6) |

Reference platform for DI and EOD comparisons: **Kaggle**.

---

## 10. CPFE Thresholds

Provisional pre-deployment readiness criteria derived from this study. **These thresholds require external validation before informing policy or regulatory decisions.**

| Axis | Threshold | Severity |
|------|-----------|----------|
| **ΔAUC** | < 15% | Acceptable |
| **ΔAUC** | 15–30% | Moderate — recalibration and monitoring required |
| **ΔAUC** | > 30% | Severe — deployment on a new platform without domain-specific retraining carries substantial discriminative performance risk |
| **Jaccard J** | ≥ 0.20 at K = 10 | Minimum for attribution stability |
| **Disparate Impact** | ≥ 0.80 | Minimum for four-fifths rule compliance |

---

## 11. Notes for Reviewers

### For Journal Reviewers (JBI)

- All analysis code is deterministic: random seeds are fixed globally via `configs/config.yaml` (`seed: 42`), and all bootstrap procedures use `np.random.default_rng(seed)`.
- The symmetric DI formula `min(a/b, b/a)` is used throughout (not the asymmetric `a/b` formulation), ensuring platform-direction invariance.
- Temperature scaling is applied to raw pre-softmax logits, not log-softmax outputs, as required for calibration correctness.
- The GoEmotions test split is used exclusively (no train/validation contamination); multi-label conflicts are resolved by dropping ambiguous examples.
- Gradient saliency (`∂P(class)/∂embedding`) is used for attribution — **not** SHAP/Shapley values. The Limitations section in the manuscript discusses this constraint.
- ECE is computed with M = 10 equal-width bins; sensitivity to M is reported in `fairness/ece_binning_sensitivity.csv`.

### For Admissions Committees (ETH Zürich, Cambridge, Oxford, and other universities)

This project demonstrates the following research engineering competencies:

- **End-to-end ML pipeline:** raw data ingestion → label harmonisation → multi-model fine-tuning → cross-platform inference → statistical audit → interpretability analysis, all reproducible from a single `config.yaml`.
- **Statistical rigour:** bootstrap confidence intervals (B = 2000), Bonferroni multiple-comparison correction, DeLong approximation, symmetric fairness metrics.
- **Interpretability:** gradient-based token saliency, Jaccard feature stability, clinical vocabulary retention analysis, within-platform split-half reliability.
- **Fairness-aware evaluation:** disparate impact (with prior-shift correction), equalized odds difference, per-class calibration, five-scheme sensitivity analysis.
- **Software quality:** PEP 8, type-annotated functions, NumPy docstrings, centralised constants in `utils.py`, config-driven paths and seeds.
- **Research scope:** 16 analysis scripts, 28+ publication figures, 14 output CSVs, 4 fine-tuned models (~1.7 GB checkpoints).

---

## 12. Citation

If you use this code or findings in your work, please cite:

```bibtex
@article{pall2025cpfe,
  title   = {Cross-Platform Generalization Failure in Mental Health Natural Language Processing:
             A Systematic Fairness Audit of Transformer Models on Social Media},
  author  = {Pall, Rajveer Singh and Yadav, Sameer},
  journal = {Journal of Biomedical Informatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## 13. License

This project is released under the [MIT License](LICENSE).

---

*For questions or issues, please open a GitHub issue at [github.com/Rajveer-code/mental-health-fairness-nlp](https://github.com/Rajveer-code/mental-health-fairness-nlp).*
