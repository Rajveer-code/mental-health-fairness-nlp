# Cross-Platform Fairness Evaluation of LLMs for Mental Health Detection

<div align="center">

**A Cross-Platform Fairness Evaluation (CPFE) Framework for Mental Health NLP:**
**Multi-Model Audit Using Clinical-Grade Statistical Methods**

*Rajveer Singh Pall · Sameer Yadav*
*Gyan Ganga Institute of Technology and Sciences, Jabalpur, India*

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Under Review](https://img.shields.io/badge/Status-Under%20Review%20%40%20JBI-blueviolet)](https://www.sciencedirect.com/journal/journal-of-biomedical-informatics)

</div>

---

## Overview

Mental health NLP models achieve impressive accuracy on the platform they are trained on. But what happens when they are deployed on a different platform — serving a different user population with different linguistic patterns?

This repository presents a systematic answer. We fine-tune four transformer models on a multi-source mental health dataset (Kaggle, n=35,556) and evaluate them on two external platforms — Reddit (n=6,330) and Twitter (n=2,883) — that the models have never seen. We apply clinical-grade statistical methodology: DeLong 95% confidence intervals, Expected Calibration Error, and Bonferroni-corrected pairwise significance testing. The results are unambiguous.

> **Key finding:** AUC degrades by 29–40% cross-platform. Expected Calibration Error rises from 0.056–0.060 within-platform to 0.499–0.542 on Twitter, rendering predicted probabilities clinically unreliable. The stress and anxiety classes — the highest clinical priority for early screening — collapse to near-chance AUC. The cause: models learn platform-specific linguistic artefacts rather than transferable clinical signals.

---

## The CPFE Framework

This study introduces the **Cross-Platform Fairness Evaluation (CPFE) Framework** — a replicable four-step methodology for auditing NLP mental health classifiers before clinical deployment.

```
Step 1 — Within-Platform Baseline
        Train + evaluate on held-out data from the training platform.
        Threshold: AUC ≥ 0.80, ECE ≤ 0.10 required to proceed.

Step 2 — Cross-Platform Evaluation
        Evaluate on ≥ 2 external platforms (never seen during training).
        ΔAUC > 20% = clinically concerning. ΔAUC > 35% = clinically unacceptable.

Step 3 — Clinical-Grade Statistical Audit
        Bonferroni-corrected DeLong Z-tests. Per-class AUC with 95% CIs.
        Disparate Impact + Equalized Odds Difference across platforms.

Step 4 — Feature Attribution Analysis
        Gradient saliency per platform. Jaccard stability of top-10 features.
        J < 0.20 = model exploiting platform artefacts, not clinical signals.
```

The CPFE Framework is **model-agnostic**, **platform-agnostic**, and requires only a trained classifier, held-out test sets from ≥ 2 platforms, and per-sample probability outputs.

---

## Results Summary

### Cross-Platform AUC Degradation

| Model | Kaggle AUC | Reddit AUC | Reddit Drop | Twitter AUC | Twitter Drop |
|---|---|---|---|---|---|
| BERT | 0.984 | 0.630 | −36.0% 🔴 | 0.596 | −39.5% 🔴 |
| RoBERTa | 0.987 | 0.629 | −36.2% 🔴 | 0.603 | −38.9% 🔴 |
| DistilRoBERTa | 0.983 | 0.673 | −31.5% 🟠 | 0.611 | −37.9% 🔴 |
| SamLowe-RoBERTa | 0.985 | 0.699 | **−29.1%** 🟢 | 0.605 | −38.6% 🔴 |

> 🔴 > 35% drop (clinically unacceptable) · 🟠 30–35% drop (concerning) · 🟢 Smallest observed

### Calibration Collapse

| Model | Kaggle ECE | Reddit ECE | Twitter ECE |
|---|---|---|---|
| BERT | 0.060 | 0.240 | 0.506 |
| RoBERTa | 0.056 | 0.235 | 0.513 |
| DistilRoBERTa | 0.058 | 0.231 | 0.542 |
| SamLowe-RoBERTa | **0.059** | **0.212** | **0.499** |

> ECE approaching 0.50 = calibration no better than random. Any clinical application relying on probability thresholds would be operating on meaningless estimates.

### Stress Class Near-Chance AUC on Twitter

| Model | Stress AUC (Kaggle) | Stress AUC (Twitter) |
|---|---|---|
| BERT | 0.969 | 0.530 |
| RoBERTa | 0.972 | 0.522 |
| DistilRoBERTa | 0.966 | 0.535 |
| SamLowe-RoBERTa | 0.971 | 0.542 |

### Sensitivity Analysis — Finding Is Robust Across Label Mappings

| Mapping | BERT Reddit Drop | RoBERTa Reddit Drop | DistilRoBERTa Reddit Drop | SamLowe Reddit Drop |
|---|---|---|---|---|
| A — 4-class (original) | 36.0% | 36.2% | 31.5% | 29.1% |
| B — Binary (normal vs. MH) | 36.5% | 35.1% | 31.7% | 26.3% |
| C — 3-class (normal/depression/distress) | 38.0% | 37.2% | 33.5% | 29.5% |

All 24 model-platform-mapping combinations exceed the 20% threshold. The finding does not depend on the label mapping scheme.

---

## Repository Structure

```
mental-health-fairness-nlp/
│
├── configs/
│   └── config.yaml                    # All hyperparameters and paths
│
├── src/
│   ├── preprocess.py                  # Dataset loading, label remapping, splits
│   ├── train.py                       # Fine-tuning all 4 models (mixed precision FP16)
│   ├── evaluate.py                    # Cross-platform evaluation, saves predictions CSV
│   ├── fairness_audit.py              # DeLong CI, ECE, Bonferroni, Figures 1–4
│   ├── shap_analysis.py               # Gradient saliency, Jaccard stability, Figures 5–6
│   ├── sensitivity_analysis.py        # Label mapping robustness check, Figures 7–8
│   └── utils.py                       # Shared utilities
│
├── data/
│   └── splits/cross_platform/
│       ├── train.csv                  # Kaggle training set (n=35,556)
│       ├── val.csv                    # Kaggle validation set (n=7,619)
│       ├── test_kaggle.csv            # Within-platform test (n=7,620)
│       ├── test_reddit.csv            # Cross-platform: Reddit (n=6,330)
│       └── test_twitter.csv           # Cross-platform: Twitter (n=2,883)
│
├── outputs/
│   ├── models/
│   │   ├── bert/                      # Best BERT checkpoint
│   │   ├── roberta/                   # Best RoBERTa checkpoint
│   │   ├── mentalbert/                # Best DistilRoBERTa checkpoint
│   │   └── mentalroberta/             # Best SamLowe-RoBERTa checkpoint
│   ├── results/
│   │   ├── master_results.csv         # All evaluation metrics
│   │   ├── fairness/
│   │   │   └── fairness_audit_full.csv
│   │   └── sensitivity/
│   │       ├── sensitivity_full_results.csv
│   │       └── sensitivity_drops.csv
│   └── figures/
│       ├── figure1_forest_plot.png
│       ├── figure2_platform_degradation.png
│       ├── figure3_calibration_curves.png
│       ├── figure4_f1_heatmap.png
│       ├── figure5_bert_shap_cross_platform.png
│       ├── figure5_roberta_shap_cross_platform.png
│       ├── figure5_mentalbert_shap_cross_platform.png
│       ├── figure5_mentalroberta_shap_cross_platform.png
│       ├── figure6_feature_stability.png
│       ├── figure7_sensitivity_analysis.png
│       └── figure8_sensitivity_heatmap.png
│
├── requirements.txt
└── README.md
```

---

## Datasets

Three publicly available datasets are used:

| Dataset | Source | Original Labels | Mapped To | n (after filter) |
|---|---|---|---|---|
| [Kaggle Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) | Multi-source social media | 7 classes | normal / depression / anxiety / stress | 50,795 |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Reddit comments | 28 emotions | normal / depression / anxiety / stress | 42,196 |
| [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | Twitter posts | 6 emotions | normal / depression / anxiety / stress | 19,220 |

**Label mapping** follows established computational psychiatry practice (Coppersmith et al., 2014; CLPsych 2016 shared task). Robustness to alternative mappings is confirmed by the sensitivity analysis.

---

## Models

| Model Key | HuggingFace ID | Best Val F1 | Best Val AUC |
|---|---|---|---|
| `bert` | `bert-base-uncased` | 0.877 | 0.978 |
| `roberta` | `roberta-base` | 0.887 | 0.987 |
| `mentalbert` | `j-hartmann/emotion-english-distilroberta-base` | 0.867 | 0.985 |
| `mentalroberta` | `SamLowe/roberta-base-go_emotions` | 0.876 | 0.985 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Rajveer-code/mental-health-fairness-nlp.git
cd mental-health-fairness-nlp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**Hardware used:** NVIDIA RTX 4060 Laptop GPU (8GB VRAM), CUDA 12.4, PyTorch 2.6.0+cu124.
Training runs on any CUDA-capable GPU with ≥ 6GB VRAM using the provided mixed-precision FP16 config.

---

## Usage

All scripts are run from the repository root. Hyperparameters are in `configs/config.yaml`.

### Step 1 — Preprocess datasets and create cross-platform splits

```bash
python src/preprocess.py
```

Downloads datasets from HuggingFace Hub, applies label remapping, and saves train/val/test splits to `data/splits/cross_platform/`.

### Step 2 — Fine-tune all four models

```bash
python src/train.py
```

Fine-tunes BERT, RoBERTa, DistilRoBERTa, and SamLowe-RoBERTa sequentially. Best checkpoints saved to `outputs/models/`. Uses mixed precision FP16, batch size 16, max sequence length 64, learning rate 2e-5, 5 epochs, seed 42.

### Step 3 — Cross-platform evaluation

```bash
python src/evaluate.py
```

Evaluates each model on all three test sets (Kaggle within-platform, Reddit cross-platform, Twitter cross-platform). Saves per-sample predictions to `outputs/results/{model}_{platform}_predictions.csv`.

### Step 4 — Fairness audit (Figures 1–4)

```bash
python src/fairness_audit.py
```

Computes DeLong 95% CIs, ECE, Bonferroni-corrected pairwise tests, Disparate Impact, and Equalized Odds Difference. Generates Figures 1–4.

### Step 5 — Feature attribution analysis (Figures 5–6)

```bash
python src/shap_analysis.py
```

Computes gradient-based token saliency and Jaccard cross-platform stability. Generates Figures 5a–5d and Figure 6.

### Step 6 — Sensitivity analysis (Figures 7–8)

```bash
python src/sensitivity_analysis.py
```

Tests label mapping robustness under three alternative schemas (4-class, binary, 3-class) without retraining. Generates Figures 7–8.

### Run full pipeline

```bash
python src/preprocess.py && \
python src/train.py && \
python src/evaluate.py && \
python src/fairness_audit.py && \
python src/shap_analysis.py && \
python src/sensitivity_analysis.py
```

---

## Configuration

Key parameters in `configs/config.yaml`:

```yaml
training:
  batch_size: 16
  max_length: 64
  learning_rate: 2.0e-5
  weight_decay: 0.01
  epochs: 5
  warmup_ratio: 0.10
  seed: 42
  fp16: true

evaluation:
  ece_bins: 10
  jaccard_k: 10
  bonferroni_n_comparisons: 3

paths:
  data: data/splits/cross_platform
  models: outputs/models
  results: outputs/results
  figures: outputs/figures
```

---

## Figures

| Figure | Description |
|---|---|
| Figure 1 | Per-class AUC forest plot with 95% DeLong CIs — all models × all platforms |
| Figure 2 | Cross-platform degradation — macro F1, AUC, ECE grouped bar chart |
| Figure 3 | Calibration curves — 4×3 grid (4 models × 3 platforms) with ECE values |
| Figure 4 | Per-class F1 heatmap — 12 model-platform rows × 4 class columns |
| Figures 5a–5d | Cross-platform SHAP feature attribution — top-15 tokens per platform per model |
| Figure 6 | Jaccard feature stability — top-10 token overlap across platforms |
| Figure 7 | Sensitivity analysis bar chart — AUC drop under 3 label mapping schemes |
| Figure 8 | Sensitivity analysis heatmap — AUC across all model-platform-mapping combinations |

---

## Key Findings

1. **29–40% AUC degradation** on external platforms — consistent across all 4 models, statistically robust after Bonferroni correction (10/12 pairwise comparisons p < 0.001).

2. **Calibration collapse** — ECE rises from 0.056–0.060 (Kaggle) to 0.499–0.542 (Twitter). Any probability-threshold-based clinical application would be unreliable.

3. **Stress and anxiety collapse** — both classes reach near-chance AUC on Twitter (0.522–0.542 for stress), directly undermining the clinical utility for early mental health screening.

4. **Feature shift confirmed** — gradient attribution shows predictive tokens shift from clinical vocabulary (medication, therapy, relationship) to stopwords (its, little, this) across platforms. Jaccard stability < 0.15 for all models.

5. **Domain pretraining helps** — SamLowe-RoBERTa, pretrained on Reddit (GoEmotions), shows the smallest Reddit AUC drop (−29.1%) and lowest Reddit ECE (0.212), supporting the domain-adaptive pretraining hypothesis.

6. **Sensitivity analysis** — all 24 model-platform-mapping combinations exceed the 20% AUC threshold under all three label mapping schemes. The finding is robust to label construction choices.

7. **Accuracy ≠ deployment fairness** — RoBERTa, the best within-platform model (AUC 0.987), shows the largest cross-platform drop (−36.2% to Reddit), demonstrating that within-platform ranking is a poor guide for cross-platform model selection.

---

## Citation

If you use this work or the CPFE Framework, please cite:

```bibtex
@article{pall2026cpfe,
  title   = {A Cross-Platform Fairness Evaluation (CPFE) Framework for Mental Health NLP:
             Multi-Model Audit Using Clinical-Grade Statistical Methods},
  author  = {Pall, Rajveer Singh and Yadav, Sameer},
  journal = {Journal of Biomedical Informatics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Requirements

```
torch>=2.6.0
transformers>=4.40.0
datasets>=2.18.0
scikit-learn>=1.4.0
scipy>=1.12.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
pyyaml>=6.0.1
tqdm>=4.66.0
```

Full list with pinned versions in `requirements.txt`.

---

## Hardware Requirements

| Component | Minimum | Used in this study |
|---|---|---|
| GPU VRAM | 6 GB | 8 GB (RTX 4060 Laptop) |
| GPU | Any CUDA 11.8+ | CUDA 12.4 |
| RAM | 16 GB | 16 GB |
| Disk | 10 GB | ~15 GB (models + data) |
| Training time per model | ~45 min | ~50 min (5 epochs, FP16) |

CPU-only training is possible but will be very slow (~8–12 hours per model). Set `fp16: false` in `config.yaml` for CPU.

---

## Reproducibility

All experiments use `seed = 42` fixed throughout preprocessing, training, and evaluation. All data splits, model checkpoints, prediction CSVs, and figure generation scripts are included. The sensitivity analysis requires only the prediction CSVs and runs in under 2 minutes without GPU.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Rajveer Singh Pall** — B.Tech CS & Business Systems, GGITS Jabalpur
GitHub: [@Rajveer-code](https://github.com/Rajveer-code)

**Sameer Yadav** (Corresponding Author) — sameeryadav@ggits.org
