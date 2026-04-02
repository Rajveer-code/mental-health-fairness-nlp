'''# Cross-Platform Fairness Evaluation of NLP Models for Mental Health Detection

<div align="center">

**A Cross-Platform Fairness Evaluation (CPFE) Framework for Mental Health NLP**  
**Multi-Model Audit Using Clinical-Grade Statistical Methods**

*Rajveer Singh Pall · Sameer Yadav*  
*Gyan Ganga Institute of Technology and Sciences, Jabalpur, India*

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 1. Project Overview

Large pre-trained language models (LLMs) can overfit linguistic artifacts and fail to generalize across social media platforms. This repository introduces the **CPFE (Cross-Platform Fairness Evaluation) framework** and demonstrates how state-of-the-art mental health models (BERT, RoBERTa, DistilRoBERTa, SamLowe-RoBERTa) degrade when tested on unseen platforms.

Core objective:
- Train on Kaggle mental health dataset (multi-source, 7 labels mapped into 4 classes)
- Validate across unseen Reddit and Twitter datasets
- Audit performance with statistical rigor (DeLong CIs, ECE, Bonferroni, fairness metrics)
- Attribute token importance and measure feature stability
- Confirm findings in label mapping sensitivity studies

> Value for reviewers/admissions committee: this is a complete, reproducible ML audit pipeline with transparent methodology, interpretable fairness measures, and cross-platform robustness validation.

---

## 2. Key Contributions

1. CPFE framework: end-to-end cross-platform model fairness audit (data prep, training, evaluation, statistical audit, attribution, sensitivity)
2. Multi-platform evidence: performance collapse on Reddit/Twitter, especially for "stress" and "anxiety"
3. Clinical-grade metrics: per-class AUC with DeLong 95% CI, ECE and calibration curves, disparate impact, equalized odds difference
4. Feature introspection: gradient-based token saliency and Jaccard stability across platform pairs
5. Mapping robustness: 4-class, binary, 3-class label schemas with stable degradation pattern

---

## 3. Repository Structure (full)

```
mental-health-fairness-nlp/
│
├── configs/
│   └── config.yaml                  # hyperparameters, dataset/model paths, auditing thresholds
│
├── data/
│   ├── raw/                         # source datasets (Kaggle CSV, HuggingFace cached directories)
│   │   ├── kaggle_mental_health/
│   │   │   └── Combined Data.csv
│   │   ├── reddit_goemotions/
│   │   │   └── dataset_dict.json
│   │   └── twitter_emotion/
│   │       └── dataset_dict.json
│   └── splits/
│       ├── cross_platform/
│       │   ├── train.csv
│       │   ├── val.csv
│       │   ├── test_kaggle.csv
│       │   ├── test_reddit.csv
│       │   └── test_twitter.csv
│       ├── kaggle/ ...
│       ├── reddit/ ...
│       └── twitter/ ...
│
├── outputs/
│   ├── models/
│   │   ├── bert/ 
│   │   ├── roberta/
│   │   ├── mentalbert/
│   │   └── mentalroberta/
│   ├── results/
│   │   ├── master_results.csv
│   │   ├── bert_eval.json, roberta_eval.json, ...
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
│       ├── figure5_*_shap_cross_platform.png
│       ├── figure6_feature_stability.png
│       ├── figure7_sensitivity_analysis.png
│       └── figure8_sensitivity_heatmap.png
│
├── src/
│   ├── preprocess.py                # data loading, cleaning, label mapping, train/test splits
│   ├── train.py                     # model fine-tuning loop + checkpoint save
│   ├── evaluate.py                  # cross-platform inference + predictions CSV
│   ├── fairness_audit.py            # DeLong AUC, ECE, Bonferroni, subgroup fairness
│   ├── shap_analysis.py             # gradient saliency, Jaccard stability of features
│   ├── sensitivity_analysis.py      # 4-class/binary/3-class robustness check from predictions
│   ├── truncation_audit.py          # token-length diagnostics (optional)
│   ├── perclass_ece_analysis.py     # per-class calibration breakdown
│   ├── ... other analysis scripts
│   └── utils.py                     # helper functions for replication
│
├── requirements.txt                 # exact Python packages (transformers, torch, datasets, sklearn, etc.)
├── README.md                        # this file
└── LICENSE
```

> Optional directory for admission reviewers: `resources/` (not in baseline). If reviewers add `resources/`, include:
> - `resources/README.md` for human language, research narrative, and highlights
> - `resources/papers/` with references and summary notes
> - `resources/visuals/` with polished figures and diagrams
> - `resources/questionnaire/` with expected review questions and author answers

---

## 4. Data and Label Schema

### 4.1 Datasets
- Kaggle Mental Health (`data/raw/kaggle_mental_health/Combined Data.csv`): 53K samples, primary training source.
- GoEmotions Reddit (`data/raw/reddit_goemotions/`): 54K comments, cross-platform validation.
- dair-ai/emotion Twitter (`data/raw/twitter_emotion/`): 20K tweets, cross-platform validation.

### 4.2 Unified label mapping (primary)
- normal  → 0
- depression → 1 (includes "Suicidal", "Bipolar", and depression-adjacent categories)
- anxiety → 2
- stress → 3 (includes "Personality disorder" in Kaggle mapping)

Mapping algorithm in `src/preprocess.py` uses corpus-specific maps (`KAGGLE_TO_UNIFIED`, `GOEMO_TO_UNIFIED`, `DAIREMO_TO_UNIFIED`) and cleans text with `clean_text()`.

### 4.3 Additional label mappings for stress test
- Mapping B: binary normal vs. mental health (depression/anxiety/stress)
- Mapping C: 3-class (normal, depression, distress [anxiety+stress])

These are computed in `src/sensitivity_analysis.py` from saved inference outputs.

---

## 5. Models and Baselines

| model key | HF model id | alias | notes |
|---|---|---|---|
| `bert` | `bert-base-uncased` | BERT-base | general baseline
| `roberta` | `roberta-base` | RoBERTa-base | strong baseline
| `mentalbert` | `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa | emotion pre-trained
| `mentalroberta` | `SamLowe/roberta-base-go_emotions` | SamLowe-RoBERTa | robust emotion pre-trained

Training hyperparameters from `configs/config.yaml`:
- `max_length`: 64
- `batch_size`: 16
- `learning_rate`: 2e-5
- `epochs`: 5
- `seed`: 42

---

## 6. Outcomes and Key Results

### 6.1 Summary results (from `outputs/results/master_results.csv`)

- Within-platform (Kaggle) AUC: 0.983–0.987
- Reddit AUC: 0.629–0.699
- Twitter AUC: 0.596–0.605
- Cross-platform drop: 29%–40% for all models
- ECE (within-platform): ~0.056–0.060
- ECE (Twitter): ~0.499–0.542

### 6.2 Output tables
- `outputs/results/master_results.csv`: combined per-model-per-platform metrics
- `outputs/results/{model}_eval.json`: model-specific evaluation by platform
- `outputs/results/fairness/fairness_audit_full.csv`: fairness breakdown with DeLong and ECE
- `outputs/results/sensitivity/sensitivity_full_results.csv`: mapping robustness per model

### 6.3 Figures
- Figures 1–4: fairness audit plots
- Figures 5–6: SHAP/grad-based feature importance and stability
- Figures 7–8: sensitivity mapping robustness

---

## 7. Reproducible Pipeline

### 7.1 Setup

```bash
git clone https://github.com/Rajveer-code/mental-health-fairness-nlp.git
cd mental-health-fairness-nlp
python -m venv .venv
.venv\Scripts\activate  # Windows
# or source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 7.2 Full flow

1. `python src/preprocess.py`  
   - loads raw data from `data/raw/` and HuggingFace
   - applies clean_text and label mappings
   - writes `data/splits/cross_platform/*`

2. `python src/train.py --model all`  
   - trains four models sequentially
   - stores best checkpoints in `outputs/models/{model}/`
   - outputs logs and metrics in `outputs/results/`

3. `python src/evaluate.py`  
   - infers all models on Kaggle/Reddit/Twitter test sets
   - saves per-sample CSVs in `outputs/results/`
   - writes `outputs/results/master_results.csv`

4. `python src/fairness_audit.py`  
   - computes DeLong CIs, ECE, subgroup measures, Bonferroni tests
   - saves `outputs/results/fairness/` and figures

5. `python src/shap_analysis.py`  
   - computes gradient-based attribution and Jaccard stability
   - saves figures in `outputs/figures/`

6. `python src/sensitivity_analysis.py`  
   - generates mapping robustness metrics and figures
   - saves under `outputs/results/sensitivity/`

### 7.3 Optional audits

- `python src/truncation_audit.py` - token length truncation effects
- `python src/perclass_ece_analysis.py` - per-class calibration plots
- `python src/code_A*` – extended analyses used in manuscript sections

---

## 8. Evaluation Metrics (for non-technical reviewers)

- **AUC (Area Under ROC Curve)**: discriminative power; 1.0 perfect, 0.5 random.
- **ECE (Expected Calibration Error)**: probability reliability; 0.0 ideal, 0.5 akin to random.
- **DeLong CI**: statistical confidence bounds around AUC.
- **Disparate Impact**: group prediction parity across demographic proxies (age/gender inferred from text).
- **Equalized Odds Difference**: fairness measure for true positive rates across groups.

> Clinical flag: cross-platform AUC drop > 20% is concerning; > 35% is unacceptable.

---

## 9. Entry for admissions/resources directory

Add a `resources/` folder in the root during application submission to highlight project narrative:
- `resources/summary.pdf` (state-of-art findings overview)
- `resources/figures/` (visually annotated charts for reviewers)
- `resources/reproducibility/` (commands, environment details)
- `resources/appendix/` (detailed statistics, contribution bullets)

---

## 10. Notes for reviewers

- All model weights, figures, and results are recoverable from the scripts and dataset splits.
- No hidden data preprocessing is required: complete transparency in `src/preprocess.py`.
- The full pipeline is configured via `configs/config.yaml`, enabling quick hyperparameter changes.
- Well-suited for masters applications due to strong cross-platform fairness narrative and reproducibility focus.

---

## 11. Licensing

MIT License (see `LICENSE`).
'''
