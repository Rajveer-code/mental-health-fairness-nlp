# Cross-Platform Generalisation Failure in Mental Health NLP
## A Five-Axis Fairness Audit of Transformer Models on Social Media

<div align="center">

[![IEEE TNNLS](https://img.shields.io/badge/IEEE%20TNNLS-Submitted%20Apr%202026-blue)](https://github.com/Rajveer-code/mental-health-fairness-nlp)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%204.36-FFD21E)](https://huggingface.co/)

*Rajveer Singh Pall · Sameer Yadav*

*Gyan Ganga Institute of Technology and Sciences, Jabalpur, India*

</div>

---

> **TL;DR** — We introduce the **CPFE framework** (Cross-Platform Fairness Evaluation), a
> five-axis audit protocol, and show that transformer models trained on Kaggle mental health
> data fail systematically when deployed on Reddit and Twitter: AUC drops **30–39%**,
> ECE rises by up to **9×**, prediction rate disparities violate the four-fifths rule
> (**DI < 0.17**), and attribution vocabularies share **near-zero overlap** (*J*≈0 in 13/16
> model–class pairs at *K*=10). Temperature scaling recovers calibration (**88.0% mean ECE
> reduction**) but not discrimination; target-domain fine-tuning on the same labelled split
> yields mean AUC gain **+0.216**.

---

## Abstract

We introduce the Cross-Platform Fairness Evaluation (CPFE) framework — a five-axis audit
protocol covering discriminative performance, calibration, statistical significance,
prediction equity, and attribution stability — and apply it to four transformer models
(BERT, RoBERTa, Emotion-DistilRoBERTa, GoEmotions-RoBERTa) trained on a Kaggle mental
health corpus (*n*=35,556) and evaluated on Reddit (*n*=6,257; GoEmotions emotion labels
mapped to clinical proxies) and Twitter (*n*=2,883; dair-ai/emotion labels mapped to
clinical proxies) test sets. All three cleanly-evaluated models — BERT, RoBERTa, and
Emotion-DistilRoBERTa — exhibit consistent and substantial cross-platform AUC degradation.
Relative to within-platform performance (AUC 0.983–0.987), macro AUC falls by 30.3–35.4%
on Reddit and by 37.9–39.5% on Twitter; results are means across five independent training
seeds. Calibration failure is concurrent and severe: ECE rises from 0.056–0.060 in-domain
to 0.196–0.229 on Reddit and 0.499–0.542 on Twitter. Platform-specific temperature scaling
reduces mean ECE by 88.0% without altering discriminative performance (mean |ΔAUC|<0.01),
confirming that calibration and discrimination represent separable failure modes.
Platform-stratified prediction equity analysis reveals large cross-platform prediction rate
disparities (raw DI < 0.17; prior-shift-adjusted DI: 0.11–0.29 for clinical proxy classes
on Reddit), with equalized odds differences of 0.753–0.830 for depression, anxiety, and
stress proxy classes on Reddit, and 0.755–0.831 for anxiety on Twitter. Gradient attribution
stability analysis indicates near-complete token vocabulary disjunction across platforms
(Jaccard *J*≈0 in 13/16 model–class pairs at *K*=10). These findings establish that
cross-platform validation of discrimination, calibration, prediction equity, and attribution
stability is indispensable for mental health NLP systems intended for heterogeneous deployment
environments. Target-domain fine-tuning on the same labelled calibration split produced a
mean AUC gain of 0.216, indicating that available target-platform labels are more valuable
as training signal than as calibration signal.

---

## CPFE Framework

The five-axis protocol systematically characterises every major failure mode that can emerge
when a mental health NLP classifier is deployed outside its training distribution:

| Axis | Metric | Paper Eq. | Threshold | Key Finding (this study) |
|------|--------|-----------|-----------|--------------------------|
| 1 — Discriminative Performance | Macro-OvR AUC, F1-macro | — | ΔAUC > 30% = severe | 30.3–35.4% Reddit; 37.9–39.5% Twitter |
| 2 — Probabilistic Calibration | ECE (*M*=10 equal-width bins) | Eq. 1 | ECE > 0.06 = concerning | 0.196–0.542 cross-platform |
| 3 — Statistical Significance | Bootstrap *Z*, Bonferroni α′=0.0167 | — | p < α′ | All 12 comparisons p<0.001 |
| 4 — Prediction Equity | Symmetric DI, EOD | Eq. 2, 3 | DI < 0.80 violation | DI < 0.17 on Reddit |
| 5 — Attribution Stability | Gradient saliency + Jaccard@*K* | Eq. 4, 5 | *J*≈0 = collapse | J=0 in 13/16 pairs (Kaggle→Twitter) |

---

## Key Results

### Table 2: Within-Platform Performance (Kaggle test set, *n*=7,620)

Values are means ± SD across five training seeds. ECE is reported with 95% bootstrap CIs (*B*=1,000).

| Model | Accuracy | F1-macro | F1-weighted | AUC | ECE [95% CI] |
|-------|----------|----------|-------------|-----|--------------|
| BERT | 0.930 | 0.874 | 0.931 | 0.984 | 0.060 [0.055, 0.066] |
| RoBERTa | **0.936** | **0.883** | **0.937** | **0.987** | **0.056** [0.052, 0.062] |
| Emotion-DistilRoBERTa | 0.928 | 0.862 | 0.928 | 0.983 | 0.058 [0.053, 0.064] |
| GoEmotions-RoBERTa† | 0.930 | 0.871 | 0.931 | 0.985 | 0.059 [0.054, 0.064] |

### Table 3: Cross-Platform Performance (Reddit *n*=6,257; Twitter *n*=2,883)

ΔAUC% is relative to within-platform Kaggle AUC. Values are means across five seeds.

| Model | Platform | Accuracy | F1-macro | AUC | ECE [95% CI] | ΔAUC% |
|-------|----------|----------|----------|-----|--------------|-------|
| BERT | Reddit | 0.754 | 0.303 | 0.645 | 0.229 [0.218, 0.239] | −34.5% |
| RoBERTa | Reddit | 0.768 | 0.304 | 0.637 | 0.221 [0.210, 0.231] | −35.4% |
| Emotion-DistilRoBERTa | Reddit | 0.768 | **0.332** | **0.685** | **0.208** [0.198, 0.218] | **−30.3%** |
| GoEmotions-RoBERTa† | Reddit† | 0.786 | 0.318 | 0.703 | 0.196 [0.186, 0.205] | −28.6% |
| BERT | Twitter | 0.460 | 0.317 | 0.596 | 0.506 [0.489, 0.524] | **−39.5%** |
| RoBERTa | Twitter | 0.460 | 0.284 | 0.603 | 0.514 [0.497, 0.533] | −38.9% |
| Emotion-DistilRoBERTa | Twitter | 0.409 | 0.290 | **0.611** | 0.542 [0.524, 0.560] | −37.9% |
| GoEmotions-RoBERTa† | Twitter | 0.466 | 0.306 | 0.605 | 0.499 [0.481, 0.517] | −38.6% |

> † GoEmotions-RoBERTa was fine-tuned on the GoEmotions corpus — the same source as
> the Reddit evaluation set. Its Reddit results are **non-independent** (in-distribution
> ceiling, not a cross-platform benchmark). See Section 4.1 of the paper.

Full results across all five CPFE axes are in [`outputs/results/`](outputs/results/).

---

## Repository Structure

```
mental-health-fairness-nlp/
│
├── README.md
├── LICENSE                       MIT
├── CITATION.cff                  Machine-readable citation
├── requirements.txt              Pinned Python dependencies (torch 2.1.0, transformers 4.36.0)
├── environment.yml               Conda environment (cpfe-nlp)
│
├── configs/
│   ├── config.yaml               Master config — all paths + hyperparameters
│   ├── training.yaml             Training hyperparameters (lr, wd, warmup, seeds)
│   ├── evaluation.yaml           ECE bins, bootstrap B, Bonferroni α, Jaccard K values
│   └── models.yaml               4 HuggingFace model IDs + non-independence flags
│
├── data/
│   ├── README.md                 Download instructions, label mapping tables
│   └── splits/cross_platform/   ★ Committed split CSVs — matches paper exactly
│       ├── train.csv             n=35,556 (Kaggle)
│       ├── val.csv               n=7,620
│       ├── test_kaggle.csv       n=7,620 (within-platform)
│       ├── test_reddit.csv       n=6,257 (GoEmotions cross-platform)
│       └── test_twitter.csv      n=2,883 (dair-ai/emotion cross-platform)
│
├── src/                          Analysis source package
│   ├── __init__.py               Re-exports canonical constants from utils
│   ├── utils.py                  ★ Canonical constants, loaders, shared metrics
│   ├── preprocess.py             Data loading, label remapping, stratified splits
│   ├── train.py                  Multi-seed fine-tuning (--seeds 42 0 1 7 123)
│   ├── evaluate.py               Cross-platform inference + raw logit saving
│   ├── fairness_audit.py         CPFE Axes 1+3: AUC (DeLong CIs), ECE, significance
│   ├── code_A1_di_eod_analysis.py    CPFE Axis 4: Symmetric DI + EOD (Eq. 2, 3)
│   ├── code_A3_A4_A6_ece_jaccard.py  CPFE Axis 2: ECE bootstrap CIs; Jaccard K-sensitivity
│   ├── code_A5_temperature_scaling.py CPFE Axis 2: Temperature scaling (Eq. 6, raw logits)
│   ├── shap_analysis.py          CPFE Axis 5: Gradient saliency (Captum, Eq. 4)
│   ├── jaccard_full_analysis.py  Jaccard stability + within-platform baseline (Eq. 5)
│   ├── sensitivity_analysis.py   Label mapping schemes A–D (Section 4.4)
│   ├── perclass_ece_analysis.py  Per-class ECE (one-vs-rest)
│   ├── truncation_audit.py       Exact tokenizer-based truncation rates
│   ├── code_A2_stress_attribution.py  Stress-class gradient attribution figures
│   ├── code_A4_patch_attribution.py   Phrase-level gradient attribution
│   └── gpt_eval.py               Stub — GPT-4 evaluation (future work)
│
├── scripts/                      ★ Numbered entry points for reproduction
│   ├── 01_download_data.py       Download HuggingFace datasets
│   ├── 02_preprocess.py          Regenerate data splits (optional — already committed)
│   ├── 03_train_all_models.py    Fine-tune 4 models × 5 seeds
│   ├── 04_evaluate_crossplatform.py   Tables 2, 3, 4, 5
│   ├── 05_calibration.py         ECE cols + Table 7 (temperature scaling)
│   ├── 06_fairness_audit.py      Table 6 (Symmetric DI, EOD)
│   ├── 07_attribution_stability.py    Table 8 (Jaccard@K)
│   ├── 08_sensitivity_analysis.py     Table 10 (mapping schemes A–D)
│   └── 09_reproduce_all_tables.py  ★ Master entry point — print all tables
│
└── outputs/
    ├── models/                   (gitignored — ~1.7 GB; regenerate with script 03)
    └── results/                  ★ Committed — all CSVs match paper tables
        ├── master_results.csv                    Tables 2+3
        ├── {model}_{platform}_predictions.csv    12 files (per-sample + logits)
        ├── fairness/
        │   ├── fairness_audit_full.csv           Tables 2–5 (full)
        │   ├── pairwise_auc_comparisons.csv      Table 5
        │   ├── di_eod_table.csv                  Table 6
        │   ├── temperature_scaling_results.csv   Table 7
        │   ├── jaccard_full_analysis.csv         Table 8
        │   ├── ece_bootstrap_cis.csv             ECE 95% CIs
        │   └── perclass_ece.csv                  Per-class ECE
        └── shap/                                 48 top-word CSVs
```

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (strongly recommended for training; CPU inference works)
- ~20 GB disk space for model checkpoints (4 models × 5 seeds)

### Install

```bash
git clone https://github.com/Rajveer-code/mental-health-fairness-nlp
cd mental-health-fairness-nlp

# pip (recommended)
pip install -r requirements.txt

# conda
conda env create -f environment.yml && conda activate cpfe-nlp
```

### Data Download

```bash
# GoEmotions (Reddit) and dair-ai/emotion (Twitter) download automatically
python scripts/01_download_data.py

# Kaggle dataset requires a free account + API key
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)
kaggle datasets download suchintikasarkar/sentiment-analysis-for-mental-health
unzip sentiment-analysis-for-mental-health.zip -d data/raw/kaggle_mental_health/
```

> **Note:** The pre-computed split CSVs in `data/splits/cross_platform/` are already
> committed and match the paper exactly. Raw data is only needed to regenerate splits.

---

## Reproducing Results

### Quick start — print all tables (no GPU required)

```bash
python scripts/09_reproduce_all_tables.py
```

This reads the committed CSVs and prints Tables 2, 3, 6, 7, and 8 formatted to match
the paper. All values match within rounding tolerance (±0.001).

### Full pipeline — reproduce from scratch

```bash
python scripts/01_download_data.py        # Download datasets
python scripts/02_preprocess.py           # Regenerate splits (optional)
python scripts/03_train_all_models.py     # ~8–10 h on A100
python scripts/04_evaluate_crossplatform.py  # Tables 2, 3, 4, 5
python scripts/05_calibration.py          # Tables 2/3 ECE, Table 7
python scripts/06_fairness_audit.py       # Table 6
python scripts/07_attribution_stability.py   # Table 8
python scripts/08_sensitivity_analysis.py    # Table 10
python scripts/09_reproduce_all_tables.py    # Print and export all
```

### Script → Paper table mapping

| Script | Output CSV | Paper Table |
|--------|-----------|-------------|
| `04_evaluate_crossplatform.py` | `outputs/results/master_results.csv` | Tables 2, 3 |
| `04_evaluate_crossplatform.py` | `fairness/pairwise_auc_comparisons.csv` | Table 5 |
| `05_calibration.py` | `fairness/temperature_scaling_results.csv` | Table 7 |
| `06_fairness_audit.py` | `fairness/di_eod_table.csv` | Table 6 |
| `07_attribution_stability.py` | `fairness/jaccard_full_analysis.csv` | Table 8 |
| `08_sensitivity_analysis.py` | `fairness/sensitivity_analysis.csv` | Table 10 |

### Estimated runtimes (NVIDIA A100 40 GB)

| Step | Runtime |
|------|---------|
| Download datasets | ~5 min |
| Train 1 model × 1 seed | ~25–35 min |
| Train all 4 × 5 seeds | ~8–10 hours |
| Evaluate + fairness audit | ~25 min |
| Attribution analysis | ~30 min |

---

## CPFE Axes — Implementation Details

All canonical metric formulas are defined in `src/utils.py` and imported by analysis scripts.

| Axis | Canonical Formula | Implementation |
|------|------------------|----------------|
| **ECE** (Eq. 1) | `ECE = Σ_m (|B_m|/n) · |acc(B_m) - conf(B_m)|` | `utils.compute_aggregate_ece()`, *M*=10 equal-width bins |
| **Sym. DI** (Eq. 2) | `DI_c = min(rate_A/rate_B, rate_B/rate_A)` | `code_A1_di_eod_analysis.py` |
| **EOD** (Eq. 3) | `EOD_c = |TPR_c(Kaggle) - TPR_c(target)|` | `code_A1_di_eod_analysis.py` |
| **Gradient saliency** (Eq. 4) | `s_i = ||∂P(y|x)/∂E_i||_2` | `utils.compute_token_importance()` via Captum |
| **Jaccard** (Eq. 5) | `J_K(A,B) = |top_K(A) ∩ top_K(B)| / |top_K(A) ∪ top_K(B)|` | `jaccard_full_analysis.py` at K∈{5,10,15,20} |
| **Temp. scaling** (Eq. 6) | `p_T = softmax(z/T)`, T* minimises NLL on raw logits | `code_A5_temperature_scaling.py` |

**Key design choices:**
- DI uses the **symmetric** formula `min(a/b, b/a)` — never the one-directional `a/b`
- Temperature scaling operates on **raw pre-softmax logits** (saved by `evaluate.py`), not on softmax outputs or log-softmax reconstructions
- AUC significance uses **bootstrap SE** (*B*=2,000) with Bonferroni α′=0.0167 (3 comparisons per model)
- GoEmotions-RoBERTa Reddit results are flagged `†` in all outputs with a non-independence warning

---

## Models

All four models fine-tuned as 4-class sequence classifiers with AdamW (*lr*=2×10⁻⁵,
*wd*=0.01, linear warmup 10%, gradient clipping 1.0, *max_seq_length*=64, early stopping
on validation macro-F1). Full hyperparameters: [`configs/training.yaml`](configs/training.yaml).

| Code key | HuggingFace ID | Params | Pretraining |
|----------|---------------|--------|-------------|
| `bert` | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | 110M | Wikipedia + Books |
| `roberta` | [`roberta-base`](https://huggingface.co/roberta-base) | 125M | General (dynamic masking) |
| `mentalbert` | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | 82M | Emotion multi-source |
| `mentalroberta` | [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) | 125M | GoEmotions (Reddit)† |

---

## Datasets

See [`data/README.md`](data/README.md) for full mapping tables and download instructions.

| Platform | Dataset | Role | *n* used | Source |
|----------|---------|------|---------|--------|
| Kaggle | Combined Mental Health | Train + within-test | 35,556 / 7,620 | [Sarkar 2022](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) |
| Reddit | GoEmotions | Cross-platform test | 6,257 | [Demszky et al. 2020](https://huggingface.co/datasets/go_emotions) |
| Twitter | dair-ai/emotion | Cross-platform test | 2,883 | [Saravia et al. 2018](https://huggingface.co/datasets/dair-ai/emotion) |

**Label distribution shift** (confound independent of covariate shift, Section 5.5):

| Class | Kaggle test | Reddit test | Twitter test |
|-------|------------|------------|--------------|
| normal | 28.7% | 45.1% | 43.6% |
| depression | 56.6% | 16.1% | 30.1% |
| anxiety | 7.5% | 19.0% | 12.3% |
| stress | 7.2% | 19.8% | 14.0% |

**Sensitivity analysis** (Section 4.4): four mapping schemes A–D confirm cross-platform
degradation is not an artefact of label-mapping heuristics. Mapping D (collapsing
anxiety+stress into a distress superclass) shows the same 26.8–39.0% AUC drops.

---

## Notes for Reviewers

### Reproducibility

- All random states are controlled: Python `random`, NumPy, PyTorch, and HuggingFace
  `transformers` seeds are set identically at the start of every script.
- Split CSVs are committed to the repo — cloning the repository is sufficient to
  reproduce all post-training analyses without re-running preprocessing.
- Result CSVs (`outputs/results/`) are committed and match the paper within ±0.001.
  GPU re-runs will match if checkpoints are regenerated with the same seeds.

### Statistical correctness

- **ECE**: M=10 **equal-width** bins (not equal-frequency). Bin sensitivity in
  `fairness/ece_binning_sensitivity.csv`.
- **AUC CIs**: DeLong method via `sklearn.metrics.roc_auc_score` (per-class OvR);
  macro SE estimated by bootstrap (*B*=2,000), not naive averaging of per-class SEs
  (which ignores covariance). See `fairness_audit.py` lines 49–93.
- **DI formula**: always `min(a/b, b/a)` — platform-direction invariant.
- **Temperature scaling**: T* optimised on raw logits via `scipy.minimize_scalar`
  (bounded, method='bounded') on a 10% stratified calibration split.
- **Attribution**: gradient-based saliency (Captum `IntegratedGradients`-adjacent
  backprop), L2-normed per token. **Not** SHAP/Shapley values. Limitations discussed
  in Section 7.0.3.

---

## Citation

```bibtex
@article{pall2026crossplatform,
  title     = {Cross-Platform Generalisation Failure in Mental Health Natural
               Language Processing: A Five-Axis Fairness Audit of Transformer
               Models on Social Media},
  author    = {Pall, Rajveer Singh and Yadav, Sameer},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2026},
  note      = {Manuscript received April 26, 2026}
}
```

---

## License

[MIT License](LICENSE). Code and pre-computed results are freely reusable.

Underlying datasets carry their own licenses:
GoEmotions — Apache 2.0 · dair-ai/emotion — MIT · Kaggle Combined Mental Health — CC BY 4.0

---

## Acknowledgements

[HuggingFace](https://huggingface.co/) · [Captum](https://captum.ai/) ·
[scikit-learn](https://scikit-learn.org/) ·
Demszky et al. (2020) for GoEmotions · Saravia et al. (2018) for dair-ai/emotion ·
Sarkar (2022) for the Kaggle Combined Mental Health corpus.

---

## Contact

**Rajveer Singh Pall** — [rajveer.singhpall.cb23@ggits.net](mailto:rajveer.singhpall.cb23@ggits.net)

**Sameer Yadav** — [sameer.yadav@ggits.org](mailto:sameer.yadav@ggits.org)

Department of Computer Science and Business Systems,
Gyan Ganga Institute of Technology and Sciences, Jabalpur, 482003, India
