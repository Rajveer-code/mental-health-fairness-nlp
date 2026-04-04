# CLAUDE.md — Project Master Instructions
# Mental Health NLP: Cross-Platform Fairness Audit
# JBI Submission | Research Codebase

---

## 1. Project Context

This is the source code for a peer-reviewed research paper submitted to the
**Journal of Biomedical Informatics (JBI)**. The paper studies cross-platform
generalization and algorithmic fairness of transformer-based mental health
classifiers.

The codebase will be reviewed by:
- **Journal reviewers** assessing methodological soundness and reproducibility
- **Graduate admissions committees** at ETH Zürich, Oxford, and Cambridge
  evaluating research engineering quality

Every file you touch must reflect the standards of a first-author submission
to a Q1 biomedical informatics journal.

---

## 2. Research Setup (Read This Before Any Task)

### 2.1 Classification Task
Four-class mental health text classification:

| ID | Class       | Notes                                      |
|----|-------------|--------------------------------------------|
| 0  | normal      | No clinical signal                         |
| 1  | depression  | Majority class (~56.6% of training data)   |
| 2  | anxiety     | Minority class (~7%)                       |
| 3  | stress      | Minority class (~7%)                       |

### 2.2 Models

| Key            | HuggingFace ID                                   | Display Name      |
|----------------|--------------------------------------------------|-------------------|
| bert           | bert-base-uncased                                | BERT              |
| roberta        | roberta-base                                     | RoBERTa           |
| mentalbert     | j-hartmann/emotion-english-distilroberta-base    | DistilRoBERTa     |
| mentalroberta  | SamLowe/roberta-base-go_emotions                 | SamLowe-RoBERTa   |

### 2.3 Platforms (Source Datasets)

| Key     | Dataset                         | Role                    |
|---------|---------------------------------|-------------------------|
| kaggle  | Kaggle Mental Health corpus     | Training + within-test  |
| reddit  | GoEmotions (28-class, remapped) | Cross-platform test 1   |
| twitter | dair-ai/emotion (remapped)      | Cross-platform test 2   |

### 2.4 Label Remapping
All three datasets are harmonized into the 4-class schema above.
The canonical mappings live in `preprocess.py`. Do not redefine them elsewhere.

### 2.5 Directory Structure
```
project_root/
├── CLAUDE.md                  ← this file
├── configs/
│   └── config.yaml            ← single source of truth for all paths/hyperparams
├── src/
│   ├── utils.py               ← shared constants and helper functions (canonical)
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── fairness_audit.py
│   ├── shap_analysis.py
│   ├── sensitivity_analysis.py
│   ├── truncation_audit.py
│   ├── gpt_eval.py
│   ├── label_sensitivity_mappings_DE.py
│   ├── code_A1_di_eod_analysis.py
│   ├── code_A2_stress_attribution.py
│   ├── code_A3_A4_A6_ece_jaccard.py
│   ├── code_A4_patch_attribution.py
│   ├── code_A5_temperature_scaling.py
│   ├── fix_di_symmetric.py
│   ├── jaccard_full_analysis.py
│   └── perclass_ece_analysis.py
├── data/
│   └── splits/cross_platform/
│       ├── train.csv
│       ├── val.csv
│       ├── test_kaggle.csv
│       ├── test_reddit.csv
│       └── test_twitter.csv
└── outputs/
    ├── models/{bert,roberta,mentalbert,mentalroberta}/
    ├── results/
    │   ├── fairness/
    │   ├── attribution/
    │   └── shap/
    └── figures/
```

### 2.6 Canonical Prediction CSV Schema
Every `outputs/results/{model_key}_{platform}_predictions.csv` must have:
```
label, pred, prob_normal, prob_depression, prob_anxiety, prob_stress, correct
```

---

## 3. The Single Most Important Rule

**`src/utils.py` is the canonical source for all shared constants and utility
functions.** Every other script must import from it — never redefine these
in-place. If you are working on any script and notice it redefines something
that belongs in `utils.py`, move it there first, then import it.

---

## 4. What Belongs in `utils.py`

`utils.py` is currently empty. It must contain the following, and nothing else
should duplicate these:

```python
# Canonical constants
MODELS     = ["bert", "roberta", "mentalbert", "mentalroberta"]
PLATFORMS  = ["kaggle", "reddit", "twitter"]
CLASSES    = ["normal", "depression", "anxiety", "stress"]
CLASS_IDS  = {c: i for i, c in enumerate(CLASSES)}
PROB_COLS  = ["prob_normal", "prob_depression", "prob_anxiety", "prob_stress"]

MODEL_DISPLAY = {
    "bert":          "BERT",
    "roberta":       "RoBERTa",
    "mentalbert":    "DistilRoBERTa",
    "mentalroberta": "SamLowe-RoBERTa",
}

MODEL_HF_IDS = {
    "bert":          "bert-base-uncased",
    "roberta":       "roberta-base",
    "mentalbert":    "j-hartmann/emotion-english-distilroberta-base",
    "mentalroberta": "SamLowe/roberta-base-go_emotions",
}

PLATFORM_COLORS = {
    "kaggle":  "#3366CC",
    "reddit":  "#2DA44E",
    "twitter": "#E8A838",
}

# Shared loaders
def load_config(path="configs/config.yaml") -> dict
def load_predictions(model_key, platform, results_dir) -> pd.DataFrame | None
def find_platform_file(platform, data_dir) -> str | None
def get_model_checkpoint(model_key, models_dir) -> str | None

# Shared metrics
def compute_aggregate_ece(probs, labels, M=10) -> float
def compute_macro_auc(probs, labels) -> float
def bootstrap_ci(fn, probs, labels, n_boots=1000, **kwargs) -> tuple[float, float]

# Shared attribution
def compute_token_importance(model, tokenizer, texts, target_class_idx,
                             max_length=64, batch_size=16, device="cpu") -> dict
```

---

## 5. Known Redundancy Map

The following functions are **duplicated across multiple files** and must be
consolidated into `utils.py` before any new analysis is added:

| Function / Constant          | Duplicated In                                                             |
|------------------------------|---------------------------------------------------------------------------|
| `MODELS`, `PLATFORMS`, etc.  | Every analysis script                                                     |
| `MODEL_DISPLAY`              | Every analysis script                                                     |
| `load_predictions()`         | `fairness_audit.py`, `code_A1_di_eod_analysis.py`, `fix_di_symmetric.py`, `code_A5_temperature_scaling.py`, `perclass_ece_analysis.py` |
| `compute_aggregate_ece()`    | `fairness_audit.py`, `code_A3_A4_A6_ece_jaccard.py`, `perclass_ece_analysis.py`, `code_A5_temperature_scaling.py` |
| `compute_token_importance()` | `shap_analysis.py`, `code_A2_stress_attribution.py`, `code_A4_patch_attribution.py` |
| `find_platform_file()`       | `code_A2_stress_attribution.py`, `code_A4_patch_attribution.py`           |
| `get_path()` / config loading| `code_A2_stress_attribution.py`, `code_A4_patch_attribution.py`           |

**Rule**: When you encounter any of the above in a script you are editing,
replace it with an import from `utils.py`. Do not leave the duplicate in place.

---

## 6. Relationship Between Scripts (Execution Order)

```
preprocess.py
    └─→ train.py
            └─→ evaluate.py
                    └─→ fairness_audit.py
                    │       └─→ (produces fairness_audit_full.csv)
                    │
                    └─→ code_A4_patch_attribution.py   [run FIRST among A-scripts]
                            └─→ code_A1_di_eod_analysis.py  (uses fix_di_symmetric formula)
                            └─→ code_A2_stress_attribution.py
                            └─→ code_A3_A4_A6_ece_jaccard.py
                            └─→ code_A5_temperature_scaling.py
                            └─→ perclass_ece_analysis.py
                            └─→ jaccard_full_analysis.py
                            └─→ shap_analysis.py
                            └─→ sensitivity_analysis.py
                            └─→ truncation_audit.py
```

`fix_di_symmetric.py` is a **patch script** — its corrected `symmetric_di()`
formula must be backported into `code_A1_di_eod_analysis.py`. The standalone
fix script should then be deprecated with a docstring note.

---

## 7. Coding Standards (Non-Negotiable)

### 7.1 Style
- **Python 3.10+** — use `X | Y` union types, structural pattern matching where
  appropriate, and `match` statements over long `if-elif` chains.
- **PEP 8** strictly. Line limit: 88 characters (Black-compatible).
- **Type annotations on all function signatures** — parameters and return types.
- No bare `except:` clauses. Catch specific exceptions only.
- No magic numbers in function bodies — define named constants at module level
  or pull from `config.yaml`.

### 7.2 Docstrings
Every module, class, and public function must have a NumPy-style docstring:

```python
def compute_aggregate_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    M: int = 10,
) -> float:
    """
    Compute aggregate Expected Calibration Error (ECE).

    Uses max-probability confidence and argmax prediction, binned into
    M equal-width intervals. This is Equation 5 in the paper.

    Parameters
    ----------
    probs : np.ndarray of shape (n_samples, n_classes)
        Predicted probability distributions.
    labels : np.ndarray of shape (n_samples,)
        Integer ground-truth labels in [0, n_classes).
    M : int, optional
        Number of calibration bins. Default is 10.

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better. 0 = perfect calibration.

    Notes
    -----
    In imbalanced settings, aggregate ECE is dominated by the majority
    class. Use ``compute_perclass_ece`` for minority-class assessment.
    """
```

### 7.3 Config Discipline
- All paths, hyperparameters, thresholds, and seeds come from
  `configs/config.yaml` via `load_config()` in `utils.py`.
- Never hardcode paths like `"outputs/results"` in analysis scripts.
  Use `cfg["paths"]["results"]`.
- Never hardcode seeds outside of `config.yaml`.

### 7.4 Reproducibility
- Every script that uses randomness must call:
  ```python
  SEED = cfg["training"]["seed"]
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  ```
- Model inference must be done under `torch.no_grad()`.
- All file saves use `index=False` for CSVs.
- Figure DPI is always 150 minimum; use `bbox_inches="tight"`.

### 7.5 File Headers
Every `.py` file must begin with a module docstring in this format:

```python
"""
module_name.py
──────────────
One-sentence description of what this script does.

Longer description of methodology, paper reference (equation/section),
and what inputs it requires and outputs it produces.

Inputs
------
outputs/results/{model}_{platform}_predictions.csv
    Per-sample predictions with columns: label, pred, prob_*, correct.

Outputs
-------
outputs/results/fairness/some_result.csv
outputs/figures/some_figure.png

Usage
-----
Run from the repository root:
    python src/module_name.py

Dependencies
------------
Requires evaluate.py to have been run first.
"""
```

### 7.6 Empty Files
- `utils.py` — populate with the contents specified in Section 4.
- `gpt_eval.py` — if GPT-4 evaluation is not yet implemented, add a module
  docstring and a single `NotImplementedError` stub with a clear comment
  explaining what the script should do when complete. Never leave a file
  completely empty.

---

## 8. Fairness Metrics — Canonical Definitions

These are the exact formulations used in the paper. Do not deviate.

### Disparate Impact (Symmetric — Post fix_di_symmetric.py)
```
DI_c = min(P(ŷ=c | G=A) / P(ŷ=c | G=B),
           P(ŷ=c | G=B) / P(ŷ=c | G=A))
```
Range: (0, 1]. DI < 0.80 = four-fifths rule violation. DI < 0.50 = severe.
Reference platform is always **Kaggle**.

### Equalized Odds Difference
```
EOD_c = |TPR_c(ref) - TPR_c(target)|
```
EOD = 0 indicates perfect equalized odds. Report absolute value only.

### Expected Calibration Error — Aggregate (Equation 5)
```
ECE = Σ_m (|B_m| / n) · |acc(B_m) - conf(B_m)|
```
Default M=10 bins. Confidence = max predicted probability.

### Expected Calibration Error — Per-Class (Equation 5b)
```
ECE_c = Σ_m (|B_m^c| / n) · |acc(B_m^c) - conf(B_m^c)|
```
Where bin membership is based on P(class=c) and accuracy is 1{y_i = c}.

---

## 9. Figure Standards

All matplotlib figures must:
- Use `matplotlib.use("Agg")` (non-interactive backend)
- Have `fontsize=11` for titles and `fontsize=10` for axis labels
- Use `fontweight="bold"` for main titles
- Include `plt.tight_layout()` before saving
- Be saved at `dpi=150` minimum with `bbox_inches="tight"`
- Have descriptive titles that include the key finding, not just the metric name

Color palette for platforms (canonical — from `utils.py`):
```python
PLATFORM_COLORS = {"kaggle": "#3366CC", "reddit": "#2DA44E", "twitter": "#E8A838"}
```

Heatmaps for DI use `cmap="RdYlGn"`, `vmin=0.0`, `vmax=1.0`, `center=0.80`.
Heatmaps for EOD use `cmap="YlOrRd"`, `vmin=0.0`, `vmax=0.8`.

---

## 10. What NOT To Do

- **Never redefine `MODEL_DISPLAY`, `MODELS`, `PLATFORMS`, or `CLASSES`** in an
  analysis script. Import them from `utils.py`.
- **Never use the asymmetric DI formula** `rate_tgt / rate_ref`. Always use the
  symmetric `min(a/b, b/a)` formulation.
- **Never print results without also saving them** to a CSV in the appropriate
  output directory.
- **Never leave `TODO` or `FIXME` comments in committed code** — resolve them or
  convert to a GitHub issue reference.
- **Never use `pd.DataFrame.append()`** — it is deprecated. Use
  `pd.concat([df, new_row_df], ignore_index=True)`.
- **Never suppress all warnings** with `warnings.filterwarnings("ignore")`
  without a comment explaining what is being suppressed and why.
- **Never hardcode the number of classes (4)** — use `len(CLASSES)` or
  `NUM_LABELS` from config.

---

## 11. Quick Reference — Paper Equations

| Equation | Description                        | Script                          |
|----------|------------------------------------|----------------------------------|
| Eq. 5    | Aggregate ECE                      | `fairness_audit.py`, `utils.py`  |
| Eq. 5b   | Per-class ECE (one-vs-rest)        | `perclass_ece_analysis.py`       |
| Eq. 6    | Symmetric DI (post-fix)            | `code_A1_di_eod_analysis.py`     |
| Eq. 7    | Equalized Odds Difference          | `code_A1_di_eod_analysis.py`     |
| Eq. 8–9  | Gradient saliency attribution      | `code_A2_stress_attribution.py`  |
| —        | Temperature scaling (NLL opt.)     | `code_A5_temperature_scaling.py` |

---

## 12. When You Are Unsure

If a task is ambiguous, always choose the option that:
1. Keeps the code more readable over more compact
2. Is more reproducible (explicit seeds, saved outputs)
3. Is more consistent with what the rest of the codebase already does
4. Reduces redundancy rather than increases it

When modifying an analysis script, check whether the same computation exists
elsewhere first. If it does, refactor — do not duplicate.
