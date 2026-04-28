# CPFE Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-seed training, Integrated Gradients attribution, fine-tuning baseline comparison, updated figures/tables, and submission-ready manuscript to the existing CPFE codebase.

**Architecture:** Six new standalone scripts in `src/` extend the existing pipeline without modifying any validated code. A root-level `run_pipeline.py` orchestrates the full sequence. All new scripts follow the established patterns from `CLAUDE.md` and import constants from `src/utils.py`.

**Tech Stack:** PyTorch, HuggingFace Transformers, Captum (IG), python-docx, openpyxl, matplotlib/seaborn, scipy, scikit-learn, tqdm

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `src/train_multiseed.py` | CREATE | Train 4 models × 5 seeds; evaluate each on 3 platforms; aggregate to mean ± std |
| `src/integrated_gradients.py` | CREATE | Captum LayerIG attribution; Table 8 update with both Grad-Sal J and IG J columns |
| `src/calibration_comparison.py` | CREATE | Fine-tuning baseline (3 epochs on 10% split) vs temperature scaling; new Table 10 |
| `src/compile_tables.py` | CREATE | Read all result CSVs → `outputs/results/all_tables.xlsx` (Tables 2–10, mean ± std) |
| `src/generate_figures_v2.py` | CREATE | Regenerate Figures 1–9 from multi-seed data; save to `outputs/figures/v2/` |
| `src/rebuild_manuscript.py` | CREATE | Read existing .docx; inject multi-seed text, Table 10, IG paragraph, updated figures |
| `run_pipeline.py` | CREATE | Root orchestrator: steps 1–6 in order, dependency checks, tqdm progress |

---

## Task 1: `src/train_multiseed.py`

**Files:** Create `src/train_multiseed.py`

Outputs:
- `outputs/models/{model}_seed{seed}/` — checkpoint per (model, seed) for seeds 0,1,7,123
- `outputs/results/multiseed/{model}_{platform}_seed{seed}_predictions.csv`
- `outputs/results/multiseed/multiseed_results.csv`

Logic:
1. Seeds = [42, 0, 1, 7, 123]. For seed=42, existing checkpoint is at `outputs/models/{model}/`.
2. For seeds [0,1,7,123]: check `outputs/models/{model}_seed{seed}/` — skip if exists (resumable).
3. Training config matches `train.py` exactly: AdamW lr=2e-5, wd=0.01, 10% warmup, clip=1.0, 5 epochs, AMP on CUDA.
4. After training, evaluate each checkpoint on kaggle/reddit/twitter test sets — save per-seed prediction CSVs with label, pred, prob_*, logit_*, correct columns.
5. Compute ECE and macro-AUC per seed. Aggregate across 5 seeds: mean ± std for accuracy, f1_macro, f1_weighted, auc_macro, ece per (model, platform).
6. Save `multiseed_results.csv` with both per-seed rows and summary rows.

---

## Task 2: `src/integrated_gradients.py`

**Files:** Create `src/integrated_gradients.py`

Outputs:
- `outputs/results/ig_attribution_results.csv`
- `outputs/results/ig_table8_update.csv`

Logic:
1. Use `captum.attr.LayerIntegratedGradients` on the embedding layer (`model.bert.embeddings.word_embeddings` for BERT, `model.roberta.embeddings.word_embeddings` for RoBERTa variants).
2. Baseline: zero-embedding vector (standard NLP baseline).
3. n_steps=50, internal_batch_size=4.
4. For each model × platform × class: sample n=200 texts, compute |IG_i| summed across embedding dims, aggregate by word (min 3 occurrences), extract top-K for K ∈ {5,10,15,20}.
5. Load existing gradient saliency top-K from `outputs/results/shap/{model}_{platform}_{class}_top_words.csv`.
6. Compute Jaccard for platform pairs (kaggle→reddit, kaggle→twitter, reddit→twitter).
7. `ig_table8_update.csv` columns: model, class, pair, k, jaccard_gradsal, jaccard_ig, agreement ("Consistent" if both < 0.10, "Disagree" otherwise).

---

## Task 3: `src/calibration_comparison.py`

**Files:** Create `src/calibration_comparison.py`

Outputs:
- `outputs/results/calibration_comparison.csv`

Logic (seed=42 models only):
1. For each model × target platform (reddit, twitter):
   a. Load logits + labels from existing `outputs/results/{model}_{platform}_predictions.csv`.
   b. Stratified 10/90 split (same RNG seed as `code_A5_temperature_scaling.py`).
   c. Part A — Temperature scaling: find T* via NLL minimisation on 10%; apply to 90%; record ECE before/after, AUC before/after.
   d. Part B — Fine-tuning baseline: load the pre-trained model checkpoint from `outputs/models/{model}/`; fine-tune on 10% split texts for 3 epochs (AdamW lr=2e-5, no early stopping); evaluate on 90%; record AUC and ECE.
2. `calibration_comparison.csv` columns: model, platform, baseline_auc, baseline_ece, tempscale_ece, tempscale_auc_delta, finetuned_auc, finetuned_ece, ft_vs_ts_auc_delta.

---

## Task 4: `src/compile_tables.py`

**Files:** Create `src/compile_tables.py`

Outputs:
- `outputs/results/all_tables.xlsx` (one sheet per table, Tables 2–10)

Logic:
- Table 2 (within-platform): read multiseed_results.csv, filter platform==kaggle, format "mean ± std".
- Table 3 (cross-platform AUC): read multiseed_results.csv, all platforms, AUC + ECE columns.
- Table 4 (per-class AUC): read multiseed_results.csv per-class AUC columns.
- Table 5 (pairwise stats): read `manuscript_inputs/fairness/pairwise_auc_comparisons.csv` as-is.
- Table 6 (DI/EOD): read `manuscript_inputs/fairness/di_eod_table.csv` as-is.
- Table 7 (temperature scaling): read `manuscript_inputs/fairness/temperature_scaling_results.csv`.
- Table 8 (Jaccard + IG): read `outputs/results/ig_table8_update.csv`, K=10 rows.
- Table 9 (sensitivity): read `manuscript_inputs/fairness/sensitivity_drops_all_mappings.csv`.
- Table 10 (calibration comparison): read `outputs/results/calibration_comparison.csv`.

---

## Task 5: `src/generate_figures_v2.py`

**Files:** Create `src/generate_figures_v2.py`

Outputs: `outputs/figures/v2/figure{N}.png` + `.pdf` (300 DPI) for N=1..9

Nine figures regenerated with multi-seed data:
1. CPFE Framework Diagram (schematic, no data needed)
2. Cross-platform degradation (3-panel: F1-macro, AUC, ECE with mean ± std error bars)
3. Reliability diagrams (4×3 grid, ECE annotations with mean ± std)
4. Per-class F1 heatmap (RdYlGn, rows = model×platform, cols = classes)
5a-5f. Token attribution figures (gradient saliency + IG side-by-side, top-15 tokens)
6. Jaccard stability heatmap (two methods side-by-side)
7. Sensitivity analysis (AUC degradation across mappings A-E)
8. Temperature scaling + fine-tuning comparison (new figure, from calibration_comparison.csv)
9. DI heatmap (RdYlGn, 0.80 and 0.50 boundary lines)

---

## Task 6: `src/rebuild_manuscript.py`

**Files:** Create `src/rebuild_manuscript.py`

Outputs: `CPFE_submission_ready.docx`

Logic:
1. Read `CPFE_Manuscript_Final_Submission (1).docx` as base.
2. Inject multi-seed language into Abstract (replace single-value AUC with mean ± std).
3. Add Section 4.5 (Multi-Seed Training Protocol) and 4.6 (Fine-Tuning Baseline) paragraphs.
4. Update Section 5.4 to reference Table 10 and discuss fine-tuning vs temperature scaling findings.
5. Add IG paragraph to Section 5.7 (Attribution).
6. Update Limitations: mark three items as "addressed."
7. Add Captum citation to References.
8. Save to `CPFE_submission_ready.docx`.

---

## Task 7: `run_pipeline.py`

**Files:** Create `run_pipeline.py` (repository root)

Orchestrates all six steps with dependency checking, ETA estimates, and a final validation checklist printed at completion. Accepts `--from-step N` to resume mid-pipeline.
