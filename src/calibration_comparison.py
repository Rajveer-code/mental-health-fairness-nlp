"""
calibration_comparison.py
─────────────────────────
Fine-tuning baseline vs temperature scaling comparison (New Table 10).

Addresses GAP 2 from the JBI revision: temperature scaling uses ~288 labelled
examples from the target platform (10% of the test set), but a reviewer will
immediately ask whether fine-tuning on those same examples would do better.

For each model × target platform (Reddit, Twitter), this script evaluates:

  (A) Baseline — the original zero-shot cross-platform model (seed=42)
  (B) Temperature Scaling — post-hoc recalibration using the 10% split
      (this replicates Table 7 but on the same 90% holdout as the FT baseline,
       ensuring a fair comparison between the two methods)
  (C) Fine-Tuning Baseline — 3 epochs of additional fine-tuning on the 10%
      split, evaluated on the remaining 90%

The comparison answers: are the 288 labelled target-domain examples more
valuable as a calibration signal (temperature scaling) or as additional
training signal (fine-tuning)?

Inputs
------
outputs/models/{model}/
    Seed=42 checkpoint.
outputs/results/{model}_{platform}_predictions.csv
    Seed=42 predictions with logit columns (from evaluate.py).
data/splits/cross_platform/test_{reddit,twitter}.csv

Outputs
-------
outputs/results/calibration_comparison.csv
    Columns: model, platform, baseline_auc, baseline_ece,
             tempscale_ece, tempscale_auc_delta,
             finetuned_auc, finetuned_ece, ft_auc_gain_vs_baseline,
             ft_vs_ts_verdict

outputs/figures/figure_calibration_comparison.png
    Grouped bar chart: ECE and AUC for all three conditions per model/platform.

Usage
-----
Run from the repository root:
    python src/calibration_comparison.py

Dependencies
------------
Requires evaluate.py to have been run (logit columns in prediction CSVs).
Requires outputs/models/{model}/ checkpoints.
"""

import os
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed as hf_set_seed,
)
from scipy.optimize import minimize_scalar
from scipy.special import softmax as sp_softmax
from sklearn.metrics import log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    MODELS,
    MODEL_DISPLAY,
    PROB_COLS,
    load_config,
    load_predictions,
    compute_aggregate_ece,
    compute_macro_auc,
    get_model_checkpoint,
)

# ── Config ─────────────────────────────────────────────────────────────────────

cfg = load_config()

MODELS_DIR  = cfg["paths"]["models"]
RESULTS_DIR = cfg["paths"]["results"]
FIGURES_DIR = cfg["paths"]["figures"]
SPLITS_DIR  = "data/splits/cross_platform"

os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler        = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

EVAL_PLATFORMS = ["reddit", "twitter"]
CAL_FRAC       = 0.10    # 10% calibration / fine-tuning split
FT_EPOCHS      = 3       # fine-tuning epochs (no early stopping — small dataset)
FT_LR          = 2e-5
SEED           = cfg["training"]["seed"]  # 42
MAX_LEN        = cfg["training"]["max_length"]
BATCH_SIZE     = cfg["training"]["batch_size"]
NUM_LABELS     = 4
LABEL_NAMES    = ["normal", "depression", "anxiety", "stress"]

LOGIT_COLS = [
    "logit_normal", "logit_depression", "logit_anxiety", "logit_stress",
]


# ── Dataset ────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Minimal Dataset for fine-tuning and inference."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int) -> None:
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Temperature scaling helpers ────────────────────────────────────────────────

def _find_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T* minimising NLL on calibration logits."""
    def nll(T: float) -> float:
        if T <= 0:
            return 1e10
        return float(log_loss(labels, sp_softmax(logits / T, axis=1)))
    result = minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    return float(result.x)


def _apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Return calibrated probabilities: softmax(logits / T)."""
    return sp_softmax(logits / T, axis=1)


# ── Stratified split helper ────────────────────────────────────────────────────

def _stratified_split(
    df: pd.DataFrame,
    cal_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (calibration, evaluation) subsets, stratified by label.

    Parameters
    ----------
    df : pd.DataFrame
        Full test set with a "label" column.
    cal_frac : float
        Fraction to use for calibration / fine-tuning.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (cal_df, eval_df)
    """
    rng    = np.random.default_rng(seed)
    cal_idx: list[int] = []

    for label_val in sorted(df["label"].unique()):
        label_idx = df.index[df["label"] == label_val].tolist()
        n_cal     = max(1, int(len(label_idx) * cal_frac))
        chosen    = rng.choice(label_idx, size=n_cal, replace=False)
        cal_idx.extend(chosen.tolist())

    cal_df  = df.loc[cal_idx].reset_index(drop=True)
    eval_df = df.drop(index=cal_idx).reset_index(drop=True)
    return cal_df, eval_df


# ── Inference helper ───────────────────────────────────────────────────────────

def _run_inference(
    model: AutoModelForSequenceClassification,
    tokenizer,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a dataframe.

    Returns
    -------
    tuple
        (probs, logits, preds) — all numpy arrays.
    """
    model.eval()
    ds     = TextDataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_probs, all_logits = [], []
    with torch.no_grad():
        for batch in loader:
            out     = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            logits  = out.logits
            probs   = F.softmax(logits, dim=-1)
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    probs_arr  = np.array(all_probs)
    logits_arr = np.array(all_logits)
    preds_arr  = probs_arr.argmax(axis=1)
    return probs_arr, logits_arr, preds_arr


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def _finetune_on_split(
    ckpt_dir: str,
    cal_df: pd.DataFrame,
    model_key: str,
    platform: str,
) -> AutoModelForSequenceClassification:
    """
    Fine-tune a pre-trained checkpoint for FT_EPOCHS on a small target-domain
    split. No early stopping (dataset too small for a held-out dev set).
    Returns the model at the last epoch.

    Parameters
    ----------
    ckpt_dir : str
        Path to the base checkpoint to fine-tune.
    cal_df : pd.DataFrame
        Fine-tuning training data (the 10% split).
    model_key : str
        Used only for progress printing.
    platform : str
        Used only for progress printing.

    Returns
    -------
    AutoModelForSequenceClassification
        Fine-tuned model (in eval mode, on DEVICE).
    """
    hf_set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    ).to(DEVICE)

    ds     = TextDataset(cal_df, tokenizer, MAX_LEN)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer    = AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
    total_steps  = len(loader) * FT_EPOCHS
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(
        f"    Fine-tuning {MODEL_DISPLAY[model_key]} on {platform} "
        f"({len(cal_df)} samples × {FT_EPOCHS} epochs) …"
    )

    model.train()
    for epoch in range(1, FT_EPOCHS + 1):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"      FT epoch {epoch}", leave=False):
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    out  = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

        print(f"      Epoch {epoch}/{FT_EPOCHS}  loss={epoch_loss/len(loader):.4f}")

    model.eval()
    return model


# ── Main analysis ──────────────────────────────────────────────────────────────

def run_calibration_comparison() -> pd.DataFrame:
    """
    Run the full calibration comparison for all models × platforms.

    Returns
    -------
    pd.DataFrame
        Table 10 data.
    """
    rows: list[dict] = []

    for model_key in MODELS:
        display = MODEL_DISPLAY[model_key]
        print(f"\n{'='*60}")
        print(f"  Model: {display}")
        print(f"{'='*60}")

        ckpt = get_model_checkpoint(model_key, MODELS_DIR)
        if ckpt is None:
            print(f"  SKIP: No checkpoint for {model_key}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(ckpt)

        for platform in EVAL_PLATFORMS:
            print(f"\n  Platform: {platform.upper()}")

            # ── Load prediction CSV (must have logit columns) ──────────────
            pred_df = load_predictions(model_key, platform, RESULTS_DIR)
            if pred_df is None:
                continue

            missing_logits = [c for c in LOGIT_COLS if c not in pred_df.columns]
            if missing_logits:
                print(
                    f"  ERROR: Logit columns missing. Re-run evaluate.py.\n"
                    f"  Missing: {missing_logits}"
                )
                continue

            # ── Load full test CSV (needed for fine-tuning) ────────────────
            test_csv = os.path.join(SPLITS_DIR, f"test_{platform}.csv")
            if not os.path.exists(test_csv):
                print(f"  ERROR: {test_csv} not found")
                continue
            full_df = pd.read_csv(test_csv)

            # ── Stratified 10/90 split ─────────────────────────────────────
            cal_df, eval_df = _stratified_split(full_df, CAL_FRAC, SEED)
            n_cal  = len(cal_df)
            n_eval = len(eval_df)
            print(f"    Cal: {n_cal}  Eval: {n_eval}")

            # ── Align prediction CSV with eval_df ──────────────────────────
            # pred_df has the same row order as full_df; extract eval rows
            eval_idx         = eval_df.index  # indices into full_df
            logits_full      = pred_df[LOGIT_COLS].values.astype(float)
            probs_full       = pred_df[PROB_COLS].values.astype(float)
            labels_full      = full_df["label"].values.astype(int)

            # Reconstruct cal/eval aligned to full_df indices
            all_indices      = np.arange(len(full_df))
            cal_mask         = np.isin(all_indices, cal_df.index)
            eval_mask        = ~cal_mask

            logits_cal       = logits_full[cal_mask]
            labels_cal       = labels_full[cal_mask]
            logits_eval      = logits_full[eval_mask]
            probs_eval_base  = probs_full[eval_mask]
            labels_eval      = labels_full[eval_mask]

            # ── Baseline (no recalibration) ────────────────────────────────
            auc_base = compute_macro_auc(probs_eval_base, labels_eval)
            ece_base = compute_aggregate_ece(probs_eval_base, labels_eval)

            # ── Temperature scaling ────────────────────────────────────────
            T_star         = _find_temperature(logits_cal, labels_cal)
            probs_scaled   = _apply_temperature(logits_eval, T_star)
            auc_ts         = compute_macro_auc(probs_scaled, labels_eval)
            ece_ts         = compute_aggregate_ece(probs_scaled, labels_eval)
            auc_ts_delta   = round(auc_ts - auc_base, 4)

            print(
                f"    Baseline  AUC={auc_base:.4f}  ECE={ece_base:.4f}\n"
                f"    TempScale T*={T_star:.3f}  ECE={ece_ts:.4f}  "
                f"ΔAUC={auc_ts_delta:+.4f}"
            )

            # ── Fine-tuning baseline ───────────────────────────────────────
            ft_model = _finetune_on_split(ckpt, cal_df, model_key, platform)

            eval_df_subset = full_df.iloc[eval_mask].reset_index(drop=True)
            probs_ft, _, _ = _run_inference(ft_model, tokenizer, eval_df_subset)
            auc_ft         = compute_macro_auc(probs_ft, labels_eval)
            ece_ft         = compute_aggregate_ece(probs_ft, labels_eval)
            auc_ft_gain    = round(auc_ft - auc_base, 4)

            print(
                f"    FineTune  AUC={auc_ft:.4f}  ECE={ece_ft:.4f}  "
                f"AUC gain vs baseline={auc_ft_gain:+.4f}"
            )

            # Verdict for manuscript discussion
            if auc_ft_gain >= 0.05:
                verdict = (
                    "Fine-tuning substantially recovers AUC "
                    f"(+{auc_ft_gain:.3f}); prefer domain adaptation."
                )
            elif auc_ft_gain >= 0.01:
                verdict = (
                    "Fine-tuning modestly improves AUC "
                    f"(+{auc_ft_gain:.3f}); temperature scaling sufficient "
                    "for calibration."
                )
            else:
                verdict = (
                    "Fine-tuning does not substantially recover AUC "
                    f"({auc_ft_gain:+.3f}); domain gap requires more data."
                )
            print(f"    Verdict: {verdict}")

            rows.append({
                "model":                  model_key,
                "model_display":          display,
                "platform":               platform,
                "n_cal":                  n_cal,
                "n_eval":                 n_eval,
                "baseline_auc":           round(auc_base, 4),
                "baseline_ece":           round(ece_base, 4),
                "tempscale_temperature":  round(T_star,   4),
                "tempscale_ece":          round(ece_ts,   4),
                "tempscale_auc_delta":    auc_ts_delta,
                "finetuned_auc":          round(auc_ft,   4),
                "finetuned_ece":          round(ece_ft,   4),
                "ft_auc_gain_vs_baseline": auc_ft_gain,
                "ft_vs_ts_verdict":       verdict,
            })

            del ft_model
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_comparison(df: pd.DataFrame) -> None:
    """
    Figure 8 (new): Grouped bar chart showing ECE and AUC for baseline,
    temperature scaling, and fine-tuning across all models and platforms.
    """
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_configs = [
        ("reddit",  "ECE",  "ece"),
        ("twitter", "ECE",  "ece"),
        ("reddit",  "AUC",  "auc"),
        ("twitter", "AUC",  "auc"),
    ]

    for ax, (platform, metric_label, metric_key) in zip(
        axes.flatten(), plot_configs
    ):
        sub = df[df["platform"] == platform]
        if sub.empty:
            ax.set_visible(False)
            continue

        model_labels = [MODEL_DISPLAY.get(m, m) for m in sub["model"]]
        x     = np.arange(len(model_labels))
        width = 0.26

        baseline_vals = sub[f"baseline_{metric_key}"].values
        ts_vals       = sub[f"tempscale_{metric_key}"].values \
                        if f"tempscale_{metric_key}" in sub.columns \
                        else sub["baseline_auc"].values  # AUC unchanged by TS
        ft_vals       = sub[f"finetuned_{metric_key}"].values

        # For AUC: temperature scaling doesn't change AUC (monotone transform)
        if metric_key == "auc":
            ts_vals = baseline_vals

        bars1 = ax.bar(x - width, baseline_vals, width,
                       label="Baseline", color="#5B8DB8", alpha=0.85,
                       edgecolor="white")
        bars2 = ax.bar(x,          ts_vals,       width,
                       label="Temperature Scaling", color="#E07B39", alpha=0.85,
                       edgecolor="white")
        bars3 = ax.bar(x + width,  ft_vals,        width,
                       label="Fine-Tuned (3 epochs)", color="#3DAA6E", alpha=0.85,
                       edgecolor="white")

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(
            f"{platform.capitalize()} — {metric_label}", fontweight="bold"
        )
        ax.legend(fontsize=8)

        if metric_key == "ece":
            ax.axhline(0.10, color="grey", linestyle=":", alpha=0.6)
        else:
            ax.axhline(0.70, color="grey", linestyle=":", alpha=0.6,
                       label="0.70 reference")
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Calibration Comparison: Temperature Scaling vs Fine-Tuning Baseline\n"
        "Both methods trained on identical 10% stratified calibration split",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "figure_calibration_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Calibration Comparison: Temperature Scaling vs Fine-Tuning Baseline")
    print("=" * 65)
    print(f"Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"FT epochs: {FT_EPOCHS}  LR={FT_LR}  Cal fraction={CAL_FRAC:.0%}")
    print()

    df = run_calibration_comparison()

    if df.empty:
        print("\nERROR: No results — check prediction CSVs and model checkpoints.")
        return

    # Save CSV
    out_csv = os.path.join(RESULTS_DIR, "calibration_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nTable 10 saved: {out_csv}")

    # Plot
    plot_comparison(df)

    # Print Table 10 for manuscript
    print(f"\n{'='*75}")
    print("NEW TABLE 10 — Calibration Comparison")
    print(f"{'='*75}")
    print(
        f"{'Model':<22} {'Platform':<10} "
        f"{'Base AUC':>9} {'Base ECE':>9} "
        f"{'TS ECE':>8} {'FT AUC':>8} {'FT ECE':>8} {'FT gain':>9}"
    )
    print("-" * 75)
    for _, row in df.iterrows():
        print(
            f"  {row['model_display']:<20} {row['platform']:<10} "
            f"{row['baseline_auc']:>9.4f} {row['baseline_ece']:>9.4f} "
            f"{row['tempscale_ece']:>8.4f} {row['finetuned_auc']:>8.4f} "
            f"{row['finetuned_ece']:>8.4f} {row['ft_auc_gain_vs_baseline']:>+9.4f}"
        )

    print(f"\n{'='*65}")
    print("DONE. Table 10 ready for manuscript insertion.")
    print("Next step: python src/compile_tables.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
