"""
code_A5_temperature_scaling.py
──────────────────────────────
CPFE Axis 2: platform-specific temperature scaling recalibration.

Temperature T is optimised by minimising NLL on a 10% stratified
calibration split drawn from each deployment platform, then applied
to the remaining 90% evaluation split. Operates on raw pre-softmax
logits (columns logit_* in prediction CSVs), not softmax probabilities:
    p_T = softmax(z / T)
This is the correct implementation per Guo et al. (2017).

Inputs
------
outputs/results/{model}_{platform}_predictions.csv
    Must include logit_normal, logit_depression, logit_anxiety,
    logit_stress columns (written by evaluate.py).

Outputs
-------
outputs/results/fairness/temperature_scaling_results.csv
outputs/figures/figure_recalibration.png

Usage
-----
Run from the repository root:
    python src/code_A5_temperature_scaling.py

Dependencies
------------
Requires evaluate.py to have been run first.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax as sp_softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from utils import (
    MODELS, MODEL_DISPLAY, PROB_COLS,
    load_config, load_predictions,
    compute_aggregate_ece, compute_macro_auc,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────

cfg = load_config()

RESULTS_DIR  = cfg["paths"]["results"]
FIGURES_DIR  = cfg["paths"]["figures"]
FAIRNESS_DIR = os.path.join(RESULTS_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)

EVAL_PLATFORMS = ["reddit", "twitter"]
CAL_FRAC       = 0.10
SEED           = cfg["training"]["seed"]

# Logit column names written by the fixed evaluate.py
LOGIT_COLS = [
    "logit_normal",
    "logit_depression",
    "logit_anxiety",
    "logit_stress",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_predictions_with_logits(model_key: str, platform: str) -> pd.DataFrame | None:
    """
    Load prediction CSV and verify it contains raw logit columns.
    If logit columns are missing the user must re-run evaluate.py [FIXED].
    """
    df = load_predictions(model_key, platform, RESULTS_DIR)
    if df is None:
        return None

    missing = [c for c in LOGIT_COLS if c not in df.columns]
    if missing:
        print(
            f"\n  ERROR: Logit columns missing in {model_key}_{platform}_predictions.csv\n"
            f"  Missing columns: {missing}\n"
            f"  You must re-run evaluate.py [FIXED] to generate logit columns.\n"
            f"  Run: python src/evaluate.py\n"
            f"  Then re-run this script.\n"
        )
        return None

    return df


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """
    Scale raw logits by temperature T and return calibrated probabilities.
    This is the CORRECT temperature scaling operation:
        p_T = softmax(logits / T)
    """
    return sp_softmax(logits / T, axis=1)


def find_optimal_temperature(logits_cal: np.ndarray,
                             labels_cal: np.ndarray) -> float:
    """
    Find temperature T* that minimises NLL on the calibration split.
    Uses bounded scalar optimisation over T ∈ [0.1, 10.0].

    Parameters
    ----------
    logits_cal : np.ndarray, shape (n_cal, 4)
        Raw pre-softmax logits for the calibration samples.
    labels_cal : np.ndarray, shape (n_cal,)
        Integer class labels for the calibration samples.
    """
    def nll(T: float) -> float:
        if T <= 0:
            return 1e10
        cal_probs = apply_temperature(logits_cal, T)
        return float(log_loss(labels_cal, cal_probs))

    result = minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    return float(result.x)


# ── Main Analysis ─────────────────────────────────────────────────────────────

def run_temperature_scaling() -> pd.DataFrame:
    rows = []
    rng  = np.random.default_rng(SEED)

    for model_key in MODELS:
        display = MODEL_DISPLAY[model_key]
        print(f"\n{'='*55}\nModel: {display}\n{'='*55}")

        for platform in EVAL_PLATFORMS:
            df = _load_predictions_with_logits(model_key, platform)
            if df is None:
                continue

            n = len(df)

            # Raw logits and probabilities
            logits = df[LOGIT_COLS].values.astype(float)   # (n, 4)
            probs  = df[PROB_COLS].values.astype(float)    # (n, 4)
            labels = df["label"].values.astype(int)

            # ── Stratified calibration split (10% cal, 90% eval) ──────────
            # Stratify by label so every class is represented in the cal set,
            # which is critical for calibration on minority classes.
            idx_cal, idx_eval = train_test_split(
                np.arange(n),
                test_size=1.0 - CAL_FRAC,
                stratify=labels,
                random_state=SEED,
            )

            logits_cal  = logits[idx_cal];  labels_cal  = labels[idx_cal]
            logits_eval = logits[idx_eval]; labels_eval = labels[idx_eval]
            probs_eval  = probs[idx_eval]

            # ── Before recalibration ──────────────────────────────────────
            ece_before = compute_aggregate_ece(probs_eval, labels_eval)
            auc_before = compute_macro_auc(probs_eval, labels_eval)

            # ── Find T* on calibration set ────────────────────────────────
            T_star = find_optimal_temperature(logits_cal, labels_cal)

            # ── Apply temperature scaling to eval logits ──────────────────
            probs_scaled = apply_temperature(logits_eval, T_star)

            # ── After recalibration ───────────────────────────────────────
            ece_after = compute_aggregate_ece(probs_scaled, labels_eval)
            auc_after = compute_macro_auc(probs_scaled, labels_eval)

            ece_reduction = round((ece_before - ece_after) / ece_before * 100, 1) \
                            if ece_before > 0 else float("nan")
            auc_change    = round(auc_after - auc_before, 4)

            print(f"\n  {platform.upper():10} n_cal={n_cal}  n_eval={len(idx_eval)}")
            print(f"    T* = {T_star:.4f}  (optimised on actual logits)")
            print(f"    ECE: {ece_before:.4f} → {ece_after:.4f}  "
                  f"({ece_reduction:.1f}% reduction)")
            print(f"    AUC: {auc_before:.4f} → {auc_after:.4f}  "
                  f"(ΔAUC = {auc_change:+.4f})")

            rows.append({
                "model":              model_key,
                "platform":           platform,
                "n_cal":              n_cal,
                "n_eval":             len(idx_eval),
                "temperature":        round(T_star, 4),
                "ece_before":         round(ece_before, 4),
                "ece_after":          round(ece_after, 4),
                "ece_reduction_pct":  ece_reduction,
                "auc_before":         round(auc_before, 4),
                "auc_after":          round(auc_after, 4),
                "auc_change":         auc_change,
            })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_recalibration(df: pd.DataFrame):
    """
    Before/after ECE bar chart with AUC overlay.
    Figure 8: shows ECE improvement; AUC unchanged (monotonic rescaling
    cannot alter ranking).
    """
    if df.empty:
        print("  No data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, platform in enumerate(["reddit", "twitter"]):
        ax  = axes[ax_idx]
        sub = df[df["platform"] == platform]

        if sub.empty:
            ax.set_title(f"{platform.capitalize()}: No data")
            continue

        model_labels = [MODEL_DISPLAY[m] for m in sub["model"].tolist()]
        x     = np.arange(len(model_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, sub["ece_before"].values, width,
                       label="ECE before recalibration",
                       color="#FF6B6B", alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, sub["ece_after"].values, width,
                       label="ECE after temp. scaling (correct)",
                       color="#4ECDC4", alpha=0.85, edgecolor="white")

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(x - width/2, sub["auc_before"].values, "b^--",
                 label="AUC before", markersize=8, linewidth=1.5)
        ax2.plot(x + width/2, sub["auc_after"].values, "bs-",
                 label="AUC after", markersize=8, linewidth=1.5)
        ax2.set_ylabel("Macro AUC", color="blue", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.set_ylim(0.4, 1.05)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Expected Calibration Error (ECE)")
        ax.set_ylim(0, 0.70)
        ax.axhline(0.10, color="gray", linestyle=":", alpha=0.6, linewidth=1)
        ax.set_title(
            f"Recalibration — {platform.capitalize()}\n"
            f"T* optimised on raw pre-softmax logits (10% stratified cal split)",
            fontweight="bold"
        )

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Temperature Scaling Recalibration (Equation 6)\n"
        "T* fitted on 10% stratified calibration split from each deployment platform",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_recalibration.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


def print_paper_summary(df: pd.DataFrame):
    if df.empty:
        return
    avg_ece_before   = df["ece_before"].mean()
    avg_ece_after    = df["ece_after"].mean()
    avg_reduction    = df["ece_reduction_pct"].mean()
    avg_auc_change   = df["auc_change"].mean()

    print(f"\n{'='*65}")
    print("TEMPERATURE SCALING SUMMARY (Section 5.6)")
    print(f"{'='*65}")
    print(f"""
We validated platform-specific temperature scaling as a post-hoc
recalibration strategy. Temperature T was optimised via NLL minimisation
on a 10% stratified calibration split drawn from each deployment platform,
and applied directly to the model's raw pre-softmax logits — the
mathematically correct implementation (Guo et al., 2017). Fitting T
on a 10% calibration split reduced mean ECE from {avg_ece_before:.3f} to
{avg_ece_after:.3f} (mean reduction {avg_reduction:.1f}% across all models and
platforms). Individual model ECE values are reported in Table S2. Macro
AUC was unchanged by temperature scaling (mean ΔAUC = {avg_auc_change:+.4f}),
confirming that monotonic confidence rescaling cannot alter
ranking order and therefore cannot substitute for domain-adaptive
retraining to recover discriminative performance. This finding supports
a two-tier remediation strategy: (1) temperature scaling for immediate
calibration improvement when labelled target-domain data are available
in small quantities; (2) domain-adaptive retraining for full
discriminative performance recovery.
""")
    print(f"  Temperature values by model and platform:")
    for _, row in df.iterrows():
        print(f"    {MODEL_DISPLAY[row['model']]:<20} {row['platform']:<10} "
              f"T*={row['temperature']:.4f}  "
              f"ECE {row['ece_before']:.3f} → {row['ece_after']:.3f} "
              f"({row['ece_reduction_pct']:.1f}%)")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("A5: Temperature Scaling Recalibration")
    print("=" * 60)
    print("Prerequisite: prediction CSVs must contain logit columns.")
    print("If logit columns are missing, re-run: python src/evaluate.py")
    print("=" * 60)

    df = run_temperature_scaling()

    if df.empty:
        print("\nERROR: No prediction files found or logit columns missing.")
        print("Re-run python src/evaluate.py first to generate logit columns.")
        sys.exit(1)

    out_csv = os.path.join(FAIRNESS_DIR, "temperature_scaling_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    plot_recalibration(df)
    print_paper_summary(df)

    print(f"\n{'='*60}")
    print("DONE. Temperature scaling results saved.")
