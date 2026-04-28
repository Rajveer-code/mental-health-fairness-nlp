"""
A1_di_eod_analysis.py
─────────────────────
Computes Disparate Impact (DI) and Equalized Odds Difference (EOD)
for all model × platform pairs and saves:
  - outputs/results/fairness/di_eod_table.csv
  - outputs/figures/figure_di_eod_heatmap.png

Run from the repository root:
    python src/A1_di_eod_analysis.py

Requires: per-sample prediction CSVs at
    outputs/results/{model}_{platform}_predictions.csv
  Each CSV must have columns:
    label          (int 0–3: 0=normal, 1=depression, 2=anxiety, 3=stress)
    pred           (int 0–3: predicted class)
    prob_normal, prob_depression, prob_anxiety, prob_stress  (float probabilities)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    MODELS, PLATFORMS, CLASSES, CLASS_IDS, MODEL_DISPLAY,
    load_config, load_predictions,
)

warnings.filterwarnings(  # suppress seaborn/matplotlib deprecation noise
    "ignore", category=FutureWarning
)

# ── Config ────────────────────────────────────────────────────────
cfg = load_config()

RESULTS_DIR  = cfg["paths"]["results"]
FIGURES_DIR  = cfg["paths"]["figures"]
FAIRNESS_DIR = os.path.join(RESULTS_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)

# ── Core Fairness Functions ────────────────────────────────────────

def disparate_impact(
    y_true_ref: np.ndarray,
    y_pred_ref: np.ndarray,
    y_true_tgt: np.ndarray,
    y_pred_tgt: np.ndarray,
    class_id: int,
) -> float:
    """
    Symmetric Disparate Impact for class ``class_id`` (Equation 6).

    .. math::

        DI_c = \\min\\!\\left(
            \\frac{P(\\hat{y}=c\\mid G=A)}{P(\\hat{y}=c\\mid G=B)},
            \\frac{P(\\hat{y}=c\\mid G=B)}{P(\\hat{y}=c\\mid G=A)}
        \\right)

    Range: (0, 1].  DI < 0.80 = four-fifths rule violation.
    DI < 0.50 = severe violation.  Reference platform is Kaggle.

    Parameters
    ----------
    y_true_ref, y_pred_ref : np.ndarray
        Ground-truth labels and predictions for the reference platform.
    y_true_tgt, y_pred_tgt : np.ndarray
        Ground-truth labels and predictions for the target platform.
    class_id : int
        Class index in [0, n_classes).

    Returns
    -------
    float
        Symmetric DI in (0, 1], or ``float("nan")`` if both rates are zero,
        or 0.0 if exactly one rate is zero.
    """
    rate_ref = float(np.mean(y_pred_ref == class_id))
    rate_tgt = float(np.mean(y_pred_tgt == class_id))

    if rate_ref <= 0 and rate_tgt <= 0:
        return float("nan")
    if rate_ref <= 0 or rate_tgt <= 0:
        return 0.0
    ratio = rate_ref / rate_tgt
    return round(float(min(ratio, 1.0 / ratio)), 4)


def disparate_impact_prior_adjusted(
    y_true_ref: np.ndarray,
    y_pred_ref: np.ndarray,
    y_true_tgt: np.ndarray,
    y_pred_tgt: np.ndarray,
    class_id: int,
) -> dict:
    """
    Prior-shift-adjusted Disparate Impact for class ``class_id``.

    Computes DI on class-balanced subsets by resampling the reference
    platform to match the target platform's class prevalence, removing
    the prior-shift confound.  Returns both raw DI and adjusted DI.

    This is needed because Reddit has ~80.9% normal samples vs Kaggle's
    ~29%: a model correctly adapting to this prevalence difference would
    show lower clinical-class prediction rates on Reddit by design,
    which reduces raw DI below 1.0 without any genuine model failure.

    Parameters
    ----------
    y_true_ref, y_pred_ref : np.ndarray
        Ground-truth labels and predictions for the reference platform.
    y_true_tgt, y_pred_tgt : np.ndarray
        Ground-truth labels and predictions for the target platform.
    class_id : int
        Class index in [0, n_classes).

    Returns
    -------
    dict
        Keys: ``di_raw``, ``di_adjusted``, ``prev_ref``, ``prev_tgt``.
        ``di_raw`` is the symmetric DI from ``disparate_impact()``.
        ``di_adjusted`` is DI after resampling ref to match tgt prevalence.
        Either may be ``float("nan")`` if resampling is not feasible.
    """
    prev_ref = float(np.mean(y_true_ref == class_id))
    prev_tgt = float(np.mean(y_true_tgt == class_id))

    di_raw = disparate_impact(y_true_ref, y_pred_ref,
                              y_true_tgt, y_pred_tgt, class_id)

    # Resample reference to match target class prevalence
    n_ref      = len(y_true_ref)
    n_pos_tgt  = int(round(prev_tgt * n_ref))
    n_neg_tgt  = n_ref - n_pos_tgt

    pos_ref_idx = np.where(y_true_ref == class_id)[0]
    neg_ref_idx = np.where(y_true_ref != class_id)[0]

    if len(pos_ref_idx) < n_pos_tgt or len(neg_ref_idx) < n_neg_tgt:
        # Cannot resample without replacement — return raw only
        return {
            "di_raw":      di_raw,
            "di_adjusted": float("nan"),
            "prev_ref":    round(prev_ref, 4),
            "prev_tgt":    round(prev_tgt, 4),
        }

    rng         = np.random.default_rng(42)
    sampled_pos = rng.choice(pos_ref_idx, size=n_pos_tgt, replace=False)
    sampled_neg = rng.choice(neg_ref_idx, size=n_neg_tgt, replace=False)
    balanced_idx = np.concatenate([sampled_pos, sampled_neg])

    y_pred_ref_balanced = y_pred_ref[balanced_idx]
    rate_ref_adj        = float(np.mean(y_pred_ref_balanced == class_id))
    rate_tgt            = float(np.mean(y_pred_tgt == class_id))

    if rate_ref_adj <= 0 and rate_tgt <= 0:
        di_adj = float("nan")
    elif rate_ref_adj <= 0 or rate_tgt <= 0:
        di_adj = 0.0
    else:
        ratio  = rate_ref_adj / rate_tgt
        di_adj = round(float(min(ratio, 1.0 / ratio)), 4)

    return {
        "di_raw":      di_raw,
        "di_adjusted": di_adj,
        "prev_ref":    round(prev_ref, 4),
        "prev_tgt":    round(prev_tgt, 4),
    }


def equalized_odds_difference(y_true_ref: np.ndarray, y_pred_ref: np.ndarray,
                              y_true_tgt: np.ndarray, y_pred_tgt: np.ndarray,
                              class_id: int) -> float:
    """
    EOD_c = |TPR_c(ref) - TPR_c(target)|
          = |P(ŷ=c | y=c, G=ref) - P(ŷ=c | y=c, G=tgt)|

    EOD = 0 indicates perfect Equalized Odds.
    """
    mask_ref = (y_true_ref == class_id)
    mask_tgt = (y_true_tgt == class_id)

    if mask_ref.sum() == 0 or mask_tgt.sum() == 0:
        return float("nan")

    tpr_ref = np.mean(y_pred_ref[mask_ref] == class_id)
    tpr_tgt = np.mean(y_pred_tgt[mask_tgt] == class_id)

    return round(float(abs(tpr_ref - tpr_tgt)), 4)


# ── Main Analysis ─────────────────────────────────────────────────

def run_di_eod():
    """
    For each model, compare reference platform (Kaggle) against
    Reddit and Twitter on DI and EOD for each class.
    """
    rows = []

    for model_key in MODELS:
        print(f"\n{'='*55}")
        print(f"Model: {MODEL_DISPLAY[model_key]}")
        print(f"{'='*55}")

        # Load reference (Kaggle) predictions
        ref_df = load_predictions(model_key, "kaggle", RESULTS_DIR)
        if ref_df is None:
            continue

        y_true_ref = ref_df["label"].values.astype(int)
        y_pred_ref = ref_df["pred"].values.astype(int)

        for tgt_platform in ["reddit", "twitter"]:
            if model_key == "mentalroberta" and tgt_platform == "reddit":
                print(
                    "  WARNING: GoEmotions-RoBERTa Reddit results are NON-INDEPENDENT. "
                    "This model was fine-tuned on GoEmotions (same source as the Reddit "
                    "test set). DI/EOD values represent in-distribution performance, "
                    "not cross-platform fairness. Flagged with † in all tables."
                )
            tgt_df = load_predictions(model_key, tgt_platform, RESULTS_DIR)
            if tgt_df is None:
                continue

            y_true_tgt = tgt_df["label"].values.astype(int)
            y_pred_tgt = tgt_df["pred"].values.astype(int)

            row = {
                "model":    model_key,
                "platform": tgt_platform,
                "n_ref":    len(y_true_ref),
                "n_tgt":    len(y_true_tgt),
            }

            print(f"\n  Kaggle (n={len(y_true_ref):,}) vs "
                  f"{tgt_platform.capitalize()} (n={len(y_true_tgt):,})")
            print(f"  {'Class':<12} {'DI_raw':>8} {'DI_adj':>10} "
                  f"{'EOD':>8} {'Interpretation'}")
            print(f"  {'-'*58}")

            for cls in CLASSES:
                cid = CLASS_IDS[cls]
                di_result = disparate_impact_prior_adjusted(
                    y_true_ref, y_pred_ref,
                    y_true_tgt, y_pred_tgt, cid
                )
                di     = di_result["di_raw"]
                di_adj = di_result["di_adjusted"]
                eod    = equalized_odds_difference(y_true_ref, y_pred_ref,
                                                   y_true_tgt, y_pred_tgt, cid)

                # Interpret DI
                if np.isnan(di):
                    di_flag = "N/A"
                elif di < 0.50:
                    di_flag = "SEVERE (<0.50)"
                elif di < 0.80:
                    di_flag = "Violation (<0.80)"
                else:
                    di_flag = "OK (≥0.80)"

                print(f"  {cls:<12} {di:>8.4f} {di_adj:>10.4f} {eod:>8.4f}  {di_flag}")

                row[f"di_{cls}"]      = di
                row[f"di_adj_{cls}"]  = di_adj
                row[f"eod_{cls}"]     = eod

            rows.append(row)

    return pd.DataFrame(rows)


def plot_di_heatmap(df: pd.DataFrame):
    """
    Heatmap: DI values for all model × platform × class combinations.
    Red = DI < 0.80 (violation); yellow = 0.80–0.95; green = close to 1.0.
    """
    # Build matrix: rows = model × platform, cols = classes
    di_cols = [f"di_{c}" for c in CLASSES]
    df_plot = df.copy()
    df_plot["label"] = (df_plot["model"].map(MODEL_DISPLAY)
                        + "\n(" + df_plot["platform"] + ")")

    mat = df_plot.set_index("label")[di_cols]
    mat.columns = [c.replace("di_", "").capitalize() for c in di_cols]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        center=0.80,          # four-fifths rule threshold
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Symmetric Disparate Impact (DI)"}
    )

    # Add threshold line annotation
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(
        "Symmetric Disparate Impact (DI) by Model, Platform, and Class\n"
        "Values < 0.80 (yellow→red) violate the four-fifths rule",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Mental Health Class", fontsize=10)
    ax.set_ylabel("")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_di_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


def plot_eod_heatmap(df: pd.DataFrame):
    """
    Heatmap: EOD values. Higher = worse Equalized Odds.
    """
    eod_cols = [f"eod_{c}" for c in CLASSES]
    df_plot = df.copy()
    df_plot["label"] = (df_plot["model"].map(MODEL_DISPLAY)
                        + "\n(" + df_plot["platform"] + ")")

    mat = df_plot.set_index("label")[eod_cols]
    mat.columns = [c.replace("eod_", "").capitalize() for c in eod_cols]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.8,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Equalized Odds Difference (EOD)"}
    )

    ax.set_title(
        "Equalized Odds Difference (EOD) by Model, Platform, and Class\n"
        "Higher values = greater disparity in True Positive Rate across platforms",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Mental Health Class", fontsize=10)
    ax.set_ylabel("")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_eod_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_paper_table(df: pd.DataFrame):
    """
    Print a formatted version of the DI/EOD table (Table 6) with both
    raw and prior-shift-adjusted DI values.
    """
    print(f"\n{'='*100}")
    print("TABLE 6 — DI (raw), DI (prior-adjusted), and Max EOD — for paper")
    print(f"{'='*100}")
    print(
        f"{'Model':<20} {'Platform':<10} "
        f"{'DI_nor':>7} {'DI_dep':>7} {'DI_anx':>7} {'DI_str':>7} "
        f"{'DI*_nor':>8} {'DI*_dep':>8} {'DI*_anx':>8} {'DI*_str':>8} "
        f"{'EOD_max':>9}"
    )
    print("-" * 100)

    for _, row in df.iterrows():
        eod_vals = [row[f"eod_{c}"] for c in CLASSES
                    if not np.isnan(row.get(f"eod_{c}", np.nan))]
        eod_max  = max(eod_vals) if eod_vals else float("nan")

        def _fmt(val: float) -> str:
            return f"{val:>7.3f}" if not np.isnan(val) else "    N/A"

        print(
            f"{MODEL_DISPLAY[row['model']]:<20} {row['platform']:<10} "
            f"{_fmt(row['di_normal'])} {_fmt(row['di_depression'])} "
            f"{_fmt(row['di_anxiety'])} {_fmt(row['di_stress'])} "
            f"{_fmt(row.get('di_adj_normal', float('nan')))} "
            f"{_fmt(row.get('di_adj_depression', float('nan')))} "
            f"{_fmt(row.get('di_adj_anxiety', float('nan')))} "
            f"{_fmt(row.get('di_adj_stress', float('nan')))} "
            f"{eod_max:>9.3f}"
        )

    print(f"\nDI* = prior-shift-adjusted DI (reference resampled to match target prevalence).")
    print("DI < 0.80 violates the four-fifths rule.  EOD = 0 = perfect Equalized Odds.")
    print("Reference platform: Kaggle.")


# ── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("A1: Disparate Impact and Equalized Odds Difference Analysis")
    print("=" * 55)

    df = run_di_eod()

    if df.empty:
        print("\nERROR: No prediction files found. Ensure CSVs exist at:")
        print("  outputs/results/{model}_{platform}_predictions.csv")
        print("Each CSV needs columns: label, pred, prob_normal, prob_depression,")
        print("  prob_anxiety, prob_stress")
        exit(1)

    # Save CSV
    out_csv = os.path.join(FAIRNESS_DIR, "di_eod_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    # Generate figures
    print("\nGenerating heatmaps...")
    plot_di_heatmap(df)
    plot_eod_heatmap(df)

    # Print paper-ready table
    print_paper_table(df)

    print(f"\n{'='*55}")
    print("DONE. Outputs:")
    print(f"  {out_csv}")
    print(f"  {FIGURES_DIR}/figure_di_heatmap.png")
    print(f"  {FIGURES_DIR}/figure_eod_heatmap.png")
    print("\nNext: Copy the TABLE 6 values above into the paper.")
    print("      Share figure_di_heatmap.png to add as Figure in Section 4.5")
