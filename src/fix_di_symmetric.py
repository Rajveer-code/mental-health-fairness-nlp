"""
fix_di_symmetric.py
───────────────────
FIX F3: Replaces the asymmetric Disparate Impact formula in
code_A1_di_eod_analysis.py and regenerates Table 6 with the
symmetric (four-fifths-compliant) formulation.

PROBLEM:
  Current formula: DI_c = P(ŷ=c | G=target) / P(ŷ=c | G=ref)
  This produces DI > 1.0 for the Normal class on Reddit (3.14–3.29),
  which is uninterpretable under the four-fifths rule (DI < 0.80 = violation).

FIX:
  Symmetric formula: DI_c = min(rate_A/rate_B, rate_B/rate_A)
  This constrains DI ∈ (0, 1] where:
    1.0 = perfect parity
    < 0.80 = violation of four-fifths rule
    < 0.50 = severe violation

IMPLICATIONS:
  - Normal class on Reddit now shows DI = 0.304–0.318 (SEVERE violation)
    instead of the meaningless 3.14–3.29
  - DistilRoBERTa Twitter depression (was 1.10) → 0.909 (borderline OK)
  - All other values remain below 0.80 — findings are unchanged

Run from repo root:
    python src/fix_di_symmetric.py

Requires: outputs/results/{model}_{platform}_predictions.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# DEPRECATED: The symmetric_di() formula in this script has been backported
# into code_A1_di_eod_analysis.py.  Do not run this script independently —
# it now produces redundant output.  Retained for git history traceability.
warnings.warn(
    "fix_di_symmetric.py is deprecated.  The symmetric DI formula has been "
    "backported into code_A1_di_eod_analysis.py.  Do not run this script.",
    DeprecationWarning,
    stacklevel=1,
)

from utils import (
    MODELS, PLATFORMS, CLASSES, MODEL_DISPLAY,
    load_config, load_predictions,
)

warnings.filterwarnings(  # suppress seaborn/matplotlib deprecation noise
    "ignore", category=FutureWarning
)

cfg = load_config()

RESULTS_DIR  = cfg["paths"]["results"]
FIGURES_DIR  = cfg["paths"]["figures"]
FAIRNESS_DIR = os.path.join(RESULTS_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)


def symmetric_di(rate_a: float, rate_b: float) -> float:
    """
    Symmetric Disparate Impact: min(rate_A/rate_B, rate_B/rate_A).

    Returns a value in (0, 1] where 1.0 = perfect parity.
    DI < 0.80 violates the four-fifths rule regardless of direction.

    Why symmetric:
    The four-fifths rule tests whether two groups receive positive
    outcomes at comparable rates. The direction of the ratio is
    irrelevant to the fairness question — a DI of 3.0 is just as
    much a violation as a DI of 0.33. The symmetric form makes the
    threshold (< 0.80) applicable in both directions.
    """
    if rate_a <= 0 and rate_b <= 0:
        return float("nan")
    if rate_a <= 0 or rate_b <= 0:
        return 0.0  # One group never predicted this class = maximum disparity
    ratio = rate_a / rate_b
    return round(float(min(ratio, 1.0 / ratio)), 4)


def equalized_odds_diff(y_true_ref, y_pred_ref, y_true_tgt, y_pred_tgt,
                        class_id):
    """Signed EOD: TPR_ref - TPR_tgt (positive = higher TPR on ref)."""
    mask_ref = (y_true_ref == class_id)
    mask_tgt = (y_true_tgt == class_id)
    if mask_ref.sum() == 0 or mask_tgt.sum() == 0:
        return float("nan")
    tpr_ref = np.mean(y_pred_ref[mask_ref] == class_id)
    tpr_tgt = np.mean(y_pred_tgt[mask_tgt] == class_id)
    return round(float(abs(tpr_ref - tpr_tgt)), 4)


def run():
    rows = []
    ref_data = {}   # cache kaggle predictions

    print("Loading Kaggle (reference) predictions for all models...")
    for model_key in MODELS:
        df = load_predictions(model_key, "kaggle", RESULTS_DIR)
        if df is not None:
            ref_data[model_key] = df

    for model_key in MODELS:
        if model_key not in ref_data:
            print(f"\nSkipping {model_key}: no Kaggle predictions found")
            continue

        ref_df = ref_data[model_key]
        y_true_ref = ref_df["label"].values.astype(int)
        y_pred_ref = ref_df["pred"].values.astype(int)

        print(f"\n{'='*60}")
        print(f"Model: {MODEL_DISPLAY[model_key]}")
        print(f"{'='*60}")

        for tgt_platform in ["reddit", "twitter"]:
            tgt_df = load_predictions(model_key, tgt_platform, RESULTS_DIR)
            if tgt_df is None:
                continue

            y_true_tgt = tgt_df["label"].values.astype(int)
            y_pred_tgt = tgt_df["pred"].values.astype(int)

            print(f"\n  vs {tgt_platform.capitalize()} "
                  f"(ref n={len(ref_df):,}, tgt n={len(tgt_df):,})")

            row = {
                "model":         model_key,
                "platform":      tgt_platform,
                "n_ref":         len(ref_df),
                "n_tgt":         len(tgt_df),
            }

            print(f"  {'Class':<14} {'rate_ref':>9} {'rate_tgt':>9} "
                  f"{'DI(sym)':>9} {'EOD':>8} {'DI_flag'}")
            print("  " + "-"*65)

            eod_vals = []
            for cls in CLASSES:
                cid = CLASS_IDS[cls] 
                rate_ref = float(np.mean(y_pred_ref == cid))
                rate_tgt = float(np.mean(y_pred_tgt == cid))

                di  = symmetric_di(rate_ref, rate_tgt)
                eod = equalized_odds_diff(y_true_ref, y_pred_ref,
                                          y_true_tgt, y_pred_tgt, cid)

                if not np.isnan(eod):
                    eod_vals.append(eod)

                if np.isnan(di):
                    flag = "N/A"
                elif di < 0.50:
                    flag = "SEVERE (<0.50)"
                elif di < 0.80:
                    flag = "Violation (<0.80)"
                else:
                    flag = "OK (≥0.80)"

                print(f"  {cls:<14} {rate_ref:>9.4f} {rate_tgt:>9.4f} "
                      f"{di:>9.4f} {eod:>8.4f} {flag}")

                row[f"di_{cls}"]  = di
                row[f"eod_{cls}"] = eod

            row["eod_max"] = round(max(eod_vals), 4) if eod_vals else float("nan")
            rows.append(row)

    return pd.DataFrame(rows)


def plot_di_heatmap(df):
    """
    Heatmap of symmetric DI values.
    Green ≥ 0.80 (compliant), yellow 0.50–0.79, red < 0.50 (severe).
    All values now ∈ (0, 1] — no more DI > 1.0 entries.
    """
    di_cols = [f"di_{c}" for c in CLASSES]
    df_plot = df.copy()
    df_plot["label"] = (
        df_plot["model"].map(MODEL_DISPLAY) + "\n(" + df_plot["platform"] + ")"
    )
    mat = df_plot.set_index("label")[di_cols]
    mat.columns = [c.replace("di_", "").capitalize() for c in di_cols]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        center=0.80,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Symmetric Disparate Impact (DI)"},
    )
    ax.set_title(
        "Symmetric Disparate Impact — Fixed Formulation\n"
        "min(rate_A/rate_B, rate_B/rate_A) ∈ (0,1]  |  "
        "DI < 0.80 = four-fifths rule violation",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Mental Health Class", fontsize=10)
    ax.set_ylabel("")
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "figure_di_heatmap_symmetric.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


def print_paper_table(df):
    """Print paper-ready Table 6 with corrected symmetric DI values."""
    print(f"\n{'='*75}")
    print("CORRECTED TABLE 6 — Symmetric Disparate Impact and Max EOD")
    print("(Replace the original Table 6 in the manuscript)")
    print(f"{'='*75}")
    print(f"{'Model':<20} {'Platform':<10} "
          f"{'DI_normal':>10} {'DI_depres':>10} "
          f"{'DI_anxty':>10} {'DI_stress':>10} {'EOD_max':>9}")
    print("-" * 75)

    for _, row in df.iterrows():
        print(f"{MODEL_DISPLAY[row['model']]:<20} {row['platform']:<10} "
              f"{row['di_normal']:>10.3f} {row['di_depression']:>10.3f} "
              f"{row['di_anxiety']:>10.3f} {row['di_stress']:>10.3f} "
              f"{row['eod_max']:>9.3f}")

    print()
    print("Notes:")
    print("  DI formula: min(P(ŷ=c|G=A)/P(ŷ=c|G=B), P(ŷ=c|G=B)/P(ŷ=c|G=A))")
    print("  All DI values now ∈ (0,1]. DI < 0.80 = four-fifths rule violation.")
    print("  DI < 0.50 = severe violation.")
    print("  Reference platform: Kaggle (within-platform training distribution).")
    print()
    print("Key change from original Table 6:")
    print("  Normal class on Reddit (was 3.14–3.29, uninterpretable) →")
    print("  now 0.304–0.318 (SEVERE violation — models massively")
    print("  over-predict 'Normal' on Reddit relative to Kaggle,")
    print("  indicating systematic failure to detect clinical conditions).")
    print()
    print("  DistilRoBERTa Twitter depression (was 1.10, uninterpretable) →")
    print("  now 0.909 (borderline OK — slightly under-predicted on Twitter).")
    print()
    print("  Finding STRENGTHENED: Normal class now universally violates")
    print("  the four-fifths rule on Reddit, which was hidden by the")
    print("  asymmetric formula. The clinical interpretation is unchanged:")
    print("  models systematically default to predicting 'Normal' when")
    print("  they cannot classify cross-platform clinical content.")


if __name__ == "__main__":
    print("FIX F3: Symmetric Disparate Impact Analysis")
    print("=" * 60)

    df = run()

    if df.empty:
        print("\nNo prediction files found.")
        exit(1)

    # Save corrected CSV
    out_csv = os.path.join(FAIRNESS_DIR, "di_eod_table_symmetric.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Heatmap
    plot_di_heatmap(df)

    # Paper table
    print_paper_table(df)

    print(f"\n{'='*60}")
    print("DONE. Replace Table 6 in the manuscript with the values above.")
    print("Update code_A1_di_eod_analysis.py: change disparate_impact()")
    print("  to use min(rate_tgt/rate_ref, rate_ref/rate_tgt)")
    print("Update paper §3.3 Equation 6 notation to clarify symmetric form.")
