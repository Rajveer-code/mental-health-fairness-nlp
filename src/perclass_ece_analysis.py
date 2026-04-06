"""
perclass_ece_analysis.py  [FIXED — v2]
─────────────────────────
Adds per-class ECE (one-vs-rest) to the fairness audit.

CRITICAL FIX FROM ORIGINAL:
  The original script used n_boots=200 for bootstrap CI estimation.
  The manuscript states "ECE 95% bootstrap confidence intervals (B=1000)"
  in Table 3.  This fix sets n_boots=1000 everywhere, making the code
  consistent with the manuscript claim.

  The runtime increase is modest: 200 → 1000 boots is ~5× slower per
  call but total wall time is still well under 5 minutes.

All other logic is unchanged.
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
    MODELS, PLATFORMS, CLASSES, PROB_COLS, MODEL_DISPLAY,
    load_config, load_predictions, compute_aggregate_ece, bootstrap_ci,
)

warnings.filterwarnings("ignore", category=FutureWarning)

cfg = load_config()

RESULTS_DIR  = cfg["paths"]["results"]
FIGURES_DIR  = cfg["paths"]["figures"]
FAIRNESS_DIR = os.path.join(RESULTS_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)

ECE_BINS = cfg["fairness"]["ece_bins"]   # 10

# FIXED: n_boots=1000 throughout (was 200 in original)
N_BOOTS = 1000


def compute_perclass_ece(probs: np.ndarray, labels: np.ndarray,
                         class_idx: int, M: int = 10) -> tuple:
    """
    Per-class ECE via one-vs-rest binary calibration.

    For class c:
      - Confidence = P(class=c) for each sample
      - Accuracy   = 1 if true label = c, else 0
      - Bin on P(class=c)

    Returns (ece_c, bin_details).
    Unchanged from original.
    """
    n       = len(labels)
    probs_c = probs[:, class_idx]
    true_c  = (labels == class_idx).astype(float)

    ece      = 0.0
    bins     = np.linspace(0, 1, M + 1)
    bin_stats = []

    for i in range(M):
        mask = (probs_c >= bins[i]) & (probs_c < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = float(true_c[mask].mean())
        conf = float(probs_c[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
        bin_stats.append({
            "bin":   f"{bins[i]:.1f}–{bins[i+1]:.1f}",
            "count": int(mask.sum()),
            "acc":   round(acc, 4),
            "conf":  round(conf, 4),
            "delta": round(abs(acc - conf), 4),
        })

    return round(float(ece), 4), bin_stats


def bootstrap_ece_ci(probs: np.ndarray, labels: np.ndarray,
                     class_idx: int = None, M: int = 10,
                     n_boots: int = N_BOOTS) -> tuple:
    """
    Bootstrap 95% CI for ECE (aggregate or per-class).

    FIXED: default n_boots=1000 (was 200).
    This makes all reported CIs consistent with the paper's claim of B=1000.

    Parameters
    ----------
    probs : np.ndarray
    labels : np.ndarray
    class_idx : int or None
        None → aggregate ECE; int → per-class ECE for that class.
    M : int
        Number of calibration bins.
    n_boots : int
        Bootstrap resamples.  Must be 1000 for publication.

    Returns
    -------
    tuple[float, float]
        (lower_95_CI, upper_95_CI)
    """
    n = len(labels)
    boot_eces = []

    rng = np.random.default_rng(42)

    for _ in range(n_boots):
        idx = rng.integers(0, n, n)
        if class_idx is None:
            e = compute_aggregate_ece(probs[idx], labels[idx], M)
        else:
            e, _ = compute_perclass_ece(probs[idx], labels[idx], class_idx, M)
        boot_eces.append(e)

    lo = float(np.percentile(boot_eces, 2.5))
    hi = float(np.percentile(boot_eces, 97.5))
    return round(lo, 4), round(hi, 4)


def run():
    """Compute aggregate and per-class ECE for all models × platforms."""
    rows = []

    for model_key in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_DISPLAY[model_key]}")
        print(f"{'='*60}")

        for platform in PLATFORMS:
            df = load_predictions(model_key, platform, RESULTS_DIR)
            if df is None:
                continue

            probs  = df[PROB_COLS].values
            labels = df["label"].values.astype(int)
            n      = len(labels)

            # Aggregate ECE with B=1000 bootstrap CI
            ece_agg         = compute_aggregate_ece(probs, labels, ECE_BINS)
            ece_lo, ece_hi  = bootstrap_ece_ci(probs, labels, class_idx=None,
                                               n_boots=N_BOOTS)

            row = {
                "model":      model_key,
                "platform":   platform,
                "n":          n,
                "ece_agg":    ece_agg,
                "ece_agg_lo": ece_lo,
                "ece_agg_hi": ece_hi,
            }

            print(f"\n  {platform.upper()} (n={n:,})")
            print(f"  {'Class':<12} {'ECE':>8} {'95% CI [B=1000]':<22} "
                  f"{'n_pos':>8} {'Interpretation'}")
            print(f"  {'-'*72}")
            print(f"  {'AGGREGATE':<12} {ece_agg:>8.4f} "
                  f"[{ece_lo:.4f}, {ece_hi:.4f}]")

            for cls_name, cls_idx in zip(CLASSES, range(len(CLASSES))):
                ece_c, _       = compute_perclass_ece(probs, labels, cls_idx, ECE_BINS)
                lo_c, hi_c     = bootstrap_ece_ci(probs, labels, class_idx=cls_idx,
                                                   n_boots=N_BOOTS)
                n_pos = int((labels == cls_idx).sum())

                if np.isnan(ece_c):
                    interp = "N/A"
                elif ece_c < 0.10:
                    interp = "Well-calibrated"
                elif ece_c < 0.20:
                    interp = "Acceptable"
                elif ece_c < 0.35:
                    interp = "Moderate miscalibration"
                else:
                    interp = "SEVERE miscalibration"

                print(f"  {cls_name:<12} {ece_c:>8.4f} "
                      f"[{lo_c:.4f}, {hi_c:.4f}]        "
                      f"{n_pos:>8,} {interp}")

                row[f"ece_{cls_name}"]    = ece_c
                row[f"ece_{cls_name}_lo"] = lo_c
                row[f"ece_{cls_name}_hi"] = hi_c
                row[f"n_{cls_name}"]      = n_pos

            rows.append(row)

    return pd.DataFrame(rows)


def plot_perclass_ece_heatmap(df: pd.DataFrame):
    """
    Heatmap: aggregate ECE (left) and per-class ECE (right).
    Shows whether minority class miscalibration is masked by aggregate ECE.
    Unchanged from original.
    """
    plot_rows = []
    for _, row in df.iterrows():
        label = f"{MODEL_DISPLAY[row['model']]}\n({row['platform']})"
        entry = {"Model-Platform": label}
        for c in CLASSES:
            entry[c.capitalize()] = row.get(f"ece_{c}", float("nan"))
        plot_rows.append(entry)

    hm = pd.DataFrame(plot_rows).set_index("Model-Platform")

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    agg_data = []
    for _, row in df.iterrows():
        agg_data.append({
            "Model-Platform": f"{MODEL_DISPLAY[row['model']]}\n({row['platform']})",
            "Aggregate ECE":  row["ece_agg"],
        })
    agg_df = pd.DataFrame(agg_data).set_index("Model-Platform")

    sns.heatmap(
        agg_df, annot=True, fmt=".3f",
        cmap="YlOrRd", vmin=0, vmax=0.6,
        ax=axes[0], linewidths=0.5,
        cbar_kws={"label": "ECE"},
    )
    axes[0].set_title(
        "Aggregate ECE\n(max-probability confidence; Equation 5)",
        fontsize=10, fontweight="bold"
    )

    sns.heatmap(
        hm, annot=True, fmt=".3f",
        cmap="YlOrRd", vmin=0, vmax=0.6,
        ax=axes[1], linewidths=0.5,
        cbar_kws={"label": "Per-Class ECE"},
    )
    axes[1].set_title(
        "Per-Class ECE (one-vs-rest)\nBootstrap 95% CIs computed with B=1000",
        fontsize=10, fontweight="bold"
    )

    fig.suptitle(
        "Aggregate vs. Per-Class Expected Calibration Error (B=1000 bootstrap)\n"
        "Per-class ECE reveals miscalibration hidden by majority-class dominance",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_perclass_ece_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


def print_paper_additions(df: pd.DataFrame):
    """Print Equation 5 extension and paper text additions."""
    print(f"\n{'='*70}")
    print("PAPER ADDITIONS — Per-Class ECE (B=1000 consistent with manuscript)")
    print(f"{'='*70}")
    print("""
ADD TO §3.2 after Equation 5:

  "Equation 5 computes aggregate ECE using max-probability confidence,
   which is dominated by the majority class in imbalanced distributions.
   To assess calibration for the clinically critical minority classes
   (anxiety, stress), we additionally compute per-class (one-vs-rest) ECE:

   ECE_c = Σ_m (|B_m^c|/n) · |acc(B_m^c) − conf(B_m^c)|   ... (5b)

   where B_m^c = {i : P(y=c|x_i) ∈ [(m-1)/M, m/M)}.
   Bootstrap 95% confidence intervals are computed with B=1000 resamples
   for both aggregate and per-class ECE."
""")


if __name__ == "__main__":
    print("Per-Class ECE Analysis — FIXED (n_boots=1000)")
    print("=" * 55)

    df = run()

    if df.empty:
        print("\nNo prediction files found.")
        exit(1)

    out = os.path.join(FAIRNESS_DIR, "perclass_ece.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    plot_perclass_ece_heatmap(df)
    print_paper_additions(df)

    print(f"\n{'='*55}")
    print("DONE — all bootstrap CIs computed with B=1000 (paper-consistent).")
    print(f"  Results: {FAIRNESS_DIR}/perclass_ece.csv")
    print(f"  Figure:  {FIGURES_DIR}/figure_perclass_ece_heatmap.png")
