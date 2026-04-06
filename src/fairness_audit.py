"""
fairness_audit.py  [FIXED — v2]
-----------------
Performs clinical-grade fairness audit on model predictions.

CRITICAL FIX FROM ORIGINAL:
  The original pairwise_auc_comparison() used an incorrect standard error
  for the macro AUC Z-test.

  ORIGINAL (WRONG):
    auc_se = average of per-class DeLong SEs
    Z = |AUC1 - AUC2| / sqrt(se1^2 + se2^2)

  WHY IT'S WRONG:
    Macro AUC = (1/K) * sum(AUC_k).
    Var(macro AUC) = (1/K^2) * [sum Var(AUC_k) + 2 * sum_{j<k} Cov(AUC_j, AUC_k)]
    The covariance terms between per-class AUC estimates are non-zero
    because negative sets overlap (a "normal" sample is in the negative
    set for BOTH depression AUC and anxiety AUC).  Averaging the per-class
    SEs ignores all covariance terms, biasing the SE and Z-statistic.

  FIX:
    Use bootstrap SE of macro AUC directly.  Bootstrap naturally captures
    the full covariance structure between per-class AUC estimates because
    it resamples the joint sample, not each class independently.

    Z = |AUC1 - AUC2| / sqrt(SE_boot(AUC1)^2 + SE_boot(AUC2)^2)

    Because the two test sets are INDEPENDENT (different platforms),
    SE_boot(difference) = sqrt(SE_boot(AUC1)^2 + SE_boot(AUC2)^2).

    Bootstrap SEs are precomputed once per model×platform (cached) and
    reused across all pairs to avoid redundant computation.

IMPORTANT: The per-class AUC DeLong CIs (delong_auc_ci, multiclass_auc_ci)
    are NOT changed — those are binary AUC CIs and are correctly implemented.
    Only the pairwise macro AUC Z-test is fixed.

Runtime note: pairwise_auc_comparison() now runs B=2000 bootstrap
    resamples × 12 model×platform combinations = 24,000 AUC computations.
    Expected time: 3–8 minutes depending on hardware.

All other functions are unchanged from the original.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from itertools import combinations

from utils import (
    MODELS, PLATFORMS, CLASSES, PROB_COLS, MODEL_DISPLAY,
    load_config, load_predictions, compute_aggregate_ece,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────

cfg = load_config()

ALPHA      = cfg["fairness"]["alpha"]
ECE_BINS   = cfg["fairness"]["ece_bins"]
NUM_LABELS = len(CLASSES)

PLATFORM_DISPLAY = {
    "kaggle":  "Kaggle (within-platform)",
    "reddit":  "Reddit (cross-platform)",
    "twitter": "Twitter (cross-platform)",
}

RESULTS_DIR  = cfg["paths"]["results"]
FIGURES_DIR  = cfg["paths"]["figures"]
FAIRNESS_DIR = os.path.join(RESULTS_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── DeLong AUC with confidence intervals (unchanged) ─────────────────────────

def delong_auc_ci(y_true, y_score, alpha=0.05):
    """
    Compute AUC with DeLong 95% confidence interval for a binary task.
    Returns (auc, ci_lower, ci_upper, se).
    Unchanged from original — this implementation is correct.
    """
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    auc, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    auc    = auc / (n_pos * n_neg)

    def structural_components(pos, neg):
        v_pos = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        v_neg = np.array([np.mean(pos > n_) + 0.5 * np.mean(pos == n_) for n_ in neg])
        return v_pos, v_neg

    v_pos, v_neg = structural_components(pos_scores, neg_scores)
    var_pos = np.var(v_pos, ddof=1) / n_pos
    var_neg = np.var(v_neg, ddof=1) / n_neg
    se      = np.sqrt(var_pos + var_neg)

    z        = stats.norm.ppf(1 - alpha / 2)
    ci_lower = max(0.0, auc - z * se)
    ci_upper = min(1.0, auc + z * se)

    return round(auc, 4), round(ci_lower, 4), round(ci_upper, 4), round(se, 6)


def multiclass_auc_ci(y_true, y_probs, class_idx, alpha=0.05):
    """One-vs-rest AUC with DeLong CI for a specific class. Unchanged."""
    y_bin   = (y_true == class_idx).astype(int)
    y_score = y_probs[:, class_idx]
    return delong_auc_ci(y_bin, y_score, alpha)


# ── ECE (unchanged) ───────────────────────────────────────────────────────────

def _compute_ece_with_bins(y_true, y_probs, n_bins=10):
    """ECE with bin statistics. Unchanged."""
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    correct     = (predictions == y_true).astype(float)

    bins      = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    bin_stats = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            bin_stats.append(None)
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
        bin_stats.append({
            "bin_lower":  round(bins[i], 2),
            "bin_upper":  round(bins[i + 1], 2),
            "count":      int(mask.sum()),
            "accuracy":   round(float(acc), 4),
            "confidence": round(float(conf), 4),
        })

    return round(float(ece), 4), bin_stats


# ── Fairness metrics (unchanged) ──────────────────────────────────────────────

def disparate_impact(y_true, y_pred, group_a_mask, group_b_mask):
    """Symmetric DI per class. Unchanged."""
    dis = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        rate_a = (y_pred[group_a_mask] == cls_idx).mean()
        rate_b = (y_pred[group_b_mask] == cls_idx).mean()

        if rate_a <= 0 and rate_b <= 0:
            di = float("nan")
        elif rate_a <= 0 or rate_b <= 0:
            di = 0.0
        else:
            ratio = rate_a / rate_b
            di    = round(float(min(ratio, 1.0 / ratio)), 4)

        dis[cls_name] = round(float(di), 4)
    return dis


def equalized_odds_diff(y_true, y_pred, group_a_mask, group_b_mask):
    """Max |TPR_A - TPR_B| across classes. Unchanged."""
    max_diff = 0.0
    diffs    = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        pos_a = (y_true[group_a_mask] == cls_idx)
        pos_b = (y_true[group_b_mask] == cls_idx)
        if pos_a.sum() == 0 or pos_b.sum() == 0:
            diffs[cls_name] = float("nan")
            continue
        tpr_a = ((y_pred[group_a_mask] == cls_idx) & pos_a).sum() / pos_a.sum()
        tpr_b = ((y_pred[group_b_mask] == cls_idx) & pos_b).sum() / pos_b.sum()
        diff  = abs(float(tpr_a) - float(tpr_b))
        diffs[cls_name] = round(diff, 4)
        max_diff = max(max_diff, diff)
    return round(max_diff, 4), diffs


# ── FIXED: Pairwise AUC comparison ───────────────────────────────────────────

def _bootstrap_macro_auc_se(
    probs:   np.ndarray,
    labels:  np.ndarray,
    n_boots: int = 2000,
    seed:    int = 42,
) -> float:
    """
    Compute bootstrap standard error of the macro OvR AUC.

    This correctly captures the covariance structure between per-class
    AUC estimates that arises because samples from one class appear in
    the negative set of every other class's binary AUC computation.

    Parameters
    ----------
    probs : ndarray, shape (n, 4)
        Predicted probability distributions.
    labels : ndarray, shape (n,)
        Integer ground-truth labels.
    n_boots : int
        Number of bootstrap resamples.  2000 is standard for SE estimation.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    float
        Bootstrap SE of macro AUC.
    """
    rng  = np.random.default_rng(seed)
    n    = len(labels)
    boot = []

    for _ in range(n_boots):
        idx = rng.integers(0, n, n)
        try:
            b_auc = roc_auc_score(
                labels[idx], probs[idx],
                multi_class="ovr", average="macro",
            )
            boot.append(b_auc)
        except ValueError:
            # Can happen if a bootstrapped sample is missing a class.
            continue

    if len(boot) < 100:
        # Too many failures — SE estimate unreliable; return conservative fallback.
        return 0.02
    return float(np.std(boot, ddof=1))


def pairwise_auc_comparison(
    results_dir: str,
    n_boots:     int = 2000,
    seed:        int = 42,
) -> pd.DataFrame:
    """
    Bootstrap-based pairwise macro AUC comparison with Bonferroni correction.

    For each model, compares every pair of platforms using:
        Z = |AUC_1 - AUC_2| / sqrt(SE_boot(AUC_1)^2 + SE_boot(AUC_2)^2)

    where SE_boot is the bootstrap standard error of the macro OvR AUC.
    Because the two test sets are independent (different platforms), the
    SE of their difference is the quadrature sum of their individual SEs.

    Bootstrap SEs are precomputed once per model×platform and cached to
    avoid running 2000 resamples multiple times for the same dataset.

    Parameters
    ----------
    results_dir : str
        Directory containing the prediction CSVs.
    n_boots : int
        Bootstrap resamples for SE estimation per platform (default 2000).
    seed : int
        Base seed; each model×platform receives seed + offset for independence.

    Returns
    -------
    pd.DataFrame
        One row per pairwise comparison with columns:
        model, platform_1, platform_2, auc_1, auc_2, auc_diff,
        se_boot_1, se_boot_2, z_stat, p_value, alpha_corrected, significant.
    """
    pairs           = list(combinations(PLATFORMS, 2))
    n_comparisons   = len(pairs)
    alpha_corrected = ALPHA / n_comparisons

    print(f"\n  Precomputing bootstrap SEs (B={n_boots}) — this takes 3–8 min...")

    # ── Step 1: Precompute bootstrap SEs for every model × platform ───────────
    # Keyed by (model_key, platform) → {"auc": float, "se": float,
    #                                    "probs": ndarray, "labels": ndarray}
    cache: dict = {}
    seed_offset = 0

    for model_key in MODELS:
        for platform in PLATFORMS:
            df = load_predictions(model_key, platform, results_dir)
            if df is None:
                continue

            probs  = df[PROB_COLS].values.astype(float)
            labels = df["label"].values.astype(int)

            try:
                auc = float(roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro"
                ))
            except ValueError:
                auc = float("nan")

            se = _bootstrap_macro_auc_se(
                probs, labels,
                n_boots=n_boots,
                seed=seed + seed_offset,
            )
            seed_offset += 1

            cache[(model_key, platform)] = {
                "auc": round(auc, 4),
                "se":  round(se, 6),
            }
            print(f"    {MODEL_DISPLAY[model_key]:<20} {platform:<10} "
                  f"AUC={auc:.4f}  SE_boot={se:.4f}")

    # ── Step 2: Pairwise tests ────────────────────────────────────────────────
    results = []

    for m in MODELS:
        for p1, p2 in pairs:
            c1 = cache.get((m, p1))
            c2 = cache.get((m, p2))
            if c1 is None or c2 is None:
                continue

            a1   = c1["auc"]
            a2   = c2["auc"]
            se1  = c1["se"]
            se2  = c2["se"]

            if np.isnan(a1) or np.isnan(a2):
                continue

            se_diff = np.sqrt(se1**2 + se2**2)

            if se_diff == 0:
                z    = float("inf")
                pval = 0.0
            else:
                z    = abs(a1 - a2) / se_diff
                pval = float(2 * (1 - stats.norm.cdf(z)))

            results.append({
                "model":            m,
                "platform_1":       p1,
                "platform_2":       p2,
                "auc_1":            a1,
                "auc_2":            a2,
                "auc_diff":         round(abs(a1 - a2), 4),
                "se_boot_auc1":     round(se1, 6),
                "se_boot_auc2":     round(se2, 6),
                "z_stat":           round(z, 4),
                "p_value":          round(pval, 6),
                "alpha_corrected":  round(alpha_corrected, 6),
                "significant":      pval < alpha_corrected,
            })

    return pd.DataFrame(results)


def between_model_auc_test(
    results_dir: str,
    target_platform: str = "reddit",
    n_boots: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap-based pairwise macro AUC comparison across models on a single platform.

    Tests whether GoEmotions-RoBERTa AUC differs significantly from each of
    the other three models on the specified platform.  This complements
    ``pairwise_auc_comparison`` (which tests the same model across platforms)
    by testing different models on the same platform.

    Uses the same bootstrap SE approach as ``pairwise_auc_comparison`` with
    Bonferroni correction for 6 pairwise comparisons (4 choose 2).

    Parameters
    ----------
    results_dir : str
        Directory containing the prediction CSVs.
    target_platform : str
        Platform on which to compare models.  Default ``"reddit"``.
    n_boots : int
        Bootstrap resamples for SE estimation.  Default 2000.
    seed : int
        Random seed for bootstrap.  Default 42.

    Returns
    -------
    pd.DataFrame
        One row per model pair with columns:
        platform, model_1, model_2, auc_1, auc_2, delta_auc,
        z_stat, p_value, alpha_bonferroni, significant.
    """
    N_PAIRS        = 6   # 4 choose 2
    alpha_bonf     = ALPHA / N_PAIRS

    model_aucs: dict[str, float] = {}
    model_ses:  dict[str, float] = {}

    print(f"\n  Between-model AUC test on {target_platform} (B={n_boots})...")

    for offset, model_key in enumerate(MODELS):
        df = load_predictions(model_key, target_platform, results_dir)
        if df is None:
            continue

        probs  = df[PROB_COLS].values.astype(float)
        labels = df["label"].values.astype(int)

        try:
            auc = float(roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            ))
        except ValueError:
            continue

        se = _bootstrap_macro_auc_se(probs, labels,
                                     n_boots=n_boots, seed=seed + offset)
        model_aucs[model_key] = auc
        model_ses[model_key]  = se
        print(f"    {MODEL_DISPLAY[model_key]:<20} AUC={auc:.4f}  SE={se:.5f}")

    rows = []
    for m1, m2 in combinations(list(model_aucs.keys()), 2):
        auc1, auc2 = model_aucs[m1], model_aucs[m2]
        se1,  se2  = model_ses[m1],  model_ses[m2]
        se_diff    = float(np.sqrt(se1 ** 2 + se2 ** 2))
        z_stat     = abs(auc1 - auc2) / se_diff if se_diff > 0 else float("nan")
        p_value    = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        rows.append({
            "platform":         target_platform,
            "model_1":          m1,
            "model_2":          m2,
            "auc_1":            round(auc1, 4),
            "auc_2":            round(auc2, 4),
            "delta_auc":        round(auc1 - auc2, 4),
            "z_stat":           round(z_stat, 3),
            "p_value":          round(p_value, 6),
            "alpha_bonferroni": round(alpha_bonf, 6),
            "significant":      bool(p_value < alpha_bonf),
        })

    return pd.DataFrame(rows)


# ── Per-model per-platform audit (unchanged) ──────────────────────────────────

def audit_model_platform(model_key, platform):
    """Full audit for one model × platform. Unchanged from original."""
    df = load_predictions(model_key, platform, RESULTS_DIR)
    if df is None:
        print(f"  WARNING: predictions not found for {model_key}/{platform}, skipping.")
        return None

    y_true  = df["label"].values
    y_pred  = df["pred"].values
    y_probs = df[PROB_COLS].values

    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc_macro = roc_auc_score(y_true, y_probs,
                                  multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")

    per_class_auc = {}
    for i, cls_name in enumerate(CLASSES):
        auc, ci_lo, ci_hi, se = multiclass_auc_ci(y_true, y_probs, i)
        per_class_auc[cls_name] = {
            "auc":      auc,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "se":       se,
            "n_pos":    int((y_true == i).sum()),
            "n_neg":    int((y_true != i).sum()),
        }

    ece, bin_stats = _compute_ece_with_bins(y_true, y_probs, ECE_BINS)

    f1_per     = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {CLASSES[i]: round(float(f1_per[i]), 4)
                    for i in range(len(f1_per))}

    return {
        "model":         model_key,
        "platform":      platform,
        "n_samples":     len(df),
        "accuracy":      round(float(acc), 4),
        "f1_macro":      round(float(f1_mac), 4),
        "auc_macro":     round(float(auc_macro), 4),
        "ece":           ece,
        "per_class_auc": per_class_auc,
        "per_class_f1":  per_class_f1,
        "bin_stats":     bin_stats,
    }


# ── Visualisations (unchanged) ────────────────────────────────────────────────

def plot_forest_plot(audit_results):
    """Forest plot — unchanged."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for col, platform in enumerate(PLATFORMS):
        ax = axes[col]
        y_pos    = 0
        y_ticks  = []
        y_labels = []

        for m_idx, model_key in enumerate(MODELS):
            key = (model_key, platform)
            if key not in audit_results:
                continue
            res = audit_results[key]

            for cls_name in CLASSES:
                cls_res = res["per_class_auc"][cls_name]
                auc     = cls_res["auc"]
                ci_lo   = cls_res["ci_lower"]
                ci_hi   = cls_res["ci_upper"]

                if np.isnan(auc):
                    y_pos += 1
                    continue

                ax.errorbar(
                    auc, y_pos,
                    xerr=[[auc - ci_lo], [ci_hi - auc]],
                    fmt="o",
                    color=colors[m_idx],
                    capsize=3,
                    markersize=5,
                    linewidth=1.5,
                    label=MODEL_DISPLAY[model_key] if cls_name == "normal" and y_pos < 8 else ""
                )
                y_ticks.append(y_pos)
                y_labels.append(cls_name)
                y_pos += 1

            y_pos += 1

        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlim(0.3, 1.05)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("AUC (95% DeLong CI)", fontsize=10)
        ax.set_title(PLATFORM_DISPLAY[platform], fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=colors[i], label=MODEL_DISPLAY[m])
        for i, m in enumerate(MODELS)
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("Per-Class AUC with 95% DeLong Confidence Intervals\nAcross Models and Platforms",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(FIGURES_DIR, "figure1_forest_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_platform_degradation(audit_results):
    """Bar chart — unchanged."""
    data = []
    for model_key in MODELS:
        for platform in PLATFORMS:
            key = (model_key, platform)
            if key in audit_results:
                data.append({
                    "Model":    MODEL_DISPLAY[model_key],
                    "Platform": PLATFORM_DISPLAY[platform],
                    "F1 Macro": audit_results[key]["f1_macro"],
                    "AUC":      audit_results[key]["auc_macro"],
                    "ECE":      audit_results[key]["ece"],
                })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    palette   = ["#1976D2", "#43A047", "#FB8C00"]

    for idx, metric in enumerate(["F1 Macro", "AUC", "ECE"]):
        ax    = axes[idx]
        pivot = df.pivot(index="Model", columns="Platform", values=metric)
        pivot.plot(kind="bar", ax=ax, color=palette, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
        ax.set_title(f"{metric} by Model and Platform",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
        ax.legend(fontsize=8, title="Platform", title_fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if metric == "ECE":
            ax.set_ylim(0, 0.65)

    plt.suptitle("Cross-Platform Performance Degradation Across Models",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure2_platform_degradation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_calibration_curves(audit_results):
    """Calibration curves — unchanged."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    for m_idx, model_key in enumerate(MODELS):
        for p_idx, platform in enumerate(PLATFORMS):
            ax  = axes[m_idx][p_idx]
            key = (model_key, platform)

            if key not in audit_results:
                ax.set_visible(False)
                continue

            bins  = audit_results[key]["bin_stats"]
            valid = [b for b in bins if b is not None]
            if not valid:
                continue

            conf = [b["confidence"] for b in valid]
            acc  = [b["accuracy"]   for b in valid]

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.plot(conf, acc, "o-", color="#2196F3",
                    markersize=4, linewidth=1.5, label="Model")
            ax.fill_between(conf, acc, conf, alpha=0.1, color="#2196F3")

            ece = audit_results[key]["ece"]
            ax.set_title(
                f"{MODEL_DISPLAY[model_key]} — {platform}\nECE={ece:.3f}",
                fontsize=9
            )
            ax.set_xlabel("Confidence", fontsize=8)
            ax.set_ylabel("Accuracy",   fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    plt.suptitle("Calibration Curves by Model and Platform",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure3_calibration_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_heatmap(audit_results):
    """F1 heatmap — unchanged."""
    rows = []
    for model_key in MODELS:
        for platform in PLATFORMS:
            key = (model_key, platform)
            if key not in audit_results:
                continue
            row = {"Model-Platform": f"{MODEL_DISPLAY[model_key]}\n{platform}"}
            for cls in CLASSES:
                row[cls] = audit_results[key]["per_class_f1"].get(cls, float("nan"))
            rows.append(row)

    df = pd.DataFrame(rows).set_index("Model-Platform")

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "F1 Score"})
    ax.set_title("Per-Class F1 Score Heatmap\nAcross Models and Platforms",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mental Health Class", fontsize=11)
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure4_f1_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Starting fairness audit (FIXED — bootstrap SE for pairwise test)...")
    print(f"Alpha: {ALPHA} | Bonferroni-corrected for "
          f"{len(list(combinations(PLATFORMS, 2)))} comparisons per model")

    audit_results = {}
    all_rows      = []

    for model_key in MODELS:
        print(f"\n[{model_key.upper()}]")
        for platform in PLATFORMS:
            print(f"  Auditing: {platform}")
            res = audit_model_platform(model_key, platform)
            if res is None:
                continue
            key = (model_key, platform)
            audit_results[key] = res

            all_rows.append({
                "model":       model_key,
                "platform":    platform,
                "n_samples":   res["n_samples"],
                "accuracy":    res["accuracy"],
                "f1_macro":    res["f1_macro"],
                "auc_macro":   res["auc_macro"],
                "ece":         res["ece"],
                **{f"auc_{cls}":       res["per_class_auc"][cls]["auc"]
                   for cls in CLASSES},
                **{f"auc_{cls}_ci_lo": res["per_class_auc"][cls]["ci_lower"]
                   for cls in CLASSES},
                **{f"auc_{cls}_ci_hi": res["per_class_auc"][cls]["ci_upper"]
                   for cls in CLASSES},
                **{f"f1_{cls}":        res["per_class_f1"][cls]
                   for cls in CLASSES},
            })

            for cls in CLASSES:
                ca = res["per_class_auc"][cls]
                print(f"    {cls:<12} AUC={ca['auc']:.4f} "
                      f"[{ca['ci_lower']:.4f}, {ca['ci_upper']:.4f}]  "
                      f"n_pos={ca['n_pos']}")
            print(f"    ECE={res['ece']:.4f}  F1={res['f1_macro']:.4f}  "
                  f"AUC={res['auc_macro']:.4f}")

    fairness_df = pd.DataFrame(all_rows)
    fairness_df.to_csv(
        os.path.join(FAIRNESS_DIR, "fairness_audit_full.csv"), index=False
    )

    # ── FIXED: Pairwise test uses bootstrap SE ──────────────────────────────
    print("\nRunning pairwise AUC comparisons (Bonferroni corrected — bootstrap SE)...")
    print("This step takes 3–8 minutes. Please wait.")
    pairwise_df = pairwise_auc_comparison(RESULTS_DIR, n_boots=2000)
    pairwise_df.to_csv(
        os.path.join(FAIRNESS_DIR, "pairwise_auc_comparisons.csv"), index=False
    )

    print("\nPairwise comparison results (Table 6):")
    print(f"{'Model':<18} {'P1':<8} {'P2':<8} {'AUC1':>6} {'AUC2':>6} "
          f"{'ΔAUC':>6} {'Z':>7} {'p':>9} {'Sig?'}")
    print("-" * 90)
    for _, row in pairwise_df.iterrows():
        sig = "YES (p<0.0167)" if row["significant"] else "no"
        print(f"{MODEL_DISPLAY[row['model']]:<18} {row['platform_1']:<8} "
              f"{row['platform_2']:<8} {row['auc_1']:>6.4f} {row['auc_2']:>6.4f} "
              f"{row['auc_diff']:>6.4f} {row['z_stat']:>7.2f} "
              f"{row['p_value']:>9.6f} {sig}")

    sig_count = pairwise_df["significant"].sum()
    print(f"\nSignificant pairs after Bonferroni: {sig_count}/{len(pairwise_df)}")

    print("\nGenerating figures...")
    plot_forest_plot(audit_results)
    plot_platform_degradation(audit_results)
    plot_calibration_curves(audit_results)
    plot_heatmap(audit_results)

    serializable = {
        f"{k[0]}_{k[1]}": {kk: vv for kk, vv in v.items() if kk != "bin_stats"}
        for k, v in audit_results.items()
    }
    with open(os.path.join(FAIRNESS_DIR, "audit_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*60}")
    print("FAIRNESS AUDIT COMPLETE")
    print(f"{'='*60}")
    print("\nKey finding — AUC drop from Kaggle to Reddit/Twitter:")
    for model_key in MODELS:
        k_auc = audit_results.get((model_key, "kaggle"),  {}).get("auc_macro", None)
        r_auc = audit_results.get((model_key, "reddit"),  {}).get("auc_macro", None)
        t_auc = audit_results.get((model_key, "twitter"), {}).get("auc_macro", None)
        if k_auc and r_auc and t_auc:
            r_drop = round((k_auc - r_auc) / k_auc * 100, 1)
            t_drop = round((k_auc - t_auc) / k_auc * 100, 1)
            print(f"  {MODEL_DISPLAY[model_key]:<20} "
                  f"Kaggle={k_auc:.4f}  "
                  f"Reddit={r_auc:.4f} (-{r_drop}%)  "
                  f"Twitter={t_auc:.4f} (-{t_drop}%)")

    # ── Between-model AUC significance tests (NE-3 / B5) ─────────────────────
    print(f"\n{'='*60}")
    print("Between-Model AUC Significance Tests")
    print(f"{'='*60}")
    for tgt_plat in ["reddit", "twitter"]:
        bm_df = between_model_auc_test(RESULTS_DIR, target_platform=tgt_plat)
        if not bm_df.empty:
            out_bm = os.path.join(
                FAIRNESS_DIR, f"between_model_auc_{tgt_plat}.csv"
            )
            bm_df.to_csv(out_bm, index=False)
            print(f"\n  Platform: {tgt_plat.upper()}")
            print(f"  {'Model 1':<20} {'Model 2':<20} {'ΔAUC':>7} "
                  f"{'Z':>7} {'p':>10} {'Sig?':>6}")
            print("  " + "-" * 75)
            for _, row in bm_df.iterrows():
                sig = "YES" if row["significant"] else "no"
                print(f"  {MODEL_DISPLAY[row['model_1']]:<20} "
                      f"{MODEL_DISPLAY[row['model_2']]:<20} "
                      f"{row['delta_auc']:>7.4f} {row['z_stat']:>7.3f} "
                      f"{row['p_value']:>10.6f} {sig:>6}")
            print(f"  Saved: {out_bm}")

    print(f"\nOutputs saved to:")
    print(f"  {FAIRNESS_DIR}")
    print(f"  {FIGURES_DIR}")
    print("\nNext step: python src/shap_analysis.py")


if __name__ == "__main__":
    main()
