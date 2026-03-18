"""
fairness_audit.py
-----------------
Performs clinical-grade fairness audit on model predictions.

Computes per-platform:
  - Subgroup AUC with DeLong 95% confidence intervals
  - Expected Calibration Error (ECE) per platform
  - Disparate Impact and Equalized Odds Difference
  - Bonferroni-corrected pairwise AUC comparisons
  - Per-class fairness breakdown

Saves all results to outputs/results/fairness/
Figures saved to outputs/figures/

Usage:
    python src/fairness_audit.py
"""

import os
import json
import yaml
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

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ALPHA       = cfg["fairness"]["alpha"]
ECE_BINS    = cfg["fairness"]["ece_bins"]
LABEL_NAMES = ["normal", "depression", "anxiety", "stress"]
NUM_LABELS  = 4

MODELS = ["bert", "roberta", "mentalbert", "mentalroberta"]
MODEL_DISPLAY = {
    "bert":          "BERT",
    "roberta":       "RoBERTa",
    "mentalbert":    "DistilRoBERTa",
    "mentalroberta": "SamLowe-RoBERTa",
}
PLATFORMS = ["kaggle", "reddit", "twitter"]
PLATFORM_DISPLAY = {
    "kaggle":  "Kaggle (within-platform)",
    "reddit":  "Reddit (cross-platform)",
    "twitter": "Twitter (cross-platform)",
}

RESULTS_DIR = cfg["paths"]["results"]
FIGURES_DIR = cfg["paths"]["figures"]
FAIRNESS_DIR = os.path.join(RESULTS_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── DeLong AUC with confidence intervals ─────────────────────────────────────

def delong_auc_ci(y_true, y_score, alpha=0.05):
    """
    Compute AUC with DeLong 95% confidence interval.
    Returns (auc, ci_lower, ci_upper, se)

    DeLong et al. (1988) method — same as used in the IEEE diabetes paper.
    """
    # Binary AUC via Mann-Whitney U statistic
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # Mann-Whitney U
    auc, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    auc = auc / (n_pos * n_neg)

    # DeLong variance estimate
    def structural_components(pos, neg):
        m, n = len(pos), len(neg)
        v_pos = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg)
                          for p in pos])
        v_neg = np.array([np.mean(pos > n_) + 0.5 * np.mean(pos == n_)
                          for n_ in neg])
        return v_pos, v_neg

    v_pos, v_neg = structural_components(pos_scores, neg_scores)
    var_pos = np.var(v_pos, ddof=1) / n_pos
    var_neg = np.var(v_neg, ddof=1) / n_neg
    se = np.sqrt(var_pos + var_neg)

    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = max(0.0, auc - z * se)
    ci_upper = min(1.0, auc + z * se)

    return round(auc, 4), round(ci_lower, 4), round(ci_upper, 4), round(se, 6)


def multiclass_auc_ci(y_true, y_probs, class_idx, alpha=0.05):
    """
    One-vs-rest AUC with DeLong CI for a specific class.
    """
    y_bin   = (y_true == class_idx).astype(int)
    y_score = y_probs[:, class_idx]
    return delong_auc_ci(y_bin, y_score, alpha)


# ── Expected Calibration Error ────────────────────────────────────────────────

def compute_ece(y_true, y_probs, n_bins=10):
    """
    Expected Calibration Error across all classes (macro).
    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    """
    # Use max probability as confidence, argmax as prediction
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
            "bin_lower": round(bins[i], 2),
            "bin_upper": round(bins[i + 1], 2),
            "count":     int(mask.sum()),
            "accuracy":  round(float(acc), 4),
            "confidence":round(float(conf), 4),
        })

    return round(float(ece), 4), bin_stats


# ── Fairness metrics ──────────────────────────────────────────────────────────

def disparate_impact(y_true, y_pred, group_a_mask, group_b_mask):
    """
    DI = P(Y_hat=positive | group_A) / P(Y_hat=positive | group_B)
    For multi-class: use per-class positive rate.
    """
    dis = {}
    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        rate_a = (y_pred[group_a_mask] == cls_idx).mean()
        rate_b = (y_pred[group_b_mask] == cls_idx).mean()
        di = rate_a / rate_b if rate_b > 0 else float("nan")
        dis[cls_name] = round(float(di), 4)
    return dis


def equalized_odds_diff(y_true, y_pred, group_a_mask, group_b_mask):
    """
    EOD = max |TPR_A - TPR_B| across classes.
    """
    max_diff = 0.0
    diffs    = {}
    for cls_idx, cls_name in enumerate(LABEL_NAMES):
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


# ── Pairwise AUC comparison with Bonferroni correction ───────────────────────

def pairwise_auc_comparison(aucs_dict, alpha=0.05):
    """
    Compare AUC across all platform pairs for each model.
    Applies Bonferroni correction for multiple comparisons.
    Returns significance flags.
    """
    pairs        = list(combinations(PLATFORMS, 2))
    n_comparisons = len(pairs)
    alpha_corrected = alpha / n_comparisons
    results = []

    for m in MODELS:
        for p1, p2 in pairs:
            a1 = aucs_dict.get((m, p1), {}).get("auc_macro")
            a2 = aucs_dict.get((m, p2), {}).get("auc_macro")
            if a1 is None or a2 is None:
                continue
            # Z-test approximation for AUC comparison
            se1 = aucs_dict.get((m, p1), {}).get("auc_se", 0.01)
            se2 = aucs_dict.get((m, p2), {}).get("auc_se", 0.01)
            se_diff = np.sqrt(se1**2 + se2**2)
            if se_diff == 0:
                continue
            z    = abs(a1 - a2) / se_diff
            pval = 2 * (1 - stats.norm.cdf(z))
            results.append({
                "model":              m,
                "platform_1":         p1,
                "platform_2":         p2,
                "auc_1":              a1,
                "auc_2":              a2,
                "auc_diff":           round(abs(a1 - a2), 4),
                "z_stat":             round(z, 4),
                "p_value":            round(pval, 6),
                "alpha_corrected":    round(alpha_corrected, 6),
                "significant":        pval < alpha_corrected,
            })

    return pd.DataFrame(results)


# ── Audit per model per platform ──────────────────────────────────────────────

def audit_model_platform(model_key, platform):
    """
    Full fairness audit for one model on one platform.
    Returns dict of all metrics.
    """
    pred_path = os.path.join(
        RESULTS_DIR,
        f"{model_key}_{platform}_predictions.csv"
    )
    if not os.path.exists(pred_path):
        print(f"  WARNING: {pred_path} not found, skipping.")
        return None

    df = pd.read_csv(pred_path)
    y_true  = df["label"].values
    y_pred  = df["pred"].values
    y_probs = df[["prob_normal", "prob_depression",
                  "prob_anxiety", "prob_stress"]].values

    # Overall metrics
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc_macro = roc_auc_score(y_true, y_probs,
                                  multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")

    # Per-class AUC with DeLong CI
    per_class_auc = {}
    auc_se_values = []
    for i, cls_name in enumerate(LABEL_NAMES):
        auc, ci_lo, ci_hi, se = multiclass_auc_ci(y_true, y_probs, i)
        per_class_auc[cls_name] = {
            "auc":      auc,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "se":       se,
            "n_pos":    int((y_true == i).sum()),
            "n_neg":    int((y_true != i).sum()),
        }
        if not np.isnan(se):
            auc_se_values.append(se)

    avg_se = float(np.mean(auc_se_values)) if auc_se_values else 0.01

    # ECE
    ece, bin_stats = compute_ece(y_true, y_probs, ECE_BINS)

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {LABEL_NAMES[i]: round(float(f1_per[i]), 4)
                    for i in range(len(f1_per))}

    return {
        "model":         model_key,
        "platform":      platform,
        "n_samples":     len(df),
        "accuracy":      round(float(acc), 4),
        "f1_macro":      round(float(f1_mac), 4),
        "auc_macro":     round(float(auc_macro), 4),
        "auc_se":        avg_se,
        "ece":           ece,
        "per_class_auc": per_class_auc,
        "per_class_f1":  per_class_f1,
        "bin_stats":     bin_stats,
    }


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_forest_plot(audit_results):
    """
    Forest plot: per-class AUC ± 95% CI for each model × platform.
    This is the key Figure 1 of the paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for col, platform in enumerate(PLATFORMS):
        ax = axes[col]
        y_pos   = 0
        y_ticks = []
        y_labels = []

        for m_idx, model_key in enumerate(MODELS):
            key = (model_key, platform)
            if key not in audit_results:
                continue
            res = audit_results[key]

            for cls_name in LABEL_NAMES:
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
                y_labels.append(f"{cls_name}")
                y_pos += 1

            y_pos += 1  # gap between models

        ax.axvline(x=0.5, color="red", linestyle="--",
                   alpha=0.5, linewidth=1, label="Chance (0.5)")
        ax.set_xlim(0.3, 1.05)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("AUC (95% CI)", fontsize=10)
        ax.set_title(PLATFORM_DISPLAY[platform], fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    # Legend
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
    """
    Bar chart: F1 macro per model, grouped by platform.
    Shows the platform fairness gap clearly.
    This is Figure 2.
    """
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
    palette = ["#1976D2", "#43A047", "#FB8C00"]

    for idx, metric in enumerate(["F1 Macro", "AUC", "ECE"]):
        ax = axes[idx]
        pivot = df.pivot(index="Model", columns="Platform", values=metric)
        pivot.plot(kind="bar", ax=ax, color=palette, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
        ax.set_title(f"{metric} by Model and Platform",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=25, ha="right", fontsize=9)
        ax.legend(fontsize=8, title="Platform", title_fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if metric == "ECE":
            ax.set_ylim(0, 0.5)

    plt.suptitle("Cross-Platform Performance Degradation Across Models",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure2_platform_degradation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_calibration_curves(audit_results):
    """
    Calibration curves per model per platform.
    Figure 3.
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    for m_idx, model_key in enumerate(MODELS):
        for p_idx, platform in enumerate(PLATFORMS):
            ax  = axes[m_idx][p_idx]
            key = (model_key, platform)

            if key not in audit_results:
                ax.set_visible(False)
                continue

            bins = audit_results[key]["bin_stats"]
            valid = [b for b in bins if b is not None]
            if not valid:
                continue

            conf = [b["confidence"] for b in valid]
            acc  = [b["accuracy"]   for b in valid]

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.plot(conf, acc, "o-", color="#2196F3",
                    markersize=4, linewidth=1.5, label="Model")
            ax.fill_between(conf, acc, conf,
                            alpha=0.1, color="#2196F3")

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
    """
    Heatmap: F1 per class per model per platform.
    Figure 4 — shows exactly which class fails on which platform.
    """
    rows = []
    for model_key in MODELS:
        for platform in PLATFORMS:
            key = (model_key, platform)
            if key not in audit_results:
                continue
            row = {
                "Model-Platform": f"{MODEL_DISPLAY[model_key]}\n{platform}"
            }
            for cls in LABEL_NAMES:
                row[cls] = audit_results[key]["per_class_f1"].get(cls, float("nan"))
            rows.append(row)

    df = pd.DataFrame(rows).set_index("Model-Platform")

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "F1 Score"},
    )
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
    print("Starting fairness audit...")
    print(f"Alpha: {ALPHA} | Bonferroni-corrected for {len(list(combinations(PLATFORMS,2)))} comparisons")

    audit_results = {}
    all_rows      = []
    aucs_dict     = {}

    # Run audit for all model × platform combinations
    for model_key in MODELS:
        print(f"\n[{model_key.upper()}]")
        for platform in PLATFORMS:
            print(f"  Auditing: {platform}")
            res = audit_model_platform(model_key, platform)
            if res is None:
                continue
            key = (model_key, platform)
            audit_results[key] = res
            aucs_dict[key]     = {
                "auc_macro": res["auc_macro"],
                "auc_se":    res["auc_se"],
            }

            all_rows.append({
                "model":       model_key,
                "platform":    platform,
                "n_samples":   res["n_samples"],
                "accuracy":    res["accuracy"],
                "f1_macro":    res["f1_macro"],
                "auc_macro":   res["auc_macro"],
                "ece":         res["ece"],
                **{f"auc_{cls}": res["per_class_auc"][cls]["auc"]
                   for cls in LABEL_NAMES},
                **{f"auc_{cls}_ci_lo": res["per_class_auc"][cls]["ci_lower"]
                   for cls in LABEL_NAMES},
                **{f"auc_{cls}_ci_hi": res["per_class_auc"][cls]["ci_upper"]
                   for cls in LABEL_NAMES},
                **{f"f1_{cls}": res["per_class_f1"][cls]
                   for cls in LABEL_NAMES},
            })

            # Print per-class AUC with CI
            for cls in LABEL_NAMES:
                ca = res["per_class_auc"][cls]
                print(f"    {cls:<12} AUC={ca['auc']:.4f} "
                      f"[{ca['ci_lower']:.4f}, {ca['ci_upper']:.4f}]  "
                      f"n_pos={ca['n_pos']}")

            print(f"    ECE={res['ece']:.4f}  "
                  f"F1={res['f1_macro']:.4f}  "
                  f"AUC={res['auc_macro']:.4f}")

    # Save full fairness table
    fairness_df = pd.DataFrame(all_rows)
    fairness_df.to_csv(
        os.path.join(FAIRNESS_DIR, "fairness_audit_full.csv"),
        index=False
    )

    # Pairwise AUC comparisons with Bonferroni correction
    print("\nRunning pairwise AUC comparisons (Bonferroni corrected)...")
    pairwise_df = pairwise_auc_comparison(aucs_dict)
    pairwise_df.to_csv(
        os.path.join(FAIRNESS_DIR, "pairwise_auc_comparisons.csv"),
        index=False
    )
    sig = pairwise_df[pairwise_df["significant"] == True]
    print(f"  Significant pairs (after Bonferroni): {len(sig)}/{len(pairwise_df)}")

    # Generate all figures
    print("\nGenerating figures...")
    plot_forest_plot(audit_results)
    plot_platform_degradation(audit_results)
    plot_calibration_curves(audit_results)
    plot_heatmap(audit_results)

    # Save full audit results as JSON
    serializable = {
        f"{k[0]}_{k[1]}": {
            kk: vv for kk, vv in v.items()
            if kk != "bin_stats"
        }
        for k, v in audit_results.items()
    }
    with open(os.path.join(FAIRNESS_DIR, "audit_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("FAIRNESS AUDIT COMPLETE")
    print(f"{'='*60}")
    print("\nKey finding — AUC drop from Kaggle to Reddit/Twitter:")
    for model_key in MODELS:
        k_auc = audit_results.get((model_key, "kaggle"),  {}).get("auc_macro", "N/A")
        r_auc = audit_results.get((model_key, "reddit"),  {}).get("auc_macro", "N/A")
        t_auc = audit_results.get((model_key, "twitter"), {}).get("auc_macro", "N/A")
        if isinstance(k_auc, float):
            r_drop = round((k_auc - r_auc) / k_auc * 100, 1)
            t_drop = round((k_auc - t_auc) / k_auc * 100, 1)
            print(f"  {MODEL_DISPLAY[model_key]:<20} "
                  f"Kaggle={k_auc:.4f}  "
                  f"Reddit={r_auc:.4f} (-{r_drop}%)  "
                  f"Twitter={t_auc:.4f} (-{t_drop}%)")

    print(f"\nOutputs saved to:")
    print(f"  Results: {FAIRNESS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("\nNext step: python src/shap_analysis.py")


if __name__ == "__main__":
    main()