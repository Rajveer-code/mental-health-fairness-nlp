"""
A2_stress_attribution.py
─────────────────────────
Generates gradient-based token attribution figures for the STRESS class
across all four models and three platforms. Mirrors the existing depression-
class figures (Figures 5a–5d) to fill the mechanistic explanation gap.

Output:
  outputs/figures/figure5e_bert_stress_attribution.png
  outputs/figures/figure5f_roberta_stress_attribution.png
  outputs/figures/figure5g_distilroberta_stress_attribution.png
  outputs/figures/figure5h_samlowe_stress_attribution.png
  outputs/figures/figure5_stress_combined.png   ← 2-panel comparison figure
                                                    (SamLowe-RoBERTa only,
                                                     depression vs stress)

Run from repo root:
    python src/A2_stress_attribution.py

Hardware: GPU recommended. Works on CPU but ~4× slower.
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import (
    MODELS, CLASSES, MODEL_HF_IDS, MODEL_DISPLAY, PLATFORM_COLORS,
    load_config,
    find_platform_file, get_model_checkpoint, compute_token_importance,
)

warnings.filterwarnings(  # suppress transformers deprecation noise
    "ignore", category=FutureWarning
)

# ── Config ────────────────────────────────────────────────────────
cfg = load_config()

DATA_DIR    = cfg["paths"]["splits"]
MODELS_DIR  = cfg["paths"]["models"]
RESULTS_DIR = cfg["paths"]["results"]
FIGURES_DIR = cfg["paths"]["figures"]
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MAX_SAMPLES = 200   # samples per platform for attribution (same as depression figures)
TOP_K       = 15    # top tokens to display
STRESS_IDX  = 3     # stress is class index 3 in the 4-class schema

PLATFORM_FILES = {
    p: find_platform_file(p, DATA_DIR) for p in ["kaggle", "reddit", "twitter"]
}


def get_top_words(scores_dict, k=TOP_K):
    """Return top-K (word, score) pairs sorted descending."""
    return sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:k]


# ── Plotting ─────────────────────────────────────────────────────

def plot_attribution_figure(model_key, platform_scores, class_name="stress"):
    """
    3-panel horizontal bar chart for one model across three platforms.
    Mirrors the format of Figures 5a-5d.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Cross-Platform Gradient Token Attribution — {MODEL_CONFIGS[model_key]['display']}\n"
        f"{class_name.capitalize()} class: which words drive predictions per platform",
        fontsize=13, fontweight="bold"
    )

    for ax, platform in zip(axes, ["kaggle", "reddit", "twitter"]):
        scores = platform_scores.get(platform, {})
        top    = get_top_words(scores)

        if not top:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{platform.upper()}\n({class_name} features)")
            continue

        words, vals = zip(*top)
        color = PLATFORM_COLORS[platform]

        ax.barh(words[::-1], vals[::-1], color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean |Gradient Saliency|", fontsize=10)
        ax.set_title(f"{platform.upper()}\n({class_name} features)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()

    fig_key = {"bert": "5e", "roberta": "5f",
               "mentalbert": "5g", "mentalroberta": "5h"}[model_key]
    out = os.path.join(FIGURES_DIR,
                       f"figure{fig_key}_{model_key}_stress_attribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def plot_combined_comparison(model_key, depression_scores, stress_scores):
    """
    Side-by-side comparison: depression class vs stress class
    for one model (SamLowe-RoBERTa) across all platforms.
    This is the 'money figure' showing that both clinical classes
    lose clinical signal on Twitter.
    """
    platforms = ["kaggle", "reddit", "twitter"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Depression vs Stress: Cross-Platform Feature Attribution\n"
        f"Model: {MODEL_CONFIGS[model_key]['display']}  "
        f"— showing that both clinical classes shift to platform artefacts",
        fontsize=13, fontweight="bold"
    )

    for row_idx, (class_name, all_scores) in enumerate([
        ("depression", depression_scores),
        ("stress",     stress_scores)
    ]):
        for col_idx, platform in enumerate(platforms):
            ax = axes[row_idx][col_idx]
            scores = all_scores.get(platform, {})
            top    = get_top_words(scores)

            if not top:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                words, vals = zip(*top[:10])
                color = PLATFORM_COLORS[platform]
                ax.barh(words[::-1], vals[::-1], color=color, alpha=0.85)
                ax.grid(axis="x", alpha=0.3)
                ax.invert_yaxis()
                ax.set_xlabel("Mean |Gradient Saliency|", fontsize=9)

            if row_idx == 0:
                ax.set_title(f"{platform.upper()}", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{class_name.upper()} class",
                              fontsize=11, fontweight="bold", color="darkred")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure5_stress_combined_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────

def load_platform_texts(platform, label_filter=None, max_n=MAX_SAMPLES):
    """Load texts from a platform CSV, optionally filtered by class."""
    path = PLATFORM_FILES.get(platform)
    if path is None or not os.path.exists(path):
        print(f"  WARNING: data file not found for platform={platform}")
        return []
    df = pd.read_csv(path)
    if label_filter is not None:
        df = df[df["label"] == label_filter]
    texts = df["text"].dropna().tolist()
    return texts[:max_n]


def run_stress_attribution():
    """Run attribution for stress class across all models and platforms."""
    for model_key in MODELS:
        display   = MODEL_DISPLAY[model_key]
        ckpt_dir  = get_model_checkpoint(model_key, MODELS_DIR)

        if ckpt_dir is None:
            print(f"\nMISSING checkpoint for {display} — skipping")
            continue

        print(f"\n{'='*55}")
        print(f"Model: {display}")
        print(f"{'='*55}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_IDS[model_key])
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir
        ).to(DEVICE)
        model.eval()

        platform_scores = {}

        for platform in ["kaggle", "reddit", "twitter"]:
            print(f"\n  Platform: {platform.upper()}")

            texts = load_platform_texts(platform, label_filter=STRESS_IDX)
            if not texts:
                texts = load_platform_texts(platform)

            print(f"    Loaded {len(texts)} texts")

            if texts:
                scores = compute_token_importance(
                    model, tokenizer, texts,
                    target_class_idx=STRESS_IDX,
                    device=DEVICE,
                )
                platform_scores[platform] = scores
                top = get_top_words(scores, k=5)
                print(f"    Top-5: {[w for w, _ in top]}")

        plot_attribution_figure(model_key, platform_scores, class_name="stress")

        if model_key == "mentalroberta":
            print(f"\n  Generating depression vs stress comparison figure...")
            depression_scores = {}
            for platform in ["kaggle", "reddit", "twitter"]:
                texts = load_platform_texts(platform, label_filter=1)
                if not texts:
                    texts = load_platform_texts(platform)
                if texts:
                    dep_scores = compute_token_importance(
                        model, tokenizer, texts,
                        target_class_idx=1,
                        device=DEVICE,
                    )
                    depression_scores[platform] = dep_scores

            plot_combined_comparison(
                model_key, depression_scores, platform_scores
            )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*55}")
    print("DONE — Stress attribution figures saved to:")
    print(f"  {FIGURES_DIR}/figure5e_bert_stress_attribution.png")
    print(f"  {FIGURES_DIR}/figure5f_roberta_stress_attribution.png")
    print(f"  {FIGURES_DIR}/figure5g_distilroberta_stress_attribution.png")
    print(f"  {FIGURES_DIR}/figure5h_samlowe_stress_attribution.png")
    print(f"  {FIGURES_DIR}/figure5_stress_combined_comparison.png")
    print("\nNext: Add these figures to the paper as Figures 5e-5h")
    print("      Reference them in Section 4.8 with the stress-class discussion.")

if __name__ == "__main__":
    run_stress_attribution()
