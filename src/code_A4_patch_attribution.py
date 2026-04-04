"""
A4_patch_gradient_attribution.py
─────────────────────────────────
Run this BEFORE running code_A3_A4_A6_ece_jaccard.py to generate
the attribution score CSV files needed for Jaccard K sensitivity.

This script adds token score saving to your existing
gradient_attribution.py (or shap_analysis.py) output.

Option 1 (recommended): Run this standalone script which directly
  reads your existing figures/shap outputs and derives the scores,
  OR re-runs attribution on prediction files if scores don't exist.

Option 2: Manually add to your gradient_attribution.py:
  After computing word_scores dict, add:
    scores_df = pd.DataFrame({
        'token': list(word_scores.keys()),
        'mean_importance': [np.mean(v) for v in word_scores.values()]
    }).sort_values('mean_importance', ascending=False)
    out_dir = os.path.join(RESULTS_DIR, 'attribution')
    os.makedirs(out_dir, exist_ok=True)
    scores_df.to_csv(f'{out_dir}/{model_key}_{platform}_{class_name}_scores.csv', index=False)

Run from repo root:
    python src/A4_patch_gradient_attribution.py
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import warnings

from utils import (
    MODELS, PLATFORMS, CLASSES, CLASS_IDS, MODEL_HF_IDS,
    load_config,
    find_platform_file, get_model_checkpoint, compute_token_importance,
)

warnings.filterwarnings(  # suppress transformers deprecation noise
    "ignore", category=FutureWarning
)

# ── Config ──────────────────────────────────────────────────────
cfg = load_config()

MODELS_DIR      = cfg["paths"]["models"]
RESULTS_DIR     = cfg["paths"]["results"]
SPLITS_DIR      = cfg["paths"]["splits"]
ATTRIBUTION_DIR = os.path.join(RESULTS_DIR, "attribution")
os.makedirs(ATTRIBUTION_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Limit texts per class for speed — attribution is expensive.
_MAX_TEXTS_PER_CLASS = 100


def save_scores(scores_dict, model_key, platform, class_name):
    df = pd.DataFrame({
        "token": list(scores_dict.keys()),
        "mean_importance": list(scores_dict.values())
    }).sort_values("mean_importance", ascending=False)
    path = os.path.join(ATTRIBUTION_DIR,
                        f"{model_key}_{platform}_{class_name}_scores.csv")
    df.to_csv(path, index=False)
    return path


def run():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    for model_key in MODELS:
        ckpt_dir = get_model_checkpoint(model_key, MODELS_DIR)
        if ckpt_dir is None:
            print(f"SKIP {model_key}: no checkpoint found")
            continue

        print(f"\n{'='*50}\nModel: {model_key}\n{'='*50}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_IDS[model_key])
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(DEVICE)
        model.eval()

        for platform in PLATFORMS:
            path = find_platform_file(platform, SPLITS_DIR)
            if path is None:
                print(f"  SKIP {platform}: no data file")
                continue
            df = pd.read_csv(path)

            for class_name, class_id in CLASS_IDS.items():
                out_path = os.path.join(ATTRIBUTION_DIR,
                    f"{model_key}_{platform}_{class_name}_scores.csv")
                if os.path.exists(out_path):
                    print(f"  EXISTS: {model_key}/{platform}/{class_name}")
                    continue

                # Get texts with this label; fall back to all texts if none found.
                texts = df[df["label"] == class_id]["text"].dropna().tolist()
                if not texts:
                    texts = df["text"].dropna().tolist()
                texts = texts[:_MAX_TEXTS_PER_CLASS]

                print(f"  Computing: {platform}/{class_name} (n={len(texts)})")
                # batch_size=8 intentional: smaller batches for memory efficiency
                # during attribution (gradient accumulation is expensive).
                scores = compute_token_importance(
                    model, tokenizer, texts, class_id,
                    batch_size=8, device=DEVICE,
                )
                if scores:
                    save_scores(scores, model_key, platform, class_name)
                    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"    Top-5: {[w for w, _ in top5]}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone. Attribution scores saved to: {ATTRIBUTION_DIR}")
    print("Now re-run: python src/code_A3_A4_A6_ece_jaccard.py")


if __name__ == "__main__":
    run()
