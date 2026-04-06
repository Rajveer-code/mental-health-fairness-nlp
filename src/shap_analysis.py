"""
shap_analysis.py
────────────────
Gradient-based saliency attribution analysis across platforms.

For each model, computes gradient saliency scores on the Kaggle test set
and cross-platform test sets to identify which linguistic features drive
prediction disparities across platforms.

Method: gradient-based saliency (backpropagation of class probability
score through the token embedding layer), implementing Equations 8–9
from the paper. Note: despite the filename, this script does NOT use
the SHAP library; the method is gradient saliency throughout.

Inputs
------
data/splits/cross_platform/test_{platform}.csv
outputs/models/{model_key}/

Outputs
-------
outputs/results/gradient_attribution/{model}_{platform}_{class}_top_words.csv
outputs/figures/figure5_{model}_gradient_cross_platform.png
outputs/figures/figure6_feature_stability.png

Usage
-----
Run from the repository root:
    python src/shap_analysis.py

Dependencies
------------
Requires train.py to have been run first (model checkpoints must exist).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import (
    MODEL_DISPLAY, PLATFORM_COLORS, CLASSES, PLATFORMS,
    load_config,
)

warnings.filterwarnings(  # suppress transformers deprecation noise
    "ignore", category=FutureWarning
)

cfg = load_config()

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use CLASSES from utils instead of a local CLASSES definition.
NUM_LABELS  = len(CLASSES)
MAX_LEN     = cfg["training"]["max_length"]
FIGURES_DIR = cfg["paths"]["figures"]
RESULTS_DIR = cfg["paths"]["results"]
GRAD_DIR    = os.path.join(RESULTS_DIR, "gradient_attribution")
os.makedirs(GRAD_DIR,    exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Build path-based model dict from config (avoids hardcoded output paths).
MODELS = {
    m: os.path.join(cfg["paths"]["models"], m) for m in MODEL_DISPLAY
}
# Build test-set paths from config splits directory.
TEST_SETS = {
    p: os.path.join(cfg["paths"]["splits"], "cross_platform", f"test_{p}.csv")
    for p in PLATFORMS
}

# Sample size for gradient saliency — limit to keep compute feasible
SHAP_SAMPLE_SIZE  = 200   # retained for backward compat with callers
GRAD_SAMPLE_SIZE  = 200


def get_prediction_function(model, tokenizer):
    """
    Returns a function that takes a list of texts and returns
    predicted probabilities. Required by shap.Explainer.
    """
    def predict(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()
    return predict


def run_shap_analysis(model_key, model_path):
    """
    Compute gradient-based saliency attribution for all platforms.

    Despite the function name (retained for backward compatibility),
    this uses gradient saliency — not SHAP.
    """
    print(f"\n  Loading model: {model_key}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    ).to(DEVICE)
    model.eval()

    platform_top_words = {}

    for platform, test_path in TEST_SETS.items():
        print(f"    Platform: {platform}")
        df = pd.read_csv(test_path)

        # Sample texts stratified by label
        samples = []
        for label_id in range(NUM_LABELS):
            subset = df[df["label"] == label_id]
            n      = min(GRAD_SAMPLE_SIZE // NUM_LABELS, len(subset))
            samples.append(subset.sample(n, random_state=42))
        sample_df = pd.concat(samples, ignore_index=True)
        texts     = sample_df["text"].tolist()
        labels    = sample_df["label"].tolist()

        print(f"      Computing gradient saliency on {len(texts)} samples...")

        try:
            top_words = _compute_token_importance_per_label(
                model, tokenizer, texts, labels
            )
            platform_top_words[platform] = top_words

            # Save top words per class
            for cls_name, words_df in top_words.items():
                if not words_df.empty:
                    words_df.to_csv(
                        os.path.join(
                            GRAD_DIR,
                            f"{model_key}_{platform}_{cls_name}_top_words.csv"
                        ),
                        index=False
                    )

        except Exception as e:
            print(f"      Failed for {model_key}/{platform}: {e}")
            platform_top_words[platform] = {}
            continue

    if platform_top_words:
        plot_cross_platform_gradient(model_key, platform_top_words)

    del model
    torch.cuda.empty_cache()
    return platform_top_words


def _compute_token_importance_per_label(model, tokenizer, texts, labels):
    """
    Token importance via gradient-based saliency, stratified by true label.

    This local implementation differs from ``utils.compute_token_importance``
    in that it iterates over each text individually and uses the true label
    to select the backprop target, returning a per-class attribution dict.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        Fine-tuned model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    texts : list[str]
        Input texts.
    labels : list[int]
        True class labels (one per text).

    Returns
    -------
    dict[str, dict[str, float]]
        ``{class_name: {token: mean_importance, ...}, ...}``
    """
    word_scores = {cls: {} for cls in CLASSES}

    model.eval()
    for text, label in zip(texts, labels):
        cls_name = CLASSES[label]

        inputs = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Get embeddings with gradient tracking
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        embeddings.retain_grad()

        # Forward pass
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"],
        )
        probs = torch.softmax(outputs.logits, dim=-1)

        # Backprop on predicted class score
        model.zero_grad()
        score = probs[0, label]
        score.backward()

        if embeddings.grad is None:
            continue

        # Gradient magnitude per token
        grad_magnitude = embeddings.grad[0].norm(dim=-1).detach().cpu().numpy()

        # Map back to tokens
        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0].cpu().numpy()
        )

        for tok, grad in zip(tokens, grad_magnitude):
            # Skip special tokens and short tokens
            if tok in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>",
                       "<pad>", "Ġ", "▁"]:
                continue
            clean = tok.replace("##", "").replace("Ġ", "").strip()
            if len(clean) < 3:
                continue
            if clean not in word_scores[cls_name]:
                word_scores[cls_name][clean] = []
            word_scores[cls_name][clean].append(float(grad))

    # Aggregate
    top_words = {}
    for cls_name, scores in word_scores.items():
        rows = [
            {"word": w, "mean_gradient": np.mean(v), "count": len(v)}
            for w, v in scores.items()
            if len(v) >= 3
        ]
        if not rows:
            top_words[cls_name] = pd.DataFrame()
            continue
        top_words[cls_name] = pd.DataFrame(rows).sort_values(
            "mean_gradient", ascending=False
        ).head(20)

    return top_words


def extract_top_words(gradient_values, texts, tokenizer, n_top=20):
    """
    Extract top words by mean gradient saliency magnitude per class.

    Parameters
    ----------
    gradient_values : object
        Gradient attribution values (structured with .values attribute).
    texts : list[str]
        Input texts.
    tokenizer : AutoTokenizer
        Tokenizer used to produce tokens.
    n_top : int
        Number of top words to retain per class.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{class_name: DataFrame with columns word, mean_gradient, count}``
    """
    top_words = {}

    for cls_idx, cls_name in enumerate(CLASSES):
        word_scores = {}

        for i, text in enumerate(texts):
            if i >= len(gradient_values):
                break
            tokens = tokenizer.tokenize(text,
                                        max_length=MAX_LEN,
                                        truncation=True)
            values = gradient_values.values[i, :len(tokens), cls_idx]

            for tok, val in zip(tokens, values):
                # Clean subword tokens
                clean = tok.replace("##", "").replace("Ġ", "").strip()
                if len(clean) < 2:
                    continue
                if clean not in word_scores:
                    word_scores[clean] = []
                word_scores[clean].append(abs(float(val)))

        # Aggregate — require at least 3 occurrences
        rows = [
            {"word": w, "mean_gradient": np.mean(scores),
             "count": len(scores)}
            for w, scores in word_scores.items()
            if len(scores) >= 3
        ]
        if not rows:
            top_words[cls_name] = pd.DataFrame()
            continue

        df = pd.DataFrame(rows).sort_values(
            "mean_gradient", ascending=False
        ).head(n_top)
        top_words[cls_name] = df

    return top_words


def plot_cross_platform_gradient(model_key, platform_top_words):
    """
    Side-by-side bar chart comparing top gradient saliency tokens
    for the depression class across Kaggle vs Reddit vs Twitter.
    Figure 5 — cross-platform feature disparity.
    """
    cls_name = "depression"   # Focus on depression — most clinically relevant
    colors   = {"kaggle": "#1976D2", "reddit": "#43A047", "twitter": "#FB8C00"}

    # Collect top words from each platform
    platform_dfs = {}
    for platform in ["kaggle", "reddit", "twitter"]:
        words_data = platform_top_words.get(platform, {})
        df = words_data.get(cls_name, pd.DataFrame())
        if not df.empty:
            platform_dfs[platform] = df.head(15)

    if not platform_dfs:
        return

    fig, axes = plt.subplots(1, len(platform_dfs),
                             figsize=(6 * len(platform_dfs), 7))
    if len(platform_dfs) == 1:
        axes = [axes]

    for idx, (platform, df) in enumerate(platform_dfs.items()):
        ax = axes[idx]
        ax.barh(
            df["word"],
            df["mean_gradient"],
            color=colors.get(platform, "steelblue"),
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_xlabel("Mean |Gradient Saliency|", fontsize=10)
        ax.set_title(
            f"{platform.upper()}\n(top {cls_name} features)",
            fontsize=11, fontweight="bold"
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Cross-Platform Gradient Saliency Attribution — {MODEL_DISPLAY[model_key]}\n"
        f"Depression class: which words drive predictions per platform",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR,
                        f"figure5_{model_key}_gradient_cross_platform.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


def plot_shap_summary_across_models():
    """
    Summary comparison: which model has the most stable gradient-based
    feature attribution across platforms.
    Computed as Jaccard similarity of top-10 word rankings.
    Figure 6.
    """
    # Load saved top word CSVs and compute overlap
    results = []
    for model_key in MODELS:
        for cls_name in CLASSES:
            try:
                kaggle_path  = os.path.join(
                    GRAD_DIR,
                    f"{model_key}_kaggle_{cls_name}_top_words.csv"
                )
                reddit_path  = os.path.join(
                    GRAD_DIR,
                    f"{model_key}_reddit_{cls_name}_top_words.csv"
                )
                twitter_path = os.path.join(
                    GRAD_DIR,
                    f"{model_key}_twitter_{cls_name}_top_words.csv"
                )

                if not all(os.path.exists(p) for p in
                           [kaggle_path, reddit_path, twitter_path]):
                    continue

                kaggle_words  = set(pd.read_csv(kaggle_path)["word"].head(10))
                reddit_words  = set(pd.read_csv(reddit_path)["word"].head(10))
                twitter_words = set(pd.read_csv(twitter_path)["word"].head(10))

                # Jaccard similarity
                kr_sim = len(kaggle_words & reddit_words) / len(
                    kaggle_words | reddit_words
                ) if kaggle_words | reddit_words else 0
                kt_sim = len(kaggle_words & twitter_words) / len(
                    kaggle_words | twitter_words
                ) if kaggle_words | twitter_words else 0

                results.append({
                    "model":       MODEL_DISPLAY[model_key],
                    "class":       cls_name,
                    "kaggle_reddit_sim":  round(kr_sim, 3),
                    "kaggle_twitter_sim": round(kt_sim, 3),
                    "avg_stability":      round((kr_sim + kt_sim) / 2, 3),
                })
            except Exception:
                continue

    if not results:
        return

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(GRAD_DIR, "feature_stability.csv"), index=False)

    # Plot
    pivot = df.pivot_table(
        index="model", columns="class",
        values="avg_stability", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, alpha=0.85, edgecolor="white")
    ax.set_title(
        "Feature Attribution Stability Across Platforms\n"
        "(Jaccard similarity of top-10 words: Kaggle vs Reddit/Twitter)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Average Jaccard Similarity", fontsize=11)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Class", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.6)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure6_feature_stability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("Starting gradient saliency attribution analysis...")
    print(f"Sample size per platform: {GRAD_SAMPLE_SIZE}")
    print("Note: gradient attribution is compute-intensive (~5–10 min per model).\n")

    # Run for all models; start with RoBERTa — best overall performer
    priority_models = ["roberta", "bert", "mentalbert", "mentalroberta"]

    for model_key in priority_models:
        print(f"\n{'='*50}")
        print(f"Gradient Saliency Attribution: {MODEL_DISPLAY[model_key]}")
        print(f"{'='*50}")
        run_shap_analysis(model_key, MODELS[model_key])

    # Cross-model feature stability summary
    print("\nGenerating feature stability summary...")
    plot_shap_summary_across_models()

    print(f"\n{'='*50}")
    print("GRADIENT SALIENCY ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Data saved to:    {GRAD_DIR}")


if __name__ == "__main__":
    main()