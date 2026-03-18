"""
shap_analysis.py
----------------
SHAP-based feature attribution analysis.

For each model, computes SHAP values on the Kaggle test set
and cross-platform test sets to identify which linguistic
features drive prediction disparities across platforms.

Saves:
  - SHAP summary plots per model
  - Cross-platform feature importance comparison
  - Top features driving platform disparities

Usage:
    python src/shap_analysis.py
"""

import os
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap

warnings.filterwarnings("ignore")

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["normal", "depression", "anxiety", "stress"]
NUM_LABELS  = 4
MAX_LEN     = cfg["training"]["max_length"]
FIGURES_DIR = cfg["paths"]["figures"]
RESULTS_DIR = cfg["paths"]["results"]
SHAP_DIR    = os.path.join(RESULTS_DIR, "shap")
os.makedirs(SHAP_DIR,    exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = {
    "bert":          "outputs/models/bert",
    "roberta":       "outputs/models/roberta",
    "mentalbert":    "outputs/models/mentalbert",
    "mentalroberta": "outputs/models/mentalroberta",
}
MODEL_DISPLAY = {
    "bert":          "BERT",
    "roberta":       "RoBERTa",
    "mentalbert":    "DistilRoBERTa",
    "mentalroberta": "SamLowe-RoBERTa",
}
TEST_SETS = {
    "kaggle":  "data/splits/cross_platform/test_kaggle.csv",
    "reddit":  "data/splits/cross_platform/test_reddit.csv",
    "twitter": "data/splits/cross_platform/test_twitter.csv",
}

# Use small sample for SHAP — it is computationally expensive
SHAP_SAMPLE_SIZE  = 200
SHAP_BG_SIZE      = 50   # background samples for explainer


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
    Run SHAP analysis for one model across all platforms.
    Uses shap.Explainer with the tokenizer-aware partition explainer.
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

    predict_fn = get_prediction_function(model, tokenizer)

    platform_top_words = {}

    for platform, test_path in TEST_SETS.items():
        print(f"    Platform: {platform}")
        df = pd.read_csv(test_path)

        # Sample texts — stratified by label for balance
        samples = []
        for label_id in range(NUM_LABELS):
            subset = df[df["label"] == label_id]
            n      = min(SHAP_SAMPLE_SIZE // NUM_LABELS, len(subset))
            samples.append(subset.sample(n, random_state=42))
        sample_df = pd.concat(samples, ignore_index=True)
        texts     = sample_df["text"].tolist()

        # Background sample for explainer
        bg_texts = df.sample(
            min(SHAP_BG_SIZE, len(df)), random_state=42
        )["text"].tolist()

        print(f"      Running SHAP on {len(texts)} samples...")

        try:
            # Partition explainer — works with any black-box model
            explainer   = shap.Explainer(predict_fn, bg_texts)
            shap_values = explainer(texts, max_evals=200, batch_size=16)

            # shap_values.values shape: (n_samples, n_tokens, n_classes)
            # Extract token-level importance
            token_importance = {}
            for cls_idx, cls_name in enumerate(LABEL_NAMES):
                # Mean absolute SHAP across samples for this class
                cls_shap = np.abs(shap_values.values[:, :, cls_idx])
                # Average per token position
                mean_abs = cls_shap.mean(axis=0)
                token_importance[cls_name] = mean_abs

            # Save raw values
            np.save(
                os.path.join(SHAP_DIR,
                             f"{model_key}_{platform}_shap.npy"),
                shap_values.values
            )

            # Get top words per class
            top_words = extract_top_words(
                shap_values, texts, tokenizer, n_top=20
            )
            platform_top_words[platform] = top_words

            # Save top words
            for cls_name, words_df in top_words.items():
                words_df.to_csv(
                    os.path.join(SHAP_DIR,
                                 f"{model_key}_{platform}_{cls_name}_top_words.csv"),
                    index=False
                )

        except Exception as e:
            print(f"      SHAP failed for {model_key}/{platform}: {e}")
            platform_top_words[platform] = {}
            continue

    # Generate cross-platform comparison plot
    if platform_top_words:
        plot_cross_platform_shap(
            model_key, platform_top_words
        )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return platform_top_words


def extract_top_words(shap_values, texts, tokenizer, n_top=20):
    """
    Extract top words by mean absolute SHAP value per class.
    Returns dict of {class_name: DataFrame with word, mean_shap}
    """
    top_words = {}

    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        word_scores = {}

        for i, text in enumerate(texts):
            if i >= len(shap_values):
                break
            tokens = tokenizer.tokenize(text,
                                        max_length=MAX_LEN,
                                        truncation=True)
            values = shap_values.values[i, :len(tokens), cls_idx]

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
            {"word": w, "mean_shap": np.mean(scores),
             "count": len(scores)}
            for w, scores in word_scores.items()
            if len(scores) >= 3
        ]
        if not rows:
            top_words[cls_name] = pd.DataFrame()
            continue

        df = pd.DataFrame(rows).sort_values(
            "mean_shap", ascending=False
        ).head(n_top)
        top_words[cls_name] = df

    return top_words


def plot_cross_platform_shap(model_key, platform_top_words):
    """
    Side-by-side bar chart comparing top SHAP words
    for depression class across Kaggle vs Reddit vs Twitter.
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
            df["mean_shap"],
            color=colors.get(platform, "steelblue"),
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title(
            f"{platform.upper()}\n(top {cls_name} features)",
            fontsize=11, fontweight="bold"
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Cross-Platform SHAP Feature Attribution — {MODEL_DISPLAY[model_key]}\n"
        f"Depression class: which words drive predictions per platform",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR,
                        f"figure5_{model_key}_shap_cross_platform.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


def plot_shap_summary_across_models():
    """
    Summary comparison: which model has the most stable
    feature attribution across platforms.
    Computed as cosine similarity of top-word rankings.
    Figure 6.
    """
    # Load saved top word CSVs and compute overlap
    results = []
    for model_key in MODELS:
        for cls_name in LABEL_NAMES:
            try:
                kaggle_path  = os.path.join(
                    SHAP_DIR,
                    f"{model_key}_kaggle_{cls_name}_top_words.csv"
                )
                reddit_path  = os.path.join(
                    SHAP_DIR,
                    f"{model_key}_reddit_{cls_name}_top_words.csv"
                )
                twitter_path = os.path.join(
                    SHAP_DIR,
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
    df.to_csv(os.path.join(SHAP_DIR, "feature_stability.csv"), index=False)

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
    print("Starting SHAP analysis...")
    print(f"Sample size per platform: {SHAP_SAMPLE_SIZE}")
    print(f"Background size: {SHAP_BG_SIZE}")
    print("Note: SHAP is slow — ~5-10 mins per model. Do not interrupt.\n")

    # Run SHAP for all models
    # Start with RoBERTa — best overall model
    priority_models = ["roberta", "bert", "mentalbert", "mentalroberta"]

    for model_key in priority_models:
        print(f"\n{'='*50}")
        print(f"SHAP Analysis: {MODEL_DISPLAY[model_key]}")
        print(f"{'='*50}")
        run_shap_analysis(model_key, MODELS[model_key])

    # Cross-model feature stability summary
    print("\nGenerating feature stability summary...")
    plot_shap_summary_across_models()

    print(f"\n{'='*50}")
    print("SHAP ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Data saved to:    {SHAP_DIR}")
    print("\nNext step: notebooks/02_results_figures.ipynb")


if __name__ == "__main__":
    main()