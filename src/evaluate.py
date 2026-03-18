"""
evaluate.py
-----------
Evaluates all trained models on three test sets:
  1. test_kaggle.csv   — within-platform (same distribution as training)
  2. test_reddit.csv   — cross-platform test 1
  3. test_twitter.csv  — cross-platform test 2

Saves per-model per-platform predictions to outputs/results/
These are used by fairness_audit.py in the next step.

Usage:
    python src/evaluate.py
"""

import os
import json
import yaml
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS  = 4
LABEL_NAMES = ["normal", "depression", "anxiety", "stress"]
MAX_LEN     = cfg["training"]["max_length"]
BATCH_SIZE  = 32   # larger batch is fine for inference, no gradients

MODELS = {
    "bert":          "outputs/models/bert",
    "roberta":       "outputs/models/roberta",
    "mentalbert":    "outputs/models/mentalbert",
    "mentalroberta": "outputs/models/mentalroberta",
}

TEST_SETS = {
    "kaggle":  "data/splits/cross_platform/test_kaggle.csv",
    "reddit":  "data/splits/cross_platform/test_reddit.csv",
    "twitter": "data/splits/cross_platform/test_twitter.csv",
}

print(f"Device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts     = df["text"].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, df):
    """
    Run model on a dataframe.
    Returns: labels, predictions, probabilities (all numpy arrays)
    """
    dataset = InferenceDataset(df, tokenizer, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="    Inferring", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_labels.extend(batch["label"].numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs):
    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_per = f1_score(labels, preds, average=None,       zero_division=0)
    try:
        auc = roc_auc_score(labels, probs,
                            multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy":    round(float(acc),    4),
        "f1_macro":    round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_wt),  4),
        "auc_macro":   round(float(auc),    4),
        "f1_per_class": {
            LABEL_NAMES[i]: round(float(f1_per[i]), 4)
            for i in range(len(f1_per))
        },
        "classification_report": classification_report(
            labels, preds,
            target_names=LABEL_NAMES,
            zero_division=0
        )
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    # Master results table — rows = model x platform
    master_rows = []

    for model_key, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_key.upper()}")
        print(f"{'='*60}")

        # Load model and tokenizer from saved checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
        ).to(DEVICE)

        model_results = {}

        for platform, test_path in TEST_SETS.items():
            print(f"\n  Platform: {platform.upper()} "
                  f"({test_path})")
            df = pd.read_csv(test_path)
            print(f"  Samples: {len(df):,}")

            labels, preds, probs = run_inference(model, tokenizer, df)
            metrics = compute_metrics(labels, preds, probs)

            print(f"  Accuracy: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1_macro']:.4f} | "
                  f"AUC: {metrics['auc_macro']:.4f}")
            print(f"  Per-class F1: {metrics['f1_per_class']}")

            # Save raw predictions + probabilities for fairness audit
            pred_df = df.copy()
            pred_df["pred"]  = preds
            pred_df["prob_normal"]     = probs[:, 0]
            pred_df["prob_depression"] = probs[:, 1]
            pred_df["prob_anxiety"]    = probs[:, 2]
            pred_df["prob_stress"]     = probs[:, 3]
            pred_df["correct"] = (preds == labels).astype(int)

            pred_path = os.path.join(
                results_dir,
                f"{model_key}_{platform}_predictions.csv"
            )
            pred_df.to_csv(pred_path, index=False)

            model_results[platform] = metrics

            master_rows.append({
                "model":       model_key,
                "platform":    platform,
                "accuracy":    metrics["accuracy"],
                "f1_macro":    metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "auc_macro":   metrics["auc_macro"],
                **{f"f1_{k}": v
                   for k, v in metrics["f1_per_class"].items()},
            })

        # Save per-model results JSON
        with open(os.path.join(results_dir,
                               f"{model_key}_eval.json"), "w") as f:
            json.dump(model_results, f, indent=2)

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # Save master results table
    master_df = pd.DataFrame(master_rows)
    master_path = os.path.join(results_dir, "master_results.csv")
    master_df.to_csv(master_path, index=False)

    # Print final summary table
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE — MASTER RESULTS TABLE")
    print(f"{'='*60}")
    print(master_df.to_string(index=False))
    print(f"\nSaved to: {master_path}")
    print("\nNext step: python src/fairness_audit.py")


if __name__ == "__main__":
    main()