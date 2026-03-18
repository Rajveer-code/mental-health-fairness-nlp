"""
train.py
--------
Fine-tunes four transformer models on the Kaggle mental health dataset.
Models: BERT-base, RoBERTa-base, MentalBERT, MentalRoBERTa

Saves model checkpoints to outputs/models/
Saves training metrics to outputs/results/

Usage:
    python src/train.py --model bert
    python src/train.py --model roberta
    python src/train.py --model mentalbert
    python src/train.py --model mentalroberta
    python src/train.py --model all
"""

import os
import json
import time
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["training"]["seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_REGISTRY = {
    "bert":          cfg["models"]["bert"],
    "roberta":       cfg["models"]["roberta"],
    "mentalbert":    cfg["models"]["mental_bert"],
    "mentalroberta": cfg["models"]["mental_roberta"],
}

NUM_LABELS  = 4
LABEL_NAMES = ["normal", "depression", "anxiety", "stress"]
MAX_LEN     = cfg["training"]["max_length"]
BATCH_SIZE  = cfg["training"]["batch_size"]
LR          = cfg["training"]["learning_rate"]
EPOCHS      = cfg["training"]["epochs"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Dataset class ─────────────────────────────────────────────────────────────

class MentalHealthDataset(Dataset):
    """PyTorch Dataset wrapper for mental health text data."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs):
    """
    Compute full evaluation metrics.
    Returns dict with accuracy, macro F1, weighted F1,
    per-class F1, and macro AUC-ROC.
    """
    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_per = f1_score(labels, preds, average=None,       zero_division=0)

    # AUC — requires probability scores
    try:
        auc = roc_auc_score(
            labels, probs,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        auc = float("nan")

    report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        zero_division=0
    )

    return {
        "accuracy":    round(acc,    4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wt,  4),
        "f1_per_class": {
            LABEL_NAMES[i]: round(float(f1_per[i]), 4)
            for i in range(len(f1_per))
        },
        "auc_macro":   round(auc, 4),
        "report":      report,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Training", leave=False):
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    """
    Evaluate model on a dataloader.
    Returns metrics dict + raw predictions and probabilities.
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            probs  = torch.softmax(logits, dim=-1)
            preds  = torch.argmax(probs, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )
    return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Main training function ────────────────────────────────────────────────────

def train_model(model_key: str):
    """
    Full training pipeline for one model.
    Loads data, trains for EPOCHS, saves best checkpoint.
    """
    model_name = MODEL_REGISTRY[model_key]
    print(f"\n{'='*60}")
    print(f"Training: {model_key.upper()}")
    print(f"Model:    {model_name}")
    print(f"Epochs:   {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"{'='*60}")

    # ── Load data ──
    train_df = pd.read_csv("data/splits/cross_platform/train.csv")
    val_df   = pd.read_csv("data/splits/cross_platform/val.csv")

    print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,}")

    # ── Tokenizer ──
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = MentalHealthDataset(train_df, tokenizer, MAX_LEN)
    val_dataset   = MentalHealthDataset(val_df,   tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ── Model ──
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model = model.to(DEVICE)

    # ── Optimizer + scheduler ──
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = total_steps // 10   # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ──
    best_val_f1  = 0.0
    best_epoch   = 0
    history      = []
    out_model_dir = os.path.join(cfg["paths"]["models"], model_key)
    os.makedirs(out_model_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_metrics, _, _, _ = evaluate(model, val_loader)

        elapsed = round(time.time() - t0, 1)
        print(f"  Loss: {train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1_macro']:.4f} | "
              f"Val AUC: {val_metrics['auc_macro']:.4f} | "
              f"Time: {elapsed}s")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            **{f"val_{k}": v for k, v in val_metrics.items()
               if k not in ["report", "f1_per_class"]},
        })

        # Save best model
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch  = epoch
            model.save_pretrained(out_model_dir)
            tokenizer.save_pretrained(out_model_dir)
            print(f"  ✓ Best model saved (F1: {best_val_f1:.4f})")

    # ── Save training history ──
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(results_dir, f"{model_key}_history.csv"),
        index=False
    )

    # Save best metrics summary
    summary = {
        "model_key":    model_key,
        "model_name":   model_name,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
        "trained_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "device":       str(DEVICE),
        "train_samples": len(train_df),
        "val_samples":   len(val_df),
    }
    with open(os.path.join(results_dir, f"{model_key}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete: {model_key.upper()}")
    print(f"Best epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {out_model_dir}")
    print(f"{'='*60}")

    return best_val_f1


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mental health NLP models")
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "roberta", "mentalbert", "mentalroberta", "all"],
        help="Which model to train"
    )
    args = parser.parse_args()

    if args.model == "all":
        results = {}
        for key in MODEL_REGISTRY:
            results[key] = train_model(key)
        print("\n\nFINAL RESULTS SUMMARY")
        print("="*40)
        for key, f1 in results.items():
            print(f"  {key:<15} Val F1: {f1:.4f}")
    else:
        train_model(args.model)