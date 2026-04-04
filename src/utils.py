"""
utils.py
────────
Canonical shared constants and utility functions for the mental health
NLP fairness audit codebase.

Every other script in src/ must import from this module rather than
redefine these values locally.  Any function or constant that is used
by more than one script belongs here.

Inputs
------
configs/config.yaml
    Single source of truth for all paths, hyperparameters, and seeds.

Outputs
-------
None — this module is imported, not run directly.

Usage
-----
    from utils import (
        MODELS, PLATFORMS, CLASSES, MODEL_DISPLAY,
        load_config, load_predictions, compute_aggregate_ece,
    )

Dependencies
------------
numpy, pandas, scikit-learn, PyTorch, transformers, PyYAML
"""

import os
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Canonical Constants ───────────────────────────────────────────────────────

MODELS: list[str] = ["bert", "roberta", "mentalbert", "mentalroberta"]
PLATFORMS: list[str] = ["kaggle", "reddit", "twitter"]
CLASSES: list[str] = ["normal", "depression", "anxiety", "stress"]
CLASS_IDS: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
PROB_COLS: list[str] = [
    "prob_normal",
    "prob_depression",
    "prob_anxiety",
    "prob_stress",
]

MODEL_DISPLAY: dict[str, str] = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "mentalbert": "DistilRoBERTa",
    "mentalroberta": "SamLowe-RoBERTa",
}

MODEL_HF_IDS: dict[str, str] = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "mentalbert": "j-hartmann/emotion-english-distilroberta-base",
    "mentalroberta": "SamLowe/roberta-base-go_emotions",
}

PLATFORM_COLORS: dict[str, str] = {
    "kaggle": "#3366CC",
    "reddit": "#2DA44E",
    "twitter": "#E8A838",
}

# ── Shared Loaders ────────────────────────────────────────────────────────────


def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Load the project configuration from a YAML file.

    All scripts must be run from the repository root so that the default
    relative path resolves correctly.

    Parameters
    ----------
    path : str, optional
        Path to the YAML config file.  Default is ``"configs/config.yaml"``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at ``path``.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_predictions(
    model_key: str,
    platform: str,
    results_dir: str,
) -> pd.DataFrame | None:
    """
    Load the canonical per-sample prediction CSV for a model × platform pair.

    Parameters
    ----------
    model_key : str
        One of ``MODELS`` (e.g. ``"bert"``).
    platform : str
        One of ``PLATFORMS`` (e.g. ``"kaggle"``).
    results_dir : str
        Directory that contains ``{model_key}_{platform}_predictions.csv``.
        Typically ``cfg["paths"]["results"]``.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: label, pred, prob_normal, prob_depression,
        prob_anxiety, prob_stress, correct.
        Returns ``None`` and prints a warning if the file does not exist.
    """
    path = os.path.join(results_dir, f"{model_key}_{platform}_predictions.csv")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    return pd.read_csv(path)


def find_platform_file(
    platform: str,
    data_dir: str,
) -> str | None:
    """
    Locate the test CSV for a given platform, searching several candidate paths.

    Tries paths in order and returns the first one that exists.  This
    accommodates the two path conventions used across analysis scripts.

    Parameters
    ----------
    platform : str
        One of ``PLATFORMS`` (e.g. ``"reddit"``).
    data_dir : str
        Base data directory, typically ``cfg["paths"]["splits"]`` or
        ``cfg["paths"]["data"]``.

    Returns
    -------
    str or None
        Absolute or relative path to the found CSV, or ``None`` if not found.
    """
    candidates = [
        os.path.join(data_dir, "splits", "cross_platform", f"test_{platform}.csv"),
        os.path.join(data_dir, "cross_platform", f"test_{platform}.csv"),
        os.path.join(data_dir, f"test_{platform}.csv"),
        os.path.join("data", "splits", "cross_platform", f"test_{platform}.csv"),
        os.path.join("data", f"test_{platform}.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    print(f"  WARNING: Could not find data file for '{platform}'. Tried:")
    for candidate in candidates:
        print(f"    {candidate}")
    return None


def get_model_checkpoint(
    model_key: str,
    models_dir: str,
) -> str | None:
    """
    Locate the fine-tuned model checkpoint directory for a given model key.

    Parameters
    ----------
    model_key : str
        One of ``MODELS`` (e.g. ``"mentalbert"``).
    models_dir : str
        Root directory containing per-model subdirectories.
        Typically ``cfg["paths"]["models"]``.

    Returns
    -------
    str or None
        Path to the checkpoint directory, or ``None`` if not found.
    """
    primary = os.path.join(models_dir, model_key)
    if os.path.isdir(primary):
        return primary
    print(f"  WARNING: Checkpoint not found for '{model_key}' at {primary}")
    return None


# ── Shared Metrics ────────────────────────────────────────────────────────────


def compute_aggregate_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    M: int = 10,
) -> float:
    """
    Compute aggregate Expected Calibration Error (ECE).

    Uses max-probability confidence and argmax prediction, binned into
    M equal-width intervals.  This is Equation 5 in the paper.

    Parameters
    ----------
    probs : np.ndarray of shape (n_samples, n_classes)
        Predicted probability distributions.
    labels : np.ndarray of shape (n_samples,)
        Integer ground-truth labels in [0, n_classes).
    M : int, optional
        Number of calibration bins.  Default is 10.

    Returns
    -------
    float
        ECE in [0, 1].  Lower is better.  0 = perfect calibration.

    Notes
    -----
    In imbalanced settings, aggregate ECE is dominated by the majority
    class.  Use ``compute_perclass_ece`` (in perclass_ece_analysis.py)
    for minority-class assessment.
    """
    n = len(labels)
    ece = 0.0
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, M + 1)

    for i in range(M):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(np.mean(preds[mask] == labels[mask]))
        conf = float(np.mean(max_probs[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)

    return round(float(ece), 4)


def compute_macro_auc(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute one-vs-rest macro-averaged AUC.

    Parameters
    ----------
    probs : np.ndarray of shape (n_samples, n_classes)
        Predicted probability distributions.
    labels : np.ndarray of shape (n_samples,)
        Integer ground-truth labels.

    Returns
    -------
    float
        Macro OvR AUC, or ``float("nan")`` if computation fails (e.g.
        a class is absent from ``labels``).
    """
    try:
        return float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")


def bootstrap_ci(
    fn: Callable[..., float],
    probs: np.ndarray,
    labels: np.ndarray,
    n_boots: int = 1000,
    **kwargs,
) -> tuple[float, float]:
    """
    Compute a 95 % bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    fn : callable
        Metric function with signature ``fn(probs, labels, **kwargs) -> float``.
    probs : np.ndarray of shape (n_samples, n_classes)
        Predicted probability distributions.
    labels : np.ndarray of shape (n_samples,)
        Integer ground-truth labels.
    n_boots : int, optional
        Number of bootstrap resamples.  Default is 1000.
    **kwargs
        Additional keyword arguments forwarded to ``fn``.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` — the 2.5th and 97.5th percentiles of the
        bootstrap distribution.
    """
    rng = np.random.default_rng()
    boot_vals: list[float] = []
    n = len(labels)
    for _ in range(n_boots):
        idx = rng.integers(0, n, size=n)
        boot_vals.append(fn(probs[idx], labels[idx], **kwargs))
    lo = float(np.percentile(boot_vals, 2.5))
    hi = float(np.percentile(boot_vals, 97.5))
    return lo, hi


# ── Shared Attribution ────────────────────────────────────────────────────────


def compute_token_importance(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    target_class_idx: int,
    max_length: int = 64,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Compute gradient-based token saliency scores for a target class.

    Implements Equations 8–9 from the paper:

    .. math::

        s_i = \\|\\partial P(y|x) / \\partial E_i\\|_2

        \\bar{S}(w) = \\frac{1}{N_w} \\sum_i s_i

    Only tokens appearing at least 3 times across all texts are retained
    to avoid noise from hapax legomena.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        Fine-tuned HuggingFace classification model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    texts : list[str]
        Input texts to attribute.
    target_class_idx : int
        Class index for which gradients are computed.
    max_length : int, optional
        Tokeniser truncation length.  Default is 64.
    batch_size : int, optional
        Tokenisation batch size.  Default is 16.
    device : str, optional
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).  Default is
        ``"cpu"``.  Callers should pass their module-level ``DEVICE`` variable
        to enable GPU acceleration.

    Returns
    -------
    dict[str, float]
        Mapping from cleaned token string to mean gradient magnitude,
        for tokens with at least 3 occurrences across ``texts``.
    """
    model.eval()
    word_scores: dict[str, list[float]] = defaultdict(list)
    n_success = 0

    _special = frozenset(
        ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", "Ġ", "▁"]
    )

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        ).to(device)

        for j in range(len(batch_texts)):
            single_input = {k: v[j : j + 1] for k, v in inputs.items()}

            try:
                embeddings = model.get_input_embeddings()(
                    single_input["input_ids"]
                )
                embeddings.retain_grad()
                embeddings = embeddings.requires_grad_(True)

                outputs = model(
                    inputs_embeds=embeddings,
                    attention_mask=single_input["attention_mask"],
                )
                prob = torch.softmax(outputs.logits, dim=-1)[0, target_class_idx]
                model.zero_grad()
                prob.backward()

                if embeddings.grad is None:
                    continue

                importance = (
                    embeddings.grad[0].norm(dim=-1).detach().cpu().numpy()
                )
                token_ids = single_input["input_ids"][0].cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)

                for tok, imp in zip(tokens, importance):
                    if tok in _special:
                        continue
                    clean = tok.lstrip("##").lstrip("Ġ").lstrip("▁").lower()
                    if len(clean) > 1:
                        word_scores[clean].append(float(imp))

                n_success += 1

            except Exception:  # noqa: BLE001 — skip malformed samples silently
                continue

    print(f"    Attribution computed for {n_success}/{len(texts)} samples")
    return {
        w: float(np.mean(scores))
        for w, scores in word_scores.items()
        if len(scores) >= 3
    }
