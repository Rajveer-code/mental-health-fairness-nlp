"""
src
───
Top-level package for the CPFE mental health NLP analysis.

All shared constants, loaders, and metrics are in :mod:`src.utils`.
Import from there rather than from individual analysis scripts.

Example
-------
>>> from src.utils import MODELS, PLATFORMS, compute_aggregate_ece
"""
from src.utils import (
    MODELS,
    PLATFORMS,
    CLASSES,
    CLASS_IDS,
    PROB_COLS,
    MODEL_DISPLAY,
    MODEL_HF_IDS,
    PLATFORM_COLORS,
    load_config,
    load_predictions,
    compute_aggregate_ece,
    compute_macro_auc,
    bootstrap_ci,
)

__all__ = [
    "MODELS",
    "PLATFORMS",
    "CLASSES",
    "CLASS_IDS",
    "PROB_COLS",
    "MODEL_DISPLAY",
    "MODEL_HF_IDS",
    "PLATFORM_COLORS",
    "load_config",
    "load_predictions",
    "compute_aggregate_ece",
    "compute_macro_auc",
    "bootstrap_ci",
]
