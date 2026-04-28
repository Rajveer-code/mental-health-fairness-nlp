"""
01_download_data.py
───────────────────
Download and cache all datasets required for the CPFE study.

This script downloads:
  1. GoEmotions (Reddit cross-platform test) from HuggingFace
  2. dair-ai/emotion (Twitter cross-platform test) from HuggingFace
  3. Prints instructions for the Kaggle dataset (requires manual download)

Usage
-----
    python scripts/01_download_data.py

Outputs
-------
data/raw/reddit_goemotions/   HuggingFace DatasetDict (arrow format)
data/raw/twitter_emotion/     HuggingFace DatasetDict (arrow format)
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def download_goemotions(save_dir: str = "data/raw/reddit_goemotions") -> None:
    """Download the GoEmotions dataset (Reddit cross-platform test)."""
    if os.path.exists(save_dir):
        log.info("GoEmotions already cached at %s — skipping.", save_dir)
        return

    log.info("Downloading GoEmotions from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("go_emotions", "simplified")
        ds.save_to_disk(save_dir)
        log.info("GoEmotions saved to %s", save_dir)
    except Exception as exc:
        log.error("Failed to download GoEmotions: %s", exc)
        raise


def download_dairemo(save_dir: str = "data/raw/twitter_emotion") -> None:
    """Download the dair-ai/emotion dataset (Twitter cross-platform test)."""
    if os.path.exists(save_dir):
        log.info("dair-ai/emotion already cached at %s — skipping.", save_dir)
        return

    log.info("Downloading dair-ai/emotion from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("dair-ai/emotion")
        ds.save_to_disk(save_dir)
        log.info("dair-ai/emotion saved to %s", save_dir)
    except Exception as exc:
        log.error("Failed to download dair-ai/emotion: %s", exc)
        raise


def check_kaggle() -> None:
    """Check for Kaggle dataset and print instructions if missing."""
    kaggle_path = "data/raw/kaggle_mental_health/Combined Data.csv"
    if os.path.exists(kaggle_path):
        log.info("Kaggle dataset found at %s.", kaggle_path)
        return

    log.warning(
        "Kaggle dataset NOT found at %s.\n\n"
        "To download it manually:\n"
        "  1. Create a Kaggle account and install the CLI: pip install kaggle\n"
        "  2. Place your API key at ~/.kaggle/kaggle.json\n"
        "  3. Run:\n"
        "     kaggle datasets download suchintikasarkar/sentiment-analysis-for-mental-health\n"
        "     unzip sentiment-analysis-for-mental-health.zip -d data/raw/kaggle_mental_health/\n\n"
        "Dataset URL: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health",
        kaggle_path,
    )


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)
    download_goemotions()
    download_dairemo()
    check_kaggle()

    log.info(
        "\nAll HuggingFace datasets downloaded.\n"
        "If the Kaggle dataset is missing, follow the instructions above.\n"
        "Next step: python scripts/02_preprocess.py"
    )


if __name__ == "__main__":
    main()
