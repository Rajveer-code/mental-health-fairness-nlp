"""
02_preprocess.py
────────────────
Preprocess all three datasets and produce the canonical cross-platform splits.

This is a thin entry-point wrapper around src/preprocess.py.
All label remapping logic lives in src/preprocess.py — do not modify here.

Usage
-----
    python scripts/02_preprocess.py

Outputs
-------
data/splits/cross_platform/train.csv          n=35,556
data/splits/cross_platform/val.csv            n=7,620
data/splits/cross_platform/test_kaggle.csv    n=7,620
data/splits/cross_platform/test_reddit.csv    n=6,257
data/splits/cross_platform/test_twitter.csv   n=2,883

Dependencies
------------
Requires data/raw/kaggle_mental_health/Combined Data.csv
Requires data/raw/reddit_goemotions/ (run 01_download_data.py first)
Requires data/raw/twitter_emotion/   (run 01_download_data.py first)
"""

import sys
import os
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    splits_dir = "data/splits/cross_platform"
    expected_files = [
        "train.csv", "val.csv",
        "test_kaggle.csv", "test_reddit.csv", "test_twitter.csv",
    ]
    all_present = all(
        os.path.exists(os.path.join(splits_dir, f)) for f in expected_files
    )
    if all_present:
        log.info(
            "All split files already present in %s.\n"
            "Delete them and re-run if you need to regenerate.",
            splits_dir,
        )
        return

    log.info("Running src/preprocess.py to generate splits...")
    result = subprocess.run(
        [sys.executable, "src/preprocess.py"],
        check=True,
    )
    log.info("Preprocessing complete. Splits saved to %s.", splits_dir)
    log.info("Next step: python scripts/03_train_all_models.py")


if __name__ == "__main__":
    main()
