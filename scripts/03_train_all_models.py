"""
03_train_all_models.py
───────────────────────
Fine-tune all four transformer models across five random seeds.

Runs src/train.py for each model × seed combination sequentially.
Checkpoints are saved to outputs/models/{model_key}_seed{seed}/.

Multi-seed protocol (Section 4.5 of paper):
  Seeds: [42, 0, 1, 7, 123]
  All Tables 2–10 report mean ± SD across these five seeds.

Usage
-----
    # All models, all 5 seeds (full run — ~4–8 hours on a single GPU)
    python scripts/03_train_all_models.py

    # Single model, single seed (for testing)
    python scripts/03_train_all_models.py --models bert --seeds 42

    # Skip training if checkpoints already exist
    python scripts/03_train_all_models.py --skip-existing

Outputs
-------
outputs/models/{model_key}_seed{seed}/   Checkpoint + tokenizer
outputs/results/{model_key}_history.csv  Per-epoch training metrics

Estimated runtime (NVIDIA A100 40GB):
  ~30 min per model per seed → ~10 hours total for all 4 × 5 = 20 runs

Dependencies
------------
Requires data/splits/cross_platform/ (run 02_preprocess.py first)
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_MODELS = ["bert", "roberta", "mentalbert", "mentalroberta"]
ALL_SEEDS = [42, 0, 1, 7, 123]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all CPFE models across seeds.")
    p.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Model(s) to train (default: all 4).",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=ALL_SEEDS,
        help="Seed(s) to use (default: all 5).",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip models whose checkpoint directory already exists.",
    )
    p.add_argument(
        "--config", default="configs/training.yaml",
        help="Path to training config YAML.",
    )
    return p.parse_args()


def checkpoint_exists(model_key: str, seed: int) -> bool:
    ckpt_dir = f"outputs/models/{model_key}_seed{seed}"
    return os.path.isdir(ckpt_dir) and bool(os.listdir(ckpt_dir))


def run_training(model_key: str, seed: int, config_path: str) -> None:
    cmd = [
        sys.executable, "src/train.py",
        "--model", model_key,
        "--seed", str(seed),
        "--config", config_path,
    ]
    log.info("Training %s seed=%d: %s", model_key, seed, " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    total = len(args.models) * len(args.seeds)
    done = 0
    skipped = 0

    for model_key in args.models:
        for seed in args.seeds:
            if args.skip_existing and checkpoint_exists(model_key, seed):
                log.info("Skipping %s seed=%d (checkpoint exists).", model_key, seed)
                skipped += 1
                continue

            run_training(model_key, seed, args.config)
            done += 1
            log.info(
                "Progress: %d/%d runs complete (%d skipped).",
                done + skipped, total, skipped,
            )

    log.info(
        "\nTraining complete. %d runs executed, %d skipped.\n"
        "Next step: python scripts/04_evaluate_crossplatform.py",
        done, skipped,
    )


if __name__ == "__main__":
    main()
