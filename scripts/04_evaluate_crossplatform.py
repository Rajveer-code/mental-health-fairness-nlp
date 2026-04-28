"""
04_evaluate_crossplatform.py
──────────────────────────────
Run inference on all three test platforms and compute primary metrics.

Reproduces Tables 2, 3, 4, and 5 of the paper:
  - Table 2: Within-platform performance (Kaggle)
  - Table 3: Cross-platform AUC, F1-macro, ECE, ΔAUC%
  - Table 4: Per-class AUC with DeLong 95% CIs
  - Table 5: Pairwise macro-AUC significance tests (Bonferroni-corrected)

Usage
-----
    python scripts/04_evaluate_crossplatform.py [--seed 42]

Outputs
-------
outputs/results/{model}_{platform}_predictions.csv  Per-sample predictions + logits
outputs/results/master_results.csv                  Table 2 + 3 summary
outputs/results/fairness/fairness_audit_full.csv    Full audit including DeLong CIs
outputs/results/fairness/pairwise_auc_comparisons.csv  Table 5

Dependencies
------------
Requires outputs/models/{model}_seed{seed}/ (run 03_train_all_models.py first)
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all models cross-platform.")
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed of the checkpoint to evaluate (default: 42).",
    )
    p.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to master config YAML.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Step 1/2: Running cross-platform inference (src/evaluate.py)...")
    subprocess.run(
        [sys.executable, "src/evaluate.py", "--seed", str(args.seed)],
        check=True,
    )

    log.info("Step 2/2: Running full fairness audit (src/fairness_audit.py)...")
    subprocess.run(
        [sys.executable, "src/fairness_audit.py"],
        check=True,
    )

    log.info(
        "\nEvaluation complete.\n"
        "  → outputs/results/master_results.csv          (Tables 2, 3)\n"
        "  → outputs/results/fairness/fairness_audit_full.csv\n"
        "  → outputs/results/fairness/pairwise_auc_comparisons.csv  (Table 5)\n\n"
        "Next step: python scripts/05_calibration.py"
    )


if __name__ == "__main__":
    main()
