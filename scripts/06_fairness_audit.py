"""
06_fairness_audit.py
─────────────────────
Compute platform-stratified prediction equity metrics.

Reproduces Table 6 of the paper:
  - Symmetric Disparate Impact (DI) for all 4 proxy classes (Equation 2)
  - Equalized Odds Difference (EOD) — max across classes (Equation 3)
  - Prior-shift-adjusted DI (Section 5.5)

Fairness metric definitions (Section 4.2.4–4.2.5):
  DI_c = min(P(ŷ=c|G=Kaggle)/P(ŷ=c|G=target),
             P(ŷ=c|G=target)/P(ŷ=c|G=Kaggle))
  EOD_c = |TPR_c(Kaggle) - TPR_c(target)|   (absolute value, all classes)

Four-fifths rule: DI < 0.80 = violation; DI < 0.50 = severe disparity.
Reference platform for all comparisons: Kaggle.

Usage
-----
    python scripts/06_fairness_audit.py

Outputs
-------
outputs/results/fairness/di_eod_table.csv           Table 6

Dependencies
------------
Requires outputs/results/{model}_{platform}_predictions.csv (run 04_evaluate first)
"""

import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Computing DI and EOD (src/code_A1_di_eod_analysis.py)...")
    subprocess.run([sys.executable, "src/code_A1_di_eod_analysis.py"], check=True)

    log.info(
        "\nFairness audit complete.\n"
        "  → outputs/results/fairness/di_eod_table.csv  (Table 6)\n\n"
        "Next step: python scripts/07_attribution_stability.py"
    )


if __name__ == "__main__":
    main()
