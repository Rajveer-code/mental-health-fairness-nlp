"""
08_sensitivity_analysis.py
───────────────────────────
Label-mapping sensitivity analysis across four alternative mapping schemes.

Reproduces Table 10 and Figure 11 of the paper:
  - Table 10: Cross-platform AUC degradation under four label mappings (A–D)

The four mapping schemes (Section 4.4):
  A: 4-class primary  (normal / depression / anxiety / stress)  — main analysis
  B: Binary           (normal vs. any mental health condition)
  C: 3-class          (normal / depression / distress)
  D: Distress super   (anxiety and stress merged into single distress category)

No model retraining is required — ground-truth labels and predicted probabilities
from the primary analysis are remapped and re-evaluated.

Usage
-----
    python scripts/08_sensitivity_analysis.py

Outputs
-------
outputs/results/fairness/sensitivity_analysis.csv   Table 10

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
    log.info("Running label-mapping sensitivity analysis (src/sensitivity_analysis.py)...")
    subprocess.run([sys.executable, "src/sensitivity_analysis.py"], check=True)

    log.info(
        "\nSensitivity analysis complete.\n"
        "  → outputs/results/fairness/sensitivity_analysis.csv  (Table 10)\n\n"
        "Next step: python scripts/09_reproduce_all_tables.py"
    )


if __name__ == "__main__":
    main()
