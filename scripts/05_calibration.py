"""
05_calibration.py
─────────────────
Run temperature scaling calibration and ECE bootstrap CI analysis.

Reproduces Tables 2 (ECE column), 3 (ECE column), and 7 of the paper:
  - Table 7: Optimal T*, ECE before/after, ECE reduction %, ΔAUC

Temperature scaling details (Section 4.3):
  - T* minimises NLL on a 10% stratified calibration split of the target test set
  - Operates on raw pre-softmax logits (saved by evaluate.py)
  - AUC is invariant to temperature scaling by construction

ECE bootstrap CI details (Section 4.2.2):
  - B=1,000 bootstrap resamples per model × platform
  - 95% CIs reported (2.5th–97.5th percentile)

Usage
-----
    python scripts/05_calibration.py

Outputs
-------
outputs/results/fairness/temperature_scaling_results.csv  Table 7
outputs/results/fairness/ece_bootstrap_cis.csv            ECE CIs (Table 2/3)
outputs/results/fairness/perclass_ece.csv                 Per-class ECE (Table 2/3)
outputs/results/fairness/ece_binning_sensitivity.csv      Appendix: M sensitivity

Dependencies
------------
Requires outputs/results/{model}_{platform}_predictions.csv (run 04_evaluate first)
Prediction CSVs must include logit_* columns (produced by evaluate.py v2).
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
    log.info("Step 1/3: Temperature scaling calibration (src/code_A5_temperature_scaling.py)...")
    subprocess.run([sys.executable, "src/code_A5_temperature_scaling.py"], check=True)

    log.info("Step 2/3: ECE bootstrap CIs (src/code_A3_A4_A6_ece_jaccard.py — A3)...")
    subprocess.run([sys.executable, "src/code_A3_A4_A6_ece_jaccard.py"], check=True)

    log.info("Step 3/3: Per-class ECE analysis (src/perclass_ece_analysis.py)...")
    subprocess.run([sys.executable, "src/perclass_ece_analysis.py"], check=True)

    log.info(
        "\nCalibration analysis complete.\n"
        "  → outputs/results/fairness/temperature_scaling_results.csv  (Table 7)\n"
        "  → outputs/results/fairness/ece_bootstrap_cis.csv\n"
        "  → outputs/results/fairness/perclass_ece.csv\n\n"
        "Next step: python scripts/06_fairness_audit.py"
    )


if __name__ == "__main__":
    main()
