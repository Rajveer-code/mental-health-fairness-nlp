"""
07_attribution_stability.py
────────────────────────────
Gradient-based saliency attribution and cross-platform feature stability.

Reproduces Table 8 and Figures 6–9 of the paper:
  - Table 8: Jaccard similarity of top-K attribution token sets at K=10
  - Figures 6–9: Top-15 gradient saliency words per model × platform × class

Attribution method (Section 4.2.6, Equation 4):
  s_i = ||∂P(y|x) / ∂E_i||_2
  where E_i is the embedding of token i (Captum gradient saliency).

Per-platform samples: n=200 per platform (Section 4.2.6).

Jaccard stability (Equation 5):
  J_K(A, B) = |top_K(A) ∩ top_K(B)| / |top_K(A) ∪ top_K(B)|
  Reported at K ∈ {5, 10, 15, 20}.

NOTE: J≈0 in 13/16 model–class pairs at K=10 for Kaggle→Twitter comparisons
(Table 8). Near-zero values indicate attribution instability, not definitively
attribution collapse — see Section 7.0.3.

Usage
-----
    python scripts/07_attribution_stability.py

Outputs
-------
outputs/results/fairness/jaccard_full_analysis.csv      Table 8 base
outputs/results/fairness/jaccard_k_sensitivity.csv      K-sensitivity
outputs/results/shap/                                    Top-word CSVs (48 files)
outputs/figures/v2/                                     Attribution bar charts

Dependencies
------------
Requires outputs/results/{model}_{platform}_predictions.csv (run 04_evaluate first)
Requires model checkpoints in outputs/models/ (run 03_train_all_models.py first)
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
    log.info("Step 1/3: Gradient saliency + top-word extraction (src/shap_analysis.py)...")
    subprocess.run([sys.executable, "src/shap_analysis.py"], check=True)

    log.info("Step 2/3: Jaccard K-sensitivity analysis (src/jaccard_full_analysis.py)...")
    subprocess.run([sys.executable, "src/jaccard_full_analysis.py"], check=True)

    log.info("Step 3/3: Stress-class attribution detail (src/code_A2_stress_attribution.py)...")
    subprocess.run([sys.executable, "src/code_A2_stress_attribution.py"], check=True)

    log.info(
        "\nAttribution stability analysis complete.\n"
        "  → outputs/results/fairness/jaccard_full_analysis.csv  (Table 8)\n"
        "  → outputs/results/fairness/jaccard_k_sensitivity.csv\n"
        "  → outputs/results/shap/  (48 top-word CSVs)\n\n"
        "Next step: python scripts/08_sensitivity_analysis.py"
    )


if __name__ == "__main__":
    main()
