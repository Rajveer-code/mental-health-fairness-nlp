"""
src.evaluation
──────────────
Evaluation metrics subpackage (discriminative performance, calibration,
statistical significance, fairness).

Canonical shared metrics are in :mod:`src.utils`:
  - compute_aggregate_ece()  — ECE with M=10 equal-width bins (Eq. 1)
  - compute_macro_auc()      — macro one-vs-rest AUC
  - bootstrap_ci()           — bootstrap 95% CI for any scalar metric

CPFE evaluation scripts:
  - :mod:`src.fairness_audit`             — Axes 1+3 (AUC, ECE, significance)
  - :mod:`src.code_A1_di_eod_analysis`   — Axis 4 (DI, EOD)
  - :mod:`src.code_A3_A4_A6_ece_jaccard` — Axis 2 (ECE bootstrap CIs, Jaccard K)
"""
