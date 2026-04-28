"""
src.calibration
───────────────
Calibration subpackage (temperature scaling, ECE analysis).

Temperature scaling (Axis 2 of CPFE):
  - Operates on raw pre-softmax logits (NOT softmax outputs)
  - T* minimises NLL on a 10% stratified calibration split
  - Implemented in :mod:`src.code_A5_temperature_scaling`

Entry point: ``python scripts/05_calibration.py``
"""
