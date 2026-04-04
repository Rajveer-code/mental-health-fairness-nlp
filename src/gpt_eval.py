"""
gpt_eval.py
───────────
Zero-shot GPT-4 evaluation baseline for mental health text classification.

This script will use a GPT-4 (or equivalent) API to classify a sample of
mental health posts into the four canonical classes (normal, depression,
anxiety, stress), providing a zero-shot baseline for comparison against
the four fine-tuned transformer models evaluated in the main fairness audit.

When complete, the implementation should:
  1. Accept a CSV of posts with columns ``text`` and ``label``.
  2. For each text, call the OpenAI Chat Completions API with a structured
     prompt that elicits class probabilities (via log-prob extraction or a
     chain-of-thought approach followed by explicit probability assignment).
  3. Parse the API response into the four class probabilities.
  4. Save results to ``output_path`` in the canonical prediction schema:
     ``label, pred, prob_normal, prob_depression, prob_anxiety, prob_stress,
     correct``.

Inputs
------
data/splits/cross_platform/test_{platform}.csv
    Per-platform test set with columns: text, label.

Outputs
-------
outputs/results/gpt4_{platform}_predictions.csv
    Canonical prediction CSV for the GPT-4 baseline.

Usage
-----
Run from the repository root:
    python src/gpt_eval.py

Dependencies
------------
Requires an OpenAI API key in the ``OPENAI_API_KEY`` environment variable.
Requires evaluate.py to have been run first (for comparison purposes).
"""


def run_gpt_evaluation(input_path: str, output_path: str) -> None:
    """
    Run GPT-4 zero-shot evaluation on mental health posts.

    Parameters
    ----------
    input_path : str
        Path to a CSV file with columns: ``text``, ``label``.
    output_path : str
        Destination path for the predictions CSV in the canonical schema:
        ``label, pred, prob_normal, prob_depression, prob_anxiety,
        prob_stress, correct``.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.  When complete it should call
        the OpenAI Chat Completions API for each input text, extract class
        probabilities, compute predictions, and save results to
        ``output_path`` using ``df.to_csv(output_path, index=False)``.
    """
    raise NotImplementedError(
        "GPT-4 evaluation pipeline is not yet implemented. "
        "See the module docstring for the intended implementation approach."
    )
