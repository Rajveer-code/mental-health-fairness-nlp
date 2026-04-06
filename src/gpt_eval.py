"""
gpt_eval.py
───────────
GPT-4 qualitative evaluation of model predictions on cross-platform test sets.

When implemented, this script should:
  1. Sample N misclassified examples per model per platform (e.g. N=50).
  2. Send each sample to GPT-4 with a structured prompt asking whether the
     model's predicted label is clinically plausible for the given text.
  3. Compute inter-rater agreement (Cohen's kappa) between GPT-4 judgements
     and ground-truth labels.
  4. Identify systematic error patterns (e.g. depression misclassified as
     normal on Twitter due to platform-specific phrasing).

Inputs
------
outputs/results/{model}_{platform}_predictions.csv
    Per-sample predictions with columns: label, pred, prob_*, correct.
data/splits/cross_platform/test_{platform}.csv
    Original text for the samples.

Outputs
-------
outputs/results/gpt_eval/gpt_eval_results.csv
    Sample ID, true label, predicted label, GPT-4 judgement, rationale.
outputs/results/gpt_eval/gpt_eval_summary.csv
    Per-model per-platform Cohen's kappa and error taxonomy.

Usage
-----
    python src/gpt_eval.py --model roberta --platform reddit --n_samples 50

Dependencies
------------
Requires evaluate.py to have been run first.
Requires an OpenAI API key set as the OPENAI_API_KEY environment variable.
"""

raise NotImplementedError(
    "gpt_eval.py is not yet implemented. "
    "See the module docstring for the intended methodology. "
    "This stub is retained as a placeholder for future GPT-4 qualitative "
    "evaluation of cross-platform misclassification patterns."
)
