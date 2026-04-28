"""
09_reproduce_all_tables.py
───────────────────────────
Master entry point: reproduce all tables from pre-computed outputs.

This script reads the existing CSV files in outputs/results/ and prints
formatted versions of Tables 2–10 from the paper. It does NOT re-run
training or inference — those are handled by scripts 03–08.

If you just want to see the numbers, run this script. If you want to
regenerate everything from scratch, run scripts 01–08 first.

Usage
-----
    # Print all tables from existing outputs (fast — no model loading)
    python scripts/09_reproduce_all_tables.py

    # Run the full pipeline then print tables
    python scripts/09_reproduce_all_tables.py --full-pipeline

Outputs
-------
Prints formatted tables to stdout.
Saves a consolidated outputs/results/all_tables.xlsx (if openpyxl is installed).

Expected outputs to verify (tolerances from Table captions):
  Table 2: Within-platform AUC 0.983–0.987, ECE 0.056–0.060
  Table 3: Cross-platform AUC 0.596–0.685 (Reddit), 0.596–0.611 (Twitter)
  Table 6: DI < 0.17 on Reddit for all models (severe disparity)
  Table 7: Mean ECE reduction 88.0% across all model-platform pairs
  Table 8: J=0.000 for 13/16 model-class pairs at K=10 (Kaggle→Twitter)
"""

import argparse
import logging
import os
import subprocess
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DISPLAY = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "mentalbert": "Emotion-DistilRoBERTa",
    "mentalroberta": "GoEmotions-RoBERTa†",
}


def run_full_pipeline() -> None:
    """Run scripts 04–08 in order (assumes data and checkpoints exist)."""
    for script in [
        "scripts/04_evaluate_crossplatform.py",
        "scripts/05_calibration.py",
        "scripts/06_fairness_audit.py",
        "scripts/07_attribution_stability.py",
        "scripts/08_sensitivity_analysis.py",
    ]:
        log.info("Running %s ...", script)
        subprocess.run([sys.executable, script], check=True)


def print_table2_3() -> None:
    """Print Tables 2 and 3: within-platform and cross-platform performance."""
    path = "outputs/results/master_results.csv"
    if not os.path.exists(path):
        log.warning("Table 2/3: %s not found — run 04_evaluate_crossplatform.py first.", path)
        return

    df = pd.read_csv(path)
    df["model_display"] = df["model"].map(MODEL_DISPLAY)

    print("\n" + "=" * 70)
    print("TABLE 2: Within-Platform Performance (Kaggle test set, n=7,620)")
    print("=" * 70)
    t2 = df[df["platform"] == "kaggle"][
        ["model_display", "accuracy", "f1_macro", "f1_weighted", "auc_macro"]
    ].copy()
    t2.columns = ["Model", "Accuracy", "F1-macro", "F1-weighted", "AUC"]
    print(t2.to_string(index=False, float_format="{:.3f}".format))

    print("\n" + "=" * 70)
    print("TABLE 3: Cross-Platform AUC Degradation")
    print("=" * 70)
    kaggle_auc = df[df["platform"] == "kaggle"].set_index("model")["auc_macro"]
    cross = df[df["platform"] != "kaggle"].copy()
    cross["delta_auc_pct"] = (
        (cross["auc_macro"] - cross["model"].map(kaggle_auc)) / cross["model"].map(kaggle_auc) * 100
    ).round(1)
    cross["model_display"] = cross["model"].map(MODEL_DISPLAY)
    t3 = cross[["model_display", "platform", "accuracy", "f1_macro", "auc_macro", "delta_auc_pct"]].copy()
    t3.columns = ["Model", "Platform", "Accuracy", "F1-macro", "AUC", "ΔAUC%"]
    print(t3.to_string(index=False, float_format="{:.3f}".format))


def print_table6() -> None:
    """Print Table 6: Symmetric DI and EOD."""
    path = "outputs/results/fairness/di_eod_table.csv"
    if not os.path.exists(path):
        log.warning("Table 6: %s not found — run 06_fairness_audit.py first.", path)
        return

    df = pd.read_csv(path)
    print("\n" + "=" * 70)
    print("TABLE 6: Symmetric Disparate Impact (DI) and EOD")
    print("DI < 0.80 = four-fifths violation | DI < 0.50 = severe disparity")
    print("=" * 70)
    print(df.to_string(index=False, float_format="{:.3f}".format))


def print_table7() -> None:
    """Print Table 7: Temperature scaling results."""
    path = "outputs/results/fairness/temperature_scaling_results.csv"
    if not os.path.exists(path):
        log.warning("Table 7: %s not found — run 05_calibration.py first.", path)
        return

    df = pd.read_csv(path)
    print("\n" + "=" * 70)
    print("TABLE 7: Temperature Scaling Calibration")
    print("=" * 70)
    print(df.to_string(index=False, float_format="{:.3f}".format))


def print_table8() -> None:
    """Print Table 8: Jaccard attribution stability."""
    path = "outputs/results/fairness/jaccard_full_analysis.csv"
    if not os.path.exists(path):
        log.warning("Table 8: %s not found — run 07_attribution_stability.py first.", path)
        return

    df = pd.read_csv(path)
    print("\n" + "=" * 70)
    print("TABLE 8: Jaccard Similarity of Top-K=10 Attribution Token Sets")
    print("J=0.000 = complete vocabulary disjunction  |  J=1.000 = identical")
    print("=" * 70)
    print(df.to_string(index=False, float_format="{:.3f}".format))


def save_all_tables_xlsx() -> None:
    """Save all available result CSVs to a single Excel workbook."""
    try:
        import openpyxl
    except ImportError:
        log.info("openpyxl not installed — skipping Excel export.")
        return

    tables = {
        "Table2_3_Performance": "outputs/results/master_results.csv",
        "Table6_Fairness_DI_EOD": "outputs/results/fairness/di_eod_table.csv",
        "Table7_TemperatureScaling": "outputs/results/fairness/temperature_scaling_results.csv",
        "Table8_Jaccard": "outputs/results/fairness/jaccard_full_analysis.csv",
        "Table9_Finetuning": "outputs/results/fairness/sensitivity_analysis.csv",
    }

    out_path = "outputs/results/all_tables.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, csv_path in tables.items():
            if os.path.exists(csv_path):
                pd.read_csv(csv_path).to_excel(writer, sheet_name=sheet_name, index=False)
    log.info("All tables saved to %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce all paper tables.")
    p.add_argument(
        "--full-pipeline", action="store_true",
        help="Run scripts 04–08 before printing tables.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.full_pipeline:
        log.info("Running full pipeline (scripts 04–08)...")
        run_full_pipeline()

    print_table2_3()
    print_table6()
    print_table7()
    print_table8()
    save_all_tables_xlsx()

    log.info(
        "\nAll available tables printed.\n"
        "If any tables were missing, run the corresponding numbered script first.\n"
        "Full pipeline: python scripts/09_reproduce_all_tables.py --full-pipeline"
    )


if __name__ == "__main__":
    main()
