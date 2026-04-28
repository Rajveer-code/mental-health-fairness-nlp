"""
run_pipeline.py
───────────────
Root-level orchestrator for the CPFE revision pipeline.

Runs all six revision analysis steps in dependency order, with ETA
estimates, dependency checks, and a final validation checklist.

Steps
-----
1. train_multiseed.py      — 20 training runs (4 models × 5 seeds) + evaluation
2. integrated_gradients.py — Captum IG attribution (Table 8 update)
3. calibration_comparison.py — Fine-tuning vs temperature scaling (Table 10)
4. compile_tables.py       — Aggregate all results → all_tables.xlsx
5. generate_figures_v2.py  — Regenerate all 9 figures with multi-seed data
6. rebuild_manuscript.py   — Update manuscript .docx with all additions

Usage
-----
Run from the repository root:

    # Run full pipeline from step 1
    python run_pipeline.py

    # Resume from a specific step (steps 1-N already done)
    python run_pipeline.py --from-step 2

    # Run only a single step
    python run_pipeline.py --only 4

    # Dry run: print what would execute without running
    python run_pipeline.py --dry-run

    # Skip training (evaluate existing checkpoints only)
    python run_pipeline.py --skip-training

Estimated runtimes (RTX 4060, 8GB VRAM)
-----------------------------------------
Step 1: ~6-8 hours  (dominant: 16 training runs for seeds 0,1,7,123)
Step 2: ~1-2 hours  (IG attribution: 200 samples × 4 models × 3 platforms × 4 classes)
Step 3: ~1 hour     (fine-tuning: 3 epochs × 4 models × 2 platforms)
Step 4: ~1 minute   (table compilation: pure pandas/openpyxl)
Step 5: ~2 minutes  (figure generation: matplotlib)
Step 6: ~1 minute   (manuscript rebuild: python-docx)

Total: ~9-12 hours on RTX 4060
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import timedelta

# ── Step definitions ───────────────────────────────────────────────────────────

STEPS = [
    {
        "number":       1,
        "name":         "Multi-Seed Training",
        "script":       "src/train_multiseed.py",
        "eta_hours":    7.0,
        "description":  "Train 4 models × 5 seeds, evaluate on 3 platforms",
        "outputs": [
            "outputs/results/multiseed/multiseed_results.csv",
        ],
        "extra_args":   [],
    },
    {
        "number":       2,
        "name":         "Integrated Gradients",
        "script":       "src/integrated_gradients.py",
        "eta_hours":    1.5,
        "description":  "Captum IG attribution, Table 8 update",
        "outputs": [
            "outputs/results/ig_attribution_results.csv",
            "outputs/results/ig_table8_update.csv",
        ],
        "extra_args":   [],
    },
    {
        "number":       3,
        "name":         "Calibration Comparison",
        "script":       "src/calibration_comparison.py",
        "eta_hours":    1.0,
        "description":  "Fine-tuning vs temperature scaling (Table 10)",
        "outputs": [
            "outputs/results/calibration_comparison.csv",
        ],
        "extra_args":   [],
    },
    {
        "number":       4,
        "name":         "Compile Tables",
        "script":       "src/compile_tables.py",
        "eta_hours":    0.02,
        "description":  "Aggregate results → all_tables.xlsx (Tables 2–10)",
        "outputs": [
            "outputs/results/all_tables.xlsx",
        ],
        "extra_args":   [],
    },
    {
        "number":       5,
        "name":         "Generate Figures v2",
        "script":       "src/generate_figures_v2.py",
        "eta_hours":    0.05,
        "description":  "Regenerate all 9 figures with multi-seed error bars",
        "outputs": [
            "outputs/figures/v2/figure1.png",
            "outputs/figures/v2/figure2.png",
            "outputs/figures/v2/figure9.png",
        ],
        "extra_args":   [],
    },
    {
        "number":       6,
        "name":         "Rebuild Manuscript",
        "script":       "src/rebuild_manuscript.py",
        "eta_hours":    0.02,
        "description":  "Update .docx with all revision additions",
        "outputs": [
            "CPFE_submission_ready.docx",
        ],
        "extra_args":   [],
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_outputs_exist(step: dict) -> bool:
    """Return True if all expected outputs for a step already exist."""
    return all(os.path.exists(p) for p in step["outputs"])


def _human_eta(hours: float) -> str:
    td = timedelta(hours=hours)
    total_min = int(td.total_seconds() / 60)
    if total_min < 2:
        return "< 1 min"
    if total_min < 60:
        return f"~{total_min} min"
    h = total_min // 60
    m = total_min % 60
    return f"~{h}h {m:02d}min"


def _run_step(
    step: dict,
    dry_run: bool = False,
    extra_flags: list[str] | None = None,
) -> bool:
    """
    Execute a pipeline step.

    Parameters
    ----------
    step : dict
        Step definition from STEPS.
    dry_run : bool
        If True, print the command without executing it.
    extra_flags : list[str] or None
        Additional command-line flags to pass to the script.

    Returns
    -------
    bool
        True if the step completed successfully.
    """
    cmd  = [sys.executable, step["script"]] + step.get("extra_args", [])
    if extra_flags:
        cmd += extra_flags

    cmd_str = " ".join(cmd)

    print(f"\n{'─'*65}")
    print(f"  Step {step['number']}: {step['name']}")
    print(f"  Command : {cmd_str}")
    print(f"  ETA     : {_human_eta(step['eta_hours'])}")
    print(f"{'─'*65}")

    if dry_run:
        print("  [DRY RUN] — not executing")
        return True

    t_start = time.time()

    result = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        text=True,
        check=False,
    )

    elapsed = time.time() - t_start

    if result.returncode != 0:
        print(f"\n  ✗ Step {step['number']} FAILED (exit code {result.returncode})")
        print(f"  Elapsed: {_human_eta(elapsed / 3600)}")
        return False

    print(f"\n  ✓ Step {step['number']} completed in {_human_eta(elapsed / 3600)}")
    return True


def _print_validation_checklist() -> None:
    """Print a structured validation checklist of all expected outputs."""
    print(f"\n{'='*65}")
    print("VALIDATION CHECKLIST")
    print(f"{'='*65}")

    ALL_EXPECTED = [
        # Step 1
        ("outputs/results/multiseed/multiseed_results.csv",
         "Multi-seed aggregated results (20 runs × 3 platforms)"),
        # Step 2
        ("outputs/results/ig_attribution_results.csv",
         "IG Jaccard results (all K values)"),
        ("outputs/results/ig_table8_update.csv",
         "Table 8 update: Grad-Sal J + IG J"),
        # Step 3
        ("outputs/results/calibration_comparison.csv",
         "Table 10: fine-tuning vs temperature scaling"),
        ("outputs/figures/figure_calibration_comparison.png",
         "Calibration comparison figure"),
        # Step 4
        ("outputs/results/all_tables.xlsx",
         "All tables (2-10) in Excel workbook"),
        ("outputs/results/tables_summary.txt",
         "Table compilation summary"),
        # Step 5
        ("outputs/figures/v2/figure1.png", "Figure 1 (framework, v2)"),
        ("outputs/figures/v2/figure2.png", "Figure 2 (degradation, v2)"),
        ("outputs/figures/v2/figure3.png", "Figure 3 (reliability, v2)"),
        ("outputs/figures/v2/figure4.png", "Figure 4 (F1 heatmap, v2)"),
        ("outputs/figures/v2/figure5a.png", "Figure 5a (attribution, v2)"),
        ("outputs/figures/v2/figure6.png", "Figure 6 (Jaccard, v2)"),
        ("outputs/figures/v2/figure7.png", "Figure 7 (sensitivity, v2)"),
        ("outputs/figures/v2/figure8.png", "Figure 8 (calibration comparison, v2)"),
        ("outputs/figures/v2/figure9.png", "Figure 9 (DI heatmap, v2)"),
        # Step 6
        ("CPFE_submission_ready.docx",
         "Submission-ready manuscript"),
    ]

    passed = 0
    for path, description in ALL_EXPECTED:
        exists  = os.path.exists(path)
        status  = "✓" if exists else "✗"
        if exists:
            passed += 1
        print(f"  [{status}] {description}")
        if not exists:
            print(f"       Missing: {path}")

    print(f"\n  {passed}/{len(ALL_EXPECTED)} outputs present")
    print()

    if passed == len(ALL_EXPECTED):
        print("  All outputs verified. Manuscript is ready for submission.")
    else:
        missing = len(ALL_EXPECTED) - passed
        print(f"  {missing} output(s) missing — re-run the relevant pipeline steps.")

    print(f"{'='*65}")


def _print_pipeline_overview(steps_to_run: list[dict]) -> None:
    """Print a summary of what will be executed."""
    total_eta = sum(s["eta_hours"] for s in steps_to_run)

    print("\nCPFE REVISION PIPELINE")
    print(f"{'='*65}")
    print(f"  Steps to run   : {len(steps_to_run)}")
    print(f"  Total ETA      : {_human_eta(total_eta)}")
    print(f"{'='*65}\n")

    for step in steps_to_run:
        done = "[DONE]" if _check_outputs_exist(step) else ""
        print(
            f"  Step {step['number']}: {step['name']:<28} "
            f"{_human_eta(step['eta_hours']):<12} {done}"
        )
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPFE revision pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from-step", type=int, default=1,
        help="Start from this step number (default: 1).",
    )
    parser.add_argument(
        "--only", type=int, default=None,
        help="Run only this step number.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Pass --skip-training to train_multiseed.py (evaluate existing only).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Print validation checklist only — do not run any steps.",
    )
    args = parser.parse_args()

    if args.validate_only:
        _print_validation_checklist()
        return

    # Select steps
    if args.only is not None:
        steps_to_run = [s for s in STEPS if s["number"] == args.only]
        if not steps_to_run:
            print(f"Unknown step number: {args.only}")
            sys.exit(1)
    else:
        steps_to_run = [s for s in STEPS if s["number"] >= args.from_step]

    # Apply --skip-training flag to step 1
    if args.skip_training:
        for step in steps_to_run:
            if step["number"] == 1:
                step["extra_args"] = ["--skip-training"]

    _print_pipeline_overview(steps_to_run)

    if args.dry_run:
        print("DRY RUN — no commands will be executed.\n")

    # Execute
    pipeline_start = time.time()
    failed_step    = None

    for step in steps_to_run:
        # Check if already done (resumability)
        if _check_outputs_exist(step) and not args.dry_run:
            print(f"\n  Step {step['number']} ({step['name']}): "
                  "outputs already exist — skipping.")
            continue

        success = _run_step(step, dry_run=args.dry_run)

        if not success:
            failed_step = step
            break

    # Final summary
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*65}")

    if failed_step is not None:
        print(f"Pipeline stopped at Step {failed_step['number']}: "
              f"{failed_step['name']}")
        print(f"Elapsed: {_human_eta(total_elapsed / 3600)}")
        print()
        print("To resume from this step:")
        print(f"  python run_pipeline.py --from-step {failed_step['number']}")
    else:
        print(
            f"Pipeline complete in {_human_eta(total_elapsed / 3600)}."
            if not args.dry_run else "Dry run complete."
        )
        print()
        _print_validation_checklist()

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
