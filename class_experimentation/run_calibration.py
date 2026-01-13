#!/usr/bin/env python3
"""
Run Classifier Calibration

Main entry point for auto-calibrating the PD classifier using labeled lab data.

Usage:
    python -m class_experimentation.run_calibration --data-dir IEEE_Example_Data/

    # With custom K threshold
    python -m class_experimentation.run_calibration --data-dir IEEE_Example_Data/ --k-threshold 7.0

    # Quick test with grid search (no optuna required)
    python -m class_experimentation.run_calibration --data-dir IEEE_Example_Data/ --method grid

    # More trials for better optimization
    python -m class_experimentation.run_calibration --data-dir IEEE_Example_Data/ --n-trials 200
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from class_experimentation.labeled_loader import LabeledDatasetLoader
from class_experimentation.feature_pipeline import FeaturePipeline
from class_experimentation.auto_calibrate import ClassifierCalibrator, run_grid_search, OPTUNA_AVAILABLE
from class_experimentation.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Auto-calibrate PD classifier thresholds using labeled lab data"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='IEEE_Example_Data',
        help='Directory containing labeled .mat files'
    )
    parser.add_argument(
        '--k-threshold',
        type=float,
        default=6.0,
        help='Noise floor K-value (default: 6.0). Pulses > K*noise_std are used for training.'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['optuna', 'grid'],
        default='optuna' if OPTUNA_AVAILABLE else 'grid',
        help='Optimization method (default: optuna if available, else grid)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of optimization trials (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration_result.json',
        help='Output file for results (default: calibration_result.json)'
    )
    parser.add_argument(
        '--output-code',
        type=str,
        default='calibrated_thresholds.py',
        help='Output file for threshold Python code (default: calibrated_thresholds.py)'
    )
    parser.add_argument(
        '--channel',
        type=str,
        default='Ch1',
        help='Channel to load from .mat files (default: Ch1)'
    )
    parser.add_argument(
        '--filter-types',
        type=str,
        nargs='+',
        default=None,
        help='Only load specific PD types (e.g., --filter-types SURFACE CORONA)'
    )
    parser.add_argument(
        '--file-level',
        action='store_true',
        help='Use file-level aggregation instead of HDBScan clustering. '
             'Treats each file as one cluster with the known label.'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PD CLASSIFIER AUTO-CALIBRATION")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"K-threshold: {args.k_threshold}")
    print(f"Method: {args.method}")
    print(f"Trials: {args.n_trials}")
    print(f"File-level mode: {args.file_level}")
    print()

    # Step 1: Load labeled datasets
    print("Step 1: Loading labeled datasets...")
    print("-" * 40)
    loader = LabeledDatasetLoader(args.data_dir, channel=args.channel)

    try:
        datasets = loader.load_all(filter_types=args.filter_types)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not datasets:
        print("Error: No datasets found!")
        print(f"Make sure .mat files are in: {args.data_dir}")
        print("Filenames should contain: surface, corona, internal, or noise")
        sys.exit(1)

    print()

    # Step 2: Extract features
    print("Step 2: Extracting features...")
    print("-" * 40)
    pipeline = FeaturePipeline(k_threshold=args.k_threshold)
    dataset_features = pipeline.extract_all(datasets)

    # Count total pulses above threshold
    total_above = sum(df.n_above_threshold for df in dataset_features)
    total_below = sum(df.n_below_threshold for df in dataset_features)
    print(f"\nTotal pulses above K={args.k_threshold}: {total_above}")
    print(f"Total pulses below K={args.k_threshold}: {total_below}")

    if total_above < 10:
        print("Error: Not enough pulses above threshold for calibration!")
        print("Try lowering K-threshold or check data quality.")
        sys.exit(1)

    print()

    # Step 3: Run calibration
    print("Step 3: Running calibration...")
    print("-" * 40)

    if args.method == 'optuna':
        if not OPTUNA_AVAILABLE:
            print("Warning: optuna not available, falling back to grid search")
            print("Install optuna for better optimization: pip install optuna")
            args.method = 'grid'

    if args.method == 'optuna':
        calibrator = ClassifierCalibrator(
            dataset_features,
            only_above_threshold=True,
            use_file_level=args.file_level,
        )
        result = calibrator.optimize(n_trials=args.n_trials)
    else:
        result = run_grid_search(dataset_features, use_file_level=args.file_level)

    print()

    # Step 4: Evaluation and reporting
    print("Step 4: Generating report...")
    print("-" * 40)
    evaluator = Evaluator(result)
    evaluator.print_report()

    # Save results
    result.save(args.output)
    print(f"Saved results to: {args.output}")

    evaluator.save_threshold_code(args.output_code)

    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Best accuracy: {result.best_accuracy:.2%}")
    print()
    print("To use the calibrated thresholds:")
    print("  from pdlib.classification import PDTypeClassifier")
    print(f"  from {args.output_code.replace('.py', '')} import CALIBRATED_THRESHOLDS")
    print("  classifier = PDTypeClassifier(thresholds=CALIBRATED_THRESHOLDS)")
    print()


if __name__ == "__main__":
    main()
