#!/usr/bin/env python3
"""
PD Analysis Pipeline Runner

Runs the complete partial discharge analysis pipeline:
1. Feature extraction from waveforms
2. Pulse clustering (DBSCAN or K-means)
3. Cluster feature aggregation

Usage:
    python run_analysis_pipeline.py [options]

Options:
    --input-dir DIR         Directory containing data files (default: "Rugged Data Files")
    --clustering-method     Clustering method: 'dbscan' or 'kmeans' (default: dbscan)
    --n-clusters N          Number of clusters for K-means (default: 5)
    --skip-extraction       Skip feature extraction step
    --skip-clustering       Skip clustering step
    --skip-aggregation      Skip aggregation step
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime

DATA_DIR = "Rugged Data Files"


def run_command(cmd, description):
    """
    Run a command and capture output.

    Args:
        cmd: Command list to execute
        description: Description of the step

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Step failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete PD analysis pipeline"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=DATA_DIR,
        help='Directory containing data files'
    )
    parser.add_argument(
        '--clustering-method',
        type=str,
        choices=['dbscan', 'kmeans', 'both'],
        default='dbscan',
        help='Clustering method (default: dbscan, use "both" for both methods)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=5,
        help='Number of clusters for K-means (default: 5)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=None,
        help='DBSCAN epsilon parameter (default: auto)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=5,
        help='DBSCAN min_samples parameter (default: 5)'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip feature extraction step'
    )
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering step'
    )
    parser.add_argument(
        '--skip-aggregation',
        action='store_true',
        help='Skip aggregation step'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Process specific file prefix only'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PD ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {args.input_dir}")
    print(f"Clustering method: {args.clustering_method}")
    print("=" * 70)

    # Track overall success
    all_success = True
    steps_completed = []

    # Determine which clustering methods to use
    if args.clustering_method == 'both':
        clustering_methods = ['dbscan', 'kmeans']
    else:
        clustering_methods = [args.clustering_method]

    # Step 1: Feature Extraction
    if not args.skip_extraction:
        cmd = [sys.executable, 'extract_features.py', '--input-dir', args.input_dir]
        if args.file:
            cmd.extend(['--file', args.file])

        success = run_command(cmd, "Feature Extraction")
        if success:
            steps_completed.append("Feature Extraction")
        else:
            all_success = False
            print("\nWARNING: Feature extraction failed. Continuing with existing files...")
    else:
        print("\n[Skipping Feature Extraction]")

    # Step 2: Clustering (for each method)
    for method in clustering_methods:
        if not args.skip_clustering:
            cmd = [
                sys.executable, 'cluster_pulses.py',
                '--method', method,
                '--input-dir', args.input_dir
            ]

            if method == 'kmeans':
                cmd.extend(['--n-clusters', str(args.n_clusters)])
            else:
                cmd.extend(['--min-samples', str(args.min_samples)])
                if args.eps is not None:
                    cmd.extend(['--eps', str(args.eps)])

            if args.file:
                # Need to specify input file with features suffix
                input_file = os.path.join(args.input_dir, f"{args.file}-features.csv")
                cmd.extend(['--input', input_file])

            success = run_command(cmd, f"Clustering ({method.upper()})")
            if success:
                steps_completed.append(f"Clustering ({method})")
            else:
                all_success = False
                print(f"\nWARNING: {method} clustering failed.")
        else:
            print(f"\n[Skipping Clustering ({method})]")

    # Step 3: Cluster Feature Aggregation (for each method)
    for method in clustering_methods:
        if not args.skip_aggregation:
            cmd = [
                sys.executable, 'aggregate_cluster_features.py',
                '--method', method,
                '--input-dir', args.input_dir
            ]

            if args.file:
                cmd.extend(['--file', args.file])

            success = run_command(cmd, f"Cluster Feature Aggregation ({method.upper()})")
            if success:
                steps_completed.append(f"Aggregation ({method})")
            else:
                all_success = False
                print(f"\nWARNING: {method} aggregation failed.")
        else:
            print(f"\n[Skipping Aggregation ({method})]")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSteps completed:")
    for step in steps_completed:
        print(f"  ✓ {step}")

    if all_success:
        print("\n✓ Pipeline completed successfully!")
    else:
        print("\n⚠ Pipeline completed with some errors.")

    print("\nOutput files:")
    print(f"  - Feature files: {args.input_dir}/*-features.csv")
    for method in clustering_methods:
        print(f"  - Cluster labels: {args.input_dir}/*-clusters-{method}.csv")
        print(f"  - Cluster features: {args.input_dir}/*-cluster-features-{method}.csv")

    print("=" * 70)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
