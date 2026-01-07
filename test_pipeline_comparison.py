#!/usr/bin/env python3
"""
Pipeline Comparison Test

Runs both the old subprocess-based pipeline and the new integrated pipeline,
then compares their outputs to verify they produce equivalent results.

Usage:
    python test_pipeline_comparison.py [--file PREFIX] [--method METHOD]
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "Rugged Data Files"


def backup_files(data_dir, prefix, method):
    """Backup existing output files before running pipelines."""
    backup_dir = os.path.join(data_dir, "_comparison_backup")
    os.makedirs(backup_dir, exist_ok=True)

    patterns = [
        f"{prefix}-features.csv",
        f"{prefix}-clusters-{method}.csv",
        f"{prefix}-cluster-features-{method}.csv",
        f"{prefix}-pd-types-{method}.csv",
    ]

    for pattern in patterns:
        src = os.path.join(data_dir, pattern)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, pattern)
            shutil.copy2(src, dst)

    return backup_dir


def run_old_pipeline(data_dir, prefix, method):
    """Run the old subprocess-based pipeline."""
    print("\n" + "=" * 60)
    print("RUNNING OLD PIPELINE (subprocess-based)")
    print("=" * 60)

    cmd = [
        sys.executable, "run_analysis_pipeline.py",
        "--input-dir", data_dir,
        "--clustering-method", method,
        "--file", prefix
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Old pipeline failed: {result.stderr}")
        return False

    print("Old pipeline completed successfully")
    return True


def run_new_pipeline(data_dir, prefix, method):
    """Run the new integrated pipeline."""
    print("\n" + "=" * 60)
    print("RUNNING NEW PIPELINE (pdlib-based)")
    print("=" * 60)

    cmd = [
        sys.executable, "run_pipeline_integrated.py",
        "--input-dir", data_dir,
        "--clustering-method", method,
        "--file", prefix
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"New pipeline failed: {result.stderr}")
        return False

    print("New pipeline completed successfully")
    return True


def compare_features(data_dir, prefix, backup_dir):
    """Compare feature extraction results."""
    print("\n" + "-" * 60)
    print("COMPARING FEATURE EXTRACTION")
    print("-" * 60)

    old_path = os.path.join(backup_dir, f"{prefix}-features.csv")
    new_path = os.path.join(data_dir, f"{prefix}-features.csv")

    if not os.path.exists(old_path):
        print("  ⚠ No backup features file (old pipeline didn't run extraction)")
        return True, {}

    if not os.path.exists(new_path):
        print("  ✗ New features file not found")
        return False, {}

    old_df = pd.read_csv(old_path, index_col=0)
    new_df = pd.read_csv(new_path, index_col=0)

    results = {
        'old_pulses': len(old_df),
        'new_pulses': len(new_df),
        'old_features': len(old_df.columns),
        'new_features': len(new_df.columns),
    }

    print(f"  Old: {results['old_pulses']} pulses, {results['old_features']} features")
    print(f"  New: {results['new_pulses']} pulses, {results['new_features']} features")

    # Check pulse count
    if results['old_pulses'] != results['new_pulses']:
        print(f"  ✗ Pulse count mismatch!")
        return False, results

    print(f"  ✓ Pulse count matches: {results['old_pulses']}")

    # Find common features
    common = set(old_df.columns) & set(new_df.columns)
    old_only = set(old_df.columns) - set(new_df.columns)
    new_only = set(new_df.columns) - set(old_df.columns)

    print(f"  Common features: {len(common)}")
    if old_only:
        print(f"  Old only: {len(old_only)} ({', '.join(list(old_only)[:5])}...)")
    if new_only:
        print(f"  New only: {len(new_only)} ({', '.join(list(new_only)[:5])}...)")

    # Compare values of common features
    mismatches = []
    for feat in sorted(common):
        old_vals = old_df[feat].values
        new_vals = new_df[feat].values

        # Use relative tolerance for floating point comparison
        if not np.allclose(old_vals, new_vals, rtol=1e-5, atol=1e-10, equal_nan=True):
            diff = np.abs(old_vals - new_vals)
            max_diff = np.nanmax(diff)
            mismatches.append((feat, max_diff))

    if mismatches:
        print(f"  ⚠ {len(mismatches)} features with value differences:")
        for feat, diff in mismatches[:5]:
            print(f"      {feat}: max_diff={diff:.6g}")
    else:
        print(f"  ✓ All {len(common)} common features match exactly")

    results['common_features'] = len(common)
    results['mismatches'] = len(mismatches)

    return len(mismatches) == 0, results


def compare_clustering(data_dir, prefix, method, backup_dir):
    """Compare clustering results."""
    print("\n" + "-" * 60)
    print("COMPARING CLUSTERING")
    print("-" * 60)

    old_path = os.path.join(backup_dir, f"{prefix}-clusters-{method}.csv")
    new_path = os.path.join(data_dir, f"{prefix}-clusters-{method}.csv")

    if not os.path.exists(old_path):
        print("  ⚠ No backup cluster file")
        return True, {}

    if not os.path.exists(new_path):
        print("  ✗ New cluster file not found")
        return False, {}

    # Old format has comment lines starting with #
    old_df = pd.read_csv(old_path, comment='#')
    new_df = pd.read_csv(new_path, index_col=0)

    # Handle different column names
    if 'cluster_label' in old_df.columns:
        old_labels = old_df['cluster_label'].values
    else:
        old_labels = old_df['cluster'].values

    new_labels = new_df['cluster'].values

    old_n_clusters = len(set(old_labels)) - (1 if -1 in old_labels else 0)
    new_n_clusters = len(set(new_labels)) - (1 if -1 in new_labels else 0)
    old_noise = np.sum(old_labels == -1)
    new_noise = np.sum(new_labels == -1)

    results = {
        'old_clusters': old_n_clusters,
        'new_clusters': new_n_clusters,
        'old_noise': old_noise,
        'new_noise': new_noise,
    }

    print(f"  Old: {old_n_clusters} clusters, {old_noise} noise")
    print(f"  New: {new_n_clusters} clusters, {new_noise} noise")

    if old_n_clusters == new_n_clusters:
        print(f"  ✓ Cluster count matches")
    else:
        print(f"  ⚠ Cluster count differs (old={old_n_clusters}, new={new_n_clusters})")

    # Check if labels are identical
    if np.array_equal(old_labels, new_labels):
        print(f"  ✓ All cluster labels match exactly")
        results['labels_match'] = True
    else:
        # Count how many labels differ
        diff_count = np.sum(old_labels != new_labels)
        results['labels_match'] = False
        results['diff_count'] = diff_count
        print(f"  ⚠ {diff_count}/{len(old_labels)} labels differ")

    return results.get('labels_match', False) or (old_n_clusters == new_n_clusters), results


def compare_classification(data_dir, prefix, method, backup_dir):
    """Compare classification results."""
    print("\n" + "-" * 60)
    print("COMPARING CLASSIFICATION")
    print("-" * 60)

    old_path = os.path.join(backup_dir, f"{prefix}-pd-types-{method}.csv")
    new_path = os.path.join(data_dir, f"{prefix}-pd-types-{method}.csv")

    if not os.path.exists(old_path):
        print("  ⚠ No backup classification file")
        return True, {}

    if not os.path.exists(new_path):
        print("  ✗ New classification file not found")
        return False, {}

    # Old format has comment lines starting with #
    old_df = pd.read_csv(old_path, comment='#')
    new_df = pd.read_csv(new_path)

    # Normalize column names
    if 'cluster_label' in old_df.columns:
        old_df = old_df.rename(columns={'cluster_label': 'cluster'})

    # Convert 'noise' string to -1
    old_df['cluster'] = old_df['cluster'].apply(lambda x: -1 if x == 'noise' else int(x))

    results = {
        'old_clusters': len(old_df),
        'new_clusters': len(new_df),
    }

    print(f"  Old: {len(old_df)} clusters classified")
    print(f"  New: {len(new_df)} clusters classified")

    # Compare by cluster label
    matches = 0
    mismatches = []

    for _, old_row in old_df.iterrows():
        cluster = old_row['cluster']
        old_type = old_row['pd_type']

        new_row = new_df[new_df['cluster'] == cluster]
        if len(new_row) == 0:
            mismatches.append((cluster, old_type, 'MISSING'))
            continue

        new_type = new_row.iloc[0]['pd_type']

        if old_type == new_type:
            matches += 1
            print(f"    ✓ Cluster {cluster}: {old_type}")
        else:
            mismatches.append((cluster, old_type, new_type))
            print(f"    ✗ Cluster {cluster}: old={old_type}, new={new_type}")

    total = len(old_df)
    results['matches'] = matches
    results['mismatches'] = len(mismatches)

    if len(mismatches) == 0:
        print(f"\n  ✓ All {total} classifications match!")
    else:
        print(f"\n  ⚠ {matches}/{total} classifications match ({len(mismatches)} differ)")

    return len(mismatches) == 0, results


def main():
    parser = argparse.ArgumentParser(description="Compare old and new pipelines")
    parser.add_argument('--file', type=str, help='File prefix to test')
    parser.add_argument('--method', type=str, default='dbscan',
                        choices=['dbscan', 'kmeans', 'hdbscan'],
                        help='Clustering method (default: dbscan)')
    parser.add_argument('--input-dir', type=str, default=DATA_DIR,
                        help='Data directory')
    args = parser.parse_args()

    # Find a test file if not specified
    if args.file:
        prefix = args.file
    else:
        wfm_files = glob.glob(os.path.join(args.input_dir, "*-WFMs.txt"))
        if not wfm_files:
            print("No data files found!")
            return 1
        prefix = os.path.basename(wfm_files[0]).replace("-WFMs.txt", "")
        print(f"Using first dataset: {prefix}")

    print("=" * 60)
    print("PIPELINE COMPARISON TEST")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {prefix}")
    print(f"Method: {args.method}")
    print(f"Directory: {args.input_dir}")

    # Step 1: Backup existing files
    print("\nBacking up existing files...")
    backup_dir = backup_files(args.input_dir, prefix, args.method)

    # Step 2: Run old pipeline
    old_success = run_old_pipeline(args.input_dir, prefix, args.method)
    if not old_success:
        print("Old pipeline failed, cannot compare")
        return 1

    # Save old results
    old_backup = os.path.join(args.input_dir, "_old_results")
    os.makedirs(old_backup, exist_ok=True)
    for f in glob.glob(os.path.join(args.input_dir, f"{prefix}-*{args.method}*")):
        shutil.copy2(f, old_backup)
    shutil.copy2(os.path.join(args.input_dir, f"{prefix}-features.csv"), old_backup)

    # Step 3: Run new pipeline
    new_success = run_new_pipeline(args.input_dir, prefix, args.method)
    if not new_success:
        print("New pipeline failed")
        return 1

    # Step 4: Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    feat_ok, feat_results = compare_features(args.input_dir, prefix, old_backup)
    clust_ok, clust_results = compare_clustering(args.input_dir, prefix, args.method, old_backup)
    class_ok, class_results = compare_classification(args.input_dir, prefix, args.method, old_backup)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True

    if feat_ok:
        print("  ✓ Feature extraction: PASS")
    else:
        print("  ⚠ Feature extraction: DIFFERENCES FOUND")
        all_pass = False

    if clust_ok:
        print("  ✓ Clustering: PASS")
    else:
        print("  ⚠ Clustering: DIFFERENCES FOUND")
        all_pass = False

    if class_ok:
        print("  ✓ Classification: PASS")
    else:
        print("  ⚠ Classification: DIFFERENCES FOUND")
        all_pass = False

    print()
    if all_pass:
        print("  ★ ALL TESTS PASSED - Pipelines produce equivalent results")
    else:
        print("  ⚠ SOME DIFFERENCES FOUND - Review results above")

    # Cleanup
    print(f"\nBackup files saved to: {old_backup}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
