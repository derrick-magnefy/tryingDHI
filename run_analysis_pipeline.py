#!/usr/bin/env python3
"""
PD Analysis Pipeline Runner

Runs the complete partial discharge analysis pipeline:
1. Feature extraction from waveforms
2. Pulse clustering (HDBSCAN, DBSCAN, or K-means)
3. Cluster feature aggregation
4. PD type classification
5. Summary report generation

Usage:
    python run_analysis_pipeline.py [options]

Options:
    --input-dir DIR         Directory containing data files (default: "Rugged Data Files")
    --clustering-method     Clustering method: 'hdbscan', 'dbscan', or 'kmeans' (default: hdbscan)
    --n-clusters N          Number of clusters for K-means (default: 5)
    --polarity-method       Method for polarity calculation (default: peak)
                            Options: peak, first_peak, integrated_charge,
                                     energy_weighted, dominant_half_cycle, initial_slope
    --skip-extraction       Skip feature extraction step
    --skip-clustering       Skip clustering step
    --skip-aggregation      Skip aggregation step
    --skip-classification   Skip PD type classification step
    --skip-summary          Skip summary report generation
"""

import subprocess
import sys
import os
import glob
import argparse
from datetime import datetime
import numpy as np
from polarity_methods import POLARITY_METHODS, DEFAULT_POLARITY_METHOD

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


def generate_summary_report(data_dir, clustering_methods, file_prefix=None):
    """
    Generate a comprehensive summary report of all analysis results.

    Args:
        data_dir: Directory containing data files
        clustering_methods: List of clustering methods used
        file_prefix: Optional specific file prefix

    Returns:
        str: Path to generated summary report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PARTIAL DISCHARGE ANALYSIS - COMPREHENSIVE SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Directory: {data_dir}")
    report_lines.append("")

    # Find all datasets
    if file_prefix:
        prefixes = [file_prefix]
    else:
        wfm_files = glob.glob(os.path.join(data_dir, "*-WFMs.txt"))
        prefixes = [os.path.basename(f).replace("-WFMs.txt", "") for f in wfm_files]

    report_lines.append(f"Datasets Analyzed: {len(prefixes)}")
    report_lines.append("")

    # Overall statistics
    total_pulses = 0
    total_clusters = 0
    pd_type_counts = {'NOISE': 0, 'CORONA': 0, 'INTERNAL': 0, 'SURFACE': 0, 'UNKNOWN': 0}

    # Process each dataset
    for prefix in sorted(prefixes):
        report_lines.append("=" * 80)
        report_lines.append(f"DATASET: {prefix}")
        report_lines.append("=" * 80)

        # Load feature file to get pulse count
        features_file = os.path.join(data_dir, f"{prefix}-features.csv")
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                pulse_count = sum(1 for line in f) - 1  # Subtract header
            report_lines.append(f"Total Pulses: {pulse_count}")
            total_pulses += pulse_count

            # Get feature statistics
            report_lines.append("")
            report_lines.append("Pulse Feature Statistics:")
            report_lines.append("-" * 40)
            try:
                features = []
                feature_names = []
                with open(features_file, 'r') as f:
                    header = f.readline().strip().split(',')[1:]
                    feature_names = header
                    for line in f:
                        parts = line.strip().split(',')[1:]
                        if parts:
                            features.append([float(v) for v in parts])

                if features:
                    features = np.array(features)
                    # Show key features
                    key_features = ['phase_angle', 'peak_amplitude_positive', 'rise_time',
                                   'energy', 'dominant_frequency']
                    for feat_name in key_features:
                        if feat_name in feature_names:
                            idx = feature_names.index(feat_name)
                            col = features[:, idx]
                            # Handle infinite values
                            col = col[np.isfinite(col)]
                            if len(col) > 0:
                                report_lines.append(
                                    f"  {feat_name:30s}: min={np.min(col):.4e}, "
                                    f"max={np.max(col):.4e}, mean={np.mean(col):.4e}"
                                )
            except Exception as e:
                report_lines.append(f"  Error reading features: {e}")

        # Process each clustering method
        for method in clustering_methods:
            report_lines.append("")
            report_lines.append(f"Clustering Method: {method.upper()}")
            report_lines.append("-" * 40)

            # Load cluster info
            cluster_file = os.path.join(data_dir, f"{prefix}-clusters-{method}.csv")
            cluster_counts = {}  # Store counts for use in PD type classification
            if os.path.exists(cluster_file):
                cluster_labels = []
                with open(cluster_file, 'r') as f:
                    for line in f:
                        if not line.startswith('#') and not line.startswith('waveform'):
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                cluster_labels.append(int(parts[1]))

                if cluster_labels:
                    unique_labels = sorted(set(cluster_labels))
                    n_clusters = len([l for l in unique_labels if l >= 0])
                    n_noise = cluster_labels.count(-1)
                    total_clusters += n_clusters

                    report_lines.append(f"  Number of Clusters: {n_clusters}")
                    report_lines.append(f"  Noise Points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

                    # Cluster sizes
                    report_lines.append("  Cluster Sizes:")
                    for label in unique_labels:
                        count = cluster_labels.count(label)
                        label_str = "noise" if label == -1 else str(label)
                        cluster_counts[label_str] = count
                        display_label = "Noise" if label == -1 else f"Cluster {label}"
                        report_lines.append(f"    {display_label}: {count} pulses ({count/len(cluster_labels)*100:.1f}%)")

            # Load cluster features
            cluster_feat_file = os.path.join(data_dir, f"{prefix}-cluster-features-{method}.csv")
            if os.path.exists(cluster_feat_file):
                report_lines.append("")
                report_lines.append("  Cluster Feature Summary:")
                try:
                    with open(cluster_feat_file, 'r') as f:
                        lines = [l for l in f.readlines() if not l.startswith('#')]
                        if len(lines) > 1:
                            header = lines[0].strip().split(',')
                            # Find key feature indices
                            key_feats = ['cross_correlation', 'discharge_asymmetry',
                                        'phase_of_max_activity', 'weibull_beta']
                            for i, line in enumerate(lines[1:4]):  # Show first 3 clusters
                                parts = line.strip().split(',')
                                label = parts[0]
                                report_lines.append(f"    Cluster {label}:")
                                for feat in key_feats:
                                    if feat in header:
                                        idx = header.index(feat)
                                        if idx < len(parts):
                                            val = float(parts[idx])
                                            report_lines.append(f"      {feat}: {val:.4f}")
                except Exception as e:
                    report_lines.append(f"    Error reading cluster features: {e}")

            # Load PD type classification
            pd_type_file = os.path.join(data_dir, f"{prefix}-pd-types-{method}.csv")
            if os.path.exists(pd_type_file):
                report_lines.append("")
                report_lines.append("  PD Type Classification:")
                try:
                    with open(pd_type_file, 'r') as f:
                        for line in f:
                            if line.startswith('#') or line.startswith('cluster'):
                                continue
                            parts = line.strip().split(',')
                            if len(parts) >= 4:
                                label = parts[0]
                                pd_type = parts[1]
                                confidence = float(parts[3])
                                # Get waveform count for this cluster
                                n_waveforms = cluster_counts.get(label, 0)
                                report_lines.append(
                                    f"    Cluster {label}: {pd_type} ({confidence:.1%} confidence) - {n_waveforms} waveforms"
                                )
                                if pd_type in pd_type_counts:
                                    pd_type_counts[pd_type] += 1
                except Exception as e:
                    report_lines.append(f"    Error reading PD types: {e}")

        report_lines.append("")

    # Overall Summary
    report_lines.append("=" * 80)
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Datasets: {len(prefixes)}")
    report_lines.append(f"Total Pulses Analyzed: {total_pulses}")
    report_lines.append(f"Total Clusters Identified: {total_clusters}")
    report_lines.append("")
    report_lines.append("PD Type Distribution (across all clusters):")
    total_typed = sum(pd_type_counts.values())
    for pd_type, count in sorted(pd_type_counts.items()):
        if total_typed > 0:
            pct = count / total_typed * 100
            report_lines.append(f"  {pd_type:10s}: {count:3d} ({pct:.1f}%)")
    report_lines.append("")

    # Recommendations
    report_lines.append("=" * 80)
    report_lines.append("ANALYSIS NOTES")
    report_lines.append("=" * 80)

    if pd_type_counts['CORONA'] > 0:
        report_lines.append("- CORONA discharges detected: Check for sharp edges, corona rings, or")
        report_lines.append("  conductor surface issues. Corona is typically less harmful but may")
        report_lines.append("  indicate insulation stress.")

    if pd_type_counts['INTERNAL'] > 0:
        report_lines.append("- INTERNAL (void) discharges detected: May indicate cavities in solid")
        report_lines.append("  insulation. Monitor for progression as this can lead to insulation")
        report_lines.append("  breakdown over time.")

    if pd_type_counts['SURFACE'] > 0:
        report_lines.append("- SURFACE discharges detected: Check for contamination, moisture, or")
        report_lines.append("  tracking on insulation surfaces. Clean and inspect bushings and")
        report_lines.append("  terminations.")

    if pd_type_counts['UNKNOWN'] > 0:
        report_lines.append("- UNKNOWN patterns detected: Manual inspection recommended. These may")
        report_lines.append("  be mixed or transitional discharge patterns.")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Write report
    report_path = os.path.join(data_dir, "analysis-summary-report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Print summary to console
    print("\nSummary Statistics:")
    print(f"  Total Datasets: {len(prefixes)}")
    print(f"  Total Pulses: {total_pulses}")
    print(f"  Total Clusters: {total_clusters}")
    print(f"  PD Types: {dict(pd_type_counts)}")

    return report_path


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
        choices=['hdbscan', 'dbscan', 'kmeans', 'both', 'all'],
        default='hdbscan',
        help='Clustering method (default: hdbscan, use "both" for dbscan+kmeans, "all" for all three)'
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
        '--polarity-method',
        type=str,
        choices=POLARITY_METHODS,
        default=DEFAULT_POLARITY_METHOD,
        help=f'Method for polarity calculation (default: {DEFAULT_POLARITY_METHOD}). '
             f'Options: {", ".join(POLARITY_METHODS)}'
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
        '--skip-classification',
        action='store_true',
        help='Skip PD type classification step'
    )
    parser.add_argument(
        '--skip-summary',
        action='store_true',
        help='Skip summary report generation'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Process specific file prefix only'
    )
    parser.add_argument(
        '--pulse-features',
        type=str,
        default=None,
        help='Comma-separated list of pulse features to use for clustering (default: all)'
    )
    parser.add_argument(
        '--cluster-features',
        type=str,
        default=None,
        help='Comma-separated list of cluster features to use for classification (default: all)'
    )
    parser.add_argument(
        '--feature-weights',
        type=str,
        default=None,
        help='Feature weights for clustering as feature:weight pairs, e.g., "energy:2.0,phase_angle:1.5". '
             'Higher weights increase feature importance in clustering.'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PD ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {args.input_dir}")
    print(f"Clustering method: {args.clustering_method}")
    print(f"Polarity method: {args.polarity_method}")
    if args.feature_weights:
        print(f"Feature weights: {args.feature_weights}")
    print("=" * 70)

    # Track overall success
    all_success = True
    steps_completed = []

    # Determine which clustering methods to use
    if args.clustering_method == 'all':
        clustering_methods = ['hdbscan', 'dbscan', 'kmeans']
    elif args.clustering_method == 'both':
        clustering_methods = ['dbscan', 'kmeans']
    else:
        clustering_methods = [args.clustering_method]

    # Step 1: Feature Extraction
    if not args.skip_extraction:
        cmd = [sys.executable, 'extract_features.py', '--input-dir', args.input_dir,
               '--polarity-method', args.polarity_method]
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
            elif method == 'hdbscan':
                cmd.extend(['--min-samples', str(args.min_samples)])
            else:  # dbscan
                cmd.extend(['--min-samples', str(args.min_samples)])
                if args.eps is not None:
                    cmd.extend(['--eps', str(args.eps)])

            # Add pulse features selection if specified
            if args.pulse_features:
                cmd.extend(['--features', args.pulse_features])

            # Add feature weights if specified
            if args.feature_weights:
                cmd.extend(['--feature-weights', args.feature_weights])

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

    # Step 4: PD Type Classification (for each method)
    for method in clustering_methods:
        if not args.skip_classification:
            cmd = [
                sys.executable, 'classify_pd_type.py',
                '--method', method,
                '--input-dir', args.input_dir
            ]

            if args.file:
                cmd.extend(['--file', args.file])

            # Add cluster features selection if specified
            if args.cluster_features:
                cmd.extend(['--cluster-features', args.cluster_features])

            success = run_command(cmd, f"PD Type Classification ({method.upper()})")
            if success:
                steps_completed.append(f"Classification ({method})")
            else:
                all_success = False
                print(f"\nWARNING: {method} classification failed.")
        else:
            print(f"\n[Skipping Classification ({method})]")

    # Step 5: Generate Summary Report
    if not args.skip_summary:
        print(f"\n{'='*70}")
        print("STEP: Generate Summary Report")
        print(f"{'='*70}")
        try:
            summary_path = generate_summary_report(args.input_dir, clustering_methods, args.file)
            steps_completed.append("Summary Report")
            print(f"\nSummary report saved to: {summary_path}")
        except Exception as e:
            all_success = False
            print(f"\nWARNING: Summary generation failed: {e}")
    else:
        print("\n[Skipping Summary Report]")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSteps completed:")
    for step in steps_completed:
        print(f"  [OK] {step}")

    if all_success:
        print("\n[SUCCESS] Pipeline completed successfully!")
    else:
        print("\n[WARNING] Pipeline completed with some errors.")

    print("\nOutput files:")
    print(f"  - Feature files: {args.input_dir}/*-features.csv")
    for method in clustering_methods:
        print(f"  - Cluster labels: {args.input_dir}/*-clusters-{method}.csv")
        print(f"  - Cluster features: {args.input_dir}/*-cluster-features-{method}.csv")
        print(f"  - PD types: {args.input_dir}/*-pd-types-{method}.csv")
    print(f"  - Summary report: {args.input_dir}/analysis-summary-report.txt")

    print("=" * 70)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
