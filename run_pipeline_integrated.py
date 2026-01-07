#!/usr/bin/env python3
"""
Integrated PD Analysis Pipeline

Uses pdlib modules directly for all processing steps:
1. Feature extraction (pdlib.features)
2. Pulse clustering (pdlib.clustering)
3. Cluster feature aggregation (pdlib.clustering)
4. PD type classification (pdlib.classification)
5. Summary report generation

This replaces the subprocess-based run_analysis_pipeline.py with
a more efficient in-process implementation.

Usage:
    python run_pipeline_integrated.py [options]

Options:
    --input-dir DIR         Directory containing data files (default: "Rugged Data Files")
    --clustering-method     Clustering method: 'hdbscan', 'dbscan', or 'kmeans' (default: hdbscan)
    --n-clusters N          Number of clusters for K-means (default: 5)
    --polarity-method       Method for polarity calculation (default: peak)
    --file PREFIX           Process specific file prefix only
    --pulse-features        Comma-separated list of pulse features for clustering
    --skip-extraction       Skip feature extraction step
    --skip-clustering       Skip clustering step
    --skip-aggregation      Skip aggregation step
    --skip-classification   Skip PD type classification step
    --skip-summary          Skip summary report generation
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Import pdlib modules
from pdlib.features import PDFeatureExtractor, FEATURE_NAMES
from pdlib.clustering import (
    cluster_pulses,
    compute_cluster_features,
    HDBSCAN_AVAILABLE,
    DEFAULT_CLUSTERING_METHOD,
)
from pdlib.classification import PDTypeClassifier, PD_TYPES

# Import middleware for data loading
from middleware.formats import RuggedLoader, list_datasets

# Import polarity methods
from polarity_methods import POLARITY_METHODS, DEFAULT_POLARITY_METHOD

# Import config
try:
    from config.loader import ConfigLoader
    _config = ConfigLoader()
    _features_config = _config.get_features()
    DEFAULT_CLUSTERING_FEATURES = _features_config.get('pulse_features', {}).get('default_clustering', [])
except Exception:
    DEFAULT_CLUSTERING_FEATURES = []

DATA_DIR = "Rugged Data Files"


def extract_features(
    data_dir: str,
    file_prefix: Optional[str] = None,
    polarity_method: str = DEFAULT_POLARITY_METHOD,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Extract features from all waveforms in directory.

    Args:
        data_dir: Directory containing data files
        file_prefix: Optional specific file prefix
        polarity_method: Method for polarity calculation
        verbose: Print progress

    Returns:
        Dict mapping prefix to DataFrame of features
    """
    loader = RuggedLoader(data_dir)

    # Get list of datasets
    if file_prefix:
        prefixes = [file_prefix]
    else:
        prefixes = list_datasets(data_dir)

    results = {}

    for prefix in prefixes:
        if verbose:
            print(f"\n  Processing: {prefix}")

        # Load data
        try:
            waveforms = loader.load_waveforms(prefix)
            settings = loader.load_settings(prefix)
            phases = loader.load_phase_angles(prefix)
        except FileNotFoundError as e:
            print(f"    Warning: {e}")
            continue

        sample_interval = settings.get('sample_interval', 4e-9)

        # Create extractor
        extractor = PDFeatureExtractor(
            sample_interval=sample_interval,
            polarity_method=polarity_method
        )

        # Extract features using batch mode (includes normalized features)
        phase_list = list(phases) if phases is not None else None
        all_features = extractor.extract_all(waveforms, phase_angles=phase_list, normalize=True)

        # Create DataFrame
        df = pd.DataFrame(all_features)
        df.index.name = 'pulse_id'

        # Save to CSV
        output_path = os.path.join(data_dir, f"{prefix}-features.csv")
        df.to_csv(output_path)

        if verbose:
            print(f"    Extracted {len(df)} pulses, {len(df.columns)} features")
            print(f"    Saved: {output_path}")

        results[prefix] = df

    return results


def run_clustering(
    data_dir: str,
    file_prefix: Optional[str] = None,
    method: str = 'hdbscan',
    n_clusters: int = 5,
    eps: Optional[float] = None,
    min_samples: int = 5,
    pulse_features: Optional[List[str]] = None,
    feature_weights: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    Run clustering on extracted features.

    Args:
        data_dir: Directory containing feature files
        file_prefix: Optional specific file prefix
        method: Clustering method (hdbscan, dbscan, kmeans)
        n_clusters: Number of clusters for k-means
        eps: DBSCAN epsilon (auto if None)
        min_samples: Minimum samples for density methods
        pulse_features: Features to use for clustering
        feature_weights: Optional feature weights
        verbose: Print progress

    Returns:
        Dict mapping prefix to (labels, info) tuples
    """
    # Find feature files
    if file_prefix:
        feature_files = [os.path.join(data_dir, f"{file_prefix}-features.csv")]
    else:
        feature_files = glob.glob(os.path.join(data_dir, "*-features.csv"))

    results = {}

    for filepath in feature_files:
        prefix = os.path.basename(filepath).replace("-features.csv", "")

        if verbose:
            print(f"\n  Clustering: {prefix}")

        # Load features
        df = pd.read_csv(filepath, index_col=0)
        feature_names = list(df.columns)

        # Select features
        if pulse_features:
            selected = [f for f in pulse_features if f in feature_names]
            if not selected:
                print(f"    Warning: No valid features selected, using all")
                selected = feature_names
        else:
            selected = feature_names

        X = df[selected].values

        # Build kwargs based on method
        kwargs = {}
        if method == 'kmeans':
            kwargs['n_clusters'] = n_clusters
        else:
            # DBSCAN and HDBSCAN use min_samples
            kwargs['min_samples'] = min_samples
            if method == 'dbscan' and eps is not None:
                kwargs['eps'] = eps

        # Run clustering
        labels, info = cluster_pulses(
            X,
            feature_names=selected,
            method=method,
            feature_weights=feature_weights,
            **kwargs
        )

        # Save results
        output_path = os.path.join(data_dir, f"{prefix}-clusters-{method}.csv")
        cluster_df = pd.DataFrame({'cluster': labels})
        cluster_df.index.name = 'pulse_id'
        cluster_df.to_csv(output_path)

        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        if verbose:
            print(f"    Found {n_clusters_found} clusters, {n_noise} noise points")
            print(f"    Saved: {output_path}")

        results[prefix] = (labels, info)

    return results


def aggregate_cluster_features(
    data_dir: str,
    file_prefix: Optional[str] = None,
    method: str = 'hdbscan',
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate features for each cluster.

    Args:
        data_dir: Directory containing feature and cluster files
        file_prefix: Optional specific file prefix
        method: Clustering method used
        verbose: Print progress

    Returns:
        Dict mapping prefix to DataFrame of cluster features
    """
    # Find cluster files
    if file_prefix:
        cluster_files = [os.path.join(data_dir, f"{file_prefix}-clusters-{method}.csv")]
    else:
        cluster_files = glob.glob(os.path.join(data_dir, f"*-clusters-{method}.csv"))

    results = {}
    loader = RuggedLoader(data_dir)

    for cluster_path in cluster_files:
        prefix = os.path.basename(cluster_path).replace(f"-clusters-{method}.csv", "")
        feature_path = os.path.join(data_dir, f"{prefix}-features.csv")

        if not os.path.exists(feature_path):
            print(f"    Warning: Features not found for {prefix}")
            continue

        if verbose:
            print(f"\n  Aggregating: {prefix}")

        # Load data
        features_df = pd.read_csv(feature_path, index_col=0)
        clusters_df = pd.read_csv(cluster_path, index_col=0)
        labels = clusters_df['cluster'].values

        # Load settings for AC frequency
        settings = loader.load_settings(prefix)
        ac_frequency = settings.get('ac_frequency', 60.0)

        # Compute cluster features (phases/amplitudes extracted from features_matrix)
        cluster_features_dict = compute_cluster_features(
            features_matrix=features_df.values,
            feature_names=list(features_df.columns),
            labels=labels,
            trigger_times=None,
            ac_frequency=ac_frequency
        )

        # Convert to DataFrame
        cluster_df = pd.DataFrame.from_dict(cluster_features_dict, orient='index')
        cluster_df.index.name = 'cluster'

        # Save results
        output_path = os.path.join(data_dir, f"{prefix}-cluster-features-{method}.csv")
        cluster_df.to_csv(output_path)

        if verbose:
            print(f"    Aggregated {len(cluster_df)} clusters, {len(cluster_df.columns)} features")
            print(f"    Saved: {output_path}")

        results[prefix] = cluster_df

    return results


def classify_clusters(
    data_dir: str,
    file_prefix: Optional[str] = None,
    method: str = 'hdbscan',
    cluster_features: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, List[Dict]]:
    """
    Classify clusters into PD types.

    Args:
        data_dir: Directory containing cluster feature files
        file_prefix: Optional specific file prefix
        method: Clustering method used
        cluster_features: Features to use for classification
        verbose: Print progress

    Returns:
        Dict mapping prefix to list of classification results
    """
    # Find cluster feature files
    if file_prefix:
        files = [os.path.join(data_dir, f"{file_prefix}-cluster-features-{method}.csv")]
    else:
        files = glob.glob(os.path.join(data_dir, f"*-cluster-features-{method}.csv"))

    classifier = PDTypeClassifier()
    results = {}

    for filepath in files:
        prefix = os.path.basename(filepath).replace(f"-cluster-features-{method}.csv", "")

        if verbose:
            print(f"\n  Classifying: {prefix}")

        # Load cluster features
        df = pd.read_csv(filepath, index_col=0)

        # Classify each cluster
        classifications = []
        for cluster_label in df.index:
            features = df.loc[cluster_label].to_dict()
            result = classifier.classify(features, int(cluster_label))
            classifications.append(result)

            if verbose:
                label_str = 'noise' if cluster_label == -1 else str(cluster_label)
                print(f"    Cluster {label_str}: {result['pd_type']} ({result['confidence']:.0%})")

        # Save results
        output_path = os.path.join(data_dir, f"{prefix}-pd-types-{method}.csv")
        results_df = pd.DataFrame([
            {
                'cluster': r['cluster_label'],
                'pd_type': r['pd_type'],
                'pd_type_code': r.get('pd_type_code', PD_TYPES.get(r['pd_type'], {}).get('code', -1)),
                'confidence': r['confidence'],
                'n_warnings': len(r.get('warnings', []))
            }
            for r in classifications
        ])
        results_df.to_csv(output_path, index=False)

        if verbose:
            print(f"    Saved: {output_path}")

        results[prefix] = classifications

    return results


def generate_summary_report(
    data_dir: str,
    methods: List[str],
    file_prefix: Optional[str] = None
) -> str:
    """
    Generate comprehensive summary report.

    Args:
        data_dir: Directory containing result files
        methods: List of clustering methods used
        file_prefix: Optional specific file prefix

    Returns:
        Path to generated report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PARTIAL DISCHARGE ANALYSIS - COMPREHENSIVE SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Directory: {data_dir}")
    report_lines.append(f"Pipeline: Integrated (pdlib-based)")
    report_lines.append("")

    # Find datasets
    if file_prefix:
        prefixes = [file_prefix]
    else:
        wfm_files = glob.glob(os.path.join(data_dir, "*-WFMs.txt"))
        prefixes = [os.path.basename(f).replace("-WFMs.txt", "") for f in wfm_files]

    report_lines.append(f"Datasets Analyzed: {len(prefixes)}")
    report_lines.append(f"Clustering Methods: {', '.join(methods)}")
    report_lines.append("")

    # Overall statistics
    total_pulses = 0
    total_clusters = {m: 0 for m in methods}
    pd_type_counts = {m: {t: 0 for t in PD_TYPES.keys()} for m in methods}

    for prefix in sorted(prefixes):
        report_lines.append("=" * 80)
        report_lines.append(f"DATASET: {prefix}")
        report_lines.append("=" * 80)

        # Load features
        features_file = os.path.join(data_dir, f"{prefix}-features.csv")
        if os.path.exists(features_file):
            df = pd.read_csv(features_file, index_col=0)
            pulse_count = len(df)
            total_pulses += pulse_count
            report_lines.append(f"Total Pulses: {pulse_count}")

            # Feature statistics
            report_lines.append("")
            report_lines.append("Key Feature Statistics:")
            report_lines.append("-" * 40)
            key_features = ['phase_angle', 'absolute_amplitude', 'rise_time', 'energy', 'dominant_frequency']
            for feat in key_features:
                if feat in df.columns:
                    values = df[feat].dropna()
                    if len(values) > 0:
                        report_lines.append(f"  {feat}: mean={values.mean():.4g}, std={values.std():.4g}")

        # Classification results per method
        for method in methods:
            pd_types_file = os.path.join(data_dir, f"{prefix}-pd-types-{method}.csv")
            if os.path.exists(pd_types_file):
                report_lines.append("")
                report_lines.append(f"Classification Results ({method.upper()}):")
                report_lines.append("-" * 40)

                types_df = pd.read_csv(pd_types_file)
                for _, row in types_df.iterrows():
                    label = 'noise' if row['cluster'] == -1 else str(int(row['cluster']))
                    report_lines.append(f"  Cluster {label:5s}: {row['pd_type']:15s} (conf: {row['confidence']:.1%})")
                    pd_type_counts[method][row['pd_type']] = pd_type_counts[method].get(row['pd_type'], 0) + 1
                    total_clusters[method] += 1

        report_lines.append("")

    # Overall summary
    report_lines.append("=" * 80)
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Datasets: {len(prefixes)}")
    report_lines.append(f"Total Pulses: {total_pulses}")
    report_lines.append("")

    for method in methods:
        report_lines.append(f"Classification Summary ({method.upper()}):")
        report_lines.append("-" * 40)
        for pd_type, count in sorted(pd_type_counts[method].items()):
            if count > 0:
                pct = count / total_clusters[method] * 100 if total_clusters[method] > 0 else 0
                report_lines.append(f"  {pd_type:15s}: {count:3d} clusters ({pct:.1f}%)")
        report_lines.append("")

    # Write report
    report_path = os.path.join(data_dir, "analysis-summary-report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nSummary report saved: {report_path}")
    return report_path


def parse_feature_weights(weights_str: str) -> Dict[str, float]:
    """Parse feature weights string into dict."""
    weights = {}
    if weights_str:
        for pair in weights_str.split(','):
            if ':' in pair:
                name, weight = pair.split(':')
                weights[name.strip()] = float(weight.strip())
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Run integrated PD analysis pipeline (pdlib-based)"
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
        help=f'Method for polarity calculation (default: {DEFAULT_POLARITY_METHOD})'
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
        help='Comma-separated list of pulse features to use for clustering. '
             'Use "all" for all features, or leave empty for config defaults.'
    )
    parser.add_argument(
        '--cluster-features',
        type=str,
        default=None,
        help='Comma-separated list of cluster features to use for classification'
    )
    parser.add_argument(
        '--feature-weights',
        type=str,
        default=None,
        help='Feature weights for clustering as feature:weight pairs, e.g., "energy:2.0,phase_angle:1.5"'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("INTEGRATED PD ANALYSIS PIPELINE (pdlib-based)")
    print("=" * 70)
    print(f"Data directory: {args.input_dir}")
    print(f"Clustering method: {args.clustering_method}")
    print(f"Polarity method: {args.polarity_method}")
    if args.file:
        print(f"File filter: {args.file}")
    print()

    # Determine clustering methods
    if args.clustering_method == 'both':
        methods = ['dbscan', 'kmeans']
    elif args.clustering_method == 'all':
        methods = ['hdbscan', 'dbscan', 'kmeans']
    else:
        methods = [args.clustering_method]

    # Check HDBSCAN availability
    if 'hdbscan' in methods and not HDBSCAN_AVAILABLE:
        print("WARNING: HDBSCAN not available, falling back to DBSCAN")
        methods = ['dbscan' if m == 'hdbscan' else m for m in methods]

    # Parse pulse features
    if args.pulse_features:
        if args.pulse_features.lower() == 'all':
            pulse_features = None  # Use all
        else:
            pulse_features = [f.strip() for f in args.pulse_features.split(',')]
    elif DEFAULT_CLUSTERING_FEATURES:
        pulse_features = DEFAULT_CLUSTERING_FEATURES
    else:
        pulse_features = None

    # Parse feature weights
    feature_weights = parse_feature_weights(args.feature_weights)

    # Parse cluster features
    cluster_features = None
    if args.cluster_features:
        cluster_features = [f.strip() for f in args.cluster_features.split(',')]

    success = True

    # Step 1: Feature Extraction
    if not args.skip_extraction:
        print("\n" + "=" * 70)
        print("STEP 1: FEATURE EXTRACTION")
        print("=" * 70)
        try:
            extract_features(
                args.input_dir,
                file_prefix=args.file,
                polarity_method=args.polarity_method
            )
            print("\n  ✓ Feature extraction complete")
        except Exception as e:
            print(f"\n  ✗ Feature extraction failed: {e}")
            success = False

    # Step 2: Clustering (for each method)
    if not args.skip_clustering and success:
        for method in methods:
            print("\n" + "=" * 70)
            print(f"STEP 2: CLUSTERING ({method.upper()})")
            print("=" * 70)
            try:
                run_clustering(
                    args.input_dir,
                    file_prefix=args.file,
                    method=method,
                    n_clusters=args.n_clusters,
                    eps=args.eps,
                    min_samples=args.min_samples,
                    pulse_features=pulse_features,
                    feature_weights=feature_weights
                )
                print(f"\n  ✓ Clustering ({method}) complete")
            except Exception as e:
                print(f"\n  ✗ Clustering ({method}) failed: {e}")
                success = False

    # Step 3: Aggregation (for each method)
    if not args.skip_aggregation and success:
        for method in methods:
            print("\n" + "=" * 70)
            print(f"STEP 3: CLUSTER FEATURE AGGREGATION ({method.upper()})")
            print("=" * 70)
            try:
                aggregate_cluster_features(
                    args.input_dir,
                    file_prefix=args.file,
                    method=method
                )
                print(f"\n  ✓ Aggregation ({method}) complete")
            except Exception as e:
                print(f"\n  ✗ Aggregation ({method}) failed: {e}")
                success = False

    # Step 4: Classification (for each method)
    if not args.skip_classification and success:
        for method in methods:
            print("\n" + "=" * 70)
            print(f"STEP 4: PD TYPE CLASSIFICATION ({method.upper()})")
            print("=" * 70)
            try:
                classify_clusters(
                    args.input_dir,
                    file_prefix=args.file,
                    method=method,
                    cluster_features=cluster_features
                )
                print(f"\n  ✓ Classification ({method}) complete")
            except Exception as e:
                print(f"\n  ✗ Classification ({method}) failed: {e}")
                success = False

    # Step 5: Summary Report
    if not args.skip_summary and success:
        print("\n" + "=" * 70)
        print("STEP 5: SUMMARY REPORT")
        print("=" * 70)
        try:
            generate_summary_report(
                args.input_dir,
                methods,
                file_prefix=args.file
            )
            print("\n  ✓ Summary report complete")
        except Exception as e:
            print(f"\n  ✗ Summary report failed: {e}")
            success = False

    # Final status
    print("\n" + "=" * 70)
    if success:
        print("PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("PIPELINE COMPLETED WITH ERRORS")
    print("=" * 70)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
