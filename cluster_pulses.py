#!/usr/bin/env python3
"""
PD Pulse Clustering Script

Clusters partial discharge pulses using DBSCAN or K-means based on extracted features.

Usage:
    python cluster_pulses.py [--method dbscan|kmeans] [--input FILE] [--n-clusters N]

Options:
    --method        Clustering method: 'dbscan' or 'kmeans' (default: dbscan)
    --input FILE    Input features CSV file (default: process all *-features.csv)
    --n-clusters N  Number of clusters for K-means (default: 5)
    --eps EPS       DBSCAN epsilon parameter (default: auto)
    --min-samples N DBSCAN min_samples parameter (default: 5)
    --features      Comma-separated list of features to use (default: all)
"""

import numpy as np
import os
import glob
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings

DATA_DIR = "Rugged Data Files"


def load_features(filepath):
    """Load features from CSV file."""
    features = []
    feature_names = None

    with open(filepath, 'r') as f:
        header = f.readline().strip()
        feature_names = header.split(',')[1:]  # Skip waveform_index

        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1:
                # Skip waveform_index (first column)
                values = [float(v) for v in parts[1:]]
                features.append(values)

    return np.array(features), feature_names


def estimate_dbscan_eps(X_scaled, k=5):
    """
    Estimate optimal epsilon for DBSCAN using k-nearest neighbors.

    Uses the "elbow method" on k-distance graph.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)

    # Get the k-th nearest neighbor distance for each point
    k_distances = np.sort(distances[:, k-1])

    # Find the "elbow" - point of maximum curvature
    # Use simple heuristic: take distance at 90th percentile
    eps = np.percentile(k_distances, 90)

    return eps


def run_dbscan(X_scaled, eps=None, min_samples=5):
    """
    Run DBSCAN clustering.

    Args:
        X_scaled: Scaled feature matrix
        eps: Epsilon parameter (auto-estimated if None)
        min_samples: Minimum samples for core point

    Returns:
        labels: Cluster labels (-1 for noise)
        info: Dict with clustering info
    """
    if eps is None:
        eps = estimate_dbscan_eps(X_scaled, k=min_samples)
        print(f"  Auto-estimated eps: {eps:.4f}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    info = {
        'method': 'DBSCAN',
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels)
    }

    # Calculate silhouette score if we have valid clusters
    if n_clusters > 1:
        # Exclude noise points for silhouette calculation
        mask = labels != -1
        if mask.sum() > 1:
            try:
                info['silhouette_score'] = silhouette_score(X_scaled[mask], labels[mask])
            except:
                info['silhouette_score'] = None

    return labels, info


def run_kmeans(X_scaled, n_clusters=5):
    """
    Run K-means clustering.

    Args:
        X_scaled: Scaled feature matrix
        n_clusters: Number of clusters

    Returns:
        labels: Cluster labels
        info: Dict with clustering info
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    info = {
        'method': 'KMeans',
        'n_clusters': n_clusters,
        'inertia': kmeans.inertia_,
        'n_iterations': kmeans.n_iter_
    }

    # Calculate silhouette score
    if n_clusters > 1:
        try:
            info['silhouette_score'] = silhouette_score(X_scaled, labels)
        except:
            info['silhouette_score'] = None

    return labels, info


def save_cluster_labels(labels, output_path, feature_file, info, used_features=None):
    """Save cluster labels and metadata to file."""
    with open(output_path, 'w') as f:
        # Write header with metadata
        f.write(f"# Clustering Results\n")
        f.write(f"# Source: {feature_file}\n")
        f.write(f"# Method: {info['method']}\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# N_clusters: {info['n_clusters']}\n")

        if 'eps' in info:
            f.write(f"# DBSCAN_eps: {info['eps']}\n")
            f.write(f"# DBSCAN_min_samples: {info['min_samples']}\n")
            f.write(f"# Noise_points: {info['n_noise']}\n")

        if 'silhouette_score' in info and info['silhouette_score'] is not None:
            f.write(f"# Silhouette_score: {info['silhouette_score']:.4f}\n")

        # Save features used for clustering
        if used_features:
            f.write(f"# Features_used: {','.join(used_features)}\n")

        f.write("#\n")
        f.write("waveform_index,cluster_label\n")

        for i, label in enumerate(labels):
            f.write(f"{i},{label}\n")


def process_file(filepath, method='dbscan', n_clusters=5, eps=None, min_samples=5, selected_features=None):
    """
    Process a single features file and perform clustering.

    Args:
        filepath: Path to features CSV file
        method: 'dbscan' or 'kmeans'
        n_clusters: Number of clusters for K-means
        eps: DBSCAN epsilon (auto if None)
        min_samples: DBSCAN min_samples
        selected_features: List of feature names to use (None = all)

    Returns:
        tuple: (labels, info, output_path)
    """
    print(f"\nProcessing: {os.path.basename(filepath)}")

    # Load features
    print("  Loading features...")
    features, feature_names = load_features(filepath)
    print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")

    # Filter to selected features if specified
    used_features = None
    if selected_features:
        # Find indices of selected features
        feature_indices = []
        used_features = []
        for feat in selected_features:
            if feat in feature_names:
                feature_indices.append(feature_names.index(feat))
                used_features.append(feat)
            else:
                print(f"  Warning: Feature '{feat}' not found, skipping")

        if not feature_indices:
            raise ValueError("No valid features selected for clustering")

        features = features[:, feature_indices]
        print(f"  Using {len(used_features)} selected features: {', '.join(used_features[:5])}{'...' if len(used_features) > 5 else ''}")
    else:
        used_features = feature_names.copy()  # All features used
        print(f"  Using all {len(feature_names)} features")

    # Handle infinite values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    print("  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Run clustering
    if method == 'dbscan':
        print(f"  Running DBSCAN (min_samples={min_samples})...")
        labels, info = run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
    else:
        print(f"  Running K-means (k={n_clusters})...")
        labels, info = run_kmeans(X_scaled, n_clusters=n_clusters)

    # Print results
    print(f"  Found {info['n_clusters']} clusters")
    if 'n_noise' in info:
        print(f"  Noise points: {info['n_noise']} ({info['noise_ratio']*100:.1f}%)")
    if 'silhouette_score' in info and info['silhouette_score'] is not None:
        print(f"  Silhouette score: {info['silhouette_score']:.4f}")

    # Print cluster distribution
    unique_labels = sorted(set(labels))
    print("  Cluster distribution:")
    for label in unique_labels:
        count = np.sum(labels == label)
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        print(f"    {label_name}: {count} pulses ({count/len(labels)*100:.1f}%)")

    # Save results
    output_path = filepath.replace('-features.csv', f'-clusters-{method}.csv')
    save_cluster_labels(labels, output_path, filepath, info, used_features)
    print(f"  Saved to: {output_path}")

    return labels, info, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Cluster PD pulses based on extracted features"
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['dbscan', 'kmeans'],
        default='dbscan',
        help='Clustering method (default: dbscan)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input features CSV file (default: all *-features.csv files)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=DATA_DIR,
        help='Directory containing feature files'
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
        help='DBSCAN epsilon parameter (default: auto-estimate)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=5,
        help='DBSCAN min_samples parameter (default: 5)'
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Comma-separated list of features to use for clustering (default: all features)'
    )
    args = parser.parse_args()

    # Parse features list if provided
    selected_features = None
    if args.features:
        selected_features = [f.strip() for f in args.features.split(',') if f.strip()]

    print("=" * 70)
    print("PD PULSE CLUSTERING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Method: {args.method.upper()}")
    if args.method == 'kmeans':
        print(f"K-means clusters: {args.n_clusters}")
    else:
        print(f"DBSCAN min_samples: {args.min_samples}")
        print(f"DBSCAN eps: {'auto' if args.eps is None else args.eps}")
    if selected_features:
        print(f"Features: {len(selected_features)} selected")
    else:
        print("Features: all")
    print("=" * 70)

    # Find files to process
    if args.input:
        feature_files = [args.input]
    else:
        feature_files = glob.glob(os.path.join(args.input_dir, "*-features.csv"))

    if not feature_files:
        print("No feature files found!")
        return

    print(f"\nFound {len(feature_files)} feature file(s) to process")

    # Process each file
    results = []
    for filepath in sorted(feature_files):
        try:
            labels, info, output_path = process_file(
                filepath,
                method=args.method,
                n_clusters=args.n_clusters,
                eps=args.eps,
                min_samples=args.min_samples,
                selected_features=selected_features
            )
            results.append({
                'file': filepath,
                'labels': labels,
                'info': info,
                'output': output_path
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Clustering complete!")
    print("=" * 70)

    # Summary
    print("\nSummary:")
    for r in results:
        basename = os.path.basename(r['file'])
        print(f"  {basename}: {r['info']['n_clusters']} clusters")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
