#!/usr/bin/env python3
"""
Cluster Feature Aggregation Script

Computes accumulated/aggregated features for each cluster from PD pulse data.
These features describe the statistical properties of each cluster's phase-resolved
partial discharge (PRPD) pattern.

Usage:
    python aggregate_cluster_features.py [--input-dir DIR] [--method METHOD]

Features computed for each cluster:
- Phase distribution statistics (Hn: pulse count histogram)
- Charge-weighted distribution statistics (Hqn)
- Amplitude statistics
- Phase-related metrics
- Weibull distribution parameters
"""

import numpy as np
import os
import glob
import argparse
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import warnings

DATA_DIR = "Rugged Data Files"

# Aggregated feature names
CLUSTER_FEATURE_NAMES = [
    'pulses_per_positive_halfcycle',
    'pulses_per_negative_halfcycle',
    'pulses_per_cycle',
    'cross_correlation',
    'discharge_asymmetry',
    'skewness_Hn_positive',
    'skewness_Hn_negative',
    'kurtosis_Hn_positive',
    'kurtosis_Hn_negative',
    'skewness_Hqn_positive',
    'skewness_Hqn_negative',
    'kurtosis_Hqn_positive',
    'kurtosis_Hqn_negative',
    'mean_amplitude_positive',
    'mean_amplitude_negative',
    'max_amplitude_positive',
    'max_amplitude_negative',
    'number_of_peaks_Hn_positive',
    'number_of_peaks_Hn_negative',
    'phase_of_max_activity',
    'phase_spread',
    'inception_phase',
    'extinction_phase',
    'quadrant_1_percentage',
    'quadrant_2_percentage',
    'quadrant_3_percentage',
    'quadrant_4_percentage',
    'weibull_alpha',
    'weibull_beta',
    'variance_amplitude_positive',
    'variance_amplitude_negative',
    'coefficient_of_variation',
    'repetition_rate',
]


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
                values = [float(v) for v in parts[1:]]
                features.append(values)

    return np.array(features), feature_names


def load_cluster_labels(filepath):
    """Load cluster labels from cluster CSV file."""
    labels = []
    metadata = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # Parse metadata
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            elif line and not line.startswith('waveform_index'):
                parts = line.split(',')
                if len(parts) >= 2:
                    labels.append(int(parts[1]))

    return np.array(labels), metadata


def weibull_cdf(x, alpha, beta):
    """Weibull cumulative distribution function."""
    return 1 - np.exp(-(x / alpha) ** beta)


def fit_weibull(data):
    """
    Fit Weibull distribution to amplitude data.

    Returns:
        alpha: Scale parameter
        beta: Shape parameter
    """
    if len(data) < 5:
        return 0.0, 0.0

    try:
        # Use scipy's fit method
        data_positive = np.abs(data[data != 0])
        if len(data_positive) < 5:
            return 0.0, 0.0

        # Fit Weibull minimum (which is standard Weibull)
        params = stats.weibull_min.fit(data_positive, floc=0)
        beta = params[0]  # shape
        alpha = params[2]  # scale

        return alpha, beta
    except:
        return 0.0, 0.0


def compute_cluster_features(phases, amplitudes, trigger_times=None, ac_frequency=60.0):
    """
    Compute aggregated features for a single cluster.

    Args:
        phases: Array of phase angles (degrees, 0-360)
        amplitudes: Array of amplitudes (can be positive or negative)
        trigger_times: Optional array of trigger times for repetition rate
        ac_frequency: AC line frequency in Hz

    Returns:
        dict: Dictionary of feature names to values
    """
    features = {}
    n_pulses = len(phases)

    if n_pulses == 0:
        return {name: 0.0 for name in CLUSTER_FEATURE_NAMES}

    # Normalize phases to 0-360 range
    phases = phases % 360

    # Split by polarity (positive vs negative half-cycle)
    # Positive half-cycle: 0-180 degrees
    # Negative half-cycle: 180-360 degrees
    positive_mask = (phases >= 0) & (phases < 180)
    negative_mask = (phases >= 180) & (phases < 360)

    positive_phases = phases[positive_mask]
    negative_phases = phases[negative_mask]
    positive_amplitudes = amplitudes[positive_mask]
    negative_amplitudes = amplitudes[negative_mask]

    # === PULSE COUNT FEATURES ===
    features['pulses_per_positive_halfcycle'] = len(positive_phases)
    features['pulses_per_negative_halfcycle'] = len(negative_phases)
    features['pulses_per_cycle'] = len(positive_phases) + len(negative_phases)

    # === PHASE DISTRIBUTION (Hn) - Histogram of pulse counts ===
    n_bins = 36  # 10-degree bins
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    Hn_full, _ = np.histogram(phases, bins=bin_edges)
    Hn_positive, _ = np.histogram(positive_phases, bins=np.linspace(0, 180, n_bins // 2 + 1))
    Hn_negative, _ = np.histogram(negative_phases, bins=np.linspace(180, 360, n_bins // 2 + 1))

    # Skewness and Kurtosis of Hn
    if len(Hn_positive) > 2 and np.sum(Hn_positive) > 0:
        features['skewness_Hn_positive'] = stats.skew(Hn_positive)
        features['kurtosis_Hn_positive'] = stats.kurtosis(Hn_positive)
    else:
        features['skewness_Hn_positive'] = 0.0
        features['kurtosis_Hn_positive'] = 0.0

    if len(Hn_negative) > 2 and np.sum(Hn_negative) > 0:
        features['skewness_Hn_negative'] = stats.skew(Hn_negative)
        features['kurtosis_Hn_negative'] = stats.kurtosis(Hn_negative)
    else:
        features['skewness_Hn_negative'] = 0.0
        features['kurtosis_Hn_negative'] = 0.0

    # === CHARGE-WEIGHTED DISTRIBUTION (Hqn) ===
    # Weight histogram by absolute amplitude (proxy for charge)
    abs_amplitudes = np.abs(amplitudes)

    # Create charge-weighted histogram
    Hqn_full = np.zeros(n_bins)
    for i, (phase, amp) in enumerate(zip(phases, abs_amplitudes)):
        bin_idx = min(int(phase / 10), n_bins - 1)
        Hqn_full[bin_idx] += amp

    # Split by half-cycle
    Hqn_positive = Hqn_full[:n_bins // 2]
    Hqn_negative = Hqn_full[n_bins // 2:]

    # Skewness and Kurtosis of Hqn
    if np.sum(Hqn_positive) > 0:
        features['skewness_Hqn_positive'] = stats.skew(Hqn_positive)
        features['kurtosis_Hqn_positive'] = stats.kurtosis(Hqn_positive)
    else:
        features['skewness_Hqn_positive'] = 0.0
        features['kurtosis_Hqn_positive'] = 0.0

    if np.sum(Hqn_negative) > 0:
        features['skewness_Hqn_negative'] = stats.skew(Hqn_negative)
        features['kurtosis_Hqn_negative'] = stats.kurtosis(Hqn_negative)
    else:
        features['skewness_Hqn_negative'] = 0.0
        features['kurtosis_Hqn_negative'] = 0.0

    # === CROSS-CORRELATION ===
    # Cross-correlation between positive and negative half-cycle patterns
    if len(Hn_positive) > 0 and len(Hn_negative) > 0:
        Hn_pos_norm = Hn_positive - np.mean(Hn_positive)
        Hn_neg_norm = Hn_negative - np.mean(Hn_negative)
        if np.std(Hn_pos_norm) > 0 and np.std(Hn_neg_norm) > 0:
            features['cross_correlation'] = np.corrcoef(Hn_pos_norm, Hn_neg_norm)[0, 1]
        else:
            features['cross_correlation'] = 0.0
    else:
        features['cross_correlation'] = 0.0

    # === DISCHARGE ASYMMETRY ===
    total_positive = len(positive_phases)
    total_negative = len(negative_phases)
    total = total_positive + total_negative
    if total > 0:
        features['discharge_asymmetry'] = (total_positive - total_negative) / total
    else:
        features['discharge_asymmetry'] = 0.0

    # === AMPLITUDE STATISTICS ===
    if len(positive_amplitudes) > 0:
        features['mean_amplitude_positive'] = np.mean(np.abs(positive_amplitudes))
        features['max_amplitude_positive'] = np.max(np.abs(positive_amplitudes))
        features['variance_amplitude_positive'] = np.var(np.abs(positive_amplitudes))
    else:
        features['mean_amplitude_positive'] = 0.0
        features['max_amplitude_positive'] = 0.0
        features['variance_amplitude_positive'] = 0.0

    if len(negative_amplitudes) > 0:
        features['mean_amplitude_negative'] = np.mean(np.abs(negative_amplitudes))
        features['max_amplitude_negative'] = np.max(np.abs(negative_amplitudes))
        features['variance_amplitude_negative'] = np.var(np.abs(negative_amplitudes))
    else:
        features['mean_amplitude_negative'] = 0.0
        features['max_amplitude_negative'] = 0.0
        features['variance_amplitude_negative'] = 0.0

    # === COEFFICIENT OF VARIATION ===
    all_abs_amplitudes = np.abs(amplitudes)
    if len(all_abs_amplitudes) > 0 and np.mean(all_abs_amplitudes) > 0:
        features['coefficient_of_variation'] = np.std(all_abs_amplitudes) / np.mean(all_abs_amplitudes)
    else:
        features['coefficient_of_variation'] = 0.0

    # === NUMBER OF PEAKS IN Hn ===
    # Find peaks in the histogram distributions
    if len(Hn_positive) > 2:
        peaks_pos, _ = find_peaks(Hn_positive, height=np.max(Hn_positive) * 0.1)
        features['number_of_peaks_Hn_positive'] = len(peaks_pos)
    else:
        features['number_of_peaks_Hn_positive'] = 0

    if len(Hn_negative) > 2:
        peaks_neg, _ = find_peaks(Hn_negative, height=np.max(Hn_negative) * 0.1)
        features['number_of_peaks_Hn_negative'] = len(peaks_neg)
    else:
        features['number_of_peaks_Hn_negative'] = 0

    # === PHASE FEATURES ===
    # Phase of maximum activity (phase bin with most pulses)
    if np.max(Hn_full) > 0:
        max_bin = np.argmax(Hn_full)
        features['phase_of_max_activity'] = bin_centers[max_bin]
    else:
        features['phase_of_max_activity'] = 0.0

    # Phase spread (standard deviation of phase distribution)
    if len(phases) > 1:
        # Use circular statistics for phase spread
        phases_rad = np.deg2rad(phases)
        mean_sin = np.mean(np.sin(phases_rad))
        mean_cos = np.mean(np.cos(phases_rad))
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        # Circular standard deviation
        if R > 0 and R < 1:
            features['phase_spread'] = np.rad2deg(np.sqrt(-2 * np.log(R)))
        else:
            features['phase_spread'] = 0.0 if R >= 1 else 180.0
    else:
        features['phase_spread'] = 0.0

    # Inception and extinction phases
    # Find first and last significant activity
    threshold = np.max(Hn_full) * 0.05 if np.max(Hn_full) > 0 else 0
    active_bins = np.where(Hn_full > threshold)[0]
    if len(active_bins) > 0:
        features['inception_phase'] = bin_centers[active_bins[0]]
        features['extinction_phase'] = bin_centers[active_bins[-1]]
    else:
        features['inception_phase'] = 0.0
        features['extinction_phase'] = 0.0

    # === QUADRANT PERCENTAGES ===
    # Q1: 0-90, Q2: 90-180, Q3: 180-270, Q4: 270-360
    q1_mask = (phases >= 0) & (phases < 90)
    q2_mask = (phases >= 90) & (phases < 180)
    q3_mask = (phases >= 180) & (phases < 270)
    q4_mask = (phases >= 270) & (phases < 360)

    if n_pulses > 0:
        features['quadrant_1_percentage'] = np.sum(q1_mask) / n_pulses * 100
        features['quadrant_2_percentage'] = np.sum(q2_mask) / n_pulses * 100
        features['quadrant_3_percentage'] = np.sum(q3_mask) / n_pulses * 100
        features['quadrant_4_percentage'] = np.sum(q4_mask) / n_pulses * 100
    else:
        features['quadrant_1_percentage'] = 0.0
        features['quadrant_2_percentage'] = 0.0
        features['quadrant_3_percentage'] = 0.0
        features['quadrant_4_percentage'] = 0.0

    # === WEIBULL PARAMETERS ===
    # Fit Weibull distribution to absolute amplitudes
    alpha, beta = fit_weibull(np.abs(amplitudes))
    features['weibull_alpha'] = alpha
    features['weibull_beta'] = beta

    # === REPETITION RATE ===
    # If trigger times available, compute pulses per second
    if trigger_times is not None and len(trigger_times) > 1:
        duration = np.max(trigger_times) - np.min(trigger_times)
        if duration > 0:
            features['repetition_rate'] = n_pulses / duration
        else:
            features['repetition_rate'] = 0.0
    else:
        # Estimate based on AC frequency (pulses per cycle)
        # Assume data covers multiple AC cycles
        features['repetition_rate'] = n_pulses * ac_frequency / 360.0  # Approximate

    return features


def process_dataset(prefix, data_dir, cluster_method='dbscan'):
    """
    Process a dataset and compute aggregated features for each cluster.

    Args:
        prefix: Dataset prefix
        data_dir: Directory containing data files
        cluster_method: 'dbscan' or 'kmeans'

    Returns:
        dict: {cluster_label: features_dict}
    """
    # File paths
    features_file = os.path.join(data_dir, f"{prefix}-features.csv")
    cluster_file = os.path.join(data_dir, f"{prefix}-clusters-{cluster_method}.csv")
    sg_file = os.path.join(data_dir, f"{prefix}-SG.txt")
    ti_file = os.path.join(data_dir, f"{prefix}-Ti.txt")

    # Check files exist
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    print(f"\nProcessing: {prefix}")

    # Load features
    print("  Loading features...")
    features, feature_names = load_features(features_file)

    # Find phase and amplitude column indices
    phase_idx = feature_names.index('phase_angle') if 'phase_angle' in feature_names else None
    amp_pos_idx = feature_names.index('peak_amplitude_positive') if 'peak_amplitude_positive' in feature_names else None
    amp_neg_idx = feature_names.index('peak_amplitude_negative') if 'peak_amplitude_negative' in feature_names else None
    polarity_idx = feature_names.index('polarity') if 'polarity' in feature_names else None

    if phase_idx is None:
        raise ValueError("Phase angle not found in features")

    # Extract phase and amplitude arrays
    phases = features[:, phase_idx]

    # Compute signed amplitude (use polarity to determine sign)
    if amp_pos_idx is not None and amp_neg_idx is not None and polarity_idx is not None:
        amplitudes = np.where(
            features[:, polarity_idx] > 0,
            features[:, amp_pos_idx],
            features[:, amp_neg_idx]
        )
    elif amp_pos_idx is not None:
        amplitudes = features[:, amp_pos_idx]
    else:
        amplitudes = np.ones(len(phases))

    # Load cluster labels
    print("  Loading cluster labels...")
    labels, cluster_metadata = load_cluster_labels(cluster_file)

    # Load settings for AC frequency
    ac_frequency = 60.0
    if os.path.exists(sg_file):
        with open(sg_file, 'r') as f:
            settings = [float(v) for v in f.read().strip().split('\t') if v.strip()]
            if len(settings) > 9:
                ac_frequency = settings[9]

    # Load trigger times if available
    trigger_times = None
    if os.path.exists(ti_file):
        with open(ti_file, 'r') as f:
            trigger_times = np.array([float(v) for v in f.read().strip().split('\t') if v.strip()])

    # Compute features for each cluster
    print("  Computing aggregated features for each cluster...")
    cluster_features = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        cluster_phases = phases[mask]
        cluster_amplitudes = amplitudes[mask]
        cluster_times = trigger_times[mask] if trigger_times is not None else None

        label_name = "noise" if label == -1 else str(label)
        print(f"    Cluster {label_name}: {np.sum(mask)} pulses")

        cluster_features[label] = compute_cluster_features(
            cluster_phases,
            cluster_amplitudes,
            cluster_times,
            ac_frequency
        )

    return cluster_features, cluster_metadata


def save_aggregated_features(cluster_features, output_path, prefix, cluster_metadata):
    """Save aggregated features to CSV file."""
    with open(output_path, 'w') as f:
        # Write header with metadata
        f.write(f"# Aggregated Cluster Features\n")
        f.write(f"# Source: {prefix}\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Clustering method: {cluster_metadata.get('Method', 'unknown')}\n")
        f.write("#\n")

        # Column header
        header = ['cluster_label', 'n_pulses'] + CLUSTER_FEATURE_NAMES
        f.write(','.join(header) + '\n')

        # Write data for each cluster
        for label in sorted(cluster_features.keys()):
            feats = cluster_features[label]
            n_pulses = int(feats['pulses_per_positive_halfcycle'] + feats['pulses_per_negative_halfcycle'])
            label_str = 'noise' if label == -1 else str(label)

            values = [label_str, str(n_pulses)]
            values += [f'{feats[name]:.6e}' for name in CLUSTER_FEATURE_NAMES]
            f.write(','.join(values) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Compute aggregated features for each PD pulse cluster"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=DATA_DIR,
        help='Directory containing data files'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['dbscan', 'kmeans'],
        default='dbscan',
        help='Clustering method used (default: dbscan)'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Process specific file prefix (default: all files)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CLUSTER FEATURE AGGREGATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Clustering method: {args.method}")
    print("=" * 70)

    # Find files to process
    if args.file:
        prefixes = [args.file]
    else:
        # Find all cluster files
        cluster_files = glob.glob(os.path.join(args.input_dir, f"*-clusters-{args.method}.csv"))
        prefixes = [
            os.path.basename(f).replace(f"-clusters-{args.method}.csv", "")
            for f in cluster_files
        ]

    if not prefixes:
        print(f"No cluster files found for method '{args.method}'!")
        return

    print(f"\nFound {len(prefixes)} dataset(s) to process")

    # Process each dataset
    for prefix in sorted(prefixes):
        try:
            cluster_features, cluster_metadata = process_dataset(
                prefix, args.input_dir, args.method
            )

            # Save results
            output_path = os.path.join(
                args.input_dir,
                f"{prefix}-cluster-features-{args.method}.csv"
            )
            save_aggregated_features(cluster_features, output_path, prefix, cluster_metadata)
            print(f"  Saved to: {output_path}")

            # Print summary
            print(f"\n  Feature Summary (Cluster 0 if exists):")
            first_cluster = 0 if 0 in cluster_features else list(cluster_features.keys())[0]
            feats = cluster_features[first_cluster]
            for name in CLUSTER_FEATURE_NAMES[:10]:
                print(f"    {name}: {feats[name]:.4f}")
            print(f"    ... and {len(CLUSTER_FEATURE_NAMES) - 10} more features")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Aggregation complete!")
    print("=" * 70)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
