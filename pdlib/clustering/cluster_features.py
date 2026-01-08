"""
Cluster feature aggregation.

Computes aggregated features for each cluster from PD pulse data.
These features describe the statistical properties of each cluster's
phase-resolved partial discharge (PRPD) pattern.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks

from .definitions import WAVEFORM_FEATURE_NAMES, CLUSTER_FEATURE_NAMES


def trimmed_mean(data: np.ndarray, trim_fraction: float = 0.1) -> float:
    """
    Compute the trimmed mean of data, removing the top and bottom trim_fraction.

    Args:
        data: Array of values
        trim_fraction: Fraction to remove from each end (default 0.1 = 10% from each end)

    Returns:
        Trimmed mean value
    """
    if len(data) == 0:
        return 0.0

    # Handle NaN values
    data = np.array(data)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return 0.0

    if len(data) < 5:
        # Not enough data to trim, just return regular mean
        return np.mean(data)

    return stats.trim_mean(data, trim_fraction)


def fit_weibull(data: np.ndarray) -> Tuple[float, float]:
    """
    Fit Weibull distribution to amplitude data.

    Args:
        data: Array of amplitude values

    Returns:
        Tuple of (alpha, beta) - scale and shape parameters
    """
    if len(data) < 5:
        return 0.0, 0.0

    try:
        data_positive = np.abs(data[data != 0])
        if len(data_positive) < 5:
            return 0.0, 0.0

        # Fit Weibull minimum (standard Weibull)
        params = stats.weibull_min.fit(data_positive, floc=0)
        beta = params[0]  # shape
        alpha = params[2]  # scale

        return alpha, beta
    except:
        return 0.0, 0.0


def compute_prpd_features(
    phases: np.ndarray,
    amplitudes: np.ndarray,
    trigger_times: Optional[np.ndarray] = None,
    ac_frequency: float = 60.0
) -> Dict[str, float]:
    """
    Compute aggregated PRPD features for a single cluster.

    Args:
        phases: Array of phase angles (degrees, 0-360)
        amplitudes: Array of amplitudes (can be positive or negative)
        trigger_times: Optional array of trigger times for repetition rate
        ac_frequency: AC line frequency in Hz

    Returns:
        Dictionary of feature names to values
    """
    features = {}
    n_pulses = len(phases)

    if n_pulses == 0:
        return {name: 0.0 for name in CLUSTER_FEATURE_NAMES}

    # Normalize phases to 0-360 range
    phases = phases % 360

    # Split by polarity (positive vs negative half-cycle)
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
    abs_amplitudes = np.abs(amplitudes)
    Hqn_full = np.zeros(n_bins)
    for i, (phase, amp) in enumerate(zip(phases, abs_amplitudes)):
        bin_idx = min(int(phase / 10), n_bins - 1)
        Hqn_full[bin_idx] += amp

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
    if np.max(Hn_full) > 0:
        max_bin = np.argmax(Hn_full)
        features['phase_of_max_activity'] = bin_centers[max_bin]
    else:
        features['phase_of_max_activity'] = 0.0

    # Phase spread (circular standard deviation)
    if len(phases) > 1:
        phases_rad = np.deg2rad(phases)
        mean_sin = np.mean(np.sin(phases_rad))
        mean_cos = np.mean(np.cos(phases_rad))
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        if R > 0 and R < 1:
            features['phase_spread'] = np.rad2deg(np.sqrt(-2 * np.log(R)))
        else:
            features['phase_spread'] = 0.0 if R >= 1 else 180.0
    else:
        features['phase_spread'] = 0.0

    # Inception and extinction phases
    threshold = np.max(Hn_full) * 0.05 if np.max(Hn_full) > 0 else 0
    active_bins = np.where(Hn_full > threshold)[0]
    if len(active_bins) > 0:
        features['inception_phase'] = bin_centers[active_bins[0]]
        features['extinction_phase'] = bin_centers[active_bins[-1]]
    else:
        features['inception_phase'] = 0.0
        features['extinction_phase'] = 0.0

    # === QUADRANT PERCENTAGES ===
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
    alpha, beta = fit_weibull(np.abs(amplitudes))
    features['weibull_alpha'] = alpha
    features['weibull_beta'] = beta

    # === REPETITION RATE ===
    if trigger_times is not None and len(trigger_times) > 1:
        duration = np.max(trigger_times) - np.min(trigger_times)
        if duration > 0:
            features['repetition_rate'] = n_pulses / duration
        else:
            features['repetition_rate'] = 0.0
    else:
        features['repetition_rate'] = n_pulses * ac_frequency / 360.0

    # === AMPLITUDE-PHASE CORRELATION ===
    if len(phases) > 10 and len(amplitudes) > 10:
        phases_rad = np.deg2rad(phases)
        expected_pattern = np.abs(np.sin(phases_rad))

        abs_amplitudes = np.abs(amplitudes)
        if np.max(abs_amplitudes) > 0:
            normalized_amplitudes = abs_amplitudes / np.max(abs_amplitudes)
        else:
            normalized_amplitudes = abs_amplitudes

        if np.std(normalized_amplitudes) > 0 and np.std(expected_pattern) > 0:
            correlation = np.corrcoef(normalized_amplitudes, expected_pattern)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            features['amplitude_phase_correlation'] = correlation
        else:
            features['amplitude_phase_correlation'] = 0.0
    else:
        features['amplitude_phase_correlation'] = 0.0

    return features


def compute_waveform_aggregates(
    features_matrix: np.ndarray,
    feature_names: List[str],
    mask: np.ndarray,
    trim_fraction: float = 0.1
) -> Dict[str, float]:
    """
    Compute mean and trimmed mean for all waveform features for a cluster.

    Args:
        features_matrix: Full features matrix (n_pulses x n_features)
        feature_names: List of feature names
        mask: Boolean mask for pulses in this cluster
        trim_fraction: Fraction to trim from each end for trimmed mean

    Returns:
        Dictionary with mean_<feature> and trimmed_mean_<feature> for each feature
    """
    aggregates = {}
    cluster_features = features_matrix[mask]

    for i, feat_name in enumerate(feature_names):
        if feat_name in WAVEFORM_FEATURE_NAMES:
            feat_values = cluster_features[:, i]

            # Handle infinite and NaN values
            feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=0.0, neginf=0.0)

            if len(feat_values) > 0:
                aggregates[f'mean_{feat_name}'] = np.mean(feat_values)
                aggregates[f'trimmed_mean_{feat_name}'] = trimmed_mean(feat_values, trim_fraction)
            else:
                aggregates[f'mean_{feat_name}'] = 0.0
                aggregates[f'trimmed_mean_{feat_name}'] = 0.0

    # Ensure all expected features are present
    for feat_name in WAVEFORM_FEATURE_NAMES:
        if f'mean_{feat_name}' not in aggregates:
            aggregates[f'mean_{feat_name}'] = 0.0
            aggregates[f'trimmed_mean_{feat_name}'] = 0.0

    return aggregates


def compute_cluster_features(
    features_matrix: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
    trigger_times: Optional[np.ndarray] = None,
    ac_frequency: float = 60.0
) -> Dict[int, Dict[str, float]]:
    """
    Compute all aggregated features for each cluster.

    Args:
        features_matrix: Full features matrix (n_pulses x n_features)
        feature_names: List of feature names
        labels: Cluster labels for each pulse
        trigger_times: Optional trigger times for repetition rate
        ac_frequency: AC line frequency

    Returns:
        Dictionary mapping cluster labels to feature dictionaries
    """
    # Find required column indices
    phase_idx = feature_names.index('phase_angle') if 'phase_angle' in feature_names else None
    amp_pos_idx = feature_names.index('peak_amplitude_positive') if 'peak_amplitude_positive' in feature_names else None
    amp_neg_idx = feature_names.index('peak_amplitude_negative') if 'peak_amplitude_negative' in feature_names else None
    polarity_idx = feature_names.index('polarity') if 'polarity' in feature_names else None

    if phase_idx is None:
        raise ValueError("Phase angle not found in features")

    phases = features_matrix[:, phase_idx]

    # Compute signed amplitude
    if amp_pos_idx is not None and amp_neg_idx is not None and polarity_idx is not None:
        amplitudes = np.where(
            features_matrix[:, polarity_idx] > 0,
            features_matrix[:, amp_pos_idx],
            features_matrix[:, amp_neg_idx]
        )
    elif amp_pos_idx is not None:
        amplitudes = features_matrix[:, amp_pos_idx]
    else:
        amplitudes = np.ones(len(phases))

    # Compute features for each cluster
    cluster_features = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        cluster_phases = phases[mask]
        cluster_amplitudes = amplitudes[mask]
        cluster_times = trigger_times[mask] if trigger_times is not None else None

        # Compute PRPD features
        prpd_features = compute_prpd_features(
            cluster_phases,
            cluster_amplitudes,
            cluster_times,
            ac_frequency
        )

        # Compute waveform aggregates
        waveform_aggregates = compute_waveform_aggregates(
            features_matrix, feature_names, mask
        )

        # Merge all features
        cluster_features[label] = {**prpd_features, **waveform_aggregates}

    return cluster_features
