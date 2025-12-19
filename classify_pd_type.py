#!/usr/bin/env python3
"""
PD Type Classification Decision Tree

Classifies partial discharge clusters into types:
- Noise: Non-PD signals identified by DBSCAN or failing phase correlation tests
- Corona: Asymmetric discharge in one half-cycle, typically at voltage peaks
- Internal (Void): Symmetric discharge in both half-cycles at voltage peaks
- Surface: Discharge near zero-crossings or with tracking patterns

The decision tree uses accumulated cluster features and validates against
individual pulse characteristics within each cluster.

Usage:
    python classify_pd_type.py [--input-dir DIR] [--method METHOD]
"""

import numpy as np
import os
import glob
import argparse
from datetime import datetime
import warnings

DATA_DIR = "Rugged Data Files"

# =============================================================================
# DECISION TREE THRESHOLDS (Based on PD Research)
# =============================================================================

# Branch 1: Noise Detection
NOISE_THRESHOLDS = {
    'dbscan_noise_label': -1,                 # DBSCAN marks noise as -1
    'min_pulse_count': 10,                    # Minimum pulses for valid cluster
    'max_coefficient_of_variation': 2.0,      # CV too high indicates random noise
    'min_repetition_rate': 1.0,               # Minimum pulses per second
}

# Branch 2: Phase Correlation & Broadband
PHASE_CORRELATION_THRESHOLDS = {
    'min_cross_correlation_symmetric': 0.7,   # High correlation = symmetric discharge
    'max_asymmetry_symmetric': 0.35,          # Low asymmetry = symmetric discharge
    'min_phase_spread': 10.0,                 # Minimum phase spread in degrees
    'max_phase_spread_corona': 60.0,          # Corona is concentrated in phase
}

# Branch 3: Symmetry & Phase Location
SYMMETRY_THRESHOLDS = {
    # Corona: asymmetric discharge
    'min_asymmetry_corona': 0.4,              # |asymmetry| > 0.4 = corona (was 0.6 - too strict)
    'min_halfcycle_dominance_corona': 65.0,   # >65% in one half-cycle suggests corona

    # Internal: symmetric, peaks at 90deg and 270deg
    'internal_phase_q1_range': (45, 135),     # Quadrant 1 peak region
    'internal_phase_q3_range': (225, 315),    # Quadrant 3 peak region

    # Surface: near zero-crossings (0deg, 180deg, 360deg)
    'surface_phase_tolerance': 45,            # Degrees from zero-crossing
}

# Branch 4: Amplitude Characteristics
AMPLITUDE_THRESHOLDS = {
    'corona_amplitude_ratio_threshold': 3.0,  # max/mean ratio for corona
    'internal_weibull_beta_min': 2.0,         # Weibull shape > 2 = internal
    'internal_weibull_beta_max': 15.0,        # Upper bound for typical internal
}

# Branch 5: Quadrant Distribution
QUADRANT_THRESHOLDS = {
    'single_halfcycle_threshold': 80.0,       # >80% in one half = corona
    'symmetric_quadrant_min': 15.0,           # Each quadrant >15% = symmetric
    'symmetric_quadrant_max': 35.0,           # Each quadrant <35% = symmetric
}


# =============================================================================
# PD TYPE DEFINITIONS
# =============================================================================

PD_TYPES = {
    'NOISE': {
        'code': 0,
        'description': 'Non-PD or random noise signals',
        'characteristics': [
            'Identified as noise by DBSCAN clustering',
            'Low pulse count or erratic distribution',
            'High coefficient of variation',
            'No clear phase correlation',
        ]
    },
    'CORONA': {
        'code': 1,
        'description': 'Corona discharge (surface ionization in gas/air)',
        'characteristics': [
            'Highly asymmetric - predominantly in one half-cycle',
            'Phase concentrated near voltage peaks (0-180deg or 180-360deg)',
            'Fast rise times (<20ns typical)',
            'Higher amplitude variability',
            '"Rabbit ear" or "wing" pattern in PRPD',
        ]
    },
    'INTERNAL': {
        'code': 2,
        'description': 'Internal/void discharge (cavities in solid insulation)',
        'characteristics': [
            'Symmetric discharge in both half-cycles',
            'High cross-correlation between half-cycles (>0.7)',
            'Phase peaks near 90deg and 270deg (voltage peaks)',
            'Uniform amplitude distribution (Weibull beta > 2)',
            'Moderate rise times',
        ]
    },
    'SURFACE': {
        'code': 3,
        'description': 'Surface discharge (tracking/creeping discharge)',
        'characteristics': [
            'Activity near zero-crossings (0deg, 180deg)',
            'Moderate asymmetry',
            'May show tracking patterns',
            'Variable rise times',
            'Can transition to flashover',
        ]
    },
    'UNKNOWN': {
        'code': 4,
        'description': 'Unclassified PD pattern',
        'characteristics': [
            'Does not match known PD signatures',
            'May be mixed or transitional pattern',
            'Requires manual inspection',
        ]
    }
}


class PDTypeClassifier:
    """
    Decision tree classifier for partial discharge types.

    The classifier uses a hierarchical decision tree with the following branches:

    Branch 1: Noise Detection
        - DBSCAN noise label (-1)
        - Low pulse count
        - High coefficient of variation

    Branch 2: Phase Correlation
        - Cross-correlation between half-cycles
        - Discharge asymmetry
        - Phase spread

    Branch 3: Symmetry Analysis
        - Quadrant distribution
        - Half-cycle balance

    Branch 4: Phase Location
        - Peak phase positions
        - Inception/extinction phases

    Branch 5: Amplitude Characteristics
        - Weibull parameters
        - Amplitude ratios
    """

    def __init__(self, verbose=True, selected_features=None):
        self.verbose = verbose
        self.classification_log = []
        # If selected_features is None, use all features
        self.selected_features = selected_features

    def _is_feature_enabled(self, feature_name):
        """Check if a feature should be used in classification."""
        if self.selected_features is None:
            return True
        return feature_name in self.selected_features

    def _get_feature(self, cluster_features, feature_name, default=0):
        """Get a feature value if enabled, otherwise return default."""
        if not self._is_feature_enabled(feature_name):
            return default
        return cluster_features.get(feature_name, default)

    def classify(self, cluster_features, cluster_label, pulse_features=None):
        """
        Classify a cluster into a PD type.

        Args:
            cluster_features: Dict of aggregated cluster features
            cluster_label: Original cluster label (e.g., -1 for noise)
            pulse_features: Optional array of individual pulse features for validation

        Returns:
            dict: Classification result with type, confidence, and reasoning
        """
        result = {
            'cluster_label': cluster_label,
            'pd_type': 'UNKNOWN',
            'confidence': 0.0,
            'branch_path': [],
            'reasoning': [],
            'warnings': [],
        }

        # Extract key features using helper to respect feature selection
        n_pulses = self._get_feature(cluster_features, 'pulses_per_positive_halfcycle', 0) + \
                   self._get_feature(cluster_features, 'pulses_per_negative_halfcycle', 0)

        # =====================================================================
        # BRANCH 1: NOISE DETECTION
        # =====================================================================
        result['branch_path'].append('Branch 1: Noise Detection')

        # Check DBSCAN noise label
        if cluster_label == NOISE_THRESHOLDS['dbscan_noise_label']:
            result['pd_type'] = 'NOISE'
            result['confidence'] = 0.95
            result['reasoning'].append(
                f"DBSCAN identified as noise (label={cluster_label})"
            )
            return result

        # Check minimum pulse count
        if n_pulses < NOISE_THRESHOLDS['min_pulse_count']:
            result['pd_type'] = 'NOISE'
            result['confidence'] = 0.8
            result['reasoning'].append(
                f"Insufficient pulse count ({n_pulses} < {NOISE_THRESHOLDS['min_pulse_count']})"
            )
            return result

        # Check coefficient of variation
        cv = self._get_feature(cluster_features, 'coefficient_of_variation', 0)
        if cv > NOISE_THRESHOLDS['max_coefficient_of_variation']:
            result['warnings'].append(
                f"High amplitude variability (CV={cv:.2f}) - possible noise contamination"
            )

        result['reasoning'].append(f"Passed noise detection (n={n_pulses}, CV={cv:.2f})")

        # =====================================================================
        # BRANCH 2: PHASE CORRELATION & SYMMETRY
        # =====================================================================
        result['branch_path'].append('Branch 2: Phase Correlation')

        cross_corr = self._get_feature(cluster_features, 'cross_correlation', 0)
        asymmetry = self._get_feature(cluster_features, 'discharge_asymmetry', 0)
        phase_spread = self._get_feature(cluster_features, 'phase_spread', 0)

        is_symmetric = (
            cross_corr > PHASE_CORRELATION_THRESHOLDS['min_cross_correlation_symmetric'] and
            abs(asymmetry) < PHASE_CORRELATION_THRESHOLDS['max_asymmetry_symmetric']
        )

        is_highly_asymmetric = abs(asymmetry) > SYMMETRY_THRESHOLDS['min_asymmetry_corona']

        # Negative cross-correlation indicates anti-correlated half-cycles (corona signature)
        has_negative_correlation = cross_corr < -0.3

        result['reasoning'].append(
            f"Phase correlation: cross_corr={cross_corr:.3f}, asymmetry={asymmetry:.3f}, "
            f"spread={phase_spread:.1f}deg"
        )

        # =====================================================================
        # BRANCH 3: QUADRANT DISTRIBUTION
        # =====================================================================
        result['branch_path'].append('Branch 3: Quadrant Distribution')

        q1 = self._get_feature(cluster_features, 'quadrant_1_percentage', 0)
        q2 = self._get_feature(cluster_features, 'quadrant_2_percentage', 0)
        q3 = self._get_feature(cluster_features, 'quadrant_3_percentage', 0)
        q4 = self._get_feature(cluster_features, 'quadrant_4_percentage', 0)

        positive_half = q1 + q2  # 0-180deg
        negative_half = q3 + q4  # 180-360deg

        result['reasoning'].append(
            f"Quadrant distribution: Q1={q1:.1f}%, Q2={q2:.1f}%, Q3={q3:.1f}%, Q4={q4:.1f}%"
        )
        result['reasoning'].append(
            f"Half-cycle: positive={positive_half:.1f}%, negative={negative_half:.1f}%"
        )

        # =====================================================================
        # BRANCH 4: PHASE LOCATION ANALYSIS
        # =====================================================================
        result['branch_path'].append('Branch 4: Phase Location')

        phase_max = self._get_feature(cluster_features, 'phase_of_max_activity', 0)
        inception = self._get_feature(cluster_features, 'inception_phase', 0)
        extinction = self._get_feature(cluster_features, 'extinction_phase', 0)

        # Check if near zero-crossings (0deg, 180deg, 360deg)
        near_zero_crossing = (
            phase_max < SYMMETRY_THRESHOLDS['surface_phase_tolerance'] or
            abs(phase_max - 180) < SYMMETRY_THRESHOLDS['surface_phase_tolerance'] or
            phase_max > (360 - SYMMETRY_THRESHOLDS['surface_phase_tolerance'])
        )

        # Check if at voltage peaks (90deg, 270deg)
        near_positive_peak = 45 < phase_max < 135
        near_negative_peak = 225 < phase_max < 315

        result['reasoning'].append(
            f"Phase location: max_activity={phase_max:.1f}deg, "
            f"inception={inception:.1f}deg, extinction={extinction:.1f}deg"
        )

        # =====================================================================
        # BRANCH 5: AMPLITUDE CHARACTERISTICS
        # =====================================================================
        result['branch_path'].append('Branch 5: Amplitude Analysis')

        weibull_beta = self._get_feature(cluster_features, 'weibull_beta', 0)
        mean_amp_pos = self._get_feature(cluster_features, 'mean_amplitude_positive', 0)
        mean_amp_neg = self._get_feature(cluster_features, 'mean_amplitude_negative', 0)
        max_amp_pos = self._get_feature(cluster_features, 'max_amplitude_positive', 0)
        max_amp_neg = self._get_feature(cluster_features, 'max_amplitude_negative', 0)

        # Amplitude ratio (max/mean) for variability assessment
        mean_amp = max(mean_amp_pos, mean_amp_neg) if max(mean_amp_pos, mean_amp_neg) > 0 else 1e-10
        max_amp = max(max_amp_pos, max_amp_neg)
        amplitude_ratio = max_amp / mean_amp if mean_amp > 0 else 0

        result['reasoning'].append(
            f"Amplitude: Weibull_beta={weibull_beta:.2f}, amp_ratio={amplitude_ratio:.2f}"
        )

        # =====================================================================
        # CLASSIFICATION DECISION
        # =====================================================================
        result['branch_path'].append('Classification Decision')

        # Decision logic with confidence scoring
        confidence_factors = []

        # ----- CORONA DETECTION -----
        # Check for corona if: highly asymmetric OR has half-cycle dominance with negative correlation
        halfcycle_dominant = (
            positive_half > SYMMETRY_THRESHOLDS['min_halfcycle_dominance_corona'] or
            negative_half > SYMMETRY_THRESHOLDS['min_halfcycle_dominance_corona']
        )
        corona_candidate = is_highly_asymmetric or (has_negative_correlation and halfcycle_dominant)

        if corona_candidate:
            # Strong asymmetry or negative correlation with half-cycle dominance indicates corona
            corona_confidence = 0.0

            # Factor 1: Asymmetry strength
            asym_factor = min(abs(asymmetry), 1.0)
            corona_confidence += asym_factor * 0.30
            confidence_factors.append(f"asymmetry={asym_factor:.2f}")

            # Factor 2: Single half-cycle dominance
            if positive_half > QUADRANT_THRESHOLDS['single_halfcycle_threshold'] or \
               negative_half > QUADRANT_THRESHOLDS['single_halfcycle_threshold']:
                corona_confidence += 0.25
                confidence_factors.append("single_halfcycle>80%=True")
            elif halfcycle_dominant:
                corona_confidence += 0.20
                confidence_factors.append(f"halfcycle_dominant={max(positive_half, negative_half):.1f}%")

            # Factor 3: Negative cross-correlation (anti-correlated half-cycles)
            if has_negative_correlation:
                neg_corr_factor = min(abs(cross_corr), 1.0) * 0.20
                corona_confidence += neg_corr_factor
                confidence_factors.append(f"negative_corr={cross_corr:.2f}")

            # Factor 4: Phase concentration (low spread)
            if phase_spread < PHASE_CORRELATION_THRESHOLDS['max_phase_spread_corona']:
                corona_confidence += 0.10
                confidence_factors.append(f"concentrated_phase={phase_spread:.1f}deg")

            # Factor 5: Near voltage peak
            if near_positive_peak or near_negative_peak:
                corona_confidence += 0.10
                confidence_factors.append("near_peak=True")

            # Factor 6: Amplitude variability
            if amplitude_ratio > AMPLITUDE_THRESHOLDS['corona_amplitude_ratio_threshold']:
                corona_confidence += 0.05
                confidence_factors.append(f"high_amp_ratio={amplitude_ratio:.2f}")

            if corona_confidence > 0.45:
                result['pd_type'] = 'CORONA'
                result['confidence'] = min(corona_confidence, 0.95)
                result['reasoning'].append(
                    f"Classified as CORONA: {', '.join(confidence_factors)}"
                )
                return result

        # ----- INTERNAL (VOID) DETECTION -----
        if is_symmetric:
            internal_confidence = 0.0
            confidence_factors = []

            # Factor 1: High cross-correlation
            corr_factor = min(cross_corr, 1.0)
            internal_confidence += corr_factor * 0.30
            confidence_factors.append(f"cross_corr={corr_factor:.2f}")

            # Factor 2: Low asymmetry
            sym_factor = 1.0 - min(abs(asymmetry), 1.0)
            internal_confidence += sym_factor * 0.25
            confidence_factors.append(f"symmetry={sym_factor:.2f}")

            # Factor 3: Balanced quadrant distribution
            all_quadrants_active = all(
                QUADRANT_THRESHOLDS['symmetric_quadrant_min'] < q <
                QUADRANT_THRESHOLDS['symmetric_quadrant_max'] + 15
                for q in [q1, q2, q3, q4]
            )
            if all_quadrants_active:
                internal_confidence += 0.20
                confidence_factors.append("balanced_quadrants=True")

            # Factor 4: Weibull distribution
            if AMPLITUDE_THRESHOLDS['internal_weibull_beta_min'] < weibull_beta < \
               AMPLITUDE_THRESHOLDS['internal_weibull_beta_max']:
                internal_confidence += 0.15
                confidence_factors.append(f"weibull_beta={weibull_beta:.2f}")

            # Factor 5: Phase at voltage peaks
            if near_positive_peak or near_negative_peak:
                internal_confidence += 0.10
                confidence_factors.append("peak_phase=True")

            if internal_confidence > 0.5:
                result['pd_type'] = 'INTERNAL'
                result['confidence'] = min(internal_confidence, 0.95)
                result['reasoning'].append(
                    f"Classified as INTERNAL: {', '.join(confidence_factors)}"
                )
                return result

        # ----- SURFACE DETECTION -----
        if near_zero_crossing:
            surface_confidence = 0.0
            confidence_factors = []

            # Factor 1: Activity near zero-crossing
            surface_confidence += 0.35
            confidence_factors.append("near_zero_crossing=True")

            # Factor 2: Moderate asymmetry
            if 0.2 < abs(asymmetry) < 0.7:
                surface_confidence += 0.25
                confidence_factors.append(f"moderate_asymmetry={asymmetry:.2f}")

            # Factor 3: Phase spread
            if phase_spread > 30:
                surface_confidence += 0.15
                confidence_factors.append(f"phase_spread={phase_spread:.1f}deg")

            # Factor 4: Concentrated in adjacent quadrants
            if (q1 + q4 > 50) or (q2 + q3 > 50):
                surface_confidence += 0.15
                confidence_factors.append("adjacent_quadrants=True")

            if surface_confidence > 0.4:
                result['pd_type'] = 'SURFACE'
                result['confidence'] = min(surface_confidence, 0.90)
                result['reasoning'].append(
                    f"Classified as SURFACE: {', '.join(confidence_factors)}"
                )
                return result

        # ----- FALLBACK: Additional analysis -----
        # Try to classify based on dominant patterns with more permissive thresholds

        # Check for partial corona (less pronounced asymmetry but clear half-cycle dominance)
        if abs(asymmetry) > 0.25 and (positive_half > 60 or negative_half > 60):
            result['pd_type'] = 'CORONA'
            result['confidence'] = 0.55
            result['reasoning'].append(
                f"Weak CORONA signature: asymmetry={asymmetry:.2f}, "
                f"half_cycle={max(positive_half, negative_half):.1f}%"
            )
            return result

        # Check for corona based on strong negative correlation alone
        if cross_corr < -0.5 and (positive_half > 55 or negative_half > 55):
            result['pd_type'] = 'CORONA'
            result['confidence'] = 0.50
            result['reasoning'].append(
                f"CORONA from anti-correlation: cross_corr={cross_corr:.2f}, "
                f"half_cycle={max(positive_half, negative_half):.1f}%"
            )
            return result

        # Check for partial internal (moderate symmetry)
        if cross_corr > 0.5 and abs(asymmetry) < 0.5:
            result['pd_type'] = 'INTERNAL'
            result['confidence'] = 0.55
            result['reasoning'].append(
                f"Weak INTERNAL signature: cross_corr={cross_corr:.2f}, asymmetry={asymmetry:.2f}"
            )
            return result

        # Check for surface discharge (moderate asymmetry, not near peaks)
        if 0.15 < abs(asymmetry) < 0.5 and not (near_positive_peak or near_negative_peak):
            result['pd_type'] = 'SURFACE'
            result['confidence'] = 0.45
            result['reasoning'].append(
                f"Possible SURFACE: asymmetry={asymmetry:.2f}, not near voltage peaks"
            )
            return result

        # Unknown pattern
        result['pd_type'] = 'UNKNOWN'
        result['confidence'] = 0.3
        result['reasoning'].append(
            "Pattern does not match known PD signatures - requires manual inspection"
        )
        result['warnings'].append("Could not reliably classify - manual review recommended")

        return result

    def validate_with_waveforms(self, result, pulse_features, cluster_mask):
        """
        Validate classification by checking individual pulse characteristics.

        Args:
            result: Classification result dict
            pulse_features: Array of all pulse features
            cluster_mask: Boolean mask for pulses in this cluster

        Returns:
            Updated result dict with validation info
        """
        if pulse_features is None or cluster_mask is None:
            return result

        cluster_pulses = pulse_features[cluster_mask]
        if len(cluster_pulses) == 0:
            return result

        result['validation'] = {}

        # Get feature indices (assuming standard order)
        # rise_time is index 5, dominant_frequency is index 16, spectral_power_high is index 20
        try:
            rise_times = cluster_pulses[:, 5]
            dominant_freqs = cluster_pulses[:, 16]
            spectral_high = cluster_pulses[:, 20]

            # Calculate statistics
            mean_rise_time = np.mean(rise_times[rise_times > 0]) if np.any(rise_times > 0) else 0
            mean_freq = np.mean(dominant_freqs)
            mean_spectral_high = np.mean(spectral_high)

            result['validation']['mean_rise_time_ns'] = mean_rise_time * 1e9
            result['validation']['mean_dominant_freq_MHz'] = mean_freq / 1e6
            result['validation']['mean_spectral_power_high'] = mean_spectral_high

            # Validate against PD type expectations
            pd_type = result['pd_type']

            if pd_type == 'CORONA':
                # Corona should have fast rise times and high frequency content
                if mean_rise_time * 1e9 < 30:
                    result['validation']['rise_time_consistent'] = True
                else:
                    result['validation']['rise_time_consistent'] = False
                    result['warnings'].append(
                        f"Rise time ({mean_rise_time*1e9:.1f}ns) higher than typical corona"
                    )

            elif pd_type == 'INTERNAL':
                # Internal has moderate rise times
                if 10e-9 < mean_rise_time < 100e-9:
                    result['validation']['rise_time_consistent'] = True
                else:
                    result['validation']['rise_time_consistent'] = False

        except Exception as e:
            result['warnings'].append(f"Validation error: {e}")

        return result


def load_cluster_features(filepath):
    """Load cluster features from CSV file."""
    clusters = {}

    with open(filepath, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if header is None:
                header = line.split(',')
                continue

            parts = line.split(',')
            if len(parts) < 2:
                continue

            label = parts[0]
            if label == 'noise':
                label = -1
            else:
                label = int(label)

            features = {}
            for i, name in enumerate(header[1:], 1):
                if i < len(parts):
                    try:
                        features[name] = float(parts[i])
                    except:
                        features[name] = 0.0

            clusters[label] = features

    return clusters


def load_cluster_labels(filepath):
    """Load cluster labels for each pulse."""
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('waveform'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                labels.append(int(parts[1]))

    return np.array(labels)


def load_pulse_features(filepath):
    """Load individual pulse features."""
    features = []

    with open(filepath, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1:
                values = [float(v) for v in parts[1:]]
                features.append(values)

    return np.array(features)


def process_dataset(prefix, data_dir, method='dbscan', selected_features=None):
    """
    Process a dataset and classify all clusters.

    Args:
        prefix: Dataset file prefix
        data_dir: Directory containing data files
        method: Clustering method used
        selected_features: List of cluster feature names to use (None = all)

    Returns:
        list: Classification results for each cluster
    """
    cluster_features_file = os.path.join(
        data_dir, f"{prefix}-cluster-features-{method}.csv"
    )
    cluster_labels_file = os.path.join(
        data_dir, f"{prefix}-clusters-{method}.csv"
    )
    pulse_features_file = os.path.join(
        data_dir, f"{prefix}-features.csv"
    )

    print(f"\nProcessing: {prefix}")
    print("-" * 60)

    # Load data
    cluster_features = load_cluster_features(cluster_features_file)
    cluster_labels = load_cluster_labels(cluster_labels_file)
    pulse_features = load_pulse_features(pulse_features_file) if os.path.exists(pulse_features_file) else None

    # Print feature selection info
    if selected_features:
        print(f"  Using {len(selected_features)} selected cluster features")
    else:
        print("  Using all cluster features")

    # Classify each cluster
    classifier = PDTypeClassifier(verbose=True, selected_features=selected_features)
    results = []

    for label, features in sorted(cluster_features.items()):
        result = classifier.classify(features, label)

        # Validate with waveform data
        if pulse_features is not None:
            cluster_mask = cluster_labels == label
            result = classifier.validate_with_waveforms(result, pulse_features, cluster_mask)

        results.append(result)

        # Print result
        label_str = "noise" if label == -1 else str(label)
        n_pulses = int(features.get('pulses_per_positive_halfcycle', 0) +
                      features.get('pulses_per_negative_halfcycle', 0))

        print(f"\n  Cluster {label_str} ({n_pulses} pulses):")
        print(f"    Type: {result['pd_type']} (confidence: {result['confidence']:.1%})")
        print(f"    Path: {' -> '.join(result['branch_path'][:3])}...")

        if result['reasoning']:
            for reason in result['reasoning'][-2:]:
                print(f"    Reason: {reason}")

        if result.get('validation'):
            val = result['validation']
            if 'mean_rise_time_ns' in val:
                print(f"    Validation: rise_time={val['mean_rise_time_ns']:.1f}ns, "
                      f"freq={val['mean_dominant_freq_MHz']:.1f}MHz")

        if result['warnings']:
            for warning in result['warnings']:
                print(f"    WARNING: {warning}")

    return results


def save_classification_results(results, output_path, prefix, method):
    """Save classification results to CSV."""
    with open(output_path, 'w') as f:
        f.write("# PD Type Classification Results\n")
        f.write(f"# Source: {prefix}\n")
        f.write(f"# Method: {method}\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")

        # Header
        f.write("cluster_label,pd_type,pd_type_code,confidence,n_warnings\n")

        # Data
        for result in results:
            label = result['cluster_label']
            label_str = 'noise' if label == -1 else str(label)
            pd_type = result['pd_type']
            pd_code = PD_TYPES[pd_type]['code']
            confidence = result['confidence']
            n_warnings = len(result['warnings'])

            f.write(f"{label_str},{pd_type},{pd_code},{confidence:.4f},{n_warnings}\n")


def generate_summary(all_results, output_path):
    """Generate a summary of all classifications."""
    summary = {
        'NOISE': 0, 'CORONA': 0, 'INTERNAL': 0, 'SURFACE': 0, 'UNKNOWN': 0
    }
    total_pulses = {'NOISE': 0, 'CORONA': 0, 'INTERNAL': 0, 'SURFACE': 0, 'UNKNOWN': 0}

    for dataset_results in all_results:
        for result in dataset_results['results']:
            pd_type = result['pd_type']
            summary[pd_type] += 1

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PD TYPE CLASSIFICATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 40 + "\n")
        total = sum(summary.values())
        for pd_type, count in summary.items():
            pct = count / total * 100 if total > 0 else 0
            f.write(f"  {pd_type:10s}: {count:3d} clusters ({pct:.1f}%)\n")
        f.write(f"\n  Total: {total} clusters\n\n")

        f.write("BY DATASET:\n")
        f.write("-" * 40 + "\n")
        for dataset in all_results:
            f.write(f"\n  {dataset['prefix']}:\n")
            for result in dataset['results']:
                label = result['cluster_label']
                label_str = 'noise' if label == -1 else str(label)
                f.write(f"    Cluster {label_str:5s}: {result['pd_type']:10s} "
                       f"(conf: {result['confidence']:.1%})\n")


def main():
    parser = argparse.ArgumentParser(
        description="Classify PD clusters into discharge types"
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
        help='Process specific file prefix'
    )
    parser.add_argument(
        '--cluster-features',
        type=str,
        default=None,
        help='Comma-separated list of cluster features to use for classification (default: all features)'
    )
    args = parser.parse_args()

    # Parse cluster features list if provided
    selected_features = None
    if args.cluster_features:
        selected_features = [f.strip() for f in args.cluster_features.split(',') if f.strip()]

    print("=" * 70)
    print("PD TYPE CLASSIFICATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Clustering method: {args.method}")
    if selected_features:
        print(f"Cluster features: {len(selected_features)} selected")
    else:
        print("Cluster features: all")
    print("=" * 70)

    # Print decision tree structure
    print("\nDECISION TREE STRUCTURE:")
    print("-" * 40)
    print("  Branch 1: Noise Detection")
    print("    |- DBSCAN noise label (-1)?")
    print("    |- Pulse count < 10?")
    print("    \\- Coefficient of variation > 2.0?")
    print("  Branch 2: Phase Correlation")
    print("    |- Cross-correlation > 0.7? (symmetric)")
    print("    \\- |Asymmetry| < 0.35? (symmetric)")
    print("  Branch 3: Quadrant Distribution")
    print("    |- Single half-cycle > 80%? (corona)")
    print("    \\- All quadrants 15-35%? (internal)")
    print("  Branch 4: Phase Location")
    print("    |- Near zero-crossing? (surface)")
    print("    \\- Near voltage peak? (corona/internal)")
    print("  Branch 5: Amplitude Analysis")
    print("    |- Weibull beta: 2-15? (internal)")
    print("    \\- Amplitude ratio > 3? (corona)")
    print("-" * 40)

    # Find files to process
    if args.file:
        prefixes = [args.file]
    else:
        cluster_files = glob.glob(
            os.path.join(args.input_dir, f"*-cluster-features-{args.method}.csv")
        )
        prefixes = [
            os.path.basename(f).replace(f"-cluster-features-{args.method}.csv", "")
            for f in cluster_files
        ]

    if not prefixes:
        print(f"\nNo cluster feature files found for method '{args.method}'!")
        return

    print(f"\nFound {len(prefixes)} dataset(s) to process")

    # Process each dataset
    all_results = []
    for prefix in sorted(prefixes):
        try:
            results = process_dataset(prefix, args.input_dir, args.method, selected_features)
            all_results.append({'prefix': prefix, 'results': results})

            # Save individual results
            output_path = os.path.join(
                args.input_dir,
                f"{prefix}-pd-types-{args.method}.csv"
            )
            save_classification_results(results, output_path, prefix, args.method)
            print(f"\n  Saved to: {output_path}")

        except Exception as e:
            print(f"\n  ERROR processing {prefix}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary
    summary_path = os.path.join(args.input_dir, f"pd-classification-summary-{args.method}.txt")
    generate_summary(all_results, summary_path)
    print(f"\n{'='*70}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
