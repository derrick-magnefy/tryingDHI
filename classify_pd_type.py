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

# Branch 1: Noise Detection (using waveform characteristics)
NOISE_THRESHOLDS = {
    'dbscan_noise_label': -1,                 # DBSCAN marks noise as -1
    # Spectral characteristics - Noise is < 1 MHz, real PD is higher frequency
    'min_spectral_flatness': 0.6,             # High flatness = random noise (white noise ~1.0)
    'max_bandwidth_3db': 2e6,                 # Noise has < 2 MHz bandwidth
    'max_dominant_frequency': 1e6,            # Noise is < 1 MHz (60Hz hum, EMI)
    # Pulse shape characteristics
    'min_slew_rate': 1e6,                     # Real PD has fast rise (V/s), noise is slower
    'min_crest_factor': 3.0,                  # Real PD is impulsive (high crest factor)
    'max_oscillation_count': 20,              # Excessive ringing = noise/EMI
    # Signal quality
    'min_signal_to_noise_ratio': 3.0,         # Minimum SNR for valid signal (dB)
    'min_cross_correlation': 0.3,             # Shape consistency across pulses
    'max_coefficient_of_variation': 2.0,      # CV too high = random amplitude noise
    # Multi-pulse detection
    'min_pulses_for_multipulse': 2,           # >= 2 pulses per waveform = multi-pulse
}

# Branch 2: Phase Spread Check (Surface PD initial detection)
PHASE_SPREAD_THRESHOLDS = {
    'surface_phase_spread_min': 120.0,        # Phase spread > 120° → Surface PD
}

# Branch 3: Surface PD vs Corona/Internal (8-feature weighted score-based)
SURFACE_DETECTION_THRESHOLDS = {
    # Weights for scoring (max score: 4 + 3*3 + 2*2 + 1*2 = 4 + 9 + 4 + 2 = 19)
    'primary_weight': 4,      # phase_spread
    'secondary_weight': 3,    # slew_rate, spectral_power_ratio, cv
    'mid_weight': 2,          # crest_factor, cross_correlation
    'supporting_weight': 1,   # spectral_flatness, repetition_rate_variance

    # Minimum score to classify as Surface PD (out of max possible 19)
    'min_surface_score': 10,

    # PRIMARY FEATURE (Weight: 4)
    # Feature 1: Phase spread (already checked in Branch 2, but included for scoring)
    'surface_phase_spread': 120.0,            # >120° suggests Surface PD
    'corona_phase_spread': 100.0,             # <100° suggests Corona/Internal

    # SECONDARY FEATURES (Weight: 3)
    # Feature 2: Slew rate (V/s) - Surface has slower rise
    'surface_max_slew_rate': 5e6,             # Surface: Low slew rate
    'corona_min_slew_rate': 1e7,              # Corona/Internal: High slew rate

    # Feature 3: Spectral power ratio (low/high frequency)
    'surface_max_spectral_power_ratio': 0.5,  # Surface: <0.5
    'corona_min_spectral_power_ratio': 0.8,   # Corona/Internal: >0.8

    # Feature 4: Coefficient of variation
    'surface_min_cv': 0.4,                    # Surface: >0.4
    'corona_max_cv': 0.3,                     # Corona/Internal: <0.3

    # MID-WEIGHT FEATURES (Weight: 2)
    # Feature 5: Crest factor
    'surface_min_crest_factor': 4.0,          # Surface: Moderate (4-6)
    'surface_max_crest_factor': 6.0,
    'corona_min_crest_factor': 6.0,           # Corona/Internal: High (>6)

    # Feature 6: Cross-correlation
    'surface_min_cross_corr': 0.4,            # Surface: Lower (0.4-0.6)
    'surface_max_cross_corr': 0.6,
    'corona_min_cross_corr': 0.7,             # Corona/Internal: Higher (>0.7)

    # SUPPORTING FEATURES (Weight: 1)
    # Feature 7: Spectral flatness
    'surface_min_spectral_flatness': 0.4,     # Surface: Higher (0.4-0.5)
    'surface_max_spectral_flatness': 0.5,
    'corona_max_spectral_flatness': 0.35,     # Corona/Internal: Lower (<0.35)

    # Feature 8: Repetition rate variance
    'surface_min_rep_rate_var': 0.5,          # Surface: High variance
    'corona_max_rep_rate_var': 0.3,           # Corona/Internal: Low variance

    # Feature 9: Dominant frequency - Surface PD is 1-5 MHz
    'surface_min_dominant_freq': 1e6,         # Surface: >= 1 MHz
    'surface_max_dominant_freq': 5e6,         # Surface: <= 5 MHz
}

# Branch 4: Corona vs Internal (Score-based detection)
CORONA_INTERNAL_THRESHOLDS = {
    # Weights for scoring
    'primary_weight': 4,
    'secondary_weight': 2,
    'supporting_weight': 1,

    # Minimum score to classify (out of max possible)
    'min_corona_score': 8,
    'min_internal_score': 8,

    # PRIMARY FEATURES (Weight: 4)
    # discharge_asymmetry: Corona (strongly asymmetric), Internal can be moderately asymmetric
    'corona_neg_max_asymmetry': -0.6,         # Negative Corona: < -0.6 (strong negative)
    'corona_pos_min_asymmetry': 0.6,          # Positive Corona: > +0.6 (strong positive)
    'internal_min_asymmetry': -0.9,           # Internal: -0.9 to +0.9 (wide range)
    'internal_max_asymmetry': 0.9,

    # phase_of_max_activity:
    # Negative Corona: 180-270° (expanded from 200-250°)
    # Positive Corona: 0-90° or 270-360°
    # Internal: 45-90° or 225-270° (near voltage peaks)
    'corona_neg_phase_min': 180,              # Negative Corona: 180°-270°
    'corona_neg_phase_max': 270,
    'corona_pos_phase_q1_min': 0,             # Positive Corona: 0°-90°
    'corona_pos_phase_q1_max': 90,
    'corona_pos_phase_q4_min': 270,           # Positive Corona: 270°-360°
    'corona_pos_phase_q4_max': 360,
    'internal_phase_q1_min': 45,              # Internal: 45°-90° or 225°-270°
    'internal_phase_q1_max': 90,
    'internal_phase_q3_min': 225,
    'internal_phase_q3_max': 270,

    # amplitude_phase_correlation: PRIMARY FEATURE - How well amplitudes track sinusoidal reference
    # Internal: High (>0.5), Corona: Low (<0.3) - Strong discriminator
    'internal_min_amp_phase_corr': 0.5,       # Internal: High correlation (>0.5)
    'corona_max_amp_phase_corr': 0.3,         # Corona: Low correlation (<0.3)

    # spectral_power_low: PRIMARY FEATURE - Fraction of power in low frequencies
    # Internal: High (>0.85 = most power in low freq), Corona: Low (<0.60 = significant high freq)
    'internal_min_spectral_power_low': 0.85,  # Internal: >85% power in low frequencies
    'corona_max_spectral_power_low': 0.60,    # Corona: <60% power in low frequencies

    # SECONDARY FEATURES (Weight: 2)
    # slew_rate: Corona very high, Internal moderate
    'corona_min_slew_rate': 5e7,              # Corona: Very high (>50 MV/s)
    'internal_min_slew_rate': 1e7,            # Internal: Moderate (10-50 MV/s)
    'internal_max_slew_rate': 5e7,

    # norm_slew_rate: Normalized slew rate (alternative to raw)
    'corona_min_norm_slew_rate': 8.0,         # Corona: >8.0 (normalized)
    'internal_max_norm_slew_rate': 5.0,       # Internal: <5.0 (normalized)

    # spectral_power_ratio: Corona > 1.5, Internal 0.8-1.5
    'corona_min_spectral_ratio': 1.5,         # Corona: > 1.5
    'internal_min_spectral_ratio': 0.8,       # Internal: 0.8-1.5
    'internal_max_spectral_ratio': 1.5,

    # oscillation_count: Corona HIGH (>=90, more ringing), Internal LOW (<90)
    'corona_min_oscillation': 90,             # Corona: >= 90 oscillations (ringing after spike)
    'internal_max_oscillation': 90,           # Internal: < 90 oscillations

    # crest_factor: Corona higher (>=7.0), Internal moderate (4.0-6.5)
    'corona_min_crest_factor': 7.0,           # Corona: >= 7.0 (very impulsive)
    'internal_min_crest_factor': 4.0,         # Internal: 4.0-6.5
    'internal_max_crest_factor': 6.5,

    # dominant_frequency:
    # Negative Corona: > 15 MHz (high frequency)
    # Positive Corona: 5-15 MHz (moderate frequency)
    # Internal: 5-30 MHz (expanded range)
    'corona_neg_min_dominant_freq': 15e6,     # Negative Corona: >= 15 MHz
    'corona_pos_min_dominant_freq': 5e6,      # Positive Corona: 5-15 MHz
    'corona_pos_max_dominant_freq': 15e6,
    'internal_min_dominant_freq': 5e6,        # Internal: 5-30 MHz (expanded)
    'internal_max_dominant_freq': 30e6,

    # SUPPORTING FEATURES (Weight: 1)
    # coefficient_of_variation: Corona < 0.15, Internal 0.15-0.35
    'corona_max_cv': 0.15,                    # Corona: < 0.15
    'internal_min_cv': 0.15,                  # Internal: 0.15-0.35
    'internal_max_cv': 0.35,

    # quadrant_3_percentage: Negative Corona > 55%, Internal 35-50%
    'corona_neg_min_q3_pct': 55,              # Negative Corona: > 55% in Q3
    'internal_min_q3_pct': 35,                # Internal: 35-50%
    'internal_max_q3_pct': 50,

    # repetition_rate: Corona high, Internal moderate
    'corona_min_rep_rate': 100,               # Corona: High (>100 pulses/cycle)
    'internal_min_rep_rate': 20,              # Internal: Moderate (20-100)
    'internal_max_rep_rate': 100,
}

# Branch 5: Amplitude Characteristics (for fallback)
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


def apply_custom_thresholds(custom_thresholds):
    """Apply custom threshold values to the global threshold dictionaries."""
    global NOISE_THRESHOLDS, PHASE_SPREAD_THRESHOLDS, SURFACE_DETECTION_THRESHOLDS
    global CORONA_INTERNAL_THRESHOLDS, AMPLITUDE_THRESHOLDS, QUADRANT_THRESHOLDS

    # Map custom threshold names to the actual dictionary keys
    threshold_mapping = {
        # Branch 1: Noise Detection
        'min_spectral_flatness': ('NOISE_THRESHOLDS', 'min_spectral_flatness'),
        'max_bandwidth_3db': ('NOISE_THRESHOLDS', 'max_bandwidth_3db'),
        'max_dominant_frequency': ('NOISE_THRESHOLDS', 'max_dominant_frequency'),
        'min_slew_rate': ('NOISE_THRESHOLDS', 'min_slew_rate'),
        'min_crest_factor': ('NOISE_THRESHOLDS', 'min_crest_factor'),
        'max_oscillation_count': ('NOISE_THRESHOLDS', 'max_oscillation_count'),
        'min_snr': ('NOISE_THRESHOLDS', 'min_signal_to_noise_ratio'),
        'min_cross_corr_noise': ('NOISE_THRESHOLDS', 'min_cross_correlation'),
        'max_cv_noise': ('NOISE_THRESHOLDS', 'max_coefficient_of_variation'),
        'min_pulses_for_multipulse': ('NOISE_THRESHOLDS', 'min_pulses_for_multipulse'),

        # Branch 2: Phase Spread (Surface initial detection)
        'surface_phase_spread_min': ('PHASE_SPREAD_THRESHOLDS', 'surface_phase_spread_min'),

        # Branch 3: Surface PD vs Corona/Internal (Weights)
        'surface_primary_weight': ('SURFACE_DETECTION_THRESHOLDS', 'primary_weight'),
        'surface_secondary_weight': ('SURFACE_DETECTION_THRESHOLDS', 'secondary_weight'),
        'surface_mid_weight': ('SURFACE_DETECTION_THRESHOLDS', 'mid_weight'),
        'surface_supporting_weight': ('SURFACE_DETECTION_THRESHOLDS', 'supporting_weight'),
        'min_surface_score': ('SURFACE_DETECTION_THRESHOLDS', 'min_surface_score'),
        # Branch 3: Surface PD vs Corona/Internal (Thresholds)
        'surface_phase_spread': ('SURFACE_DETECTION_THRESHOLDS', 'surface_phase_spread'),
        'corona_phase_spread': ('SURFACE_DETECTION_THRESHOLDS', 'corona_phase_spread'),
        'surface_max_slew_rate': ('SURFACE_DETECTION_THRESHOLDS', 'surface_max_slew_rate'),
        'corona_min_slew_rate': ('SURFACE_DETECTION_THRESHOLDS', 'corona_min_slew_rate'),
        'surface_max_spectral_power_ratio': ('SURFACE_DETECTION_THRESHOLDS', 'surface_max_spectral_power_ratio'),
        'corona_min_spectral_power_ratio': ('SURFACE_DETECTION_THRESHOLDS', 'corona_min_spectral_power_ratio'),
        'surface_min_cv': ('SURFACE_DETECTION_THRESHOLDS', 'surface_min_cv'),
        'corona_max_cv': ('SURFACE_DETECTION_THRESHOLDS', 'corona_max_cv'),
        'surface_min_crest_factor': ('SURFACE_DETECTION_THRESHOLDS', 'surface_min_crest_factor'),
        'surface_max_crest_factor': ('SURFACE_DETECTION_THRESHOLDS', 'surface_max_crest_factor'),
        'corona_min_crest_factor': ('SURFACE_DETECTION_THRESHOLDS', 'corona_min_crest_factor'),
        'surface_min_cross_corr': ('SURFACE_DETECTION_THRESHOLDS', 'surface_min_cross_corr'),
        'surface_max_cross_corr': ('SURFACE_DETECTION_THRESHOLDS', 'surface_max_cross_corr'),
        'corona_min_cross_corr': ('SURFACE_DETECTION_THRESHOLDS', 'corona_min_cross_corr'),
        'surface_min_spectral_flatness': ('SURFACE_DETECTION_THRESHOLDS', 'surface_min_spectral_flatness'),
        'surface_max_spectral_flatness': ('SURFACE_DETECTION_THRESHOLDS', 'surface_max_spectral_flatness'),
        'corona_max_spectral_flatness': ('SURFACE_DETECTION_THRESHOLDS', 'corona_max_spectral_flatness'),
        'surface_min_rep_rate_var': ('SURFACE_DETECTION_THRESHOLDS', 'surface_min_rep_rate_var'),
        'corona_max_rep_rate_var': ('SURFACE_DETECTION_THRESHOLDS', 'corona_max_rep_rate_var'),

        # Branch 4: Corona vs Internal (Score-based)
        # Weights
        'primary_weight': ('CORONA_INTERNAL_THRESHOLDS', 'primary_weight'),
        'secondary_weight': ('CORONA_INTERNAL_THRESHOLDS', 'secondary_weight'),
        'supporting_weight': ('CORONA_INTERNAL_THRESHOLDS', 'supporting_weight'),
        'min_corona_score': ('CORONA_INTERNAL_THRESHOLDS', 'min_corona_score'),
        'min_internal_score': ('CORONA_INTERNAL_THRESHOLDS', 'min_internal_score'),
        # Primary: Asymmetry
        'corona_max_asymmetry': ('CORONA_INTERNAL_THRESHOLDS', 'corona_max_asymmetry'),
        'internal_min_asymmetry': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_asymmetry'),
        'internal_max_asymmetry': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_asymmetry'),
        # Primary: Phase of max activity
        'corona_phase_min': ('CORONA_INTERNAL_THRESHOLDS', 'corona_phase_min'),
        'corona_phase_max': ('CORONA_INTERNAL_THRESHOLDS', 'corona_phase_max'),
        'internal_phase_q1_min': ('CORONA_INTERNAL_THRESHOLDS', 'internal_phase_q1_min'),
        'internal_phase_q1_max': ('CORONA_INTERNAL_THRESHOLDS', 'internal_phase_q1_max'),
        'internal_phase_q3_min': ('CORONA_INTERNAL_THRESHOLDS', 'internal_phase_q3_min'),
        'internal_phase_q3_max': ('CORONA_INTERNAL_THRESHOLDS', 'internal_phase_q3_max'),
        # Secondary: Slew rate
        'ci_corona_min_slew_rate': ('CORONA_INTERNAL_THRESHOLDS', 'corona_min_slew_rate'),
        'ci_internal_min_slew_rate': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_slew_rate'),
        'ci_internal_max_slew_rate': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_slew_rate'),
        # Secondary: Spectral power ratio
        'ci_corona_min_spectral_ratio': ('CORONA_INTERNAL_THRESHOLDS', 'corona_min_spectral_ratio'),
        'ci_internal_min_spectral_ratio': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_spectral_ratio'),
        'ci_internal_max_spectral_ratio': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_spectral_ratio'),
        # Supporting: Q3 percentage
        'corona_min_q3_pct': ('CORONA_INTERNAL_THRESHOLDS', 'corona_min_q3_pct'),
        'internal_min_q3_pct': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_q3_pct'),
        'internal_max_q3_pct': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_q3_pct'),
        # Secondary: Oscillation count (Corona HIGH, Internal LOW)
        'corona_min_oscillation': ('CORONA_INTERNAL_THRESHOLDS', 'corona_min_oscillation'),
        'internal_max_oscillation': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_oscillation'),
        # Primary: Spectral power low
        'internal_min_spectral_power_low': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_spectral_power_low'),
        'corona_max_spectral_power_low': ('CORONA_INTERNAL_THRESHOLDS', 'corona_max_spectral_power_low'),
        # Secondary: Norm slew rate
        'corona_min_norm_slew_rate': ('CORONA_INTERNAL_THRESHOLDS', 'corona_min_norm_slew_rate'),
        'internal_max_norm_slew_rate': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_norm_slew_rate'),
        # Secondary: Crest factor
        'corona_min_crest_factor': ('CORONA_INTERNAL_THRESHOLDS', 'corona_min_crest_factor'),
        'internal_min_crest_factor': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_crest_factor'),
        'internal_max_crest_factor': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_crest_factor'),
        # Supporting: CV
        'ci_corona_max_cv': ('CORONA_INTERNAL_THRESHOLDS', 'corona_max_cv'),
        'ci_internal_min_cv': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_cv'),
        'ci_internal_max_cv': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_cv'),
        # Supporting: Repetition rate
        'corona_min_rep_rate': ('CORONA_INTERNAL_THRESHOLDS', 'corona_min_rep_rate'),
        'internal_min_rep_rate': ('CORONA_INTERNAL_THRESHOLDS', 'internal_min_rep_rate'),
        'internal_max_rep_rate': ('CORONA_INTERNAL_THRESHOLDS', 'internal_max_rep_rate'),

        # Branch 5: Internal/Amplitude (fallback)
        'weibull_beta_min': ('AMPLITUDE_THRESHOLDS', 'internal_weibull_beta_min'),
        'sym_quadrant_min': ('QUADRANT_THRESHOLDS', 'symmetric_quadrant_min'),
        'sym_quadrant_max': ('QUADRANT_THRESHOLDS', 'symmetric_quadrant_max'),
    }

    # Get the actual threshold dictionaries
    threshold_dicts = {
        'NOISE_THRESHOLDS': NOISE_THRESHOLDS,
        'PHASE_SPREAD_THRESHOLDS': PHASE_SPREAD_THRESHOLDS,
        'SURFACE_DETECTION_THRESHOLDS': SURFACE_DETECTION_THRESHOLDS,
        'CORONA_INTERNAL_THRESHOLDS': CORONA_INTERNAL_THRESHOLDS,
        'AMPLITUDE_THRESHOLDS': AMPLITUDE_THRESHOLDS,
        'QUADRANT_THRESHOLDS': QUADRANT_THRESHOLDS,
    }

    # Apply custom values
    for custom_name, value in custom_thresholds.items():
        if custom_name in threshold_mapping:
            dict_name, key_name = threshold_mapping[custom_name]
            threshold_dicts[dict_name][key_name] = value
            print(f"  Custom threshold: {custom_name} = {value}")


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
    'NOISE_MULTIPULSE': {
        'code': 5,
        'description': 'Multi-pulse waveform (multiple PD events in single acquisition)',
        'characteristics': [
            'Multiple distinct pulses detected in waveform',
            'Pulse count >= 2 per waveform window',
            'May indicate high PD activity or overlapping events',
            'Requires separation before individual classification',
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

        # Check for multi-pulse waveforms
        # Multi-pulse = multiple distinct PD events captured in a single waveform acquisition
        pulses_per_waveform = self._get_feature(cluster_features, 'mean_pulse_count',
                             self._get_feature(cluster_features, 'pulses_per_waveform',
                             self._get_feature(cluster_features, 'mean_pulses_per_waveform', 1)))
        is_multi_pulse = self._get_feature(cluster_features, 'mean_is_multi_pulse',
                        self._get_feature(cluster_features, 'is_multi_pulse', 0))

        min_pulses_multipulse = NOISE_THRESHOLDS['min_pulses_for_multipulse']
        # Check if majority of waveforms are multi-pulse (mean > 0.5) or pulse count >= threshold
        if is_multi_pulse > 0.5 or pulses_per_waveform >= min_pulses_multipulse:
            result['pd_type'] = 'NOISE_MULTIPULSE'
            result['confidence'] = 0.90
            result['reasoning'].append(
                f"Multi-pulse waveform detected: {pulses_per_waveform:.1f} pulses/waveform, "
                f"{is_multi_pulse*100:.0f}% multi-pulse (threshold: {min_pulses_multipulse} pulses or >50% multi-pulse)"
            )
            return result

        # Extract noise-related features (using mean values for cluster)
        spectral_flatness = self._get_feature(cluster_features, 'mean_spectral_flatness',
                           self._get_feature(cluster_features, 'spectral_flatness', 0))
        slew_rate = self._get_feature(cluster_features, 'mean_slew_rate',
                   self._get_feature(cluster_features, 'slew_rate', 1e9))
        crest_factor = self._get_feature(cluster_features, 'mean_crest_factor',
                      self._get_feature(cluster_features, 'crest_factor', 10))
        cross_corr_noise = self._get_feature(cluster_features, 'mean_cross_correlation',
                          self._get_feature(cluster_features, 'cross_correlation', 0.5))
        oscillation_count = self._get_feature(cluster_features, 'mean_oscillation_count',
                           self._get_feature(cluster_features, 'oscillation_count', 5))
        snr = self._get_feature(cluster_features, 'mean_signal_to_noise_ratio',
             self._get_feature(cluster_features, 'signal_to_noise_ratio', 10))
        cv = self._get_feature(cluster_features, 'coefficient_of_variation', 0)
        bandwidth = self._get_feature(cluster_features, 'mean_bandwidth_3db',
                   self._get_feature(cluster_features, 'bandwidth_3db', 1e9))
        dominant_freq = self._get_feature(cluster_features, 'mean_dominant_frequency',
                       self._get_feature(cluster_features, 'dominant_frequency', 1e6))

        # Score-based noise detection (higher score = more likely noise)
        noise_score = 0.0
        noise_reasons = []

        # 1. Spectral flatness: high flatness = random noise (white noise ~1.0)
        if spectral_flatness > NOISE_THRESHOLDS['min_spectral_flatness']:
            noise_score += 0.15
            noise_reasons.append(f"high_spectral_flatness={spectral_flatness:.2f}")

        # 2. Slew rate: real PD has fast rise times
        if slew_rate < NOISE_THRESHOLDS['min_slew_rate']:
            noise_score += 0.15
            noise_reasons.append(f"slow_slew_rate={slew_rate:.2e}")

        # 3. Crest factor: real PD is impulsive (high crest factor)
        if crest_factor < NOISE_THRESHOLDS['min_crest_factor']:
            noise_score += 0.15
            noise_reasons.append(f"low_crest_factor={crest_factor:.2f}")

        # 4. Cross-correlation: consistent shape across pulses
        if cross_corr_noise < NOISE_THRESHOLDS['min_cross_correlation']:
            noise_score += 0.10
            noise_reasons.append(f"low_cross_corr={cross_corr_noise:.2f}")

        # 5. Oscillation count: excessive ringing = EMI
        if oscillation_count > NOISE_THRESHOLDS['max_oscillation_count']:
            noise_score += 0.10
            noise_reasons.append(f"excessive_oscillations={oscillation_count}")

        # 6. Signal-to-noise ratio: poor signal quality
        if snr < NOISE_THRESHOLDS['min_signal_to_noise_ratio']:
            noise_score += 0.15
            noise_reasons.append(f"low_snr={snr:.2f}")

        # 7. Coefficient of variation: random amplitude
        if cv > NOISE_THRESHOLDS['max_coefficient_of_variation']:
            noise_score += 0.10
            noise_reasons.append(f"high_cv={cv:.2f}")

        # 8. Bandwidth: narrowband EMI has very low bandwidth
        if bandwidth < NOISE_THRESHOLDS['max_bandwidth_3db']:
            noise_score += 0.05
            noise_reasons.append(f"narrowband={bandwidth:.2e}Hz")

        # 9. Dominant frequency: 60Hz hum or low-freq interference
        if dominant_freq < NOISE_THRESHOLDS['max_dominant_frequency']:
            noise_score += 0.10
            noise_reasons.append(f"low_freq={dominant_freq:.0f}Hz")

        # Classify as noise if score exceeds threshold
        if noise_score >= 0.45:
            result['pd_type'] = 'NOISE'
            result['confidence'] = min(0.5 + noise_score, 0.95)
            result['reasoning'].append(
                f"Noise indicators: {', '.join(noise_reasons)}"
            )
            return result

        # Add warning if some noise indicators present
        if noise_score > 0.2:
            result['warnings'].append(
                f"Partial noise indicators (score={noise_score:.2f}): {', '.join(noise_reasons)}"
            )

        result['reasoning'].append(f"Passed noise detection (score={noise_score:.2f}, n={n_pulses})")

        # =====================================================================
        # BRANCH 2: PHASE SPREAD CHECK (Surface PD Initial Detection)
        # =====================================================================
        result['branch_path'].append('Branch 2: Phase Spread')

        phase_spread = self._get_feature(cluster_features, 'phase_spread', 0)

        result['reasoning'].append(f"Phase spread: {phase_spread:.1f}deg")

        # If phase spread > 120° → immediate Surface PD classification
        if phase_spread > PHASE_SPREAD_THRESHOLDS['surface_phase_spread_min']:
            result['pd_type'] = 'SURFACE'
            result['confidence'] = 0.85
            result['reasoning'].append(
                f"Classified as SURFACE: phase_spread={phase_spread:.1f}° > {PHASE_SPREAD_THRESHOLDS['surface_phase_spread_min']}°"
            )
            return result

        # =====================================================================
        # BRANCH 3: SURFACE PD vs CORONA/INTERNAL (8-feature weighted scoring)
        # =====================================================================
        result['branch_path'].append('Branch 3: Surface Detection')

        # Extract features for Surface vs Corona/Internal comparison
        slew_rate = self._get_feature(cluster_features, 'mean_slew_rate',
                   self._get_feature(cluster_features, 'slew_rate', 1e7))
        spectral_power_ratio = self._get_feature(cluster_features, 'spectral_power_ratio',
                              self._get_feature(cluster_features, 'mean_spectral_power_ratio', 0.5))
        cv = self._get_feature(cluster_features, 'coefficient_of_variation', 0.3)
        crest_factor = self._get_feature(cluster_features, 'mean_crest_factor',
                      self._get_feature(cluster_features, 'crest_factor', 5))
        cross_corr = self._get_feature(cluster_features, 'cross_correlation', 0.5)
        spectral_flatness = self._get_feature(cluster_features, 'mean_spectral_flatness',
                           self._get_feature(cluster_features, 'spectral_flatness', 0.4))
        rep_rate_var = self._get_feature(cluster_features, 'repetition_rate_variance',
                      self._get_feature(cluster_features, 'rep_rate_variance', 0.4))

        # Get weights
        primary_weight = int(SURFACE_DETECTION_THRESHOLDS['primary_weight'])
        secondary_weight = int(SURFACE_DETECTION_THRESHOLDS['secondary_weight'])
        mid_weight = int(SURFACE_DETECTION_THRESHOLDS['mid_weight'])
        supporting_weight = int(SURFACE_DETECTION_THRESHOLDS['supporting_weight'])

        # Score Surface PD indicators with weights
        surface_score = 0
        surface_indicators = []

        # PRIMARY FEATURE (Weight: 4)
        # Feature 1: Phase spread (>120° already handled, 100-120° is borderline)
        if phase_spread > SURFACE_DETECTION_THRESHOLDS['surface_phase_spread']:
            surface_score += primary_weight
            surface_indicators.append(f"phase_spread={phase_spread:.1f}°>120° [+{primary_weight}]")
        elif phase_spread < SURFACE_DETECTION_THRESHOLDS['corona_phase_spread']:
            surface_indicators.append(f"phase_spread={phase_spread:.1f}°<100° (Corona/Internal)")

        # SECONDARY FEATURES (Weight: 3)
        # Feature 2: Slew rate - Surface has low slew rate
        if slew_rate < SURFACE_DETECTION_THRESHOLDS['surface_max_slew_rate']:
            surface_score += secondary_weight
            surface_indicators.append(f"low_slew_rate={slew_rate:.2e} [+{secondary_weight}]")

        # Feature 3: Spectral power ratio - Surface has lower ratio
        if spectral_power_ratio < SURFACE_DETECTION_THRESHOLDS['surface_max_spectral_power_ratio']:
            surface_score += secondary_weight
            surface_indicators.append(f"low_spectral_ratio={spectral_power_ratio:.2f} [+{secondary_weight}]")

        # Feature 4: Coefficient of variation - Surface has higher CV
        if cv > SURFACE_DETECTION_THRESHOLDS['surface_min_cv']:
            surface_score += secondary_weight
            surface_indicators.append(f"high_cv={cv:.2f} [+{secondary_weight}]")

        # MID-WEIGHT FEATURES (Weight: 2)
        # Feature 5: Crest factor - Surface has moderate crest factor (4-6)
        if (SURFACE_DETECTION_THRESHOLDS['surface_min_crest_factor'] <= crest_factor <=
            SURFACE_DETECTION_THRESHOLDS['surface_max_crest_factor']):
            surface_score += mid_weight
            surface_indicators.append(f"moderate_crest={crest_factor:.1f} [+{mid_weight}]")

        # Feature 6: Cross-correlation - Surface has lower correlation (0.4-0.6)
        if (SURFACE_DETECTION_THRESHOLDS['surface_min_cross_corr'] <= cross_corr <=
            SURFACE_DETECTION_THRESHOLDS['surface_max_cross_corr']):
            surface_score += mid_weight
            surface_indicators.append(f"lower_corr={cross_corr:.2f} [+{mid_weight}]")

        # SUPPORTING FEATURES (Weight: 1)
        # Feature 7: Spectral flatness - Surface has higher flatness (0.4-0.5)
        if (SURFACE_DETECTION_THRESHOLDS['surface_min_spectral_flatness'] <= spectral_flatness <=
            SURFACE_DETECTION_THRESHOLDS['surface_max_spectral_flatness']):
            surface_score += supporting_weight
            surface_indicators.append(f"higher_flatness={spectral_flatness:.2f} [+{supporting_weight}]")

        # Feature 8: Repetition rate variance - Surface has high variance
        if rep_rate_var > SURFACE_DETECTION_THRESHOLDS['surface_min_rep_rate_var']:
            surface_score += supporting_weight
            surface_indicators.append(f"high_rep_var={rep_rate_var:.2f} [+{supporting_weight}]")

        # Feature 9: Dominant frequency - Surface PD is 1-5 MHz
        dominant_freq = self._get_feature(cluster_features, 'mean_dominant_frequency',
                       self._get_feature(cluster_features, 'dominant_frequency', 3e6))
        if (SURFACE_DETECTION_THRESHOLDS['surface_min_dominant_freq'] <= dominant_freq <=
            SURFACE_DETECTION_THRESHOLDS['surface_max_dominant_freq']):
            surface_score += supporting_weight
            surface_indicators.append(f"freq={dominant_freq/1e6:.1f}MHz in [1-5MHz] [+{supporting_weight}]")

        # Max possible score: 4 + 3*3 + 2*2 + 1*3 = 4 + 9 + 4 + 3 = 20
        max_score = primary_weight + 3 * secondary_weight + 2 * mid_weight + 3 * supporting_weight
        min_surface = int(SURFACE_DETECTION_THRESHOLDS['min_surface_score'])
        result['reasoning'].append(
            f"Surface score: {surface_score}/{max_score} (need {min_surface}): {', '.join(surface_indicators[:3])}..."
        )

        # Classify as Surface if score meets threshold
        if surface_score >= min_surface:
            result['pd_type'] = 'SURFACE'
            result['confidence'] = min(0.5 + (surface_score / max_score) * 0.4, 0.90)
            result['reasoning'].append(
                f"Classified as SURFACE: {surface_score}/{max_score} ({', '.join(surface_indicators)})"
            )
            return result

        # =====================================================================
        # BRANCH 4: CORONA vs INTERNAL (Score-based detection)
        # =====================================================================
        result['branch_path'].append('Branch 4: Corona vs Internal')

        # Extract all features needed for scoring
        asymmetry = self._get_feature(cluster_features, 'discharge_asymmetry', 0)
        q3 = self._get_feature(cluster_features, 'quadrant_3_percentage', 0)
        phase_max = self._get_feature(cluster_features, 'phase_of_max_activity', 0)
        ci_slew_rate = self._get_feature(cluster_features, 'mean_slew_rate',
                       self._get_feature(cluster_features, 'slew_rate', 1e7))
        ci_spectral_ratio = self._get_feature(cluster_features, 'spectral_power_ratio',
                           self._get_feature(cluster_features, 'mean_spectral_power_ratio', 1.0))
        ci_cv = self._get_feature(cluster_features, 'coefficient_of_variation', 0.2)
        rep_rate = self._get_feature(cluster_features, 'repetition_rate',
                   self._get_feature(cluster_features, 'pulses_per_cycle', 50))
        ci_oscillation = self._get_feature(cluster_features, 'mean_oscillation_count',
                         self._get_feature(cluster_features, 'oscillation_count', 5))

        # Get weights
        primary_weight = int(CORONA_INTERNAL_THRESHOLDS['primary_weight'])
        secondary_weight = int(CORONA_INTERNAL_THRESHOLDS['secondary_weight'])
        supporting_weight = int(CORONA_INTERNAL_THRESHOLDS['supporting_weight'])

        # Score Corona and Internal separately
        corona_score = 0
        internal_score = 0
        corona_indicators = []
        internal_indicators = []

        # PRIMARY FEATURES (Weight: 4)
        # 1. discharge_asymmetry - Negative Corona (<-0.6), Positive Corona (>+0.6), Internal wide range
        is_negative_corona = False
        is_positive_corona = False
        if asymmetry < CORONA_INTERNAL_THRESHOLDS['corona_neg_max_asymmetry']:
            corona_score += primary_weight
            is_negative_corona = True
            corona_indicators.append(f"asymmetry={asymmetry:.2f}<-0.6 (neg corona) [+{primary_weight}]")
        elif asymmetry > CORONA_INTERNAL_THRESHOLDS['corona_pos_min_asymmetry']:
            corona_score += primary_weight
            is_positive_corona = True
            corona_indicators.append(f"asymmetry={asymmetry:.2f}>+0.6 (pos corona) [+{primary_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_asymmetry'] <= asymmetry <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_asymmetry']):
            internal_score += primary_weight
            internal_indicators.append(f"asymmetry={asymmetry:.2f} in [-0.9,0.9] [+{primary_weight}]")

        # 2. phase_of_max_activity
        # Negative Corona: 180-270°, Positive Corona: 0-90° or 270-360°, Internal: 45-90° or 225-270°
        if (CORONA_INTERNAL_THRESHOLDS['corona_neg_phase_min'] <= phase_max <=
            CORONA_INTERNAL_THRESHOLDS['corona_neg_phase_max']):
            corona_score += primary_weight
            corona_indicators.append(f"phase_max={phase_max:.0f}° in [180-270] (neg corona) [+{primary_weight}]")
        elif ((CORONA_INTERNAL_THRESHOLDS['corona_pos_phase_q1_min'] <= phase_max <=
               CORONA_INTERNAL_THRESHOLDS['corona_pos_phase_q1_max']) or
              (CORONA_INTERNAL_THRESHOLDS['corona_pos_phase_q4_min'] <= phase_max <=
               CORONA_INTERNAL_THRESHOLDS['corona_pos_phase_q4_max'])):
            corona_score += primary_weight
            corona_indicators.append(f"phase_max={phase_max:.0f}° in [0-90,270-360] (pos corona) [+{primary_weight}]")
        if ((CORONA_INTERNAL_THRESHOLDS['internal_phase_q1_min'] <= phase_max <=
             CORONA_INTERNAL_THRESHOLDS['internal_phase_q1_max']) or
            (CORONA_INTERNAL_THRESHOLDS['internal_phase_q3_min'] <= phase_max <=
             CORONA_INTERNAL_THRESHOLDS['internal_phase_q3_max'])):
            internal_score += primary_weight
            internal_indicators.append(f"phase_max={phase_max:.0f}° at peaks [+{primary_weight}]")

        # SECONDARY FEATURES (Weight: 2)
        # 3. slew_rate
        if ci_slew_rate > CORONA_INTERNAL_THRESHOLDS['corona_min_slew_rate']:
            corona_score += secondary_weight
            corona_indicators.append(f"slew={ci_slew_rate:.1e}>50M [+{secondary_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_slew_rate'] <= ci_slew_rate <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_slew_rate']):
            internal_score += secondary_weight
            internal_indicators.append(f"slew={ci_slew_rate:.1e} in [10M,50M] [+{secondary_weight}]")

        # 4. spectral_power_ratio
        if ci_spectral_ratio > CORONA_INTERNAL_THRESHOLDS['corona_min_spectral_ratio']:
            corona_score += secondary_weight
            corona_indicators.append(f"spectral_ratio={ci_spectral_ratio:.2f}>1.5 [+{secondary_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_spectral_ratio'] <= ci_spectral_ratio <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_spectral_ratio']):
            internal_score += secondary_weight
            internal_indicators.append(f"spectral_ratio={ci_spectral_ratio:.2f} in [0.8,1.5] [+{secondary_weight}]")

        # 5. oscillation_count: Corona has MORE oscillations (ringing), Internal has fewer
        if ci_oscillation >= CORONA_INTERNAL_THRESHOLDS['corona_min_oscillation']:
            corona_score += secondary_weight
            corona_indicators.append(f"oscillation={ci_oscillation:.0f}>=90 [+{secondary_weight}]")
        if ci_oscillation < CORONA_INTERNAL_THRESHOLDS['internal_max_oscillation']:
            internal_score += secondary_weight
            internal_indicators.append(f"oscillation={ci_oscillation:.0f}<90 [+{secondary_weight}]")

        # SUPPORTING FEATURES (Weight: 1)
        # 6. coefficient_of_variation
        if ci_cv < CORONA_INTERNAL_THRESHOLDS['corona_max_cv']:
            corona_score += supporting_weight
            corona_indicators.append(f"cv={ci_cv:.2f}<0.15 [+{supporting_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_cv'] <= ci_cv <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_cv']):
            internal_score += supporting_weight
            internal_indicators.append(f"cv={ci_cv:.2f} in [0.15,0.35] [+{supporting_weight}]")

        # 7. quadrant_3_percentage (for negative corona)
        if q3 > CORONA_INTERNAL_THRESHOLDS['corona_neg_min_q3_pct']:
            corona_score += supporting_weight
            corona_indicators.append(f"q3={q3:.1f}%>55% [+{supporting_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_q3_pct'] <= q3 <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_q3_pct']):
            internal_score += supporting_weight
            internal_indicators.append(f"q3={q3:.1f}% in [35,50] [+{supporting_weight}]")

        # 8. repetition_rate
        if rep_rate > CORONA_INTERNAL_THRESHOLDS['corona_min_rep_rate']:
            corona_score += supporting_weight
            corona_indicators.append(f"rep_rate={rep_rate:.0f}>100 [+{supporting_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_rep_rate'] <= rep_rate <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_rep_rate']):
            internal_score += supporting_weight
            internal_indicators.append(f"rep_rate={rep_rate:.0f} in [20,100] [+{supporting_weight}]")

        # 9. dominant_frequency:
        # Negative Corona: >= 15 MHz, Positive Corona: 5-15 MHz, Internal: 5-15 MHz
        ci_dom_freq = self._get_feature(cluster_features, 'mean_dominant_frequency',
                      self._get_feature(cluster_features, 'dominant_frequency', 8e6))
        if ci_dom_freq >= CORONA_INTERNAL_THRESHOLDS['corona_neg_min_dominant_freq']:
            corona_score += secondary_weight
            corona_indicators.append(f"freq={ci_dom_freq/1e6:.1f}MHz>=15MHz (neg corona) [+{secondary_weight}]")
        elif (CORONA_INTERNAL_THRESHOLDS['corona_pos_min_dominant_freq'] <= ci_dom_freq <=
              CORONA_INTERNAL_THRESHOLDS['corona_pos_max_dominant_freq']):
            # Positive corona and Internal overlap in frequency - check asymmetry context
            if is_positive_corona:
                corona_score += secondary_weight
                corona_indicators.append(f"freq={ci_dom_freq/1e6:.1f}MHz in [5-15MHz] (pos corona) [+{secondary_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_dominant_freq'] <= ci_dom_freq <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_dominant_freq']):
            internal_score += secondary_weight
            internal_indicators.append(f"freq={ci_dom_freq/1e6:.1f}MHz in [5-30MHz] [+{secondary_weight}]")

        # 10. amplitude_phase_correlation: PRIMARY FEATURE - Internal high (>0.5), Corona low (<0.3)
        # This is a strong discriminator - high correlation means amplitude tracks sinusoidal reference
        amp_phase_corr = self._get_feature(cluster_features, 'mean_amplitude_phase_correlation',
                         self._get_feature(cluster_features, 'amplitude_phase_correlation', 0.0))
        if amp_phase_corr >= CORONA_INTERNAL_THRESHOLDS['internal_min_amp_phase_corr']:
            internal_score += primary_weight
            internal_indicators.append(f"amp_phase_corr={amp_phase_corr:.2f}>=0.5 [+{primary_weight}]")
        if amp_phase_corr <= CORONA_INTERNAL_THRESHOLDS['corona_max_amp_phase_corr']:
            corona_score += primary_weight
            corona_indicators.append(f"amp_phase_corr={amp_phase_corr:.2f}<=0.3 [+{primary_weight}]")

        # 11. spectral_power_low: PRIMARY FEATURE - Internal high (>0.85), Corona low (<0.60)
        # This is a very strong discriminator - Internal has most power in low freq, Corona has high freq content
        spectral_power_low = self._get_feature(cluster_features, 'mean_spectral_power_low',
                             self._get_feature(cluster_features, 'spectral_power_low', 0.5))
        if spectral_power_low >= CORONA_INTERNAL_THRESHOLDS['internal_min_spectral_power_low']:
            internal_score += primary_weight
            internal_indicators.append(f"spectral_power_low={spectral_power_low:.2f}>=0.85 [+{primary_weight}]")
        if spectral_power_low <= CORONA_INTERNAL_THRESHOLDS['corona_max_spectral_power_low']:
            corona_score += primary_weight
            corona_indicators.append(f"spectral_power_low={spectral_power_low:.2f}<=0.60 [+{primary_weight}]")

        # 12. norm_slew_rate: SECONDARY FEATURE - Corona very high (>8), Internal low (<5)
        norm_slew_rate = self._get_feature(cluster_features, 'mean_norm_slew_rate',
                         self._get_feature(cluster_features, 'norm_slew_rate', 3.0))
        if norm_slew_rate >= CORONA_INTERNAL_THRESHOLDS['corona_min_norm_slew_rate']:
            corona_score += secondary_weight
            corona_indicators.append(f"norm_slew={norm_slew_rate:.1f}>=8.0 [+{secondary_weight}]")
        if norm_slew_rate <= CORONA_INTERNAL_THRESHOLDS['internal_max_norm_slew_rate']:
            internal_score += secondary_weight
            internal_indicators.append(f"norm_slew={norm_slew_rate:.1f}<=5.0 [+{secondary_weight}]")

        # 13. crest_factor: SECONDARY FEATURE - Corona high (>=7), Internal moderate (4-6.5)
        crest_factor = self._get_feature(cluster_features, 'mean_crest_factor',
                       self._get_feature(cluster_features, 'crest_factor', 5.0))
        if crest_factor >= CORONA_INTERNAL_THRESHOLDS['corona_min_crest_factor']:
            corona_score += secondary_weight
            corona_indicators.append(f"crest={crest_factor:.1f}>=7.0 [+{secondary_weight}]")
        if (CORONA_INTERNAL_THRESHOLDS['internal_min_crest_factor'] <= crest_factor <=
            CORONA_INTERNAL_THRESHOLDS['internal_max_crest_factor']):
            internal_score += secondary_weight
            internal_indicators.append(f"crest={crest_factor:.1f} in [4.0,6.5] [+{secondary_weight}]")

        # Max possible score: 4*4 + 6*2 + 3*1 = 16 + 12 + 3 = 31
        # (4 primary: asymmetry, phase, amp_phase_corr, spectral_power_low)
        # (6 secondary: slew, spectral_ratio, oscillation, freq, norm_slew, crest)
        # (3 supporting: cv, q3, rep_rate)
        max_score = 4 * primary_weight + 6 * secondary_weight + 3 * supporting_weight
        min_corona = int(CORONA_INTERNAL_THRESHOLDS['min_corona_score'])
        min_internal = int(CORONA_INTERNAL_THRESHOLDS['min_internal_score'])

        result['reasoning'].append(
            f"Corona score: {corona_score}/{max_score} (need {min_corona})"
        )
        result['reasoning'].append(
            f"Internal score: {internal_score}/{max_score} (need {min_internal})"
        )

        # Classify based on scores
        if corona_score >= min_corona and corona_score > internal_score:
            result['pd_type'] = 'CORONA'
            result['confidence'] = min(0.5 + (corona_score / max_score) * 0.4, 0.95)
            result['reasoning'].append(
                f"Classified as CORONA: {', '.join(corona_indicators[:4])}..."
            )
            return result

        if internal_score >= min_internal and internal_score > corona_score:
            result['pd_type'] = 'INTERNAL'
            result['confidence'] = min(0.5 + (internal_score / max_score) * 0.4, 0.95)
            result['reasoning'].append(
                f"Classified as INTERNAL: {', '.join(internal_indicators[:4])}..."
            )
            return result

        # If both scores are above threshold but equal, use additional heuristics
        if corona_score >= min_corona and internal_score >= min_internal:
            # Tie-breaker: use asymmetry
            if asymmetry < -0.2:
                result['pd_type'] = 'CORONA'
                result['confidence'] = 0.60
                result['reasoning'].append(
                    f"Tie-breaker CORONA: asymmetry={asymmetry:.2f} (scores: C={corona_score}, I={internal_score})"
                )
                return result
            else:
                result['pd_type'] = 'INTERNAL'
                result['confidence'] = 0.60
                result['reasoning'].append(
                    f"Tie-breaker INTERNAL: asymmetry={asymmetry:.2f} (scores: C={corona_score}, I={internal_score})"
                )
                return result

        # ----- FALLBACK: Low confidence classification -----
        # If scores are below threshold but one is higher
        if corona_score > internal_score and corona_score >= min_corona / 2:
            result['pd_type'] = 'CORONA'
            result['confidence'] = 0.45
            result['reasoning'].append(
                f"Weak CORONA: score={corona_score}/{max_score} (below threshold but higher than internal)"
            )
            return result

        if internal_score > corona_score and internal_score >= min_internal / 2:
            result['pd_type'] = 'INTERNAL'
            result['confidence'] = 0.45
            result['reasoning'].append(
                f"Weak INTERNAL: score={internal_score}/{max_score} (below threshold but higher than corona)"
            )
            return result

        # Unknown pattern
        result['pd_type'] = 'UNKNOWN'
        result['confidence'] = 0.3
        result['reasoning'].append(
            f"Pattern unclear - Corona score: {corona_score}, Internal score: {internal_score}"
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
    parser.add_argument(
        '--thresholds',
        type=str,
        default=None,
        help='Custom threshold values as key=value pairs (e.g., min_pulse_count=10,max_cv=2.0)'
    )
    args = parser.parse_args()

    # Parse cluster features list if provided
    selected_features = None
    if args.cluster_features:
        selected_features = [f.strip() for f in args.cluster_features.split(',') if f.strip()]

    # Parse custom thresholds if provided
    custom_thresholds = None
    if args.thresholds:
        custom_thresholds = {}
        for pair in args.thresholds.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                try:
                    custom_thresholds[key.strip()] = float(value.strip())
                except ValueError:
                    pass
        if custom_thresholds:
            # Apply custom thresholds to global constants
            apply_custom_thresholds(custom_thresholds)

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
    print("  Branch 1: Noise Detection (9 features)")
    print("    |- DBSCAN noise label (-1)?")
    print("    \\- Score-based: spectral_flatness, slew_rate, crest_factor, etc.")
    print("  Branch 2: Phase Spread Check")
    print("    \\- phase_spread > 120deg? -> SURFACE PD")
    print("  Branch 3: Surface Detection (10 features)")
    print("    |- phase_spread, slew_rate, spectral_power_ratio, cv")
    print("    |- dominant_freq, crest_factor, cross_corr, spectral_flatness")
    print("    \\- bandwidth, repetition_rate_variance -> score >= N? SURFACE")
    print("  Branch 4: Corona vs Internal")
    print("    |- asymmetry > 0.4? (corona)")
    print("    \\- cross_corr > 0.7 && |asymmetry| < 0.35? (internal)")
    print("  Branch 5: Phase Location")
    print("    \\- Near voltage peak? (corona/internal)")
    print("  Branch 6: Amplitude Analysis")
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
