"""
Clustering feature definitions and constants.

This module defines the feature names and groups for cluster-level analysis.
"""

from typing import List, Dict

# Waveform-level feature names (aggregated per cluster)
WAVEFORM_FEATURE_NAMES: List[str] = [
    'phase_angle',
    'peak_amplitude_positive',
    'peak_amplitude_negative',
    'absolute_amplitude',
    'polarity',
    'rise_time',
    'fall_time',
    'pulse_width',
    'slew_rate',
    'energy',
    'charge',
    'equivalent_time',
    'equivalent_bandwidth',
    'cumulative_energy_peak',
    'cumulative_energy_rise_time',
    'cumulative_energy_shape_factor',
    'cumulative_energy_area_ratio',
    'dominant_frequency',
    'center_frequency',
    'bandwidth_3db',
    'spectral_power_low',
    'spectral_power_high',
    'spectral_flatness',
    'spectral_entropy',
    'peak_to_peak_amplitude',
    'rms_amplitude',
    'crest_factor',
    'rise_fall_ratio',
    'zero_crossing_count',
    'oscillation_count',
    'energy_charge_ratio',
    'signal_to_noise_ratio',
    'pulse_count',
    'is_multi_pulse',
    'norm_absolute_amplitude',
    'norm_peak_amplitude_positive',
    'norm_peak_amplitude_negative',
    'norm_peak_to_peak_amplitude',
    'norm_rms_amplitude',
    'norm_slew_rate',
    'norm_energy',
    'norm_charge',
    'norm_rise_time',
    'norm_fall_time',
    'norm_equivalent_time',
    'norm_equivalent_bandwidth',
    'norm_cumulative_energy_rise_time',
    'norm_pulse_width',
    'norm_dominant_frequency',
    'norm_center_frequency',
    'norm_bandwidth_3db',
    'norm_zero_crossing_rate',
    'norm_oscillation_rate',
]

# PRPD cluster feature names (aggregated from phase-amplitude patterns)
CLUSTER_FEATURE_NAMES: List[str] = [
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
    'amplitude_phase_correlation',
]

# Generate mean and trimmed mean feature names
WAVEFORM_MEAN_FEATURE_NAMES: List[str] = [f'mean_{feat}' for feat in WAVEFORM_FEATURE_NAMES]
WAVEFORM_TRIMMED_MEAN_FEATURE_NAMES: List[str] = [f'trimmed_mean_{feat}' for feat in WAVEFORM_FEATURE_NAMES]

# All cluster feature names combined
ALL_CLUSTER_FEATURE_NAMES: List[str] = (
    CLUSTER_FEATURE_NAMES +
    WAVEFORM_MEAN_FEATURE_NAMES +
    WAVEFORM_TRIMMED_MEAN_FEATURE_NAMES
)

# Cluster feature groups for organization
CLUSTER_FEATURE_GROUPS: Dict[str, List[str]] = {
    'pulse_count': [
        'pulses_per_positive_halfcycle',
        'pulses_per_negative_halfcycle',
        'pulses_per_cycle',
    ],
    'distribution': [
        'skewness_Hn_positive',
        'skewness_Hn_negative',
        'kurtosis_Hn_positive',
        'kurtosis_Hn_negative',
        'skewness_Hqn_positive',
        'skewness_Hqn_negative',
        'kurtosis_Hqn_positive',
        'kurtosis_Hqn_negative',
    ],
    'amplitude': [
        'mean_amplitude_positive',
        'mean_amplitude_negative',
        'max_amplitude_positive',
        'max_amplitude_negative',
        'variance_amplitude_positive',
        'variance_amplitude_negative',
        'coefficient_of_variation',
    ],
    'phase': [
        'phase_of_max_activity',
        'phase_spread',
        'inception_phase',
        'extinction_phase',
        'quadrant_1_percentage',
        'quadrant_2_percentage',
        'quadrant_3_percentage',
        'quadrant_4_percentage',
    ],
    'correlation': [
        'cross_correlation',
        'discharge_asymmetry',
        'amplitude_phase_correlation',
    ],
    'statistical': [
        'weibull_alpha',
        'weibull_beta',
        'repetition_rate',
        'number_of_peaks_Hn_positive',
        'number_of_peaks_Hn_negative',
    ],
}

# Default features used for classification
DEFAULT_CLASSIFICATION_FEATURES: List[str] = [
    'discharge_asymmetry',
    'cross_correlation',
    'phase_spread',
    'phase_of_max_activity',
    'coefficient_of_variation',
    'weibull_beta',
    'mean_spectral_flatness',
    'mean_slew_rate',
]

# Supported clustering methods
CLUSTERING_METHODS: List[str] = ['hdbscan', 'dbscan', 'kmeans']
DEFAULT_CLUSTERING_METHOD: str = 'hdbscan'
