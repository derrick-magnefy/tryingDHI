"""
Feature Definitions and Constants

This module contains all feature name definitions, groupings, and related constants
for PD waveform analysis.
"""

# ADC configuration for noise floor calculation
# 12-bit ADC with -2V to +2V range
ADC_BITS = 12
ADC_RANGE_V = 4.0  # -2V to +2V = 4V total range
ADC_STEP_V = ADC_RANGE_V / (2 ** ADC_BITS)  # ~0.977 mV per step

# Complete list of pulse-level feature names in order
FEATURE_NAMES = [
    # Amplitude features
    'phase_angle',
    'peak_amplitude_positive',
    'peak_amplitude_negative',
    'absolute_amplitude',
    'polarity',
    # Timing features
    'rise_time',
    'fall_time',
    'pulse_width',
    'slew_rate',
    # Energy features
    'energy',
    'charge',
    'equivalent_time',
    'equivalent_bandwidth',
    'cumulative_energy_peak',
    'cumulative_energy_rise_time',
    'cumulative_energy_shape_factor',
    'cumulative_energy_area_ratio',
    # Spectral features
    'dominant_frequency',
    'center_frequency',
    'bandwidth_3db',
    'spectral_power_low',
    'spectral_power_high',
    'spectral_flatness',
    'spectral_entropy',
    # Amplitude statistics
    'peak_to_peak_amplitude',
    'rms_amplitude',
    'crest_factor',
    'rise_fall_ratio',
    'zero_crossing_count',
    'oscillation_count',
    'energy_charge_ratio',
    'signal_to_noise_ratio',
    # Multi-pulse features
    'pulse_count',
    'is_multi_pulse',
    # Wavelet features (DWT decomposition)
    'wavelet_energy_approx',
    'wavelet_energy_d1',
    'wavelet_energy_d2',
    'wavelet_energy_d3',
    'wavelet_energy_d4',
    'wavelet_energy_d5',
    'wavelet_rel_energy_approx',
    'wavelet_rel_energy_d1',
    'wavelet_rel_energy_d2',
    'wavelet_rel_energy_d3',
    'wavelet_rel_energy_d4',
    'wavelet_rel_energy_d5',
    'wavelet_detail_approx_ratio',
    'wavelet_dominant_level',
    'wavelet_entropy',
    'wavelet_approx_mean',
    'wavelet_approx_std',
    'wavelet_approx_max',
    'wavelet_d1_mean',
    'wavelet_d1_std',
    'wavelet_d1_max',
    # Normalized features (scale-independent)
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

# Feature groupings for organization
FEATURE_GROUPS = {
    'amplitude': [
        'phase_angle',
        'peak_amplitude_positive',
        'peak_amplitude_negative',
        'absolute_amplitude',
        'polarity',
        'peak_to_peak_amplitude',
        'rms_amplitude',
        'crest_factor',
    ],
    'timing': [
        'rise_time',
        'fall_time',
        'pulse_width',
        'slew_rate',
        'rise_fall_ratio',
        'zero_crossing_count',
        'oscillation_count',
    ],
    'energy': [
        'energy',
        'charge',
        'equivalent_time',
        'equivalent_bandwidth',
        'cumulative_energy_peak',
        'cumulative_energy_rise_time',
        'cumulative_energy_shape_factor',
        'cumulative_energy_area_ratio',
        'energy_charge_ratio',
    ],
    'spectral': [
        'dominant_frequency',
        'center_frequency',
        'bandwidth_3db',
        'spectral_power_low',
        'spectral_power_high',
        'spectral_flatness',
        'spectral_entropy',
    ],
    'multi_pulse': [
        'pulse_count',
        'is_multi_pulse',
    ],
    'wavelet': [
        'wavelet_energy_approx',
        'wavelet_energy_d1',
        'wavelet_energy_d2',
        'wavelet_energy_d3',
        'wavelet_energy_d4',
        'wavelet_energy_d5',
        'wavelet_rel_energy_approx',
        'wavelet_rel_energy_d1',
        'wavelet_rel_energy_d2',
        'wavelet_rel_energy_d3',
        'wavelet_rel_energy_d4',
        'wavelet_rel_energy_d5',
        'wavelet_detail_approx_ratio',
        'wavelet_dominant_level',
        'wavelet_entropy',
        'wavelet_approx_mean',
        'wavelet_approx_std',
        'wavelet_approx_max',
        'wavelet_d1_mean',
        'wavelet_d1_std',
        'wavelet_d1_max',
    ],
    'normalized': [
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
    ],
}

# Normalized feature names (useful for clustering, scale-independent)
NORMALIZED_FEATURES = FEATURE_GROUPS['normalized']

# Features that require normalization context (noise floor, duration, etc.)
CONTEXT_DEPENDENT_FEATURES = [
    'signal_to_noise_ratio',
] + NORMALIZED_FEATURES

# Default features recommended for clustering
DEFAULT_CLUSTERING_FEATURES = [
    'phase_angle',
    'absolute_amplitude',
    'rise_time',
    'fall_time',
    'energy',
    'dominant_frequency',
    'spectral_power_low',
    'spectral_power_high',
    'crest_factor',
    'oscillation_count',
    'norm_slew_rate',
    'norm_energy',
]
