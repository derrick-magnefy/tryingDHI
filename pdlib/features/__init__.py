"""
pdlib.features - Waveform Feature Extraction

This module provides tools for extracting features from PD waveforms:
- PDFeatureExtractor: Main class for extracting pulse-level features
- Polarity calculation methods
- Feature definitions and groupings

Usage:
    from pdlib.features import PDFeatureExtractor

    extractor = PDFeatureExtractor(sample_interval=4e-9)
    features = extractor.extract_features(waveform, phase_angle=45.0)
"""

from .extractor import PDFeatureExtractor
from .polarity import (
    calculate_polarity,
    POLARITY_METHODS,
    DEFAULT_POLARITY_METHOD,
    get_method_description,
    compare_methods,
)
from .pulse_detection import detect_pulses, is_multi_pulse
from .definitions import (
    FEATURE_NAMES,
    FEATURE_GROUPS,
    NORMALIZED_FEATURES,
    DEFAULT_CLUSTERING_FEATURES,
    ADC_BITS,
    ADC_RANGE_V,
    ADC_STEP_V,
)

__all__ = [
    # Main extractor
    'PDFeatureExtractor',
    # Polarity
    'calculate_polarity',
    'POLARITY_METHODS',
    'DEFAULT_POLARITY_METHOD',
    'get_method_description',
    'compare_methods',
    # Pulse detection
    'detect_pulses',
    'is_multi_pulse',
    # Definitions
    'FEATURE_NAMES',
    'FEATURE_GROUPS',
    'NORMALIZED_FEATURES',
    'DEFAULT_CLUSTERING_FEATURES',
    'ADC_BITS',
    'ADC_RANGE_V',
    'ADC_STEP_V',
]
