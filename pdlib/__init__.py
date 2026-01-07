"""
pdlib - Partial Discharge Analysis Library

A reusable library for PD (Partial Discharge) waveform analysis, clustering,
and classification. This library can be used independently of the tryingDHI
middleware and GUI components.

Core modules:
- features: Waveform feature extraction and polarity calculation
- clustering: Clustering algorithms (DBSCAN, HDBSCAN, K-means) and cluster features
- classification: PD type classification using decision tree
- utils: Mathematical and signal processing utilities

Example usage:
    from pdlib import PDFeatureExtractor
    from pdlib.features import calculate_polarity, POLARITY_METHODS

    # Extract features from waveforms
    extractor = PDFeatureExtractor(sample_interval=4e-9)
    features = extractor.extract_all(waveforms, phase_angles)

    # Or extract from a single waveform
    single_features = extractor.extract_features(waveform, phase_angle=45.0)
"""

__version__ = "0.1.0"

# Feature extraction
from pdlib.features import (
    PDFeatureExtractor,
    calculate_polarity,
    POLARITY_METHODS,
    DEFAULT_POLARITY_METHOD,
    FEATURE_NAMES,
    FEATURE_GROUPS,
)

# Clustering
from pdlib.clustering import (
    cluster_pulses,
    run_hdbscan,
    run_dbscan,
    run_kmeans,
    compute_cluster_features,
    HDBSCAN_AVAILABLE,
    CLUSTERING_METHODS,
    DEFAULT_CLUSTERING_METHOD,
)

# Classification
from pdlib.classification import (
    PDTypeClassifier,
    PD_TYPES,
    PD_TYPE_CODES,
    get_pd_type_info,
    get_pd_type_code,
)

__all__ = [
    # Feature extraction
    'PDFeatureExtractor',
    'calculate_polarity',
    'POLARITY_METHODS',
    'DEFAULT_POLARITY_METHOD',
    'FEATURE_NAMES',
    'FEATURE_GROUPS',
    # Clustering
    'cluster_pulses',
    'run_hdbscan',
    'run_dbscan',
    'run_kmeans',
    'compute_cluster_features',
    'HDBSCAN_AVAILABLE',
    'CLUSTERING_METHODS',
    'DEFAULT_CLUSTERING_METHOD',
    # Classification
    'PDTypeClassifier',
    'PD_TYPES',
    'PD_TYPE_CODES',
    'get_pd_type_info',
    'get_pd_type_code',
]
