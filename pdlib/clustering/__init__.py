"""
pdlib.clustering - Clustering Algorithms and Cluster Features

This module provides:
- Clustering algorithms: DBSCAN, HDBSCAN, K-means
- Cluster-level feature aggregation (PRPD features, mean/trimmed-mean)
- Cluster feature definitions

Usage:
    from pdlib.clustering import cluster_pulses, run_hdbscan
    from pdlib.clustering import compute_cluster_features

    # Cluster pulses using HDBSCAN (default)
    labels, info = cluster_pulses(features, feature_names)

    # Compute cluster-level features
    cluster_features = compute_cluster_features(features, feature_names, labels)
"""

from .algorithms import (
    run_dbscan,
    run_hdbscan,
    run_kmeans,
    estimate_dbscan_eps,
    scale_features,
    cluster_pulses,
    HDBSCAN_AVAILABLE,
)

from .cluster_features import (
    compute_prpd_features,
    compute_waveform_aggregates,
    compute_cluster_features,
    trimmed_mean,
    fit_weibull,
)

from .definitions import (
    WAVEFORM_FEATURE_NAMES,
    CLUSTER_FEATURE_NAMES,
    WAVEFORM_MEAN_FEATURE_NAMES,
    WAVEFORM_TRIMMED_MEAN_FEATURE_NAMES,
    ALL_CLUSTER_FEATURE_NAMES,
    CLUSTER_FEATURE_GROUPS,
    DEFAULT_CLASSIFICATION_FEATURES,
    CLUSTERING_METHODS,
    DEFAULT_CLUSTERING_METHOD,
)

__all__ = [
    # Algorithms
    'run_dbscan',
    'run_hdbscan',
    'run_kmeans',
    'estimate_dbscan_eps',
    'scale_features',
    'cluster_pulses',
    'HDBSCAN_AVAILABLE',
    # Cluster features
    'compute_prpd_features',
    'compute_waveform_aggregates',
    'compute_cluster_features',
    'trimmed_mean',
    'fit_weibull',
    # Definitions
    'WAVEFORM_FEATURE_NAMES',
    'CLUSTER_FEATURE_NAMES',
    'WAVEFORM_MEAN_FEATURE_NAMES',
    'WAVEFORM_TRIMMED_MEAN_FEATURE_NAMES',
    'ALL_CLUSTER_FEATURE_NAMES',
    'CLUSTER_FEATURE_GROUPS',
    'DEFAULT_CLASSIFICATION_FEATURES',
    'CLUSTERING_METHODS',
    'DEFAULT_CLUSTERING_METHOD',
]
