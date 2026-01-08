"""
pdlib.classification - PD Type Classification

This module provides:
- PDTypeClassifier: Decision tree classifier for PD types
- PD type definitions (CORONA, INTERNAL, SURFACE, NOISE)
- Scoring utilities for classification

Usage:
    from pdlib.classification import PDTypeClassifier, PD_TYPES

    # Create classifier
    classifier = PDTypeClassifier()

    # Classify a cluster
    result = classifier.classify(cluster_features, cluster_label=0)
    print(f"Type: {result['pd_type']}, Confidence: {result['confidence']}")

    # Classify all clusters
    all_results = classifier.classify_all(cluster_features_dict)
"""

from .pd_types import (
    PD_TYPES,
    PD_TYPE_CODES,
    PD_CODE_NAMES,
    get_pd_type_info,
    get_pd_type_code,
    get_pd_type_name,
)

from .classifier import PDTypeClassifier

__all__ = [
    # PD Types
    'PD_TYPES',
    'PD_TYPE_CODES',
    'PD_CODE_NAMES',
    'get_pd_type_info',
    'get_pd_type_code',
    'get_pd_type_name',
    # Classifier
    'PDTypeClassifier',
]
