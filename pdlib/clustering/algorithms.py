"""
Clustering algorithms for PD pulse analysis.

Provides DBSCAN, HDBSCAN, and K-means clustering with automatic parameter estimation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Try to import HDBSCAN (optional dependency)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def estimate_dbscan_eps(X_scaled: np.ndarray, k: int = 5, percentile: int = 60) -> float:
    """
    Estimate optimal epsilon for DBSCAN using k-nearest neighbors.

    Uses the k-distance graph with configurable percentile.
    Lower percentile = tighter clusters, higher = looser clusters.

    Args:
        X_scaled: Scaled feature matrix
        k: Number of neighbors (usually same as min_samples)
        percentile: Percentile of k-distances to use (default 60, range 50-90)

    Returns:
        Estimated epsilon value
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)

    # Get the k-th nearest neighbor distance for each point
    k_distances = np.sort(distances[:, k-1])

    # Use specified percentile (lower = tighter clusters)
    eps = np.percentile(k_distances, percentile)

    return eps


def run_dbscan(
    X_scaled: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = 5,
    auto_percentile: int = 60
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run DBSCAN clustering.

    Args:
        X_scaled: Scaled feature matrix
        eps: Epsilon parameter (auto-estimated if None)
        min_samples: Minimum samples for core point
        auto_percentile: Percentile for auto eps estimation (default 60)

    Returns:
        Tuple of (labels, info_dict)
        - labels: Cluster labels (-1 for noise)
        - info_dict: Dict with clustering info
    """
    if eps is None:
        eps = estimate_dbscan_eps(X_scaled, k=min_samples, percentile=auto_percentile)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    info = {
        'method': 'DBSCAN',
        'eps': eps,
        'min_samples': min_samples,
        'auto_percentile': auto_percentile,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels) if len(labels) > 0 else 0
    }

    # Calculate silhouette score if we have valid clusters
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            try:
                info['silhouette_score'] = silhouette_score(X_scaled[mask], labels[mask])
            except:
                info['silhouette_score'] = None

    return labels, info


def run_hdbscan(
    X_scaled: np.ndarray,
    min_samples: int = 5,
    min_cluster_size: Optional[int] = None,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run HDBSCAN clustering.

    HDBSCAN automatically determines the number of clusters and handles varying
    density better than DBSCAN. No eps parameter needed.

    Args:
        X_scaled: Scaled feature matrix
        min_samples: Minimum samples for core point (default: 5)
        min_cluster_size: Minimum cluster size (default: same as min_samples)
        cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'
        metric: Distance metric (default: 'euclidean')

    Returns:
        Tuple of (labels, info_dict)
        - labels: Cluster labels (-1 for noise)
        - info_dict: Dict with clustering info
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("HDBSCAN not installed. Install with: pip install hdbscan")

    if min_cluster_size is None:
        min_cluster_size = min_samples

    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_method=cluster_selection_method
    )
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    info = {
        'method': 'HDBSCAN',
        'min_samples': min_samples,
        'min_cluster_size': min_cluster_size,
        'cluster_selection_method': cluster_selection_method,
        'metric': metric,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels) if len(labels) > 0 else 0
    }

    # Calculate silhouette score if we have valid clusters
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            try:
                info['silhouette_score'] = silhouette_score(X_scaled[mask], labels[mask])
            except:
                info['silhouette_score'] = None

    return labels, info


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 10,
    random_state: int = 42,
    max_iter: int = 300
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run K-means clustering.

    Args:
        X_scaled: Scaled feature matrix
        n_clusters: Number of clusters
        n_init: Number of initializations
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations

    Returns:
        Tuple of (labels, info_dict)
        - labels: Cluster labels
        - info_dict: Dict with clustering info
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    labels = kmeans.fit_predict(X_scaled)

    info = {
        'method': 'KMeans',
        'n_clusters': n_clusters,
        'n_init': n_init,
        'inertia': kmeans.inertia_,
        'n_iterations': kmeans.n_iter_
    }

    # Calculate silhouette score
    if n_clusters > 1:
        try:
            info['silhouette_score'] = silhouette_score(X_scaled, labels)
        except:
            info['silhouette_score'] = None

    return labels, info


def scale_features(
    features: np.ndarray,
    feature_weights: Optional[Dict[str, float]] = None,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler and optionally apply weights.

    Args:
        features: Feature matrix (n_samples, n_features)
        feature_weights: Optional dict mapping feature names to weights
        feature_names: Feature names (required if weights provided)

    Returns:
        Tuple of (scaled_features, scaler)
    """
    # Handle infinite values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Apply feature weights if specified
    if feature_weights and feature_names:
        weight_array = np.ones(len(feature_names))
        for i, feat_name in enumerate(feature_names):
            if feat_name in feature_weights:
                weight_array[i] = feature_weights[feat_name]
        X_scaled = X_scaled * weight_array

    return X_scaled, scaler


def cluster_pulses(
    features: np.ndarray,
    feature_names: List[str],
    method: str = 'hdbscan',
    feature_weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    High-level function to cluster pulse features.

    Args:
        features: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        method: Clustering method ('hdbscan', 'dbscan', 'kmeans')
        feature_weights: Optional feature weights for clustering
        **kwargs: Additional arguments passed to the clustering function

    Returns:
        Tuple of (labels, info_dict)
    """
    # Scale features
    X_scaled, _ = scale_features(features, feature_weights, feature_names)

    # Run clustering
    if method == 'hdbscan':
        return run_hdbscan(X_scaled, **kwargs)
    elif method == 'dbscan':
        return run_dbscan(X_scaled, **kwargs)
    elif method == 'kmeans':
        return run_kmeans(X_scaled, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
