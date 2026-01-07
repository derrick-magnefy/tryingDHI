#!/usr/bin/env python3
"""
Test script to validate pdlib modules against existing implementations.

Compares:
1. Feature extraction: pdlib.features vs extract_features.py
2. Clustering: pdlib.clustering vs cluster_pulses.py
3. Classification: pdlib.classification vs classify_pd_type.py

Usage:
    python test_pdlib_comparison.py [--data-dir DIR] [--dataset PREFIX]
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

# ============================================================================
# TEST RESULTS TRACKING
# ============================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    def ok(self, msg):
        self.passed += 1
        self.details.append(f"✓ PASS: {msg}")
        print(f"  ✓ {msg}")

    def fail(self, msg):
        self.failed += 1
        self.details.append(f"✗ FAIL: {msg}")
        print(f"  ✗ {msg}")

    def warn(self, msg):
        self.warnings += 1
        self.details.append(f"⚠ WARN: {msg}")
        print(f"  ⚠ {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed, {self.failed} failed, {self.warnings} warnings")
        print(f"{'='*60}")
        return self.failed == 0


# ============================================================================
# TEST 1: IMPORTS
# ============================================================================

def test_imports(results):
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)

    # Test pdlib imports
    try:
        from pdlib import PDFeatureExtractor, FEATURE_NAMES, FEATURE_GROUPS
        results.ok(f"pdlib.features: {len(FEATURE_NAMES)} features defined")
    except Exception as e:
        results.fail(f"pdlib.features import failed: {e}")
        return False

    try:
        from pdlib import cluster_pulses, run_hdbscan, run_dbscan, HDBSCAN_AVAILABLE
        results.ok(f"pdlib.clustering: HDBSCAN available = {HDBSCAN_AVAILABLE}")
    except Exception as e:
        results.fail(f"pdlib.clustering import failed: {e}")
        return False

    try:
        from pdlib import PDTypeClassifier, PD_TYPES
        results.ok(f"pdlib.classification: {len(PD_TYPES)} PD types defined")
    except Exception as e:
        results.fail(f"pdlib.classification import failed: {e}")
        return False

    try:
        from middleware import AutoLoader, RuggedLoader
        results.ok("middleware.formats: AutoLoader, RuggedLoader")
    except Exception as e:
        results.fail(f"middleware import failed: {e}")
        return False

    try:
        from config.loader import ConfigLoader
        loader = ConfigLoader()
        thresholds = loader.get_thresholds()
        results.ok(f"config.loader: {len(thresholds)} threshold sections")
    except Exception as e:
        results.fail(f"config.loader import failed: {e}")
        return False

    return True


# ============================================================================
# TEST 2: DATA LOADING COMPARISON
# ============================================================================

def test_data_loading(results, data_dir, prefix):
    """Compare data loading between old and new implementations."""
    print("\n" + "="*60)
    print("TEST 2: Data Loading Comparison")
    print("="*60)

    # Old way: direct file reading
    from extract_features import load_waveforms, load_settings, load_single_line_data

    old_wfm_path = os.path.join(data_dir, f"{prefix}-WFMs.txt")
    old_sg_path = os.path.join(data_dir, f"{prefix}-SG.txt")
    old_ph_path = os.path.join(data_dir, f"{prefix}-Ph.txt")

    try:
        old_waveforms = load_waveforms(old_wfm_path)
        old_settings = load_settings(old_sg_path)
        old_phases = load_single_line_data(old_ph_path) if os.path.exists(old_ph_path) else None
        results.ok(f"Old loader: {len(old_waveforms)} waveforms loaded")
    except Exception as e:
        results.fail(f"Old loader failed: {e}")
        return None, None

    # New way: middleware
    from middleware import RuggedLoader

    try:
        loader = RuggedLoader(data_dir)
        new_waveforms = loader.load_waveforms(prefix)
        new_settings = loader.load_settings(prefix)
        new_phases = loader.load_phase_angles(prefix)
        results.ok(f"New loader: {len(new_waveforms)} waveforms loaded")
    except Exception as e:
        results.fail(f"New loader failed: {e}")
        return None, None

    # Compare waveform counts
    if len(old_waveforms) == len(new_waveforms):
        results.ok(f"Waveform count matches: {len(old_waveforms)}")
    else:
        results.fail(f"Waveform count mismatch: old={len(old_waveforms)}, new={len(new_waveforms)}")

    # Compare first waveform
    if len(old_waveforms) > 0 and len(new_waveforms) > 0:
        if np.allclose(old_waveforms[0], new_waveforms[0]):
            results.ok("First waveform data matches exactly")
        else:
            results.fail("First waveform data differs")

    # Compare settings
    old_sample_interval = old_settings[4] if len(old_settings) > 4 else 4e-9
    new_sample_interval = new_settings.get('sample_interval', 4e-9)

    if np.isclose(old_sample_interval, new_sample_interval):
        results.ok(f"Sample interval matches: {old_sample_interval}")
    else:
        results.fail(f"Sample interval mismatch: old={old_sample_interval}, new={new_sample_interval}")

    # Compare phases
    if old_phases is not None and new_phases is not None:
        if np.allclose(old_phases, new_phases):
            results.ok(f"Phase angles match: {len(old_phases)} values")
        else:
            results.fail("Phase angles differ")

    return (old_waveforms, old_settings, old_phases), (new_waveforms, new_settings, new_phases)


# ============================================================================
# TEST 3: FEATURE EXTRACTION COMPARISON
# ============================================================================

def test_feature_extraction(results, old_data, new_data, data_dir, prefix):
    """Compare feature extraction between old and new implementations."""
    print("\n" + "="*60)
    print("TEST 3: Feature Extraction Comparison")
    print("="*60)

    if old_data is None or new_data is None:
        results.warn("Skipping feature extraction test (data loading failed)")
        return None, None

    old_waveforms, old_settings, old_phases = old_data
    new_waveforms, new_settings, new_phases = new_data

    # Old way: extract_features.py logic
    from extract_features import FEATURE_NAMES as OLD_FEATURE_NAMES

    # New way: pdlib
    from pdlib import PDFeatureExtractor, FEATURE_NAMES as NEW_FEATURE_NAMES

    # Compare feature name lists
    print(f"\n  Old implementation: {len(OLD_FEATURE_NAMES)} features")
    print(f"  New implementation: {len(NEW_FEATURE_NAMES)} features")

    # Check for matching feature names
    old_set = set(OLD_FEATURE_NAMES)
    new_set = set(NEW_FEATURE_NAMES)

    common = old_set & new_set
    only_old = old_set - new_set
    only_new = new_set - old_set

    results.ok(f"Common features: {len(common)}")
    if only_old:
        results.warn(f"Features only in old: {only_old}")
    if only_new:
        results.warn(f"Features only in new: {only_new}")

    # Extract features using new implementation
    sample_interval = new_settings.get('sample_interval', 4e-9)
    ac_frequency = new_settings.get('ac_frequency', 60.0)

    extractor = PDFeatureExtractor(
        sample_interval=sample_interval,
        ac_frequency=ac_frequency,
        polarity_method='first_peak'
    )

    # Extract for first N waveforms
    n_test = min(10, len(new_waveforms))
    print(f"\n  Extracting features for {n_test} waveforms...")

    try:
        new_features = []
        for i in range(n_test):
            phase = new_phases[i] if new_phases is not None else None
            feat = extractor.extract_features(new_waveforms[i], phase_angle=phase)
            new_features.append(feat)
        results.ok(f"New extractor: extracted {len(new_features)} feature vectors")
    except Exception as e:
        results.fail(f"New extractor failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # Load existing features file for comparison if available
    features_file = os.path.join(data_dir, f"{prefix}-features.csv")
    if os.path.exists(features_file):
        print(f"\n  Comparing with existing features file: {features_file}")

        # Load old features
        old_features = []
        with open(features_file, 'r') as f:
            header = f.readline().strip().split(',')[1:]  # Skip waveform_index
            for i, line in enumerate(f):
                if i >= n_test:
                    break
                parts = line.strip().split(',')
                values = {header[j]: float(parts[j+1]) for j in range(len(header))}
                old_features.append(values)

        # Compare specific features
        features_to_compare = [
            'phase_angle', 'absolute_amplitude', 'polarity',
            'energy', 'rise_time', 'dominant_frequency'
        ]

        print("\n  Feature comparison (first waveform):")
        for feat_name in features_to_compare:
            if feat_name in old_features[0] and feat_name in new_features[0]:
                old_val = old_features[0][feat_name]
                new_val = new_features[0][feat_name]

                if old_val == 0 and new_val == 0:
                    match = True
                elif old_val == 0:
                    match = False
                else:
                    rel_diff = abs(new_val - old_val) / abs(old_val) if old_val != 0 else 0
                    match = rel_diff < 0.01  # 1% tolerance

                status = "✓" if match else "✗"
                print(f"    {status} {feat_name}: old={old_val:.6g}, new={new_val:.6g}")

                if match:
                    results.ok(f"Feature '{feat_name}' matches")
                else:
                    results.warn(f"Feature '{feat_name}' differs: old={old_val:.6g}, new={new_val:.6g}")
    else:
        results.warn(f"No existing features file found at {features_file}")

    return old_features if 'old_features' in dir() else None, new_features


# ============================================================================
# TEST 4: CLUSTERING COMPARISON
# ============================================================================

def test_clustering(results, data_dir, prefix):
    """Compare clustering between old and new implementations."""
    print("\n" + "="*60)
    print("TEST 4: Clustering Comparison")
    print("="*60)

    # Load features file
    features_file = os.path.join(data_dir, f"{prefix}-features.csv")
    if not os.path.exists(features_file):
        results.warn("Skipping clustering test (no features file)")
        return

    # Old way: cluster_pulses.py
    from cluster_pulses import load_features, run_dbscan, run_hdbscan as old_run_hdbscan
    from sklearn.preprocessing import StandardScaler

    # New way: pdlib
    from pdlib.clustering import run_dbscan as new_run_dbscan, run_hdbscan as new_run_hdbscan
    from pdlib.clustering import scale_features, HDBSCAN_AVAILABLE

    # Load features
    features, feature_names = load_features(features_file)
    print(f"  Loaded {features.shape[0]} samples, {features.shape[1]} features")

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features (same way for both)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Test DBSCAN
    print("\n  Testing DBSCAN...")
    try:
        old_labels, old_info = run_dbscan(X_scaled, min_samples=5)
        results.ok(f"Old DBSCAN: {old_info['n_clusters']} clusters, {old_info['n_noise']} noise")
    except Exception as e:
        results.fail(f"Old DBSCAN failed: {e}")
        old_labels, old_info = None, None

    try:
        new_labels, new_info = new_run_dbscan(X_scaled, min_samples=5)
        results.ok(f"New DBSCAN: {new_info['n_clusters']} clusters, {new_info['n_noise']} noise")
    except Exception as e:
        results.fail(f"New DBSCAN failed: {e}")
        new_labels, new_info = None, None

    if old_labels is not None and new_labels is not None:
        if old_info['n_clusters'] == new_info['n_clusters']:
            results.ok("DBSCAN cluster count matches")
        else:
            results.warn(f"DBSCAN cluster count differs: old={old_info['n_clusters']}, new={new_info['n_clusters']}")

    # Test HDBSCAN if available
    if HDBSCAN_AVAILABLE:
        print("\n  Testing HDBSCAN...")
        try:
            old_labels_h, old_info_h = old_run_hdbscan(X_scaled, min_samples=5)
            results.ok(f"Old HDBSCAN: {old_info_h['n_clusters']} clusters, {old_info_h['n_noise']} noise")
        except Exception as e:
            results.fail(f"Old HDBSCAN failed: {e}")
            old_labels_h, old_info_h = None, None

        try:
            new_labels_h, new_info_h = new_run_hdbscan(X_scaled, min_samples=5)
            results.ok(f"New HDBSCAN: {new_info_h['n_clusters']} clusters, {new_info_h['n_noise']} noise")
        except Exception as e:
            results.fail(f"New HDBSCAN failed: {e}")
            new_labels_h, new_info_h = None, None

        if old_labels_h is not None and new_labels_h is not None:
            if old_info_h['n_clusters'] == new_info_h['n_clusters']:
                results.ok("HDBSCAN cluster count matches")
            else:
                results.warn(f"HDBSCAN cluster count differs: old={old_info_h['n_clusters']}, new={new_info_h['n_clusters']}")
    else:
        results.warn("HDBSCAN not available, skipping HDBSCAN test")


# ============================================================================
# TEST 5: CLASSIFICATION COMPARISON
# ============================================================================

def test_classification(results, data_dir, prefix):
    """Compare classification between old and new implementations."""
    print("\n" + "="*60)
    print("TEST 5: Classification Comparison")
    print("="*60)

    # Check for cluster features file
    cluster_features_file = os.path.join(data_dir, f"{prefix}-cluster-features-hdbscan.csv")
    if not os.path.exists(cluster_features_file):
        cluster_features_file = os.path.join(data_dir, f"{prefix}-cluster-features-dbscan.csv")

    if not os.path.exists(cluster_features_file):
        results.warn("Skipping classification test (no cluster features file)")
        return

    # Old way
    from classify_pd_type import load_cluster_features, PDTypeClassifier as OldClassifier

    # New way
    from pdlib.classification import PDTypeClassifier as NewClassifier

    # Load cluster features
    old_features = load_cluster_features(cluster_features_file)
    print(f"  Loaded features for {len(old_features)} clusters")

    # Classify with both
    old_classifier = OldClassifier(verbose=False)
    new_classifier = NewClassifier(verbose=False)

    print("\n  Classification comparison:")
    matches = 0
    total = 0

    for label, features in old_features.items():
        old_result = old_classifier.classify(features, label)
        new_result = new_classifier.classify(features, label)

        old_type = old_result['pd_type']
        new_type = new_result['pd_type']

        status = "✓" if old_type == new_type else "✗"
        print(f"    {status} Cluster {label}: old={old_type}, new={new_type}")

        if old_type == new_type:
            matches += 1
        total += 1

    if matches == total:
        results.ok(f"All {total} classifications match")
    else:
        results.warn(f"{matches}/{total} classifications match")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test pdlib against existing implementations")
    parser.add_argument('--data-dir', default='Rugged Data Files', help='Data directory')
    parser.add_argument('--dataset', default=None, help='Specific dataset prefix to test')
    args = parser.parse_args()

    print("="*60)
    print("PDLIB COMPARISON TEST")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")

    results = TestResults()

    # Test imports
    if not test_imports(results):
        print("\nImport tests failed, cannot continue")
        return 1

    # Find a dataset to test
    if args.dataset:
        prefix = args.dataset
    else:
        # Find first available dataset
        from middleware import RuggedLoader
        loader = RuggedLoader(args.data_dir)
        datasets = loader.list_datasets()
        if not datasets:
            print(f"\nNo datasets found in {args.data_dir}")
            return 1
        prefix = datasets[0]

    print(f"\nTesting with dataset: {prefix}")

    # Run comparison tests
    old_data, new_data = test_data_loading(results, args.data_dir, prefix)
    old_features, new_features = test_feature_extraction(results, old_data, new_data, args.data_dir, prefix)
    test_clustering(results, args.data_dir, prefix)
    test_classification(results, args.data_dir, prefix)

    # Summary
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
