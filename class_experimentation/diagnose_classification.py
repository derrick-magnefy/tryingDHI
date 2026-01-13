#!/usr/bin/env python3
"""
Diagnose why files are being misclassified.

Prints key features for each file to understand what's triggering
NOISE or NOISE_MULTIPULSE classifications.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from class_experimentation.labeled_loader import LabeledDatasetLoader
from class_experimentation.feature_pipeline import FeaturePipeline
from pdlib.classification import PDTypeClassifier
from pdlib.clustering import compute_cluster_features


def diagnose(data_dir: str = "Example Data", k_threshold: float = 6.0):
    """Print diagnostic info for each file."""

    print("=" * 70)
    print("CLASSIFICATION DIAGNOSIS")
    print("=" * 70)

    # Load datasets
    loader = LabeledDatasetLoader(data_dir)
    datasets = loader.load_all()

    # Extract features
    pipeline = FeaturePipeline(k_threshold=k_threshold)
    dataset_features = pipeline.extract_all(datasets)

    # Classifier with default thresholds
    classifier = PDTypeClassifier(verbose=True)

    print("\n" + "=" * 70)
    print("FILE-BY-FILE DIAGNOSIS")
    print("=" * 70)

    for df in dataset_features:
        print(f"\n{'=' * 70}")
        print(f"FILE: {df.dataset.filename}")
        print(f"EXPECTED TYPE: {df.dataset.expected_type}")
        print(f"Pulses: {len(df.pulses)} total, {df.n_above_threshold} above K={k_threshold}")
        print("-" * 70)

        # Get pulses above threshold
        valid_indices = [i for i, p in enumerate(df.pulses) if p.above_noise_floor]

        if len(valid_indices) < 5:
            print("  SKIPPED: Too few pulses above threshold")
            continue

        # Compute file-level features
        feature_matrix = df.feature_matrix[valid_indices]
        all_same_label = np.zeros(len(valid_indices), dtype=int)

        cluster_features = compute_cluster_features(
            features_matrix=feature_matrix,
            feature_names=df.feature_names,
            labels=all_same_label,
            trigger_times=None,
            ac_frequency=60.0
        )

        features = cluster_features[0]  # Single cluster

        # Print key noise-related features
        print("\n  KEY NOISE DETECTION FEATURES:")
        print("  " + "-" * 40)

        noise_features = [
            ('mean_signal_to_noise_ratio', 'SNR (min=6.0 for non-noise)'),
            ('mean_spectral_flatness', 'Spectral Flatness (>0.6 = noise)'),
            ('mean_slew_rate', 'Slew Rate (<1e6 = noise)'),
            ('mean_crest_factor', 'Crest Factor (<3.0 = noise)'),
            ('phase_entropy', 'Phase Entropy (>0.95 = noise)'),
            ('amplitude_coefficient_of_variation', 'Amplitude CV (>1.5 = noise)'),
            ('pulses_per_cycle', 'Pulses/Cycle (>20 = noise)'),
            ('mean_pulse_count', 'Mean Pulse Count (multipulse detection)'),
            ('mean_is_multi_pulse', 'Is Multi-pulse (>0.5 = multipulse)'),
        ]

        for feat_name, description in noise_features:
            value = features.get(feat_name, features.get(feat_name.replace('mean_', ''), 'N/A'))
            if isinstance(value, float):
                if value > 1e5:
                    print(f"    {description}: {value:.2e}")
                else:
                    print(f"    {description}: {value:.4f}")
            else:
                print(f"    {description}: {value}")

        # Print key Corona/Internal features
        print("\n  KEY CORONA/INTERNAL FEATURES:")
        print("  " + "-" * 40)

        ci_features = [
            ('discharge_asymmetry', 'Discharge Asymmetry'),
            ('phase_of_max_activity', 'Phase of Max Activity'),
            ('phase_spread', 'Phase Spread'),
            ('amplitude_phase_correlation', 'Amp-Phase Correlation'),
            ('mean_spectral_power_low', 'Spectral Power Low'),
            ('quadrant_3_percentage', 'Q3 Percentage'),
            ('pulses_per_positive_halfcycle', 'Pulses per +halfcycle'),
            ('pulses_per_negative_halfcycle', 'Pulses per -halfcycle'),
        ]

        for feat_name, description in ci_features:
            value = features.get(feat_name, 'N/A')
            if isinstance(value, float):
                if value > 1e5:
                    print(f"    {description}: {value:.2e}")
                else:
                    print(f"    {description}: {value:.4f}")
            else:
                print(f"    {description}: {value}")

        # Run classification
        print("\n  CLASSIFICATION RESULT:")
        print("  " + "-" * 40)
        result = classifier.classify(features, 0)
        print(f"    Predicted: {result['pd_type']}")
        print(f"    Confidence: {result['confidence']:.2f}")
        print(f"    Branch Path: {' → '.join(result['branch_path'])}")
        print(f"    Reasoning:")
        for reason in result['reasoning']:
            print(f"      - {reason}")
        if result['warnings']:
            print(f"    Warnings:")
            for warn in result['warnings']:
                print(f"      ! {warn}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "Example Data"
    diagnose(data_dir)
