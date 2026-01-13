"""
Auto-Calibration using Optuna

Optimizes classifier thresholds using labeled lab data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Install with: pip install optuna")

from pdlib.classification import PDTypeClassifier
from pdlib.clustering import cluster_pulses, compute_cluster_features
from pdlib.features import FEATURE_NAMES

from .feature_pipeline import DatasetFeatures, prepare_training_data


@dataclass
class CalibrationResult:
    """Results from a calibration run."""
    best_thresholds: Dict[str, Any]
    best_accuracy: float
    confusion_matrix: Dict[str, Dict[str, int]]
    per_type_accuracy: Dict[str, float]
    n_trials: int
    timestamp: str
    k_threshold: float

    def save(self, filepath: str):
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'CalibrationResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ClassifierCalibrator:
    """
    Optuna-based calibrator for PD classifier thresholds.

    This optimizer finds threshold values that maximize classification
    accuracy on labeled lab data.

    Usage:
        calibrator = ClassifierCalibrator(dataset_features)
        result = calibrator.optimize(n_trials=100)

        # Apply optimized thresholds
        classifier = PDTypeClassifier(thresholds=result.best_thresholds)
    """

    # Define the parameter search space
    # Each parameter has (min, max, step) or (min, max) for continuous
    PARAM_SPACE = {
        # Noise detection
        'noise_detection.min_phase_entropy': (0.8, 0.99, 0.01),
        'noise_detection.noise_score_threshold': (0.2, 0.5, 0.05),
        'noise_detection.min_mean_snr_for_pd': (3.0, 10.0, 0.5),

        # Phase spread
        'phase_spread.surface_phase_spread_min': (10.0, 120.0, 5.0),

        # Surface detection
        'surface_detection.min_surface_score': (4, 12, 1),
        'surface_detection.surface_phase_spread': (10.0, 120.0, 5.0),
        'surface_detection.surface_max_phase_entropy': (0.4, 0.8, 0.05),
        'surface_detection.surface_min_snr_for_focused': (4.0, 12.0, 0.5),

        # Corona/Internal
        'corona_internal.min_corona_score': (6, 20, 1),
        'corona_internal.min_internal_score': (6, 20, 1),
        'corona_internal.corona_neg_max_asymmetry': (-0.9, -0.3, 0.1),
        'corona_internal.corona_pos_min_asymmetry': (0.3, 0.9, 0.1),
    }

    def __init__(
        self,
        dataset_features: List[DatasetFeatures],
        ac_frequency: float = 60.0,
        only_above_threshold: bool = True,
    ):
        """
        Initialize the calibrator.

        Args:
            dataset_features: List of DatasetFeatures from FeaturePipeline
            ac_frequency: AC frequency for phase calculations
            only_above_threshold: Only use pulses above noise floor for training
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna required for calibration. Install with: pip install optuna")

        self.dataset_features = dataset_features
        self.ac_frequency = ac_frequency
        self.only_above_threshold = only_above_threshold

        # Prepare the training data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare clustered data for evaluation."""
        self.evaluation_data = []

        for df in self.dataset_features:
            expected_type = df.dataset.expected_type

            # Filter pulses
            if self.only_above_threshold:
                valid_indices = [
                    i for i, p in enumerate(df.pulses)
                    if p.above_noise_floor
                ]
            else:
                valid_indices = list(range(len(df.pulses)))

            if len(valid_indices) < 5:
                print(f"  Skipping {df.dataset.filename}: only {len(valid_indices)} valid pulses")
                continue

            # Get feature matrix for valid pulses
            feature_matrix = df.feature_matrix[valid_indices]
            phases = [df.pulses[i].phase for i in valid_indices]

            # Cluster the pulses
            labels, cluster_info = cluster_pulses(
                feature_matrix,
                df.feature_names,
                method='hdbscan',
                min_samples=5,
                min_cluster_size=5
            )

            # Compute cluster features
            cluster_features = compute_cluster_features(
                feature_matrix=feature_matrix,
                feature_names=df.feature_names,
                labels=labels,
                trigger_times=None,
                ac_frequency=self.ac_frequency
            )

            # Store for evaluation
            self.evaluation_data.append({
                'filename': df.dataset.filename,
                'expected_type': expected_type,
                'cluster_features': cluster_features,
                'labels': labels,
                'n_pulses': len(valid_indices),
            })

        print(f"Prepared {len(self.evaluation_data)} datasets for calibration")

    def _build_thresholds(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """Build threshold dict from Optuna trial suggestions."""
        thresholds = {}

        for param_name, bounds in self.PARAM_SPACE.items():
            section, key = param_name.split('.', 1)

            if section not in thresholds:
                thresholds[section] = {}

            if len(bounds) == 3:
                # Discrete with step
                low, high, step = bounds
                if isinstance(step, int) or step == int(step):
                    value = trial.suggest_int(param_name, int(low), int(high), step=int(step))
                else:
                    value = trial.suggest_float(param_name, low, high, step=step)
            else:
                # Continuous
                low, high = bounds
                value = trial.suggest_float(param_name, low, high)

            thresholds[section][key] = value

        return thresholds

    def _evaluate(self, thresholds: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        Evaluate classifier with given thresholds.

        Returns:
            Tuple of (accuracy, detailed_results)
        """
        classifier = PDTypeClassifier(thresholds=thresholds, verbose=False)

        correct = 0
        total = 0
        confusion = {}  # confusion[expected][predicted] = count

        for data in self.evaluation_data:
            expected_type = data['expected_type']
            cluster_features = data['cluster_features']
            labels = data['labels']

            # Initialize confusion matrix rows
            if expected_type not in confusion:
                confusion[expected_type] = {}

            # Classify each cluster
            for cluster_label, features in cluster_features.items():
                result = classifier.classify(features, int(cluster_label))
                predicted_type = result['pd_type']

                # Handle noise cluster (-1)
                if cluster_label == -1:
                    # DBSCAN noise should map to NOISE
                    if predicted_type in ['NOISE', 'NOISE_MULTIPULSE']:
                        # This is expected behavior, don't count against accuracy
                        continue

                # Count pulses in this cluster
                n_in_cluster = np.sum(labels == cluster_label)

                # Update confusion matrix
                if predicted_type not in confusion[expected_type]:
                    confusion[expected_type][predicted_type] = 0
                confusion[expected_type][predicted_type] += n_in_cluster

                # Check if correct
                # For CORONA, allow both CORONA subtypes
                # For NOISE, allow NOISE and NOISE_MULTIPULSE
                is_correct = False
                if expected_type == 'CORONA' and predicted_type == 'CORONA':
                    is_correct = True
                elif expected_type == 'INTERNAL' and predicted_type == 'INTERNAL':
                    is_correct = True
                elif expected_type == 'SURFACE' and predicted_type == 'SURFACE':
                    is_correct = True
                elif expected_type == 'NOISE' and predicted_type in ['NOISE', 'NOISE_MULTIPULSE']:
                    is_correct = True
                elif expected_type == predicted_type:
                    is_correct = True

                if is_correct:
                    correct += n_in_cluster
                total += n_in_cluster

        accuracy = correct / total if total > 0 else 0.0

        return accuracy, {
            'confusion': confusion,
            'correct': correct,
            'total': total,
        }

    def objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective function."""
        thresholds = self._build_thresholds(trial)
        accuracy, _ = self._evaluate(thresholds)
        return accuracy

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> CalibrationResult:
        """
        Run optimization to find best thresholds.

        Args:
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            show_progress: Show progress bar

        Returns:
            CalibrationResult with best thresholds and metrics
        """
        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
        )

        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        # Get best thresholds
        best_thresholds = self._build_thresholds_from_params(study.best_params)

        # Final evaluation with best thresholds
        accuracy, details = self._evaluate(best_thresholds)

        # Compute per-type accuracy
        per_type_accuracy = {}
        for expected_type, predictions in details['confusion'].items():
            type_total = sum(predictions.values())
            type_correct = predictions.get(expected_type, 0)
            per_type_accuracy[expected_type] = type_correct / type_total if type_total > 0 else 0.0

        # Get K threshold from first dataset
        k_threshold = self.dataset_features[0].k_threshold if self.dataset_features else 6.0

        return CalibrationResult(
            best_thresholds=best_thresholds,
            best_accuracy=accuracy,
            confusion_matrix=details['confusion'],
            per_type_accuracy=per_type_accuracy,
            n_trials=n_trials,
            timestamp=datetime.now().isoformat(),
            k_threshold=k_threshold,
        )

    def _build_thresholds_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build threshold dict from flat params dict."""
        thresholds = {}

        for param_name, value in params.items():
            section, key = param_name.split('.', 1)
            if section not in thresholds:
                thresholds[section] = {}
            thresholds[section][key] = value

        return thresholds


def run_grid_search(
    dataset_features: List[DatasetFeatures],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    ac_frequency: float = 60.0,
) -> CalibrationResult:
    """
    Simple grid search alternative (doesn't require optuna).

    This is slower but doesn't have dependencies.
    """
    if param_grid is None:
        # Default small grid
        param_grid = {
            'phase_spread.surface_phase_spread_min': [20.0, 30.0, 50.0, 80.0],
            'surface_detection.min_surface_score': [6, 8, 10],
            'corona_internal.min_corona_score': [10, 15, 20],
            'corona_internal.min_internal_score': [10, 15, 20],
        }

    # Generate all combinations
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_accuracy = 0.0
    best_thresholds = {}
    best_confusion = {}

    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    print(f"Running grid search with {total_combinations} combinations...")

    for i, combination in enumerate(product(*param_values)):
        # Build thresholds
        thresholds = {}
        for name, value in zip(param_names, combination):
            section, key = name.split('.', 1)
            if section not in thresholds:
                thresholds[section] = {}
            thresholds[section][key] = value

        # Evaluate (simplified - just use first dataset for speed)
        classifier = PDTypeClassifier(thresholds=thresholds, verbose=False)

        correct = 0
        total = 0
        confusion = {}

        for df in dataset_features:
            expected_type = df.dataset.expected_type

            # Quick evaluation using pulse-level classification
            for pulse in df.pulses:
                if not pulse.above_noise_floor:
                    continue

                # Classify based on features
                result = classifier.classify(pulse.features, 0)
                predicted = result['pd_type']

                if expected_type not in confusion:
                    confusion[expected_type] = {}
                if predicted not in confusion[expected_type]:
                    confusion[expected_type][predicted] = 0
                confusion[expected_type][predicted] += 1

                if predicted == expected_type:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_thresholds = thresholds
            best_confusion = confusion

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total_combinations}, best accuracy: {best_accuracy:.2%}")

    k_threshold = dataset_features[0].k_threshold if dataset_features else 6.0

    return CalibrationResult(
        best_thresholds=best_thresholds,
        best_accuracy=best_accuracy,
        confusion_matrix=best_confusion,
        per_type_accuracy={},
        n_trials=total_combinations,
        timestamp=datetime.now().isoformat(),
        k_threshold=k_threshold,
    )


if __name__ == "__main__":
    # Test
    print("Auto-calibration module loaded successfully")
    print(f"Optuna available: {OPTUNA_AVAILABLE}")
