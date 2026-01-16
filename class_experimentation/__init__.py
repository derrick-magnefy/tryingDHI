"""
Classifier Experimentation Framework

Auto-calibration system for PD classifier thresholds using labeled lab datasets.

Usage:
    python -m class_experimentation.run_calibration --data-dir IEEE_Example_Data/

Components:
    - labeled_loader: Load .mat files with PD type labels
    - feature_pipeline: Extract features from waveforms
    - auto_calibrate: Optuna-based threshold optimization
    - evaluate: Confusion matrix and accuracy reporting
"""

from .labeled_loader import LabeledDatasetLoader
from .feature_pipeline import FeaturePipeline
from .auto_calibrate import ClassifierCalibrator
from .evaluate import Evaluator

__all__ = [
    'LabeledDatasetLoader',
    'FeaturePipeline',
    'ClassifierCalibrator',
    'Evaluator',
]
