"""
Evaluation and Reporting

Generates confusion matrices, accuracy reports, and visualizations.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from .auto_calibrate import CalibrationResult


class Evaluator:
    """
    Evaluation and reporting for classifier calibration.

    Usage:
        evaluator = Evaluator(calibration_result)
        evaluator.print_report()
        evaluator.print_confusion_matrix()
    """

    def __init__(self, result: CalibrationResult):
        """
        Initialize evaluator.

        Args:
            result: CalibrationResult from calibration
        """
        self.result = result

    def print_report(self):
        """Print a comprehensive report."""
        print("=" * 70)
        print("CLASSIFIER CALIBRATION REPORT")
        print("=" * 70)
        print(f"Timestamp: {self.result.timestamp}")
        print(f"K-Threshold: {self.result.k_threshold}")
        print(f"Trials: {self.result.n_trials}")
        print(f"Best Overall Accuracy: {self.result.best_accuracy:.2%}")
        print()

        # Per-type accuracy
        print("Per-Type Accuracy:")
        print("-" * 40)
        for pd_type, acc in sorted(self.result.per_type_accuracy.items()):
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {pd_type:12s} {bar} {acc:.1%}")
        print()

        # Confusion matrix
        self.print_confusion_matrix()

        # Best thresholds
        print("\nOptimized Thresholds:")
        print("-" * 40)
        for section, params in sorted(self.result.best_thresholds.items()):
            print(f"  [{section}]")
            for key, value in sorted(params.items()):
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        print()

    def print_confusion_matrix(self):
        """Print confusion matrix."""
        confusion = self.result.confusion_matrix

        if not confusion:
            print("No confusion matrix data available.")
            return

        # Get all labels
        all_labels = set(confusion.keys())
        for expected, predictions in confusion.items():
            all_labels.update(predictions.keys())

        # Sort labels
        labels = sorted(all_labels)

        # Build matrix
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(labels)}

        for expected, predictions in confusion.items():
            i = label_to_idx.get(expected, -1)
            if i < 0:
                continue
            for predicted, count in predictions.items():
                j = label_to_idx.get(predicted, -1)
                if j >= 0:
                    matrix[i, j] = count

        # Print matrix
        print("\nConfusion Matrix:")
        print("-" * 70)

        # Header
        header = "Expected\\Predicted"
        col_width = max(12, max(len(l) for l in labels) + 2)
        print(f"{header:18s}", end="")
        for label in labels:
            print(f"{label:>{col_width}}", end="")
        print(f"{'Total':>{col_width}}")

        # Rows
        print("-" * (18 + col_width * (len(labels) + 1)))
        for i, expected in enumerate(labels):
            print(f"{expected:18s}", end="")
            row_total = 0
            for j, predicted in enumerate(labels):
                count = matrix[i, j]
                row_total += count
                if i == j:
                    # Highlight diagonal (correct predictions)
                    print(f"{count:>{col_width}}", end="")
                else:
                    print(f"{count:>{col_width}}", end="")
            print(f"{row_total:>{col_width}}")

        # Column totals
        print("-" * (18 + col_width * (len(labels) + 1)))
        print(f"{'Total':18s}", end="")
        grand_total = 0
        for j in range(len(labels)):
            col_total = matrix[:, j].sum()
            grand_total += col_total
            print(f"{col_total:>{col_width}}", end="")
        print(f"{grand_total:>{col_width}}")
        print()

    def get_threshold_code(self) -> str:
        """Generate Python code for the optimized thresholds."""
        lines = [
            "# Optimized thresholds from auto-calibration",
            f"# Accuracy: {self.result.best_accuracy:.2%}",
            f"# K-Threshold: {self.result.k_threshold}",
            f"# Generated: {self.result.timestamp}",
            "",
            "CALIBRATED_THRESHOLDS = {",
        ]

        for section, params in sorted(self.result.best_thresholds.items()):
            lines.append(f"    '{section}': {{")
            for key, value in sorted(params.items()):
                if isinstance(value, float):
                    lines.append(f"        '{key}': {value:.6f},")
                else:
                    lines.append(f"        '{key}': {value},")
            lines.append("    },")

        lines.append("}")
        lines.append("")
        lines.append("# Usage:")
        lines.append("# classifier = PDTypeClassifier(thresholds=CALIBRATED_THRESHOLDS)")

        return "\n".join(lines)

    def save_threshold_code(self, filepath: str):
        """Save optimized thresholds as Python code."""
        code = self.get_threshold_code()
        with open(filepath, 'w') as f:
            f.write(code)
        print(f"Saved threshold code to: {filepath}")


def compare_results(
    results: List[CalibrationResult],
    names: Optional[List[str]] = None
) -> None:
    """
    Compare multiple calibration results.

    Args:
        results: List of CalibrationResult objects
        names: Optional names for each result
    """
    if names is None:
        names = [f"Run {i+1}" for i in range(len(results))]

    print("=" * 70)
    print("CALIBRATION COMPARISON")
    print("=" * 70)
    print()

    # Overall accuracy comparison
    print("Overall Accuracy:")
    print("-" * 40)
    for name, result in zip(names, results):
        bar = "█" * int(result.best_accuracy * 30) + "░" * (30 - int(result.best_accuracy * 30))
        print(f"  {name:20s} {bar} {result.best_accuracy:.1%}")
    print()

    # Per-type comparison
    all_types = set()
    for result in results:
        all_types.update(result.per_type_accuracy.keys())

    for pd_type in sorted(all_types):
        print(f"\n{pd_type} Accuracy:")
        for name, result in zip(names, results):
            acc = result.per_type_accuracy.get(pd_type, 0.0)
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {name:20s} {bar} {acc:.1%}")


if __name__ == "__main__":
    # Test
    print("Evaluation module loaded successfully")
