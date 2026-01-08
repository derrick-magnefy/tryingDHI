"""
Trigger Detection Methods for Raw PD Data Streams

Provides multiple methods to detect trigger points (PD pulse locations)
in continuous data streams that don't have hardware triggering.

Methods:
- stdev: Threshold based on standard deviation above baseline
- pulse_rate: Adaptive threshold targeting a maximum pulse rate per AC cycle
- histogram_knee: Find the "knee" in amplitude histogram (Kneedle algorithm)

Usage:
    detector = TriggerDetector(method='histogram_knee')
    triggers = detector.detect(signal, sample_rate=250e6, ac_frequency=60)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# Available trigger detection methods
TRIGGER_METHODS = [
    'stdev',           # Threshold = mean + k * stdev
    'pulse_rate',      # Adaptive threshold targeting max pulses per cycle
    'histogram_knee',  # Find knee in amplitude histogram
]

# Default method (histogram_knee is often most robust)
DEFAULT_TRIGGER_METHOD = 'histogram_knee'


@dataclass
class TriggerResult:
    """Result of trigger detection."""
    triggers: np.ndarray          # Trigger sample indices
    threshold: float              # Threshold value used
    method: str                   # Method used
    stats: Dict[str, Any]         # Additional statistics


class TriggerDetector:
    """
    Detect trigger points in continuous PD data streams.

    Attributes:
        method: Detection method ('stdev', 'pulse_rate', 'histogram_knee')
        polarity: Which polarity to trigger on ('positive', 'negative', 'both')
        min_separation: Minimum samples between triggers (prevents double-triggering)
    """

    def __init__(
        self,
        method: str = DEFAULT_TRIGGER_METHOD,
        polarity: str = 'both',
        min_separation: int = 100,
        **kwargs
    ):
        """
        Initialize trigger detector.

        Args:
            method: Detection method (see TRIGGER_METHODS)
            polarity: 'positive', 'negative', or 'both'
            min_separation: Minimum samples between triggers
            **kwargs: Method-specific parameters
        """
        if method not in TRIGGER_METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {TRIGGER_METHODS}")

        self.method = method
        self.polarity = polarity
        self.min_separation = min_separation
        self.params = kwargs

    def detect(
        self,
        signal: np.ndarray,
        sample_rate: float,
        ac_frequency: float = 60.0,
        **kwargs
    ) -> TriggerResult:
        """
        Detect trigger points in the signal.

        Args:
            signal: Raw signal data (1D numpy array)
            sample_rate: Sample rate in Hz
            ac_frequency: AC power frequency in Hz (for pulse_rate method)
            **kwargs: Additional method-specific parameters

        Returns:
            TriggerResult with trigger indices and metadata
        """
        # Merge kwargs with stored params
        params = {**self.params, **kwargs}

        if self.method == 'stdev':
            return self._detect_stdev(signal, sample_rate, **params)
        elif self.method == 'pulse_rate':
            return self._detect_pulse_rate(signal, sample_rate, ac_frequency, **params)
        elif self.method == 'histogram_knee':
            return self._detect_histogram_knee(signal, sample_rate, **params)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_stdev(
        self,
        signal: np.ndarray,
        sample_rate: float,
        k_sigma: float = 5.0,
        baseline_percentile: float = 50.0,
        **kwargs
    ) -> TriggerResult:
        """
        Standard deviation threshold method.

        Threshold = baseline + k * stdev

        Args:
            signal: Raw signal
            sample_rate: Sample rate in Hz
            k_sigma: Number of standard deviations above baseline
            baseline_percentile: Percentile to use as baseline (50 = median)

        Returns:
            TriggerResult
        """
        # Calculate baseline and noise level
        baseline = np.percentile(signal, baseline_percentile)

        # Use MAD (median absolute deviation) for robust noise estimate
        mad = np.median(np.abs(signal - baseline))
        stdev_estimate = 1.4826 * mad  # Convert MAD to stdev equivalent

        # Calculate threshold
        threshold = k_sigma * stdev_estimate

        # Find trigger points
        triggers = self._find_crossings(signal - baseline, threshold)

        return TriggerResult(
            triggers=triggers,
            threshold=threshold,
            method='stdev',
            stats={
                'baseline': baseline,
                'stdev_estimate': stdev_estimate,
                'k_sigma': k_sigma,
                'num_triggers': len(triggers),
            }
        )

    def _detect_pulse_rate(
        self,
        signal: np.ndarray,
        sample_rate: float,
        ac_frequency: float = 60.0,
        target_rate_per_cycle: float = 100.0,
        max_iterations: int = 50,
        initial_k_sigma: float = 3.0,
        **kwargs
    ) -> TriggerResult:
        """
        Pulse rate targeting method.

        Iteratively adjusts threshold until pulse rate falls below target.

        Args:
            signal: Raw signal
            sample_rate: Sample rate in Hz
            ac_frequency: AC frequency in Hz
            target_rate_per_cycle: Maximum acceptable pulses per AC cycle
            max_iterations: Maximum iterations for threshold search
            initial_k_sigma: Starting point for threshold (in stdevs)

        Returns:
            TriggerResult
        """
        # Calculate baseline
        baseline = np.median(signal)
        mad = np.median(np.abs(signal - baseline))
        stdev_estimate = 1.4826 * mad

        # Calculate number of cycles in the signal
        duration = len(signal) / sample_rate
        num_cycles = duration * ac_frequency

        # Binary search for optimal threshold
        k_low, k_high = initial_k_sigma, 20.0
        best_threshold = k_high * stdev_estimate
        best_triggers = np.array([])

        for _ in range(max_iterations):
            k_mid = (k_low + k_high) / 2
            threshold = k_mid * stdev_estimate

            triggers = self._find_crossings(signal - baseline, threshold)
            pulse_rate = len(triggers) / num_cycles if num_cycles > 0 else 0

            if pulse_rate <= target_rate_per_cycle:
                # Threshold is high enough, try lower
                k_high = k_mid
                best_threshold = threshold
                best_triggers = triggers
            else:
                # Too many pulses, need higher threshold
                k_low = k_mid

            # Convergence check
            if k_high - k_low < 0.1:
                break

        # Final detection with best threshold
        if len(best_triggers) == 0:
            best_triggers = self._find_crossings(signal - baseline, best_threshold)

        return TriggerResult(
            triggers=best_triggers,
            threshold=best_threshold,
            method='pulse_rate',
            stats={
                'baseline': baseline,
                'stdev_estimate': stdev_estimate,
                'target_rate_per_cycle': target_rate_per_cycle,
                'achieved_rate_per_cycle': len(best_triggers) / num_cycles if num_cycles > 0 else 0,
                'num_cycles': num_cycles,
                'num_triggers': len(best_triggers),
            }
        )

    def _detect_histogram_knee(
        self,
        signal: np.ndarray,
        sample_rate: float,
        num_bins: int = 1000,
        sensitivity: float = 1.0,
        **kwargs
    ) -> TriggerResult:
        """
        Histogram knee detection method.

        Finds the "knee" or "elbow" point in the amplitude histogram where
        the distribution transitions from noise floor to signal.

        Args:
            signal: Raw signal
            sample_rate: Sample rate in Hz
            num_bins: Number of histogram bins
            sensitivity: Knee detection sensitivity (higher = more sensitive)

        Returns:
            TriggerResult
        """
        # Remove baseline
        baseline = np.median(signal)
        centered = signal - baseline

        # Work with absolute values for threshold detection
        abs_signal = np.abs(centered)

        # Create histogram of absolute amplitudes
        hist, bin_edges = np.histogram(abs_signal, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find the knee using the Kneedle algorithm
        # Normalize the curve
        x_norm = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min() + 1e-10)

        # Use cumulative sum (CDF) for smoother knee detection
        cumsum = np.cumsum(hist)
        y_norm = cumsum / (cumsum.max() + 1e-10)

        # Find knee: maximum distance from the line connecting endpoints
        # Line from (0, y_norm[0]) to (1, y_norm[-1])
        line_y = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm

        # Distance from each point to the line
        distances = np.abs(y_norm - line_y)

        # Weight by sensitivity - prefer earlier knees for higher sensitivity
        weighted_distances = distances * (1 - sensitivity * 0.3 * x_norm)

        # Find the knee point
        knee_idx = np.argmax(weighted_distances)
        threshold = bin_centers[knee_idx]

        # Ensure minimum threshold (at least 3 sigma)
        mad = np.median(np.abs(centered))
        min_threshold = 3 * 1.4826 * mad
        threshold = max(threshold, min_threshold)

        # Find trigger points
        triggers = self._find_crossings(centered, threshold)

        return TriggerResult(
            triggers=triggers,
            threshold=threshold,
            method='histogram_knee',
            stats={
                'baseline': baseline,
                'knee_bin_index': knee_idx,
                'knee_amplitude': bin_centers[knee_idx],
                'min_threshold': min_threshold,
                'num_triggers': len(triggers),
                'histogram_bins': num_bins,
            }
        )

    def _find_crossings(
        self,
        signal: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Find threshold crossing points in the signal.

        Args:
            signal: Baseline-corrected signal
            threshold: Absolute threshold value

        Returns:
            Array of trigger sample indices
        """
        triggers = []

        if self.polarity in ('positive', 'both'):
            # Find positive crossings
            above = signal > threshold
            # Find rising edges (transitions from below to above)
            crossings = np.where(np.diff(above.astype(int)) == 1)[0] + 1
            triggers.extend(crossings.tolist())

        if self.polarity in ('negative', 'both'):
            # Find negative crossings
            below = signal < -threshold
            # Find falling edges (transitions from above to below)
            crossings = np.where(np.diff(below.astype(int)) == 1)[0] + 1
            triggers.extend(crossings.tolist())

        # Sort and remove duplicates
        triggers = sorted(set(triggers))

        # Apply minimum separation filter
        if self.min_separation > 0 and len(triggers) > 1:
            filtered = [triggers[0]]
            for t in triggers[1:]:
                if t - filtered[-1] >= self.min_separation:
                    filtered.append(t)
            triggers = filtered

        return np.array(triggers, dtype=np.int64)

    def estimate_noise_floor(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Estimate the noise floor of the signal.

        Returns:
            Dict with noise statistics
        """
        baseline = np.median(signal)
        centered = signal - baseline

        # MAD-based noise estimate
        mad = np.median(np.abs(centered))
        stdev_estimate = 1.4826 * mad

        # RMS of lower quartile (mostly noise)
        lower_quartile = np.abs(centered) < np.percentile(np.abs(centered), 25)
        rms_noise = np.sqrt(np.mean(centered[lower_quartile]**2)) if np.any(lower_quartile) else stdev_estimate

        return {
            'baseline': baseline,
            'mad': mad,
            'stdev_estimate': stdev_estimate,
            'rms_noise': rms_noise,
            'peak_to_noise': np.max(np.abs(centered)) / stdev_estimate,
        }


def compare_methods(
    signal: np.ndarray,
    sample_rate: float,
    ac_frequency: float = 60.0,
    **kwargs
) -> Dict[str, TriggerResult]:
    """
    Compare all trigger detection methods on the same signal.

    Args:
        signal: Raw signal data
        sample_rate: Sample rate in Hz
        ac_frequency: AC frequency in Hz
        **kwargs: Parameters passed to all methods

    Returns:
        Dict mapping method name to TriggerResult
    """
    results = {}

    for method in TRIGGER_METHODS:
        detector = TriggerDetector(method=method, **kwargs)
        try:
            results[method] = detector.detect(signal, sample_rate, ac_frequency)
        except Exception as e:
            print(f"Method {method} failed: {e}")

    return results
