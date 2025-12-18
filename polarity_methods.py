#!/usr/bin/env python3
"""
Polarity Calculation Methods for Partial Discharge Waveforms

This module provides multiple methods for determining the polarity of PD pulses.
Different methods may be more appropriate depending on the type of discharge
and measurement conditions.

Available Methods:
- peak: Based on which peak (positive or negative) has the larger absolute value
- first_peak: Based on which significant peak occurs first in time
- integrated_charge: Based on the sign of the total integrated charge
- energy_weighted: Based on energy-weighted signal centroid
- dominant_half_cycle: Based on which half-cycle contains more energy
- initial_slope: Based on the initial slope direction of the pulse

Usage:
    from polarity_methods import calculate_polarity, POLARITY_METHODS

    polarity = calculate_polarity(waveform, method='peak', sample_interval=4e-9)
"""

import numpy as np
from scipy import signal

# List of available polarity methods
POLARITY_METHODS = [
    'peak',
    'first_peak',
    'integrated_charge',
    'energy_weighted',
    'dominant_half_cycle',
    'initial_slope',
]

# Default method
DEFAULT_POLARITY_METHOD = 'first_peak'


def calculate_polarity(waveform, method='peak', sample_interval=4e-9, baseline_samples=50):
    """
    Calculate polarity of a waveform using the specified method.

    Args:
        waveform: numpy array of waveform samples
        method: Polarity calculation method (see POLARITY_METHODS)
        sample_interval: Time between samples in seconds (default: 4ns)
        baseline_samples: Number of samples to use for baseline correction

    Returns:
        int: 1 for positive polarity, -1 for negative polarity
    """
    # Remove DC offset using baseline from first samples
    baseline = np.mean(waveform[:baseline_samples]) if len(waveform) > baseline_samples else np.mean(waveform)
    wfm = waveform - baseline

    if method == 'peak':
        return _polarity_peak(wfm)
    elif method == 'first_peak':
        return _polarity_first_peak(wfm, sample_interval)
    elif method == 'integrated_charge':
        return _polarity_integrated_charge(wfm, sample_interval)
    elif method == 'energy_weighted':
        return _polarity_energy_weighted(wfm)
    elif method == 'dominant_half_cycle':
        return _polarity_dominant_half_cycle(wfm, sample_interval)
    elif method == 'initial_slope':
        return _polarity_initial_slope(wfm)
    else:
        raise ValueError(f"Unknown polarity method: {method}. Available: {POLARITY_METHODS}")


def _polarity_peak(wfm):
    """
    Peak method: Polarity based on which peak has the larger absolute amplitude.

    This is the simplest method - returns positive if the positive peak is larger
    than the absolute value of the negative peak.

    Pros:
    - Simple and fast
    - Works well for unipolar pulses

    Cons:
    - May be confused by noise or oscillations
    - Doesn't consider temporal information
    """
    peak_pos = np.max(wfm)
    peak_neg = np.min(wfm)

    if abs(peak_pos) >= abs(peak_neg):
        return 1
    else:
        return -1


def _polarity_first_peak(wfm, sample_interval):
    """
    First Peak method: Polarity based on which significant peak occurs first.

    Finds the first peak that exceeds a threshold (20% of max absolute value)
    and uses its sign to determine polarity.

    Pros:
    - Better for bipolar pulses where both peaks are similar in magnitude
    - Captures the initial discharge direction

    Cons:
    - May be affected by noise before the main pulse
    - Requires threshold tuning
    """
    abs_wfm = np.abs(wfm)
    max_abs = np.max(abs_wfm)

    if max_abs == 0:
        return 1

    # Threshold at 20% of maximum
    threshold = 0.2 * max_abs

    # Find all local maxima (peaks)
    pos_peaks, _ = signal.find_peaks(wfm, height=threshold)
    neg_peaks, _ = signal.find_peaks(-wfm, height=threshold)

    # Find the first significant peak
    first_pos_idx = pos_peaks[0] if len(pos_peaks) > 0 else float('inf')
    first_neg_idx = neg_peaks[0] if len(neg_peaks) > 0 else float('inf')

    if first_pos_idx <= first_neg_idx:
        return 1
    else:
        return -1


def _polarity_integrated_charge(wfm, sample_interval):
    """
    Integrated Charge method: Polarity based on the sign of total integrated charge.

    Integrates the waveform (signed area under curve) and returns positive if
    the net charge is positive.

    Pros:
    - Considers the entire waveform shape
    - Robust to small oscillations if main pulse dominates

    Cons:
    - May return near-zero for symmetric bipolar pulses
    - Heavily influenced by pulse duration
    """
    # Signed integral of the waveform
    charge = np.sum(wfm) * sample_interval

    if charge >= 0:
        return 1
    else:
        return -1


def _polarity_energy_weighted(wfm):
    """
    Energy-Weighted method: Polarity based on energy-weighted signal centroid.

    Calculates separate energies for positive and negative portions of the signal,
    and returns positive if the positive energy dominates.

    Pros:
    - Energy-based, so larger amplitude portions have more influence
    - Less sensitive to small oscillations

    Cons:
    - May not capture the "true" polarity for symmetric pulses
    """
    # Separate positive and negative portions
    positive_mask = wfm > 0
    negative_mask = wfm < 0

    # Calculate energy in each portion
    energy_positive = np.sum(wfm[positive_mask]**2) if np.any(positive_mask) else 0
    energy_negative = np.sum(wfm[negative_mask]**2) if np.any(negative_mask) else 0

    if energy_positive >= energy_negative:
        return 1
    else:
        return -1


def _polarity_dominant_half_cycle(wfm, sample_interval):
    """
    Dominant Half-Cycle method: Polarity based on which half-cycle contains more energy.

    Identifies zero crossings, calculates energy in each half-cycle, and
    returns polarity based on which half-cycle (positive or negative) has the most energy.

    Pros:
    - Good for oscillatory signals
    - Identifies the dominant oscillation direction

    Cons:
    - May be sensitive to zero-crossing detection
    - Computationally more expensive
    """
    # Find zero crossings
    zero_crossings = np.where(np.diff(np.signbit(wfm)))[0]

    if len(zero_crossings) == 0:
        # No zero crossings - use peak method
        return _polarity_peak(wfm)

    # Add boundaries
    boundaries = np.concatenate([[0], zero_crossings + 1, [len(wfm)]])

    # Calculate energy in each segment and sum by polarity
    positive_energy = 0
    negative_energy = 0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment = wfm[start:end]

        if len(segment) == 0:
            continue

        segment_energy = np.sum(segment**2)

        # Determine segment polarity by its mean or first significant value
        if np.mean(segment) > 0:
            positive_energy += segment_energy
        else:
            negative_energy += segment_energy

    if positive_energy >= negative_energy:
        return 1
    else:
        return -1


def _polarity_initial_slope(wfm):
    """
    Initial Slope method: Polarity based on the initial slope direction.

    Finds when the pulse starts (signal exceeds noise floor) and determines
    polarity based on whether it's initially rising or falling.

    Pros:
    - Captures the initial discharge direction
    - Good for fast rise-time pulses

    Cons:
    - Sensitive to noise at pulse onset
    - May fail for slowly-developing pulses
    """
    abs_wfm = np.abs(wfm)
    max_abs = np.max(abs_wfm)

    if max_abs == 0:
        return 1

    # Find noise floor from first samples
    noise_floor = np.std(wfm[:50]) if len(wfm) > 50 else np.std(wfm[:10])
    threshold = max(3 * noise_floor, 0.05 * max_abs)  # 3-sigma or 5% of max

    # Find the first point that exceeds threshold
    exceeds_threshold = np.where(abs_wfm > threshold)[0]

    if len(exceeds_threshold) == 0:
        return _polarity_peak(wfm)

    start_idx = exceeds_threshold[0]

    # Look at a window after the start to determine initial direction
    window_size = min(20, len(wfm) - start_idx)

    if window_size < 3:
        return _polarity_peak(wfm)

    # Calculate slope over the initial window
    window = wfm[start_idx:start_idx + window_size]
    initial_slope = np.mean(np.diff(window))

    # Also check the sign of the first significant value
    first_value = wfm[start_idx]

    # Combine slope and first value information
    # If first value and slope agree, use that
    if (first_value > 0 and initial_slope > 0) or (first_value < 0 and initial_slope < 0):
        return 1 if first_value > 0 else -1

    # If they disagree, use the first value sign
    if abs(first_value) > threshold:
        return 1 if first_value > 0 else -1

    # Fallback to slope direction
    return 1 if initial_slope > 0 else -1


def get_method_description(method):
    """
    Get a human-readable description of a polarity method.

    Args:
        method: Name of the polarity method

    Returns:
        str: Description of the method
    """
    descriptions = {
        'peak': 'Peak: Uses the larger absolute peak value',
        'first_peak': 'First Peak: Uses whichever peak occurs first in time',
        'integrated_charge': 'Integrated Charge: Based on net signed area under curve',
        'energy_weighted': 'Energy-Weighted: Based on energy in positive vs negative portions',
        'dominant_half_cycle': 'Dominant Half-Cycle: Based on which half-cycle has more energy',
        'initial_slope': 'Initial Slope: Based on the initial direction of the pulse',
    }
    return descriptions.get(method, f'Unknown method: {method}')


def compare_methods(waveform, sample_interval=4e-9, baseline_samples=50):
    """
    Compare all polarity methods for a given waveform.

    Args:
        waveform: numpy array of waveform samples
        sample_interval: Time between samples in seconds
        baseline_samples: Number of samples to use for baseline correction

    Returns:
        dict: Method names mapped to their polarity results
    """
    results = {}
    for method in POLARITY_METHODS:
        results[method] = calculate_polarity(
            waveform, method=method,
            sample_interval=sample_interval,
            baseline_samples=baseline_samples
        )
    return results


if __name__ == '__main__':
    # Test with a simple synthetic waveform
    print("Polarity Methods Test")
    print("=" * 50)

    # Create test waveforms
    t = np.linspace(0, 1e-6, 250)  # 1 microsecond, 250 samples

    # Test 1: Positive unipolar pulse
    wfm_pos = np.exp(-((t - 0.3e-6)**2) / (0.05e-6)**2)
    print("\nTest 1: Positive Unipolar Pulse")
    results = compare_methods(wfm_pos)
    for method, polarity in results.items():
        print(f"  {method}: {'+' if polarity > 0 else '-'}")

    # Test 2: Negative unipolar pulse
    wfm_neg = -np.exp(-((t - 0.3e-6)**2) / (0.05e-6)**2)
    print("\nTest 2: Negative Unipolar Pulse")
    results = compare_methods(wfm_neg)
    for method, polarity in results.items():
        print(f"  {method}: {'+' if polarity > 0 else '-'}")

    # Test 3: Bipolar pulse (positive first)
    wfm_bipolar_pos = np.exp(-((t - 0.3e-6)**2) / (0.05e-6)**2) - 0.8 * np.exp(-((t - 0.5e-6)**2) / (0.05e-6)**2)
    print("\nTest 3: Bipolar Pulse (positive first)")
    results = compare_methods(wfm_bipolar_pos)
    for method, polarity in results.items():
        print(f"  {method}: {'+' if polarity > 0 else '-'}")

    # Test 4: Bipolar pulse (negative first)
    wfm_bipolar_neg = -np.exp(-((t - 0.3e-6)**2) / (0.05e-6)**2) + 0.8 * np.exp(-((t - 0.5e-6)**2) / (0.05e-6)**2)
    print("\nTest 4: Bipolar Pulse (negative first)")
    results = compare_methods(wfm_bipolar_neg)
    for method, polarity in results.items():
        print(f"  {method}: {'+' if polarity > 0 else '-'}")

    print("\n" + "=" * 50)
    print("Test complete!")
