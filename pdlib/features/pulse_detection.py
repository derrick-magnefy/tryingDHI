"""
Pulse Detection Module

Detects multiple distinct PD pulses within a single waveform.
"""

import numpy as np
from scipy.signal import find_peaks


def detect_pulses(waveform, sample_interval, amplitude_threshold_ratio=0.3, min_pulse_separation_us=0.5):
    """
    Detect multiple distinct PD pulses within a single waveform.

    A pulse is defined as a significant peak (above threshold) that is separated
    from other pulses by a minimum time gap.

    Args:
        waveform: numpy array of waveform samples (baseline-corrected)
        sample_interval: time between samples in seconds
        amplitude_threshold_ratio: minimum peak height as ratio of max amplitude (default: 0.3 = 30%)
        min_pulse_separation_us: minimum separation between pulses in microseconds (default: 0.5 µs)

    Returns:
        dict with:
            - pulse_count: number of distinct pulses detected
            - pulse_indices: list of peak indices for each pulse
            - pulse_amplitudes: list of amplitudes for each pulse
    """
    # Get absolute waveform for peak detection
    abs_wfm = np.abs(waveform)
    max_amplitude = np.max(abs_wfm)

    if max_amplitude == 0:
        return {'pulse_count': 0, 'pulse_indices': [], 'pulse_amplitudes': []}

    # Minimum height threshold for peaks
    height_threshold = amplitude_threshold_ratio * max_amplitude

    # Minimum distance between peaks (in samples)
    min_distance_samples = int(min_pulse_separation_us * 1e-6 / sample_interval)
    min_distance_samples = max(1, min_distance_samples)  # At least 1 sample

    # Find peaks in absolute waveform
    peaks, properties = find_peaks(
        abs_wfm,
        height=height_threshold,
        distance=min_distance_samples,
        prominence=height_threshold * 0.5  # Peaks should be prominent
    )

    # Get amplitudes at peak locations (use original waveform for signed amplitude)
    pulse_amplitudes = [waveform[idx] for idx in peaks]

    return {
        'pulse_count': len(peaks),
        'pulse_indices': peaks.tolist(),
        'pulse_amplitudes': pulse_amplitudes
    }


def is_multi_pulse(waveform, sample_interval, threshold_ratio=0.3, min_separation_us=0.5):
    """
    Check if a waveform contains multiple distinct pulses.

    Args:
        waveform: numpy array of waveform samples
        sample_interval: time between samples in seconds
        threshold_ratio: minimum peak height as ratio of max amplitude
        min_separation_us: minimum separation between pulses in microseconds

    Returns:
        bool: True if multiple pulses detected
    """
    result = detect_pulses(waveform, sample_interval, threshold_ratio, min_separation_us)
    return result['pulse_count'] > 1
