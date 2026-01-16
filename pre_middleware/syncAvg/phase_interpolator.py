"""
Phase Interpolator for Reference Signal

Extracts instantaneous phase angle (0-360 degrees) for every sample
in a raw data stream using the reference signal (50/60 Hz sine wave).

Methods:
1. Zero-crossing detection with linear interpolation
2. Hilbert transform for instantaneous phase
3. Sine wave fitting (for clean reference signals)

Usage:
    interpolator = PhaseInterpolator(ac_frequency=60.0)
    phases = interpolator.interpolate(reference_signal, sample_rate)
    # phases[i] = phase angle in degrees for sample i
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Literal
from dataclasses import dataclass
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d


@dataclass
class PhaseResult:
    """Result of phase interpolation."""
    phases: np.ndarray          # Phase angle for each sample (0-360 degrees)
    zero_crossings: np.ndarray  # Indices of positive zero crossings
    estimated_frequency: float  # Estimated AC frequency from data
    cycles_detected: int        # Number of complete AC cycles
    method: str                 # Method used for interpolation
    quality: float              # Quality metric (0-1, higher is better)


class PhaseInterpolator:
    """
    Interpolate phase angles from reference signal.

    The reference signal is typically a 50/60 Hz sine wave captured
    alongside the PD sensor signal for phase synchronization.

    Attributes:
        ac_frequency: Expected AC frequency (50 or 60 Hz)
        method: Interpolation method ('zero_crossing', 'hilbert', 'fit')
    """

    def __init__(
        self,
        ac_frequency: float = 60.0,
        method: Literal['zero_crossing', 'hilbert', 'fit'] = 'zero_crossing',
    ):
        """
        Initialize phase interpolator.

        Args:
            ac_frequency: Expected AC frequency in Hz (default: 60)
            method: Phase extraction method
                   'zero_crossing': Detect zero crossings and interpolate
                   'hilbert': Use Hilbert transform for instantaneous phase
                   'fit': Fit sine wave to reference
        """
        self.ac_frequency = ac_frequency
        self.method = method

    def interpolate(
        self,
        reference: np.ndarray,
        sample_rate: float,
        remove_dc: bool = True,
    ) -> PhaseResult:
        """
        Extract phase angle for every sample.

        Args:
            reference: Reference signal (50/60 Hz sine wave)
            sample_rate: Sample rate in Hz
            remove_dc: Remove DC offset before processing

        Returns:
            PhaseResult with phases array and metadata
        """
        # Remove DC offset
        if remove_dc:
            ref = reference - np.mean(reference)
        else:
            ref = reference.copy()

        # Normalize amplitude
        ref = ref / (np.max(np.abs(ref)) + 1e-10)

        if self.method == 'zero_crossing':
            return self._interpolate_zero_crossing(ref, sample_rate)
        elif self.method == 'hilbert':
            return self._interpolate_hilbert(ref, sample_rate)
        elif self.method == 'fit':
            return self._interpolate_fit(ref, sample_rate)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _interpolate_zero_crossing(
        self,
        ref: np.ndarray,
        sample_rate: float,
    ) -> PhaseResult:
        """
        Interpolate phase using zero-crossing detection.

        Most robust method for typical reference signals.
        Detects positive-going zero crossings and linearly
        interpolates phase between them.
        """
        n_samples = len(ref)

        # Find positive-going zero crossings
        # Sign changes from negative to positive
        sign = np.sign(ref)
        sign[sign == 0] = 1  # Treat zero as positive
        crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]

        if len(crossings) < 2:
            # Fallback to Hilbert if not enough crossings
            return self._interpolate_hilbert(ref, sample_rate)

        # Refine crossing positions using linear interpolation
        refined_crossings = []
        for idx in crossings:
            # Linear interpolation between samples
            y0, y1 = ref[idx], ref[idx + 1]
            frac = -y0 / (y1 - y0)  # Fractional position
            refined_crossings.append(idx + frac)

        refined_crossings = np.array(refined_crossings)

        # Estimate actual frequency from crossing intervals
        if len(refined_crossings) >= 2:
            periods = np.diff(refined_crossings) / sample_rate
            estimated_freq = 1.0 / np.median(periods)
        else:
            estimated_freq = self.ac_frequency

        # Build phase array by linear interpolation between crossings
        phases = np.zeros(n_samples, dtype=np.float64)

        # Before first crossing: extrapolate backward
        if refined_crossings[0] > 0:
            period_samples = refined_crossings[1] - refined_crossings[0] if len(refined_crossings) > 1 else sample_rate / estimated_freq
            for i in range(int(refined_crossings[0]) + 1):
                phases[i] = (360.0 * (i - refined_crossings[0]) / period_samples) % 360.0

        # Between crossings: linear interpolation
        for j in range(len(refined_crossings) - 1):
            start_idx = int(np.floor(refined_crossings[j]))
            end_idx = int(np.ceil(refined_crossings[j + 1]))
            period = refined_crossings[j + 1] - refined_crossings[j]

            for i in range(start_idx, min(end_idx + 1, n_samples)):
                frac = (i - refined_crossings[j]) / period
                phases[i] = (360.0 * frac) % 360.0

        # After last crossing: extrapolate forward
        last_crossing = refined_crossings[-1]
        if len(refined_crossings) > 1:
            period_samples = refined_crossings[-1] - refined_crossings[-2]
        else:
            period_samples = sample_rate / estimated_freq

        for i in range(int(np.floor(last_crossing)), n_samples):
            frac = (i - last_crossing) / period_samples
            phases[i] = (360.0 * frac) % 360.0

        # Quality metric based on frequency consistency
        if len(refined_crossings) >= 3:
            periods = np.diff(refined_crossings) / sample_rate
            freq_variance = np.std(periods) / np.mean(periods)
            quality = max(0, 1 - freq_variance * 10)  # Penalize variance
        else:
            quality = 0.5

        return PhaseResult(
            phases=phases,
            zero_crossings=crossings,
            estimated_frequency=estimated_freq,
            cycles_detected=len(crossings),
            method='zero_crossing',
            quality=quality,
        )

    def _interpolate_hilbert(
        self,
        ref: np.ndarray,
        sample_rate: float,
    ) -> PhaseResult:
        """
        Extract instantaneous phase using Hilbert transform.

        Works well for clean signals but can be noisy.
        """
        # Apply Hilbert transform
        analytic = scipy_signal.hilbert(ref)
        instantaneous_phase = np.angle(analytic)  # -pi to pi

        # Convert to degrees (0-360)
        phases = np.degrees(instantaneous_phase) % 360.0

        # Estimate frequency from phase derivative
        phase_diff = np.diff(np.unwrap(instantaneous_phase))
        freq_per_sample = phase_diff / (2 * np.pi)
        estimated_freq = np.median(freq_per_sample) * sample_rate

        # Find zero crossings for cycle count
        sign = np.sign(ref)
        sign[sign == 0] = 1
        crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]

        # Quality based on frequency stability
        freq_variance = np.std(freq_per_sample) / (np.mean(np.abs(freq_per_sample)) + 1e-10)
        quality = max(0, 1 - freq_variance)

        return PhaseResult(
            phases=phases,
            zero_crossings=crossings,
            estimated_frequency=abs(estimated_freq),
            cycles_detected=len(crossings),
            method='hilbert',
            quality=quality,
        )

    def _interpolate_fit(
        self,
        ref: np.ndarray,
        sample_rate: float,
    ) -> PhaseResult:
        """
        Extract phase by fitting a sine wave to the reference.

        Best for very clean reference signals.
        """
        from scipy.optimize import curve_fit

        n_samples = len(ref)
        t = np.arange(n_samples) / sample_rate

        # Initial guess
        amp_guess = np.std(ref) * np.sqrt(2)
        freq_guess = self.ac_frequency
        phase_guess = 0.0
        offset_guess = np.mean(ref)

        def sine_model(t, amp, freq, phase, offset):
            return amp * np.sin(2 * np.pi * freq * t + phase) + offset

        try:
            # Fit sine wave
            popt, pcov = curve_fit(
                sine_model, t, ref,
                p0=[amp_guess, freq_guess, phase_guess, offset_guess],
                bounds=(
                    [0, freq_guess * 0.9, -np.pi, -np.inf],
                    [np.inf, freq_guess * 1.1, np.pi, np.inf]
                ),
                maxfev=5000
            )
            amp, freq, phase_offset, offset = popt

            # Compute phase for each sample
            phases = np.degrees(2 * np.pi * freq * t + phase_offset) % 360.0

            # Quality from fit residuals
            fitted = sine_model(t, *popt)
            residual = np.sqrt(np.mean((ref - fitted) ** 2))
            quality = max(0, 1 - residual / amp)

            # Find zero crossings
            sign = np.sign(fitted)
            sign[sign == 0] = 1
            crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]

            return PhaseResult(
                phases=phases,
                zero_crossings=crossings,
                estimated_frequency=freq,
                cycles_detected=len(crossings),
                method='fit',
                quality=quality,
            )

        except Exception:
            # Fallback to zero crossing method
            return self._interpolate_zero_crossing(ref, sample_rate)

    def get_phase_at_index(
        self,
        phases: np.ndarray,
        index: int,
    ) -> float:
        """Get phase angle at a specific sample index."""
        if 0 <= index < len(phases):
            return phases[index]
        else:
            raise IndexError(f"Index {index} out of range for phases array of length {len(phases)}")

    def get_indices_at_phase(
        self,
        phases: np.ndarray,
        target_phase: float,
        tolerance: float = 1.0,
    ) -> np.ndarray:
        """
        Find all sample indices at a specific phase angle.

        Args:
            phases: Phase array from interpolate()
            target_phase: Target phase in degrees (0-360)
            tolerance: Phase tolerance in degrees

        Returns:
            Array of sample indices near target phase
        """
        target_phase = target_phase % 360.0

        # Handle wraparound at 0/360
        if target_phase < tolerance:
            mask = (phases <= target_phase + tolerance) | (phases >= 360 - tolerance + target_phase)
        elif target_phase > 360 - tolerance:
            mask = (phases >= target_phase - tolerance) | (phases <= target_phase + tolerance - 360)
        else:
            mask = np.abs(phases - target_phase) <= tolerance

        return np.where(mask)[0]


def interpolate_phase(
    reference: np.ndarray,
    sample_rate: float,
    ac_frequency: float = 60.0,
    method: str = 'zero_crossing',
) -> PhaseResult:
    """
    Convenience function to interpolate phase.

    Args:
        reference: Reference signal (50/60 Hz sine wave)
        sample_rate: Sample rate in Hz
        ac_frequency: Expected AC frequency
        method: Interpolation method

    Returns:
        PhaseResult with phases array and metadata
    """
    interpolator = PhaseInterpolator(ac_frequency, method)
    return interpolator.interpolate(reference, sample_rate)
