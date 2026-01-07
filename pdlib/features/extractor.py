"""
PD Feature Extractor

Main class for extracting features from partial discharge pulse waveforms.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Any

from .polarity import calculate_polarity, DEFAULT_POLARITY_METHOD
from .pulse_detection import detect_pulses
from .definitions import FEATURE_NAMES, ADC_STEP_V


class PDFeatureExtractor:
    """
    Extract features from partial discharge pulse waveforms.

    This class provides comprehensive feature extraction including:
    - Time-domain features (amplitude, timing, shape)
    - Frequency-domain features (spectral characteristics)
    - Energy-based features (cumulative energy analysis)
    - Multi-pulse detection

    Usage:
        extractor = PDFeatureExtractor(sample_interval=4e-9)
        features = extractor.extract_features(waveform, phase_angle=45.0)

        # Or extract from multiple waveforms
        all_features = extractor.extract_all(waveforms, phase_angles)
    """

    def __init__(self, sample_interval: float = 4e-9, ac_frequency: float = 60.0,
                 polarity_method: str = DEFAULT_POLARITY_METHOD):
        """
        Initialize the feature extractor.

        Args:
            sample_interval: Time between samples in seconds (default: 4ns = 250 MHz)
            ac_frequency: AC line frequency in Hz (default: 60Hz)
            polarity_method: Method for polarity calculation (default: 'first_peak')
                            Options: peak, first_peak, integrated_charge,
                                     energy_weighted, dominant_half_cycle, initial_slope
        """
        self.sample_interval = sample_interval
        self.ac_frequency = ac_frequency
        self.sample_rate = 1.0 / sample_interval
        self.polarity_method = polarity_method

    def extract_features(self, waveform: np.ndarray, phase_angle: Optional[float] = None) -> Dict[str, float]:
        """
        Extract all features from a single waveform.

        Args:
            waveform: numpy array of waveform samples
            phase_angle: Optional phase angle from timing data (degrees)

        Returns:
            dict: Dictionary of feature names to values
        """
        features = {}

        # Remove DC offset using baseline from first samples
        baseline = np.mean(waveform[:50]) if len(waveform) > 50 else np.mean(waveform)
        wfm = waveform - baseline

        # Time array
        n_samples = len(wfm)
        time = np.arange(n_samples) * self.sample_interval

        # === AMPLITUDE FEATURES ===
        features['phase_angle'] = phase_angle if phase_angle is not None else 0.0
        features['peak_amplitude_positive'] = float(np.max(wfm))
        features['peak_amplitude_negative'] = float(np.min(wfm))
        features['absolute_amplitude'] = max(abs(features['peak_amplitude_positive']),
                                             abs(features['peak_amplitude_negative']))
        features['peak_to_peak_amplitude'] = features['peak_amplitude_positive'] - features['peak_amplitude_negative']

        # Polarity: calculated using the configured polarity method
        features['polarity'] = float(calculate_polarity(
            waveform,
            method=self.polarity_method,
            sample_interval=self.sample_interval
        ))

        # RMS amplitude
        features['rms_amplitude'] = float(np.sqrt(np.mean(wfm**2)))

        # Crest factor (peak / RMS)
        if features['rms_amplitude'] > 0:
            peak = max(abs(features['peak_amplitude_positive']), abs(features['peak_amplitude_negative']))
            features['crest_factor'] = peak / features['rms_amplitude']
        else:
            features['crest_factor'] = 0.0

        # === TIME-DOMAIN FEATURES ===
        timing = self._extract_timing_features(wfm, time)
        features.update(timing)

        # === ENERGY FEATURES ===
        energy = self._extract_energy_features(wfm, time)
        features.update(energy)

        # === FREQUENCY-DOMAIN FEATURES ===
        spectral = self._extract_spectral_features(wfm)
        features.update(spectral)

        # === MULTI-PULSE DETECTION ===
        pulse_info = detect_pulses(wfm, self.sample_interval)
        features['pulse_count'] = float(pulse_info['pulse_count'])
        features['is_multi_pulse'] = 1.0 if pulse_info['pulse_count'] > 1 else 0.0

        return features

    def _extract_timing_features(self, wfm: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """Extract timing-related features."""
        features = {}

        # Find the main peak (largest absolute value)
        abs_wfm = np.abs(wfm)
        peak_idx = np.argmax(abs_wfm)
        peak_value = wfm[peak_idx]

        # Determine thresholds for rise/fall time (10% to 90% of peak)
        threshold_low = 0.1 * abs(peak_value)
        threshold_high = 0.9 * abs(peak_value)

        # Find rise time (time from 10% to 90% of peak on rising edge)
        rise_start_idx = None
        rise_end_idx = None

        # Search backwards from peak for rising edge
        for i in range(peak_idx, -1, -1):
            if abs(wfm[i]) <= threshold_low:
                rise_start_idx = i
                break
            if abs(wfm[i]) <= threshold_high and rise_end_idx is None:
                rise_end_idx = i

        if rise_start_idx is not None and rise_end_idx is not None and rise_end_idx > rise_start_idx:
            features['rise_time'] = (rise_end_idx - rise_start_idx) * self.sample_interval
        else:
            features['rise_time'] = 0.0

        # Find fall time (time from 90% to 10% of peak on falling edge)
        fall_start_idx = None
        fall_end_idx = None

        # Search forward from peak for falling edge
        for i in range(peak_idx, len(wfm)):
            if fall_start_idx is None and abs(wfm[i]) <= threshold_high:
                fall_start_idx = i
            if abs(wfm[i]) <= threshold_low:
                fall_end_idx = i
                break

        if fall_start_idx is not None and fall_end_idx is not None and fall_end_idx > fall_start_idx:
            features['fall_time'] = (fall_end_idx - fall_start_idx) * self.sample_interval
        else:
            features['fall_time'] = 0.0

        # Pulse width (time above 50% of peak)
        threshold_50 = 0.5 * abs(peak_value)
        above_50 = np.where(abs_wfm >= threshold_50)[0]
        if len(above_50) > 0:
            features['pulse_width'] = (above_50[-1] - above_50[0]) * self.sample_interval
        else:
            features['pulse_width'] = 0.0

        # Slew rate (maximum rate of change)
        diff_wfm = np.diff(wfm) / self.sample_interval
        features['slew_rate'] = float(np.max(np.abs(diff_wfm)))

        # Rise/fall ratio
        if features['fall_time'] > 0:
            features['rise_fall_ratio'] = features['rise_time'] / features['fall_time']
        else:
            features['rise_fall_ratio'] = 0.0

        # Zero crossing count
        zero_crossings = np.where(np.diff(np.signbit(wfm)))[0]
        features['zero_crossing_count'] = float(len(zero_crossings))

        # Oscillation count (number of local maxima/minima)
        local_max = signal.argrelmax(wfm, order=3)[0]
        local_min = signal.argrelmin(wfm, order=3)[0]
        features['oscillation_count'] = float(len(local_max) + len(local_min))

        return features

    def _extract_energy_features(self, wfm: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """Extract energy-related features."""
        features = {}

        # Total energy (integral of squared signal)
        features['energy'] = float(np.sum(wfm**2) * self.sample_interval)

        # Charge (integral of absolute signal - apparent charge proxy)
        features['charge'] = float(np.sum(np.abs(wfm)) * self.sample_interval)

        # Energy/Charge ratio
        if features['charge'] > 0:
            features['energy_charge_ratio'] = features['energy'] / features['charge']
        else:
            features['energy_charge_ratio'] = 0.0

        # Equivalent time (energy-weighted time centroid)
        if features['energy'] > 0:
            features['equivalent_time'] = float(np.sum(time * wfm**2) * self.sample_interval / features['energy'])
        else:
            features['equivalent_time'] = 0.0

        # Equivalent bandwidth (related to signal spread in time)
        if features['energy'] > 0:
            time_spread = np.sum((time - features['equivalent_time'])**2 * wfm**2) * self.sample_interval
            features['equivalent_bandwidth'] = float(np.sqrt(time_spread / features['energy']))
        else:
            features['equivalent_bandwidth'] = 0.0

        # Cumulative energy analysis
        cumulative_energy = np.cumsum(wfm**2) * self.sample_interval
        total_energy = cumulative_energy[-1] if len(cumulative_energy) > 0 else 0

        if total_energy > 0:
            # Normalized cumulative energy
            norm_cum_energy = cumulative_energy / total_energy

            # Cumulative energy at peak
            peak_idx = np.argmax(np.abs(wfm))
            features['cumulative_energy_peak'] = float(norm_cum_energy[peak_idx])

            # Time to reach 10% to 90% of total energy (rise time in energy domain)
            idx_10 = np.searchsorted(norm_cum_energy, 0.1)
            idx_90 = np.searchsorted(norm_cum_energy, 0.9)
            features['cumulative_energy_rise_time'] = (idx_90 - idx_10) * self.sample_interval

            # Shape factor: ratio of energy in first half vs second half
            mid_idx = len(wfm) // 2
            energy_first_half = cumulative_energy[mid_idx]
            energy_second_half = total_energy - energy_first_half
            if energy_second_half > 0:
                features['cumulative_energy_shape_factor'] = float(energy_first_half / energy_second_half)
            else:
                features['cumulative_energy_shape_factor'] = 0.0

            # Area ratio: ratio of positive to negative areas
            positive_energy = np.sum(wfm[wfm > 0]**2) * self.sample_interval
            negative_energy = np.sum(wfm[wfm < 0]**2) * self.sample_interval
            if negative_energy > 0:
                features['cumulative_energy_area_ratio'] = float(positive_energy / negative_energy)
            else:
                features['cumulative_energy_area_ratio'] = float('inf') if positive_energy > 0 else 1.0
        else:
            features['cumulative_energy_peak'] = 0.0
            features['cumulative_energy_rise_time'] = 0.0
            features['cumulative_energy_shape_factor'] = 0.0
            features['cumulative_energy_area_ratio'] = 1.0

        return features

    def _extract_spectral_features(self, wfm: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features."""
        features = {}

        n_samples = len(wfm)

        # Apply window to reduce spectral leakage
        window = signal.windows.hann(n_samples)
        windowed_wfm = wfm * window

        # Compute FFT
        fft_vals = fft(windowed_wfm)
        freqs = fftfreq(n_samples, self.sample_interval)

        # Use only positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        # Power spectrum
        power_spectrum = pos_fft**2
        total_power = np.sum(power_spectrum)

        if total_power > 0 and len(pos_freqs) > 0:
            # Dominant frequency (frequency with maximum power)
            features['dominant_frequency'] = float(pos_freqs[np.argmax(power_spectrum)])

            # Center frequency (power-weighted centroid)
            features['center_frequency'] = float(np.sum(pos_freqs * power_spectrum) / total_power)

            # 3dB bandwidth
            max_power = np.max(power_spectrum)
            half_power = max_power / 2
            above_half = pos_freqs[power_spectrum >= half_power]
            if len(above_half) > 1:
                features['bandwidth_3db'] = float(above_half[-1] - above_half[0])
            else:
                features['bandwidth_3db'] = 0.0

            # Spectral power in low and high bands
            # Define low band as 0-25% of Nyquist, high band as 75-100% of Nyquist
            nyquist = self.sample_rate / 2
            low_band_mask = pos_freqs < (0.25 * nyquist)
            high_band_mask = pos_freqs > (0.75 * nyquist)

            features['spectral_power_low'] = float(np.sum(power_spectrum[low_band_mask]) / total_power)
            features['spectral_power_high'] = float(np.sum(power_spectrum[high_band_mask]) / total_power)

            # Spectral flatness (geometric mean / arithmetic mean)
            # High flatness = noise-like, low flatness = tonal
            geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-12)))
            arithmetic_mean = np.mean(power_spectrum)
            if arithmetic_mean > 0:
                features['spectral_flatness'] = float(geometric_mean / arithmetic_mean)
            else:
                features['spectral_flatness'] = 0.0

            # Spectral entropy
            # Normalized power spectrum as probability distribution
            p = power_spectrum / total_power
            p = p[p > 0]  # Remove zeros for log
            features['spectral_entropy'] = float(-np.sum(p * np.log2(p)))

        else:
            features['dominant_frequency'] = 0.0
            features['center_frequency'] = 0.0
            features['bandwidth_3db'] = 0.0
            features['spectral_power_low'] = 0.0
            features['spectral_power_high'] = 0.0
            features['spectral_flatness'] = 0.0
            features['spectral_entropy'] = 0.0

        return features

    def extract_all(self, waveforms: List[np.ndarray],
                    phase_angles: Optional[List[float]] = None,
                    normalize: bool = True) -> List[Dict[str, float]]:
        """
        Extract features from multiple waveforms.

        Args:
            waveforms: List of waveform arrays
            phase_angles: Optional list of phase angles (same length as waveforms)
            normalize: Whether to compute normalized features (requires all waveforms)

        Returns:
            List of feature dictionaries
        """
        all_features = []

        for i, wfm in enumerate(waveforms):
            phase = phase_angles[i] if phase_angles is not None and i < len(phase_angles) else None
            features = self.extract_features(wfm, phase_angle=phase)
            all_features.append(features)

        if normalize and len(all_features) > 0:
            all_features = self._compute_normalized_features(all_features, waveforms)

        return all_features

    def _compute_normalized_features(self, all_features: List[Dict[str, float]],
                                     waveforms: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Compute normalized (scale-independent) features.

        These features are normalized by noise floor, duration, and other context.
        """
        # Calculate noise floor from minimum absolute amplitude across all pulses
        abs_amplitudes = np.array([f['absolute_amplitude'] for f in all_features])
        min_amplitude = np.min(abs_amplitudes)
        noise_floor = max(ADC_STEP_V, min_amplitude - ADC_STEP_V)  # At least one ADC step

        # Get normalization constants
        n_samples = len(waveforms[0]) if waveforms else 1
        waveform_duration = n_samples * self.sample_interval
        nyquist_freq = self.sample_rate / 2

        # Compute normalized features for each pulse
        for features in all_features:
            # Signal-to-noise ratio
            features['signal_to_noise_ratio'] = features['absolute_amplitude'] / noise_floor

            # Normalized amplitude features (by noise_floor)
            features['norm_absolute_amplitude'] = features['absolute_amplitude'] / noise_floor
            features['norm_peak_amplitude_positive'] = features['peak_amplitude_positive'] / noise_floor
            features['norm_peak_amplitude_negative'] = features['peak_amplitude_negative'] / noise_floor
            features['norm_peak_to_peak_amplitude'] = features['peak_to_peak_amplitude'] / noise_floor
            features['norm_rms_amplitude'] = features['rms_amplitude'] / noise_floor

            # Normalized slew rate
            slew_rate_ref = noise_floor / self.sample_interval
            features['norm_slew_rate'] = features['slew_rate'] / slew_rate_ref if slew_rate_ref > 0 else 0.0

            # Normalized energy and charge
            energy_ref = (noise_floor ** 2) * waveform_duration
            charge_ref = noise_floor * waveform_duration
            features['norm_energy'] = features['energy'] / energy_ref if energy_ref > 0 else 0.0
            features['norm_charge'] = features['charge'] / charge_ref if charge_ref > 0 else 0.0

            # Normalized timing features
            pulse_width = features.get('pulse_width', waveform_duration)
            if pulse_width > 0:
                features['norm_rise_time'] = features['rise_time'] / pulse_width
                features['norm_fall_time'] = features['fall_time'] / pulse_width
                features['norm_cumulative_energy_rise_time'] = features['cumulative_energy_rise_time'] / pulse_width
            else:
                features['norm_rise_time'] = 0.0
                features['norm_fall_time'] = 0.0
                features['norm_cumulative_energy_rise_time'] = 0.0

            features['norm_equivalent_time'] = features['equivalent_time'] / waveform_duration
            features['norm_equivalent_bandwidth'] = features['equivalent_bandwidth'] / waveform_duration
            features['norm_pulse_width'] = features.get('pulse_width', 0.0) / waveform_duration

            # Normalized frequency features
            features['norm_dominant_frequency'] = features['dominant_frequency'] / nyquist_freq
            features['norm_center_frequency'] = features['center_frequency'] / nyquist_freq
            features['norm_bandwidth_3db'] = features['bandwidth_3db'] / nyquist_freq

            # Normalized rates (per sample)
            features['norm_zero_crossing_rate'] = features['zero_crossing_count'] / n_samples
            features['norm_oscillation_rate'] = features['oscillation_count'] / n_samples

        return all_features

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all feature names in order."""
        return FEATURE_NAMES.copy()
