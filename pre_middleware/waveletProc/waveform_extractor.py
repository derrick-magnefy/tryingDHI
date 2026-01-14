"""
Wavelet-based Waveform Extractor

Extracts waveforms around wavelet-detected events with band-specific
window sizes. Different bands require different window lengths:

    Band    Pre-Samples    Post-Samples    Total (250 MSPS)    Duration
    D1      50             200             250                 1 us
    D2      100            400             500                 2 us
    D3      250            1000            1250                5 us
    D4      500            2000            2500                10 us
    D5      1000           4000            5000                20 us

Window sizes scale proportionally for different sample rates.

Output Format:
    Each extracted waveform includes:
    - sample_index: Location in original block
    - phase_degrees: Phase angle within AC cycle (0-360)
    - band: Which wavelet band triggered (D1/D2/D3)
    - waveform: Numpy array of sample values
    - source: 'wavelet' or 'sync_averaged'
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import os
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

from .dwt_detector import (
    DetectionEvent,
    DetectionResult,
    BAND_WINDOWS,
    BAND_CHARACTERISTICS,
)


# =============================================================================
# ADAPTIVE WINDOWING (SmartBounds)
# =============================================================================

def smart_bounds(
    waveform: np.ndarray,
    sample_interval: float,
    noise_window: int = 50,
    snr_threshold: float = 2.0,
    min_window_us: float = 0.5,
) -> Dict[str, Any]:
    """
    Smart boundary detection using envelope analysis and noise floor estimation.

    Finds true signal boundaries based on SNR, with a minimum window guarantee.
    This is the default adaptive windowing method: SmartBounds 2x+ (min 0.5µs).

    Args:
        waveform: Input waveform array
        sample_interval: Time between samples in seconds
        noise_window: Number of samples at edges to estimate noise floor
        snr_threshold: Multiplier above noise floor to define signal boundary
                      (2.0=aggressive, 3.0=default, 5.0=conservative)
        min_window_us: Minimum window size in microseconds

    Returns:
        Dict with 'waveform', 'start_idx', 'end_idx', 'name', 'description'
    """
    n = len(waveform)
    min_samples = int(min_window_us * 1e-6 / sample_interval)

    if n < noise_window * 2:
        # Waveform too short for noise estimation
        return {
            'waveform': waveform,
            'start_idx': 0,
            'end_idx': n,
            'name': f'SmartBounds {snr_threshold:.0f}x+ (min {min_window_us}µs)',
            'description': 'Waveform too short for adaptive windowing',
        }

    # Estimate noise floor from edges
    noise_start = waveform[:noise_window]
    noise_end = waveform[-noise_window:]
    noise_rms_start = np.sqrt(np.mean(noise_start ** 2))
    noise_rms_end = np.sqrt(np.mean(noise_end ** 2))
    noise_floor = min(noise_rms_start, noise_rms_end)

    if noise_floor < 1e-10:
        noise_floor = np.max(np.abs(waveform)) * 0.01

    # Compute envelope using Hilbert transform
    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)
    envelope_smooth = gaussian_filter1d(envelope, sigma=3)

    # Find signal threshold
    signal_threshold = snr_threshold * noise_floor
    above_threshold = envelope_smooth > signal_threshold

    # Find peak location
    peak_idx = np.argmax(envelope_smooth)

    if not np.any(above_threshold):
        # Low SNR - use minimum window around peak
        half_win = max(min_samples // 2, 25)
        start_idx = max(0, peak_idx - half_win)
        end_idx = min(n, peak_idx + half_win)
    else:
        # Find boundaries where envelope crosses threshold
        start_idx = 0
        for i in range(peak_idx, -1, -1):
            if envelope_smooth[i] < signal_threshold:
                start_idx = i
                break

        end_idx = n - 1
        for i in range(peak_idx, n):
            if envelope_smooth[i] < signal_threshold:
                # Check if signal stays low (not just a dip)
                remaining = envelope_smooth[i:min(i + 50, n)]
                if len(remaining) == 0 or np.max(remaining) < signal_threshold * 1.5:
                    end_idx = i
                    break

        # Add padding proportional to signal duration
        signal_duration = end_idx - start_idx
        padding = max(10, signal_duration // 10)
        start_idx = max(0, start_idx - padding)
        end_idx = min(n, end_idx + padding)

    # Ensure minimum window size
    current_samples = end_idx - start_idx
    if current_samples < min_samples:
        # Expand to minimum window centered on peak
        half_window = min_samples // 2
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(n, peak_idx + half_window)

    return {
        'waveform': waveform[start_idx:end_idx],
        'start_idx': start_idx,
        'end_idx': end_idx,
        'name': f'SmartBounds {snr_threshold:.0f}x+ (min {min_window_us}µs)',
        'description': f'Envelope-based bounds, {snr_threshold:.0f}x noise, min {min_window_us}µs',
    }


@dataclass
class WaveletWaveform:
    """A single extracted waveform with metadata."""
    sample_index: int            # Location in original block
    phase_degrees: float         # Phase angle within AC cycle (0-360)
    band: str                    # Which wavelet band triggered (D1/D2/D3)
    waveform: np.ndarray         # Numpy array of sample values
    source: str                  # 'wavelet' or 'sync_averaged'
    amplitude: float             # Peak amplitude of waveform
    significance: float          # Detection significance
    pre_samples: int             # Samples before trigger
    post_samples: int            # Samples after trigger
    sample_rate: float           # Sample rate for this waveform

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sample_index': self.sample_index,
            'phase_degrees': self.phase_degrees,
            'band': self.band,
            'waveform': self.waveform.tolist(),
            'source': self.source,
            'amplitude': self.amplitude,
            'significance': self.significance,
            'pre_samples': self.pre_samples,
            'post_samples': self.post_samples,
            'sample_rate': self.sample_rate,
        }


@dataclass
class ExtractionResult:
    """Result of waveform extraction."""
    waveforms: List[WaveletWaveform]
    stats: Dict[str, Any]
    sample_rate: float

    @property
    def num_waveforms(self) -> int:
        return len(self.waveforms)

    def get_by_band(self, band: str) -> List[WaveletWaveform]:
        """Get waveforms from a specific band."""
        return [w for w in self.waveforms if w.band == band]

    def get_waveform_matrix(self, band: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Get waveforms as a 2D matrix.

        Since different bands have different window sizes, this only works
        when filtering to a single band or when all waveforms have the
        same window size.

        Args:
            band: Optional band to filter (required if mixed bands)

        Returns:
            (waveform_matrix, sample_indices)
        """
        if band:
            wfms = self.get_by_band(band)
        else:
            wfms = self.waveforms

        if not wfms:
            return np.array([]), []

        # Check all have same length
        lengths = set(len(w.waveform) for w in wfms)
        if len(lengths) > 1:
            raise ValueError(
                f"Mixed waveform lengths: {lengths}. "
                "Filter by band or use same window size."
            )

        matrix = np.array([w.waveform for w in wfms])
        indices = [w.sample_index for w in wfms]

        return matrix, indices


class WaveletExtractor:
    """
    Extract waveforms around wavelet-detected events.

    Handles band-specific window sizes and scales appropriately
    for different sample rates.

    Attributes:
        sample_rate: Sample rate in Hz
        overlap_handling: How to handle overlapping windows
    """

    def __init__(
        self,
        sample_rate: float = 250e6,
        overlap_handling: str = 'skip',
        validate_extraction: bool = True,
        adaptive_window: bool = True,
        snr_threshold: float = 2.0,
        min_window_us: float = 0.5,
    ):
        """
        Initialize waveform extractor.

        Args:
            sample_rate: Sample rate in Hz (default: 250 MHz)
            overlap_handling: How to handle overlapping windows
                             'skip': Skip overlapping extractions
                             'allow': Allow overlaps
            validate_extraction: Validate peak is near trigger point
            adaptive_window: Apply SmartBounds adaptive windowing (default: True)
            snr_threshold: SNR threshold for SmartBounds (default: 2.0 = aggressive)
            min_window_us: Minimum window size in µs for SmartBounds (default: 0.5)
        """
        self.sample_rate = sample_rate
        self.overlap_handling = overlap_handling
        self.validate_extraction = validate_extraction
        self.adaptive_window = adaptive_window
        self.snr_threshold = snr_threshold
        self.min_window_us = min_window_us

        # Pre-compute scaled window parameters
        self._scaled_windows = {}
        for band, params in BAND_WINDOWS.items():
            scale = sample_rate / 250e6
            self._scaled_windows[band] = {
                'pre': int(params['pre'] * scale),
                'post': int(params['post'] * scale),
                'total': int(params['total'] * scale),
                'duration_us': params['duration_us'],
            }

    def extract(
        self,
        signal: np.ndarray,
        detection_result: DetectionResult,
        source: str = 'wavelet',
    ) -> ExtractionResult:
        """
        Extract waveforms around detected events.

        Args:
            signal: Raw signal data
            detection_result: DetectionResult from DWTDetector
            source: Source identifier for waveforms

        Returns:
            ExtractionResult with extracted waveforms
        """
        n_samples = len(signal)
        waveforms = []

        stats = {
            'total_events': len(detection_result.events),
            'extracted': 0,
            'skipped_bounds': 0,
            'skipped_overlap': 0,
            'by_band': {band: 0 for band in self._scaled_windows.keys()},
            'adaptive_window': self.adaptive_window,
            'snr_threshold': self.snr_threshold if self.adaptive_window else None,
            'min_window_us': self.min_window_us if self.adaptive_window else None,
        }

        last_end = -1  # Track end of last extraction for overlap detection

        for event in detection_result.events:
            band = event.band
            window = self._scaled_windows.get(band)

            if window is None:
                continue

            pre = window['pre']
            post = window['post']

            start_idx = event.sample_index - pre
            end_idx = event.sample_index + post

            # Check bounds
            if start_idx < 0 or end_idx > n_samples:
                stats['skipped_bounds'] += 1
                continue

            # Check overlap
            if self.overlap_handling == 'skip' and start_idx <= last_end:
                stats['skipped_overlap'] += 1
                continue

            # Extract waveform
            wfm_data = signal[start_idx:end_idx].copy()

            # Apply adaptive windowing if enabled
            sample_interval = 1.0 / self.sample_rate
            if self.adaptive_window:
                bounds_result = smart_bounds(
                    wfm_data,
                    sample_interval,
                    snr_threshold=self.snr_threshold,
                    min_window_us=self.min_window_us,
                )
                wfm_data = bounds_result['waveform']
                # Update pre/post samples based on adaptive window
                adaptive_pre = bounds_result['start_idx']
                adaptive_post = len(bounds_result['waveform']) - (pre - adaptive_pre)
                actual_pre = pre - bounds_result['start_idx']
                actual_post = bounds_result['end_idx'] - pre
            else:
                actual_pre = pre
                actual_post = post

            # Compute amplitude
            amplitude = float(np.max(np.abs(wfm_data)))

            wfm = WaveletWaveform(
                sample_index=event.sample_index,
                phase_degrees=event.phase_degrees,
                band=band,
                waveform=wfm_data,
                source=source,
                amplitude=amplitude,
                significance=event.significance,
                pre_samples=actual_pre,
                post_samples=actual_post,
                sample_rate=self.sample_rate,
            )
            waveforms.append(wfm)

            stats['extracted'] += 1
            stats['by_band'][band] = stats['by_band'].get(band, 0) + 1
            last_end = end_idx

        return ExtractionResult(
            waveforms=waveforms,
            stats=stats,
            sample_rate=self.sample_rate,
        )

    def extract_from_events(
        self,
        signal: np.ndarray,
        events: List[DetectionEvent],
        phases: Optional[np.ndarray] = None,
        source: str = 'wavelet',
    ) -> ExtractionResult:
        """
        Extract waveforms from a list of events.

        Args:
            signal: Raw signal data
            events: List of DetectionEvent
            phases: Optional phase array (updates event phases if provided)
            source: Source identifier

        Returns:
            ExtractionResult
        """
        # Update phases if provided
        if phases is not None:
            for event in events:
                if 0 <= event.sample_index < len(phases):
                    event.phase_degrees = phases[event.sample_index]

        # Create a dummy DetectionResult
        dummy_result = DetectionResult(
            events=events,
            band_stats={},
            sample_rate=self.sample_rate,
            wavelet='',
            num_levels=0,
            thresholds={},
            total_samples=len(signal),
        )

        return self.extract(signal, dummy_result, source)

    def save_rugged_format(
        self,
        result: ExtractionResult,
        output_dir: str,
        prefix: str,
        band: Optional[str] = None,
        ac_frequency: float = 60.0,
    ) -> Dict[str, str]:
        """
        Save extracted waveforms in Rugged Data Files format.

        Since different bands have different window sizes, you should
        either specify a band or ensure all waveforms have the same size.

        Args:
            result: ExtractionResult from extract()
            output_dir: Output directory
            prefix: File prefix
            band: Optional band filter (recommended)
            ac_frequency: AC frequency for metadata

        Returns:
            Dict mapping file type to file path
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get waveforms
        if band:
            wfms = result.get_by_band(band)
            file_prefix = f"{prefix}_{band}"
        else:
            wfms = result.waveforms
            file_prefix = prefix

        if not wfms:
            return {}

        # Check consistent window size
        lengths = set(len(w.waveform) for w in wfms)
        if len(lengths) > 1:
            raise ValueError(
                f"Cannot save mixed window sizes: {lengths}. "
                "Specify a band parameter."
            )

        wfm_length = lengths.pop()
        pre_samples = wfms[0].pre_samples
        post_samples = wfms[0].post_samples

        files = {}

        # Build arrays
        waveform_matrix = np.array([w.waveform for w in wfms])
        trigger_times = np.array([w.sample_index / self.sample_rate for w in wfms])
        phases = np.array([w.phase_degrees for w in wfms])
        amplitudes = np.array([w.amplitude for w in wfms])

        # Save waveforms (-WFMs.txt)
        wfm_path = os.path.join(output_dir, f"{file_prefix}-WFMs.txt")
        np.savetxt(wfm_path, waveform_matrix, delimiter='\t', fmt='%.10e')
        files['waveforms'] = wfm_path

        # Save trigger times (-Ti.txt)
        ti_path = os.path.join(output_dir, f"{file_prefix}-Ti.txt")
        np.savetxt(ti_path, trigger_times.reshape(1, -1), delimiter='\t', fmt='%.10e')
        files['trigger_times'] = ti_path

        # Save phases (-Ph.txt)
        ph_path = os.path.join(output_dir, f"{file_prefix}-Ph.txt")
        np.savetxt(ph_path, phases.reshape(1, -1), delimiter='\t', fmt='%.6f')
        files['phases'] = ph_path

        # Save amplitudes (-A.txt)
        a_path = os.path.join(output_dir, f"{file_prefix}-A.txt")
        np.savetxt(a_path, amplitudes.reshape(1, -1), delimiter='\t', fmt='%.10e')
        files['amplitudes'] = a_path

        # Save settings (-SG.txt)
        sg_path = os.path.join(output_dir, f"{file_prefix}-SG.txt")
        sample_interval = 1.0 / self.sample_rate
        sg_values = [
            1.0,                              # [0] voltage_scale
            trigger_times[-1] - trigger_times[0] if len(trigger_times) > 1 else 0,  # [1] acquisition_time
            len(wfms),                        # [2] num_waveforms
            0,                                # [3] mode
            0, 0, 0, 0, 0,                    # [4-8] reserved
            ac_frequency,                     # [9] ac_frequency
            sample_interval,                  # [10] sample_interval
            0, 0, 0, 0,                       # [11-14] reserved
            3,                                # [15] processing mode (wavelet)
            wfm_length,                       # [16] samples_per_waveform
            pre_samples,                      # [17] pre_trigger_samples
            post_samples,                     # [18] post_trigger_samples
        ]
        np.savetxt(sg_path, [sg_values], delimiter='\t', fmt='%.10e')
        files['settings'] = sg_path

        # Save band info (-Band.txt) - new file for wavelet-specific metadata
        band_path = os.path.join(output_dir, f"{file_prefix}-Band.txt")
        bands = [w.band for w in wfms]
        with open(band_path, 'w') as f:
            f.write('\t'.join(bands))
        files['bands'] = band_path

        return files

    def get_window_params(self, band: str) -> Dict[str, int]:
        """Get window parameters for a band."""
        return self._scaled_windows.get(band, self._scaled_windows['D1'])


def extract_wavelet_waveforms(
    signal: np.ndarray,
    detection_result: DetectionResult,
    sample_rate: float = 250e6,
    adaptive_window: bool = True,
    snr_threshold: float = 2.0,
    min_window_us: float = 0.5,
) -> ExtractionResult:
    """
    Convenience function for waveform extraction.

    By default, uses SmartBounds 2x+ (min 0.5µs) adaptive windowing to
    find true signal boundaries and reduce false multi-pulse detections.

    Args:
        signal: Raw signal data
        detection_result: DetectionResult from DWTDetector
        sample_rate: Sample rate in Hz
        adaptive_window: Apply SmartBounds adaptive windowing (default: True)
        snr_threshold: SNR threshold for boundary detection (default: 2.0)
        min_window_us: Minimum window size in µs (default: 0.5)

    Returns:
        ExtractionResult
    """
    extractor = WaveletExtractor(
        sample_rate=sample_rate,
        adaptive_window=adaptive_window,
        snr_threshold=snr_threshold,
        min_window_us=min_window_us,
    )
    return extractor.extract(signal, detection_result)
