"""
DWT-based PD Event Detector

Uses Discrete Wavelet Transform to detect PD events at multiple frequency
scales with independent thresholding per band.

Algorithm:
1. Compute DWT coefficients for D1, D2, D3 (and optionally D4, D5)
2. Calculate independent threshold for each band using 99.5th percentile
3. Find indices where each band exceeds its threshold
4. Convert indices to original sample positions (D1 x2, D2 x4, D3 x8)
5. Merge nearby detections from all bands
6. Tag each event with originating band for classification context

Band mapping at 250 MSPS (Nyquist = 125 MHz):
    D1: 31.25 - 62.5 MHz  (fast transients)
    D2: 15.63 - 31.25 MHz (medium speed)
    D3: 7.81 - 15.63 MHz  (slower events)
    D4: 3.91 - 7.81 MHz   (optional, slow)
    D5: 1.95 - 3.91 MHz   (optional, very slow)

At 125 MSPS (Nyquist = 62.5 MHz):
    D1: 15.63 - 31.25 MHz
    D2: 7.81 - 15.63 MHz
    D3: 3.91 - 7.81 MHz
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Literal
from dataclasses import dataclass, field

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


# Band-specific window sizes (samples at 250 MSPS)
# Pre-samples, Post-samples, Total window
BAND_WINDOWS = {
    'D1': {'pre': 50, 'post': 200, 'total': 250, 'duration_us': 1.0},
    'D2': {'pre': 100, 'post': 400, 'total': 500, 'duration_us': 2.0},
    'D3': {'pre': 250, 'post': 1000, 'total': 1250, 'duration_us': 5.0},
    'D4': {'pre': 500, 'post': 2000, 'total': 2500, 'duration_us': 10.0},
    'D5': {'pre': 1000, 'post': 4000, 'total': 5000, 'duration_us': 20.0},
}

# Band characteristics for classification hints
BAND_CHARACTERISTICS = {
    'D1': {
        'likely_types': ['internal_void', 'corona'],
        'description': 'Fast rise, short duration',
        'freq_range_250msps': (31.25, 62.5),  # MHz
        'freq_range_125msps': (15.63, 31.25),
    },
    'D2': {
        'likely_types': ['corona', 'surface'],
        'description': 'Medium characteristics',
        'freq_range_250msps': (15.63, 31.25),
        'freq_range_125msps': (7.81, 15.63),
    },
    'D3': {
        'likely_types': ['surface', 'tracking'],
        'description': 'Slow rise, longer duration',
        'freq_range_250msps': (7.81, 15.63),
        'freq_range_125msps': (3.91, 7.81),
    },
    'D4': {
        'likely_types': ['surface', 'tracking', 'interference'],
        'description': 'Very slow, may include interference',
        'freq_range_250msps': (3.91, 7.81),
        'freq_range_125msps': (1.95, 3.91),
    },
    'D5': {
        'likely_types': ['tracking', 'interference'],
        'description': 'Slowest, often interference',
        'freq_range_250msps': (1.95, 3.91),
        'freq_range_125msps': (0.98, 1.95),
    },
}


@dataclass
class DetectionEvent:
    """A single detected PD event."""
    sample_index: int            # Location in original signal
    phase_degrees: float         # Phase angle within AC cycle (0-360)
    band: str                    # Which wavelet band triggered (D1/D2/D3/etc)
    amplitude: float             # Wavelet coefficient amplitude at detection
    threshold: float             # Threshold that was exceeded
    significance: float          # How many sigma above threshold
    source: str = 'wavelet'      # Detection source identifier

    # Additional fields populated during extraction
    waveform: Optional[np.ndarray] = None
    pre_samples: int = 0
    post_samples: int = 0


@dataclass
class DetectionResult:
    """Result of wavelet-based detection."""
    events: List[DetectionEvent]          # All detected events
    band_stats: Dict[str, Dict[str, Any]] # Statistics per band
    sample_rate: float                     # Sample rate used
    wavelet: str                           # Wavelet used
    num_levels: int                        # DWT decomposition levels
    thresholds: Dict[str, float]           # Threshold per band
    total_samples: int                     # Total samples processed


class DWTDetector:
    """
    Wavelet-based PD event detector.

    Uses multi-level DWT decomposition to detect events at different
    frequency scales with independent thresholding per band.

    Attributes:
        sample_rate: Sample rate in Hz
        wavelet: Wavelet to use (default: 'db4')
        bands: Which bands to analyze (default: ['D1', 'D2', 'D3'])
        threshold_percentile: Percentile for threshold (default: 99.5)
        min_separation: Minimum samples between events
    """

    def __init__(
        self,
        sample_rate: float = 250e6,
        wavelet: str = 'db4',
        bands: Optional[List[str]] = None,
        threshold_percentile: float = 99.5,
        min_separation_us: float = 1.0,
    ):
        """
        Initialize DWT detector.

        Args:
            sample_rate: Sample rate in Hz (default: 250 MHz)
            wavelet: PyWavelets wavelet name (default: 'db4')
            bands: Bands to analyze (default: ['D1', 'D2', 'D3'])
            threshold_percentile: Percentile for threshold (default: 99.5)
            min_separation_us: Minimum separation between events in microseconds
        """
        if not PYWT_AVAILABLE:
            raise ImportError("pywt is required for wavelet detection. Install with: pip install PyWavelets")

        self.sample_rate = sample_rate
        self.wavelet = wavelet
        self.bands = bands or ['D1', 'D2', 'D3']
        self.threshold_percentile = threshold_percentile
        self.min_separation = int(min_separation_us * sample_rate / 1e6)

        # Map band names to DWT levels
        self._band_to_level = {'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4, 'D5': 5}
        self._level_to_band = {v: k for k, v in self._band_to_level.items()}

        # Calculate max level needed
        self._max_level = max(self._band_to_level[b] for b in self.bands)

    def detect(
        self,
        signal: np.ndarray,
        phases: Optional[np.ndarray] = None,
        ac_frequency: float = 60.0,
    ) -> DetectionResult:
        """
        Detect PD events using wavelet decomposition.

        Args:
            signal: Raw signal data
            phases: Optional pre-computed phase angles (0-360 degrees)
            ac_frequency: AC frequency for phase calculation if phases not provided

        Returns:
            DetectionResult with detected events and statistics
        """
        n_samples = len(signal)

        # Compute phases if not provided
        if phases is None:
            # Simple phase calculation based on sample position
            ac_period_samples = self.sample_rate / ac_frequency
            sample_indices = np.arange(n_samples)
            phases = (sample_indices % ac_period_samples) / ac_period_samples * 360.0

        # Perform DWT decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, level=self._max_level)
        # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
        # D1 is the last element, D2 is second to last, etc.

        # Collect events from each band
        all_events = []
        band_stats = {}
        thresholds = {}

        for band in self.bands:
            level = self._band_to_level[band]
            # D1 is at index -1, D2 at -2, etc.
            coeff_idx = -level
            detail_coeffs = coeffs[coeff_idx]

            # Calculate threshold using percentile of absolute values
            abs_coeffs = np.abs(detail_coeffs)
            threshold = np.percentile(abs_coeffs, self.threshold_percentile)
            thresholds[band] = threshold

            # Find indices exceeding threshold
            above_threshold = abs_coeffs > threshold

            # Get indices of threshold crossings
            detection_indices = np.where(above_threshold)[0]

            # Convert coefficient indices to original sample positions
            # D1 coefficients are downsampled by 2, D2 by 4, D3 by 8, etc.
            scale_factor = 2 ** level
            sample_indices = detection_indices * scale_factor + scale_factor // 2

            # Filter out-of-bounds indices
            sample_indices = sample_indices[sample_indices < n_samples]

            # Store band statistics
            band_stats[band] = {
                'num_detections': len(detection_indices),
                'threshold': threshold,
                'coeff_mean': np.mean(abs_coeffs),
                'coeff_std': np.std(abs_coeffs),
                'coeff_max': np.max(abs_coeffs),
                'scale_factor': scale_factor,
            }

            # Create events
            for i, det_idx in enumerate(detection_indices):
                sample_idx = det_idx * scale_factor + scale_factor // 2
                if sample_idx >= n_samples:
                    continue

                amplitude = abs_coeffs[det_idx]
                significance = (amplitude - threshold) / (band_stats[band]['coeff_std'] + 1e-10)

                event = DetectionEvent(
                    sample_index=int(sample_idx),
                    phase_degrees=float(phases[sample_idx]),
                    band=band,
                    amplitude=float(amplitude),
                    threshold=float(threshold),
                    significance=float(significance),
                    source='wavelet',
                )
                all_events.append(event)

        # Sort events by sample index
        all_events.sort(key=lambda e: e.sample_index)

        # Merge nearby detections
        merged_events = self._merge_nearby_events(all_events)

        return DetectionResult(
            events=merged_events,
            band_stats=band_stats,
            sample_rate=self.sample_rate,
            wavelet=self.wavelet,
            num_levels=self._max_level,
            thresholds=thresholds,
            total_samples=n_samples,
        )

    def _merge_nearby_events(
        self,
        events: List[DetectionEvent],
    ) -> List[DetectionEvent]:
        """
        Merge nearby detections, keeping the most significant one.

        When multiple bands detect the same event, keep the detection
        from the band with highest significance.
        """
        if len(events) <= 1:
            return events

        merged = []
        current_group = [events[0]]

        for event in events[1:]:
            # Check if this event is close to the current group
            if event.sample_index - current_group[-1].sample_index <= self.min_separation:
                current_group.append(event)
            else:
                # Process current group - keep most significant
                best = max(current_group, key=lambda e: e.significance)
                merged.append(best)
                current_group = [event]

        # Don't forget the last group
        if current_group:
            best = max(current_group, key=lambda e: e.significance)
            merged.append(best)

        return merged

    def get_band_frequency_range(self, band: str) -> Tuple[float, float]:
        """
        Get frequency range for a band at current sample rate.

        Returns:
            (low_freq_mhz, high_freq_mhz)
        """
        level = self._band_to_level[band]
        nyquist = self.sample_rate / 2e6  # MHz

        high_freq = nyquist / (2 ** (level - 1))
        low_freq = nyquist / (2 ** level)

        return (low_freq, high_freq)

    def get_window_params(
        self,
        band: str,
        sample_rate: Optional[float] = None,
    ) -> Dict[str, int]:
        """
        Get window parameters for a band, adjusted for sample rate.

        The default windows are specified for 250 MSPS. This method
        scales them for other sample rates.

        Args:
            band: Band name (D1, D2, D3, etc.)
            sample_rate: Sample rate (uses detector's rate if not specified)

        Returns:
            Dict with 'pre', 'post', 'total' sample counts
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        base = BAND_WINDOWS[band]
        scale = sample_rate / 250e6  # Scale relative to 250 MSPS

        return {
            'pre': int(base['pre'] * scale),
            'post': int(base['post'] * scale),
            'total': int(base['total'] * scale),
            'duration_us': base['duration_us'],
        }

    def detect_single_cycle(
        self,
        signal: np.ndarray,
        cycle_start: int,
        cycle_end: int,
        phases: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """
        Detect events in a single AC cycle.

        Args:
            signal: Full signal data
            cycle_start: Start sample index of cycle
            cycle_end: End sample index of cycle
            phases: Optional pre-computed phases

        Returns:
            DetectionResult for this cycle only
        """
        cycle_signal = signal[cycle_start:cycle_end]

        if phases is not None:
            cycle_phases = phases[cycle_start:cycle_end]
        else:
            cycle_phases = None

        result = self.detect(cycle_signal, cycle_phases)

        # Adjust sample indices to be relative to full signal
        for event in result.events:
            event.sample_index += cycle_start

        return result


def detect_with_wavelets(
    signal: np.ndarray,
    sample_rate: float = 250e6,
    phases: Optional[np.ndarray] = None,
    bands: Optional[List[str]] = None,
    threshold_percentile: float = 99.5,
) -> DetectionResult:
    """
    Convenience function for wavelet-based detection.

    Args:
        signal: Raw signal data
        sample_rate: Sample rate in Hz
        phases: Optional phase angles
        bands: Bands to analyze (default: D1, D2, D3)
        threshold_percentile: Threshold percentile

    Returns:
        DetectionResult
    """
    detector = DWTDetector(
        sample_rate=sample_rate,
        bands=bands,
        threshold_percentile=threshold_percentile,
    )
    return detector.detect(signal, phases)
