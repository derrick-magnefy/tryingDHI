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

Wavelet Selection:
    Different wavelets have different characteristics for PD detection:
    - db4, db6, db8: Daubechies - good general purpose, asymmetric
    - sym4, sym6, sym8: Symlets - more symmetric than Daubechies
    - coif2, coif4: Coiflets - more vanishing moments, smoother
    - bior1.5, bior2.4: Biorthogonal - linear phase, good for localization
    - haar: Simplest, good for sharp transients
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Literal
from dataclasses import dataclass, field

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


# Recommended wavelets for PD detection with characteristics
WAVELETS = {
    # Daubechies family - asymmetric, good general purpose
    'db4': {
        'description': 'Daubechies 4 - Good balance of smoothness and localization',
        'best_for': ['general', 'fast_transients'],
        'filter_length': 8,
    },
    'db6': {
        'description': 'Daubechies 6 - Smoother than db4, slightly less localized',
        'best_for': ['general', 'medium_transients'],
        'filter_length': 12,
    },
    'db8': {
        'description': 'Daubechies 8 - Very smooth, good for slower events',
        'best_for': ['slow_transients', 'surface_pd'],
        'filter_length': 16,
    },
    # Symlets - more symmetric than Daubechies
    'sym4': {
        'description': 'Symlet 4 - Nearly symmetric version of db4',
        'best_for': ['general', 'symmetric_pulses'],
        'filter_length': 8,
    },
    'sym6': {
        'description': 'Symlet 6 - Good symmetry and smoothness',
        'best_for': ['general', 'symmetric_pulses'],
        'filter_length': 12,
    },
    'sym8': {
        'description': 'Symlet 8 - Very symmetric, good for clean signals',
        'best_for': ['slow_transients', 'clean_signals'],
        'filter_length': 16,
    },
    # Coiflets - more vanishing moments
    'coif2': {
        'description': 'Coiflet 2 - Good for signals with polynomial trends',
        'best_for': ['noisy_signals', 'trending_baseline'],
        'filter_length': 12,
    },
    'coif4': {
        'description': 'Coiflet 4 - Higher order, very smooth',
        'best_for': ['noisy_signals', 'slow_transients'],
        'filter_length': 24,
    },
    # Biorthogonal - linear phase
    'bior1.5': {
        'description': 'Biorthogonal 1.5 - Linear phase, good localization',
        'best_for': ['timing_critical', 'fast_transients'],
        'filter_length': 10,
    },
    'bior2.4': {
        'description': 'Biorthogonal 2.4 - Smooth with linear phase',
        'best_for': ['timing_critical', 'general'],
        'filter_length': 10,
    },
    # Haar - simplest
    'haar': {
        'description': 'Haar - Simplest wavelet, excellent for sharp edges',
        'best_for': ['sharp_transients', 'step_changes'],
        'filter_length': 2,
    },
}

# Default wavelet
DEFAULT_WAVELET = 'db4'


def list_wavelets() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of recommended wavelets with descriptions."""
    return WAVELETS.copy()


def get_wavelet_info(wavelet: str) -> Dict[str, Any]:
    """Get information about a specific wavelet."""
    if wavelet in WAVELETS:
        return WAVELETS[wavelet]
    else:
        # Try to get info from pywt for custom wavelets
        if PYWT_AVAILABLE:
            try:
                w = pywt.Wavelet(wavelet)
                return {
                    'description': f'{wavelet} - Custom wavelet',
                    'best_for': ['custom'],
                    'filter_length': w.dec_len,
                }
            except Exception:
                pass
        return {'description': f'{wavelet} - Unknown wavelet', 'best_for': [], 'filter_length': 0}


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


@dataclass
class KurtosisResult:
    """Result of kurtosis pre-check."""
    kurtosis: float              # Excess kurtosis of signal (0 = Gaussian)
    has_impulsive_content: bool  # Whether kurtosis exceeds threshold
    threshold_used: float        # Threshold that was used
    recommendation: str          # 'process', 'skip', or 'uncertain'


def compute_kurtosis(
    signal: np.ndarray,
    remove_dc: bool = True,
) -> float:
    """
    Compute excess kurtosis of signal.

    Excess kurtosis:
    - 0 = Gaussian (normal distribution)
    - > 0 = Heavy tails (leptokurtic) - indicates impulsive events
    - < 0 = Light tails (platykurtic)

    PD signals typically have high kurtosis (>> 3) due to impulsive spikes.

    Args:
        signal: Input signal
        remove_dc: Remove DC offset before computing

    Returns:
        Excess kurtosis value
    """
    if remove_dc:
        sig = signal - np.mean(signal)
    else:
        sig = signal

    n = len(sig)
    if n < 4:
        return 0.0

    # Compute moments
    mean = np.mean(sig)
    m2 = np.mean((sig - mean) ** 2)
    m4 = np.mean((sig - mean) ** 4)

    if m2 < 1e-20:
        return 0.0

    # Kurtosis = m4 / m2^2, excess kurtosis = kurtosis - 3
    kurtosis = m4 / (m2 ** 2) - 3.0

    return float(kurtosis)


def check_kurtosis(
    signal: np.ndarray,
    threshold: float = 5.0,
    remove_dc: bool = True,
) -> KurtosisResult:
    """
    Quick kurtosis check to determine if signal has impulsive content.

    Use this before running expensive wavelet decomposition to skip
    blocks that are unlikely to contain PD events.

    Typical kurtosis values:
    - Gaussian noise: ~0 (excess kurtosis)
    - Signal with rare spikes: > 10
    - Active PD signal: > 20-50
    - Very impulsive PD: > 100

    Args:
        signal: Input signal
        threshold: Excess kurtosis threshold (default: 5.0)
        remove_dc: Remove DC offset before computing

    Returns:
        KurtosisResult with recommendation
    """
    kurt = compute_kurtosis(signal, remove_dc)

    if kurt >= threshold * 2:
        recommendation = 'process'  # Definitely has impulsive content
    elif kurt >= threshold:
        recommendation = 'uncertain'  # Borderline, might have weak events
    else:
        recommendation = 'skip'  # Likely just noise

    return KurtosisResult(
        kurtosis=kurt,
        has_impulsive_content=kurt >= threshold,
        threshold_used=threshold,
        recommendation=recommendation,
    )


def compute_kurtosis_per_cycle(
    signal: np.ndarray,
    sample_rate: float,
    ac_frequency: float = 60.0,
    remove_dc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kurtosis for each AC cycle.

    Useful for identifying which cycles have impulsive activity.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        ac_frequency: AC frequency in Hz
        remove_dc: Remove DC offset before computing

    Returns:
        (kurtosis_values, cycle_start_indices)
    """
    samples_per_cycle = int(sample_rate / ac_frequency)
    n_cycles = len(signal) // samples_per_cycle

    kurtosis_values = []
    cycle_starts = []

    for i in range(n_cycles):
        start = i * samples_per_cycle
        end = start + samples_per_cycle
        cycle = signal[start:end]

        kurt = compute_kurtosis(cycle, remove_dc)
        kurtosis_values.append(kurt)
        cycle_starts.append(start)

    return np.array(kurtosis_values), np.array(cycle_starts)


@dataclass
class QuadrantKurtosisResult:
    """Result of per-quadrant kurtosis analysis."""
    quadrant_kurtosis: Dict[str, float]   # Kurtosis per quadrant (Q1, Q2, Q3, Q4)
    halfcycle_kurtosis: Dict[str, float]  # Kurtosis per half-cycle (positive, negative)
    overall_kurtosis: float               # Overall signal kurtosis
    max_quadrant: str                     # Quadrant with highest kurtosis
    max_kurtosis: float                   # Highest quadrant kurtosis value
    active_quadrants: List[str]           # Quadrants exceeding threshold
    recommendation: str                   # 'process', 'skip', or 'uncertain'


def compute_kurtosis_per_quadrant(
    signal: np.ndarray,
    phases: np.ndarray,
    threshold: float = 5.0,
    remove_dc: bool = True,
) -> QuadrantKurtosisResult:
    """
    Compute kurtosis for each phase quadrant.

    Useful for identifying phase regions with impulsive activity,
    even when overall kurtosis is low due to dilution by quiet phases.

    Quadrants:
        Q1: 0-90° (positive half-cycle, rising)
        Q2: 90-180° (positive half-cycle, falling)
        Q3: 180-270° (negative half-cycle, rising)
        Q4: 270-360° (negative half-cycle, falling)

    Args:
        signal: Input signal
        phases: Phase angle for each sample (0-360 degrees)
        threshold: Kurtosis threshold for "active" classification
        remove_dc: Remove DC offset before computing

    Returns:
        QuadrantKurtosisResult with per-quadrant analysis
    """
    if len(signal) != len(phases):
        raise ValueError("Signal and phases must have same length")

    # Define quadrants
    quadrants = {
        'Q1': (0, 90),
        'Q2': (90, 180),
        'Q3': (180, 270),
        'Q4': (270, 360),
    }

    # Define half-cycles
    halfcycles = {
        'positive': (0, 180),
        'negative': (180, 360),
    }

    # Compute kurtosis for each quadrant
    quadrant_kurtosis = {}
    for name, (start, end) in quadrants.items():
        mask = (phases >= start) & (phases < end)
        if np.sum(mask) > 4:  # Need at least 4 samples
            quadrant_kurtosis[name] = compute_kurtosis(signal[mask], remove_dc)
        else:
            quadrant_kurtosis[name] = 0.0

    # Compute kurtosis for each half-cycle
    halfcycle_kurtosis = {}
    for name, (start, end) in halfcycles.items():
        mask = (phases >= start) & (phases < end)
        if np.sum(mask) > 4:
            halfcycle_kurtosis[name] = compute_kurtosis(signal[mask], remove_dc)
        else:
            halfcycle_kurtosis[name] = 0.0

    # Overall kurtosis
    overall_kurtosis = compute_kurtosis(signal, remove_dc)

    # Find max quadrant
    max_quadrant = max(quadrant_kurtosis, key=quadrant_kurtosis.get)
    max_kurtosis = quadrant_kurtosis[max_quadrant]

    # Find active quadrants (above threshold)
    active_quadrants = [q for q, k in quadrant_kurtosis.items() if k >= threshold]

    # Recommendation based on max quadrant kurtosis
    if max_kurtosis >= threshold * 2:
        recommendation = 'process'
    elif max_kurtosis >= threshold:
        recommendation = 'uncertain'
    else:
        recommendation = 'skip'

    return QuadrantKurtosisResult(
        quadrant_kurtosis=quadrant_kurtosis,
        halfcycle_kurtosis=halfcycle_kurtosis,
        overall_kurtosis=overall_kurtosis,
        max_quadrant=max_quadrant,
        max_kurtosis=max_kurtosis,
        active_quadrants=active_quadrants,
        recommendation=recommendation,
    )


def check_kurtosis_per_quadrant(
    signal: np.ndarray,
    sample_rate: float,
    ac_frequency: float = 60.0,
    reference: Optional[np.ndarray] = None,
    threshold: float = 5.0,
) -> QuadrantKurtosisResult:
    """
    Convenience function to compute per-quadrant kurtosis.

    If reference signal is provided, uses zero-crossings for accurate
    phase alignment. Otherwise, assumes phase starts at sample 0.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        ac_frequency: AC frequency in Hz
        reference: Optional reference signal (50/60 Hz) for phase alignment
        threshold: Kurtosis threshold

    Returns:
        QuadrantKurtosisResult
    """
    n_samples = len(signal)
    samples_per_cycle = sample_rate / ac_frequency

    if reference is not None:
        # Find zero crossings for phase alignment
        ref = reference - np.mean(reference)
        sign = np.sign(ref)
        sign[sign == 0] = 1
        crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]
        phase_start = crossings[0] if len(crossings) > 0 else 0
    else:
        phase_start = 0

    # Compute phase for each sample
    sample_indices = np.arange(n_samples)
    phases = ((sample_indices - phase_start) % samples_per_cycle) / samples_per_cycle * 360.0

    return compute_kurtosis_per_quadrant(signal, phases, threshold)


class DWTDetector:
    """
    Wavelet-based PD event detector.

    Uses multi-level DWT decomposition to detect events at different
    frequency scales with independent thresholding per band.

    Attributes:
        sample_rate: Sample rate in Hz
        wavelet: Wavelet to use (default: 'db4', see WAVELETS for options)
        bands: Which bands to analyze (default: ['D1', 'D2', 'D3'])
        threshold_percentile: Percentile for threshold (default: 99.5)
        min_separation: Minimum samples between events
    """

    def __init__(
        self,
        sample_rate: float = 250e6,
        wavelet: str = DEFAULT_WAVELET,
        bands: Optional[List[str]] = None,
        threshold_percentile: float = 99.5,
        min_separation_us: float = 1.0,
    ):
        """
        Initialize DWT detector.

        Args:
            sample_rate: Sample rate in Hz (default: 250 MHz)
            wavelet: PyWavelets wavelet name (default: 'db4')
                    Options: db4, db6, db8, sym4, sym6, sym8, coif2, coif4,
                            bior1.5, bior2.4, haar (see WAVELETS dict)
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

        # Validate wavelet
        try:
            pywt.Wavelet(wavelet)
        except Exception as e:
            raise ValueError(f"Invalid wavelet '{wavelet}': {e}. See WAVELETS for options.")

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
