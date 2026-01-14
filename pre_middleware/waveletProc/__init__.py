"""
waveletProc - Wavelet-based PD Detection

Uses Discrete Wavelet Transform (DWT) to detect PD events at multiple
frequency scales (D1, D2, D3, optionally D4, D5) with independent
thresholding per band.

Key Features:
- Multi-band detection (D1=fast PD, D2=medium, D3=slow/surface)
- Independent 99.5th percentile thresholding per band
- Sample rate adaptive (125 MSPS, 250 MSPS, etc.)
- Band-specific waveform extraction windows
- Event tagging with originating band for classification
- SmartBounds adaptive windowing (default: 2x SNR threshold, 0.5µs minimum)

Band Characteristics:
    Band    Frequency Range     Likely PD Type          Window (250 MSPS)
    D1      31-62 MHz           Internal void, corona   250 samples (1 us)
    D2      16-31 MHz           Corona, some surface    500 samples (2 us)
    D3      8-16 MHz            Surface, tracking       1250 samples (5 us)

Usage:
    from pre_middleware.waveletProc import DWTDetector, WaveletExtractor

    # Detect events
    detector = DWTDetector(sample_rate=250e6)
    events = detector.detect(signal, phases)

    # Extract waveforms with SmartBounds adaptive windowing (default)
    extractor = WaveletExtractor(sample_rate=250e6)
    waveforms = extractor.extract(signal, events)

    # Or disable adaptive windowing
    extractor = WaveletExtractor(sample_rate=250e6, adaptive_window=False)
"""

from .dwt_detector import (
    DWTDetector,
    DetectionEvent,
    DetectionResult,
    BAND_WINDOWS,
    BAND_CHARACTERISTICS,
    # Wavelet options
    WAVELETS,
    DEFAULT_WAVELET,
    list_wavelets,
    get_wavelet_info,
    # Kurtosis pre-check
    KurtosisResult,
    QuadrantKurtosisResult,
    compute_kurtosis,
    check_kurtosis,
    compute_kurtosis_per_cycle,
    compute_kurtosis_per_quadrant,
    check_kurtosis_per_quadrant,
)
from .waveform_extractor import (
    WaveletExtractor,
    WaveletWaveform,
    ExtractionResult,
    smart_bounds,
    extract_wavelet_waveforms,
)

__all__ = [
    # Detection
    'DWTDetector',
    'DetectionEvent',
    'DetectionResult',
    'BAND_WINDOWS',
    'BAND_CHARACTERISTICS',
    # Wavelet options
    'WAVELETS',
    'DEFAULT_WAVELET',
    'list_wavelets',
    'get_wavelet_info',
    # Kurtosis pre-check
    'KurtosisResult',
    'QuadrantKurtosisResult',
    'compute_kurtosis',
    'check_kurtosis',
    'compute_kurtosis_per_cycle',
    'compute_kurtosis_per_quadrant',
    'check_kurtosis_per_quadrant',
    # Extraction
    'WaveletExtractor',
    'WaveletWaveform',
    'ExtractionResult',
    'smart_bounds',
    'extract_wavelet_waveforms',
]
