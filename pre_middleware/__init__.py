"""
pre_middleware - Raw Data Stream Processing

Converts continuous raw data streams (without triggers) into triggered
waveform format compatible with the PD analysis pipeline.

This module handles data that comes as a continuous acquisition stream
(e.g., IEEE .mat files) and needs processing to extract individual
PD pulse waveforms.

Submodules:
-----------
triggerProc : Trigger-based detection and waveform extraction
    Traditional threshold-based trigger detection with configurable
    methods (stdev, pulse_rate, histogram_knee).

syncAvg : Synchronous averaging for phase-locked analysis
    Phase-locked averaging across AC cycles to reveal consistent
    PD patterns buried in noise. Useful for finding hotspots.

waveletProc : Wavelet-based multi-band detection
    DWT-based detection at multiple frequency scales (D1, D2, D3)
    with band-specific windowing. Good for classifying PD types
    based on frequency content.

loaders : Data file loaders (MatLoader, etc.)
    Load various data formats including IEEE .mat files.

Usage Examples:
---------------

1. Trigger-based processing (traditional):
    from pre_middleware.triggerProc import TriggerDetector, WaveformExtractor
    from pre_middleware.loaders import MatLoader

    loader = MatLoader('data.mat')
    data = loader.load()

    detector = TriggerDetector(method='histogram_knee')
    triggers = detector.detect(data['signal'], sample_rate=data['sample_rate'])

    extractor = WaveformExtractor(pre_samples=62, post_samples=188)
    result = extractor.extract(data['signal'], triggers.triggers, data['sample_rate'])

2. Synchronous averaging:
    from pre_middleware.syncAvg import PhaseInterpolator, SyncAverager
    from pre_middleware.loaders import MatLoader

    loader = MatLoader('data.mat')
    data = loader.load(channel='Ch1', include_reference=True)

    interpolator = PhaseInterpolator(ac_frequency=60.0)
    phases = interpolator.interpolate(data['reference'], data['sample_rate'])

    averager = SyncAverager(num_bins=360)
    result = averager.compute(data['signal'], phases.phases)
    hotspots = averager.find_hotspots(result, threshold_sigma=3.0)

3. Wavelet-based detection:
    from pre_middleware.waveletProc import DWTDetector, WaveletExtractor
    from pre_middleware.loaders import MatLoader

    loader = MatLoader('data.mat')
    data = loader.load()

    detector = DWTDetector(sample_rate=data['sample_rate'], bands=['D1', 'D2', 'D3'])
    events = detector.detect(data['signal'])

    extractor = WaveletExtractor(sample_rate=data['sample_rate'])
    waveforms = extractor.extract(data['signal'], events)

CLI Usage:
----------
    # Trigger-based processing
    python -m pre_middleware.triggerProc.process_raw_stream data.mat -o output/

    # Compare trigger methods
    python -m pre_middleware.triggerProc.process_raw_stream data.mat --compare-methods
"""

# Re-export from triggerProc for backward compatibility
from .triggerProc import (
    TriggerDetector,
    TriggerResult,
    TRIGGER_METHODS,
    DEFAULT_TRIGGER_METHOD,
    compare_methods,
    WaveformExtractor,
    ExtractionResult,
    extract_waveforms,
    process_raw_stream,
)

__all__ = [
    # Trigger detection (triggerProc)
    'TriggerDetector',
    'TriggerResult',
    'TRIGGER_METHODS',
    'DEFAULT_TRIGGER_METHOD',
    'compare_methods',
    # Waveform extraction (triggerProc)
    'WaveformExtractor',
    'ExtractionResult',
    'extract_waveforms',
    # Processing (triggerProc)
    'process_raw_stream',
]
