"""
triggerProc - Trigger-based Raw Data Processing

Converts continuous raw data streams (without triggers) into triggered
waveform format using threshold-based trigger detection.

This module handles data that comes as a continuous acquisition stream
(e.g., IEEE .mat files) and needs trigger detection to extract individual
PD pulse waveforms.

Key Features:
- Multiple trigger detection methods (stdev, pulse_rate, histogram_knee)
- Configurable pre/post trigger sample windows
- Outputs Rugged-compatible format for downstream processing

Usage:
    from pre_middleware.triggerProc import TriggerDetector, WaveformExtractor
    from pre_middleware.loaders import MatLoader

    # Load raw data
    loader = MatLoader('IEEE Data/dataset.mat')
    data = loader.load()

    # Detect triggers
    detector = TriggerDetector(method='histogram_knee')
    triggers = detector.detect(data['signal'], sample_rate=data['sample_rate'])

    # Extract waveforms (2us @ 125 MSPS, 25% pre-trigger = 250 samples)
    extractor = WaveformExtractor(pre_samples=62, post_samples=188)
    waveforms = extractor.extract(data['signal'], triggers.triggers)

CLI Usage:
    # Process a .mat file with default settings
    python -m pre_middleware.triggerProc.process_raw_stream data.mat -o output/

    # Compare trigger methods
    python -m pre_middleware.triggerProc.process_raw_stream data.mat --compare-methods
"""

from .trigger_detection import (
    TriggerDetector,
    TriggerResult,
    TRIGGER_METHODS,
    DEFAULT_TRIGGER_METHOD,
    compare_methods,
)
from .waveform_extraction import (
    WaveformExtractor,
    ExtractionResult,
    extract_waveforms,
)
from .process_raw_stream import process_raw_stream

__all__ = [
    # Trigger detection
    'TriggerDetector',
    'TriggerResult',
    'TRIGGER_METHODS',
    'DEFAULT_TRIGGER_METHOD',
    'compare_methods',
    # Waveform extraction
    'WaveformExtractor',
    'ExtractionResult',
    'extract_waveforms',
    # Processing
    'process_raw_stream',
]
