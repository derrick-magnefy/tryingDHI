"""
middleware.formats - Format-Specific Data Loaders

Provides loaders for different PD data formats:
- RuggedLoader: Rugged format (-WFMs.txt, -SG.txt files)
- TektronixWFMParser: TU Delft/Tektronix binary WFM format
- PDN utilities: Binary PDN file decoder

Usage:
    from middleware.formats import AutoLoader, RuggedLoader

    # Auto-detect format and load
    loader = AutoLoader('Rugged Data Files')
    data = loader.load('dataset_name')

    # Or use specific loader
    rugged = RuggedLoader('Rugged Data Files')
    waveforms = rugged.load_waveforms('dataset_name')

    # Tektronix WFM parser
    from middleware.formats import TektronixWFMParser
    parser = TektronixWFMParser('file.wfm')
"""

from .base import BaseLoader, DatasetInfo
from .rugged import RuggedLoader, load_rugged_waveforms, load_rugged_settings
from .detection import (
    detect_format,
    get_loader,
    get_loader_for_format,
    list_datasets,
    get_dataset_info,
    AutoLoader,
)

# Tektronix/TU Delft format
try:
    from .tektronix import TektronixWFMParser, load_tu_delft_timing, convert_timing_to_phase
    TEKTRONIX_AVAILABLE = True
except ImportError:
    TEKTRONIX_AVAILABLE = False

__all__ = [
    # Base
    'BaseLoader',
    'DatasetInfo',
    # Rugged
    'RuggedLoader',
    'load_rugged_waveforms',
    'load_rugged_settings',
    # Detection
    'detect_format',
    'get_loader',
    'get_loader_for_format',
    'list_datasets',
    'get_dataset_info',
    'AutoLoader',
    # Tektronix
    'TektronixWFMParser',
    'TEKTRONIX_AVAILABLE',
]
