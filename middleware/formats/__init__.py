"""
middleware.formats - Format-Specific Data Loaders

Provides loaders for different PD data formats:
- RuggedLoader: Rugged format (-WFMs.txt, -SG.txt files)
- TUDelftLoader: TU Delft/Tektronix binary WFM format (future)
- GenericLoader: CSV/generic formats (future)

Usage:
    from middleware.formats import AutoLoader, RuggedLoader

    # Auto-detect format and load
    loader = AutoLoader('Rugged Data Files')
    data = loader.load('dataset_name')

    # Or use specific loader
    rugged = RuggedLoader('Rugged Data Files')
    waveforms = rugged.load_waveforms('dataset_name')
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
]
