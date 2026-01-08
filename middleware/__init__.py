"""
middleware - Data Source Handling

This module handles ingestion of PD data from various formats:
- Auto-detection of data source formats
- Format-specific loaders (Rugged, TU Delft, etc.)
- Data validation and caching

This is tryingDHI-specific and not intended for reuse by other repositories.
Other repositories should use pdlib directly and implement their own data loading.

Usage:
    from middleware import AutoLoader

    loader = AutoLoader('Rugged Data Files')
    datasets = loader.list_datasets()
    data = loader.load('dataset_name')
"""

from .formats import (
    AutoLoader,
    RuggedLoader,
    BaseLoader,
    DatasetInfo,
    detect_format,
    get_loader,
    list_datasets,
)

__all__ = [
    'AutoLoader',
    'RuggedLoader',
    'BaseLoader',
    'DatasetInfo',
    'detect_format',
    'get_loader',
    'list_datasets',
]
