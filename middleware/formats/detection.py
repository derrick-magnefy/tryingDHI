"""
Format detection and auto-loader selection.

Automatically detects the format of PD data files and returns the appropriate loader.
"""

import os
from typing import Optional, List, Tuple, Type

from .base import BaseLoader, DatasetInfo
from .rugged import RuggedLoader


# Registry of available loaders in priority order
LOADER_CLASSES: List[Type[BaseLoader]] = [
    RuggedLoader,
    # Add more loaders here as they are implemented:
    # TUDelftLoader,
    # GenericLoader,
]


def detect_format(prefix: str, data_dir: str) -> Optional[str]:
    """
    Detect the format of a dataset.

    Args:
        prefix: Dataset prefix/name
        data_dir: Directory containing data files

    Returns:
        Format type string, or None if not detected
    """
    for loader_class in LOADER_CLASSES:
        loader = loader_class(data_dir)
        if loader.detect(prefix):
            return loader.FORMAT_TYPE
    return None


def get_loader(prefix: str, data_dir: str) -> Optional[BaseLoader]:
    """
    Get the appropriate loader for a dataset.

    Args:
        prefix: Dataset prefix/name
        data_dir: Directory containing data files

    Returns:
        Loader instance, or None if no matching loader found
    """
    for loader_class in LOADER_CLASSES:
        loader = loader_class(data_dir)
        if loader.detect(prefix):
            return loader
    return None


def get_loader_for_format(format_type: str, data_dir: str) -> Optional[BaseLoader]:
    """
    Get a loader for a specific format type.

    Args:
        format_type: Format type string (e.g., 'rugged', 'tudelft')
        data_dir: Directory containing data files

    Returns:
        Loader instance, or None if format not supported
    """
    for loader_class in LOADER_CLASSES:
        loader = loader_class(data_dir)
        if loader.FORMAT_TYPE == format_type:
            return loader
    return None


def list_datasets(data_dir: str) -> List[Tuple[str, str]]:
    """
    List all detected datasets in a directory.

    Args:
        data_dir: Directory to scan

    Returns:
        List of (prefix, format_type) tuples
    """
    datasets = []
    seen_prefixes = set()

    for loader_class in LOADER_CLASSES:
        loader = loader_class(data_dir)
        if hasattr(loader, 'list_datasets'):
            for prefix in loader.list_datasets():
                if prefix not in seen_prefixes:
                    datasets.append((prefix, loader.FORMAT_TYPE))
                    seen_prefixes.add(prefix)

    return sorted(datasets, key=lambda x: x[0])


def get_dataset_info(prefix: str, data_dir: str) -> Optional[DatasetInfo]:
    """
    Get information about a dataset.

    Args:
        prefix: Dataset prefix/name
        data_dir: Directory containing data files

    Returns:
        DatasetInfo object, or None if dataset not found
    """
    loader = get_loader(prefix, data_dir)
    if loader and hasattr(loader, 'get_dataset_info'):
        return loader.get_dataset_info(prefix)
    return None


class AutoLoader:
    """
    Convenience class that auto-detects format and loads data.

    Usage:
        loader = AutoLoader('Rugged Data Files')
        data = loader.load('dataset_name')
    """

    def __init__(self, data_dir: str):
        """
        Initialize the auto-loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self._loader_cache = {}

    def _get_loader(self, prefix: str) -> Optional[BaseLoader]:
        """Get or create a loader for the given prefix."""
        if prefix not in self._loader_cache:
            self._loader_cache[prefix] = get_loader(prefix, self.data_dir)
        return self._loader_cache[prefix]

    def load(self, prefix: str) -> dict:
        """
        Load all data for a dataset.

        Args:
            prefix: Dataset prefix/name

        Returns:
            Dictionary with waveforms, settings, phase_angles, etc.
        """
        loader = self._get_loader(prefix)
        if loader is None:
            raise ValueError(f"No loader found for dataset: {prefix}")
        return loader.load_all(prefix)

    def load_waveforms(self, prefix: str) -> list:
        """Load waveforms for a dataset."""
        loader = self._get_loader(prefix)
        if loader is None:
            raise ValueError(f"No loader found for dataset: {prefix}")
        return loader.load_waveforms(prefix)

    def load_settings(self, prefix: str) -> dict:
        """Load settings for a dataset."""
        loader = self._get_loader(prefix)
        if loader is None:
            raise ValueError(f"No loader found for dataset: {prefix}")
        return loader.load_settings(prefix)

    def list_datasets(self) -> List[Tuple[str, str]]:
        """List all datasets in the data directory."""
        return list_datasets(self.data_dir)

    def detect_format(self, prefix: str) -> Optional[str]:
        """Detect the format of a dataset."""
        return detect_format(prefix, self.data_dir)
