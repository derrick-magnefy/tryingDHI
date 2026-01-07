"""
Base loader interface for PD data formats.

Defines the abstract interface that all format-specific loaders must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BaseLoader(ABC):
    """
    Abstract base class for PD data format loaders.

    All format-specific loaders (Rugged, TU Delft, etc.) should inherit
    from this class and implement the required methods.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir

    @abstractmethod
    def detect(self, prefix: str) -> bool:
        """
        Check if this loader can handle the given dataset.

        Args:
            prefix: Dataset prefix/name

        Returns:
            True if this loader can handle the dataset
        """
        pass

    @abstractmethod
    def load_waveforms(self, prefix: str) -> List[np.ndarray]:
        """
        Load waveform data from files.

        Args:
            prefix: Dataset prefix/name

        Returns:
            List of numpy arrays, one per waveform
        """
        pass

    @abstractmethod
    def load_settings(self, prefix: str) -> Dict[str, Any]:
        """
        Load acquisition settings.

        Args:
            prefix: Dataset prefix/name

        Returns:
            Dictionary of settings including sample_interval, ac_frequency, etc.
        """
        pass

    @abstractmethod
    def load_phase_angles(self, prefix: str) -> Optional[np.ndarray]:
        """
        Load phase angle data for each waveform.

        Args:
            prefix: Dataset prefix/name

        Returns:
            numpy array of phase angles (degrees), or None if not available
        """
        pass

    @abstractmethod
    def load_trigger_times(self, prefix: str) -> Optional[np.ndarray]:
        """
        Load trigger timestamps for each waveform.

        Args:
            prefix: Dataset prefix/name

        Returns:
            numpy array of trigger times (seconds), or None if not available
        """
        pass

    def get_sample_interval(self, prefix: str) -> float:
        """
        Get the sample interval for a dataset.

        Args:
            prefix: Dataset prefix/name

        Returns:
            Sample interval in seconds
        """
        settings = self.load_settings(prefix)
        return settings.get('sample_interval', 4e-9)  # Default 4ns

    def get_ac_frequency(self, prefix: str) -> float:
        """
        Get the AC frequency for a dataset.

        Args:
            prefix: Dataset prefix/name

        Returns:
            AC frequency in Hz
        """
        settings = self.load_settings(prefix)
        return settings.get('ac_frequency', 60.0)  # Default 60Hz

    def load_all(self, prefix: str) -> Dict[str, Any]:
        """
        Load all available data for a dataset.

        Args:
            prefix: Dataset prefix/name

        Returns:
            Dictionary containing waveforms, settings, phase_angles, etc.
        """
        return {
            'waveforms': self.load_waveforms(prefix),
            'settings': self.load_settings(prefix),
            'phase_angles': self.load_phase_angles(prefix),
            'trigger_times': self.load_trigger_times(prefix),
        }


class DatasetInfo:
    """Information about a dataset."""

    def __init__(
        self,
        prefix: str,
        format_type: str,
        data_dir: str,
        n_waveforms: int = 0,
        sample_interval: float = 4e-9,
        ac_frequency: float = 60.0
    ):
        self.prefix = prefix
        self.format_type = format_type
        self.data_dir = data_dir
        self.n_waveforms = n_waveforms
        self.sample_interval = sample_interval
        self.ac_frequency = ac_frequency

    def __repr__(self):
        return (
            f"DatasetInfo(prefix='{self.prefix}', format='{self.format_type}', "
            f"n_waveforms={self.n_waveforms})"
        )
