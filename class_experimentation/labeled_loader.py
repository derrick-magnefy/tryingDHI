"""
Labeled Dataset Loader

Loads .mat files from IEEE_Example_Data/ and assigns expected PD type labels
based on filename patterns.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Import the existing MatLoader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pre_middleware.loaders.mat_loader import MatLoader


# Filename patterns to PD type mapping
# These patterns are matched against lowercase filenames
PD_TYPE_PATTERNS = {
    'SURFACE': [
        r'surface',
        r'tracking',
        r'surf_pd',
    ],
    'CORONA': [
        r'corona',
        r'needle',
        r'point[\s_-]?plane',
        r'grounded[\s_-]?needle',
        r'grounded[\s_-]?plane',
    ],
    'INTERNAL': [
        r'internal',
        r'void',
        r'cavity',
        r'in[\s_-]?oil',
        r'in[\s_-]?solid',
        r'oil',
        r'solid',
    ],
    'NOISE': [
        r'noise',
        r'background',
        r'no[\s_-]?pd',
    ],
}


class LabeledDataset:
    """Container for a labeled dataset."""

    def __init__(
        self,
        filepath: str,
        expected_type: str,
        signal: np.ndarray,
        sample_rate: float,
        ac_frequency: float = 60.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.expected_type = expected_type
        self.signal = signal
        self.sample_rate = sample_rate
        self.ac_frequency = ac_frequency
        self.metadata = metadata or {}

    def __repr__(self):
        return f"LabeledDataset({self.filename}, type={self.expected_type}, samples={len(self.signal)})"


class LabeledDatasetLoader:
    """
    Loads labeled .mat files and assigns expected PD type based on filename.

    Usage:
        loader = LabeledDatasetLoader("IEEE_Example_Data/")
        datasets = loader.load_all()

        for ds in datasets:
            print(f"{ds.filename}: expected {ds.expected_type}")
    """

    def __init__(
        self,
        data_dir: str,
        channel: str = "Ch1",
        custom_patterns: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing .mat files
            channel: Channel to load from .mat files (default: Ch1)
            custom_patterns: Optional custom filename patterns to PD type mapping
        """
        self.data_dir = Path(data_dir)
        self.channel = channel
        self.patterns = custom_patterns or PD_TYPE_PATTERNS

        # Compile regex patterns
        self._compiled_patterns = {}
        for pd_type, patterns in self.patterns.items():
            self._compiled_patterns[pd_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def infer_pd_type(self, filename: str) -> str:
        """
        Infer expected PD type from filename.

        Args:
            filename: Name of the file

        Returns:
            Inferred PD type (SURFACE, CORONA, INTERNAL, NOISE, or UNKNOWN)
        """
        filename_lower = filename.lower()

        for pd_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(filename_lower):
                    return pd_type

        return "UNKNOWN"

    def find_mat_files(self) -> List[Path]:
        """Find all .mat files in the data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        mat_files = list(self.data_dir.glob("**/*.mat"))
        return sorted(mat_files)

    def load_file(self, filepath: Path) -> Optional[LabeledDataset]:
        """
        Load a single .mat file and create a LabeledDataset.

        Args:
            filepath: Path to .mat file

        Returns:
            LabeledDataset or None if loading fails
        """
        try:
            loader = MatLoader(str(filepath))

            # Try to load the specified channel
            available_channels = loader.list_channels()

            if self.channel in available_channels:
                channel = self.channel
            elif available_channels:
                channel = available_channels[0]
                print(f"  Warning: {self.channel} not found, using {channel}")
            else:
                print(f"  Error: No channels found in {filepath.name}")
                return None

            data = loader.load_channel(channel)
            signal = data['signal']
            sample_rate = data.get('sample_rate', 1e9)  # Default 1 GS/s

            # Detect AC frequency if possible
            ac_frequency = self._detect_ac_frequency(data) or 60.0

            # Infer PD type from filename
            expected_type = self.infer_pd_type(filepath.name)

            return LabeledDataset(
                filepath=str(filepath),
                expected_type=expected_type,
                signal=signal,
                sample_rate=sample_rate,
                ac_frequency=ac_frequency,
                metadata={
                    'channel': channel,
                    'available_channels': available_channels,
                    'original_data': data,
                }
            )

        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            return None

    def _detect_ac_frequency(self, data: Dict) -> Optional[float]:
        """Try to detect AC frequency from data."""
        # Check if AC frequency is stored in metadata
        if 'ac_frequency' in data:
            return data['ac_frequency']
        if 'frequency' in data:
            return data['frequency']

        # Could add zero-crossing detection here if needed
        return None

    def load_all(
        self,
        filter_types: Optional[List[str]] = None,
        exclude_unknown: bool = True
    ) -> List[LabeledDataset]:
        """
        Load all .mat files from the data directory.

        Args:
            filter_types: Only load files matching these PD types (None = all)
            exclude_unknown: Skip files where PD type couldn't be inferred

        Returns:
            List of LabeledDataset objects
        """
        mat_files = self.find_mat_files()
        print(f"Found {len(mat_files)} .mat files in {self.data_dir}")

        datasets = []
        for filepath in mat_files:
            print(f"Loading {filepath.name}...")

            # Check if we should skip based on inferred type
            inferred_type = self.infer_pd_type(filepath.name)

            if exclude_unknown and inferred_type == "UNKNOWN":
                print(f"  Skipping (unknown type)")
                continue

            if filter_types and inferred_type not in filter_types:
                print(f"  Skipping (type={inferred_type}, filter={filter_types})")
                continue

            dataset = self.load_file(filepath)
            if dataset:
                datasets.append(dataset)
                print(f"  Loaded: {len(dataset.signal)} samples, type={dataset.expected_type}")

        print(f"\nLoaded {len(datasets)} datasets:")
        type_counts = {}
        for ds in datasets:
            type_counts[ds.expected_type] = type_counts.get(ds.expected_type, 0) + 1
        for pd_type, count in sorted(type_counts.items()):
            print(f"  {pd_type}: {count} files")

        return datasets

    def get_summary(self, datasets: List[LabeledDataset]) -> Dict[str, Any]:
        """Get summary statistics for loaded datasets."""
        summary = {
            'total_files': len(datasets),
            'by_type': {},
            'total_samples': 0,
        }

        for ds in datasets:
            pd_type = ds.expected_type
            if pd_type not in summary['by_type']:
                summary['by_type'][pd_type] = {
                    'count': 0,
                    'files': [],
                    'total_samples': 0,
                }

            summary['by_type'][pd_type]['count'] += 1
            summary['by_type'][pd_type]['files'].append(ds.filename)
            summary['by_type'][pd_type]['total_samples'] += len(ds.signal)
            summary['total_samples'] += len(ds.signal)

        return summary


if __name__ == "__main__":
    # Test the loader
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "IEEE_Example_Data"

    loader = LabeledDatasetLoader(data_dir)
    datasets = loader.load_all()

    print("\nSummary:")
    summary = loader.get_summary(datasets)
    print(f"Total files: {summary['total_files']}")
    print(f"Total samples: {summary['total_samples']:,}")
    for pd_type, info in summary['by_type'].items():
        print(f"  {pd_type}: {info['count']} files, {info['total_samples']:,} samples")
