"""
Rugged format loader.

Handles the Rugged data format with these files:
- {prefix}-WFMs.txt: Tab-separated waveform samples (one waveform per line)
- {prefix}-SG.txt: Settings (sample interval, thresholds, etc.)
- {prefix}-Ti.txt: Trigger timestamps
- {prefix}-Ph.txt: Phase angles
- {prefix}-Amp.txt: Amplitude values (optional)
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional

from .base import BaseLoader, DatasetInfo


# Rugged SG.txt indices
SG_INDICES = {
    'threshold_negative': 0,
    'threshold_positive': 1,
    'pretrigger_samples': 2,
    'num_samples': 3,
    'sample_interval': 4,
    'measurement_duration': 5,
    'start_timestamp': 6,
    'end_timestamp': 7,
    'total_records': 8,
    'ac_frequency': 9,
    'voltage_channel': 10,
}


class RuggedLoader(BaseLoader):
    """
    Loader for Rugged data format.

    File structure:
    - {prefix}-WFMs.txt: Waveform samples (tab-separated, one per line)
    - {prefix}-SG.txt: Settings
    - {prefix}-Ti.txt: Trigger times
    - {prefix}-Ph.txt: Phase angles
    """

    FORMAT_TYPE = 'rugged'

    def detect(self, prefix: str) -> bool:
        """Check if Rugged format files exist."""
        wfm_file = os.path.join(self.data_dir, f"{prefix}-WFMs.txt")
        return os.path.exists(wfm_file)

    def load_waveforms(self, prefix: str) -> List[np.ndarray]:
        """Load waveforms from -WFMs.txt file."""
        filepath = os.path.join(self.data_dir, f"{prefix}-WFMs.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Waveform file not found: {filepath}")

        waveforms = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    values = [float(v) for v in line.split('\t') if v.strip()]
                    waveforms.append(np.array(values))
        return waveforms

    def load_settings(self, prefix: str) -> Dict[str, Any]:
        """Load settings from -SG.txt file."""
        filepath = os.path.join(self.data_dir, f"{prefix}-SG.txt")
        settings = {
            'sample_interval': 4e-9,  # Default 4ns
            'ac_frequency': 60.0,     # Default 60Hz
            'format': self.FORMAT_TYPE,
        }

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read().strip()
                values = [float(v) for v in content.split('\t') if v.strip()]

            # Map values to settings
            if len(values) > SG_INDICES['sample_interval']:
                settings['sample_interval'] = values[SG_INDICES['sample_interval']]
            if len(values) > SG_INDICES['ac_frequency']:
                settings['ac_frequency'] = values[SG_INDICES['ac_frequency']]
            if len(values) > SG_INDICES['threshold_negative']:
                settings['threshold_negative'] = values[SG_INDICES['threshold_negative']]
            if len(values) > SG_INDICES['threshold_positive']:
                settings['threshold_positive'] = values[SG_INDICES['threshold_positive']]
            if len(values) > SG_INDICES['num_samples']:
                settings['num_samples'] = int(values[SG_INDICES['num_samples']])
            if len(values) > SG_INDICES['pretrigger_samples']:
                settings['pretrigger_samples'] = int(values[SG_INDICES['pretrigger_samples']])
            if len(values) > SG_INDICES['measurement_duration']:
                settings['measurement_duration'] = values[SG_INDICES['measurement_duration']]
            if len(values) > SG_INDICES['total_records']:
                settings['total_records'] = int(values[SG_INDICES['total_records']])

            settings['raw_values'] = values

        return settings

    def load_phase_angles(self, prefix: str) -> Optional[np.ndarray]:
        """Load phase angles from -Ph.txt file."""
        filepath = os.path.join(self.data_dir, f"{prefix}-Ph.txt")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            content = f.read().strip()
            values = [float(v) for v in content.split('\t') if v.strip()]

        return np.array(values)

    def load_trigger_times(self, prefix: str) -> Optional[np.ndarray]:
        """Load trigger timestamps from -Ti.txt file."""
        filepath = os.path.join(self.data_dir, f"{prefix}-Ti.txt")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            content = f.read().strip()
            values = [float(v) for v in content.split('\t') if v.strip()]

        return np.array(values)

    def load_amplitudes(self, prefix: str) -> Optional[np.ndarray]:
        """Load amplitude data from -Amp.txt file (optional)."""
        filepath = os.path.join(self.data_dir, f"{prefix}-Amp.txt")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            content = f.read().strip()
            values = [float(v) for v in content.split('\t') if v.strip()]

        return np.array(values)

    def get_dataset_info(self, prefix: str) -> DatasetInfo:
        """Get information about a dataset."""
        settings = self.load_settings(prefix)
        waveforms = self.load_waveforms(prefix)

        return DatasetInfo(
            prefix=prefix,
            format_type=self.FORMAT_TYPE,
            data_dir=self.data_dir,
            n_waveforms=len(waveforms),
            sample_interval=settings.get('sample_interval', 4e-9),
            ac_frequency=settings.get('ac_frequency', 60.0)
        )

    def list_datasets(self) -> List[str]:
        """List all Rugged datasets in the data directory."""
        wfm_files = [
            f for f in os.listdir(self.data_dir)
            if f.endswith('-WFMs.txt')
        ]
        prefixes = [f.replace('-WFMs.txt', '') for f in wfm_files]
        return sorted(prefixes)


def load_rugged_waveforms(filepath: str) -> List[np.ndarray]:
    """
    Convenience function to load waveforms from a Rugged -WFMs.txt file.

    Args:
        filepath: Path to the -WFMs.txt file

    Returns:
        List of numpy arrays, one per waveform
    """
    waveforms = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(v) for v in line.split('\t') if v.strip()]
                waveforms.append(np.array(values))
    return waveforms


def load_rugged_settings(filepath: str) -> Dict[str, Any]:
    """
    Convenience function to load settings from a Rugged -SG.txt file.

    Args:
        filepath: Path to the -SG.txt file

    Returns:
        Dictionary of settings
    """
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]

    settings = {'raw_values': values}
    if len(values) > SG_INDICES['sample_interval']:
        settings['sample_interval'] = values[SG_INDICES['sample_interval']]
    if len(values) > SG_INDICES['ac_frequency']:
        settings['ac_frequency'] = values[SG_INDICES['ac_frequency']]

    return settings
