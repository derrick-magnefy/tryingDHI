"""
MATLAB .mat File Loader for PD Data

Loads continuous PD data from MATLAB .mat files (IEEE format).
These files typically contain:
- Raw signal data (continuous acquisition)
- Sample rate / time vector
- Metadata (AC frequency, voltage, etc.)

Usage:
    loader = MatLoader('IEEE Data/dataset.mat')
    data = loader.load()

    signal = data['signal']
    sample_rate = data['sample_rate']
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class MatLoader:
    """
    Loader for MATLAB .mat files containing PD data.

    Supports both MATLAB v5 (.mat) and v7.3 (HDF5-based .mat) formats.

    Common variable names for PD data:
    - Signal: 'signal', 'data', 'waveform', 'voltage', 'v', 'y'
    - Sample rate: 'fs', 'Fs', 'sample_rate', 'samplerate', 'sr'
    - Time: 't', 'time', 'Time'
    - AC frequency: 'f_ac', 'fac', 'ac_freq', 'frequency'
    """

    # Common variable name patterns
    SIGNAL_NAMES = ['signal', 'data', 'waveform', 'voltage', 'v', 'y', 'ch1', 'ch2', 'channel1']
    CHANNEL_PATTERN = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'ch1', 'ch2', 'ch3', 'ch4', 'CH1', 'CH2', 'CH3', 'CH4']
    SAMPLE_RATE_NAMES = ['fs', 'Fs', 'FS', 'sample_rate', 'samplerate', 'sr', 'SampleRate']
    SAMPLE_INTERVAL_NAMES = ['dt', 'Dt', 'DT', 'sample_interval', 'delta_t', 'ts', 'Ts']
    TIME_NAMES = ['t', 'time', 'Time']  # Removed 'T' - often used for total time in IEEE format
    TOTAL_TIME_NAMES = ['T', 'total_time', 'duration', 'acq_time']
    AC_FREQ_NAMES = ['f_ac', 'fac', 'ac_freq', 'frequency', 'f_line', 'line_freq']

    def __init__(
        self,
        filepath: Union[str, Path],
        signal_var: Optional[str] = None,
        sample_rate_var: Optional[str] = None,
        time_var: Optional[str] = None,
    ):
        """
        Initialize the .mat file loader.

        Args:
            filepath: Path to the .mat file
            signal_var: Variable name for signal data (auto-detect if None)
            sample_rate_var: Variable name for sample rate (auto-detect if None)
            time_var: Variable name for time vector (auto-detect if None)
        """
        self.filepath = Path(filepath)
        self.signal_var = signal_var
        self.sample_rate_var = sample_rate_var
        self.time_var = time_var

        self._mat_data: Optional[Dict] = None
        self._is_v73 = False

    def _load_mat_file(self):
        """Load the .mat file using appropriate loader."""
        try:
            import scipy.io as sio
            self._mat_data = sio.loadmat(str(self.filepath), squeeze_me=True)
            self._is_v73 = False
        except NotImplementedError:
            # v7.3 files are HDF5 format
            try:
                import h5py
                self._mat_data = {}
                with h5py.File(str(self.filepath), 'r') as f:
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            self._mat_data[key] = f[key][:]
                        elif isinstance(f[key], h5py.Group):
                            # Handle structs/cell arrays
                            self._mat_data[key] = self._load_h5_group(f[key])
                self._is_v73 = True
            except ImportError:
                raise ImportError(
                    "h5py required for MATLAB v7.3 files. Install with: pip install h5py"
                )

    def _load_h5_group(self, group) -> Dict:
        """Recursively load HDF5 group (for v7.3 files)."""
        import h5py
        result = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                result[key] = group[key][:]
            elif isinstance(group[key], h5py.Group):
                result[key] = self._load_h5_group(group[key])
        return result

    def _find_signal(self) -> tuple:
        """Find and return the signal data."""
        # Check explicit variable name first
        if self.signal_var and self.signal_var in self._mat_data:
            signal = self._mat_data[self.signal_var]
            return np.asarray(signal).flatten(), self.signal_var

        # Check for channel patterns (common in IEEE format: Ch1, Ch2, etc.)
        for name in self.CHANNEL_PATTERN:
            if name in self._mat_data:
                signal = self._mat_data[name]
                if isinstance(signal, np.ndarray) and signal.size > 1000:
                    return np.asarray(signal).flatten(), name

        # Auto-detect from common names
        for name in self.SIGNAL_NAMES:
            if name in self._mat_data:
                signal = self._mat_data[name]
                if isinstance(signal, np.ndarray) and signal.size > 1000:
                    return np.asarray(signal).flatten(), name

            # Try case-insensitive match
            for key in self._mat_data.keys():
                if key.lower() == name.lower():
                    signal = self._mat_data[key]
                    if isinstance(signal, np.ndarray) and signal.size > 1000:
                        return np.asarray(signal).flatten(), key

        # Find largest array as fallback
        largest_name = None
        largest_size = 0
        for key, value in self._mat_data.items():
            if key.startswith('_'):
                continue
            if isinstance(value, np.ndarray) and value.size > largest_size:
                largest_size = value.size
                largest_name = key

        if largest_name:
            return np.asarray(self._mat_data[largest_name]).flatten(), largest_name

        raise ValueError("Could not find signal data in .mat file")

    def _find_sample_rate(self) -> tuple:
        """Find or calculate the sample rate."""
        # Check explicit variable name
        if self.sample_rate_var and self.sample_rate_var in self._mat_data:
            sr = self._mat_data[self.sample_rate_var]
            return float(np.asarray(sr).flatten()[0]), 'variable'

        # Auto-detect from common names
        for name in self.SAMPLE_RATE_NAMES:
            if name in self._mat_data:
                sr = self._mat_data[name]
                return float(np.asarray(sr).flatten()[0]), 'variable'

            # Case-insensitive
            for key in self._mat_data.keys():
                if key.lower() == name.lower():
                    sr = self._mat_data[key]
                    return float(np.asarray(sr).flatten()[0]), 'variable'

        # Check for sample interval (dt) - common in IEEE format
        for name in self.SAMPLE_INTERVAL_NAMES:
            if name in self._mat_data:
                dt = self._mat_data[name]
                dt_val = float(np.asarray(dt).flatten()[0])
                if dt_val > 0:
                    return 1.0 / dt_val, 'calculated_from_dt'

            # Case-insensitive
            for key in self._mat_data.keys():
                if key.lower() == name.lower():
                    dt = self._mat_data[key]
                    dt_val = float(np.asarray(dt).flatten()[0])
                    if dt_val > 0:
                        return 1.0 / dt_val, 'calculated_from_dt'

        # Try to calculate from time vector
        time_vec = self._find_time_vector()
        if time_vec is not None and len(time_vec) > 1:
            dt = np.median(np.diff(time_vec))
            if dt > 0:
                return 1.0 / dt, 'calculated_from_time'

        # Default assumption for typical oscilloscope data
        print("Warning: Could not find sample rate. Using default 250 MHz.")
        return 250e6, 'default'

    def _find_time_vector(self) -> Optional[np.ndarray]:
        """Find the time vector if available."""
        if self.time_var and self.time_var in self._mat_data:
            return np.asarray(self._mat_data[self.time_var]).flatten()

        for name in self.TIME_NAMES:
            if name in self._mat_data:
                t = self._mat_data[name]
                if isinstance(t, np.ndarray) and t.size > 1:
                    return np.asarray(t).flatten()

        return None

    def _find_ac_frequency(self) -> Optional[float]:
        """Find the AC line frequency if available."""
        for name in self.AC_FREQ_NAMES:
            if name in self._mat_data:
                freq = self._mat_data[name]
                return float(np.asarray(freq).flatten()[0])

        return None

    def get_info(self) -> Dict[str, Any]:
        """Get information about the .mat file without fully loading data."""
        self._load_mat_file()

        info = {
            'filepath': str(self.filepath),
            'format': 'v7.3 (HDF5)' if self._is_v73 else 'v5',
            'variables': {},
        }

        for key, value in self._mat_data.items():
            if key.startswith('_'):
                continue
            if isinstance(value, np.ndarray):
                info['variables'][key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'size': value.size,
                }
            else:
                info['variables'][key] = {
                    'type': type(value).__name__,
                    'value': value if np.isscalar(value) else 'complex',
                }

        return info

    def list_variables(self) -> List[str]:
        """List all variable names in the .mat file."""
        self._load_mat_file()
        return [k for k in self._mat_data.keys() if not k.startswith('_')]

    def list_channels(self) -> List[str]:
        """
        List all available signal channels in the .mat file.

        Looks for common channel naming patterns: Ch1, Ch2, channel1, etc.

        Returns:
            List of channel variable names found
        """
        self._load_mat_file()
        channels = []

        # Check for common channel patterns
        channel_patterns = ['Ch', 'ch', 'CH', 'Channel', 'channel', 'CHANNEL']

        for key in self._mat_data.keys():
            if key.startswith('_'):
                continue
            value = self._mat_data[key]
            if not isinstance(value, np.ndarray) or value.size < 1000:
                continue

            # Check if it matches a channel pattern
            for pattern in channel_patterns:
                if key.startswith(pattern):
                    channels.append(key)
                    break

        return sorted(channels)

    def load_channel(self, channel: str) -> Dict[str, Any]:
        """
        Load a specific channel from a multi-channel .mat file.

        Args:
            channel: Channel variable name (e.g., 'Ch1', 'Ch2')

        Returns:
            Dict with signal data and metadata for the specified channel
        """
        return self.load(signal_var_override=channel)

    def load(self, signal_var_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the .mat file and extract PD data.

        Args:
            signal_var_override: Override the signal variable to load

        Returns:
            Dict with keys:
            - 'signal': numpy array of signal data
            - 'sample_rate': sample rate in Hz
            - 'time': time vector (if available)
            - 'ac_frequency': AC line frequency (if available)
            - 'metadata': dict of other variables
            - 'filepath': source file path
            - 'available_channels': list of channel names found
        """
        self._load_mat_file()

        result = {
            'filepath': str(self.filepath),
            'metadata': {},
        }

        # Use override if provided
        effective_signal_var = signal_var_override or self.signal_var

        # Extract signal
        if effective_signal_var:
            self.signal_var = effective_signal_var
        signal, signal_name = self._find_signal()
        result['signal'] = signal
        result['signal_var'] = signal_name

        # List available channels
        result['available_channels'] = self.list_channels()

        # Extract or calculate sample rate
        sample_rate, sr_source = self._find_sample_rate()
        result['sample_rate'] = sample_rate
        result['sample_rate_source'] = sr_source

        # Extract time vector if available
        time_vec = self._find_time_vector()
        if time_vec is not None:
            result['time'] = time_vec

        # Extract AC frequency if available
        ac_freq = self._find_ac_frequency()
        if ac_freq is not None:
            result['ac_frequency'] = ac_freq
        else:
            result['ac_frequency'] = 60.0  # Default

        # Store remaining variables as metadata
        for key, value in self._mat_data.items():
            if key.startswith('_'):
                continue
            if key not in [signal_name, self.sample_rate_var, self.time_var]:
                if isinstance(value, np.ndarray):
                    result['metadata'][key] = value
                elif np.isscalar(value):
                    result['metadata'][key] = value

        return result


def load_mat_file(
    filepath: Union[str, Path],
    signal_var: Optional[str] = None,
    sample_rate_var: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load a .mat file.

    Args:
        filepath: Path to .mat file
        signal_var: Variable name for signal (auto-detect if None)
        sample_rate_var: Variable name for sample rate (auto-detect if None)

    Returns:
        Dict with signal data and metadata
    """
    loader = MatLoader(filepath, signal_var, sample_rate_var)
    return loader.load()
