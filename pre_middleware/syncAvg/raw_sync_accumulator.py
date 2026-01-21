"""
Raw Synchronous Average Accumulator

Maintains sample-by-sample (not binned) sync average data that can be
incrementally updated as new AC cycles are processed.

At 250 MHz sample rate and 60 Hz AC frequency:
- samples_per_cycle = 250,000,000 / 60 ≈ 4,166,667 samples

This allows:
1. Incremental updates without reprocessing all data
2. Mean AND peak amplitude at each sample position
3. Compression to display format (360 bins) when needed
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import gzip
import json


@dataclass
class RawSyncAverageState:
    """State of raw sync average accumulator."""
    sum_values: np.ndarray       # Running sum at each sample position
    sample_counts: np.ndarray    # Count at each sample position
    max_values: np.ndarray       # Peak value at each sample position
    samples_per_cycle: int       # Number of samples in one AC cycle
    sample_rate: float           # Sample rate in Hz
    ac_frequency: float          # AC frequency in Hz
    num_cycles: int              # Total cycles accumulated


class RawSyncAccumulator:
    """
    Accumulator for raw (sample-resolution) synchronous averaging.

    Maintains three arrays of ~4.17M samples each:
    - sum_values: Running sum for mean calculation
    - sample_counts: Count for mean calculation
    - max_values: Peak at each sample position

    Can be incrementally updated and serialized/deserialized.
    """

    def __init__(
        self,
        sample_rate: float = 250e6,
        ac_frequency: float = 60.0,
        samples_per_cycle: Optional[int] = None,
    ):
        """
        Initialize accumulator.

        Args:
            sample_rate: Sample rate in Hz (default: 250 MHz)
            ac_frequency: AC frequency in Hz (default: 60 Hz)
            samples_per_cycle: Override computed samples per cycle
        """
        self.sample_rate = sample_rate
        self.ac_frequency = ac_frequency

        if samples_per_cycle is not None:
            self.samples_per_cycle = samples_per_cycle
        else:
            self.samples_per_cycle = int(sample_rate / ac_frequency)

        # Initialize arrays
        self.sum_values = np.zeros(self.samples_per_cycle, dtype=np.float64)
        self.sample_counts = np.zeros(self.samples_per_cycle, dtype=np.int64)
        self.max_values = np.zeros(self.samples_per_cycle, dtype=np.float64)
        self.num_cycles = 0

    def add_cycle(self, cycle_data: np.ndarray, use_absolute: bool = True):
        """
        Add one AC cycle of data to the accumulator.

        Args:
            cycle_data: Array of samples for one AC cycle
            use_absolute: Use absolute values (default: True)
        """
        if len(cycle_data) != self.samples_per_cycle:
            # Resample if needed
            cycle_data = self._resample(cycle_data, self.samples_per_cycle)

        if use_absolute:
            cycle_data = np.abs(cycle_data)

        # Update accumulators
        self.sum_values += cycle_data
        self.sample_counts += 1
        self.max_values = np.maximum(self.max_values, cycle_data)
        self.num_cycles += 1

    def add_signal(
        self,
        signal: np.ndarray,
        phases: Optional[np.ndarray] = None,
        reference: Optional[np.ndarray] = None,
        use_absolute: bool = True,
    ):
        """
        Add continuous signal data, splitting into aligned AC cycles.

        Args:
            signal: Raw signal data
            phases: Optional phase array (0-360 degrees)
            reference: Optional AC reference signal for phase alignment
            use_absolute: Use absolute values
        """
        # If we have reference signal, find zero crossings for alignment
        if reference is not None:
            cycle_starts = self._find_zero_crossings(reference)
        elif phases is not None:
            # Find where phase wraps from ~360 to ~0
            phase_diff = np.diff(phases)
            cycle_starts = np.where(phase_diff < -180)[0] + 1
            cycle_starts = np.concatenate([[0], cycle_starts])
        else:
            # Assume cycles start at regular intervals
            cycle_starts = np.arange(0, len(signal), self.samples_per_cycle)

        # Process each complete cycle
        for i in range(len(cycle_starts) - 1):
            start = cycle_starts[i]
            end = cycle_starts[i + 1]

            cycle_data = signal[start:end]

            # Only add if we have enough samples (at least 90% of expected)
            if len(cycle_data) >= self.samples_per_cycle * 0.9:
                self.add_cycle(cycle_data, use_absolute)

    def _find_zero_crossings(self, reference: np.ndarray) -> np.ndarray:
        """Find positive-going zero crossings in reference signal."""
        ref = reference - np.mean(reference)
        sign = np.sign(ref)
        sign[sign == 0] = 1
        crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]
        return crossings

    def _resample(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample data to target length using linear interpolation."""
        if len(data) == target_length:
            return data

        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, data)

    def get_mean(self) -> np.ndarray:
        """Get mean amplitude at each sample position."""
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = self.sum_values / self.sample_counts
            mean = np.nan_to_num(mean, nan=0.0)
        return mean

    def get_peak(self) -> np.ndarray:
        """Get peak amplitude at each sample position."""
        return self.max_values.copy()

    def get_state(self) -> RawSyncAverageState:
        """Get current state."""
        return RawSyncAverageState(
            sum_values=self.sum_values.copy(),
            sample_counts=self.sample_counts.copy(),
            max_values=self.max_values.copy(),
            samples_per_cycle=self.samples_per_cycle,
            sample_rate=self.sample_rate,
            ac_frequency=self.ac_frequency,
            num_cycles=self.num_cycles,
        )

    def load_state(self, state: RawSyncAverageState):
        """Load state from RawSyncAverageState."""
        self.sum_values = state.sum_values.copy()
        self.sample_counts = state.sample_counts.copy()
        self.max_values = state.max_values.copy()
        self.samples_per_cycle = state.samples_per_cycle
        self.sample_rate = state.sample_rate
        self.ac_frequency = state.ac_frequency
        self.num_cycles = state.num_cycles

    def to_display_format(self, num_bins: int = 360) -> Dict[str, Any]:
        """
        Compress to display format (360 bins).

        Args:
            num_bins: Number of phase bins for display

        Returns:
            Dict with phase_bins, mean_amplitude, peak_amplitude, etc.
        """
        samples_per_bin = self.samples_per_cycle // num_bins

        mean_full = self.get_mean()
        peak_full = self.get_peak()

        # Bin the data
        mean_bins = np.zeros(num_bins)
        peak_bins = np.zeros(num_bins)
        count_bins = np.zeros(num_bins, dtype=np.int64)

        for i in range(num_bins):
            start = i * samples_per_bin
            end = min((i + 1) * samples_per_bin, self.samples_per_cycle)

            mean_bins[i] = np.mean(mean_full[start:end])
            peak_bins[i] = np.max(peak_full[start:end])
            count_bins[i] = np.sum(self.sample_counts[start:end])

        phase_bins = np.arange(num_bins) * (360.0 / num_bins) + (180.0 / num_bins)

        return {
            'phase_bins': phase_bins.tolist(),
            'mean_amplitude': mean_bins.tolist(),
            'peak_amplitude': peak_bins.tolist(),
            'sample_count': count_bins.tolist(),
            'num_bins': num_bins,
            'num_cycles': self.num_cycles,
            'samples_per_cycle': self.samples_per_cycle,
            'sample_rate': self.sample_rate,
            'ac_frequency': self.ac_frequency,
        }

    def save_npz(self, filepath: str, compress: bool = True):
        """
        Save raw sync average to .npz file.

        Args:
            filepath: Output file path
            compress: Use compression (default: True)
        """
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(
            filepath,
            sum_values=self.sum_values,
            sample_counts=self.sample_counts,
            max_values=self.max_values,
            samples_per_cycle=self.samples_per_cycle,
            sample_rate=self.sample_rate,
            ac_frequency=self.ac_frequency,
            num_cycles=self.num_cycles,
        )

    @classmethod
    def load_npz(cls, filepath: str) -> 'RawSyncAccumulator':
        """
        Load raw sync average from .npz file.

        Args:
            filepath: Input file path

        Returns:
            RawSyncAccumulator instance
        """
        data = np.load(filepath)

        accumulator = cls(
            sample_rate=float(data['sample_rate']),
            ac_frequency=float(data['ac_frequency']),
            samples_per_cycle=int(data['samples_per_cycle']),
        )

        accumulator.sum_values = data['sum_values']
        accumulator.sample_counts = data['sample_counts']
        accumulator.max_values = data['max_values']
        accumulator.num_cycles = int(data['num_cycles'])

        return accumulator

    def to_bytes(self, compress: bool = True) -> bytes:
        """
        Serialize to bytes for S3 storage.

        Args:
            compress: Use gzip compression

        Returns:
            Bytes representation
        """
        import io

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            sum_values=self.sum_values,
            sample_counts=self.sample_counts,
            max_values=self.max_values,
            samples_per_cycle=self.samples_per_cycle,
            sample_rate=self.sample_rate,
            ac_frequency=self.ac_frequency,
            num_cycles=self.num_cycles,
        )

        data = buffer.getvalue()

        if compress:
            data = gzip.compress(data)

        return data

    @classmethod
    def from_bytes(cls, data: bytes, compressed: bool = True) -> 'RawSyncAccumulator':
        """
        Deserialize from bytes.

        Args:
            data: Bytes to deserialize
            compressed: Data is gzip compressed

        Returns:
            RawSyncAccumulator instance
        """
        import io

        if compressed:
            data = gzip.decompress(data)

        buffer = io.BytesIO(data)
        npz_data = np.load(buffer)

        accumulator = cls(
            sample_rate=float(npz_data['sample_rate']),
            ac_frequency=float(npz_data['ac_frequency']),
            samples_per_cycle=int(npz_data['samples_per_cycle']),
        )

        accumulator.sum_values = npz_data['sum_values']
        accumulator.sample_counts = npz_data['sample_counts']
        accumulator.max_values = npz_data['max_values']
        accumulator.num_cycles = int(npz_data['num_cycles'])

        return accumulator


def merge_accumulators(accumulators: list) -> RawSyncAccumulator:
    """
    Merge multiple accumulators into one.

    All accumulators must have the same samples_per_cycle.

    Args:
        accumulators: List of RawSyncAccumulator instances

    Returns:
        Merged RawSyncAccumulator
    """
    if not accumulators:
        raise ValueError("No accumulators to merge")

    first = accumulators[0]
    merged = RawSyncAccumulator(
        sample_rate=first.sample_rate,
        ac_frequency=first.ac_frequency,
        samples_per_cycle=first.samples_per_cycle,
    )

    for acc in accumulators:
        if acc.samples_per_cycle != merged.samples_per_cycle:
            raise ValueError("All accumulators must have same samples_per_cycle")

        merged.sum_values += acc.sum_values
        merged.sample_counts += acc.sample_counts
        merged.max_values = np.maximum(merged.max_values, acc.max_values)
        merged.num_cycles += acc.num_cycles

    return merged
