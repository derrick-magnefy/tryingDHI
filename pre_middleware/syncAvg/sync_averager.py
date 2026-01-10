"""
Synchronous Averager for PD Detection

Performs phase-locked averaging of raw data across multiple AC cycles
to reveal consistent PD activity patterns buried in noise.

The core idea: PD events that occur consistently at specific AC phases
will accumulate in the average, while random noise will cancel out.

Usage:
    averager = SyncAverager(num_bins=360)
    result = averager.compute(signal, phases)
    hotspots = averager.find_hotspots(result, threshold_sigma=3.0)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class SyncAverageResult:
    """Result of synchronous averaging."""
    phase_bins: np.ndarray       # Center of each phase bin (degrees)
    mean_amplitude: np.ndarray   # Mean amplitude at each phase bin
    std_amplitude: np.ndarray    # Std dev at each phase bin
    max_amplitude: np.ndarray    # Max amplitude at each phase bin
    min_amplitude: np.ndarray    # Min amplitude at each phase bin
    sample_count: np.ndarray     # Number of samples in each bin
    rms_amplitude: np.ndarray    # RMS amplitude at each phase bin
    num_cycles: int              # Number of AC cycles averaged
    num_bins: int                # Number of phase bins
    bin_width: float             # Width of each bin in degrees


@dataclass
class PhaseHotspot:
    """A region of elevated PD activity at a specific phase."""
    center_phase: float          # Center phase in degrees
    phase_range: Tuple[float, float]  # (start, end) phase range
    mean_amplitude: float        # Mean amplitude in this region
    peak_amplitude: float        # Peak amplitude in this region
    significance: float          # How many sigma above background
    bin_indices: np.ndarray      # Indices of bins in this hotspot


class SyncAverager:
    """
    Synchronous averaging for phase-locked PD detection.

    Divides the AC cycle into bins and computes statistics for
    signal amplitude in each bin across multiple cycles.

    Attributes:
        num_bins: Number of phase bins (default: 360 = 1 degree per bin)
        use_absolute: Use absolute values for averaging
        use_rectified: Use rectified (positive only) values
    """

    def __init__(
        self,
        num_bins: int = 360,
        use_absolute: bool = True,
        use_rectified: bool = False,
    ):
        """
        Initialize synchronous averager.

        Args:
            num_bins: Number of phase bins (default: 360)
            use_absolute: Use absolute value of signal (default: True)
            use_rectified: Use only positive values (default: False)
        """
        self.num_bins = num_bins
        self.use_absolute = use_absolute
        self.use_rectified = use_rectified
        self.bin_width = 360.0 / num_bins

    def compute(
        self,
        signal: np.ndarray,
        phases: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> SyncAverageResult:
        """
        Compute synchronous average of signal over phase bins.

        Args:
            signal: Raw signal data
            phases: Phase angle for each sample (0-360 degrees)
            weights: Optional weights for each sample

        Returns:
            SyncAverageResult with statistics for each phase bin
        """
        if len(signal) != len(phases):
            raise ValueError("Signal and phases must have same length")

        # Prepare signal
        if self.use_absolute:
            sig = np.abs(signal)
        elif self.use_rectified:
            sig = np.maximum(signal, 0)
        else:
            sig = signal.copy()

        # Initialize accumulators
        sum_amplitude = np.zeros(self.num_bins)
        sum_sq_amplitude = np.zeros(self.num_bins)
        max_amplitude = np.full(self.num_bins, -np.inf)
        min_amplitude = np.full(self.num_bins, np.inf)
        sample_count = np.zeros(self.num_bins, dtype=np.int64)

        # Bin phases
        bin_indices = np.floor(phases / self.bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Accumulate statistics
        if weights is not None:
            for i in range(len(signal)):
                b = bin_indices[i]
                w = weights[i]
                sum_amplitude[b] += sig[i] * w
                sum_sq_amplitude[b] += (sig[i] ** 2) * w
                sample_count[b] += w
                max_amplitude[b] = max(max_amplitude[b], sig[i])
                min_amplitude[b] = min(min_amplitude[b], sig[i])
        else:
            for i in range(len(signal)):
                b = bin_indices[i]
                sum_amplitude[b] += sig[i]
                sum_sq_amplitude[b] += sig[i] ** 2
                sample_count[b] += 1
                max_amplitude[b] = max(max_amplitude[b], sig[i])
                min_amplitude[b] = min(min_amplitude[b], sig[i])

        # Compute statistics
        # Avoid division by zero
        count_safe = np.maximum(sample_count, 1)

        mean_amplitude = sum_amplitude / count_safe
        mean_sq = sum_sq_amplitude / count_safe
        variance = mean_sq - mean_amplitude ** 2
        variance = np.maximum(variance, 0)  # Numerical stability
        std_amplitude = np.sqrt(variance)
        rms_amplitude = np.sqrt(mean_sq)

        # Handle bins with no samples
        empty_mask = sample_count == 0
        mean_amplitude[empty_mask] = 0
        std_amplitude[empty_mask] = 0
        max_amplitude[empty_mask] = 0
        min_amplitude[empty_mask] = 0
        rms_amplitude[empty_mask] = 0

        # Phase bin centers
        phase_bins = np.arange(self.num_bins) * self.bin_width + self.bin_width / 2

        # Estimate number of cycles from phase wraparounds
        phase_diff = np.diff(phases)
        wraparounds = np.sum(phase_diff < -180)  # Large negative jumps
        num_cycles = max(1, wraparounds + 1)

        return SyncAverageResult(
            phase_bins=phase_bins,
            mean_amplitude=mean_amplitude,
            std_amplitude=std_amplitude,
            max_amplitude=max_amplitude,
            min_amplitude=min_amplitude,
            sample_count=sample_count,
            rms_amplitude=rms_amplitude,
            num_cycles=num_cycles,
            num_bins=self.num_bins,
            bin_width=self.bin_width,
        )

    def find_hotspots(
        self,
        result: SyncAverageResult,
        threshold_sigma: float = 3.0,
        min_bins: int = 3,
        merge_gap: int = 2,
    ) -> List[PhaseHotspot]:
        """
        Find phase regions with elevated PD activity.

        Args:
            result: SyncAverageResult from compute()
            threshold_sigma: Number of standard deviations above mean
            min_bins: Minimum number of consecutive bins for a hotspot
            merge_gap: Merge hotspots separated by this many bins or fewer

        Returns:
            List of PhaseHotspot objects
        """
        # Compute threshold
        background_mean = np.median(result.mean_amplitude)
        background_std = np.std(result.mean_amplitude)
        threshold = background_mean + threshold_sigma * background_std

        # Find bins above threshold
        above_threshold = result.mean_amplitude > threshold

        # Find contiguous regions
        hotspots = []
        in_hotspot = False
        start_bin = 0

        for i in range(self.num_bins):
            if above_threshold[i]:
                if not in_hotspot:
                    in_hotspot = True
                    start_bin = i
            else:
                if in_hotspot:
                    # End of hotspot
                    if i - start_bin >= min_bins:
                        hotspots.append((start_bin, i - 1))
                    in_hotspot = False

        # Handle hotspot at end
        if in_hotspot and self.num_bins - start_bin >= min_bins:
            hotspots.append((start_bin, self.num_bins - 1))

        # Handle wraparound (hotspot crossing 360/0)
        if len(hotspots) >= 2:
            first = hotspots[0]
            last = hotspots[-1]
            if last[1] == self.num_bins - 1 and first[0] == 0:
                # Merge wraparound hotspot
                merged_bins = list(range(last[0], self.num_bins)) + list(range(0, first[1] + 1))
                if len(merged_bins) >= min_bins:
                    hotspots = hotspots[1:-1]  # Remove first and last
                    # Add as special case with wraparound
                    hotspots.append(('wrap', last[0], first[1]))

        # Merge nearby hotspots
        if merge_gap > 0 and len(hotspots) >= 2:
            merged = []
            current = hotspots[0]
            for next_hs in hotspots[1:]:
                if isinstance(current, tuple) and len(current) == 2 and isinstance(next_hs, tuple) and len(next_hs) == 2:
                    if next_hs[0] - current[1] <= merge_gap:
                        current = (current[0], next_hs[1])
                    else:
                        merged.append(current)
                        current = next_hs
                else:
                    merged.append(current)
                    current = next_hs
            merged.append(current)
            hotspots = merged

        # Convert to PhaseHotspot objects
        phase_hotspots = []
        for hs in hotspots:
            if isinstance(hs, tuple) and hs[0] == 'wrap':
                # Wraparound hotspot
                _, start, end = hs
                bin_indices = np.concatenate([
                    np.arange(start, self.num_bins),
                    np.arange(0, end + 1)
                ])
                center_phase = (result.phase_bins[start] + result.phase_bins[end]) / 2
                if start > end:
                    center_phase = (center_phase + 180) % 360
                phase_range = (result.phase_bins[start] - self.bin_width / 2,
                              result.phase_bins[end] + self.bin_width / 2)
            else:
                start, end = hs
                bin_indices = np.arange(start, end + 1)
                center_phase = result.phase_bins[(start + end) // 2]
                phase_range = (result.phase_bins[start] - self.bin_width / 2,
                              result.phase_bins[end] + self.bin_width / 2)

            mean_amp = np.mean(result.mean_amplitude[bin_indices])
            peak_amp = np.max(result.max_amplitude[bin_indices])
            significance = (mean_amp - background_mean) / (background_std + 1e-10)

            phase_hotspots.append(PhaseHotspot(
                center_phase=center_phase,
                phase_range=phase_range,
                mean_amplitude=mean_amp,
                peak_amplitude=peak_amp,
                significance=significance,
                bin_indices=bin_indices,
            ))

        return phase_hotspots

    def get_samples_in_phase_range(
        self,
        signal: np.ndarray,
        phases: np.ndarray,
        phase_start: float,
        phase_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract samples within a specific phase range.

        Args:
            signal: Original signal data
            phases: Phase array
            phase_start: Start of phase range (degrees)
            phase_end: End of phase range (degrees)

        Returns:
            (sample_indices, sample_values) for samples in range
        """
        phase_start = phase_start % 360
        phase_end = phase_end % 360

        if phase_start <= phase_end:
            mask = (phases >= phase_start) & (phases <= phase_end)
        else:
            # Wraparound case
            mask = (phases >= phase_start) | (phases <= phase_end)

        indices = np.where(mask)[0]
        values = signal[indices]

        return indices, values

    def to_dict(self, result: SyncAverageResult) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'phase_bins': result.phase_bins.tolist(),
            'mean_amplitude': result.mean_amplitude.tolist(),
            'std_amplitude': result.std_amplitude.tolist(),
            'max_amplitude': result.max_amplitude.tolist(),
            'min_amplitude': result.min_amplitude.tolist(),
            'sample_count': result.sample_count.tolist(),
            'rms_amplitude': result.rms_amplitude.tolist(),
            'num_cycles': result.num_cycles,
            'num_bins': result.num_bins,
            'bin_width': result.bin_width,
        }


def compute_sync_average(
    signal: np.ndarray,
    phases: np.ndarray,
    num_bins: int = 360,
    use_absolute: bool = True,
) -> SyncAverageResult:
    """
    Convenience function for synchronous averaging.

    Args:
        signal: Raw signal data
        phases: Phase angle for each sample (0-360 degrees)
        num_bins: Number of phase bins
        use_absolute: Use absolute value of signal

    Returns:
        SyncAverageResult
    """
    averager = SyncAverager(num_bins=num_bins, use_absolute=use_absolute)
    return averager.compute(signal, phases)
