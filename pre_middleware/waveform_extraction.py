"""
Waveform Extraction from Continuous Data Streams

Extracts individual PD pulse waveforms around detected trigger points,
producing output compatible with the Rugged Data Files format.

Usage:
    extractor = WaveformExtractor(pre_samples=500, post_samples=1500)
    waveforms, trigger_times = extractor.extract(signal, triggers, sample_rate)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """Result of waveform extraction."""
    waveforms: np.ndarray         # Shape: (num_waveforms, samples_per_waveform)
    trigger_indices: np.ndarray   # Original trigger sample indices
    trigger_times: np.ndarray     # Trigger times in seconds
    valid_mask: np.ndarray        # Boolean mask of valid extractions
    sample_interval: float        # Time between samples
    stats: Dict[str, Any]         # Extraction statistics


class WaveformExtractor:
    """
    Extract waveform windows around trigger points.

    Produces output compatible with Rugged Data Files format:
    - Fixed-length waveforms with pre and post trigger samples
    - Trigger times for phase calculation
    - Settings metadata

    Attributes:
        pre_samples: Number of samples before trigger point (default: 500)
        post_samples: Number of samples after trigger point (default: 1500)
        total_samples: Total waveform length (pre_samples + post_samples)
    """

    def __init__(
        self,
        pre_samples: int = 500,
        post_samples: int = 1500,
        overlap_handling: str = 'skip',
        validate_peak_position: bool = False,
        peak_tolerance: float = 0.5,
    ):
        """
        Initialize waveform extractor.

        Args:
            pre_samples: Samples before trigger (default: 500)
            post_samples: Samples after trigger (default: 1500)
            overlap_handling: How to handle overlapping windows
                             'skip': Skip overlapping triggers
                             'truncate': Truncate to non-overlapping portion
                             'allow': Allow overlaps (may duplicate data)
            validate_peak_position: If True, discard waveforms where peak is far from trigger
            peak_tolerance: Fraction of window where peak must appear (0.5 = first half)
        """
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.total_samples = pre_samples + post_samples
        self.overlap_handling = overlap_handling
        self.validate_peak_position = validate_peak_position
        self.peak_tolerance = peak_tolerance

    def extract(
        self,
        signal: np.ndarray,
        triggers: np.ndarray,
        sample_rate: float,
        start_time: float = 0.0,
    ) -> ExtractionResult:
        """
        Extract waveforms around trigger points.

        Args:
            signal: Raw signal data (1D array)
            triggers: Trigger sample indices
            sample_rate: Sample rate in Hz
            start_time: Absolute start time of signal (for timestamp calculation)

        Returns:
            ExtractionResult with extracted waveforms and metadata
        """
        sample_interval = 1.0 / sample_rate
        signal_length = len(signal)

        # Filter triggers that would go out of bounds
        valid_triggers = []
        skipped_start = 0
        skipped_end = 0

        for t in triggers:
            start_idx = t - self.pre_samples
            end_idx = t + self.post_samples

            if start_idx < 0:
                skipped_start += 1
                continue
            if end_idx > signal_length:
                skipped_end += 1
                continue

            valid_triggers.append(t)

        # Handle overlapping windows
        if self.overlap_handling == 'skip' and len(valid_triggers) > 1:
            non_overlapping = [valid_triggers[0]]
            for t in valid_triggers[1:]:
                # Check if this window overlaps with the previous
                prev_end = non_overlapping[-1] + self.post_samples
                curr_start = t - self.pre_samples
                if curr_start >= prev_end:
                    non_overlapping.append(t)
            skipped_overlap = len(valid_triggers) - len(non_overlapping)
            valid_triggers = non_overlapping
        else:
            skipped_overlap = 0

        valid_triggers = np.array(valid_triggers, dtype=np.int64)

        # Extract waveforms
        num_waveforms = len(valid_triggers)
        waveforms = np.zeros((num_waveforms, self.total_samples), dtype=np.float64)
        valid_mask = np.ones(num_waveforms, dtype=bool)

        for i, t in enumerate(valid_triggers):
            start_idx = t - self.pre_samples
            end_idx = t + self.post_samples
            waveforms[i, :] = signal[start_idx:end_idx]

        # Validate peak positions if enabled
        skipped_peak_position = 0
        if self.validate_peak_position:
            # Check that peak is within tolerance of trigger position
            # Peak should be near pre_samples (the trigger point in the waveform)
            max_peak_idx = int(self.total_samples * self.peak_tolerance)
            valid_indices = []
            for i, wfm in enumerate(waveforms):
                peak_idx = np.argmax(np.abs(wfm))
                if peak_idx <= max_peak_idx:
                    valid_indices.append(i)
                else:
                    valid_mask[i] = False
                    skipped_peak_position += 1

            if valid_indices:
                waveforms = waveforms[valid_indices]
                valid_triggers = valid_triggers[valid_indices]
                valid_mask = valid_mask[valid_indices]
                num_waveforms = len(valid_indices)
            else:
                # All waveforms failed validation
                waveforms = np.array([]).reshape(0, self.total_samples)
                valid_triggers = np.array([], dtype=np.int64)
                valid_mask = np.array([], dtype=bool)
                num_waveforms = 0

        # Calculate trigger times
        trigger_times = start_time + valid_triggers * sample_interval

        return ExtractionResult(
            waveforms=waveforms,
            trigger_indices=valid_triggers,
            trigger_times=trigger_times,
            valid_mask=valid_mask,
            sample_interval=sample_interval,
            stats={
                'total_triggers': len(triggers),
                'extracted': num_waveforms,
                'skipped_start': skipped_start,
                'skipped_end': skipped_end,
                'skipped_overlap': skipped_overlap,
                'skipped_peak_position': skipped_peak_position,
                'pre_samples': self.pre_samples,
                'post_samples': self.post_samples,
                'sample_rate': sample_rate,
            }
        )

    def calculate_phases(
        self,
        trigger_times: np.ndarray,
        ac_frequency: float = 60.0,
        phase_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate AC phase angles for each trigger.

        Args:
            trigger_times: Trigger times in seconds
            ac_frequency: AC power frequency in Hz
            phase_offset: Phase offset in degrees

        Returns:
            Phase angles in degrees (0-360)
        """
        ac_period = 1.0 / ac_frequency

        # Calculate phase within AC cycle
        phases = ((trigger_times % ac_period) / ac_period) * 360.0

        # Apply offset
        phases = (phases + phase_offset) % 360.0

        return phases

    def to_rugged_format(
        self,
        result: ExtractionResult,
        ac_frequency: float = 60.0,
        voltage_scale: float = 1.0,
        phase_offset: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Convert extraction result to Rugged Data Files format.

        Returns dict with keys matching Rugged file types:
        - 'waveforms': 2D array for -WFMs.txt
        - 'trigger_times': 1D array for -Ti.txt
        - 'phases': 1D array for -Ph.txt
        - 'amplitudes': 1D array for -A.txt (max absolute amplitude)
        - 'settings': dict for -SG.txt values

        Args:
            result: ExtractionResult from extract()
            ac_frequency: AC frequency in Hz
            voltage_scale: Voltage scaling factor
            phase_offset: Phase offset in degrees

        Returns:
            Dict with Rugged-compatible data structures
        """
        # Calculate phases
        phases = self.calculate_phases(
            result.trigger_times, ac_frequency, phase_offset
        )

        # Calculate amplitudes (signed absolute maximum)
        amplitudes = np.array([
            wfm[np.argmax(np.abs(wfm))] for wfm in result.waveforms
        ]) * voltage_scale

        # Build settings
        settings = {
            'voltage_scale': voltage_scale,
            'acquisition_time': (result.trigger_times[-1] - result.trigger_times[0])
                                if len(result.trigger_times) > 1 else 0,
            'num_waveforms': len(result.waveforms),
            'mode': 0,  # Standard mode
            'ac_frequency': ac_frequency,
            'sample_interval': result.sample_interval,
            'samples_per_waveform': self.total_samples,
            'pre_trigger_samples': self.pre_samples,
            'post_trigger_samples': self.post_samples,
        }

        return {
            'waveforms': result.waveforms * voltage_scale,
            'trigger_times': result.trigger_times,
            'phases': phases,
            'amplitudes': amplitudes,
            'settings': settings,
        }

    def save_rugged_format(
        self,
        result: ExtractionResult,
        output_dir: str,
        prefix: str,
        ac_frequency: float = 60.0,
        voltage_scale: float = 1.0,
        phase_offset: float = 0.0,
    ) -> Dict[str, str]:
        """
        Save extraction result in Rugged Data Files format.

        Args:
            result: ExtractionResult from extract()
            output_dir: Output directory path
            prefix: File prefix (e.g., 'Dataset1')
            ac_frequency: AC frequency in Hz
            voltage_scale: Voltage scaling factor
            phase_offset: Phase offset in degrees

        Returns:
            Dict mapping file type to file path
        """
        import os

        # Convert to Rugged format
        data = self.to_rugged_format(result, ac_frequency, voltage_scale, phase_offset)

        os.makedirs(output_dir, exist_ok=True)

        files = {}

        # Save waveforms (-WFMs.txt)
        wfm_path = os.path.join(output_dir, f"{prefix}-WFMs.txt")
        np.savetxt(wfm_path, data['waveforms'], delimiter='\t', fmt='%.10e')
        files['waveforms'] = wfm_path

        # Save trigger times (-Ti.txt)
        ti_path = os.path.join(output_dir, f"{prefix}-Ti.txt")
        np.savetxt(ti_path, data['trigger_times'].reshape(1, -1), delimiter='\t', fmt='%.10e')
        files['trigger_times'] = ti_path

        # Save phases (-Ph.txt)
        ph_path = os.path.join(output_dir, f"{prefix}-Ph.txt")
        np.savetxt(ph_path, data['phases'].reshape(1, -1), delimiter='\t', fmt='%.6f')
        files['phases'] = ph_path

        # Save amplitudes (-A.txt)
        a_path = os.path.join(output_dir, f"{prefix}-A.txt")
        np.savetxt(a_path, data['amplitudes'].reshape(1, -1), delimiter='\t', fmt='%.10e')
        files['amplitudes'] = a_path

        # Save settings (-SG.txt)
        sg_path = os.path.join(output_dir, f"{prefix}-SG.txt")
        settings = data['settings']
        # Convert to list format matching Rugged SG.txt
        sg_values = [
            settings['voltage_scale'],       # [0]
            settings['acquisition_time'],    # [1]
            settings['num_waveforms'],       # [2]
            settings['mode'],                # [3]
            0, 0, 0, 0, 0,                   # [4-8] reserved
            settings['ac_frequency'],        # [9]
            settings['sample_interval'],     # [10]
            0, 0, 0, 0,                      # [11-14] reserved
            2,                               # [15] processing mode (signed_abs_max)
            settings['samples_per_waveform'],# [16]
            settings['pre_trigger_samples'], # [17]
            settings['post_trigger_samples'],# [18]
        ]
        np.savetxt(sg_path, [sg_values], delimiter='\t', fmt='%.10e')
        files['settings'] = sg_path

        return files


def extract_waveforms(
    signal: np.ndarray,
    triggers: np.ndarray,
    sample_rate: float,
    pre_samples: int = 500,
    post_samples: int = 1500,
    **kwargs
) -> ExtractionResult:
    """
    Convenience function to extract waveforms.

    Args:
        signal: Raw signal data
        triggers: Trigger sample indices
        sample_rate: Sample rate in Hz
        pre_samples: Samples before trigger
        post_samples: Samples after trigger
        **kwargs: Additional arguments for WaveformExtractor

    Returns:
        ExtractionResult
    """
    extractor = WaveformExtractor(pre_samples, post_samples, **kwargs)
    return extractor.extract(signal, triggers, sample_rate)
