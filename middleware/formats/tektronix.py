#!/usr/bin/env python3
"""
Tektronix WFM file parser for TU Delft PD datasets.

Parses Tektronix WFM003 (FastFrame) format files commonly used
in partial discharge measurement systems.
"""

import struct
import os


class TektronixWFMParser:
    """Parser for Tektronix WFM003 format files."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.header = {}
        self.waveforms = []
        self.timestamps = []
        self._parse()

    def _parse(self):
        """Parse the WFM file."""
        with open(self.filepath, 'rb') as f:
            data = f.read()

        # Check magic number
        if data[2:10] != b':WFM#003':
            raise ValueError(f"Not a valid Tektronix WFM003 file: {self.filepath}")

        # Parse header info
        self.header['magic'] = data[2:10].decode('ascii')
        self.header['bytes_per_point'] = struct.unpack('<H', data[14:16])[0]

        # Record length at offset 0x48
        self.header['record_length'] = struct.unpack('<I', data[0x48:0x4C])[0]

        # Try to find sample interval - search for reasonable values
        # Common locations in WFM003 format
        for offset in [0xa0, 0xa8, 0xe0, 0x20]:
            try:
                val = struct.unpack('<d', data[offset:offset+8])[0]
                if 1e-12 < val < 1e-3:  # Reasonable sample interval range
                    self.header['sample_interval'] = val
                    break
            except:
                pass

        if 'sample_interval' not in self.header:
            # Default to 4ns if not found
            self.header['sample_interval'] = 4e-9

        # Calculate data structure
        file_size = len(data)
        self.header['file_size'] = file_size

        # Find data start by looking for consistent waveform patterns
        # The header is typically 300-1000 bytes
        data_start = self._find_data_start(data)
        self.header['data_start'] = data_start

        # Extract waveforms
        self._extract_waveforms(data, data_start)

    def _find_data_start(self, data):
        """Find where the actual waveform data starts."""
        # Try common header sizes for WFM003
        # Check for pattern consistency at different offsets
        best_offset = 838  # Common for Tektronix WFM003

        for test_offset in [300, 500, 750, 838, 1000]:
            if test_offset + 1000 < len(data):
                # Check if data looks like waveform (reasonable int16 values)
                samples = []
                valid = True
                for i in range(100):
                    try:
                        val = struct.unpack('<h', data[test_offset+i*2:test_offset+i*2+2])[0]
                        samples.append(val)
                    except:
                        valid = False
                        break

                if valid and samples:
                    # Check for reasonable variance (not all zeros or constant)
                    avg = sum(samples) / len(samples)
                    variance = sum((x - avg) ** 2 for x in samples) / len(samples)
                    if variance > 1000:  # Has actual signal variation
                        best_offset = test_offset
                        break

        return best_offset

    def _extract_waveforms(self, data, data_start):
        """Extract individual waveforms from the data."""
        bytes_per_sample = 2  # int16

        # record_length in FastFrame format is the number of frames (waveforms)
        # not samples per waveform
        record_length = self.header.get('record_length', 500)

        # Calculate data size and determine structure
        data_size = len(data) - data_start

        # If record_length is large (>1000), it's likely the frame count
        if record_length > 1000:
            num_waveforms = record_length
            bytes_per_wfm = data_size // num_waveforms
            samples_per_wfm = bytes_per_wfm // bytes_per_sample
        else:
            # Small record_length means it's samples per waveform
            samples_per_wfm = min(record_length, 500)
            bytes_per_wfm = samples_per_wfm * bytes_per_sample
            num_waveforms = data_size // bytes_per_wfm

        if bytes_per_wfm == 0 or samples_per_wfm == 0:
            return

        self.header['num_waveforms'] = num_waveforms
        self.header['samples_per_waveform'] = samples_per_wfm

        # Extract each waveform
        self.waveforms = []
        for i in range(num_waveforms):
            offset = data_start + i * bytes_per_wfm
            samples = []
            for j in range(samples_per_wfm):
                sample_offset = offset + j * bytes_per_sample
                if sample_offset + bytes_per_sample <= len(data):
                    val = struct.unpack('<h', data[sample_offset:sample_offset+bytes_per_sample])[0]
                    samples.append(val)

            if samples:
                # Convert to voltage (simple scaling - adjust as needed)
                # Typical Tektronix scaling: val * y_scale / 32768 + y_offset
                voltage_samples = [s / 32768.0 for s in samples]  # Normalize to -1 to 1
                self.waveforms.append(voltage_samples)

    def get_waveforms(self):
        """Return extracted waveforms as list of lists."""
        return self.waveforms

    def get_sample_interval(self):
        """Return sample interval in seconds."""
        return self.header.get('sample_interval', 4e-9)

    def get_info(self):
        """Return header information."""
        return self.header


def load_tu_delft_timing(txt_filepath):
    """Load timing information from TU Delft .txt file.

    Args:
        txt_filepath: Path to the timing .txt file

    Returns:
        List of (index, timestamp_seconds) tuples
    """
    timing = []

    if not os.path.exists(txt_filepath):
        return timing

    with open(txt_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: "1, 16 Sep 2016 15:05:56.172 869 519 614"
            # The digits after the time are additional decimal places
            # So 56.172 869 519 614 = 56.172869519614 seconds
            parts = line.split(',', 1)
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    # Parse timestamp - extract seconds with full precision
                    time_part = parts[1].strip()
                    # Find the time portion (HH:MM:SS.nnn...)
                    time_parts = time_part.split()
                    for i, p in enumerate(time_parts):
                        if ':' in p:  # Found time
                            time_str = p
                            # Collect fractional parts (additional decimal digits)
                            frac_parts = time_parts[i+1:] if i+1 < len(time_parts) else []

                            # Parse HH:MM:SS.fraction
                            h, m, s_with_frac = time_str.split(':')

                            # Append additional fractional digits
                            # 56.172 + "869 519 614" -> "56.172869519614"
                            if frac_parts:
                                additional_digits = ''.join(frac_parts)
                                s_with_frac = s_with_frac + additional_digits

                            seconds = int(h) * 3600 + int(m) * 60 + float(s_with_frac)

                            timing.append((idx, seconds))
                            break
                except (ValueError, IndexError):
                    continue

    return timing


def convert_timing_to_phase(timing, ac_frequency=50.0, reference_time=None):
    """Convert absolute timestamps to phase angles.

    Args:
        timing: List of (index, seconds) tuples
        ac_frequency: AC frequency in Hz (default 50 Hz)
        reference_time: Reference time for phase calculation (default: first timestamp)

    Returns:
        List of phase angles in degrees (0-360)
    """
    if not timing:
        return []

    if reference_time is None:
        reference_time = timing[0][1]

    period = 1.0 / ac_frequency
    phases = []

    for idx, seconds in timing:
        elapsed = seconds - reference_time
        phase = (elapsed % period) / period * 360.0
        phases.append(phase)

    return phases


if __name__ == '__main__':
    # Test with TU Delft data
    import sys

    if len(sys.argv) > 1:
        wfm_file = sys.argv[1]
    else:
        wfm_file = "TU Delft WFMs/1-Internal_45mm33/1-Internal_45mm33_Ch1.wfm"

    if os.path.exists(wfm_file):
        print(f"Parsing: {wfm_file}")
        parser = TektronixWFMParser(wfm_file)

        print(f"\nHeader info:")
        for k, v in parser.get_info().items():
            print(f"  {k}: {v}")

        waveforms = parser.get_waveforms()
        print(f"\nExtracted {len(waveforms)} waveforms")

        if waveforms:
            print(f"Samples per waveform: {len(waveforms[0])}")
            print(f"First waveform range: [{min(waveforms[0]):.4f}, {max(waveforms[0]):.4f}]")
    else:
        print(f"File not found: {wfm_file}")
