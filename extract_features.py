#!/usr/bin/env python3
"""
Partial Discharge Pulse Feature Extraction Script

Extracts comprehensive features from each PD pulse waveform including:
- Time-domain features (amplitude, timing, shape)
- Frequency-domain features (spectral characteristics)
- Energy-based features (cumulative energy analysis)

Usage:
    python extract_features.py [--input-dir DIR] [--output FILE] [--format FORMAT]

Options:
    --input-dir DIR     Directory containing data files (default: "Rugged Data Files")
    --output FILE       Output file path (default: features.csv)
    --format FORMAT     Output format: csv, tsv, json (default: csv)
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import glob
import argparse
import json
from datetime import datetime

DATA_DIR = "Rugged Data Files"

# Feature names in order
FEATURE_NAMES = [
    'phase_angle',
    'peak_amplitude_positive',
    'peak_amplitude_negative',
    'polarity',
    'rise_time',
    'fall_time',
    'pulse_width',
    'slew_rate',
    'energy',
    'equivalent_time',
    'equivalent_bandwidth',
    'cumulative_energy_peak',
    'cumulative_energy_rise_time',
    'cumulative_energy_shape_factor',
    'cumulative_energy_area_ratio',
    'dominant_frequency',
    'center_frequency',
    'bandwidth_3db',
    'spectral_power_low',
    'spectral_power_high',
    'spectral_flatness',
    'spectral_entropy',
    'peak_to_peak_amplitude',
    'rms_amplitude',
    'crest_factor',
    'rise_fall_ratio',
    'zero_crossing_count',
    'oscillation_count',
    'energy_charge_ratio',
]


def load_waveforms(filepath):
    """Load waveform data from -WFMs.txt file."""
    waveforms = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(v) for v in line.split('\t') if v.strip()]
                waveforms.append(np.array(values))
    return waveforms


def load_single_line_data(filepath):
    """Load data from single-line files."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return np.array(values)


def load_settings(filepath):
    """Load settings from -SG.txt file."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return values


class PDFeatureExtractor:
    """Extract features from partial discharge pulse waveforms."""

    def __init__(self, sample_interval=4e-9, ac_frequency=60.0):
        """
        Initialize the feature extractor.

        Args:
            sample_interval: Time between samples in seconds (default: 4ns)
            ac_frequency: AC line frequency in Hz (default: 60Hz)
        """
        self.sample_interval = sample_interval
        self.ac_frequency = ac_frequency
        self.sample_rate = 1.0 / sample_interval

    def extract_features(self, waveform, phase_angle=None):
        """
        Extract all features from a single waveform.

        Args:
            waveform: numpy array of waveform samples
            phase_angle: Optional phase angle from Ti.txt (degrees)

        Returns:
            dict: Dictionary of feature names to values
        """
        features = {}

        # Remove DC offset using baseline from first samples
        baseline = np.mean(waveform[:50]) if len(waveform) > 50 else np.mean(waveform)
        wfm = waveform - baseline

        # Time array
        n_samples = len(wfm)
        time = np.arange(n_samples) * self.sample_interval

        # === AMPLITUDE FEATURES ===
        features['phase_angle'] = phase_angle if phase_angle is not None else 0.0
        features['peak_amplitude_positive'] = np.max(wfm)
        features['peak_amplitude_negative'] = np.min(wfm)
        features['peak_to_peak_amplitude'] = features['peak_amplitude_positive'] - features['peak_amplitude_negative']

        # Polarity: 1 if positive peak is larger, -1 if negative peak is larger
        if abs(features['peak_amplitude_positive']) >= abs(features['peak_amplitude_negative']):
            features['polarity'] = 1
        else:
            features['polarity'] = -1

        # RMS amplitude
        features['rms_amplitude'] = np.sqrt(np.mean(wfm**2))

        # Crest factor (peak / RMS)
        if features['rms_amplitude'] > 0:
            peak = max(abs(features['peak_amplitude_positive']), abs(features['peak_amplitude_negative']))
            features['crest_factor'] = peak / features['rms_amplitude']
        else:
            features['crest_factor'] = 0.0

        # === TIME-DOMAIN FEATURES ===
        timing = self._extract_timing_features(wfm, time)
        features.update(timing)

        # === ENERGY FEATURES ===
        energy = self._extract_energy_features(wfm, time)
        features.update(energy)

        # === FREQUENCY-DOMAIN FEATURES ===
        spectral = self._extract_spectral_features(wfm)
        features.update(spectral)

        return features

    def _extract_timing_features(self, wfm, time):
        """Extract timing-related features."""
        features = {}

        # Find the main peak (largest absolute value)
        abs_wfm = np.abs(wfm)
        peak_idx = np.argmax(abs_wfm)
        peak_value = wfm[peak_idx]

        # Determine thresholds for rise/fall time (10% to 90% of peak)
        threshold_low = 0.1 * abs(peak_value)
        threshold_high = 0.9 * abs(peak_value)

        # Find rise time (time from 10% to 90% of peak on rising edge)
        rise_start_idx = None
        rise_end_idx = None

        # Search backwards from peak for rising edge
        for i in range(peak_idx, -1, -1):
            if abs(wfm[i]) <= threshold_low:
                rise_start_idx = i
                break
            if abs(wfm[i]) <= threshold_high and rise_end_idx is None:
                rise_end_idx = i

        if rise_start_idx is not None and rise_end_idx is not None and rise_end_idx > rise_start_idx:
            features['rise_time'] = (rise_end_idx - rise_start_idx) * self.sample_interval
        else:
            features['rise_time'] = 0.0

        # Find fall time (time from 90% to 10% of peak on falling edge)
        fall_start_idx = None
        fall_end_idx = None

        # Search forward from peak for falling edge
        for i in range(peak_idx, len(wfm)):
            if fall_start_idx is None and abs(wfm[i]) <= threshold_high:
                fall_start_idx = i
            if abs(wfm[i]) <= threshold_low:
                fall_end_idx = i
                break

        if fall_start_idx is not None and fall_end_idx is not None and fall_end_idx > fall_start_idx:
            features['fall_time'] = (fall_end_idx - fall_start_idx) * self.sample_interval
        else:
            features['fall_time'] = 0.0

        # Pulse width (time above 50% of peak)
        threshold_50 = 0.5 * abs(peak_value)
        above_50 = np.where(abs_wfm >= threshold_50)[0]
        if len(above_50) > 0:
            features['pulse_width'] = (above_50[-1] - above_50[0]) * self.sample_interval
        else:
            features['pulse_width'] = 0.0

        # Slew rate (maximum rate of change)
        diff_wfm = np.diff(wfm) / self.sample_interval
        features['slew_rate'] = np.max(np.abs(diff_wfm))

        # Rise/fall ratio
        if features['fall_time'] > 0:
            features['rise_fall_ratio'] = features['rise_time'] / features['fall_time']
        else:
            features['rise_fall_ratio'] = 0.0

        # Zero crossing count
        zero_crossings = np.where(np.diff(np.signbit(wfm)))[0]
        features['zero_crossing_count'] = len(zero_crossings)

        # Oscillation count (number of local maxima/minima)
        # Find local extrema
        local_max = signal.argrelmax(wfm, order=3)[0]
        local_min = signal.argrelmin(wfm, order=3)[0]
        features['oscillation_count'] = len(local_max) + len(local_min)

        return features

    def _extract_energy_features(self, wfm, time):
        """Extract energy-related features."""
        features = {}

        # Total energy (integral of squared signal)
        features['energy'] = np.sum(wfm**2) * self.sample_interval

        # Charge (integral of absolute signal)
        charge = np.sum(np.abs(wfm)) * self.sample_interval

        # Energy/Charge ratio
        if charge > 0:
            features['energy_charge_ratio'] = features['energy'] / charge
        else:
            features['energy_charge_ratio'] = 0.0

        # Equivalent time (energy-weighted time centroid)
        if features['energy'] > 0:
            features['equivalent_time'] = np.sum(time * wfm**2) * self.sample_interval / features['energy']
        else:
            features['equivalent_time'] = 0.0

        # Equivalent bandwidth (related to signal spread in time)
        if features['energy'] > 0:
            time_spread = np.sum((time - features['equivalent_time'])**2 * wfm**2) * self.sample_interval
            features['equivalent_bandwidth'] = np.sqrt(time_spread / features['energy'])
        else:
            features['equivalent_bandwidth'] = 0.0

        # Cumulative energy analysis
        cumulative_energy = np.cumsum(wfm**2) * self.sample_interval
        total_energy = cumulative_energy[-1] if len(cumulative_energy) > 0 else 0

        if total_energy > 0:
            # Normalized cumulative energy
            norm_cum_energy = cumulative_energy / total_energy

            # Cumulative energy at peak
            peak_idx = np.argmax(np.abs(wfm))
            features['cumulative_energy_peak'] = norm_cum_energy[peak_idx]

            # Time to reach 10% of total energy (rise time in energy domain)
            idx_10 = np.searchsorted(norm_cum_energy, 0.1)
            idx_90 = np.searchsorted(norm_cum_energy, 0.9)
            features['cumulative_energy_rise_time'] = (idx_90 - idx_10) * self.sample_interval

            # Shape factor: ratio of energy in first half vs second half
            mid_idx = len(wfm) // 2
            energy_first_half = cumulative_energy[mid_idx]
            energy_second_half = total_energy - energy_first_half
            if energy_second_half > 0:
                features['cumulative_energy_shape_factor'] = energy_first_half / energy_second_half
            else:
                features['cumulative_energy_shape_factor'] = 0.0

            # Area ratio: ratio of positive to negative areas
            positive_energy = np.sum(wfm[wfm > 0]**2) * self.sample_interval
            negative_energy = np.sum(wfm[wfm < 0]**2) * self.sample_interval
            if negative_energy > 0:
                features['cumulative_energy_area_ratio'] = positive_energy / negative_energy
            else:
                features['cumulative_energy_area_ratio'] = float('inf') if positive_energy > 0 else 1.0
        else:
            features['cumulative_energy_peak'] = 0.0
            features['cumulative_energy_rise_time'] = 0.0
            features['cumulative_energy_shape_factor'] = 0.0
            features['cumulative_energy_area_ratio'] = 1.0

        return features

    def _extract_spectral_features(self, wfm):
        """Extract frequency-domain features."""
        features = {}

        n_samples = len(wfm)

        # Apply window to reduce spectral leakage
        window = signal.windows.hann(n_samples)
        windowed_wfm = wfm * window

        # Compute FFT
        fft_vals = fft(windowed_wfm)
        freqs = fftfreq(n_samples, self.sample_interval)

        # Use only positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        # Power spectrum
        power_spectrum = pos_fft**2
        total_power = np.sum(power_spectrum)

        if total_power > 0 and len(pos_freqs) > 0:
            # Dominant frequency (frequency with maximum power)
            features['dominant_frequency'] = pos_freqs[np.argmax(power_spectrum)]

            # Center frequency (power-weighted centroid)
            features['center_frequency'] = np.sum(pos_freqs * power_spectrum) / total_power

            # 3dB bandwidth
            max_power = np.max(power_spectrum)
            half_power = max_power / 2
            above_half = pos_freqs[power_spectrum >= half_power]
            if len(above_half) > 1:
                features['bandwidth_3db'] = above_half[-1] - above_half[0]
            else:
                features['bandwidth_3db'] = 0.0

            # Spectral power in low and high bands
            # Define low band as 0-25% of Nyquist, high band as 75-100% of Nyquist
            nyquist = self.sample_rate / 2
            low_band_mask = pos_freqs < (0.25 * nyquist)
            high_band_mask = pos_freqs > (0.75 * nyquist)

            features['spectral_power_low'] = np.sum(power_spectrum[low_band_mask]) / total_power
            features['spectral_power_high'] = np.sum(power_spectrum[high_band_mask]) / total_power

            # Spectral flatness (geometric mean / arithmetic mean)
            # High flatness = noise-like, low flatness = tonal
            geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-12)))
            arithmetic_mean = np.mean(power_spectrum)
            if arithmetic_mean > 0:
                features['spectral_flatness'] = geometric_mean / arithmetic_mean
            else:
                features['spectral_flatness'] = 0.0

            # Spectral entropy
            # Normalized power spectrum as probability distribution
            p = power_spectrum / total_power
            p = p[p > 0]  # Remove zeros for log
            features['spectral_entropy'] = -np.sum(p * np.log2(p))

        else:
            features['dominant_frequency'] = 0.0
            features['center_frequency'] = 0.0
            features['bandwidth_3db'] = 0.0
            features['spectral_power_low'] = 0.0
            features['spectral_power_high'] = 0.0
            features['spectral_flatness'] = 0.0
            features['spectral_entropy'] = 0.0

        return features


def process_dataset(prefix, data_dir=DATA_DIR):
    """
    Process a complete dataset and extract features from all waveforms.

    Args:
        prefix: Base filename prefix (without -WFMs.txt suffix)
        data_dir: Directory containing the data files

    Returns:
        tuple: (feature_matrix, metadata_dict)
    """
    wfm_file = os.path.join(data_dir, f"{prefix}-WFMs.txt")
    sg_file = os.path.join(data_dir, f"{prefix}-SG.txt")
    p_file = os.path.join(data_dir, f"{prefix}-P.txt")
    ti_file = os.path.join(data_dir, f"{prefix}-Ti.txt")

    # Load waveforms
    print(f"Loading waveforms from {prefix}...")
    waveforms = load_waveforms(wfm_file)
    print(f"  Loaded {len(waveforms)} waveforms")

    # Load settings
    settings = load_settings(sg_file) if os.path.exists(sg_file) else None
    sample_interval = settings[10] if settings and len(settings) > 10 else 4e-9
    ac_frequency = settings[9] if settings and len(settings) > 9 else 60.0

    # Load phase data
    phase_data = load_single_line_data(p_file) if os.path.exists(p_file) else None

    # Initialize feature extractor
    extractor = PDFeatureExtractor(sample_interval=sample_interval, ac_frequency=ac_frequency)

    # Extract features from each waveform
    all_features = []
    print(f"  Extracting features...")

    for i, wfm in enumerate(waveforms):
        phase = phase_data[i] if phase_data is not None and i < len(phase_data) else None
        features = extractor.extract_features(wfm, phase_angle=phase)
        all_features.append(features)

        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(waveforms)} waveforms")

    # Convert to matrix
    feature_matrix = np.array([[f[name] for name in FEATURE_NAMES] for f in all_features])

    metadata = {
        'prefix': prefix,
        'num_waveforms': len(waveforms),
        'samples_per_waveform': len(waveforms[0]) if waveforms else 0,
        'sample_interval': sample_interval,
        'ac_frequency': ac_frequency,
        'feature_names': FEATURE_NAMES,
    }

    return feature_matrix, metadata


def save_features(feature_matrix, metadata, output_path, format='csv'):
    """
    Save extracted features to file.

    Args:
        feature_matrix: numpy array of shape (n_waveforms, n_features)
        metadata: dict with dataset metadata
        output_path: Output file path
        format: Output format ('csv', 'tsv', 'json')
    """
    if format == 'csv':
        delimiter = ','
        _save_delimited(feature_matrix, metadata, output_path, delimiter)
    elif format == 'tsv':
        delimiter = '\t'
        _save_delimited(feature_matrix, metadata, output_path, delimiter)
    elif format == 'json':
        _save_json(feature_matrix, metadata, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def _save_delimited(feature_matrix, metadata, output_path, delimiter):
    """Save features as delimited text file."""
    with open(output_path, 'w') as f:
        # Write header
        header = ['waveform_index'] + FEATURE_NAMES
        f.write(delimiter.join(header) + '\n')

        # Write data
        for i, row in enumerate(feature_matrix):
            values = [str(i)] + [f'{v:.6e}' for v in row]
            f.write(delimiter.join(values) + '\n')


def _save_json(feature_matrix, metadata, output_path):
    """Save features as JSON file."""
    data = {
        'metadata': metadata,
        'features': {
            'names': FEATURE_NAMES,
            'data': feature_matrix.tolist()
        }
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from PD pulse waveforms"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=DATA_DIR,
        help='Directory containing data files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: <prefix>-features.csv)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'tsv', 'json'],
        default='csv',
        help='Output format'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Process specific file prefix (default: all files)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PARTIAL DISCHARGE PULSE FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output format: {args.format}")
    print("=" * 70)

    # Find files to process
    if args.file:
        prefixes = [args.file]
    else:
        wfm_files = glob.glob(os.path.join(args.input_dir, "*-WFMs.txt"))
        prefixes = [os.path.basename(f).replace("-WFMs.txt", "") for f in wfm_files]

    if not prefixes:
        print("No waveform files found!")
        return

    print(f"\nFound {len(prefixes)} dataset(s) to process\n")

    # Process each dataset
    for prefix in sorted(prefixes):
        print(f"\n{'='*70}")
        print(f"Processing: {prefix}")
        print("="*70)

        try:
            feature_matrix, metadata = process_dataset(prefix, args.input_dir)

            # Determine output path
            if args.output:
                output_path = args.output
            else:
                ext = args.format
                output_path = os.path.join(args.input_dir, f"{prefix}-features.{ext}")

            # Save features
            save_features(feature_matrix, metadata, output_path, args.format)
            print(f"\n  Features saved to: {output_path}")

            # Print summary statistics
            print(f"\n  Feature Statistics:")
            print(f"  {'Feature':<35} {'Min':>12} {'Max':>12} {'Mean':>12}")
            print(f"  {'-'*71}")
            for i, name in enumerate(FEATURE_NAMES[:10]):  # Show first 10
                col = feature_matrix[:, i]
                print(f"  {name:<35} {np.min(col):>12.4e} {np.max(col):>12.4e} {np.mean(col):>12.4e}")
            if len(FEATURE_NAMES) > 10:
                print(f"  ... and {len(FEATURE_NAMES) - 10} more features")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
