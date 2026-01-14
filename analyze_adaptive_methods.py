#!/usr/bin/env python3
"""
Analyze all adaptive window methods across all example datasets.
Produces a comprehensive comparison table to recommend the best method.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pre_middleware.loaders.mat_loader import MatLoader
from pre_middleware.waveletProc.dwt_detector import DWTDetector
from pre_middleware.waveletProc.waveform_extractor import WaveletExtractor
from pdlib.features.pulse_detection import detect_pulses
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# ADAPTIVE WINDOW METHODS (copied from GUI)
# =============================================================================

def method_original(waveform: np.ndarray, sample_interval: float) -> Dict:
    return {
        'name': 'Original',
        'waveform': waveform,
        'start_idx': 0,
        'end_idx': len(waveform),
    }


def method_rise_fall_based(waveform: np.ndarray, sample_interval: float,
                           threshold_pct: float = 0.1, padding_factor: float = 1.5) -> Dict:
    abs_wfm = np.abs(waveform)
    max_amp = np.max(abs_wfm)
    if max_amp == 0:
        return method_original(waveform, sample_interval)
    threshold = threshold_pct * max_amp
    above_threshold = abs_wfm > threshold
    if not np.any(above_threshold):
        return method_original(waveform, sample_interval)
    indices = np.where(above_threshold)[0]
    start_idx = indices[0]
    end_idx = indices[-1]
    signal_width = end_idx - start_idx
    padding = int(signal_width * (padding_factor - 1) / 2)
    new_start = max(0, start_idx - padding)
    new_end = min(len(waveform), end_idx + padding)
    return {
        'name': 'Rise-Fall',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
    }


def method_energy_based(waveform: np.ndarray, sample_interval: float,
                        energy_fraction: float = 0.90) -> Dict:
    energy = waveform ** 2
    total_energy = np.sum(energy)
    if total_energy == 0:
        return method_original(waveform, sample_interval)
    cumsum = np.cumsum(energy)
    normalized = cumsum / total_energy
    margin = (1 - energy_fraction) / 2
    start_idx = np.searchsorted(normalized, margin)
    end_idx = np.searchsorted(normalized, 1 - margin)
    buffer = max(10, (end_idx - start_idx) // 10)
    new_start = max(0, start_idx - buffer)
    new_end = min(len(waveform), end_idx + buffer)
    return {
        'name': f'Energy{int(energy_fraction*100)}',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
    }


def method_peak_centered(waveform: np.ndarray, sample_interval: float,
                         window_us: float = 0.5) -> Dict:
    abs_wfm = np.abs(waveform)
    peak_idx = np.argmax(abs_wfm)
    window_samples = int(window_us * 1e-6 / sample_interval)
    half_window = window_samples // 2
    new_start = max(0, peak_idx - half_window)
    new_end = min(len(waveform), peak_idx + half_window)
    return {
        'name': f'Peak{window_us}us',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
    }


def method_adaptive_shrink(waveform: np.ndarray, sample_interval: float,
                           min_separation_us: float = 0.5) -> Dict:
    pulse_info = detect_pulses(waveform, sample_interval, 0.3, min_separation_us)
    if pulse_info['pulse_count'] <= 1:
        result = method_rise_fall_based(waveform, sample_interval)
        result['name'] = 'AdaptShrink'
        return result
    abs_wfm = np.abs(waveform)
    pulse_indices = pulse_info['pulse_indices']
    pulse_amps = [abs_wfm[i] for i in pulse_indices]
    main_pulse_idx = pulse_indices[np.argmax(pulse_amps)]
    min_dist = len(waveform)
    for idx in pulse_indices:
        if idx != main_pulse_idx:
            min_dist = min(min_dist, abs(idx - main_pulse_idx))
    half_window = max(50, min_dist // 2)
    new_start = max(0, main_pulse_idx - half_window)
    new_end = min(len(waveform), main_pulse_idx + half_window)
    return {
        'name': 'AdaptShrink',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
    }


def method_derivative_based(waveform: np.ndarray, sample_interval: float,
                            smoothing_sigma: float = 2.0) -> Dict:
    smoothed = gaussian_filter1d(np.abs(waveform), sigma=smoothing_sigma)
    deriv = np.abs(np.diff(smoothed))
    max_deriv = np.max(deriv)
    if max_deriv == 0:
        return method_original(waveform, sample_interval)
    threshold = 0.05 * max_deriv
    significant = deriv > threshold
    if not np.any(significant):
        return method_original(waveform, sample_interval)
    indices = np.where(significant)[0]
    start_idx = max(0, indices[0] - 20)
    end_idx = min(len(waveform), indices[-1] + 20)
    return {
        'name': 'Derivative',
        'waveform': waveform[start_idx:end_idx],
        'start_idx': start_idx,
        'end_idx': end_idx,
    }


def method_smart_bounds(waveform: np.ndarray, sample_interval: float,
                        noise_window: int = 50, snr_threshold: float = 3.0,
                        min_separation_us: float = 0.5) -> Dict:
    """
    Smart boundary detection using envelope analysis and noise floor estimation.
    Works for both single-pulse (finds true boundaries) and multi-pulse (isolates main).
    """
    n = len(waveform)
    if n < noise_window * 2:
        return method_original(waveform, sample_interval)

    # Estimate noise floor from edges
    noise_start = waveform[:noise_window]
    noise_end = waveform[-noise_window:]
    noise_rms_start = np.sqrt(np.mean(noise_start ** 2))
    noise_rms_end = np.sqrt(np.mean(noise_end ** 2))
    noise_floor = min(noise_rms_start, noise_rms_end)

    if noise_floor < 1e-10:
        noise_floor = np.max(np.abs(waveform)) * 0.01

    # Compute envelope using Hilbert transform
    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)
    envelope_smooth = gaussian_filter1d(envelope, sigma=3)

    signal_threshold = snr_threshold * noise_floor
    above_threshold = envelope_smooth > signal_threshold

    if not np.any(above_threshold):
        peak_idx = np.argmax(envelope_smooth)
        half_win = max(25, int(0.2e-6 / sample_interval))
        start_idx = max(0, peak_idx - half_win)
        end_idx = min(n, peak_idx + half_win)
        return {
            'name': f'SmartBounds {snr_threshold:.0f}x',
            'waveform': waveform[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
        }

    # Check for multi-pulse
    pulse_info = detect_pulses(waveform, sample_interval, 0.3, min_separation_us)

    if pulse_info['pulse_count'] > 1:
        # Multi-pulse: isolate main pulse using envelope
        pulse_indices = pulse_info['pulse_indices']
        pulse_amps = [envelope_smooth[i] for i in pulse_indices]
        main_pulse_idx = pulse_indices[np.argmax(pulse_amps)]

        # Find boundaries using envelope
        start_idx = main_pulse_idx
        for i in range(main_pulse_idx, -1, -1):
            if envelope_smooth[i] < signal_threshold:
                start_idx = i
                break
            for other_idx in pulse_indices:
                if other_idx < main_pulse_idx and i <= other_idx:
                    start_idx = max(i, other_idx + 10)
                    break

        end_idx = main_pulse_idx
        for i in range(main_pulse_idx, n):
            if envelope_smooth[i] < signal_threshold:
                end_idx = i
                break
            for other_idx in pulse_indices:
                if other_idx > main_pulse_idx and i >= other_idx - 10:
                    end_idx = min(i, other_idx - 10)
                    break

        padding = max(10, int(0.1e-6 / sample_interval))
        start_idx = max(0, start_idx - padding)
        end_idx = min(n, end_idx + padding)

        return {
            'name': f'SmartBounds {snr_threshold:.0f}x',
            'waveform': waveform[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
        }

    # Single pulse - find true boundaries
    peak_idx = np.argmax(envelope_smooth)

    start_idx = 0
    for i in range(peak_idx, -1, -1):
        if envelope_smooth[i] < signal_threshold:
            start_idx = i
            break

    end_idx = n - 1
    for i in range(peak_idx, n):
        if envelope_smooth[i] < signal_threshold:
            remaining = envelope_smooth[i:min(i+50, n)]
            if np.max(remaining) < signal_threshold * 1.5:
                end_idx = i
                break

    signal_duration = end_idx - start_idx
    padding = max(10, signal_duration // 10)
    start_idx = max(0, start_idx - padding)
    end_idx = min(n, end_idx + padding)

    return {
        'name': f'SmartBounds {snr_threshold:.0f}x',
        'waveform': waveform[start_idx:end_idx],
        'start_idx': start_idx,
        'end_idx': end_idx,
    }


def method_smart_bounds_padded(waveform: np.ndarray, sample_interval: float,
                                min_window_us: float = 0.5) -> Dict:
    """SmartBounds with minimum window guarantee."""
    result = method_smart_bounds(waveform, sample_interval)
    min_samples = int(min_window_us * 1e-6 / sample_interval)
    current_samples = len(result['waveform'])

    if current_samples >= min_samples:
        result['name'] = f'SB+{min_window_us}us'
        return result

    # Expand to minimum window
    abs_wfm = np.abs(result['waveform'])
    local_peak = np.argmax(abs_wfm)
    peak_in_original = result['start_idx'] + local_peak
    half_window = min_samples // 2
    new_start = max(0, peak_in_original - half_window)
    new_end = min(len(waveform), peak_in_original + half_window)

    return {
        'name': f'SB+{min_window_us}us',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
    }


# All methods to test
METHODS = {
    'Original': method_original,
    'Rise-Fall': method_rise_fall_based,
    'Energy90': lambda w, s: method_energy_based(w, s, 0.90),
    'Energy95': lambda w, s: method_energy_based(w, s, 0.95),
    'Peak0.5us': lambda w, s: method_peak_centered(w, s, 0.5),
    'Peak1.0us': lambda w, s: method_peak_centered(w, s, 1.0),
    'Peak2.0us': lambda w, s: method_peak_centered(w, s, 2.0),
    'AdaptShrink': method_adaptive_shrink,
    'Derivative': method_derivative_based,
    'SmartBounds': method_smart_bounds,
    'SmartBounds5x': lambda w, s: method_smart_bounds(w, s, snr_threshold=5.0),
    'SmartBounds2x': lambda w, s: method_smart_bounds(w, s, snr_threshold=2.0),
    'SB+0.5us': lambda w, s: method_smart_bounds_padded(w, s, 0.5),
    'SB+1.0us': lambda w, s: method_smart_bounds_padded(w, s, 1.0),
}


def analyze_file(filepath: str, band: str = 'D3', max_waveforms: int = 200) -> Dict:
    """Analyze a single file with all methods."""

    loader = MatLoader(filepath)
    channels = loader.list_channels()

    # Find data channel
    data_channel = None
    for ch in channels:
        if 'Ch2' not in ch:
            data_channel = ch
            break

    if data_channel is None:
        return None

    # Load signal
    load_result = loader.load_channel(data_channel)
    raw_signal = load_result['signal']
    sample_rate = load_result.get('sample_rate', 250e6)
    sample_interval = 1.0 / sample_rate

    # Detect events
    detector = DWTDetector(sample_rate=sample_rate)
    detection_result = detector.detect(raw_signal)
    events = detection_result.events

    # Filter by band
    band_events = [e for e in events if e.band == band]

    if len(band_events) == 0:
        return None

    # Extract waveforms
    extractor = WaveletExtractor(sample_rate=sample_rate)
    limited_events = band_events[:max_waveforms]
    result = extractor.extract_from_events(raw_signal, limited_events)

    if len(result.waveforms) == 0:
        return None

    # Analyze each waveform with each method
    results = {name: {'total': 0, 'multi_pulse': 0, 'avg_samples': 0, 'avg_duration_us': 0}
               for name in METHODS.keys()}

    for wfm_obj in result.waveforms:
        waveform = wfm_obj.waveform

        # Remove baseline
        baseline = np.mean(waveform[:50]) if len(waveform) > 50 else np.mean(waveform)
        waveform = waveform - baseline

        for method_name, method_func in METHODS.items():
            try:
                method_result = method_func(waveform, sample_interval)
                extracted = method_result['waveform']

                # Check for multi-pulse
                pulse_info = detect_pulses(extracted, sample_interval)
                is_multi = pulse_info['pulse_count'] > 1

                results[method_name]['total'] += 1
                if is_multi:
                    results[method_name]['multi_pulse'] += 1
                results[method_name]['avg_samples'] += len(extracted)
                results[method_name]['avg_duration_us'] += len(extracted) * sample_interval * 1e6
            except Exception as e:
                pass

    # Calculate averages
    for method_name in results:
        total = results[method_name]['total']
        if total > 0:
            results[method_name]['avg_samples'] /= total
            results[method_name]['avg_duration_us'] /= total
            results[method_name]['multi_pulse_pct'] = 100 * results[method_name]['multi_pulse'] / total
        else:
            results[method_name]['multi_pulse_pct'] = 0

    return results


def main():
    print("=" * 100)
    print("ADAPTIVE WINDOW METHOD ANALYSIS")
    print("=" * 100)

    # Find all example data files
    data_dir = Path("Example Data")
    mat_files = list(data_dir.glob("*.mat"))

    if not mat_files:
        print("No .mat files found in Example Data/")
        return

    print(f"\nFound {len(mat_files)} files to analyze")

    # Analyze each band
    bands = ['D1', 'D2', 'D3']

    all_results = {}

    for band in bands:
        print(f"\n{'='*100}")
        print(f"BAND: {band}")
        print("=" * 100)

        band_results = defaultdict(lambda: {'total': 0, 'multi_pulse': 0, 'files': 0, 'avg_samples': 0})

        for mat_file in mat_files:
            print(f"\nProcessing: {mat_file.name} (Band {band})...")

            try:
                results = analyze_file(str(mat_file), band=band)

                if results is None:
                    print(f"  No {band} events found")
                    continue

                # Aggregate results
                for method_name, method_results in results.items():
                    band_results[method_name]['total'] += method_results['total']
                    band_results[method_name]['multi_pulse'] += method_results['multi_pulse']
                    band_results[method_name]['files'] += 1
                    # Weighted average of samples
                    n = method_results['total']
                    if n > 0:
                        prev_total = band_results[method_name]['total'] - n
                        prev_avg = band_results[method_name]['avg_samples']
                        new_avg = method_results['avg_samples']
                        if prev_total > 0:
                            band_results[method_name]['avg_samples'] = (prev_avg * prev_total + new_avg * n) / band_results[method_name]['total']
                        else:
                            band_results[method_name]['avg_samples'] = new_avg

                # Print file summary
                orig_mp = results['Original']['multi_pulse_pct']
                adapt_mp = results['AdaptShrink']['multi_pulse_pct']
                total = results['Original']['total']
                print(f"  {total} waveforms: Original={orig_mp:.1f}% multi-pulse, AdaptShrink={adapt_mp:.1f}% multi-pulse")

            except Exception as e:
                print(f"  Error: {e}")

        all_results[band] = dict(band_results)

    # Print summary tables
    print("\n" + "=" * 100)
    print("SUMMARY TABLES")
    print("=" * 100)

    for band in bands:
        print(f"\n{'='*80}")
        print(f"BAND {band} - Multi-Pulse Reduction Summary")
        print("=" * 80)

        if band not in all_results or not all_results[band]:
            print("No data for this band")
            continue

        # Header
        print(f"{'Method':<15} {'Total WFMs':>12} {'Multi-Pulse':>12} {'MP %':>10} {'Reduction':>12} {'Avg Samples':>12}")
        print("-" * 95)

        # Get original baseline
        orig_data = all_results[band].get('Original', {'total': 0, 'multi_pulse': 0, 'avg_samples': 0})
        orig_total = orig_data['total']
        orig_mp = orig_data['multi_pulse']
        orig_pct = 100 * orig_mp / orig_total if orig_total > 0 else 0
        orig_samples = orig_data.get('avg_samples', 0)

        # Print each method
        for method_name in METHODS.keys():
            data = all_results[band].get(method_name, {'total': 0, 'multi_pulse': 0, 'avg_samples': 0})
            total = data['total']
            mp = data['multi_pulse']
            pct = 100 * mp / total if total > 0 else 0
            avg_samples = data.get('avg_samples', 0)

            # Calculate reduction from original
            if orig_mp > 0:
                reduction = 100 * (orig_mp - mp) / orig_mp
            else:
                reduction = 0

            reduction_str = f"{reduction:+.1f}%" if method_name != 'Original' else "baseline"

            print(f"{method_name:<15} {total:>12} {mp:>12} {pct:>9.1f}% {reduction_str:>12} {avg_samples:>12.0f}")

    # Overall recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    # Find best method for each band
    for band in bands:
        if band not in all_results or not all_results[band]:
            continue

        orig_data = all_results[band].get('Original', {'total': 0, 'multi_pulse': 0})
        orig_mp = orig_data['multi_pulse']

        best_method = None
        best_reduction = 0

        for method_name, data in all_results[band].items():
            if method_name == 'Original':
                continue
            mp = data['multi_pulse']
            if orig_mp > 0:
                reduction = (orig_mp - mp) / orig_mp
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_method = method_name

        if best_method:
            print(f"\nBand {band}: Best method is '{best_method}' with {best_reduction*100:.1f}% multi-pulse reduction")
        else:
            print(f"\nBand {band}: No significant improvement from any method")


if __name__ == "__main__":
    main()
