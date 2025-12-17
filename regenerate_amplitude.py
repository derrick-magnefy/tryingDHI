#!/usr/bin/env python3
"""
Partial Discharge Waveform Amplitude Analysis Script

This script regenerates the *-A.txt (amplitude) files from the *-WFMs.txt
(waveform) files and compares against the existing amplitude files.

METHODOLOGY FINDINGS:
=====================
After extensive analysis, we identified TWO amplitude calculation methods
based on the settings[15] value in the SG.txt file:

METHOD 1: SIGNED ABSOLUTE MAXIMUM (when settings[15] = 2)
    Formula: A[i] = wfm[argmax(|wfm|)]
    Select the value with largest absolute magnitude, preserving sign.
    Works for: Motor Ch1 (100% match)

METHOD 2: FIRST PEAK (when settings[15] = 0 or 1)
    Formula: A[i] = min(wfm) if argmin(wfm) < argmax(wfm) else max(wfm)
    Select whichever extreme value (min or max) occurs first in time.
    Works for: Motor Ch2 (86.3%), Corona Ch1 (98.8%)

FILE FORMAT FINDINGS:
=====================
PDN Files (.pdn):
    - Header: 8 bytes = two big-endian int32 [num_waveforms, samples_per_wfm]
    - Data: Big-endian float32 waveform samples (flattened)
    - Contains same data as -WFMs.txt but in binary format

SG.txt Files (Settings):
    Index   Meaning
    -----   -------
    [0]     Voltage range/scale
    [1]     Acquisition time (seconds)
    [2]     Number of waveforms
    [3]     Mode/trigger setting (0=standard, 1/4=different modes)
    [9]     AC frequency (Hz, typically 60)
    [10]    Sample interval (seconds, typically 4e-9)
    [15]    Processing mode (2=signed_abs_max, 0/1=first_peak)

Ti.txt Files (Trigger Times):
    - Contains trigger time (seconds) for each waveform
    - Non-uniform intervals (triggered acquisition)
    - Phase = (Ti mod AC_period) / AC_period × 360° + offset

Usage:
    python regenerate_amplitude.py [--regenerate] [--output-dir DIR] [--method METHOD]

Options:
    --regenerate        Write new *-A.txt files (default: compare only)
    --output-dir DIR    Directory for regenerated files
    --method METHOD     Force method: auto, signed_abs_max, first_peak
"""

import numpy as np
import os
import glob
import argparse
from datetime import datetime

DATA_DIR = "Rugged Data Files"


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


def load_amplitude_data(filepath):
    """Load existing amplitude data from -A.txt file."""
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


def signed_abs_max(wfm):
    """
    Method 1: Signed Absolute Maximum

    Select the value with the largest absolute magnitude, preserving sign.
    Used when settings[15] = 2.
    """
    idx = np.argmax(np.abs(wfm))
    return wfm[idx]


def first_peak(wfm):
    """
    Method 2: First Peak

    Select whichever extreme value (min or max) occurs first in time.
    Used when settings[15] = 0 or 1.
    """
    min_idx = np.argmin(wfm)
    max_idx = np.argmax(wfm)
    if min_idx < max_idx:
        return wfm[min_idx]
    else:
        return wfm[max_idx]


def get_method_for_settings(settings):
    """
    Determine the appropriate amplitude method based on settings.

    Returns:
        tuple: (method_function, method_name)
    """
    if settings is None:
        return signed_abs_max, "signed_abs_max"

    setting_15 = settings[15] if len(settings) > 15 else 0

    if setting_15 == 2:
        return signed_abs_max, "signed_abs_max"
    else:
        return first_peak, "first_peak"


def compute_amplitudes(waveforms, method):
    """Compute amplitude values for all waveforms using the specified method."""
    return np.array([method(wfm) for wfm in waveforms])


def format_scientific(values):
    """Format values as tab-separated scientific notation string."""
    return '\t'.join(f'{v:.6E}' for v in values)


def compare_amplitudes(actual, computed, tolerance=1e-12):
    """Compare actual and computed amplitude arrays."""
    matches = np.sum(np.isclose(actual, computed, rtol=tolerance, atol=tolerance))
    total = len(actual)

    correlation = np.corrcoef(actual, computed)[0, 1] if total > 1 else 0
    rmse = np.sqrt(np.mean((actual - computed)**2))
    max_error = np.max(np.abs(actual - computed))

    return {
        'matches': matches,
        'total': total,
        'match_percentage': 100 * matches / total,
        'correlation': correlation,
        'rmse': rmse,
        'max_error': max_error,
        'is_perfect': matches == total
    }


def process_file(wfm_path, regenerate=False, output_dir=None, force_method=None):
    """
    Process a single waveform file.

    Args:
        wfm_path: Path to the -WFMs.txt file
        regenerate: If True, write a new -A.txt file
        output_dir: Directory for output (default: same as input)
        force_method: Force a specific method (None=auto, or method function)

    Returns:
        dict with processing results
    """
    base = wfm_path.replace("-WFMs.txt", "")
    a_path = base + "-A.txt"
    sg_path = base + "-SG.txt"

    basename = os.path.basename(base)

    result = {
        'name': basename,
        'wfm_path': wfm_path,
        'a_path': a_path,
        'status': 'unknown',
    }

    # Load waveforms
    try:
        waveforms = load_waveforms(wfm_path)
        result['num_waveforms'] = len(waveforms)
        result['samples_per_wfm'] = len(waveforms[0]) if waveforms else 0
    except Exception as e:
        result['status'] = 'error'
        result['error'] = f"Failed to load waveforms: {e}"
        return result

    # Load settings if available
    settings = None
    if os.path.exists(sg_path):
        try:
            settings = load_settings(sg_path)
            result['settings'] = settings
        except:
            pass

    # Determine method
    if force_method:
        method = force_method
        method_name = force_method.__name__
    else:
        method, method_name = get_method_for_settings(settings)

    result['method'] = method_name

    # Compute amplitudes
    computed = compute_amplitudes(waveforms, method)
    result['computed_amplitudes'] = computed

    # Also compute with both methods for comparison
    computed_sam = compute_amplitudes(waveforms, signed_abs_max)
    computed_fp = compute_amplitudes(waveforms, first_peak)

    # Compare with actual if available
    has_actual = os.path.exists(a_path)
    if has_actual:
        try:
            actual = load_amplitude_data(a_path)
            result['actual_amplitudes'] = actual

            # Compare with selected method
            comparison = compare_amplitudes(actual, computed)
            result.update(comparison)

            # Also compare with both methods
            result['sam_match'] = compare_amplitudes(actual, computed_sam)['match_percentage']
            result['fp_match'] = compare_amplitudes(actual, computed_fp)['match_percentage']

            if comparison['is_perfect']:
                result['status'] = 'perfect_match'
            elif comparison['match_percentage'] > 95:
                result['status'] = 'high_match'
            elif comparison['match_percentage'] > 50:
                result['status'] = 'partial_match'
            else:
                result['status'] = 'low_match'

        except Exception as e:
            result['status'] = 'comparison_error'
            result['error'] = f"Failed to compare: {e}"
    else:
        result['status'] = 'no_reference'

    # Regenerate file if requested
    if regenerate:
        if output_dir:
            out_path = os.path.join(output_dir, os.path.basename(a_path))
        else:
            out_path = a_path.replace("-A.txt", "-A_regenerated.txt")

        try:
            with open(out_path, 'w') as f:
                f.write(format_scientific(computed))
                f.write('\n')
            result['regenerated_path'] = out_path
        except Exception as e:
            result['regeneration_error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate PD amplitude files from waveform data"
    )
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help='Write regenerated -A.txt files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for regenerated files'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['auto', 'signed_abs_max', 'first_peak'],
        default='auto',
        help='Force amplitude calculation method'
    )
    args = parser.parse_args()

    # Determine force method
    force_method = None
    if args.method == 'signed_abs_max':
        force_method = signed_abs_max
    elif args.method == 'first_peak':
        force_method = first_peak

    print("=" * 80)
    print("PARTIAL DISCHARGE AMPLITUDE ANALYSIS")
    print("=" * 80)
    print(f"Method: {args.method} (settings[15]=2 → signed_abs_max, else → first_peak)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Find all WFM files
    wfm_files = glob.glob(os.path.join(DATA_DIR, "*-WFMs.txt"))

    if not wfm_files:
        print(f"No waveform files found in {DATA_DIR}")
        return

    print(f"\nFound {len(wfm_files)} waveform files\n")

    results = []
    for wfm_path in sorted(wfm_files):
        result = process_file(wfm_path, args.regenerate, args.output_dir, force_method)
        results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'File':<45} {'Method':<16} {'Match %':>8} {'SAM %':>8} {'FP %':>8}")
    print("-" * 90)

    for r in results:
        name = r['name'][:42] + "..." if len(r['name']) > 45 else r['name']
        method = r.get('method', 'N/A')
        match_pct = f"{r.get('match_percentage', 0):.1f}" if 'match_percentage' in r else "N/A"
        sam_pct = f"{r.get('sam_match', 0):.1f}" if 'sam_match' in r else "N/A"
        fp_pct = f"{r.get('fp_match', 0):.1f}" if 'fp_match' in r else "N/A"
        print(f"{name:<45} {method:<16} {match_pct:>8} {sam_pct:>8} {fp_pct:>8}")

    # Settings analysis
    print("\n" + "=" * 80)
    print("SETTINGS ANALYSIS")
    print("=" * 80)
    print(f"{'File':<45} {'Set[3]':>8} {'Set[15]':>8} {'Best Method':<16}")
    print("-" * 80)

    for r in results:
        name = r['name'][:42] + "..." if len(r['name']) > 45 else r['name']
        settings = r.get('settings', [])
        set3 = f"{settings[3]:.0f}" if len(settings) > 3 else "N/A"
        set15 = f"{settings[15]:.0f}" if len(settings) > 15 else "N/A"

        sam_match = r.get('sam_match', 0)
        fp_match = r.get('fp_match', 0)
        if sam_match > fp_match:
            best = f"signed_abs_max ({sam_match:.0f}%)"
        else:
            best = f"first_peak ({fp_match:.0f}%)"

        print(f"{name:<45} {set3:>8} {set15:>8} {best:<16}")

    print("\n" + "=" * 80)
    print("METHODOLOGY SUMMARY")
    print("=" * 80)
    print("""
Two amplitude calculation methods were identified:

1. SIGNED_ABS_MAX (settings[15] = 2):
   - Formula: A = wfm[argmax(|wfm|)]
   - Selects value with largest absolute magnitude
   - Works 100% for Motor Ch1

2. FIRST_PEAK (settings[15] = 0 or 1):
   - Formula: A = min(wfm) if argmin < argmax else max(wfm)
   - Selects whichever peak (min/max) occurs first in time
   - Works 86-99% for other files

The method selection is based on settings[15] in the SG.txt file.
Ch3 files appear to use a different amplitude definition entirely
(values not derived from waveform peaks).
""")

    if args.regenerate:
        print(f"\nRegenerated files written to: {args.output_dir or 'same directory (*_regenerated.txt)'}")


if __name__ == "__main__":
    main()
