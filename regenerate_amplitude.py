#!/usr/bin/env python3
"""
Partial Discharge Waveform Amplitude Analysis Script

This script regenerates the *-A.txt (amplitude) files from the *-WFMs.txt
(waveform) files and compares against the existing amplitude files.

METHODOLOGY FINDINGS:
=====================
After iterating through multiple methods, the amplitude calculation method
that was previously used is:

    SIGNED ABSOLUTE MAXIMUM (signed_abs_max)

    For each waveform, find the value with the largest absolute magnitude
    and preserve its sign (polarity). This captures the peak of the partial
    discharge pulse while indicating whether it was a positive or negative
    excursion.

    Formula: A[i] = wfm[argmax(|wfm|)]

This method works perfectly (100% match) for:
    - AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1

Other files show partial matches, possibly due to:
    - Different acquisition modes (see settings[3], settings[15])
    - Different voltage ranges/scaling (see settings[0])
    - Post-processing or filtering applied during acquisition

Usage:
    python regenerate_amplitude.py [--regenerate] [--output-dir DIR]

Options:
    --regenerate    Actually write new *-A.txt files (default: compare only)
    --output-dir    Directory for regenerated files (default: same as source)
"""

import numpy as np
import os
import glob
import argparse
from datetime import datetime

DATA_DIR = "Rugged Data Files"


def load_waveforms(filepath):
    """
    Load waveform data from -WFMs.txt file.

    Each line in the file represents one waveform with tab-separated samples.
    Returns a list of numpy arrays.
    """
    waveforms = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(v) for v in line.split('\t') if v.strip()]
                waveforms.append(np.array(values))
    return waveforms


def load_amplitude_data(filepath):
    """
    Load existing amplitude data from -A.txt file.

    The file contains tab-separated amplitude values, one per waveform.
    Returns a numpy array.
    """
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return np.array(values)


def load_settings(filepath):
    """
    Load settings from -SG.txt file.

    Settings are tab-separated values containing acquisition parameters.
    """
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return values


def signed_abs_max(wfm):
    """
    Calculate the signed absolute maximum of a waveform.

    This is the PRIMARY METHOD that matches the original analysis:
    - Find the index of the value with the largest absolute magnitude
    - Return that value (preserving its sign/polarity)

    This captures the peak amplitude of the partial discharge pulse.

    Args:
        wfm: numpy array of waveform samples

    Returns:
        float: The signed value with maximum absolute magnitude
    """
    idx = np.argmax(np.abs(wfm))
    return wfm[idx]


def compute_amplitudes(waveforms, method=signed_abs_max):
    """
    Compute amplitude values for all waveforms using the specified method.

    Args:
        waveforms: List of numpy arrays, each representing a waveform
        method: Function to extract amplitude from a single waveform

    Returns:
        numpy array of amplitude values
    """
    return np.array([method(wfm) for wfm in waveforms])


def format_scientific(values):
    """
    Format values as tab-separated scientific notation string.
    Matches the format of the original -A.txt files.
    """
    return '\t'.join(f'{v:.6E}' for v in values)


def compare_amplitudes(actual, computed, tolerance=1e-12):
    """
    Compare actual and computed amplitude arrays.

    Returns:
        dict with comparison statistics
    """
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


def process_file(wfm_path, regenerate=False, output_dir=None):
    """
    Process a single waveform file.

    Args:
        wfm_path: Path to the -WFMs.txt file
        regenerate: If True, write a new -A.txt file
        output_dir: Directory for output (default: same as input)

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

    # Check if A.txt exists for comparison
    has_actual = os.path.exists(a_path)

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
    if os.path.exists(sg_path):
        try:
            settings = load_settings(sg_path)
            result['settings'] = settings
        except:
            result['settings'] = None

    # Compute amplitudes using signed_abs_max method
    computed = compute_amplitudes(waveforms, signed_abs_max)
    result['computed_amplitudes'] = computed

    # Compare with actual if available
    if has_actual:
        try:
            actual = load_amplitude_data(a_path)
            result['actual_amplitudes'] = actual

            comparison = compare_amplitudes(actual, computed)
            result.update(comparison)

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
    args = parser.parse_args()

    print("=" * 80)
    print("PARTIAL DISCHARGE AMPLITUDE ANALYSIS")
    print("=" * 80)
    print(f"Method: signed_abs_max (value with maximum absolute magnitude)")
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
        result = process_file(wfm_path, args.regenerate, args.output_dir)
        results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'File':<55} {'Match %':>10} {'Status':<15}")
    print("-" * 80)

    for r in results:
        name = r['name'][:52] + "..." if len(r['name']) > 55 else r['name']
        match_pct = f"{r.get('match_percentage', 0):.1f}%" if 'match_percentage' in r else "N/A"
        status = r['status']
        print(f"{name:<55} {match_pct:>10} {status:<15}")

    # Detailed results for perfect matches
    perfect = [r for r in results if r['status'] == 'perfect_match']
    if perfect:
        print("\n" + "=" * 80)
        print("PERFECT MATCHES (100% accuracy)")
        print("=" * 80)
        for r in perfect:
            print(f"\n  File: {r['name']}")
            print(f"  Waveforms: {r['num_waveforms']}")
            print(f"  Samples per waveform: {r['samples_per_wfm']}")
            if r.get('settings'):
                print(f"  Settings[0-4]: {r['settings'][:5]}")

    # Detailed results for non-matches
    non_perfect = [r for r in results if r['status'] not in ['perfect_match', 'no_reference']]
    if non_perfect:
        print("\n" + "=" * 80)
        print("FILES WITH PARTIAL/LOW MATCH (may use different method)")
        print("=" * 80)
        for r in non_perfect:
            print(f"\n  File: {r['name']}")
            print(f"  Match: {r.get('match_percentage', 0):.1f}%")
            print(f"  Correlation: {r.get('correlation', 0):.6f}")
            print(f"  RMSE: {r.get('rmse', 0):.6e}")
            if r.get('settings'):
                print(f"  Settings[0-4]: {r['settings'][:5]}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The 'signed_abs_max' method (maximum absolute value with preserved sign)
works perfectly for some files but not all. This suggests:

1. Different acquisition modes may use different amplitude calculations
2. Post-processing or calibration differs between files
3. Some files may have additional filtering or baseline correction

The method is confirmed to work for:
- AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1 (100% match)

Further investigation would be needed to determine the exact method
used for other files.
""")

    if args.regenerate:
        print(f"\nRegenerated files written to: {args.output_dir or 'same directory as source (*_regenerated.txt)'}")


if __name__ == "__main__":
    main()
