#!/usr/bin/env python3
"""
Decode PDN files and analyze settings correlation with amplitude methods.
"""

import struct
import os
import glob
import numpy as np

DATA_DIR = "Rugged Data Files"

def load_waveforms(filepath):
    """Load waveform data from -WFMs.txt file"""
    waveforms = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(v) for v in line.split('\t') if v.strip()]
                waveforms.append(np.array(values))
    return waveforms

def load_single_line_data(filepath):
    """Load data from single-line files"""
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return np.array(values)

def decode_pdn_file(pdn_path):
    """Decode a .pdn file structure."""
    print(f"\n{'='*70}")
    print(f"DECODING PDN FILE: {os.path.basename(pdn_path)}")
    print(f"{'='*70}")

    with open(pdn_path, 'rb') as f:
        data = f.read()

    filesize = len(data)
    print(f"File size: {filesize} bytes")

    # Decode header (first 8 bytes)
    print(f"\nHeader analysis (first 8 bytes):")

    # Try different interpretations of the header
    header = data[:8]
    print(f"  Raw bytes: {header.hex()}")

    # As 2 big-endian 32-bit integers
    h1, h2 = struct.unpack('>ii', header)
    print(f"  As 2 x int32 (big-endian): {h1}, {h2}")

    # As 2 little-endian 32-bit integers
    h1_le, h2_le = struct.unpack('<ii', header)
    print(f"  As 2 x int32 (little-endian): {h1_le}, {h2_le}")

    # As 2 big-endian 32-bit floats
    f1, f2 = struct.unpack('>ff', header)
    print(f"  As 2 x float32 (big-endian): {f1}, {f2}")

    # Check if header values match expected dimensions
    data_size = filesize - 8
    num_floats = data_size // 4
    print(f"\n  Data size (after header): {data_size} bytes")
    print(f"  Number of floats: {num_floats}")
    print(f"  If 500 samples: {num_floats // 500} waveforms")
    print(f"  If 2000 waveforms: {num_floats // 2000} samples per wfm")

    # Decode waveform data (big-endian floats after 8-byte header)
    print(f"\nWaveform data (big-endian float32, offset 8):")
    floats = struct.unpack(f'>{num_floats}f', data[8:8+num_floats*4])
    floats = np.array(floats)

    print(f"  First 20 values: {floats[:20]}")
    print(f"  Value range: [{floats.min():.6e}, {floats.max():.6e}]")
    print(f"  Mean: {floats.mean():.6e}")

    return floats, (h1, h2)

def compare_pdn_with_wfm(pdn_path, wfm_path):
    """Compare PDN data with WFM text file."""
    print(f"\n{'='*70}")
    print(f"COMPARING PDN WITH WFM")
    print(f"{'='*70}")

    # Load WFM text file
    waveforms = load_waveforms(wfm_path)
    wfm_flat = np.concatenate(waveforms)

    # Decode PDN
    with open(pdn_path, 'rb') as f:
        data = f.read()

    num_floats = (len(data) - 8) // 4
    pdn_floats = struct.unpack(f'>{num_floats}f', data[8:8+num_floats*4])
    pdn_floats = np.array(pdn_floats)

    print(f"WFM file: {len(waveforms)} waveforms × {len(waveforms[0])} samples = {len(wfm_flat)} total")
    print(f"PDN file: {num_floats} floats")

    # Compare values
    min_len = min(len(wfm_flat), len(pdn_floats))
    print(f"\nComparing first {min_len} values:")

    if np.allclose(wfm_flat[:min_len], pdn_floats[:min_len], rtol=1e-5):
        print("  [OK] Values match exactly!")
    else:
        diff = np.abs(wfm_flat[:min_len] - pdn_floats[:min_len])
        print(f"  Max difference: {diff.max():.6e}")
        print(f"  Mean difference: {diff.mean():.6e}")
        mismatches = np.where(diff > 1e-6)[0]
        if len(mismatches) > 0:
            print(f"  First mismatch at index {mismatches[0]}:")
            i = mismatches[0]
            print(f"    WFM: {wfm_flat[i]:.6e}")
            print(f"    PDN: {pdn_floats[i]:.6e}")

    # Check correlation
    corr = np.corrcoef(wfm_flat[:min_len], pdn_floats[:min_len])[0, 1]
    print(f"  Correlation: {corr:.8f}")

    return pdn_floats, wfm_flat

def analyze_settings_vs_amplitude():
    """Analyze correlation between settings and amplitude calculation method."""
    print(f"\n{'#'*70}")
    print("SETTINGS VS AMPLITUDE METHOD CORRELATION")
    print(f"{'#'*70}")

    prefixes = [
        "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1",
        "AC Motor 1.5 kV _Ch2_2025-10-31_11-39-24_2",
        "AC Motor 1.5 kV _Ch3_2025-10-31_11-45-29_3",
        "AC Corona-9.3 kV- Ch1_2025-10-31_12-32-36_1",
        "AC Corona-9.3 kV- Ch2_2025-10-31_12-37-52_2",
        "AC Corona-9.3 kV- Ch3_2025-10-31_12-53-33_3",
    ]

    print("\n{:20s} | {:>8s} | {:>8s} | {:>10s} | {:>12s}".format(
        "Dataset", "Set[3]", "Set[15]", "Match %", "Best Method"))
    print("-" * 70)

    for prefix in prefixes:
        sg_file = os.path.join(DATA_DIR, f"{prefix}-SG.txt")
        wfm_file = os.path.join(DATA_DIR, f"{prefix}-WFMs.txt")
        a_file = os.path.join(DATA_DIR, f"{prefix}-A.txt")

        if not all(os.path.exists(f) for f in [sg_file, wfm_file, a_file]):
            continue

        settings = load_single_line_data(sg_file)
        waveforms = load_waveforms(wfm_file)
        actual_a = load_single_line_data(a_file)

        # Test signed_abs_max
        computed_a = np.array([wfm[np.argmax(np.abs(wfm))] for wfm in waveforms])
        matches = np.sum(np.isclose(actual_a, computed_a, rtol=1e-10, atol=1e-15))
        match_pct = 100 * matches / len(actual_a)

        # Determine best method based on match %
        if match_pct == 100:
            best = "signed_abs_max"
        elif match_pct > 90:
            best = "~signed_abs_max"
        else:
            best = "different"

        short_name = prefix.split("_")[0][:10] + " " + prefix.split("_")[1][:5]
        print(f"{short_name:20s} | {settings[3]:8.0f} | {settings[15]:8.0f} | {match_pct:10.1f} | {best}")

    print("\n" + "="*70)
    print("KEY FINDING:")
    print("="*70)
    print("""
    settings[15] = 2: Uses signed_abs_max method (100% match)
    settings[15] = 0 or 1: Uses a different amplitude calculation method

    settings[3] may also affect processing:
    - 0: Standard mode
    - 1: Different trigger/processing
    - 4: Different trigger/processing

    The signed_abs_max method works perfectly when settings[15] = 2.
    """)

def analyze_ti_generation():
    """Analyze how Ti (trigger time) values are generated."""
    print(f"\n{'#'*70}")
    print("Ti FILE GENERATION ANALYSIS")
    print(f"{'#'*70}")

    prefix = "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1"
    ti_file = os.path.join(DATA_DIR, f"{prefix}-Ti.txt")
    sg_file = os.path.join(DATA_DIR, f"{prefix}-SG.txt")
    p_file = os.path.join(DATA_DIR, f"{prefix}-P.txt")

    ti_data = load_single_line_data(ti_file)
    settings = load_single_line_data(sg_file)
    phase_data = load_single_line_data(p_file)

    ac_freq = settings[9]
    ac_period = 1.0 / ac_freq

    print(f"\nAC Period: {ac_period*1000:.4f} ms ({ac_freq} Hz)")
    print(f"Number of trigger times: {len(ti_data)}")

    # Calculate phase from trigger time
    print(f"\nPhase calculation verification:")
    print(f"{'Idx':>5} {'Ti (ms)':>12} {'Calc Phase':>12} {'Actual P':>12} {'Diff':>10}")
    print("-" * 55)

    for i in [0, 1, 2, 3, 10, 100, 500, 1000]:
        if i < len(ti_data):
            ti = ti_data[i]
            # Calculate phase: (ti % ac_period) / ac_period * 360
            phase_calc = (ti % ac_period) / ac_period * 360
            actual_phase = phase_data[i] if i < len(phase_data) else 0
            diff = phase_calc - actual_phase

            print(f"{i:5d} {ti*1000:12.4f} {phase_calc:12.2f} {actual_phase:12.2f} {diff:10.2f}")

    print(f"\nTi FILE GENERATION:")
    print("""
    The Ti.txt file contains the trigger time (in seconds) for each PD event.
    These are the actual times when the partial discharge pulses were detected
    and captured by the acquisition system.

    Ti values are NON-UNIFORM because they represent actual triggered events,
    not a continuous sampling. The PD events occur at random times, but
    their phase relationship to the AC cycle is preserved.

    The P.txt file contains the phase angle (in degrees) of each event,
    calculated from: Phase = (Ti mod AC_period) / AC_period × 360°
    """)

if __name__ == "__main__":
    # Decode Motor Ch1 PDN file
    pdn_path = os.path.join(DATA_DIR, "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-27_1.pdn")
    wfm_path = os.path.join(DATA_DIR, "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1-WFMs.txt")

    if os.path.exists(pdn_path):
        pdn_floats, header = decode_pdn_file(pdn_path)

        if os.path.exists(wfm_path):
            compare_pdn_with_wfm(pdn_path, wfm_path)

    # Analyze settings correlation
    analyze_settings_vs_amplitude()

    # Analyze Ti generation
    analyze_ti_generation()
