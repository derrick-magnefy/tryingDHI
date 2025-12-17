#!/usr/bin/env python3
"""
Analyze .pdn, -SG.txt, and -Ti.txt file structures.
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
    """Load data from single-line files like -A.txt, -SG.txt, -Ti.txt"""
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return np.array(values)

def analyze_sg_file(prefix):
    """Analyze the -SG.txt (settings) file structure."""
    sg_file = os.path.join(DATA_DIR, f"{prefix}-SG.txt")

    if not os.path.exists(sg_file):
        print(f"  SG file not found")
        return None

    settings = load_single_line_data(sg_file)

    print(f"\n{'='*70}")
    print(f"SG FILE ANALYSIS: {prefix}")
    print(f"{'='*70}")
    print(f"Number of settings: {len(settings)}")
    print(f"\nAll settings values:")

    # Known or guessed setting meanings
    setting_names = [
        "Voltage range/scale",     # 0
        "Acquisition time (s)",    # 1
        "Number of waveforms",     # 2
        "Mode/trigger setting",    # 3
        "Unknown",                 # 4
        "Unknown",                 # 5
        "Unknown",                 # 6
        "Unknown",                 # 7
        "Unknown",                 # 8
        "AC frequency (Hz)",       # 9
        "Sample interval (s)",     # 10
        "Unknown",                 # 11
        "Unknown",                 # 12
        "Unknown",                 # 13
        "Unknown",                 # 14
        "Processing mode",         # 15
        "Unknown",                 # 16
        "Unknown",                 # 17
    ]

    for i, val in enumerate(settings):
        name = setting_names[i] if i < len(setting_names) else "Unknown"
        print(f"  [{i:2d}] {val:15.6e}  ({name})")

    return settings

def analyze_ti_file(prefix):
    """Analyze the -Ti.txt (timing) file structure."""
    ti_file = os.path.join(DATA_DIR, f"{prefix}-Ti.txt")
    wfm_file = os.path.join(DATA_DIR, f"{prefix}-WFMs.txt")
    sg_file = os.path.join(DATA_DIR, f"{prefix}-SG.txt")

    if not os.path.exists(ti_file):
        print(f"  Ti file not found")
        return None

    ti_data = load_single_line_data(ti_file)
    waveforms = load_waveforms(wfm_file)
    settings = load_single_line_data(sg_file) if os.path.exists(sg_file) else None

    print(f"\n{'='*70}")
    print(f"Ti FILE ANALYSIS: {prefix}")
    print(f"{'='*70}")
    print(f"Number of Ti values: {len(ti_data)}")
    print(f"Number of waveforms: {len(waveforms)}")

    print(f"\nFirst 10 Ti values:")
    for i in range(min(10, len(ti_data))):
        print(f"  [{i}] {ti_data[i]:.6e} seconds")

    print(f"\nLast 10 Ti values:")
    for i in range(max(0, len(ti_data)-10), len(ti_data)):
        print(f"  [{i}] {ti_data[i]:.6e} seconds")

    # Analyze time differences
    if len(ti_data) > 1:
        diffs = np.diff(ti_data)
        print(f"\nTime difference analysis:")
        print(f"  Min diff: {np.min(diffs):.6e} s")
        print(f"  Max diff: {np.max(diffs):.6e} s")
        print(f"  Mean diff: {np.mean(diffs):.6e} s")
        print(f"  Std diff: {np.std(diffs):.6e} s")

        # Check if uniform
        if np.std(diffs) < np.mean(diffs) * 0.01:
            print(f"  => Time intervals appear UNIFORM")
        else:
            print(f"  => Time intervals are NON-UNIFORM (triggered events)")

    # Relationship to waveforms
    print(f"\n  Does len(Ti) == len(waveforms)? {len(ti_data) == len(waveforms)}")

    # Check if Ti represents trigger times
    if settings is not None:
        ac_freq = settings[9]
        ac_period = 1.0 / ac_freq
        print(f"\n  AC frequency: {ac_freq} Hz (period: {ac_period:.6e} s)")
        print(f"  Total time span: {ti_data[-1] - ti_data[0]:.6e} s")
        print(f"  Expected cycles: {(ti_data[-1] - ti_data[0]) * ac_freq:.1f}")

    return ti_data

def analyze_pdn_vs_wfm(pdn_path, wfm_path):
    """Compare .pdn file with corresponding WFMs.txt file."""
    print(f"\n{'='*70}")
    print(f"PDN vs WFM COMPARISON")
    print(f"{'='*70}")

    # Load WFM data
    waveforms = load_waveforms(wfm_path)
    num_wfm = len(waveforms)
    samples_per_wfm = len(waveforms[0]) if waveforms else 0
    total_samples = num_wfm * samples_per_wfm

    print(f"WFM file: {os.path.basename(wfm_path)}")
    print(f"  Waveforms: {num_wfm}")
    print(f"  Samples per waveform: {samples_per_wfm}")
    print(f"  Total samples: {total_samples}")

    # Analyze PDN file
    pdn_size = os.path.getsize(pdn_path)
    print(f"\nPDN file: {os.path.basename(pdn_path)}")
    print(f"  Size: {pdn_size} bytes")
    print(f"  Expected size (float32): {total_samples * 4} bytes")
    print(f"  Difference: {pdn_size - total_samples * 4} bytes (potential header)")

    with open(pdn_path, 'rb') as f:
        pdn_data = f.read()

    # Try to find WFM values in PDN file
    print(f"\nSearching for WFM values in PDN file...")

    # First waveform first few values
    test_values = waveforms[0][:5]
    print(f"  Looking for first waveform values: {test_values}")

    # Try different endianness and offsets
    for offset in [0, 4, 8, 16, 32, 64, 100, 108, 128, 256, 512, 1000]:
        if offset + total_samples * 4 <= len(pdn_data):
            floats = struct.unpack(f'<{min(100, total_samples)}f',
                                   pdn_data[offset:offset + min(100*4, total_samples*4)])
            floats = np.array(floats)

            # Check if first few values match
            if np.allclose(floats[:5], test_values, rtol=1e-5):
                print(f"  MATCH FOUND at offset {offset}!")
                print(f"  First 10 values: {floats[:10]}")
                return offset

            # Also check correlation
            if len(waveforms[0]) <= len(floats):
                corr = np.corrcoef(waveforms[0][:len(floats)], floats[:len(waveforms[0])])[0, 1]
                if abs(corr) > 0.9:
                    print(f"  High correlation at offset {offset}: {corr:.4f}")

    print("  No direct match found for float32 little-endian")

    # Try big-endian
    print("\n  Trying big-endian...")
    for offset in [0, 4, 8, 100]:
        if offset + 100*4 <= len(pdn_data):
            floats = struct.unpack(f'>{min(100, total_samples)}f',
                                   pdn_data[offset:offset + min(100*4, total_samples*4)])
            floats = np.array(floats)
            if np.allclose(floats[:5], test_values, rtol=1e-5):
                print(f"  MATCH FOUND (big-endian) at offset {offset}!")
                return offset

    return None

# Main analysis
if __name__ == "__main__":
    # Analyze SG and Ti files for all datasets
    print("\n" + "#"*70)
    print("COMPARING SG FILES ACROSS DATASETS")
    print("#"*70)

    prefixes = [
        "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1",
        "AC Motor 1.5 kV _Ch2_2025-10-31_11-39-24_2",
        "AC Motor 1.5 kV _Ch3_2025-10-31_11-45-29_3",
        "AC Corona-9.3 kV- Ch1_2025-10-31_12-32-36_1",
        "AC Corona-9.3 kV- Ch2_2025-10-31_12-37-52_2",
        "AC Corona-9.3 kV- Ch3_2025-10-31_12-53-33_3",
    ]

    all_settings = {}
    for prefix in prefixes:
        settings = analyze_sg_file(prefix)
        if settings is not None:
            all_settings[prefix] = settings

    # Compare settings
    print("\n" + "#"*70)
    print("SETTINGS COMPARISON TABLE")
    print("#"*70)

    if all_settings:
        print("\n{:5s} | {:>12s} | {:>12s} | {:>12s} | {:>12s} | {:>12s} | {:>12s}".format(
            "Idx", "Motor Ch1", "Motor Ch2", "Motor Ch3", "Corona Ch1", "Corona Ch2", "Corona Ch3"))
        print("-" * 90)

        num_settings = max(len(s) for s in all_settings.values())
        short_names = ["M_Ch1", "M_Ch2", "M_Ch3", "C_Ch1", "C_Ch2", "C_Ch3"]

        for i in range(num_settings):
            row = [f"{i:5d}"]
            for prefix in prefixes:
                if prefix in all_settings and i < len(all_settings[prefix]):
                    val = all_settings[prefix][i]
                    row.append(f"{val:12.4g}")
                else:
                    row.append(f"{'N/A':>12s}")
            print(" | ".join(row))

    # Analyze Ti files
    print("\n" + "#"*70)
    print("Ti FILE ANALYSIS")
    print("#"*70)

    for prefix in prefixes[:2]:  # Just analyze first two for brevity
        analyze_ti_file(prefix)

    # Compare PDN with WFM
    print("\n" + "#"*70)
    print("PDN FILE ANALYSIS")
    print("#"*70)

    # Motor Ch1 PDN
    pdn_path = os.path.join(DATA_DIR, "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-27_1.pdn")
    wfm_path = os.path.join(DATA_DIR, "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1-WFMs.txt")
    if os.path.exists(pdn_path) and os.path.exists(wfm_path):
        analyze_pdn_vs_wfm(pdn_path, wfm_path)
