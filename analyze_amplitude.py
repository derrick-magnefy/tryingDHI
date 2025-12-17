#!/usr/bin/env python3
"""
Script to analyze and reverse-engineer the amplitude calculation method
for partial discharge waveform data.
"""

import numpy as np
import os

# Data directory
DATA_DIR = "Rugged Data Files"

# Test file prefix
TEST_PREFIX = "AC Motor 1.5 kV _Ch1_2025-10-31_11-35-39_1"

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
    """Load data from single-line files like -A.txt, -P.txt"""
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return np.array(values)

def load_settings(filepath):
    """Load settings from -SG.txt file"""
    with open(filepath, 'r') as f:
        content = f.read().strip()
        values = [float(v) for v in content.split('\t') if v.strip()]
    return values

# Amplitude calculation methods
def method_max(wfm):
    """Maximum value"""
    return np.max(wfm)

def method_min(wfm):
    """Minimum value"""
    return np.min(wfm)

def method_abs_max(wfm):
    """Absolute maximum value"""
    return np.max(np.abs(wfm))

def method_signed_abs_max(wfm):
    """Signed value of absolute maximum (preserves polarity)"""
    idx = np.argmax(np.abs(wfm))
    return wfm[idx]

def method_peak_to_peak(wfm):
    """Peak-to-peak amplitude"""
    return np.max(wfm) - np.min(wfm)

def method_mean_subtracted_max(wfm):
    """Max value minus mean baseline"""
    baseline = np.mean(wfm)
    return np.max(wfm) - baseline

def method_mean_subtracted_signed_max(wfm):
    """Signed max deviation from mean baseline"""
    baseline = np.mean(wfm)
    deviations = wfm - baseline
    idx = np.argmax(np.abs(deviations))
    return deviations[idx]

def method_first_n_baseline_max(wfm, n=100):
    """Max deviation from baseline computed from first n samples"""
    baseline = np.mean(wfm[:n])
    deviations = wfm - baseline
    idx = np.argmax(np.abs(deviations))
    return deviations[idx]

def method_first_n_baseline_max_50(wfm):
    return method_first_n_baseline_max(wfm, n=50)

def method_first_n_baseline_max_200(wfm):
    return method_first_n_baseline_max(wfm, n=200)

def method_median_subtracted_signed_max(wfm):
    """Signed max deviation from median baseline"""
    baseline = np.median(wfm)
    deviations = wfm - baseline
    idx = np.argmax(np.abs(deviations))
    return deviations[idx]

def method_mode_baseline_signed_max(wfm, bins=100):
    """Signed max deviation from mode (most common value) baseline"""
    hist, bin_edges = np.histogram(wfm, bins=bins)
    mode_idx = np.argmax(hist)
    baseline = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    deviations = wfm - baseline
    idx = np.argmax(np.abs(deviations))
    return deviations[idx]

def method_rms(wfm):
    """RMS value"""
    return np.sqrt(np.mean(wfm**2))

def method_percentile_baseline_max(wfm, percentile=10):
    """Max deviation from percentile-based baseline"""
    baseline = np.percentile(wfm, percentile)
    deviations = wfm - baseline
    idx = np.argmax(np.abs(deviations))
    return deviations[idx]

def compute_correlation(actual, computed):
    """Compute correlation between actual and computed amplitudes"""
    return np.corrcoef(actual, computed)[0, 1]

def compute_rmse(actual, computed):
    """Compute RMSE between actual and computed amplitudes"""
    return np.sqrt(np.mean((actual - computed)**2))

def compute_max_error(actual, computed):
    """Compute maximum absolute error"""
    return np.max(np.abs(actual - computed))

def main():
    # Load data
    wfm_file = os.path.join(DATA_DIR, f"{TEST_PREFIX}-WFMs.txt")
    a_file = os.path.join(DATA_DIR, f"{TEST_PREFIX}-A.txt")
    sg_file = os.path.join(DATA_DIR, f"{TEST_PREFIX}-SG.txt")

    print(f"Loading waveforms from: {wfm_file}")
    waveforms = load_waveforms(wfm_file)
    print(f"Loaded {len(waveforms)} waveforms")
    print(f"First waveform has {len(waveforms[0])} samples")

    print(f"\nLoading actual amplitudes from: {a_file}")
    actual_amplitudes = load_single_line_data(a_file)
    print(f"Loaded {len(actual_amplitudes)} amplitude values")

    print(f"\nLoading settings from: {sg_file}")
    settings = load_settings(sg_file)
    print(f"Settings: {settings}")

    # Analyze first waveform
    wfm0 = waveforms[0]
    print(f"\n=== First Waveform Analysis ===")
    print(f"Length: {len(wfm0)}")
    print(f"Min: {np.min(wfm0):.6e}")
    print(f"Max: {np.max(wfm0):.6e}")
    print(f"Mean: {np.mean(wfm0):.6e}")
    print(f"Median: {np.median(wfm0):.6e}")
    print(f"Std: {np.std(wfm0):.6e}")
    print(f"Actual A value: {actual_amplitudes[0]:.6e}")

    # Define methods to test
    methods = [
        ("max", method_max),
        ("min", method_min),
        ("abs_max", method_abs_max),
        ("signed_abs_max", method_signed_abs_max),
        ("peak_to_peak", method_peak_to_peak),
        ("mean_subtracted_max", method_mean_subtracted_max),
        ("mean_subtracted_signed_max", method_mean_subtracted_signed_max),
        ("first_50_baseline_max", method_first_n_baseline_max_50),
        ("first_100_baseline_max", lambda w: method_first_n_baseline_max(w, 100)),
        ("first_200_baseline_max", method_first_n_baseline_max_200),
        ("median_subtracted_signed_max", method_median_subtracted_signed_max),
        ("mode_baseline_signed_max", method_mode_baseline_signed_max),
        ("rms", method_rms),
        ("percentile_10_baseline", lambda w: method_percentile_baseline_max(w, 10)),
        ("percentile_50_baseline", lambda w: method_percentile_baseline_max(w, 50)),
    ]

    print("\n=== Testing Methods on First Waveform ===")
    for name, method in methods:
        try:
            result = method(wfm0)
            diff = result - actual_amplitudes[0]
            print(f"{name:35s}: {result:12.6e}  (diff: {diff:+12.6e})")
        except Exception as e:
            print(f"{name:35s}: ERROR - {e}")

    # Test all methods on all waveforms
    print("\n=== Full Dataset Comparison ===")
    print(f"{'Method':<35s} {'Correlation':>12s} {'RMSE':>14s} {'Max Error':>14s}")
    print("-" * 80)

    results = []
    for name, method in methods:
        try:
            computed = np.array([method(wfm) for wfm in waveforms])
            corr = compute_correlation(actual_amplitudes, computed)
            rmse = compute_rmse(actual_amplitudes, computed)
            max_err = compute_max_error(actual_amplitudes, computed)
            print(f"{name:<35s} {corr:12.8f} {rmse:14.6e} {max_err:14.6e}")
            results.append((name, corr, rmse, max_err, computed))
        except Exception as e:
            print(f"{name:<35s} ERROR: {e}")

    # Find best method
    print("\n=== Best Method Analysis ===")

    # Sort by correlation (highest first)
    results.sort(key=lambda x: -x[1])
    print("\nTop 3 by correlation:")
    for name, corr, rmse, max_err, _ in results[:3]:
        print(f"  {name}: correlation={corr:.8f}, RMSE={rmse:.6e}")

    # Sort by RMSE (lowest first)
    results.sort(key=lambda x: x[2])
    print("\nTop 3 by RMSE:")
    for name, corr, rmse, max_err, _ in results[:3]:
        print(f"  {name}: RMSE={rmse:.6e}, correlation={corr:.8f}")

    # If perfect match found
    for name, corr, rmse, max_err, computed in results:
        if rmse < 1e-12:  # Essentially zero error
            print(f"\n*** PERFECT MATCH FOUND: {name} ***")
            print(f"    RMSE: {rmse:.6e}")
            print(f"    Max Error: {max_err:.6e}")
            break

    # Sample comparison for best method
    best_name, best_corr, best_rmse, best_max_err, best_computed = results[0]
    print(f"\n=== Sample Comparison for Best Method: {best_name} ===")
    print(f"{'Index':<8s} {'Actual':>14s} {'Computed':>14s} {'Difference':>14s}")
    for i in [0, 1, 2, 10, 100, 500, 1000, 1500, 1999]:
        if i < len(actual_amplitudes):
            diff = best_computed[i] - actual_amplitudes[i]
            print(f"{i:<8d} {actual_amplitudes[i]:14.6e} {best_computed[i]:14.6e} {diff:+14.6e}")

if __name__ == "__main__":
    main()
