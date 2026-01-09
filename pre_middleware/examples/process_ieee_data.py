#!/usr/bin/env python3
"""
Example: Processing IEEE Data Files

This script demonstrates how to process continuous raw PD data streams
from IEEE .mat files into triggered waveform format compatible with
the PD analysis pipeline.

IEEE Data files typically contain:
- Multiple channels (Ch1, Ch2, Ch3, Ch4)
- Sample interval (dt) instead of sample rate
- Continuous data without hardware triggers

The pre_middleware module handles:
1. Loading multi-channel .mat files
2. Detecting trigger points using configurable methods
3. Extracting waveforms around triggers
4. Saving in Rugged Data Files format

Usage:
    python -m pre_middleware.examples.process_ieee_data

    # Or from command line with custom settings:
    python -m pre_middleware.process_raw_stream "IEEE Data/dataset.mat" \\
        --channel Ch1 \\
        --method histogram_knee \\
        --output-dir output/
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pre_middleware import (
    TriggerDetector,
    WaveformExtractor,
    process_raw_stream,
    compare_methods,
    TRIGGER_METHODS,
    DEFAULT_TRIGGER_METHOD,
)
from pre_middleware.loaders import MatLoader


def explore_ieee_file(filepath: str):
    """Explore the structure of an IEEE .mat file."""
    print(f"\n{'='*60}")
    print(f"Exploring: {filepath}")
    print(f"{'='*60}")

    loader = MatLoader(filepath)
    info = loader.get_info()

    print(f"\nFile format: {info['format']}")
    print(f"\nVariables found:")
    print("-" * 40)

    for name, details in info['variables'].items():
        if 'shape' in details:
            print(f"  {name}: {details['dtype']} {details['shape']}")
        else:
            print(f"  {name}: {details['type']} = {details.get('value', 'complex')}")

    # List channels
    channels = loader.list_channels()
    if channels:
        print(f"\nAvailable channels: {channels}")

    return loader


def demonstrate_trigger_methods(signal, sample_rate, ac_frequency=60.0):
    """Compare different trigger detection methods."""
    print(f"\n{'='*60}")
    print("Comparing Trigger Detection Methods")
    print(f"{'='*60}")

    # Estimate noise floor first
    detector = TriggerDetector()
    noise_stats = detector.estimate_noise_floor(signal)

    print(f"\nSignal characteristics:")
    print(f"  Samples: {len(signal):,}")
    print(f"  Duration: {len(signal)/sample_rate*1000:.2f} ms")
    print(f"  Noise floor (stdev): {noise_stats['stdev_estimate']:.4e}")
    print(f"  Peak-to-noise ratio: {noise_stats['peak_to_noise']:.1f}")

    # Calculate number of AC cycles
    duration = len(signal) / sample_rate
    num_cycles = duration * ac_frequency

    print(f"\n{'Method':<20} {'Threshold':>12} {'Triggers':>10} {'Rate/cycle':>12}")
    print("-" * 56)

    results = compare_methods(signal, sample_rate, ac_frequency)

    for method, result in results.items():
        rate = len(result.triggers) / num_cycles if num_cycles > 0 else 0
        print(f"{method:<20} {result.threshold:>12.4e} {len(result.triggers):>10,} {rate:>12.1f}")

    return results


def process_single_channel(
    filepath: str,
    channel: str,
    output_dir: str,
    method: str = DEFAULT_TRIGGER_METHOD,
    pre_samples: int = 62,
    post_samples: int = 188,
):
    """Process a single channel from an IEEE data file."""
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"Channel: {channel}")
    print(f"Method: {method}")
    print(f"{'='*60}")

    result = process_raw_stream(
        filepath=filepath,
        output_dir=output_dir,
        output_prefix=f"{Path(filepath).stem}_{channel}",
        trigger_method=method,
        pre_samples=pre_samples,
        post_samples=post_samples,
        signal_var=channel,
        verbose=True,
    )

    return result


def process_all_channels(
    filepath: str,
    output_dir: str,
    method: str = DEFAULT_TRIGGER_METHOD,
    pre_samples: int = 62,
    post_samples: int = 188,
):
    """Process all channels in an IEEE data file."""
    loader = MatLoader(filepath)
    channels = loader.list_channels()

    if not channels:
        print("No channels found in file.")
        return {}

    print(f"\nProcessing {len(channels)} channels: {channels}")

    results = {}
    for channel in channels:
        try:
            result = process_single_channel(
                filepath=filepath,
                channel=channel,
                output_dir=output_dir,
                method=method,
                pre_samples=pre_samples,
                post_samples=post_samples,
            )
            results[channel] = result
        except Exception as e:
            print(f"Error processing {channel}: {e}")
            results[channel] = {'status': 'error', 'error': str(e)}

    return results


def main():
    """Main example demonstrating IEEE data processing."""
    # Path to example IEEE data file
    ieee_file = "IEEE Data/Euller_Aq03_1.mat"

    if not os.path.exists(ieee_file):
        print(f"IEEE data file not found: {ieee_file}")
        print("Please ensure the 'IEEE Data' folder contains .mat files.")
        return

    # Step 1: Explore the file structure
    loader = explore_ieee_file(ieee_file)

    # Step 2: Load and examine data
    print(f"\n{'='*60}")
    print("Loading Data")
    print(f"{'='*60}")

    data = loader.load()
    print(f"\nLoaded signal: {data['signal_var']}")
    print(f"  Length: {len(data['signal']):,} samples")
    print(f"  Sample rate: {data['sample_rate']/1e6:.2f} MHz")
    print(f"  Source: {data['sample_rate_source']}")

    # Step 3: Compare trigger methods
    demonstrate_trigger_methods(
        data['signal'],
        data['sample_rate'],
        data['ac_frequency']
    )

    # Step 4: Process with default settings
    output_dir = "output/ieee_processed"
    print(f"\n{'='*60}")
    print("Processing with Default Settings")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    result = process_single_channel(
        filepath=ieee_file,
        channel='Ch1',
        output_dir=output_dir,
        method='histogram_knee',  # Default - often most robust
        pre_samples=62,   # 25% of 2µs @ 125 MSPS
        post_samples=188,  # 75% of 2µs @ 125 MSPS
    )

    if result['status'] == 'success':
        print(f"\nSuccessfully extracted {result['num_waveforms']} waveforms")
        print(f"Output files in: {result['output_dir']}")

    # Step 5: Show CLI usage examples
    print(f"\n{'='*60}")
    print("CLI Usage Examples")
    print(f"{'='*60}")
    print("""
# List available channels
python -m pre_middleware.process_raw_stream "IEEE Data/Euller_Aq03_1.mat" --list-channels

# Process Ch1 with histogram_knee method (default)
python -m pre_middleware.process_raw_stream "IEEE Data/Euller_Aq03_1.mat" \\
    -c Ch1 -o output/ -m histogram_knee

# Process with pulse rate targeting (max 50 pulses per AC cycle)
python -m pre_middleware.process_raw_stream "IEEE Data/Euller_Aq03_1.mat" \\
    -c Ch1 -o output/ -m pulse_rate --target-rate 50

# Process with standard deviation method (5 sigma threshold)
python -m pre_middleware.process_raw_stream "IEEE Data/Euller_Aq03_1.mat" \\
    -c Ch1 -o output/ -m stdev -k 5

# Compare all methods
python -m pre_middleware.process_raw_stream "IEEE Data/Euller_Aq03_1.mat" --compare-methods

# Adjust trigger window
python -m pre_middleware.process_raw_stream "IEEE Data/Euller_Aq03_1.mat" \\
    -c Ch1 -o output/ --pre 500 --post 1500
""")


if __name__ == '__main__':
    main()
