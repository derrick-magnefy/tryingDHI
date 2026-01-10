#!/usr/bin/env python3
"""
Process Raw PD Data Streams

Converts continuous raw data streams (e.g., IEEE .mat files) into
triggered waveform format compatible with the PD analysis pipeline.

This script:
1. Loads raw continuous data from various formats
2. Detects trigger points using configurable methods
3. Extracts waveforms around each trigger
4. Saves in Rugged Data Files format for downstream processing

Trigger Detection Methods:
- stdev: Threshold = k * standard deviation (default k=5)
- pulse_rate: Adaptive threshold targeting max pulses per AC cycle
- histogram_knee: Find knee in amplitude histogram (often most robust)

Usage:
    # Basic usage with defaults
    python -m pre_middleware.triggerProc.process_raw_stream input.mat --output-dir output/

    # Specify trigger method
    python -m pre_middleware.triggerProc.process_raw_stream input.mat --method histogram_knee

    # Adjust trigger windows
    python -m pre_middleware.triggerProc.process_raw_stream input.mat --pre-samples 500 --post-samples 1500

    # Pulse rate targeting
    python -m pre_middleware.triggerProc.process_raw_stream input.mat --method pulse_rate --target-rate 50

Examples:
    # Process IEEE .mat file
    python -m pre_middleware.triggerProc.process_raw_stream "IEEE Data/dataset.mat" \\
        --output-dir "Rugged Data Files" \\
        --method histogram_knee \\
        --ac-frequency 60

    # Compare all methods
    python -m pre_middleware.triggerProc.process_raw_stream "IEEE Data/dataset.mat" --compare-methods
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .trigger_detection import (
    TriggerDetector,
    TRIGGER_METHODS,
    DEFAULT_TRIGGER_METHOD,
    compare_methods as compare_trigger_methods,
)
from .waveform_extraction import WaveformExtractor, ExtractionResult
from ..loaders import MatLoader


def process_raw_stream(
    filepath: str,
    output_dir: str,
    output_prefix: Optional[str] = None,
    trigger_method: str = DEFAULT_TRIGGER_METHOD,
    pre_samples: int = 50,
    post_samples: int = 200,
    ac_frequency: float = 60.0,
    polarity: str = 'both',
    min_separation: int = 100,
    dead_time_us: Optional[float] = None,
    refine_to_onset: bool = False,
    refine_to_peak: bool = False,
    validate_peak_position: bool = False,
    peak_tolerance: float = 0.5,
    signal_var: Optional[str] = None,
    sample_rate_var: Optional[str] = None,
    verbose: bool = True,
    **trigger_kwargs
) -> Dict[str, Any]:
    """
    Process a raw data stream file and convert to Rugged format.

    Args:
        filepath: Path to input file (.mat, etc.)
        output_dir: Output directory for Rugged format files
        output_prefix: Prefix for output files (default: input filename)
        trigger_method: Detection method ('stdev', 'pulse_rate', 'histogram_knee')
        pre_samples: Samples before trigger (default: 50)
        post_samples: Samples after trigger (default: 200)
        ac_frequency: AC power frequency in Hz (default: 60)
        polarity: Trigger polarity ('positive', 'negative', 'both')
        min_separation: Minimum samples between triggers (default: 100)
        dead_time_us: Dead time between triggers in microseconds (overrides min_separation if set)
        refine_to_onset: Adjust triggers backward to pulse onset (default: False)
        refine_to_peak: Adjust triggers forward to pulse peak (default: False)
        validate_peak_position: Discard waveforms where peak is far from trigger (default: False)
        peak_tolerance: Fraction of window where peak must appear (default: 0.5 = first half)
        signal_var: Variable name for signal in .mat file (auto-detect if None)
        sample_rate_var: Variable name for sample rate (auto-detect if None)
        verbose: Print progress messages
        **trigger_kwargs: Additional arguments for trigger detector

    Returns:
        Dict with processing results and statistics
    """
    filepath = Path(filepath)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {filepath.name}")
        print(f"{'='*60}")

    # Determine output prefix
    if output_prefix is None:
        output_prefix = filepath.stem

    # Load data
    if verbose:
        print(f"\n[1/4] Loading data...")

    if filepath.suffix.lower() == '.mat':
        loader = MatLoader(filepath, signal_var, sample_rate_var)
        data = loader.load()
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    signal = data.get('signal')
    sample_rate = data.get('sample_rate')

    # Validate loaded data
    if signal is None or len(signal) == 0:
        available_vars = list(data.get('metadata', {}).keys())
        raise ValueError(
            f"Could not find signal data in {filepath.name}. "
            f"Available variables: {available_vars}. "
            f"Try specifying --signal-var or --channel."
        )

    if sample_rate is None or sample_rate <= 0:
        raise ValueError(
            f"Could not determine sample rate for {filepath.name}. "
            f"Try specifying --sample-rate-var."
        )

    if verbose:
        print(f"  Signal length: {len(signal):,} samples")
        print(f"  Duration: {len(signal)/sample_rate:.3f} seconds")
        print(f"  Sample rate: {sample_rate/1e6:.2f} MHz")
        print(f"  AC frequency: {data.get('ac_frequency', ac_frequency)} Hz")

    # Convert dead_time_us to samples if specified
    if dead_time_us is not None:
        min_separation = int(dead_time_us * sample_rate / 1e6)
        if verbose:
            print(f"  Dead time: {dead_time_us} µs = {min_separation} samples")

    # Detect triggers
    if verbose:
        print(f"\n[2/4] Detecting triggers (method: {trigger_method})...")
        print(f"  Min separation: {min_separation} samples ({1e6 * min_separation / sample_rate:.2f} µs)")

    detector = TriggerDetector(
        method=trigger_method,
        polarity=polarity,
        min_separation=min_separation,
        refine_to_onset=refine_to_onset,
        refine_to_peak=refine_to_peak,
        **trigger_kwargs
    )
    trigger_result = detector.detect(signal, sample_rate, ac_frequency)

    if verbose:
        print(f"  Threshold: {trigger_result.threshold:.4e}")
        print(f"  Triggers found: {len(trigger_result.triggers):,}")
        if 'refinement' in trigger_result.stats:
            print(f"  Trigger refinement: {trigger_result.stats['refinement']}")
        if 'achieved_rate_per_cycle' in trigger_result.stats:
            print(f"  Rate per cycle: {trigger_result.stats['achieved_rate_per_cycle']:.1f}")

    if len(trigger_result.triggers) == 0:
        print("Warning: No triggers detected!")
        return {
            'status': 'no_triggers',
            'filepath': str(filepath),
            'trigger_result': trigger_result,
        }

    # Extract waveforms
    if verbose:
        print(f"\n[3/4] Extracting waveforms...")
        print(f"  Pre-trigger samples: {pre_samples}")
        print(f"  Post-trigger samples: {post_samples}")
        print(f"  Total waveform length: {pre_samples + post_samples}")

    extractor = WaveformExtractor(
        pre_samples, post_samples,
        validate_peak_position=validate_peak_position,
        peak_tolerance=peak_tolerance
    )
    extraction_result = extractor.extract(
        signal, trigger_result.triggers, sample_rate
    )

    if verbose:
        stats = extraction_result.stats
        print(f"  Extracted: {stats['extracted']:,} waveforms")
        if stats['skipped_start'] > 0:
            print(f"  Skipped (start): {stats['skipped_start']}")
        if stats['skipped_end'] > 0:
            print(f"  Skipped (end): {stats['skipped_end']}")
        if stats['skipped_overlap'] > 0:
            print(f"  Skipped (overlap): {stats['skipped_overlap']}")
        if stats.get('skipped_peak_position', 0) > 0:
            print(f"  Skipped (peak position): {stats['skipped_peak_position']}")

    # Save in Rugged format
    if verbose:
        print(f"\n[4/4] Saving Rugged format...")
        print(f"  Output directory: {output_dir}")
        print(f"  Prefix: {output_prefix}")

    files = extractor.save_rugged_format(
        extraction_result,
        output_dir,
        output_prefix,
        ac_frequency=ac_frequency,
    )

    if verbose:
        print(f"\n  Created files:")
        for file_type, file_path in files.items():
            print(f"    {file_type}: {os.path.basename(file_path)}")

    # Calculate phase distribution
    rugged_data = extractor.to_rugged_format(extraction_result, ac_frequency)
    phases = rugged_data['phases']

    if verbose:
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"  Waveforms extracted: {len(extraction_result.waveforms):,}")
        print(f"  Phase range: {phases.min():.1f}° - {phases.max():.1f}°")
        print(f"  Output: {output_dir}/{output_prefix}-*.txt")

    return {
        'status': 'success',
        'filepath': str(filepath),
        'output_dir': output_dir,
        'output_prefix': output_prefix,
        'files': files,
        'trigger_result': trigger_result,
        'extraction_result': extraction_result,
        'num_waveforms': len(extraction_result.waveforms),
        'sample_rate': sample_rate,
        'ac_frequency': ac_frequency,
    }


def compare_methods_report(
    filepath: str,
    signal_var: Optional[str] = None,
    sample_rate_var: Optional[str] = None,
    ac_frequency: float = 60.0,
) -> Dict[str, Any]:
    """
    Compare all trigger detection methods on a dataset.

    Args:
        filepath: Path to input file
        signal_var: Variable name for signal (auto-detect if None)
        sample_rate_var: Variable name for sample rate (auto-detect if None)
        ac_frequency: AC frequency in Hz

    Returns:
        Dict with comparison results
    """
    filepath = Path(filepath)

    print(f"\n{'='*60}")
    print(f"Comparing Trigger Methods: {filepath.name}")
    print(f"{'='*60}")

    # Load data
    if filepath.suffix.lower() == '.mat':
        loader = MatLoader(filepath, signal_var, sample_rate_var)
        data = loader.load()
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    signal = data['signal']
    sample_rate = data['sample_rate']

    print(f"\nSignal: {len(signal):,} samples ({len(signal)/sample_rate:.3f} s)")
    print(f"Sample rate: {sample_rate/1e6:.2f} MHz")

    # Estimate noise floor
    detector = TriggerDetector()
    noise_stats = detector.estimate_noise_floor(signal)
    print(f"\nNoise floor:")
    print(f"  Baseline: {noise_stats['baseline']:.4e}")
    print(f"  Stdev estimate: {noise_stats['stdev_estimate']:.4e}")
    print(f"  Peak-to-noise: {noise_stats['peak_to_noise']:.1f}")

    # Compare methods
    print(f"\n{'Method':<20} {'Threshold':>12} {'Triggers':>10} {'Rate/cycle':>12}")
    print("-" * 56)

    results = compare_trigger_methods(signal, sample_rate, ac_frequency)

    num_cycles = (len(signal) / sample_rate) * ac_frequency

    for method, result in results.items():
        rate = len(result.triggers) / num_cycles if num_cycles > 0 else 0
        print(f"{method:<20} {result.threshold:>12.4e} {len(result.triggers):>10,} {rate:>12.1f}")

    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Process raw PD data streams into triggered waveform format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings (histogram_knee method)
  python -m pre_middleware.triggerProc.process_raw_stream data.mat -o output/

  # Use pulse rate targeting method
  python -m pre_middleware.triggerProc.process_raw_stream data.mat -o output/ -m pulse_rate --target-rate 50

  # Adjust trigger window
  python -m pre_middleware.triggerProc.process_raw_stream data.mat -o output/ --pre 500 --post 1500

  # Compare all methods without processing
  python -m pre_middleware.triggerProc.process_raw_stream data.mat --compare-methods
        """
    )

    parser.add_argument(
        'input',
        help='Input file path (.mat file)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '-p', '--prefix',
        help='Output file prefix (default: input filename)'
    )

    # Trigger detection options
    parser.add_argument(
        '-m', '--method',
        choices=TRIGGER_METHODS,
        default=DEFAULT_TRIGGER_METHOD,
        help=f'Trigger detection method (default: {DEFAULT_TRIGGER_METHOD})'
    )
    parser.add_argument(
        '--polarity',
        choices=['positive', 'negative', 'both'],
        default='both',
        help='Trigger polarity (default: both)'
    )
    parser.add_argument(
        '--min-separation',
        type=int,
        default=100,
        help='Minimum samples between triggers (default: 100)'
    )
    parser.add_argument(
        '--dead-time', '--dead-time-us',
        type=float,
        dest='dead_time_us',
        help='Dead time between triggers in microseconds (overrides --min-separation)'
    )

    # Method-specific options
    parser.add_argument(
        '-k', '--k-sigma',
        type=float,
        default=5.0,
        help='Stdev multiplier for stdev method (default: 5.0)'
    )
    parser.add_argument(
        '--target-rate',
        type=float,
        default=100.0,
        help='Target pulses per cycle for pulse_rate method (default: 100)'
    )
    parser.add_argument(
        '--sensitivity',
        type=float,
        default=1.0,
        help='Knee detection sensitivity for histogram_knee (default: 1.0)'
    )

    # Trigger refinement options
    parser.add_argument(
        '--refine-to-onset',
        action='store_true',
        help='Adjust triggers backward to pulse onset (helps when pulse starts before threshold crossing)'
    )
    parser.add_argument(
        '--refine-to-peak',
        action='store_true',
        help='Adjust triggers forward to pulse peak (helps when trigger is early)'
    )
    parser.add_argument(
        '--validate-peak-position',
        action='store_true',
        help='Discard waveforms where peak is far from trigger (for high pulse density data)'
    )
    parser.add_argument(
        '--peak-tolerance',
        type=float,
        default=0.5,
        help='Fraction of window where peak must appear (default: 0.5 = first half)'
    )

    # Waveform extraction options
    parser.add_argument(
        '--pre', '--pre-samples',
        type=int,
        default=500,
        dest='pre_samples',
        help='Pre-trigger samples (default: 500)'
    )
    parser.add_argument(
        '--post', '--post-samples',
        type=int,
        default=1500,
        dest='post_samples',
        help='Post-trigger samples (default: 1500)'
    )

    # Data options
    parser.add_argument(
        '--ac-frequency',
        type=float,
        default=60.0,
        help='AC power frequency in Hz (default: 60)'
    )
    parser.add_argument(
        '--signal-var',
        help='Variable name for signal in .mat file (auto-detect if not specified)'
    )
    parser.add_argument(
        '-c', '--channel',
        help='Channel to process for multi-channel files (e.g., Ch1, Ch2). Same as --signal-var but more intuitive for IEEE data.'
    )
    parser.add_argument(
        '--sample-rate-var',
        help='Variable name for sample rate (auto-detect if not specified)'
    )
    parser.add_argument(
        '--list-channels',
        action='store_true',
        help='List available channels in the .mat file and exit'
    )

    # Other options
    parser.add_argument(
        '--compare-methods',
        action='store_true',
        help='Compare all trigger methods without processing'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    parser.add_argument(
        '--list-vars',
        action='store_true',
        help='List variables in .mat file and exit'
    )

    args = parser.parse_args()

    # List variables mode
    if args.list_vars:
        loader = MatLoader(args.input)
        info = loader.get_info()
        print(f"\nVariables in {args.input}:")
        print(f"Format: {info['format']}")
        print("-" * 50)
        for name, details in info['variables'].items():
            if 'shape' in details:
                print(f"  {name}: {details['dtype']} {details['shape']}")
            else:
                print(f"  {name}: {details['type']}")
        return

    # List channels mode
    if args.list_channels:
        loader = MatLoader(args.input)
        channels = loader.list_channels()
        print(f"\nAvailable channels in {args.input}:")
        print("-" * 50)
        if channels:
            for ch in channels:
                print(f"  {ch}")
        else:
            print("  No channels found. Use --list-vars to see all variables.")
        return

    # Handle --channel as alias for --signal-var
    signal_var = args.signal_var or args.channel

    # Compare methods mode
    if args.compare_methods:
        compare_methods_report(
            args.input,
            signal_var=signal_var,
            sample_rate_var=args.sample_rate_var,
            ac_frequency=args.ac_frequency,
        )
        return

    # Build trigger kwargs based on method
    trigger_kwargs = {}
    if args.method == 'stdev':
        trigger_kwargs['k_sigma'] = args.k_sigma
    elif args.method == 'pulse_rate':
        trigger_kwargs['target_rate_per_cycle'] = args.target_rate
    elif args.method == 'histogram_knee':
        trigger_kwargs['sensitivity'] = args.sensitivity

    # Process the file
    try:
        result = process_raw_stream(
            filepath=args.input,
            output_dir=args.output_dir,
            output_prefix=args.prefix,
            trigger_method=args.method,
            pre_samples=args.pre_samples,
            post_samples=args.post_samples,
            ac_frequency=args.ac_frequency,
            polarity=args.polarity,
            min_separation=args.min_separation,
            dead_time_us=args.dead_time_us,
            refine_to_onset=args.refine_to_onset,
            refine_to_peak=args.refine_to_peak,
            validate_peak_position=args.validate_peak_position,
            peak_tolerance=args.peak_tolerance,
            signal_var=signal_var,
            sample_rate_var=args.sample_rate_var,
            verbose=not args.quiet,
            **trigger_kwargs
        )

        if result['status'] == 'no_triggers':
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
