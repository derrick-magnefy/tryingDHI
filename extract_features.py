#!/usr/bin/env python3
"""
Extract Features from Waveform Data

Standalone script for extracting features from partial discharge waveforms.
This script is called by the GUI when features need to be extracted or re-extracted.

Usage:
    python extract_features.py --input-dir "IEEE Data Processed" --file Euller_Aq03_1
    python extract_features.py --input-dir "Rugged Data Files"  # Process all files
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline_integrated import extract_features


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from waveform data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Directory containing waveform files'
    )

    parser.add_argument(
        '--file', '-f',
        help='Specific file prefix to process (optional, processes all if not specified)'
    )

    parser.add_argument(
        '--polarity-method',
        default='first_peak',
        choices=['peak', 'first_peak', 'integrated_charge', 'energy_weighted',
                 'dominant_half_cycle', 'initial_slope'],
        help='Method for calculating polarity (default: first_peak)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Run extraction
    try:
        results = extract_features(
            data_dir=args.input_dir,
            file_prefix=args.file,
            polarity_method=args.polarity_method,
            verbose=not args.quiet
        )

        if not results:
            print("Warning: No features extracted. Check that waveform files exist.", file=sys.stderr)
            sys.exit(1)

        print(f"\nFeature extraction complete. Processed {len(results)} dataset(s).")

    except Exception as e:
        print(f"Error during feature extraction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
