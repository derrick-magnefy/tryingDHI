#!/usr/bin/env python3
"""
PD Visualization GUI (Web-based using Plotly Dash)

Interactive GUI for visualizing Partial Discharge analysis results.
Features:
- Dataset selector dropdown
- Main PRPD plot with clickable dots (phase vs amplitude)
- Cluster-colored PRPD view
- PD Type-colored PRPD view
- Waveform detail viewer (shows actual waveform when dot is clicked)

Usage:
    python pd_visualization_gui.py [--data-dir DIR] [--port PORT]

Then open browser to http://localhost:8050
"""

import numpy as np
import os
import glob
import argparse
import subprocess
import sys
import tempfile
import json

try:
    from dash import Dash, html, dcc, callback, Output, Input, State, no_update
    from dash import ctx  # For callback context
    from dash.exceptions import PreventUpdate
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Dash not available. Install with: pip install dash plotly")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    print("sklearn not available for PCA. Install with: pip install scikit-learn")

from polarity_methods import (
    calculate_polarity, compare_methods, POLARITY_METHODS,
    DEFAULT_POLARITY_METHOD, get_method_description
)
from classify_pd_type import (
    PDTypeClassifier, load_cluster_features,
    NOISE_THRESHOLDS, PHASE_CORRELATION_THRESHOLDS,
    SYMMETRY_THRESHOLDS, AMPLITUDE_THRESHOLDS, QUADRANT_THRESHOLDS
)

# Try to import the Tektronix WFM parser
try:
    from wfm_parser import TektronixWFMParser, load_tu_delft_timing, convert_timing_to_phase
    WFM_PARSER_AVAILABLE = True
except ImportError:
    WFM_PARSER_AVAILABLE = False
    print("WFM parser not available. TU Delft format files will not be supported.")

# Data directories - primary and external
DATA_DIR = "Rugged Data Files"
EXTERNAL_DATA_DIRS = [
    "../datasets",  # External datasets not in git
    "TU Delft WFMs",  # TU Delft format data
]

# Clustering methods available
CLUSTERING_METHODS = ['dbscan', 'kmeans']
DEFAULT_CLUSTERING_METHOD = 'dbscan'

# Progress file for batch operations
PROGRESS_FILE = os.path.join(tempfile.gettempdir(), 'pd_gui_progress.json')

# Pulse features used for clustering (from extract_features.py)
PULSE_FEATURES = [
    'phase_angle',
    'peak_amplitude_positive',
    'peak_amplitude_negative',
    'absolute_amplitude',
    'polarity',
    'rise_time',
    'fall_time',
    'pulse_width',
    'slew_rate',
    'energy',
    'charge',
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
    'signal_to_noise_ratio',
    'pulse_count',
    'is_multi_pulse',
    # Normalized features (scale-independent)
    'norm_absolute_amplitude',
    'norm_peak_amplitude_positive',
    'norm_peak_amplitude_negative',
    'norm_peak_to_peak_amplitude',
    'norm_rms_amplitude',
    'norm_slew_rate',
    'norm_energy',
    'norm_charge',
    'norm_rise_time',
    'norm_fall_time',
    'norm_equivalent_time',
    'norm_equivalent_bandwidth',
    'norm_cumulative_energy_rise_time',
    'norm_pulse_width',
    'norm_dominant_frequency',
    'norm_center_frequency',
    'norm_bandwidth_3db',
    'norm_zero_crossing_rate',
    'norm_oscillation_rate',
]

# Default pulse features for clustering (all features selected by default)
DEFAULT_CLUSTERING_FEATURES = PULSE_FEATURES.copy()

# Cluster-level aggregated features (from aggregate_cluster_features.py)
CLUSTER_FEATURES = [
    'pulses_per_positive_halfcycle',
    'pulses_per_negative_halfcycle',
    'pulses_per_cycle',
    'cross_correlation',
    'discharge_asymmetry',
    'skewness_Hn_positive',
    'skewness_Hn_negative',
    'kurtosis_Hn_positive',
    'kurtosis_Hn_negative',
    'skewness_Hqn_positive',
    'skewness_Hqn_negative',
    'kurtosis_Hqn_positive',
    'kurtosis_Hqn_negative',
    'mean_amplitude_positive',
    'mean_amplitude_negative',
    'max_amplitude_positive',
    'max_amplitude_negative',
    'number_of_peaks_Hn_positive',
    'number_of_peaks_Hn_negative',
    'phase_of_max_activity',
    'phase_spread',
    'inception_phase',
    'extinction_phase',
    'quadrant_1_percentage',
    'quadrant_2_percentage',
    'quadrant_3_percentage',
    'quadrant_4_percentage',
    'weibull_alpha',
    'weibull_beta',
    'variance_amplitude_positive',
    'variance_amplitude_negative',
    'coefficient_of_variation',
    'repetition_rate',
]

# Default cluster features for classification (all features selected by default)
DEFAULT_CLASSIFICATION_FEATURES = CLUSTER_FEATURES.copy()

# ADC configuration for noise threshold calculation
# 12-bit ADC with -2V to +2V range
ADC_BITS = 12
ADC_RANGE_V = 4.0  # -2V to +2V = 4V total range
ADC_STEP_V = ADC_RANGE_V / (2 ** ADC_BITS)  # ~0.977 mV per step


def calculate_noise_threshold(features, feature_names):
    """Calculate noise threshold from dataset features.

    The noise threshold is the minimum absolute peak amplitude minus one ADC step.
    This represents the smallest signal that can be reliably distinguished from noise.

    Args:
        features: Feature matrix (n_samples x n_features)
        feature_names: List of feature names

    Returns:
        dict with noise threshold info, or None if features not available
    """
    if features is None or feature_names is None:
        return None

    # Get absolute amplitude or calculate from peak amplitudes
    if 'absolute_amplitude' in feature_names:
        amp_idx = feature_names.index('absolute_amplitude')
        abs_amplitudes = features[:, amp_idx]
    else:
        # Fall back to calculating from positive/negative peaks
        pos_idx = feature_names.index('peak_amplitude_positive') if 'peak_amplitude_positive' in feature_names else None
        neg_idx = feature_names.index('peak_amplitude_negative') if 'peak_amplitude_negative' in feature_names else None

        if pos_idx is None or neg_idx is None:
            return None

        pos_amps = features[:, pos_idx]
        neg_amps = features[:, neg_idx]
        abs_amplitudes = np.maximum(np.abs(pos_amps), np.abs(neg_amps))

    # Calculate statistics
    min_amplitude = float(np.min(abs_amplitudes))
    noise_threshold = max(0, min_amplitude - ADC_STEP_V)

    return {
        'min_absolute_amplitude': min_amplitude,
        'adc_step_v': ADC_STEP_V,
        'noise_threshold': noise_threshold,
        'adc_bits': ADC_BITS,
        'adc_range_v': ADC_RANGE_V,
        'n_samples': len(abs_amplitudes),
        'mean_amplitude': float(np.mean(abs_amplitudes)),
        'std_amplitude': float(np.std(abs_amplitudes)),
    }


def save_dataset_metadata(data_path, prefix, metadata):
    """Save dataset metadata to JSON file.

    Args:
        data_path: Directory containing dataset files
        prefix: Dataset prefix (clean name)
        metadata: Dict of metadata to save
    """
    metadata_file = os.path.join(data_path, f"{prefix}-metadata.json")

    # Load existing metadata if present
    existing = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                existing = json.load(f)
        except Exception:
            pass

    # Merge with new metadata
    existing.update(metadata)

    # Save
    with open(metadata_file, 'w') as f:
        json.dump(existing, f, indent=2)


def load_dataset_metadata(data_path, prefix):
    """Load dataset metadata from JSON file.

    Args:
        data_path: Directory containing dataset files
        prefix: Dataset prefix (clean name)

    Returns:
        Dict of metadata, or empty dict if not found
    """
    metadata_file = os.path.join(data_path, f"{prefix}-metadata.json")

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass

    return {}


# Color schemes
PD_TYPE_COLORS = {
    'NOISE': '#808080',      # Gray
    'CORONA': '#FF6B6B',     # Red
    'INTERNAL': '#4ECDC4',   # Teal
    'SURFACE': '#FFE66D',    # Yellow
    'UNKNOWN': '#95A5A6',    # Light gray
}

CLUSTER_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

# Feature groups for organized display
FEATURE_GROUPS = {
    'Time Domain': ['phase_angle', 'rise_time', 'fall_time', 'pulse_width', 'slew_rate'],
    'Amplitude': ['peak_amplitude_positive', 'peak_amplitude_negative', 'polarity',
                  'peak_to_peak_amplitude', 'rms_amplitude', 'crest_factor'],
    'Energy': ['energy', 'equivalent_time', 'equivalent_bandwidth', 'cumulative_energy_peak',
               'cumulative_energy_rise_time', 'cumulative_energy_shape_factor',
               'cumulative_energy_area_ratio', 'energy_charge_ratio'],
    'Frequency': ['dominant_frequency', 'center_frequency', 'bandwidth_3db',
                  'spectral_power_low', 'spectral_power_high', 'spectral_flatness', 'spectral_entropy'],
    'Shape': ['rise_fall_ratio', 'zero_crossing_count', 'oscillation_count']
}

# Default visible features
DEFAULT_VISIBLE_FEATURES = [
    'phase_angle', 'peak_amplitude_positive', 'peak_amplitude_negative',
    'rise_time', 'energy', 'dominant_frequency'
]


def format_feature_value(name, value):
    """Format a feature value for display."""
    if 'frequency' in name.lower() or 'bandwidth' in name.lower():
        if abs(value) >= 1e6:
            return f"{value/1e6:.2f} MHz"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.2f} kHz"
        else:
            return f"{value:.2f} Hz"
    elif 'time' in name.lower() or name in ['rise_time', 'fall_time', 'pulse_width']:
        if abs(value) < 1e-6:
            return f"{value*1e9:.2f} ns"
        elif abs(value) < 1e-3:
            return f"{value*1e6:.2f} µs"
        else:
            return f"{value*1e3:.2f} ms"
    elif 'energy' in name.lower():
        return f"{value:.4e}"
    elif 'phase' in name.lower() or 'angle' in name.lower():
        return f"{value:.1f}°"
    elif 'amplitude' in name.lower():
        return f"{value:.6f} V"
    elif 'polarity' in name.lower():
        return "Positive" if value > 0 else "Negative"
    elif isinstance(value, float):
        if abs(value) < 0.01 or abs(value) >= 1000:
            return f"{value:.4e}"
        else:
            return f"{value:.4f}"
    else:
        return str(value)


class PDDataLoader:
    """Handles loading PD analysis data from multiple directories and formats."""

    # Format types
    FORMAT_RUGGED = 'rugged'  # Original format with -WFMs.txt, -SG.txt, etc.
    FORMAT_TUDELFT = 'tudelft'  # TU Delft format with .wfm binary files
    FORMAT_UNKNOWN = 'unknown'

    def __init__(self, data_dir=DATA_DIR, external_dirs=None):
        self.data_dir = data_dir
        self.external_dirs = external_dirs or EXTERNAL_DATA_DIRS
        self.datasets = []
        self.dataset_info = {}  # Maps dataset name to {path, format, ...}
        self.find_datasets()

    def find_datasets(self):
        """Find all available datasets from all directories."""
        self.datasets = []
        self.dataset_info = {}

        # Search directories
        all_dirs = [self.data_dir] + self.external_dirs

        for data_dir in all_dirs:
            if not os.path.exists(data_dir):
                continue

            # Find processed datasets (have -features.csv)
            feature_files = glob.glob(os.path.join(data_dir, "*-features.csv"))
            for f in sorted(feature_files):
                basename = os.path.basename(f)
                prefix = basename.replace("-features.csv", "")
                if prefix not in self.dataset_info:
                    self.datasets.append(prefix)
                    self.dataset_info[prefix] = {
                        'path': data_dir,
                        'format': self.FORMAT_RUGGED,
                        'has_features': True
                    }

            # Find TU Delft format datasets (subdirectories with .wfm files)
            if WFM_PARSER_AVAILABLE:
                for subdir in os.listdir(data_dir):
                    subdir_path = os.path.join(data_dir, subdir)
                    if os.path.isdir(subdir_path):
                        wfm_files = glob.glob(os.path.join(subdir_path, "*.wfm"))
                        if wfm_files:
                            # This is a TU Delft format dataset
                            dataset_name = f"[TUD] {subdir}"
                            if dataset_name not in self.dataset_info:
                                self.datasets.append(dataset_name)
                                self.dataset_info[dataset_name] = {
                                    'path': subdir_path,
                                    'format': self.FORMAT_TUDELFT,
                                    'wfm_files': wfm_files,
                                    'has_features': os.path.exists(
                                        os.path.join(subdir_path, f"{subdir}-features.csv")
                                    )
                                }

            # Find unprocessed Rugged format datasets (have -WFMs.txt but no -features.csv)
            wfm_txt_files = glob.glob(os.path.join(data_dir, "*-WFMs.txt"))
            for f in sorted(wfm_txt_files):
                basename = os.path.basename(f)
                prefix = basename.replace("-WFMs.txt", "")
                features_file = os.path.join(data_dir, f"{prefix}-features.csv")
                if prefix not in self.dataset_info and not os.path.exists(features_file):
                    dataset_name = f"[RAW] {prefix}"
                    self.datasets.append(dataset_name)
                    self.dataset_info[dataset_name] = {
                        'path': data_dir,
                        'format': self.FORMAT_RUGGED,
                        'has_features': False,
                        'raw_file': f
                    }

            # Find raw single .wfm files (ASCII format)
            raw_wfm_files = glob.glob(os.path.join(data_dir, "*.wfm"))
            for f in raw_wfm_files:
                basename = os.path.basename(f)
                prefix = basename.replace(".wfm", "")
                dataset_name = f"[RAW] {prefix}"
                if dataset_name not in self.dataset_info:
                    self.datasets.append(dataset_name)
                    self.dataset_info[dataset_name] = {
                        'path': data_dir,
                        'format': self.FORMAT_RUGGED,
                        'has_features': False,
                        'raw_file': f
                    }

        return self.datasets

    def get_dataset_path(self, prefix):
        """Get the data directory for a dataset."""
        info = self.dataset_info.get(prefix, {})
        return info.get('path', self.data_dir)

    def get_dataset_format(self, prefix):
        """Get the format type for a dataset."""
        info = self.dataset_info.get(prefix, {})
        return info.get('format', self.FORMAT_RUGGED)

    def get_clean_prefix(self, prefix):
        """Remove format tags from prefix for file loading."""
        # Remove [TUD], [RAW], etc. prefixes
        if prefix.startswith('[TUD] '):
            return prefix[6:]
        elif prefix.startswith('[RAW] '):
            return prefix[6:]
        return prefix

    def load_features(self, prefix):
        """Load features from CSV."""
        data_path = self.get_dataset_path(prefix)
        clean_prefix = self.get_clean_prefix(prefix)

        filepath = os.path.join(data_path, f"{clean_prefix}-features.csv")
        if not os.path.exists(filepath):
            return None, None

        features = []
        feature_names = None

        with open(filepath, 'r') as f:
            header = f.readline().strip()
            feature_names = header.split(',')[1:]  # Skip waveform_index

            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    values = [float(v) for v in parts[1:]]
                    features.append(values)

        return np.array(features), feature_names

    def load_cluster_labels(self, prefix, method='dbscan'):
        """Load cluster labels."""
        data_path = self.get_dataset_path(prefix)
        clean_prefix = self.get_clean_prefix(prefix)

        filepath = os.path.join(data_path, f"{clean_prefix}-clusters-{method}.csv")
        if not os.path.exists(filepath):
            return None

        labels = []
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith('#') and not line.startswith('waveform'):
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        labels.append(int(parts[1]))

        return np.array(labels)

    def load_pd_types(self, prefix, method='dbscan'):
        """Load PD type classifications."""
        data_path = self.get_dataset_path(prefix)
        clean_prefix = self.get_clean_prefix(prefix)

        filepath = os.path.join(data_path, f"{clean_prefix}-pd-types-{method}.csv")
        if not os.path.exists(filepath):
            return None

        pd_types = {}
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith('#') and not line.startswith('cluster'):
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        label = -1 if parts[0] == 'noise' else int(parts[0])
                        pd_types[label] = parts[1]

        return pd_types

    def load_waveforms(self, prefix):
        """Load raw waveforms - auto-detects format."""
        data_path = self.get_dataset_path(prefix)
        clean_prefix = self.get_clean_prefix(prefix)
        fmt = self.get_dataset_format(prefix)

        if fmt == self.FORMAT_TUDELFT:
            return self._load_waveforms_tudelft(prefix)
        else:
            return self._load_waveforms_rugged(data_path, clean_prefix)

    def _load_waveforms_rugged(self, data_path, clean_prefix):
        """Load waveforms from -WFMs.txt file (original format)."""
        filepath = os.path.join(data_path, f"{clean_prefix}-WFMs.txt")
        if not os.path.exists(filepath):
            # Try .wfm extension (single ASCII wfm file format)
            filepath = os.path.join(data_path, f"{clean_prefix}.wfm")
            if not os.path.exists(filepath):
                return None
            # Check if it's a binary file (TU Delft format) - don't try to read as text
            try:
                with open(filepath, 'rb') as f:
                    header = f.read(20)
                    if b':WFM#' in header:
                        # This is a binary Tektronix file, not ASCII
                        return None
            except:
                pass

        try:
            waveforms = []
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        values = [float(v) for v in line.split('\t') if v.strip()]
                        waveforms.append(np.array(values))
            return waveforms if waveforms else None
        except Exception as e:
            print(f"Error loading waveforms: {e}")
            return None

    def _load_waveforms_tudelft(self, prefix):
        """Load waveforms from TU Delft format .wfm files."""
        if not WFM_PARSER_AVAILABLE:
            print("TU Delft format not supported - WFM parser not available")
            return None

        info = self.dataset_info.get(prefix, {})
        wfm_files = info.get('wfm_files', [])

        if not wfm_files:
            return None

        try:
            # Use the first channel (Ch1) by default
            ch1_file = None
            for f in wfm_files:
                if 'Ch1' in f or '_1.' in f:
                    ch1_file = f
                    break
            if not ch1_file:
                ch1_file = wfm_files[0]

            parser = TektronixWFMParser(ch1_file)
            waveforms = parser.get_waveforms()
            return [np.array(w) for w in waveforms] if waveforms else None
        except Exception as e:
            print(f"Error loading TU Delft waveforms: {e}")
            return None

    def load_settings(self, prefix):
        """Load settings from -SG.txt file or extract from WFM header."""
        data_path = self.get_dataset_path(prefix)
        clean_prefix = self.get_clean_prefix(prefix)
        fmt = self.get_dataset_format(prefix)

        if fmt == self.FORMAT_TUDELFT:
            return self._load_settings_tudelft(prefix)
        else:
            return self._load_settings_rugged(data_path, clean_prefix)

    def _load_settings_rugged(self, data_path, clean_prefix):
        """Load settings from -SG.txt file."""
        filepath = os.path.join(data_path, f"{clean_prefix}-SG.txt")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                values = [float(v) for v in content.split('\t') if v.strip()]
            return values
        except Exception as e:
            print(f"Error loading settings: {e}")
            return None

    def _load_settings_tudelft(self, prefix):
        """Extract settings from TU Delft WFM file header."""
        if not WFM_PARSER_AVAILABLE:
            return None

        info = self.dataset_info.get(prefix, {})
        wfm_files = info.get('wfm_files', [])

        if not wfm_files:
            return None

        try:
            parser = TektronixWFMParser(wfm_files[0])
            sample_interval = parser.get_sample_interval()

            # Return settings array compatible with existing format
            # Index 10 is sample_interval, index 9 is AC frequency
            settings = [0.0] * 18
            settings[9] = 50.0  # Default 50 Hz AC
            settings[10] = sample_interval
            return settings
        except Exception as e:
            print(f"Error loading TU Delft settings: {e}")
            return None

    def load_all(self, prefix, method='dbscan'):
        """Load all data for a dataset."""
        features, feature_names = self.load_features(prefix)
        cluster_labels = self.load_cluster_labels(prefix, method)
        pd_types = self.load_pd_types(prefix, method)
        waveforms = self.load_waveforms(prefix)
        settings = self.load_settings(prefix)

        # Get sample_interval from settings (index 10) or use default
        sample_interval = settings[10] if settings and len(settings) > 10 else 4e-9

        return {
            'features': features,
            'feature_names': feature_names,
            'cluster_labels': cluster_labels,
            'pd_types': pd_types,
            'waveforms': waveforms,
            'sample_interval': sample_interval
        }


def create_prpd_plot(features, feature_names, cluster_labels, pd_types, color_by='cluster',
                     waveforms=None, polarity_method=None, sample_interval=4e-9):
    """Create PRPD scatter plot.

    Args:
        features: Feature matrix
        feature_names: List of feature names
        cluster_labels: Cluster labels for each waveform
        pd_types: PD type classification for each cluster
        color_by: 'cluster' or 'pdtype'
        waveforms: Optional raw waveforms for polarity recalculation
        polarity_method: If provided and waveforms available, recalculate polarity
        sample_interval: Sample interval for polarity calculation
    """
    if features is None:
        return go.Figure()

    # Get phase and amplitude
    phase_idx = feature_names.index('phase_angle') if 'phase_angle' in feature_names else 0
    amp_pos_idx = feature_names.index('peak_amplitude_positive') if 'peak_amplitude_positive' in feature_names else 1
    amp_neg_idx = feature_names.index('peak_amplitude_negative') if 'peak_amplitude_negative' in feature_names else 2
    polarity_idx = feature_names.index('polarity') if 'polarity' in feature_names else None

    phases = features[:, phase_idx]
    amp_pos = features[:, amp_pos_idx]
    amp_neg = features[:, amp_neg_idx]  # Already negative values

    # Determine polarity to use
    if polarity_method is not None and waveforms is not None and polarity_method != 'stored':
        # Recalculate polarity using selected method
        polarity = np.array([
            calculate_polarity(wfm, method=polarity_method, sample_interval=sample_interval)
            for wfm in waveforms
        ])
    elif polarity_idx is not None:
        polarity = features[:, polarity_idx]
    else:
        polarity = None

    # Use polarity to determine which amplitude to show
    if polarity is not None:
        # polarity = 1 means positive dominant, polarity = -1 means negative dominant
        amplitudes = np.where(polarity > 0, amp_pos, amp_neg)
    else:
        # Fallback: use the larger magnitude with appropriate sign
        amplitudes = np.where(amp_pos >= np.abs(amp_neg), amp_pos, amp_neg)

    fig = go.Figure()

    if color_by == 'cluster' and cluster_labels is not None:
        unique_labels = sorted(set(cluster_labels))
        for label in unique_labels:
            mask = cluster_labels == label
            if label == -1:
                color = '#808080'
                name = 'Noise'
            else:
                color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
                name = f'Cluster {label}'

            # Get original indices for this cluster
            original_indices = np.where(mask)[0].tolist()
            fig.add_trace(go.Scatter(
                x=phases[mask],
                y=amplitudes[mask],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.7),
                name=name,
                customdata=original_indices,
                hovertemplate='Phase: %{x:.1f}°<br>Amplitude: %{y:.4f} V<br>Index: %{customdata}<extra></extra>'
            ))

    elif color_by == 'pdtype' and cluster_labels is not None and pd_types is not None:
        pulse_types = [pd_types.get(l, 'UNKNOWN') for l in cluster_labels]

        for pd_type in ['CORONA', 'INTERNAL', 'SURFACE', 'NOISE', 'UNKNOWN']:
            mask = np.array([t == pd_type for t in pulse_types])
            if np.any(mask):
                color = PD_TYPE_COLORS.get(pd_type, '#000000')

                # Get original indices for this PD type
                original_indices = np.where(mask)[0].tolist()
                fig.add_trace(go.Scatter(
                    x=phases[mask],
                    y=amplitudes[mask],
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.7),
                    name=pd_type,
                    customdata=original_indices,
                    hovertemplate='Phase: %{x:.1f}°<br>Amplitude: %{y:.4f} V<br>Index: %{customdata}<extra></extra>'
                ))
    else:
        fig.add_trace(go.Scatter(
            x=phases,
            y=amplitudes,
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.7),
            name='All pulses',
            customdata=np.arange(len(phases)),
            hovertemplate='Phase: %{x:.1f}°<br>Amplitude: %{y:.4f} V<br>Index: %{customdata}<extra></extra>'
        ))

    # Add reference lines
    for phase in [90, 180, 270]:
        fig.add_vline(x=phase, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_hline(y=0, line_color="gray", opacity=0.5)

    title = "PRPD by Cluster" if color_by == 'cluster' else "PRPD by PD Type"
    fig.update_layout(
        title=title,
        xaxis_title="Phase (degrees)",
        yaxis_title="Amplitude (V)",
        xaxis=dict(range=[0, 360]),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(r=150),
        dragmode='select'  # Enable box selection by default
    )

    # Add modebar buttons for selection
    fig.update_layout(
        modebar_add=['select2d', 'lasso2d'],
        modebar_remove=['autoScale2d']
    )

    return fig


def create_waveform_plot(waveforms, idx, features, feature_names, cluster_labels, pd_types):
    """Create waveform plot for selected point."""
    fig = go.Figure()

    # Check if we can display a waveform
    can_display = (
        waveforms is not None and
        len(waveforms) > 0 and
        idx is not None and
        0 <= idx < len(waveforms)
    )

    if not can_display:
        msg = "Click on a point in the PRPD plot to view waveform"
        if waveforms is None:
            msg = "Waveforms not loaded for this dataset"
        elif len(waveforms) == 0:
            msg = "No waveforms available"
        elif idx is not None and idx >= len(waveforms):
            msg = f"Waveform index {idx} out of range (max: {len(waveforms)-1})"

        fig.add_annotation(
            text=msg,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Waveform Viewer",
            xaxis_title="Sample",
            yaxis_title="Amplitude (V)"
        )
        return fig

    waveform = waveforms[idx]
    fig.add_trace(go.Scatter(
        y=waveform,
        mode='lines',
        line=dict(color='blue', width=1),
        name='Waveform'
    ))

    # Build title with info
    info_parts = [f"Waveform #{idx}"]

    if features is not None and feature_names is not None:
        phase_idx = feature_names.index('phase_angle') if 'phase_angle' in feature_names else 0
        phase = features[idx, phase_idx]
        info_parts.append(f"Phase: {phase:.1f}°")

    if cluster_labels is not None:
        cluster = cluster_labels[idx]
        info_parts.append(f"Cluster: {cluster}")

        if pd_types is not None:
            pd_type = pd_types.get(cluster, 'UNKNOWN')
            info_parts.append(f"Type: {pd_type}")

    fig.update_layout(
        title=" | ".join(info_parts),
        xaxis_title="Sample",
        yaxis_title="Amplitude (V)"
    )

    return fig


def create_histogram(features, feature_names, cluster_labels, pd_types):
    """Create phase distribution histogram."""
    fig = go.Figure()

    if features is None:
        return fig

    phase_idx = feature_names.index('phase_angle') if 'phase_angle' in feature_names else 0
    phases = features[:, phase_idx]

    if cluster_labels is not None and pd_types is not None:
        pulse_types = [pd_types.get(l, 'UNKNOWN') for l in cluster_labels]

        for pd_type in ['CORONA', 'INTERNAL', 'SURFACE', 'NOISE', 'UNKNOWN']:
            mask = np.array([t == pd_type for t in pulse_types])
            if np.any(mask):
                color = PD_TYPE_COLORS.get(pd_type, '#000000')
                fig.add_trace(go.Histogram(
                    x=phases[mask],
                    nbinsx=36,
                    name=pd_type,
                    marker_color=color,
                    opacity=0.7
                ))
    else:
        fig.add_trace(go.Histogram(
            x=phases,
            nbinsx=36,
            name='All pulses',
            marker_color='blue',
            opacity=0.7
        ))

    # Add reference lines
    for phase in [90, 180, 270]:
        fig.add_vline(x=phase, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title="Phase Distribution",
        xaxis_title="Phase (degrees)",
        yaxis_title="Count",
        xaxis=dict(range=[0, 360]),
        barmode='overlay'
    )

    return fig


def create_stats_text(features, cluster_labels, pd_types):
    """Create statistics summary text."""
    if features is None:
        return "No data loaded"

    lines = []
    total = len(features)
    lines.append(f"**Total Pulses:** {total}")

    if cluster_labels is not None:
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        lines.append(f"**Clusters:** {n_clusters}")
        lines.append(f"**Noise Points:** {n_noise} ({n_noise/total*100:.1f}%)")

    if cluster_labels is not None and pd_types is not None:
        lines.append("")
        lines.append("**PD Type Distribution:**")
        type_counts = {}
        for label in cluster_labels:
            pd_type = pd_types.get(label, 'UNKNOWN')
            type_counts[pd_type] = type_counts.get(pd_type, 0) + 1

        for pd_type in ['CORONA', 'INTERNAL', 'SURFACE', 'NOISE', 'UNKNOWN']:
            if pd_type in type_counts:
                count = type_counts[pd_type]
                pct = count / total * 100
                lines.append(f"- {pd_type}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def create_pca_plot(features, feature_names, cluster_labels):
    """Create PCA plot (PC1 vs PC2) colored by cluster.

    Args:
        features: Feature matrix
        feature_names: List of feature names
        cluster_labels: Cluster labels for each point

    Returns:
        Plotly figure with PCA scatter plot
    """
    if features is None or not PCA_AVAILABLE:
        fig = go.Figure()
        fig.update_layout(
            title="PCA Plot (PC1 vs PC2)",
            annotations=[{
                'text': 'PCA not available' if not PCA_AVAILABLE else 'No data',
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.5, 'showarrow': False,
                'font': {'size': 16, 'color': 'gray'}
            }]
        )
        return fig

    # Handle infinite/NaN values
    features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)

    fig = go.Figure()

    if cluster_labels is not None:
        unique_labels = sorted(set(cluster_labels))
        for label in unique_labels:
            mask = cluster_labels == label
            if label == -1:
                color = '#808080'
                name = 'Noise'
            else:
                color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
                name = f'Cluster {label}'

            original_indices = np.where(mask)[0].tolist()
            fig.add_trace(go.Scatter(
                x=pca_result[mask, 0],
                y=pca_result[mask, 1],
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.7),
                name=name,
                customdata=original_indices,
                hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Index: %{customdata}<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.7),
            name='All pulses'
        ))

    # Add explained variance to axis labels
    var_explained = pca.explained_variance_ratio_ * 100

    fig.update_layout(
        title=f"PCA Plot - Feature Space Visualization",
        xaxis_title=f"PC1 ({var_explained[0]:.1f}% variance)",
        yaxis_title=f"PC2 ({var_explained[1]:.1f}% variance)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        hovermode='closest'
    )

    return fig


def read_features_from_cluster_file(cluster_file_path):
    """Read the features used for clustering from a cluster file's metadata.

    Args:
        cluster_file_path: Path to the cluster CSV file

    Returns:
        List of feature names used for clustering, or None if not found
    """
    if not os.path.exists(cluster_file_path):
        return None

    try:
        with open(cluster_file_path, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if line.startswith('# Features_used:'):
                    features_str = line.replace('# Features_used:', '').strip()
                    if features_str:
                        return [f.strip() for f in features_str.split(',')]
    except Exception:
        pass

    return None


def create_correlation_matrix(features, feature_names, selected_features=None):
    """Create correlation matrix heatmap for selected features.

    Args:
        features: Feature matrix (n_samples x n_features)
        feature_names: List of all feature names
        selected_features: List of feature names to include (None = all)

    Returns:
        Plotly figure with correlation matrix heatmap
    """
    if features is None or feature_names is None:
        fig = go.Figure()
        fig.update_layout(
            title="Feature Correlation Matrix",
            annotations=[{
                'text': 'No data available',
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.5, 'showarrow': False,
                'font': {'size': 16, 'color': 'gray'}
            }]
        )
        return fig

    # Filter to selected features
    if selected_features and len(selected_features) > 0:
        # Get indices of selected features
        indices = []
        labels = []
        for feat in selected_features:
            if feat in feature_names:
                indices.append(feature_names.index(feat))
                labels.append(feat)

        if len(indices) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Feature Correlation Matrix",
                annotations=[{
                    'text': 'Select at least 2 features',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5, 'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )
            return fig

        feature_subset = features[:, indices]
    else:
        feature_subset = features
        labels = feature_names

    # Handle NaN/Inf values
    feature_subset = np.nan_to_num(feature_subset, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute correlation matrix
    n_features = feature_subset.shape[1]
    corr_matrix = np.corrcoef(feature_subset.T)

    # Handle case where corrcoef returns NaN (constant features)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Create text annotations with R values
    text_annotations = [[f'{corr_matrix[i, j]:.2f}' for j in range(n_features)] for i in range(n_features)]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=labels,
        y=labels,
        text=text_annotations,
        texttemplate='%{text}',
        textfont={'size': 9},
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title=dict(text='R value', side='right')
        ),
        hovertemplate='%{x} vs %{y}<br>R = %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Feature Correlation Matrix ({len(labels)} features)',
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            autorange='reversed'
        ),
        width=max(600, len(labels) * 35 + 150),
        height=max(600, len(labels) * 35 + 150),
    )

    return fig


def create_pca_loadings(features, feature_names, selected_features=None, n_components=10):
    """Create PCA loadings heatmap showing feature contributions to each PC.

    Args:
        features: Feature matrix (n_samples x n_features)
        feature_names: List of all feature names
        selected_features: List of feature names to include (None = all)
        n_components: Number of principal components to show

    Returns:
        Plotly figure with PCA loadings heatmap
    """
    if features is None or feature_names is None or not PCA_AVAILABLE:
        fig = go.Figure()
        fig.update_layout(
            title="PCA Loadings",
            annotations=[{
                'text': 'PCA not available' if not PCA_AVAILABLE else 'No data available',
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.5, 'showarrow': False,
                'font': {'size': 16, 'color': 'gray'}
            }]
        )
        return fig

    # Filter to selected features
    if selected_features and len(selected_features) > 0:
        indices = []
        labels = []
        for feat in selected_features:
            if feat in feature_names:
                indices.append(feature_names.index(feat))
                labels.append(feat)

        if len(indices) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="PCA Loadings",
                annotations=[{
                    'text': 'Select at least 2 features',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5, 'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )
            return fig

        feature_subset = features[:, indices]
    else:
        feature_subset = features
        labels = list(feature_names)

    # Handle NaN/Inf values
    feature_subset = np.nan_to_num(feature_subset, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_subset)

    # Perform PCA
    n_components_actual = min(n_components, len(labels), features_scaled.shape[0])
    pca = PCA(n_components=n_components_actual)
    pca.fit(features_scaled)

    # Get loadings (components_ is n_components x n_features)
    loadings = pca.components_.T  # Transpose to n_features x n_components

    # Get explained variance for column labels
    var_explained = pca.explained_variance_ratio_ * 100
    pc_labels = [f'PC{i+1}\n({var_explained[i]:.1f}%)' for i in range(n_components_actual)]

    # Create text annotations
    text_annotations = [[f'{loadings[i, j]:.2f}' for j in range(n_components_actual)] for i in range(len(labels))]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=loadings,
        x=pc_labels,
        y=labels,
        text=text_annotations,
        texttemplate='%{text}',
        textfont={'size': 10},
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(
            title=dict(text='Loading', side='right')
        ),
        hovertemplate='%{y}<br>%{x}<br>Loading = %{z:.3f}<extra></extra>'
    ))

    # Calculate total variance explained
    total_var = sum(var_explained)

    fig.update_layout(
        title=f'PCA Loadings ({n_components_actual} components, {total_var:.1f}% total variance explained)',
        xaxis=dict(
            title='Principal Components',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='Features',
            tickfont=dict(size=10),
            autorange='reversed'
        ),
        width=max(500, n_components_actual * 80 + 200),
        height=max(600, len(labels) * 25 + 150),
    )

    return fig


def compute_recommended_features(features, feature_names, n_features=5, correlation_threshold=0.85):
    """Compute recommended features based on PCA importance and correlation analysis.

    Algorithm:
    1. Compute PCA to get feature importance (sum of |loadings| * variance_explained)
    2. Rank features by importance
    3. Select top features while avoiding highly correlated ones

    Args:
        features: Feature matrix (n_samples x n_features)
        feature_names: List of all feature names
        n_features: Number of features to recommend
        correlation_threshold: Threshold above which features are considered redundant (0-1)

    Returns:
        dict with:
            - 'recommended': List of recommended feature names
            - 'scores': Dict of feature_name -> importance score
            - 'removed_correlations': Dict of removed feature -> correlated with
            - 'variance_explained': Total variance explained by top PCs
            - 'details': List of (feature, score, reason) tuples for all features
    """
    if not PCA_AVAILABLE or features is None or feature_names is None:
        return {
            'recommended': [],
            'scores': {},
            'removed_correlations': {},
            'variance_explained': 0,
            'details': [],
            'error': 'PCA not available or no data'
        }

    # Handle NaN/Inf values
    features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove constant features
    feature_stds = np.std(features_clean, axis=0)
    valid_indices = np.where(feature_stds > 1e-10)[0]

    if len(valid_indices) < 2:
        return {
            'recommended': list(feature_names)[:n_features],
            'scores': {f: 1.0 for f in feature_names[:n_features]},
            'removed_correlations': {},
            'variance_explained': 100,
            'details': [(f, 1.0, 'Only valid feature') for f in feature_names[:n_features]],
            'error': None
        }

    valid_features = features_clean[:, valid_indices]
    valid_names = [feature_names[i] for i in valid_indices]

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(valid_features)

    # Compute PCA
    n_components = min(len(valid_names), features_scaled.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(features_scaled)

    # Get loadings and variance explained
    loadings = np.abs(pca.components_.T)  # n_features x n_components
    var_explained = pca.explained_variance_ratio_

    # Compute feature importance: weighted sum of absolute loadings
    # Weight by variance explained for each component
    importance_scores = np.sum(loadings * var_explained, axis=1)

    # Normalize to 0-100 scale
    if importance_scores.max() > 0:
        importance_scores = 100 * importance_scores / importance_scores.max()

    # Create score dictionary
    scores = {name: float(importance_scores[i]) for i, name in enumerate(valid_names)}

    # Compute correlation matrix
    corr_matrix = np.corrcoef(features_scaled.T)

    # Rank features by importance
    ranked_indices = np.argsort(importance_scores)[::-1]

    # Select features avoiding high correlation
    selected = []
    selected_indices = []
    removed_correlations = {}
    details = []

    for idx in ranked_indices:
        feat_name = valid_names[idx]
        feat_score = importance_scores[idx]

        if len(selected) >= n_features:
            details.append((feat_name, feat_score, 'Not needed (have enough)'))
            continue

        # Check correlation with already selected features
        is_redundant = False
        correlated_with = None
        for sel_idx in selected_indices:
            corr = abs(corr_matrix[idx, sel_idx])
            if corr > correlation_threshold:
                is_redundant = True
                correlated_with = valid_names[sel_idx]
                break

        if is_redundant:
            removed_correlations[feat_name] = correlated_with
            details.append((feat_name, feat_score, f'Redundant (r={corr:.2f} with {correlated_with})'))
        else:
            selected.append(feat_name)
            selected_indices.append(idx)
            details.append((feat_name, feat_score, 'Selected'))

    # If we couldn't get enough features due to correlation, add some back
    if len(selected) < n_features:
        for idx in ranked_indices:
            if len(selected) >= n_features:
                break
            feat_name = valid_names[idx]
            if feat_name not in selected:
                selected.append(feat_name)
                # Update details
                for i, (name, score, reason) in enumerate(details):
                    if name == feat_name and 'Redundant' in reason:
                        details[i] = (name, score, 'Added (needed more features)')
                        if feat_name in removed_correlations:
                            del removed_correlations[feat_name]
                        break

    # Calculate total variance that could be explained
    # Using as many PCs as we have selected features
    total_var = sum(var_explained[:min(len(selected), len(var_explained))]) * 100

    return {
        'recommended': selected,
        'scores': scores,
        'removed_correlations': removed_correlations,
        'variance_explained': total_var,
        'details': details,
        'error': None
    }


def create_app(data_dir=DATA_DIR):
    """Create the Dash application."""
    app = Dash(__name__)
    loader = PDDataLoader(data_dir)

    app.layout = html.Div([
        html.H1("PD Analysis Visualization", style={'textAlign': 'center'}),

        # Controls
        html.Div([
            html.Div([
                html.Label("Select Dataset:"),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[{'label': d, 'value': d} for d in loader.datasets],
                    value=loader.datasets[0] if loader.datasets else None,
                    style={'width': '100%'},
                    persistence=True,
                    persistence_type='local'
                ),
            ], style={'width': '60%', 'display': 'inline-block', 'marginRight': '2%'}),

            # Noise Threshold Display
            html.Div([
                html.Div([
                    html.Label("Noise Threshold:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Span(id='noise-threshold-display', style={'marginRight': '10px'}),
                    html.Button("Calculate", id='calc-noise-threshold-btn', n_clicks=0,
                               style={'padding': '2px 8px', 'fontSize': '12px', 'cursor': 'pointer'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div(id='noise-threshold-details', style={'fontSize': '11px', 'color': '#666', 'marginTop': '3px'}),
            ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '90%', 'margin': '10px auto'}),

        # Advanced Options (collapsible)
        html.Div([
            html.Details([
                html.Summary("Advanced Analysis Options", style={
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'padding': '10px',
                    'backgroundColor': '#e9ecef',
                    'borderRadius': '4px',
                    'marginBottom': '10px'
                }),
                html.Div([
                    # Row 1: Clustering Method
                    html.Div([
                        html.Details([
                            html.Summary("Clustering Method", style={
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'padding': '8px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '4px'
                            }),
                            html.Div([
                                html.Div([
                                    html.Label("Method:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                                    dcc.RadioItems(
                                        id='clustering-method-radio',
                                        options=[
                                            {'label': 'DBSCAN (density-based)', 'value': 'dbscan'},
                                            {'label': 'K-Means (centroid-based)', 'value': 'kmeans'}
                                        ],
                                        value=DEFAULT_CLUSTERING_METHOD,
                                        inline=True,
                                        style={'display': 'inline-block'},
                                        inputStyle={'marginRight': '5px'},
                                        labelStyle={'marginRight': '20px'},
                                        persistence=True,
                                        persistence_type='local'
                                    ),
                                ], style={'marginBottom': '10px'}),
                                html.Div([
                                    html.Div([
                                        html.Label("DBSCAN min_samples:", style={'marginRight': '10px'}),
                                        dcc.Input(
                                            id='dbscan-min-samples',
                                            type='number',
                                            value=5,
                                            min=2,
                                            max=50,
                                            style={'width': '80px'}
                                        ),
                                    ], style={'display': 'inline-block', 'marginRight': '30px'}),
                                    html.Div([
                                        html.Label("K-Means clusters:", style={'marginRight': '10px'}),
                                        dcc.Input(
                                            id='kmeans-n-clusters',
                                            type='number',
                                            value=5,
                                            min=2,
                                            max=20,
                                            style={'width': '80px'}
                                        ),
                                    ], style={'display': 'inline-block'}),
                                ]),
                            ], style={'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'marginTop': '5px'})
                        ], style={'marginBottom': '15px'}),

                        # Row 2: Pulse Features for Clustering
                        html.Details([
                            html.Summary("Pulse Features (for clustering)", style={
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'padding': '8px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '4px'
                            }),
                            html.Div([
                                html.Div([
                                    html.Button("Select All", id='pulse-features-select-all', n_clicks=0,
                                               style={'marginRight': '10px', 'padding': '5px 10px'}),
                                    html.Button("Select None", id='pulse-features-select-none', n_clicks=0,
                                               style={'marginRight': '10px', 'padding': '5px 10px'}),
                                    html.Button("Reset Defaults", id='pulse-features-reset', n_clicks=0,
                                               style={'marginRight': '20px', 'padding': '5px 10px'}),
                                    html.Button("Recluster", id='recluster-main-btn',
                                               style={'backgroundColor': '#e91e63', 'color': 'white',
                                                      'padding': '5px 15px', 'border': 'none', 'borderRadius': '4px',
                                                      'cursor': 'pointer', 'fontWeight': 'bold', 'marginRight': '10px'}),
                                    html.Button("Recluster All Datasets", id='recluster-all-btn',
                                               style={'backgroundColor': '#c2185b', 'color': 'white',
                                                      'padding': '5px 15px', 'border': 'none', 'borderRadius': '4px',
                                                      'cursor': 'pointer', 'fontWeight': 'bold'}),
                                    html.Span(id='recluster-all-progress', style={'marginLeft': '10px', 'fontStyle': 'italic', 'color': '#666'}),
                                ], style={'marginBottom': '10px'}),
                                html.Div(id='recluster-main-result', style={'marginBottom': '10px'}),
                                dcc.Loading(
                                    id='recluster-all-loading',
                                    type='circle',
                                    children=html.Div(id='recluster-all-result', style={'marginBottom': '10px'})
                                ),
                                html.Div(id='features-source-info', style={'fontSize': '11px', 'color': '#666', 'fontStyle': 'italic', 'marginBottom': '5px'}),
                                dcc.Checklist(
                                    id='pulse-features-checklist',
                                    options=[{'label': f, 'value': f} for f in PULSE_FEATURES],
                                    value=DEFAULT_CLUSTERING_FEATURES,
                                    inline=True,
                                    style={'fontSize': '12px', 'maxHeight': '150px', 'overflowY': 'auto'},
                                    inputStyle={'marginRight': '5px'},
                                    labelStyle={'marginRight': '15px', 'marginBottom': '5px', 'display': 'inline-block'}
                                ),
                            ], style={'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'marginTop': '5px'})
                        ], style={'marginBottom': '15px'}),

                        # Cluster Decision Explanation Section
                        html.Details([
                            html.Summary("Cluster Decision Explanation", style={
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'padding': '8px',
                                'backgroundColor': '#e3f2fd',
                                'borderRadius': '4px'
                            }),
                            html.Div([
                                html.Div([
                                    dcc.Checklist(
                                        id='show-cluster-explanation-checkbox',
                                        options=[{'label': ' Enable cluster decision explanation', 'value': 'show'}],
                                        value=[],
                                        style={'display': 'inline-block', 'marginRight': '20px'}
                                    ),
                                    html.Button("Generate Explanation", id='generate-cluster-explanation-btn',
                                               style={'backgroundColor': '#1976d2', 'color': 'white',
                                                      'padding': '5px 15px', 'border': 'none', 'borderRadius': '4px',
                                                      'cursor': 'pointer', 'fontWeight': 'bold'}),
                                ], style={'marginBottom': '10px'}),
                                html.P("Uses a decision tree to explain how clusters were assigned based on the selected features. "
                                       "Shows interpretable rules like 'if phase_angle > 45° and rise_time < 10ns → Cluster 0'.",
                                       style={'color': '#666', 'fontSize': '12px', 'marginBottom': '10px', 'fontStyle': 'italic'}),
                                dcc.Loading(
                                    id='cluster-explanation-loading',
                                    type='circle',
                                    children=html.Div(id='cluster-explanation-display', style={
                                        'fontSize': '12px',
                                        'fontFamily': 'monospace',
                                        'backgroundColor': '#f8f9fa',
                                        'padding': '10px',
                                        'borderRadius': '4px',
                                        'maxHeight': '400px',
                                        'overflowY': 'auto',
                                        'whiteSpace': 'pre-wrap'
                                    })
                                ),
                            ], style={'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'marginTop': '5px'})
                        ], style={'marginBottom': '15px'}),

                        # Row 3: Cluster Features for Classification
                        html.Details([
                            html.Summary("Cluster Features (for classification)", style={
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'padding': '8px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '4px'
                            }),
                            html.Div([
                                html.Div([
                                    html.Button("Select All", id='cluster-features-select-all', n_clicks=0,
                                               style={'marginRight': '10px', 'padding': '5px 10px'}),
                                    html.Button("Select None", id='cluster-features-select-none', n_clicks=0,
                                               style={'marginRight': '10px', 'padding': '5px 10px'}),
                                    html.Button("Reset Defaults", id='cluster-features-reset', n_clicks=0,
                                               style={'marginRight': '20px', 'padding': '5px 10px'}),
                                    html.Button("Reclassify with Selected", id='reclassify-btn',
                                               style={'backgroundColor': '#9c27b0', 'color': 'white',
                                                      'padding': '5px 15px', 'border': 'none', 'borderRadius': '4px',
                                                      'cursor': 'pointer', 'fontWeight': 'bold', 'marginRight': '10px'}),
                                    html.Button("Reclassify All Datasets", id='reclassify-all-btn',
                                               style={'backgroundColor': '#7b1fa2', 'color': 'white',
                                                      'padding': '5px 15px', 'border': 'none', 'borderRadius': '4px',
                                                      'cursor': 'pointer', 'fontWeight': 'bold'}),
                                    html.Span(id='reclassify-all-progress', style={'marginLeft': '10px', 'fontStyle': 'italic', 'color': '#666'}),
                                ], style={'marginBottom': '10px'}),
                                html.Div(id='reclassify-result', style={'marginBottom': '10px'}),
                                dcc.Loading(
                                    id='reclassify-all-loading',
                                    type='circle',
                                    children=html.Div(id='reclassify-all-result', style={'marginBottom': '10px'})
                                ),
                                dcc.Checklist(
                                    id='cluster-features-checklist',
                                    options=[{'label': f, 'value': f} for f in CLUSTER_FEATURES],
                                    value=DEFAULT_CLASSIFICATION_FEATURES,
                                    inline=True,
                                    style={'fontSize': '12px', 'maxHeight': '150px', 'overflowY': 'auto'},
                                    inputStyle={'marginRight': '5px'},
                                    labelStyle={'marginRight': '15px', 'marginBottom': '5px', 'display': 'inline-block'}
                                ),
                            ], style={'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'marginTop': '5px'})
                        ]),

                        # Row 4: Polarity Method
                        html.Details([
                            html.Summary("Polarity Method", style={
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'padding': '8px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '4px'
                            }),
                            html.Div([
                                html.Div([
                                    html.Label("Display Method:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                                    dcc.Dropdown(
                                        id='polarity-method-dropdown',
                                        options=[{'label': 'Stored (from features)', 'value': 'stored'}] +
                                                [{'label': get_method_description(m), 'value': m} for m in POLARITY_METHODS],
                                        value='stored',
                                        style={'width': '300px', 'display': 'inline-block'},
                                        persistence=True,
                                        persistence_type='local'
                                    ),
                                    html.Button("Re-analyze with Polarity", id='reanalyze-button',
                                               n_clicks=0,
                                               style={'backgroundColor': '#007bff', 'color': 'white',
                                                      'padding': '8px 15px', 'border': 'none', 'borderRadius': '4px',
                                                      'cursor': 'pointer', 'fontWeight': 'bold', 'marginLeft': '15px'}),
                                ], style={'marginBottom': '10px'}),
                                html.Div(id='reanalyze-polarity-result', style={'marginTop': '10px'}),
                                html.P("Note: 'Stored' uses the polarity saved in the features file. Other options recalculate polarity from raw waveforms for display only. "
                                       "Click 'Re-analyze' to re-extract features with a new polarity method.",
                                       style={'fontSize': '11px', 'color': '#666', 'fontStyle': 'italic', 'marginTop': '10px'})
                            ], style={'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'marginTop': '5px'})
                        ]),
                    ]),
                ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'marginTop': '5px'})
            ])
        ], style={'width': '90%', 'margin': '10px auto'}),

        # Re-analysis status message
        html.Div(id='reanalysis-status', style={
            'width': '90%', 'margin': '5px auto', 'padding': '8px',
            'textAlign': 'center', 'display': 'none'
        }),

        # Statistics
        html.Div([
            dcc.Markdown(id='stats-text', style={'padding': '10px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})
        ], style={'width': '90%', 'margin': '10px auto'}),

        # Cluster PRPD - full row
        html.Div([
            dcc.Graph(id='cluster-prpd', style={'height': '600px'})
        ], style={'width': '95%', 'margin': 'auto'}),

        # Manual Cluster Definition Section - Full dedicated mode
        html.Div([
            html.Div([
                html.Button("Enter Manual Cluster Mode", id='enter-manual-mode-btn', n_clicks=0,
                           style={'backgroundColor': '#17a2b8', 'color': 'white',
                                  'padding': '10px 20px', 'border': 'none', 'borderRadius': '4px',
                                  'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '14px'}),
                html.Span(" Define your own clusters to discover which features best separate them",
                         style={'marginLeft': '15px', 'color': '#666', 'fontStyle': 'italic'})
            ], style={'marginBottom': '10px'}),
        ], style={'width': '95%', 'margin': '10px auto'}),

        # Manual Cluster Mode Container (hidden by default)
        html.Div(id='manual-cluster-mode-container', children=[
            html.Div([
                html.H4("Manual Cluster Definition Mode", style={'color': '#17a2b8', 'marginBottom': '10px'}),
                html.P([
                    "1. Use ",
                    html.B("box select"),
                    " or ",
                    html.B("lasso select"),
                    " (toolbar icons) to select points on the plot below. ",
                    "2. Click a cluster button to assign selected points. ",
                    "3. Repeat until you've defined all clusters. ",
                    "4. Click 'Analyze Features' to find the best separating features."
                ], style={'color': '#555', 'marginBottom': '15px', 'fontSize': '13px'}),

                # Controls row
                html.Div([
                    # Selection count
                    html.Span([
                        html.Span("Selected: ", style={'fontWeight': 'bold'}),
                        html.Span(id='manual-selection-count', children="0"),
                        html.Span(" points", style={'marginRight': '20px'})
                    ]),

                    # Cluster assignment buttons
                    html.Button("Cluster 1", id='assign-cluster-1-btn', n_clicks=0,
                               style={'backgroundColor': '#1f77b4', 'color': 'white', 'marginRight': '5px',
                                      'padding': '6px 12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Button("Cluster 2", id='assign-cluster-2-btn', n_clicks=0,
                               style={'backgroundColor': '#ff7f0e', 'color': 'white', 'marginRight': '5px',
                                      'padding': '6px 12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Button("Cluster 3", id='assign-cluster-3-btn', n_clicks=0,
                               style={'backgroundColor': '#2ca02c', 'color': 'white', 'marginRight': '5px',
                                      'padding': '6px 12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Button("Cluster 4", id='assign-cluster-4-btn', n_clicks=0,
                               style={'backgroundColor': '#d62728', 'color': 'white', 'marginRight': '15px',
                                      'padding': '6px 12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),

                    html.Button("Clear All", id='clear-manual-clusters-btn', n_clicks=0,
                               style={'backgroundColor': '#6c757d', 'color': 'white', 'marginRight': '15px',
                                      'padding': '6px 12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),

                    html.Button("Analyze Features", id='analyze-manual-clusters-btn', n_clicks=0,
                               style={'backgroundColor': '#28a745', 'color': 'white', 'marginRight': '15px',
                                      'padding': '6px 15px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontWeight': 'bold'}),

                    html.Button("Exit Manual Mode", id='exit-manual-mode-btn', n_clicks=0,
                               style={'backgroundColor': '#dc3545', 'color': 'white',
                                      'padding': '6px 12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                ], style={'marginBottom': '10px'}),

                # Status display
                html.Div(id='manual-cluster-status', style={'marginBottom': '10px', 'fontSize': '13px'}),

            ], style={'padding': '15px', 'backgroundColor': '#e8f4f8', 'borderRadius': '4px', 'marginBottom': '10px'}),

            # Manual cluster plot
            html.Div([
                dcc.Graph(id='manual-cluster-plot', style={'height': '600px'})
            ]),

            # Analysis results
            dcc.Loading(
                id='feature-analysis-loading',
                type='circle',
                children=html.Div(id='manual-cluster-analysis-result', style={'marginTop': '15px'})
            ),
        ], style={'width': '95%', 'margin': '10px auto', 'display': 'none'}),

        # Cluster details toggle and display
        html.Div([
            html.Div([
                dcc.Checklist(
                    id='show-cluster-details-checkbox',
                    options=[{'label': ' Show cluster details on click', 'value': 'show'}],
                    value=[],
                    style={'display': 'inline-block', 'marginRight': '20px'}
                ),
                html.Span("(Click on any point in the PRPD plot above to see cluster statistics and classification details)",
                         style={'color': '#666', 'fontSize': '12px', 'fontStyle': 'italic'})
            ], style={'marginBottom': '10px'}),

            # Cluster feature selector (collapsible)
            html.Details([
                html.Summary("Select Cluster Features to Display", style={
                    'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '12px',
                    'padding': '5px', 'backgroundColor': '#e3f2fd', 'borderRadius': '4px'
                }),
                html.Div([
                    html.Div([
                        html.Button("Select All Mean", id='cluster-feat-select-mean', n_clicks=0,
                                   style={'marginRight': '5px', 'padding': '3px 8px', 'fontSize': '11px'}),
                        html.Button("Select All Trimmed Mean", id='cluster-feat-select-trimmed', n_clicks=0,
                                   style={'marginRight': '5px', 'padding': '3px 8px', 'fontSize': '11px'}),
                        html.Button("Select PRPD Features", id='cluster-feat-select-prpd', n_clicks=0,
                                   style={'marginRight': '5px', 'padding': '3px 8px', 'fontSize': '11px'}),
                        html.Button("Clear All", id='cluster-feat-select-none', n_clicks=0,
                                   style={'marginRight': '5px', 'padding': '3px 8px', 'fontSize': '11px'}),
                    ], style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='cluster-feature-selector',
                        options=[],  # Will be populated dynamically
                        value=['mean_absolute_amplitude', 'mean_rise_time', 'mean_phase_angle',
                               'trimmed_mean_absolute_amplitude', 'trimmed_mean_rise_time'],
                        multi=True,
                        placeholder="Select features to display...",
                        style={'fontSize': '11px'}
                    ),
                ], style={'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'marginTop': '5px'})
            ], style={'marginBottom': '10px'}),

            html.Div(id='cluster-details-display', style={
                'display': 'none',
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'border': '1px solid #dee2e6',
                'borderRadius': '5px',
                'maxHeight': '500px',
                'overflowY': 'auto'
            })
        ], style={'width': '95%', 'margin': '10px auto'}),

        # PD Type PRPD - full row
        html.Div([
            dcc.Graph(id='pdtype-prpd', style={'height': '600px'})
        ], style={'width': '95%', 'margin': 'auto'}),

        # Third row: Waveform (left) and Features panel (right)
        html.Div([
            # Waveform viewer
            html.Div([
                dcc.Graph(id='waveform-plot', style={'height': '450px'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            # Feature display area
            html.Div([
                html.Div([
                    html.Label("Show Features:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                    dcc.Checklist(
                        id='feature-toggles',
                        options=[],  # Will be populated by callback
                        value=DEFAULT_VISIBLE_FEATURES,
                        inline=True,
                        style={'fontSize': '11px'},
                        inputStyle={'marginRight': '3px'},
                        labelStyle={'marginRight': '12px'}
                    ),
                ], style={'marginBottom': '10px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '10px'}),
                html.Div(id='feature-display', style={
                    'fontSize': '12px',
                    'fontFamily': 'monospace',
                    'backgroundColor': '#f8f8f8',
                    'padding': '10px',
                    'borderRadius': '4px',
                    'maxHeight': '350px',
                    'overflowY': 'auto'
                })
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                      'padding': '10px', 'backgroundColor': '#fff', 'border': '1px solid #ddd', 'borderRadius': '4px'})
        ], style={'width': '95%', 'margin': 'auto'}),

        # Fourth row: Phase Distribution Histogram
        html.Div([
            dcc.Graph(id='histogram', style={'height': '450px'})
        ], style={'width': '95%', 'margin': '20px auto'}),

        # Feature Analysis Section (unified)
        html.Div([
            html.Details([
                html.Summary("Feature Analysis", style={
                    'fontWeight': 'bold', 'fontSize': '16px', 'cursor': 'pointer',
                    'padding': '10px', 'backgroundColor': '#e8f4f8', 'borderRadius': '4px'
                }),
                html.Div([
                    # Feature Recommendation Controls
                    html.Div([
                        html.H4("Optimal Feature Selection", style={'marginTop': '15px', 'marginBottom': '10px'}),
                        html.P("Select optimal features based on PCA importance while avoiding redundant correlated features.",
                               style={'color': '#666', 'fontSize': '12px', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div([
                                html.Label("Number of Features:", style={'fontWeight': 'bold'}),
                                dcc.Input(
                                    id='n-features-input',
                                    type='number',
                                    value=5,
                                    min=1,
                                    max=20,
                                    step=1,
                                    style={'width': '80px', 'marginLeft': '10px'}
                                ),
                            ], style={'display': 'inline-block', 'marginRight': '30px'}),
                            html.Div([
                                html.Label("Correlation Threshold:", style={'fontWeight': 'bold'}),
                                dcc.Input(
                                    id='correlation-threshold-input',
                                    type='number',
                                    value=0.85,
                                    min=0.5,
                                    max=0.99,
                                    step=0.05,
                                    style={'width': '80px', 'marginLeft': '10px'}
                                ),
                                html.Span(" (features above this |r| are considered redundant)",
                                         style={'color': '#666', 'fontSize': '11px', 'marginLeft': '5px'})
                            ], style={'display': 'inline-block', 'marginRight': '30px'}),
                            html.Button("Calculate Best Features", id='calc-best-features-btn',
                                       style={'backgroundColor': '#4CAF50', 'color': 'white',
                                              'padding': '8px 16px', 'border': 'none', 'borderRadius': '4px',
                                              'cursor': 'pointer', 'fontWeight': 'bold'}),
                            html.Button("Apply to Clustering", id='apply-recommended-features-btn',
                                       style={'backgroundColor': '#2196F3', 'color': 'white',
                                              'padding': '8px 16px', 'border': 'none', 'borderRadius': '4px',
                                              'cursor': 'pointer', 'fontWeight': 'bold', 'marginLeft': '10px',
                                              'display': 'none'},
                                       disabled=True),
                            html.Button("Analyze All Datasets", id='analyze-all-datasets-btn',
                                       style={'backgroundColor': '#9C27B0', 'color': 'white',
                                              'padding': '8px 16px', 'border': 'none', 'borderRadius': '4px',
                                              'cursor': 'pointer', 'fontWeight': 'bold', 'marginLeft': '20px'}),
                        ], style={'marginBottom': '15px'}),

                        # Recommended Features Display (single dataset)
                        html.Div(id='recommended-features-display', style={
                            'padding': '15px', 'backgroundColor': '#f5f5f5', 'borderRadius': '4px',
                            'marginBottom': '20px', 'display': 'none'
                        }),

                        # All Datasets Analysis Display
                        html.Div(id='all-datasets-analysis-display', style={
                            'padding': '15px', 'backgroundColor': '#f0f0ff', 'borderRadius': '4px',
                            'marginBottom': '20px', 'display': 'none'
                        }),
                    ], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),

                    # Visualization Toggles
                    html.Div([
                        html.H4("Visualizations", style={'marginTop': '15px', 'marginBottom': '10px'}),
                        html.Div([
                            dcc.Checklist(
                                id='show-correlation-matrix-checkbox',
                                options=[{'label': ' Correlation Matrix', 'value': 'show'}],
                                value=[],
                                style={'display': 'inline-block', 'marginRight': '30px', 'fontWeight': 'bold'}
                            ),
                            dcc.Checklist(
                                id='show-pca-loadings-checkbox',
                                options=[{'label': ' PCA Loadings Table', 'value': 'show'}],
                                value=[],
                                style={'display': 'inline-block', 'fontWeight': 'bold'}
                            ),
                        ], style={'marginBottom': '10px'}),
                        html.Span("Correlation Matrix: Pearson R values | PCA Loadings: Feature contributions to principal components",
                                 style={'color': '#666', 'fontSize': '11px', 'fontStyle': 'italic'}),
                    ], style={'padding': '10px'}),

                    # Side-by-side visualization containers
                    html.Div([
                        html.Div([
                            dcc.Graph(id='correlation-matrix', style={'height': '600px'})
                        ], id='correlation-matrix-container', style={'display': 'none', 'width': '49%',
                                                                      'verticalAlign': 'top'}),
                        html.Div([
                            dcc.Graph(id='pca-loadings', style={'height': '600px'})
                        ], id='pca-loadings-container', style={'display': 'none', 'width': '49%',
                                                               'verticalAlign': 'top'}),
                    ], style={'marginTop': '10px'}),

                ], style={'padding': '10px'})
            ], open=False)
        ], style={'width': '95%', 'margin': '20px auto', 'border': '1px solid #ddd', 'borderRadius': '4px'}),

        # PCA Plot (shown only for K-means)
        html.Div([
            html.Div([
                dcc.Graph(id='pca-plot', style={'height': '500px'})
            ])
        ], id='pca-container', style={'width': '95%', 'margin': '20px auto', 'display': 'none'}),

        # Hidden stores
        dcc.Store(id='selected-index'),
        dcc.Store(id='current-data-store'),
        dcc.Store(id='selected-waveform-idx'),
        dcc.Store(id='feature-toggle-store', data=DEFAULT_VISIBLE_FEATURES),
        dcc.Store(id='reanalysis-trigger', data=0),
        dcc.Store(id='recommended-features-store', data=[]),
        dcc.Store(id='feature-analysis-data-store', data={}),  # Stores PCA data for recalculation
        # Per-dataset feature selections (persisted in localStorage)
        dcc.Store(id='pulse-features-per-dataset', storage_type='local', data={}),
        dcc.Store(id='cluster-features-per-dataset', storage_type='local', data={}),
        # Progress tracking for batch operations
        dcc.Store(id='batch-progress-store', data={'active': False, 'current': 0, 'total': 0, 'operation': '', 'dataset': ''}),
        dcc.Interval(id='progress-interval', interval=500, disabled=True),  # Poll every 500ms when active
        # Manual cluster definition storage
        dcc.Store(id='manual-cluster-assignments', data={}),  # {point_index: cluster_number}
        dcc.Store(id='current-selection-indices', data=[]),  # Currently selected point indices
        dcc.Store(id='manual-mode-active', data=False),  # Whether manual mode is active
    ])

    # Callbacks for pulse features selection buttons
    @app.callback(
        Output('pulse-features-checklist', 'value'),
        [Input('pulse-features-select-all', 'n_clicks'),
         Input('pulse-features-select-none', 'n_clicks'),
         Input('pulse-features-reset', 'n_clicks')],
        [State('pulse-features-checklist', 'value')],
        prevent_initial_call=True
    )
    def update_pulse_features(select_all, select_none, reset, current_value):
        triggered = ctx.triggered_id
        if triggered == 'pulse-features-select-all':
            return PULSE_FEATURES
        elif triggered == 'pulse-features-select-none':
            return []
        elif triggered == 'pulse-features-reset':
            return DEFAULT_CLUSTERING_FEATURES
        return current_value

    # Callbacks for cluster features selection buttons
    @app.callback(
        Output('cluster-features-checklist', 'value'),
        [Input('cluster-features-select-all', 'n_clicks'),
         Input('cluster-features-select-none', 'n_clicks'),
         Input('cluster-features-reset', 'n_clicks')],
        [State('cluster-features-checklist', 'value')],
        prevent_initial_call=True
    )
    def update_cluster_features(select_all, select_none, reset, current_value):
        triggered = ctx.triggered_id
        if triggered == 'cluster-features-select-all':
            return CLUSTER_FEATURES
        elif triggered == 'cluster-features-select-none':
            return []
        elif triggered == 'cluster-features-reset':
            return DEFAULT_CLASSIFICATION_FEATURES
        return current_value

    # Save pulse features to per-dataset store when they change
    @app.callback(
        Output('pulse-features-per-dataset', 'data'),
        [Input('pulse-features-checklist', 'value')],
        [State('dataset-dropdown', 'value'),
         State('pulse-features-per-dataset', 'data')],
        prevent_initial_call=True
    )
    def save_pulse_features_per_dataset(features, dataset, stored_data):
        if not dataset or features is None:
            raise PreventUpdate
        stored_data = stored_data or {}
        stored_data[dataset] = features
        return stored_data

    # Save cluster features to per-dataset store when they change
    @app.callback(
        Output('cluster-features-per-dataset', 'data'),
        [Input('cluster-features-checklist', 'value')],
        [State('dataset-dropdown', 'value'),
         State('cluster-features-per-dataset', 'data')],
        prevent_initial_call=True
    )
    def save_cluster_features_per_dataset(features, dataset, stored_data):
        if not dataset or features is None:
            raise PreventUpdate
        stored_data = stored_data or {}
        stored_data[dataset] = features
        return stored_data

    # Load pulse features when dataset changes
    @app.callback(
        [Output('pulse-features-checklist', 'value', allow_duplicate=True),
         Output('features-source-info', 'children')],
        [Input('dataset-dropdown', 'value')],
        [State('pulse-features-per-dataset', 'data'),
         State('clustering-method-radio', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def load_pulse_features_for_dataset(dataset, stored_data, clustering_method):
        if not dataset:
            raise PreventUpdate

        # First check if we have saved selections for this dataset
        stored_data = stored_data or {}
        if dataset in stored_data:
            n_features = len(stored_data[dataset])
            return stored_data[dataset], f"Loaded {n_features} features from session storage"

        # If no saved selection, try to read from cluster file
        data_path = loader.get_dataset_path(dataset)
        clean_prefix = loader.get_clean_prefix(dataset)
        if data_path and clean_prefix:
            method = clustering_method or 'dbscan'
            cluster_file = os.path.join(data_path, f"{clean_prefix}-clusters-{method}.csv")
            features_from_file = read_features_from_cluster_file(cluster_file)
            if features_from_file:
                # Filter to only include features that exist in PULSE_FEATURES
                valid_features = [f for f in features_from_file if f in PULSE_FEATURES]
                if valid_features:
                    return valid_features, f"Loaded {len(valid_features)} features from last {method.upper()} clustering"

        # Don't change if no saved selection and no cluster file - keeps current/default selection
        raise PreventUpdate

    # Load cluster features when dataset changes
    @app.callback(
        Output('cluster-features-checklist', 'value', allow_duplicate=True),
        [Input('dataset-dropdown', 'value')],
        [State('cluster-features-per-dataset', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def load_cluster_features_for_dataset(dataset, stored_data):
        if not dataset:
            raise PreventUpdate
        stored_data = stored_data or {}
        if dataset in stored_data:
            return stored_data[dataset]
        # Don't change if no saved selection - keeps current/default selection
        raise PreventUpdate

    # Load noise threshold when dataset changes
    @app.callback(
        [Output('noise-threshold-display', 'children'),
         Output('noise-threshold-details', 'children')],
        [Input('dataset-dropdown', 'value')]
    )
    def load_noise_threshold(dataset):
        if not dataset:
            return "N/A", ""

        data_path = loader.get_dataset_path(dataset)
        clean_prefix = loader.get_clean_prefix(dataset)

        if not data_path or not clean_prefix:
            return "N/A", ""

        # Try to load from metadata file
        metadata = load_dataset_metadata(data_path, clean_prefix)

        if 'noise_threshold' in metadata:
            threshold = metadata['noise_threshold']
            min_amp = metadata.get('min_absolute_amplitude', 0)
            n_samples = metadata.get('n_samples', 0)
            return (
                f"{threshold*1000:.3f} mV",
                f"Min amplitude: {min_amp*1000:.3f} mV | Samples: {n_samples} | ADC step: {ADC_STEP_V*1000:.3f} mV"
            )

        return "Not calculated", "Click Calculate to compute"

    # Calculate noise threshold when button is clicked
    @app.callback(
        [Output('noise-threshold-display', 'children', allow_duplicate=True),
         Output('noise-threshold-details', 'children', allow_duplicate=True)],
        [Input('calc-noise-threshold-btn', 'n_clicks')],
        [State('dataset-dropdown', 'value')],
        prevent_initial_call=True
    )
    def calculate_and_save_noise_threshold(n_clicks, dataset):
        if not n_clicks or not dataset:
            raise PreventUpdate

        data_path = loader.get_dataset_path(dataset)
        clean_prefix = loader.get_clean_prefix(dataset)

        if not data_path or not clean_prefix:
            return "Error", "Could not determine dataset path"

        # Load features
        data = loader.load_all(dataset)
        if data['features'] is None or data['feature_names'] is None:
            return "Error", "No feature data available"

        # Calculate noise threshold
        threshold_info = calculate_noise_threshold(data['features'], data['feature_names'])

        if threshold_info is None:
            return "Error", "Could not calculate threshold (missing amplitude features)"

        # Save to metadata file
        save_dataset_metadata(data_path, clean_prefix, threshold_info)

        threshold = threshold_info['noise_threshold']
        min_amp = threshold_info['min_absolute_amplitude']
        n_samples = threshold_info['n_samples']

        return (
            f"{threshold*1000:.3f} mV",
            f"Min amplitude: {min_amp*1000:.3f} mV | Samples: {n_samples} | ADC step: {ADC_STEP_V*1000:.3f} mV"
        )

    @app.callback(
        [Output('recommended-features-display', 'children'),
         Output('recommended-features-display', 'style'),
         Output('recommended-features-store', 'data'),
         Output('apply-recommended-features-btn', 'style'),
         Output('apply-recommended-features-btn', 'disabled'),
         Output('feature-analysis-data-store', 'data')],
        [Input('calc-best-features-btn', 'n_clicks')],
        [State('dataset-dropdown', 'value'),
         State('n-features-input', 'value'),
         State('correlation-threshold-input', 'value'),
         State('pulse-features-checklist', 'value')],
        prevent_initial_call=True
    )
    def calculate_recommended_features(n_clicks, prefix, n_features, corr_threshold, current_features):
        """Calculate and display recommended features based on PCA and correlation analysis."""
        hidden_style = {'display': 'none'}
        btn_hidden = {'display': 'none'}
        empty_data = {}

        if not n_clicks or not prefix:
            return [], hidden_style, [], btn_hidden, True, empty_data

        # Load data
        data = loader.load_all(prefix)
        if data['features'] is None or data['feature_names'] is None:
            return html.Div("No feature data available", style={'color': 'red'}), \
                   {'padding': '15px', 'backgroundColor': '#ffebee', 'borderRadius': '4px',
                    'marginBottom': '20px', 'display': 'block'}, [], btn_hidden, True, empty_data

        # Get subset of features based on current selection
        if current_features and len(current_features) > 0:
            indices = []
            names = []
            for feat in current_features:
                if feat in data['feature_names']:
                    indices.append(data['feature_names'].index(feat))
                    names.append(feat)
            if len(indices) < 2:
                return html.Div("Select at least 2 features to analyze", style={'color': 'orange'}), \
                       {'padding': '15px', 'backgroundColor': '#fff3e0', 'borderRadius': '4px',
                        'marginBottom': '20px', 'display': 'block'}, [], btn_hidden, True, empty_data
            features_subset = data['features'][:, indices]
            feature_names = names
        else:
            features_subset = data['features']
            feature_names = list(data['feature_names'])

        # Compute recommendations
        n_features = min(n_features or 5, len(feature_names))
        corr_threshold = corr_threshold or 0.85

        result = compute_recommended_features(
            features_subset, feature_names, n_features, corr_threshold
        )

        if result.get('error'):
            return html.Div(f"Error: {result['error']}", style={'color': 'red'}), \
                   {'padding': '15px', 'backgroundColor': '#ffebee', 'borderRadius': '4px',
                    'marginBottom': '20px', 'display': 'block'}, [], btn_hidden, True, empty_data

        # Build display
        recommended = result['recommended']
        scores = result['scores']
        details = result['details']
        variance = result['variance_explained']

        # Create checklist options for manual selection (sorted by importance)
        checklist_options = []
        for feat, score, reason in details:
            label = f"{feat} (importance: {score:.1f})"
            if 'Redundant' in reason:
                # Extract correlation info
                label += f" - {reason}"
            checklist_options.append({'label': label, 'value': feat})

        # Store analysis data for recalculation
        analysis_data = {
            'scores': scores,
            'details': [[f, s, r] for f, s, r in details],  # Convert tuples to lists for JSON
            'feature_names': feature_names,
            'prefix': prefix
        }

        display_content = html.Div([
            html.H5("Feature Analysis Results", style={'marginBottom': '10px'}),

            # Initial variance display
            html.Div([
                html.Span("Algorithm-Recommended Variance Coverage: ", style={'fontWeight': 'bold'}),
                html.Span(f"{variance:.1f}%", style={'color': '#1976d2', 'fontWeight': 'bold', 'fontSize': '16px'}),
                html.Span(f" ({len(recommended)} features)", style={'color': '#666', 'fontSize': '12px', 'marginLeft': '10px'})
            ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': '#e3f2fd', 'borderRadius': '4px'}),

            # Manual selection section
            html.Div([
                html.H6("Manual Feature Selection", style={'marginBottom': '10px'}),
                html.P("Select/deselect features below, then click 'Recalculate Variance' to see updated coverage.",
                       style={'color': '#666', 'fontSize': '12px', 'marginBottom': '10px'}),

                # Checklist for manual selection
                dcc.Checklist(
                    id='manual-feature-selection',
                    options=checklist_options,
                    value=recommended,  # Pre-select recommended features
                    style={'columns': '2', 'columnGap': '20px'},
                    inputStyle={'marginRight': '8px'},
                    labelStyle={'display': 'block', 'marginBottom': '8px', 'fontSize': '12px'}
                ),

                # Recalculate button and result
                html.Div([
                    html.Button("Recalculate Variance", id='recalc-variance-btn',
                               style={'backgroundColor': '#ff9800', 'color': 'white',
                                      'padding': '8px 16px', 'border': 'none', 'borderRadius': '4px',
                                      'cursor': 'pointer', 'fontWeight': 'bold', 'marginTop': '15px'}),
                    html.Button("Recluster with Selected Features", id='recluster-btn',
                               style={'backgroundColor': '#e91e63', 'color': 'white',
                                      'padding': '8px 16px', 'border': 'none', 'borderRadius': '4px',
                                      'cursor': 'pointer', 'fontWeight': 'bold', 'marginTop': '15px',
                                      'marginLeft': '10px'}),
                    html.Div(id='recalc-variance-result', style={'marginTop': '10px'}),
                    html.Div(id='recluster-result', style={'marginTop': '10px'})
                ]),
            ], style={'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '4px',
                      'border': '1px solid #ddd', 'marginBottom': '15px'}),

            # Show correlation info for reference
            html.Details([
                html.Summary("Feature Importance & Correlation Notes", style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.Div([
                    html.P("Features marked 'Redundant' are highly correlated with higher-ranked features. "
                           "Including both may not improve results significantly.", style={'fontSize': '12px', 'color': '#666'}),
                    html.Table([
                        html.Tr([
                            html.Th("Feature", style={'textAlign': 'left', 'padding': '6px', 'borderBottom': '1px solid #ddd'}),
                            html.Th("Score", style={'textAlign': 'center', 'padding': '6px', 'borderBottom': '1px solid #ddd'}),
                            html.Th("Note", style={'textAlign': 'left', 'padding': '6px', 'borderBottom': '1px solid #ddd'})
                        ])
                    ] + [
                        html.Tr([
                            html.Td(feat, style={'padding': '4px 6px', 'fontSize': '12px'}),
                            html.Td(f"{score:.1f}", style={'padding': '4px 6px', 'textAlign': 'center', 'fontSize': '12px'}),
                            html.Td(reason, style={'padding': '4px 6px', 'fontSize': '11px',
                                                   'color': '#2e7d32' if 'Selected' in reason else '#f57c00' if 'Redundant' in reason else '#757575'})
                        ]) for feat, score, reason in details
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '10px'})
                ])
            ], open=False)
        ])

        show_style = {'padding': '15px', 'backgroundColor': '#f5f5f5', 'borderRadius': '4px',
                      'marginBottom': '20px', 'display': 'block'}

        btn_style = {'backgroundColor': '#2196F3', 'color': 'white',
                     'padding': '8px 16px', 'border': 'none', 'borderRadius': '4px',
                     'cursor': 'pointer', 'fontWeight': 'bold', 'marginLeft': '10px',
                     'display': 'inline-block'}

        return display_content, show_style, recommended, btn_style, False, analysis_data

    @app.callback(
        Output('pulse-features-checklist', 'value', allow_duplicate=True),
        [Input('apply-recommended-features-btn', 'n_clicks')],
        [State('manual-feature-selection', 'value')],
        prevent_initial_call=True
    )
    def apply_recommended_features(n_clicks, manual_selection):
        """Apply manually selected features to the pulse features checklist."""
        if not n_clicks or not manual_selection:
            raise PreventUpdate
        return manual_selection

    @app.callback(
        [Output('recalc-variance-result', 'children'),
         Output('recommended-features-store', 'data', allow_duplicate=True)],
        [Input('recalc-variance-btn', 'n_clicks')],
        [State('manual-feature-selection', 'value'),
         State('feature-analysis-data-store', 'data'),
         State('dataset-dropdown', 'value'),
         State('pulse-features-checklist', 'value')],
        prevent_initial_call=True
    )
    def recalculate_variance(n_clicks, manual_selection, analysis_data, prefix, current_features):
        """Recalculate variance coverage for manually selected features."""
        if not n_clicks:
            raise PreventUpdate

        if not manual_selection:
            return html.Div("Please select at least one feature", style={'color': 'orange'}), []

        if not prefix or not analysis_data:
            return html.Div("No analysis data available. Click 'Calculate Best Features' first.",
                          style={'color': 'red'}), []

        # Load data
        data = loader.load_all(prefix)
        if data['features'] is None or data['feature_names'] is None:
            return html.Div("No feature data available", style={'color': 'red'}), []

        # Get subset of features based on current pulse features selection
        if current_features and len(current_features) > 0:
            indices = []
            names = []
            for feat in current_features:
                if feat in data['feature_names']:
                    indices.append(data['feature_names'].index(feat))
                    names.append(feat)
            if len(indices) < 2:
                return html.Div("Too few features available", style={'color': 'red'}), []
            features_subset = data['features'][:, indices]
            feature_names = names
        else:
            features_subset = data['features']
            feature_names = list(data['feature_names'])

        # Get indices for manually selected features
        selected_indices = []
        selected_names = []
        for feat in manual_selection:
            if feat in feature_names:
                selected_indices.append(feature_names.index(feat))
                selected_names.append(feat)

        if len(selected_indices) < 1:
            return html.Div("No valid features selected", style={'color': 'orange'}), []

        # Compute variance explained by selected features using PCA
        if not PCA_AVAILABLE:
            return html.Div("PCA not available", style={'color': 'red'}), []

        try:
            # Get selected feature data
            selected_features = features_subset[:, selected_indices]
            selected_features = np.nan_to_num(selected_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Check for constant features
            feature_stds = np.std(selected_features, axis=0)
            valid_mask = feature_stds > 1e-10
            if not np.any(valid_mask):
                return html.Div("All selected features are constant", style={'color': 'orange'}), manual_selection

            valid_features = selected_features[:, valid_mask]
            valid_names = [selected_names[i] for i in range(len(selected_names)) if valid_mask[i]]

            # Scale and run PCA
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(valid_features)

            n_components = min(len(valid_names), features_scaled.shape[0])
            pca = PCA(n_components=n_components)
            pca.fit(features_scaled)

            # Get variance explained
            var_explained = pca.explained_variance_ratio_
            total_variance = sum(var_explained) * 100

            # Build result display
            result_content = html.Div([
                html.Div([
                    html.Span("Your Selection Variance Coverage: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{total_variance:.1f}%", style={'color': '#2e7d32' if total_variance >= 70 else '#f57c00',
                                                               'fontWeight': 'bold', 'fontSize': '18px'}),
                    html.Span(f" ({len(valid_names)} features)", style={'color': '#666', 'marginLeft': '10px'})
                ], style={'padding': '10px', 'backgroundColor': '#e8f5e9' if total_variance >= 70 else '#fff3e0',
                          'borderRadius': '4px', 'marginBottom': '10px'}),
                html.Div([
                    html.Span("Selected: ", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    html.Span(", ".join(valid_names), style={'fontFamily': 'monospace', 'fontSize': '12px'})
                ]),
                html.Div([
                    html.Span("Variance per PC: ", style={'fontSize': '11px', 'color': '#666'}),
                    html.Span(", ".join([f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(var_explained[:5])]),
                             style={'fontSize': '11px', 'color': '#666'})
                ], style={'marginTop': '5px'}) if len(var_explained) > 0 else None
            ])

            return result_content, manual_selection

        except Exception as e:
            return html.Div(f"Error calculating variance: {str(e)}", style={'color': 'red'}), manual_selection

    @app.callback(
        Output('recluster-result', 'children'),
        [Input('recluster-btn', 'n_clicks')],
        [State('manual-feature-selection', 'value'),
         State('dataset-dropdown', 'value'),
         State('clustering-method-radio', 'value')],
        prevent_initial_call=True
    )
    def recluster_with_features(n_clicks, selected_features, prefix, clustering_method):
        """Run clustering with the selected features."""
        if not n_clicks:
            raise PreventUpdate

        if not selected_features or len(selected_features) < 2:
            return html.Div("Please select at least 2 features for clustering",
                          style={'color': 'orange', 'padding': '10px'})

        if not prefix:
            return html.Div("No dataset selected", style={'color': 'red', 'padding': '10px'})

        # Get dataset info
        data_path = loader.get_dataset_path(prefix)
        clean_prefix = loader.get_clean_prefix(prefix)

        if not data_path or not clean_prefix:
            return html.Div("Could not determine dataset path", style={'color': 'red', 'padding': '10px'})

        # Build the clustering command
        features_str = ','.join(selected_features)
        method = clustering_method or 'dbscan'

        import subprocess
        import sys

        try:
            # Run clustering
            cmd = [
                sys.executable, 'cluster_pulses.py',
                '--input-dir', data_path,
                '--method', method,
                '--features', features_str
            ]

            # Add input file for specific dataset
            input_file = os.path.join(data_path, f"{clean_prefix}-features.csv")
            if os.path.exists(input_file):
                cmd.extend(['--input', input_file])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                # Also run aggregation and classification
                agg_cmd = [
                    sys.executable, 'aggregate_cluster_features.py',
                    '--input-dir', data_path,
                    '--method', method,
                    '--file', clean_prefix
                ]
                subprocess.run(agg_cmd, capture_output=True, text=True, timeout=60)

                class_cmd = [
                    sys.executable, 'classify_pd_type.py',
                    '--input-dir', data_path,
                    '--method', method,
                    '--file', clean_prefix
                ]
                subprocess.run(class_cmd, capture_output=True, text=True, timeout=60)

                return html.Div([
                    html.Div("✓ Reclustering complete!", style={'color': '#2e7d32', 'fontWeight': 'bold'}),
                    html.Div(f"Method: {method.upper()}", style={'fontSize': '12px', 'color': '#666'}),
                    html.Div(f"Features: {features_str}", style={'fontSize': '12px', 'color': '#666'}),
                    html.Div("Refresh the page or change dataset and back to see updated results.",
                            style={'fontSize': '12px', 'color': '#1976d2', 'marginTop': '5px', 'fontStyle': 'italic'})
                ], style={'padding': '10px', 'backgroundColor': '#e8f5e9', 'borderRadius': '4px'})
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                return html.Div([
                    html.Div("✗ Clustering failed", style={'color': '#d32f2f', 'fontWeight': 'bold'}),
                    html.Pre(error_msg, style={'fontSize': '10px', 'color': '#666', 'whiteSpace': 'pre-wrap'})
                ], style={'padding': '10px', 'backgroundColor': '#ffebee', 'borderRadius': '4px'})

        except subprocess.TimeoutExpired:
            return html.Div("Clustering timed out (>120s)", style={'color': 'red', 'padding': '10px'})
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={'color': 'red', 'padding': '10px'})

    @app.callback(
        Output('recluster-main-result', 'children'),
        [Input('recluster-main-btn', 'n_clicks')],
        [State('pulse-features-checklist', 'value'),
         State('dataset-dropdown', 'value'),
         State('clustering-method-radio', 'value')],
        prevent_initial_call=True
    )
    def recluster_main_with_features(n_clicks, selected_features, prefix, clustering_method):
        """Run clustering with the selected pulse features from Advanced Options."""
        if not n_clicks:
            raise PreventUpdate

        if not selected_features or len(selected_features) < 2:
            return html.Div("Please select at least 2 features for clustering",
                          style={'color': 'orange', 'padding': '10px'})

        if not prefix:
            return html.Div("No dataset selected", style={'color': 'red', 'padding': '10px'})

        # Get dataset info
        data_path = loader.get_dataset_path(prefix)
        clean_prefix = loader.get_clean_prefix(prefix)

        if not data_path or not clean_prefix:
            return html.Div("Could not determine dataset path", style={'color': 'red', 'padding': '10px'})

        # Build the clustering command
        features_str = ','.join(selected_features)
        method = clustering_method or 'dbscan'

        import subprocess
        import sys

        try:
            # Run clustering
            cmd = [
                sys.executable, 'cluster_pulses.py',
                '--input-dir', data_path,
                '--method', method,
                '--features', features_str
            ]

            # Add input file for specific dataset
            input_file = os.path.join(data_path, f"{clean_prefix}-features.csv")
            if os.path.exists(input_file):
                cmd.extend(['--input', input_file])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                # Also run aggregation and classification
                agg_cmd = [
                    sys.executable, 'aggregate_cluster_features.py',
                    '--input-dir', data_path,
                    '--method', method,
                    '--file', clean_prefix
                ]
                subprocess.run(agg_cmd, capture_output=True, text=True, timeout=60)

                class_cmd = [
                    sys.executable, 'classify_pd_type.py',
                    '--input-dir', data_path,
                    '--method', method,
                    '--file', clean_prefix
                ]
                subprocess.run(class_cmd, capture_output=True, text=True, timeout=60)

                return html.Div([
                    html.Div("✓ Reclustering complete!", style={'color': '#2e7d32', 'fontWeight': 'bold'}),
                    html.Div(f"Method: {method.upper()} | Features: {len(selected_features)}", style={'fontSize': '12px', 'color': '#666'}),
                    html.Div("Refresh the page or change dataset and back to see updated results.",
                            style={'fontSize': '12px', 'color': '#1976d2', 'marginTop': '5px', 'fontStyle': 'italic'})
                ], style={'padding': '10px', 'backgroundColor': '#e8f5e9', 'borderRadius': '4px'})
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                return html.Div([
                    html.Div("✗ Clustering failed", style={'color': '#d32f2f', 'fontWeight': 'bold'}),
                    html.Pre(error_msg, style={'fontSize': '10px', 'color': '#666', 'whiteSpace': 'pre-wrap'})
                ], style={'padding': '10px', 'backgroundColor': '#ffebee', 'borderRadius': '4px'})

        except subprocess.TimeoutExpired:
            return html.Div("Clustering timed out (>120s)", style={'color': 'red', 'padding': '10px'})
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={'color': 'red', 'padding': '10px'})

    @app.callback(
        Output('reclassify-result', 'children'),
        [Input('reclassify-btn', 'n_clicks')],
        [State('cluster-features-checklist', 'value'),
         State('dataset-dropdown', 'value'),
         State('clustering-method-radio', 'value')],
        prevent_initial_call=True
    )
    def reclassify_with_features(n_clicks, selected_features, prefix, clustering_method):
        """Run classification with the selected cluster features."""
        if not n_clicks:
            raise PreventUpdate

        if not selected_features or len(selected_features) < 1:
            return html.Div("Please select at least 1 feature for classification",
                          style={'color': 'orange', 'padding': '10px'})

        if not prefix:
            return html.Div("No dataset selected", style={'color': 'red', 'padding': '10px'})

        # Get dataset info
        data_path = loader.get_dataset_path(prefix)
        clean_prefix = loader.get_clean_prefix(prefix)

        if not data_path or not clean_prefix:
            return html.Div("Could not determine dataset path", style={'color': 'red', 'padding': '10px'})

        # Build the classification command
        features_str = ','.join(selected_features)
        method = clustering_method or 'dbscan'

        import subprocess
        import sys

        try:
            # Run classification with selected features
            cmd = [
                sys.executable, 'classify_pd_type.py',
                '--input-dir', data_path,
                '--method', method,
                '--file', clean_prefix,
                '--cluster-features', features_str
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return html.Div([
                    html.Div("✓ Reclassification complete!", style={'color': '#2e7d32', 'fontWeight': 'bold'}),
                    html.Div(f"Method: {method.upper()}", style={'fontSize': '12px', 'color': '#666'}),
                    html.Div(f"Features used: {len(selected_features)}", style={'fontSize': '12px', 'color': '#666'}),
                    html.Div("Refresh the page or change dataset and back to see updated results.",
                            style={'fontSize': '12px', 'color': '#1976d2', 'marginTop': '5px', 'fontStyle': 'italic'})
                ], style={'padding': '10px', 'backgroundColor': '#f3e5f5', 'borderRadius': '4px'})
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                return html.Div([
                    html.Div("✗ Classification failed", style={'color': '#d32f2f', 'fontWeight': 'bold'}),
                    html.Pre(error_msg, style={'fontSize': '10px', 'color': '#666', 'whiteSpace': 'pre-wrap'})
                ], style={'padding': '10px', 'backgroundColor': '#ffebee', 'borderRadius': '4px'})

        except subprocess.TimeoutExpired:
            return html.Div("Classification timed out (>60s)", style={'color': 'red', 'padding': '10px'})
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={'color': 'red', 'padding': '10px'})

    @app.callback(
        Output('cluster-explanation-display', 'children'),
        [Input('generate-cluster-explanation-btn', 'n_clicks')],
        [State('show-cluster-explanation-checkbox', 'value'),
         State('dataset-dropdown', 'value'),
         State('clustering-method-radio', 'value'),
         State('pulse-features-checklist', 'value')],
        prevent_initial_call=True
    )
    def generate_cluster_explanation(n_clicks, show_explanation, dataset, clustering_method, selected_features):
        """Generate a decision tree explanation of how clusters were assigned."""
        if not n_clicks:
            raise PreventUpdate

        if not show_explanation or 'show' not in show_explanation:
            return html.Div("Enable the checkbox above to generate cluster explanations.",
                          style={'color': '#666', 'fontStyle': 'italic'})

        if not dataset:
            return html.Div("No dataset selected", style={'color': 'red'})

        # Get dataset paths
        data_path = loader.get_dataset_path(dataset)
        clean_prefix = loader.get_clean_prefix(dataset)

        if not data_path or not clean_prefix:
            return html.Div("Could not determine dataset path", style={'color': 'red'})

        method = clustering_method or 'dbscan'

        # Load features file
        features_file = os.path.join(data_path, f"{clean_prefix}-features.csv")
        cluster_file = os.path.join(data_path, f"{clean_prefix}-clusters-{method}.csv")

        if not os.path.exists(features_file):
            return html.Div(f"Features file not found: {features_file}", style={'color': 'red'})

        if not os.path.exists(cluster_file):
            return html.Div(f"Cluster file not found: {cluster_file}", style={'color': 'red'})

        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.preprocessing import StandardScaler
            import numpy as np

            # Load features
            features_data = []
            feature_names = None
            with open(features_file, 'r') as f:
                header = f.readline().strip()
                feature_names = header.split(',')[1:]  # Skip waveform_index
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        values = [float(v) if v else 0.0 for v in parts[1:]]
                        features_data.append(values)

            features_array = np.array(features_data)

            # Filter to selected features if specified
            if selected_features:
                feature_indices = []
                used_feature_names = []
                for feat in selected_features:
                    if feat in feature_names:
                        feature_indices.append(feature_names.index(feat))
                        used_feature_names.append(feat)
                if feature_indices:
                    features_array = features_array[:, feature_indices]
                    feature_names = used_feature_names

            # Handle NaN/Inf
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Load cluster labels
            cluster_labels = []
            with open(cluster_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.startswith('waveform'):
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        cluster_labels.append(int(parts[1]))

            cluster_labels = np.array(cluster_labels)

            if len(features_array) != len(cluster_labels):
                return html.Div(f"Mismatch: {len(features_array)} features vs {len(cluster_labels)} labels",
                              style={'color': 'red'})

            # Scale features for consistency with original clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)

            # Train decision tree to explain clusters
            # Use max_depth to keep it interpretable
            dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=max(5, len(cluster_labels) // 100), random_state=42)
            dt.fit(features_scaled, cluster_labels)

            # Calculate accuracy
            accuracy = dt.score(features_scaled, cluster_labels)

            # Get unique clusters
            unique_clusters = sorted(set(cluster_labels))
            n_clusters = len([c for c in unique_clusters if c != -1])
            n_noise = np.sum(cluster_labels == -1) if -1 in cluster_labels else 0

            # Extract rules from decision tree
            def get_rules(tree, feature_names, scaler):
                """Extract human-readable rules from a decision tree."""
                tree_ = tree.tree_
                feature_name = [
                    feature_names[i] if i != -2 else "undefined!"
                    for i in tree_.feature
                ]

                rules_by_class = {}

                def recurse(node, depth, rule_parts):
                    if tree_.feature[node] != -2:  # Not a leaf
                        name = feature_name[node]
                        threshold_scaled = tree_.threshold[node]

                        # Get the feature index to unscale
                        feat_idx = tree_.feature[node]
                        # Unscale threshold: threshold_original = threshold_scaled * std + mean
                        threshold_original = threshold_scaled * scaler.scale_[feat_idx] + scaler.mean_[feat_idx]

                        # Left branch: <= threshold
                        left_rule = f"{name} ≤ {threshold_original:.4g}"
                        recurse(tree_.children_left[node], depth + 1, rule_parts + [left_rule])

                        # Right branch: > threshold
                        right_rule = f"{name} > {threshold_original:.4g}"
                        recurse(tree_.children_right[node], depth + 1, rule_parts + [right_rule])
                    else:
                        # Leaf node - get class
                        class_counts = tree_.value[node][0]
                        predicted_class = int(np.argmax(class_counts))
                        # Map back to actual cluster label
                        actual_class = dt.classes_[predicted_class]
                        n_samples = int(np.sum(class_counts))
                        confidence = class_counts[predicted_class] / n_samples if n_samples > 0 else 0

                        if actual_class not in rules_by_class:
                            rules_by_class[actual_class] = []

                        rules_by_class[actual_class].append({
                            'conditions': rule_parts.copy(),
                            'samples': n_samples,
                            'confidence': confidence
                        })

                recurse(0, 0, [])
                return rules_by_class

            rules = get_rules(dt, feature_names, scaler)

            # Build output
            output_parts = []

            # Summary
            output_parts.append(html.Div([
                html.H4("Cluster Decision Explanation", style={'marginBottom': '10px', 'color': '#1976d2'}),
                html.Div(f"Dataset: {dataset}", style={'fontSize': '11px', 'color': '#666'}),
                html.Div(f"Method: {method.upper()} | Clusters: {n_clusters}" +
                        (f" | Noise points: {n_noise}" if n_noise > 0 else ""),
                        style={'fontSize': '11px', 'color': '#666'}),
                html.Div(f"Decision tree accuracy: {accuracy:.1%} (how well the tree approximates the clustering)",
                        style={'fontSize': '11px', 'color': '#666', 'marginBottom': '15px'}),
            ]))

            # Rules for each cluster
            for cluster_id in sorted(rules.keys()):
                cluster_rules = rules[cluster_id]
                cluster_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
                total_samples = sum(r['samples'] for r in cluster_rules)

                cluster_section = [
                    html.Div(f"━━━ {cluster_name} ({total_samples} pulses) ━━━",
                            style={'fontWeight': 'bold', 'marginTop': '10px', 'marginBottom': '5px',
                                   'color': '#d32f2f' if cluster_id == -1 else '#2e7d32'})
                ]

                # Sort rules by sample count
                cluster_rules_sorted = sorted(cluster_rules, key=lambda x: -x['samples'])

                for i, rule in enumerate(cluster_rules_sorted[:5]):  # Show top 5 rules per cluster
                    if rule['conditions']:
                        rule_text = " AND ".join(rule['conditions'])
                    else:
                        rule_text = "(default path)"

                    cluster_section.append(
                        html.Div([
                            html.Span(f"Rule {i+1}: ", style={'fontWeight': 'bold'}),
                            html.Span(f"IF {rule_text}"),
                            html.Br(),
                            html.Span(f"         → {rule['samples']} pulses ({rule['confidence']:.0%} confidence)",
                                     style={'color': '#666', 'fontSize': '11px'})
                        ], style={'marginBottom': '8px', 'paddingLeft': '10px'})
                    )

                if len(cluster_rules_sorted) > 5:
                    cluster_section.append(
                        html.Div(f"         ... and {len(cluster_rules_sorted) - 5} more rules",
                                style={'color': '#999', 'fontSize': '11px', 'paddingLeft': '10px'})
                    )

                output_parts.append(html.Div(cluster_section))

            # Feature importance
            importances = dt.feature_importances_
            importance_pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
            top_features = [(name, imp) for name, imp in importance_pairs if imp > 0.01][:10]

            if top_features:
                output_parts.append(html.Div([
                    html.Div("━━━ Feature Importance ━━━",
                            style={'fontWeight': 'bold', 'marginTop': '15px', 'marginBottom': '5px', 'color': '#1565c0'}),
                    html.Div([
                        html.Div(f"  {name}: {imp:.1%}", style={'fontSize': '11px'})
                        for name, imp in top_features
                    ])
                ]))

            return html.Div(output_parts)

        except Exception as e:
            import traceback
            return html.Div([
                html.Div(f"Error generating explanation: {str(e)}", style={'color': 'red'}),
                html.Pre(traceback.format_exc(), style={'fontSize': '10px', 'color': '#666'})
            ])

    def write_progress(operation, current, total, dataset):
        """Write progress to file for polling."""
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({
                    'active': True,
                    'operation': operation,
                    'current': current,
                    'total': total,
                    'dataset': dataset
                }, f)
        except:
            pass

    def clear_progress():
        """Clear progress file."""
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({'active': False, 'current': 0, 'total': 0, 'operation': '', 'dataset': ''}, f)
        except:
            pass

    @app.callback(
        [Output('recluster-all-result', 'children'),
         Output('pulse-features-per-dataset', 'data', allow_duplicate=True)],
        [Input('recluster-all-btn', 'n_clicks')],
        [State('pulse-features-checklist', 'value'),
         State('clustering-method-radio', 'value'),
         State('pulse-features-per-dataset', 'data')],
        prevent_initial_call=True
    )
    def recluster_all_datasets(n_clicks, selected_features, clustering_method, stored_features_data):
        """Run clustering on all datasets with the selected features."""
        if not n_clicks:
            raise PreventUpdate

        if not selected_features or len(selected_features) < 2:
            return (html.Div("Please select at least 2 features for clustering",
                          style={'color': 'orange', 'padding': '10px'}), no_update)

        # Get all datasets
        datasets = loader.datasets
        if not datasets:
            return (html.Div("No datasets available", style={'color': 'red', 'padding': '10px'}), no_update)

        features_str = ','.join(selected_features)
        method = clustering_method or 'dbscan'

        import subprocess
        import sys

        success_count = 0
        fail_count = 0
        results_details = []

        # Write initial progress
        write_progress('recluster', 0, len(datasets), 'Starting...')

        for i, dataset in enumerate(datasets):
            # Update progress
            write_progress('recluster', i + 1, len(datasets), dataset)

            try:
                data_path = loader.get_dataset_path(dataset)
                clean_prefix = loader.get_clean_prefix(dataset)

                if not data_path or not clean_prefix:
                    fail_count += 1
                    results_details.append(f"✗ {dataset}: Could not determine path")
                    continue

                # Check if features file exists and has all required features
                input_file = os.path.join(data_path, f"{clean_prefix}-features.csv")
                needs_extraction = False
                missing_features = []

                if not os.path.exists(input_file):
                    needs_extraction = True
                else:
                    # Check which selected features exist in the file
                    try:
                        with open(input_file, 'r') as f:
                            header = f.readline().strip().split(',')
                            available_features = set(header[1:])  # Skip waveform_index

                        missing_features = [f for f in selected_features if f not in available_features]
                        if missing_features:
                            needs_extraction = True
                    except Exception as e:
                        fail_count += 1
                        results_details.append(f"✗ {dataset}: Error reading features file: {str(e)[:30]}")
                        continue

                # Re-extract features if needed
                if needs_extraction:
                    write_progress('recluster', i + 1, len(datasets), f"{dataset} (extracting features...)")
                    try:
                        extract_cmd = [
                            sys.executable, 'extract_features.py',
                            '--input-dir', data_path,
                            '--file', clean_prefix
                        ]
                        extract_result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=300)

                        if extract_result.returncode != 0:
                            fail_count += 1
                            error_msg = extract_result.stderr[:100] if extract_result.stderr else "Unknown error"
                            results_details.append(f"✗ {dataset}: Feature extraction failed - {error_msg}")
                            continue

                        results_details.append(f"↻ {dataset}: Re-extracted features ({len(missing_features)} were missing)")
                    except subprocess.TimeoutExpired:
                        fail_count += 1
                        results_details.append(f"✗ {dataset}: Feature extraction timeout (>300s)")
                        continue
                    except Exception as e:
                        fail_count += 1
                        results_details.append(f"✗ {dataset}: Feature extraction error: {str(e)[:50]}")
                        continue

                # Run clustering
                cmd = [
                    sys.executable, 'cluster_pulses.py',
                    '--input-dir', data_path,
                    '--method', method,
                    '--features', features_str,
                    '--input', input_file
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    # Also run aggregation and classification
                    agg_cmd = [
                        sys.executable, 'aggregate_cluster_features.py',
                        '--input-dir', data_path,
                        '--method', method,
                        '--file', clean_prefix
                    ]
                    subprocess.run(agg_cmd, capture_output=True, text=True, timeout=60)

                    class_cmd = [
                        sys.executable, 'classify_pd_type.py',
                        '--input-dir', data_path,
                        '--method', method,
                        '--file', clean_prefix
                    ]
                    subprocess.run(class_cmd, capture_output=True, text=True, timeout=60)

                    success_count += 1
                    # Update re-extraction entry to show success, or add new success entry
                    updated = False
                    for j, d in enumerate(results_details):
                        if dataset in d and (d.startswith("↻") or d.startswith("⚠")):
                            results_details[j] = f"✓ {dataset}" + (" (re-extracted)" if d.startswith("↻") else "")
                            updated = True
                            break
                    if not updated:
                        results_details.append(f"✓ {dataset}")
                else:
                    fail_count += 1
                    error_hint = ""
                    if result.stderr:
                        if "No valid features" in result.stderr:
                            error_hint = " (no valid features found)"
                        elif "not found" in result.stderr.lower():
                            error_hint = " (features not in file)"
                    results_details.append(f"✗ {dataset}: Clustering failed{error_hint}")

            except subprocess.TimeoutExpired:
                fail_count += 1
                results_details.append(f"✗ {dataset}: Timeout (>120s)")
            except Exception as e:
                fail_count += 1
                results_details.append(f"✗ {dataset}: {str(e)[:50]}")

        # Clear progress when done
        clear_progress()

        # Update per-dataset feature storage for all datasets with the selected features
        # This ensures switching datasets will show the same features that were used for clustering
        updated_features_data = stored_features_data or {}
        for dataset in datasets:
            updated_features_data[dataset] = selected_features

        # Summary message
        if fail_count > 0:
            summary_msg = f"Reclustered {success_count}/{len(datasets)} datasets ({fail_count} failed)"
        else:
            summary_msg = f"Reclustered {success_count}/{len(datasets)} datasets"

        result_div = html.Div([
            html.Div(summary_msg,
                    style={'color': '#2e7d32' if fail_count == 0 else '#ff9800', 'fontWeight': 'bold'}),
            html.Div(f"Method: {method.upper()} | Features: {len(selected_features)}", style={'fontSize': '12px', 'color': '#666'}),
            html.Details([
                html.Summary("Details", style={'cursor': 'pointer', 'fontSize': '12px'}),
                html.Div([html.Div(d, style={'fontSize': '11px'}) for d in results_details],
                        style={'maxHeight': '150px', 'overflowY': 'auto', 'marginTop': '5px'})
            ]) if results_details else None,
            html.Div("Feature selection copied to all datasets.",
                    style={'fontSize': '12px', 'color': '#1976d2', 'marginTop': '5px', 'fontStyle': 'italic'})
        ], style={'padding': '10px', 'backgroundColor': '#e8f5e9', 'borderRadius': '4px'})

        return (result_div, updated_features_data)

    @app.callback(
        Output('reclassify-all-result', 'children'),
        [Input('reclassify-all-btn', 'n_clicks')],
        [State('cluster-features-checklist', 'value'),
         State('clustering-method-radio', 'value')],
        prevent_initial_call=True
    )
    def reclassify_all_datasets(n_clicks, selected_features, clustering_method):
        """Run classification on all datasets with the selected cluster features."""
        if not n_clicks:
            raise PreventUpdate

        if not selected_features or len(selected_features) < 1:
            return html.Div("Please select at least 1 feature for classification",
                          style={'color': 'orange', 'padding': '10px'})

        # Get all datasets
        datasets = loader.datasets
        if not datasets:
            return html.Div("No datasets available", style={'color': 'red', 'padding': '10px'})

        features_str = ','.join(selected_features)
        method = clustering_method or 'dbscan'

        import subprocess
        import sys

        success_count = 0
        fail_count = 0
        results_details = []

        # Write initial progress
        write_progress('reclassify', 0, len(datasets), 'Starting...')

        for i, dataset in enumerate(datasets):
            # Update progress
            write_progress('reclassify', i + 1, len(datasets), dataset)

            try:
                data_path = loader.get_dataset_path(dataset)
                clean_prefix = loader.get_clean_prefix(dataset)

                if not data_path or not clean_prefix:
                    fail_count += 1
                    results_details.append(f"✗ {dataset}: Could not determine path")
                    continue

                # Run classification with selected features
                cmd = [
                    sys.executable, 'classify_pd_type.py',
                    '--input-dir', data_path,
                    '--method', method,
                    '--file', clean_prefix,
                    '--cluster-features', features_str
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    success_count += 1
                    results_details.append(f"✓ {dataset}")
                else:
                    fail_count += 1
                    results_details.append(f"✗ {dataset}: Classification failed")

            except subprocess.TimeoutExpired:
                fail_count += 1
                results_details.append(f"✗ {dataset}: Timeout")
            except Exception as e:
                fail_count += 1
                results_details.append(f"✗ {dataset}: {str(e)[:50]}")

        # Clear progress when done
        clear_progress()

        return html.Div([
            html.Div(f"Reclassified {success_count}/{len(datasets)} datasets",
                    style={'color': '#2e7d32' if fail_count == 0 else '#ff9800', 'fontWeight': 'bold'}),
            html.Div(f"Method: {method.upper()} | Features: {len(selected_features)}", style={'fontSize': '12px', 'color': '#666'}),
            html.Details([
                html.Summary("Details", style={'cursor': 'pointer', 'fontSize': '12px'}),
                html.Div([html.Div(d, style={'fontSize': '11px'}) for d in results_details],
                        style={'maxHeight': '100px', 'overflowY': 'auto', 'marginTop': '5px'})
            ]) if results_details else None,
            html.Div("Refresh the page to see updated results.",
                    style={'fontSize': '12px', 'color': '#1976d2', 'marginTop': '5px', 'fontStyle': 'italic'})
        ], style={'padding': '10px', 'backgroundColor': '#f3e5f5', 'borderRadius': '4px'})

    # Enable interval when batch operation buttons are clicked
    @app.callback(
        Output('progress-interval', 'disabled'),
        [Input('recluster-all-btn', 'n_clicks'),
         Input('reclassify-all-btn', 'n_clicks'),
         Input('recluster-all-result', 'children'),
         Input('reclassify-all-result', 'children')],
        prevent_initial_call=True
    )
    def toggle_progress_interval(recluster_clicks, reclassify_clicks, recluster_result, reclassify_result):
        """Enable interval when operation starts, disable when result is received."""
        triggered = ctx.triggered_id
        if triggered in ['recluster-all-btn', 'reclassify-all-btn']:
            return False  # Enable interval
        else:
            return True  # Disable interval when result comes in

    # Poll progress file and update progress displays
    @app.callback(
        [Output('recluster-all-progress', 'children'),
         Output('reclassify-all-progress', 'children')],
        [Input('progress-interval', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_progress_display(n_intervals):
        """Read progress file and update progress spans."""
        recluster_progress = ''
        reclassify_progress = ''

        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, 'r') as f:
                    progress = json.load(f)

                if progress.get('active'):
                    operation = progress.get('operation', '')
                    current = progress.get('current', 0)
                    total = progress.get('total', 0)
                    dataset = progress.get('dataset', '')

                    progress_text = f"Processing {current}/{total}: {dataset}"

                    if operation == 'recluster':
                        recluster_progress = progress_text
                    elif operation == 'reclassify':
                        reclassify_progress = progress_text
        except:
            pass

        return recluster_progress, reclassify_progress

    @app.callback(
        [Output('all-datasets-analysis-display', 'children'),
         Output('all-datasets-analysis-display', 'style')],
        [Input('analyze-all-datasets-btn', 'n_clicks')],
        [State('n-features-input', 'value'),
         State('correlation-threshold-input', 'value'),
         State('pulse-features-checklist', 'value')],
        prevent_initial_call=True
    )
    def analyze_all_datasets(n_clicks, n_features, corr_threshold, current_features):
        """Analyze all datasets and display summary of recommended features."""
        hidden_style = {'display': 'none'}

        if not n_clicks:
            return [], hidden_style

        n_features = n_features or 5
        corr_threshold = corr_threshold or 0.85

        # Get all datasets
        datasets = loader.datasets
        if not datasets:
            return html.Div("No datasets available", style={'color': 'red'}), \
                   {'padding': '15px', 'backgroundColor': '#ffebee', 'borderRadius': '4px',
                    'marginBottom': '20px', 'display': 'block'}

        # Analyze each dataset
        results = []
        feature_counts = {}  # Track how often each feature is recommended

        for dataset in datasets:
            data = loader.load_all(dataset)
            if data['features'] is None or data['feature_names'] is None:
                results.append({
                    'dataset': dataset,
                    'error': 'No data',
                    'recommended': [],
                    'variance': 0
                })
                continue

            # Get subset of features based on current selection
            if current_features and len(current_features) > 0:
                indices = []
                names = []
                for feat in current_features:
                    if feat in data['feature_names']:
                        indices.append(data['feature_names'].index(feat))
                        names.append(feat)
                if len(indices) < 2:
                    results.append({
                        'dataset': dataset,
                        'error': 'Too few features',
                        'recommended': [],
                        'variance': 0
                    })
                    continue
                features_subset = data['features'][:, indices]
                feature_names = names
            else:
                features_subset = data['features']
                feature_names = list(data['feature_names'])

            # Compute recommendations
            result = compute_recommended_features(
                features_subset, feature_names,
                min(n_features, len(feature_names)), corr_threshold
            )

            if result.get('error'):
                results.append({
                    'dataset': dataset,
                    'error': result['error'],
                    'recommended': [],
                    'variance': 0
                })
            else:
                results.append({
                    'dataset': dataset,
                    'error': None,
                    'recommended': result['recommended'],
                    'variance': result['variance_explained']
                })
                # Count feature occurrences
                for feat in result['recommended']:
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1

        # Build summary table
        table_rows = [
            html.Tr([
                html.Th("Dataset", style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '2px solid #666', 'backgroundColor': '#e0e0e0'}),
                html.Th("Variance", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #666', 'backgroundColor': '#e0e0e0'}),
                html.Th("Recommended Features", style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '2px solid #666', 'backgroundColor': '#e0e0e0'})
            ])
        ]

        for r in results:
            if r['error']:
                table_rows.append(html.Tr([
                    html.Td(r['dataset'], style={'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                    html.Td('-', style={'padding': '8px', 'textAlign': 'center', 'borderBottom': '1px solid #ddd'}),
                    html.Td(f"Error: {r['error']}", style={'padding': '8px', 'color': 'red', 'borderBottom': '1px solid #ddd'})
                ]))
            else:
                variance_color = '#2e7d32' if r['variance'] >= 80 else ('#f57c00' if r['variance'] >= 60 else '#d32f2f')
                table_rows.append(html.Tr([
                    html.Td(r['dataset'], style={'padding': '8px', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold'}),
                    html.Td(f"{r['variance']:.1f}%", style={'padding': '8px', 'textAlign': 'center',
                                                            'borderBottom': '1px solid #ddd', 'color': variance_color,
                                                            'fontWeight': 'bold'}),
                    html.Td(", ".join(r['recommended']), style={'padding': '8px', 'borderBottom': '1px solid #ddd',
                                                                 'fontFamily': 'monospace', 'fontSize': '12px'})
                ]))

        # Build feature frequency summary table
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        total_datasets = len([r for r in results if not r['error']])

        # Create feature summary table
        feature_summary_rows = [
            html.Tr([
                html.Th("Feature", style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '2px solid #666', 'backgroundColor': '#e8f5e9'}),
                html.Th("Datasets", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #666', 'backgroundColor': '#e8f5e9'}),
                html.Th("Percentage", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #666', 'backgroundColor': '#e8f5e9'}),
                html.Th("Recommendation", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #666', 'backgroundColor': '#e8f5e9'})
            ])
        ]

        for feat, count in sorted_features:
            pct = (count / total_datasets * 100) if total_datasets > 0 else 0
            # Determine recommendation level
            if pct >= 80:
                rec_text = "Strongly Recommended"
                rec_color = '#2e7d32'
                rec_bg = '#e8f5e9'
            elif pct >= 50:
                rec_text = "Recommended"
                rec_color = '#1976d2'
                rec_bg = '#e3f2fd'
            elif pct >= 25:
                rec_text = "Consider"
                rec_color = '#f57c00'
                rec_bg = '#fff3e0'
            else:
                rec_text = "Dataset-Specific"
                rec_color = '#757575'
                rec_bg = '#fafafa'

            feature_summary_rows.append(html.Tr([
                html.Td(feat, style={'padding': '8px', 'borderBottom': '1px solid #ddd', 'fontFamily': 'monospace', 'fontWeight': 'bold'}),
                html.Td(f"{count} / {total_datasets}", style={'padding': '8px', 'textAlign': 'center', 'borderBottom': '1px solid #ddd'}),
                html.Td(f"{pct:.0f}%", style={'padding': '8px', 'textAlign': 'center', 'borderBottom': '1px solid #ddd',
                                               'fontWeight': 'bold', 'color': rec_color}),
                html.Td(rec_text, style={'padding': '8px', 'textAlign': 'center', 'borderBottom': '1px solid #ddd',
                                          'backgroundColor': rec_bg, 'color': rec_color, 'fontWeight': 'bold', 'fontSize': '12px'})
            ]))

        # Calculate average variance
        valid_variances = [r['variance'] for r in results if not r['error'] and r['variance'] > 0]
        avg_variance = sum(valid_variances) / len(valid_variances) if valid_variances else 0

        display_content = html.Div([
            html.H5(f"Analysis Across All Datasets ({len(datasets)} datasets)", style={'marginBottom': '15px'}),

            # Summary stats
            html.Div([
                html.Span("Average Variance Coverage: ", style={'fontWeight': 'bold'}),
                html.Span(f"{avg_variance:.1f}%", style={'color': '#1976d2', 'fontWeight': 'bold', 'fontSize': '16px'}),
                html.Span(f" | Datasets analyzed: {total_datasets}/{len(datasets)}", style={'marginLeft': '20px', 'color': '#666'})
            ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': '#e3f2fd', 'borderRadius': '4px'}),

            # Feature frequency summary table
            html.Div([
                html.H6("Feature Recommendation Summary", style={'marginBottom': '10px'}),
                html.P("Features ranked by how often they were selected across all datasets:",
                       style={'color': '#666', 'fontSize': '12px', 'marginBottom': '10px'}),
                html.Table(feature_summary_rows, style={'width': '100%', 'borderCollapse': 'collapse'})
            ], style={'marginBottom': '15px', 'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'border': '1px solid #ddd'}),

            # Per-dataset table
            html.Details([
                html.Summary("Per-Dataset Details", style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.Table(table_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '10px'})
            ], open=False)
        ])

        show_style = {'padding': '15px', 'backgroundColor': '#f0f0ff', 'borderRadius': '4px',
                      'marginBottom': '20px', 'display': 'block'}

        return display_content, show_style

    @app.callback(
        [Output('reanalysis-status', 'children'),
         Output('reanalysis-status', 'style'),
         Output('reanalysis-trigger', 'data')],
        [Input('reanalyze-button', 'n_clicks')],
        [State('dataset-dropdown', 'value'),
         State('polarity-method-dropdown', 'value'),
         State('clustering-method-radio', 'value'),
         State('dbscan-min-samples', 'value'),
         State('kmeans-n-clusters', 'value'),
         State('pulse-features-checklist', 'value'),
         State('cluster-features-checklist', 'value'),
         State('reanalysis-trigger', 'data')],
        prevent_initial_call=True
    )
    def run_reanalysis(n_clicks, prefix, polarity_method, clustering_method,
                       dbscan_min_samples, kmeans_n_clusters,
                       pulse_features, cluster_features, current_trigger):
        """Run the full analysis pipeline with the selected options."""
        if not n_clicks or not prefix:
            return "", {'display': 'none'}, current_trigger

        if polarity_method == 'stored' or polarity_method is None:
            return html.Div([
                "Please select a polarity method (not 'Stored') to re-analyze"
            ], style={'color': '#856404', 'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '4px'}), \
                {'display': 'block', 'width': '90%', 'margin': '5px auto'}, current_trigger

        if not pulse_features:
            return html.Div([
                "Please select at least one pulse feature for clustering"
            ], style={'color': '#856404', 'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '4px'}), \
                {'display': 'block', 'width': '90%', 'margin': '5px auto'}, current_trigger

        if not cluster_features:
            return html.Div([
                "Please select at least one cluster feature for classification"
            ], style={'color': '#856404', 'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '4px'}), \
                {'display': 'block', 'width': '90%', 'margin': '5px auto'}, current_trigger

        try:
            # Build command with all selected options
            cmd = [
                sys.executable, 'run_analysis_pipeline.py',
                '--input-dir', data_dir,
                '--polarity-method', polarity_method,
                '--clustering-method', clustering_method,
                '--file', prefix
            ]

            # Add clustering-specific parameters
            if clustering_method == 'dbscan':
                cmd.extend(['--min-samples', str(dbscan_min_samples or 5)])
            else:
                cmd.extend(['--n-clusters', str(kmeans_n_clusters or 5)])

            # Add selected pulse features for clustering
            if pulse_features:
                cmd.extend(['--pulse-features', ','.join(pulse_features)])

            # Add selected cluster features for classification
            if cluster_features:
                cmd.extend(['--cluster-features', ','.join(cluster_features)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Increment trigger to force data reload
                feature_info = f"Pulse features: {len(pulse_features)}, Cluster features: {len(cluster_features)}"
                return html.Div([
                    f"Re-analysis complete! ",
                    f"Polarity: {polarity_method}, Clustering: {clustering_method.upper()}. ",
                    f"{feature_info}. ",
                    "Data has been updated - select 'Stored' to see new results."
                ], style={'color': '#155724', 'backgroundColor': '#d4edda', 'padding': '10px', 'borderRadius': '4px'}), \
                    {'display': 'block', 'width': '90%', 'margin': '5px auto'}, current_trigger + 1
            else:
                # Show more of the error message
                error_msg = result.stderr if result.stderr else result.stdout if result.stdout else 'Unknown error'
                # Get last 500 chars which usually contain the actual error
                if len(error_msg) > 500:
                    error_msg = "..." + error_msg[-500:]
                return html.Div([
                    html.Div("❌ Re-analysis failed:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    html.Pre(error_msg, style={'whiteSpace': 'pre-wrap', 'fontSize': '11px', 'margin': '0'})
                ], style={'color': '#721c24', 'backgroundColor': '#f8d7da', 'padding': '10px', 'borderRadius': '4px'}), \
                    {'display': 'block', 'width': '90%', 'margin': '5px auto'}, current_trigger

        except subprocess.TimeoutExpired:
            return html.Div([
                "❌ Re-analysis timed out (exceeded 5 minutes)"
            ], style={'color': '#721c24', 'backgroundColor': '#f8d7da', 'padding': '10px', 'borderRadius': '4px'}), \
                {'display': 'block', 'width': '90%', 'margin': '5px auto'}, current_trigger
        except Exception as e:
            return html.Div([
                f"❌ Error running re-analysis: {str(e)}"
            ], style={'color': '#721c24', 'backgroundColor': '#f8d7da', 'padding': '10px', 'borderRadius': '4px'}), \
                {'display': 'block', 'width': '90%', 'margin': '5px auto'}, current_trigger

    @app.callback(
        [Output('cluster-prpd', 'figure'),
         Output('pdtype-prpd', 'figure'),
         Output('histogram', 'figure'),
         Output('stats-text', 'children'),
         Output('current-data-store', 'data')],
        [Input('dataset-dropdown', 'value'),
         Input('polarity-method-dropdown', 'value'),
         Input('reanalysis-trigger', 'data')]
    )
    def update_dataset(prefix, polarity_method, reanalysis_trigger):
        if not prefix:
            empty_fig = go.Figure()
            return empty_fig, empty_fig, empty_fig, "No dataset selected", None

        # Check if this is an unprocessed dataset
        dataset_info = loader.dataset_info.get(prefix, {})
        has_features = dataset_info.get('has_features', True)

        if not has_features:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Dataset not processed yet",
                xref="paper", yref="paper",
                x=0.5, y=0.6, showarrow=False,
                font=dict(size=20, color="orange")
            )
            empty_fig.add_annotation(
                text="Run the analysis pipeline to extract features:",
                xref="paper", yref="paper",
                x=0.5, y=0.45, showarrow=False,
                font=dict(size=14, color="gray")
            )
            clean_prefix = loader.get_clean_prefix(prefix)
            data_path = dataset_info.get('path', 'Rugged Data Files')
            cmd = f"python run_analysis_pipeline.py --data-dir \"{data_path}\" --prefix \"{clean_prefix}\""
            empty_fig.add_annotation(
                text=cmd,
                xref="paper", yref="paper",
                x=0.5, y=0.35, showarrow=False,
                font=dict(size=11, color="blue", family="monospace")
            )
            empty_fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )

            fmt = dataset_info.get('format', 'unknown')
            stats_msg = f"Dataset: {prefix}\nFormat: {fmt}\nStatus: Not processed\n\nRun the analysis pipeline to extract features and generate PRPD plots."

            return empty_fig, empty_fig, empty_fig, stats_msg, prefix

        data = loader.load_all(prefix)

        # Determine polarity method to use
        pm = polarity_method if polarity_method and polarity_method != 'stored' else None
        sample_interval = data.get('sample_interval', 4e-9)

        cluster_fig = create_prpd_plot(
            data['features'], data['feature_names'],
            data['cluster_labels'], data['pd_types'],
            color_by='cluster',
            waveforms=data['waveforms'],
            polarity_method=pm,
            sample_interval=sample_interval
        )

        pdtype_fig = create_prpd_plot(
            data['features'], data['feature_names'],
            data['cluster_labels'], data['pd_types'],
            color_by='pdtype',
            waveforms=data['waveforms'],
            polarity_method=pm,
            sample_interval=sample_interval
        )

        hist_fig = create_histogram(
            data['features'], data['feature_names'],
            data['cluster_labels'], data['pd_types']
        )

        stats = create_stats_text(
            data['features'], data['cluster_labels'], data['pd_types']
        )

        return cluster_fig, pdtype_fig, hist_fig, stats, prefix

    @app.callback(
        [Output('pca-plot', 'figure'),
         Output('pca-container', 'style')],
        [Input('clustering-method-radio', 'value'),
         Input('dataset-dropdown', 'value'),
         Input('reanalysis-trigger', 'data')]
    )
    def update_pca_plot(clustering_method, prefix, reanalysis_trigger):
        """Update PCA plot - only shown for K-means clustering."""
        # Only show PCA for K-means
        if clustering_method != 'kmeans':
            empty_fig = go.Figure()
            return empty_fig, {'display': 'none'}

        if not prefix:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="PCA Plot - No data")
            return empty_fig, {'display': 'none'}

        # Load data
        data = loader.load_all(prefix)

        # Create PCA plot
        pca_fig = create_pca_plot(
            data['features'], data['feature_names'],
            data['cluster_labels']
        )

        return pca_fig, {'width': '95%', 'margin': '20px auto', 'display': 'block'}

    @app.callback(
        [Output('correlation-matrix', 'figure'),
         Output('correlation-matrix-container', 'style')],
        [Input('show-correlation-matrix-checkbox', 'value'),
         Input('dataset-dropdown', 'value'),
         Input('pulse-features-checklist', 'value'),
         Input('reanalysis-trigger', 'data')]
    )
    def update_correlation_matrix(show_corr, prefix, selected_features, reanalysis_trigger):
        """Update correlation matrix based on checkbox and selected features."""
        hidden_style = {'display': 'none', 'width': '49%', 'verticalAlign': 'top'}
        visible_style = {'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'}

        # Check if enabled
        if not show_corr or 'show' not in show_corr:
            empty_fig = go.Figure()
            return empty_fig, hidden_style

        if not prefix:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Correlation Matrix - No data")
            return empty_fig, hidden_style

        # Load data
        data = loader.load_all(prefix)

        # Create correlation matrix with selected features
        corr_fig = create_correlation_matrix(
            data['features'],
            data['feature_names'],
            selected_features
        )

        return corr_fig, visible_style

    @app.callback(
        [Output('pca-loadings', 'figure'),
         Output('pca-loadings-container', 'style')],
        [Input('show-pca-loadings-checkbox', 'value'),
         Input('dataset-dropdown', 'value'),
         Input('pulse-features-checklist', 'value'),
         Input('reanalysis-trigger', 'data')]
    )
    def update_pca_loadings(show_loadings, prefix, selected_features, reanalysis_trigger):
        """Update PCA loadings table based on checkbox and selected features."""
        hidden_style = {'display': 'none', 'width': '49%', 'verticalAlign': 'top'}
        visible_style = {'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'}

        # Check if enabled
        if not show_loadings or 'show' not in show_loadings:
            empty_fig = go.Figure()
            return empty_fig, hidden_style

        if not prefix:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="PCA Loadings - No data")
            return empty_fig, hidden_style

        # Load data
        data = loader.load_all(prefix)

        # Create PCA loadings with selected features
        loadings_fig = create_pca_loadings(
            data['features'],
            data['feature_names'],
            selected_features
        )

        return loadings_fig, visible_style

    @app.callback(
        [Output('waveform-plot', 'figure'),
         Output('selected-waveform-idx', 'data')],
        [Input('cluster-prpd', 'clickData'),
         Input('pdtype-prpd', 'clickData')],
        [State('current-data-store', 'data')],
        prevent_initial_call=True
    )
    def update_waveform(cluster_click, pdtype_click, prefix):
        # Use callback context to determine which input triggered
        idx = None

        # Check which input triggered the callback
        triggered_id = ctx.triggered_id
        if triggered_id == 'cluster-prpd' and cluster_click:
            click_data = cluster_click
        elif triggered_id == 'pdtype-prpd' and pdtype_click:
            click_data = pdtype_click
        else:
            click_data = cluster_click or pdtype_click

        if click_data and 'points' in click_data and len(click_data['points']) > 0:
            point = click_data['points'][0]
            if 'customdata' in point:
                idx = int(point['customdata'])
            elif 'pointIndex' in point:
                # Fallback to pointIndex if customdata not available
                idx = int(point['pointIndex'])

        if prefix and idx is not None:
            try:
                # Reload data for the waveform
                data = loader.load_all(prefix)
                waveforms = data['waveforms']

                # Debug: check waveform data
                if waveforms is None:
                    print(f"DEBUG: Waveforms is None for {prefix}")
                elif idx >= len(waveforms):
                    print(f"DEBUG: Index {idx} >= waveform count {len(waveforms)}")

                return create_waveform_plot(
                    waveforms, idx,
                    data['features'], data['feature_names'],
                    data['cluster_labels'], data['pd_types']
                ), idx
            except Exception as e:
                print(f"ERROR in update_waveform: {e}")
                import traceback
                traceback.print_exc()
                return create_waveform_plot(None, None, None, None, None, None), None
        elif prefix:
            # No point clicked yet, show placeholder
            data = loader.load_all(prefix)
            return create_waveform_plot(
                data['waveforms'], None,
                data['features'], data['feature_names'],
                data['cluster_labels'], data['pd_types']
            ), None

        return create_waveform_plot(None, None, None, None, None, None), None

    # Define cluster feature lists for the selector
    from aggregate_cluster_features import (
        CLUSTER_FEATURE_NAMES as AGG_CLUSTER_FEATURE_NAMES,
        WAVEFORM_MEAN_FEATURE_NAMES,
        WAVEFORM_TRIMMED_MEAN_FEATURE_NAMES,
        ALL_CLUSTER_FEATURE_NAMES
    )

    @app.callback(
        Output('cluster-feature-selector', 'options'),
        [Input('current-data-store', 'data')],
        prevent_initial_call=True
    )
    def populate_cluster_feature_options(prefix):
        """Populate the cluster feature selector with all available features."""
        options = []

        # Group 1: PRPD-based features
        for feat in AGG_CLUSTER_FEATURE_NAMES:
            options.append({'label': f'[PRPD] {feat}', 'value': feat})

        # Group 2: Mean features
        for feat in WAVEFORM_MEAN_FEATURE_NAMES:
            display_name = feat.replace('mean_', '')
            options.append({'label': f'[Mean] {display_name}', 'value': feat})

        # Group 3: Trimmed mean features
        for feat in WAVEFORM_TRIMMED_MEAN_FEATURE_NAMES:
            display_name = feat.replace('trimmed_mean_', '')
            options.append({'label': f'[TrimMean] {display_name}', 'value': feat})

        return options

    @app.callback(
        Output('cluster-feature-selector', 'value'),
        [Input('cluster-feat-select-mean', 'n_clicks'),
         Input('cluster-feat-select-trimmed', 'n_clicks'),
         Input('cluster-feat-select-prpd', 'n_clicks'),
         Input('cluster-feat-select-none', 'n_clicks')],
        [State('cluster-feature-selector', 'value')],
        prevent_initial_call=True
    )
    def update_cluster_feature_selection(mean_clicks, trimmed_clicks, prpd_clicks, none_clicks, current_value):
        """Handle cluster feature selection buttons."""
        triggered = ctx.triggered_id

        if triggered == 'cluster-feat-select-mean':
            return WAVEFORM_MEAN_FEATURE_NAMES
        elif triggered == 'cluster-feat-select-trimmed':
            return WAVEFORM_TRIMMED_MEAN_FEATURE_NAMES
        elif triggered == 'cluster-feat-select-prpd':
            return list(AGG_CLUSTER_FEATURE_NAMES)
        elif triggered == 'cluster-feat-select-none':
            return []

        return current_value or []

    @app.callback(
        [Output('cluster-details-display', 'children'),
         Output('cluster-details-display', 'style')],
        [Input('cluster-prpd', 'clickData'),
         Input('pdtype-prpd', 'clickData')],
        [State('show-cluster-details-checkbox', 'value'),
         State('current-data-store', 'data'),
         State('clustering-method-radio', 'value'),
         State('cluster-feature-selector', 'value')],
        prevent_initial_call=True
    )
    def update_cluster_details(cluster_click, pdtype_click, show_details, prefix, clustering_method, selected_cluster_features):
        """Show cluster statistics and decision tree details when clicking on PRPD."""
        # Check if feature is enabled
        if not show_details or 'show' not in show_details:
            return "", {'display': 'none'}

        if not prefix:
            return "", {'display': 'none'}

        # Determine which click triggered the callback
        triggered_id = ctx.triggered_id
        if triggered_id == 'cluster-prpd' and cluster_click:
            click_data = cluster_click
        elif triggered_id == 'pdtype-prpd' and pdtype_click:
            click_data = pdtype_click
        else:
            click_data = cluster_click or pdtype_click

        if not click_data or 'points' not in click_data or len(click_data['points']) == 0:
            return "", {'display': 'none'}

        # Get the clicked point index
        point = click_data['points'][0]
        if 'customdata' in point:
            idx = int(point['customdata'])
        elif 'pointIndex' in point:
            idx = int(point['pointIndex'])
        else:
            return "", {'display': 'none'}

        try:
            # Load data to get cluster label for this point
            data = loader.load_all(prefix)
            cluster_labels = data['cluster_labels']
            cluster_label = cluster_labels[idx]

            # Load cluster features
            method = clustering_method or 'dbscan'
            cluster_features_file = os.path.join(data_dir, f"{prefix}-cluster-features-{method}.csv")

            if not os.path.exists(cluster_features_file):
                return html.Div([
                    html.P(f"Cluster features file not found for method '{method}'",
                          style={'color': '#721c24'})
                ]), {'display': 'block', 'padding': '15px', 'backgroundColor': '#f8d7da',
                     'border': '1px solid #f5c6cb', 'borderRadius': '5px'}

            # Load cluster features
            all_cluster_features = load_cluster_features(cluster_features_file)

            if cluster_label not in all_cluster_features:
                return html.Div([
                    html.P(f"No features found for cluster {cluster_label}",
                          style={'color': '#721c24'})
                ]), {'display': 'block', 'padding': '15px', 'backgroundColor': '#f8d7da',
                     'border': '1px solid #f5c6cb', 'borderRadius': '5px'}

            cluster_feats = all_cluster_features[cluster_label]

            # Run classification to get decision tree details
            classifier = PDTypeClassifier(verbose=False)
            result = classifier.classify(cluster_feats, cluster_label)

            # Build display content
            label_str = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}"
            n_pulses = int(cluster_feats.get('pulses_per_positive_halfcycle', 0) +
                         cluster_feats.get('pulses_per_negative_halfcycle', 0))

            content = []

            # Header
            content.append(html.H4(f"{label_str} Details", style={'marginTop': '0', 'color': '#333'}))

            # Classification result
            content.append(html.Div([
                html.Span("Classification: ", style={'fontWeight': 'bold'}),
                html.Span(f"{result['pd_type']} ", style={'color': PD_TYPE_COLORS.get(result['pd_type'], '#000'),
                                                          'fontWeight': 'bold', 'fontSize': '16px'}),
                html.Span(f"({result['confidence']:.1%} confidence)", style={'color': '#666'})
            ], style={'marginBottom': '15px', 'fontSize': '14px'}))

            # Pulse count
            content.append(html.P(f"Total pulses: {n_pulses}", style={'margin': '5px 0'}))

            # Decision tree branch path
            content.append(html.H5("Decision Tree Path:", style={'marginTop': '15px', 'marginBottom': '10px'}))
            branch_list = []
            for branch in result['branch_path']:
                branch_list.append(html.Li(branch, style={'marginBottom': '3px'}))
            content.append(html.Ol(branch_list, style={'marginLeft': '20px', 'marginTop': '0'}))

            # Reasoning
            content.append(html.H5("Feature Analysis:", style={'marginTop': '15px', 'marginBottom': '10px'}))
            reasoning_list = []
            for reason in result['reasoning']:
                reasoning_list.append(html.Li(reason, style={'marginBottom': '3px', 'fontSize': '12px'}))
            content.append(html.Ul(reasoning_list, style={'marginLeft': '20px', 'marginTop': '0'}))

            # Feature Values vs Thresholds Table
            content.append(html.H5("Feature Values vs Decision Tree Thresholds:",
                                   style={'marginTop': '15px', 'marginBottom': '10px'}))

            # Build table rows with feature values and thresholds
            table_header = html.Tr([
                html.Th("Feature", style={'padding': '6px', 'borderBottom': '2px solid #333', 'textAlign': 'left'}),
                html.Th("Value", style={'padding': '6px', 'borderBottom': '2px solid #333', 'textAlign': 'right'}),
                html.Th("Threshold", style={'padding': '6px', 'borderBottom': '2px solid #333', 'textAlign': 'center'}),
                html.Th("Status", style={'padding': '6px', 'borderBottom': '2px solid #333', 'textAlign': 'center'}),
            ])

            table_rows = [table_header]

            # Helper to create a row
            def make_row(feature_name, value, threshold_str, passes, unit=''):
                status_color = '#28a745' if passes else '#dc3545'
                status_text = 'PASS' if passes else 'FAIL'
                val_str = f"{value:.4f}" if isinstance(value, float) and abs(value) < 100 else f"{value:.1f}"
                return html.Tr([
                    html.Td(feature_name, style={'padding': '4px', 'fontSize': '11px', 'borderBottom': '1px solid #ddd'}),
                    html.Td(f"{val_str}{unit}", style={'padding': '4px', 'fontSize': '11px', 'textAlign': 'right',
                                                       'fontFamily': 'monospace', 'borderBottom': '1px solid #ddd'}),
                    html.Td(threshold_str, style={'padding': '4px', 'fontSize': '11px', 'textAlign': 'center',
                                                  'fontFamily': 'monospace', 'borderBottom': '1px solid #ddd'}),
                    html.Td(status_text, style={'padding': '4px', 'fontSize': '11px', 'textAlign': 'center',
                                                'color': status_color, 'fontWeight': 'bold', 'borderBottom': '1px solid #ddd'}),
                ])

            # Section header helper
            def section_header(text):
                return html.Tr([
                    html.Td(text, colSpan=4, style={'padding': '8px 4px 4px 4px', 'fontSize': '12px',
                                                    'fontWeight': 'bold', 'backgroundColor': '#e9ecef',
                                                    'borderBottom': '1px solid #ccc'})
                ])

            # Branch 1: Noise Detection
            table_rows.append(section_header("Branch 1: Noise Detection"))

            cv = cluster_feats.get('coefficient_of_variation', 0)
            cv_thresh = NOISE_THRESHOLDS['max_coefficient_of_variation']
            table_rows.append(make_row('Coefficient of Variation', cv, f'< {cv_thresh}', cv <= cv_thresh))

            # Branch 2: Phase Correlation
            table_rows.append(section_header("Branch 2: Phase Correlation"))

            cross_corr = cluster_feats.get('cross_correlation', 0)
            cc_thresh = PHASE_CORRELATION_THRESHOLDS['min_cross_correlation_symmetric']
            table_rows.append(make_row('Cross Correlation', cross_corr, f'>= {cc_thresh} (symmetric)', cross_corr >= cc_thresh))

            asymmetry = cluster_feats.get('discharge_asymmetry', 0)
            asym_thresh = PHASE_CORRELATION_THRESHOLDS['max_asymmetry_symmetric']
            table_rows.append(make_row('Discharge Asymmetry', abs(asymmetry), f'< {asym_thresh} (symmetric)', abs(asymmetry) < asym_thresh))

            phase_spread = cluster_feats.get('phase_spread', 0)
            ps_thresh = PHASE_CORRELATION_THRESHOLDS['max_phase_spread_corona']
            table_rows.append(make_row('Phase Spread', phase_spread, f'< {ps_thresh} (corona)', phase_spread < ps_thresh, ' deg'))

            # Branch 3: Symmetry & Quadrants
            table_rows.append(section_header("Branch 3: Quadrant Distribution"))

            q1 = cluster_feats.get('quadrant_1_percentage', 0)
            q2 = cluster_feats.get('quadrant_2_percentage', 0)
            q3 = cluster_feats.get('quadrant_3_percentage', 0)
            q4 = cluster_feats.get('quadrant_4_percentage', 0)
            positive_half = q1 + q2
            negative_half = q3 + q4

            single_half_thresh = QUADRANT_THRESHOLDS['single_halfcycle_threshold']
            table_rows.append(make_row('Q1 (0-90 deg)', q1, f'{QUADRANT_THRESHOLDS["symmetric_quadrant_min"]}-{QUADRANT_THRESHOLDS["symmetric_quadrant_max"]}%',
                                       QUADRANT_THRESHOLDS["symmetric_quadrant_min"] <= q1 <= QUADRANT_THRESHOLDS["symmetric_quadrant_max"], '%'))
            table_rows.append(make_row('Q2 (90-180 deg)', q2, f'{QUADRANT_THRESHOLDS["symmetric_quadrant_min"]}-{QUADRANT_THRESHOLDS["symmetric_quadrant_max"]}%',
                                       QUADRANT_THRESHOLDS["symmetric_quadrant_min"] <= q2 <= QUADRANT_THRESHOLDS["symmetric_quadrant_max"], '%'))
            table_rows.append(make_row('Q3 (180-270 deg)', q3, f'{QUADRANT_THRESHOLDS["symmetric_quadrant_min"]}-{QUADRANT_THRESHOLDS["symmetric_quadrant_max"]}%',
                                       QUADRANT_THRESHOLDS["symmetric_quadrant_min"] <= q3 <= QUADRANT_THRESHOLDS["symmetric_quadrant_max"], '%'))
            table_rows.append(make_row('Q4 (270-360 deg)', q4, f'{QUADRANT_THRESHOLDS["symmetric_quadrant_min"]}-{QUADRANT_THRESHOLDS["symmetric_quadrant_max"]}%',
                                       QUADRANT_THRESHOLDS["symmetric_quadrant_min"] <= q4 <= QUADRANT_THRESHOLDS["symmetric_quadrant_max"], '%'))
            table_rows.append(make_row('Positive Half (Q1+Q2)', positive_half, f'< {single_half_thresh}% (not corona)', positive_half < single_half_thresh, '%'))
            table_rows.append(make_row('Negative Half (Q3+Q4)', negative_half, f'< {single_half_thresh}% (not corona)', negative_half < single_half_thresh, '%'))

            # Branch 4: Phase Location
            table_rows.append(section_header("Branch 4: Phase Location"))

            phase_max = cluster_feats.get('phase_of_max_activity', 0)
            surface_tol = SYMMETRY_THRESHOLDS['surface_phase_tolerance']
            near_zero = phase_max < surface_tol or abs(phase_max - 180) < surface_tol or phase_max > (360 - surface_tol)
            table_rows.append(make_row('Phase of Max Activity', phase_max, f'Near 0/180/360 deg +/-{surface_tol} (surface)', near_zero, ' deg'))

            inception = cluster_feats.get('inception_phase', 0)
            extinction = cluster_feats.get('extinction_phase', 0)
            table_rows.append(make_row('Inception Phase', inception, 'Info only', True, ' deg'))
            table_rows.append(make_row('Extinction Phase', extinction, 'Info only', True, ' deg'))

            # Branch 5: Amplitude Characteristics
            table_rows.append(section_header("Branch 5: Amplitude Characteristics"))

            weibull_beta = cluster_feats.get('weibull_beta', 0)
            wb_min = AMPLITUDE_THRESHOLDS['internal_weibull_beta_min']
            wb_max = AMPLITUDE_THRESHOLDS['internal_weibull_beta_max']
            table_rows.append(make_row('Weibull Beta', weibull_beta, f'{wb_min}-{wb_max} (internal)', wb_min <= weibull_beta <= wb_max))

            mean_amp_pos = cluster_feats.get('mean_amplitude_positive', 0)
            mean_amp_neg = cluster_feats.get('mean_amplitude_negative', 0)
            max_amp_pos = cluster_feats.get('max_amplitude_positive', 0)
            max_amp_neg = cluster_feats.get('max_amplitude_negative', 0)
            mean_amp = max(mean_amp_pos, abs(mean_amp_neg)) if max(mean_amp_pos, abs(mean_amp_neg)) > 0 else 1e-10
            max_amp = max(max_amp_pos, abs(max_amp_neg))
            amp_ratio = max_amp / mean_amp if mean_amp > 0 else 0
            ar_thresh = AMPLITUDE_THRESHOLDS['corona_amplitude_ratio_threshold']
            table_rows.append(make_row('Amplitude Ratio (max/mean)', amp_ratio, f'> {ar_thresh} (corona)', amp_ratio > ar_thresh))

            # Create the table
            feature_table = html.Table(
                table_rows,
                style={
                    'width': '100%',
                    'borderCollapse': 'collapse',
                    'fontSize': '11px',
                    'backgroundColor': '#fff'
                }
            )
            content.append(html.Div(feature_table, style={'maxHeight': '300px', 'overflowY': 'auto', 'border': '1px solid #ddd', 'borderRadius': '4px'}))

            # Warnings
            if result['warnings']:
                content.append(html.H5("Warnings:", style={'marginTop': '15px', 'marginBottom': '10px', 'color': '#856404'}))
                warning_list = []
                for warning in result['warnings']:
                    warning_list.append(html.Li(warning, style={'fontSize': '12px', 'color': '#856404'}))
                content.append(html.Ul(warning_list, style={'marginLeft': '20px', 'marginTop': '0'}))

            # Selected Cluster Features Table
            if selected_cluster_features and len(selected_cluster_features) > 0:
                content.append(html.H5("Selected Cluster Features:",
                                       style={'marginTop': '20px', 'marginBottom': '10px', 'color': '#1565c0'}))

                # Build feature table
                feat_header = html.Tr([
                    html.Th("Feature", style={'padding': '6px', 'borderBottom': '2px solid #333', 'textAlign': 'left', 'backgroundColor': '#e3f2fd'}),
                    html.Th("Value", style={'padding': '6px', 'borderBottom': '2px solid #333', 'textAlign': 'right', 'backgroundColor': '#e3f2fd'}),
                ])

                feat_rows = [feat_header]

                # Group features by type
                mean_feats = [f for f in selected_cluster_features if f.startswith('mean_') and not f.startswith('mean_amplitude')]
                trimmed_feats = [f for f in selected_cluster_features if f.startswith('trimmed_mean_')]
                prpd_feats = [f for f in selected_cluster_features if f in AGG_CLUSTER_FEATURE_NAMES]
                # Also catch mean_amplitude_positive/negative which are PRPD features
                prpd_feats += [f for f in selected_cluster_features if f.startswith('mean_amplitude')]

                def format_value(val):
                    """Format a feature value for display."""
                    if val is None:
                        return "N/A"
                    if isinstance(val, float):
                        if abs(val) < 0.001 or abs(val) > 10000:
                            return f"{val:.4e}"
                        elif abs(val) < 1:
                            return f"{val:.6f}"
                        else:
                            return f"{val:.4f}"
                    return str(val)

                def add_section(title, features, color):
                    if features:
                        feat_rows.append(html.Tr([
                            html.Td(title, colSpan=2, style={
                                'padding': '6px 4px', 'fontSize': '11px', 'fontWeight': 'bold',
                                'backgroundColor': color, 'borderBottom': '1px solid #ccc'
                            })
                        ]))
                        for feat in sorted(features):
                            val = cluster_feats.get(feat, 0.0)
                            display_name = feat
                            if feat.startswith('mean_'):
                                display_name = feat.replace('mean_', '')
                            elif feat.startswith('trimmed_mean_'):
                                display_name = feat.replace('trimmed_mean_', '')

                            feat_rows.append(html.Tr([
                                html.Td(display_name, style={'padding': '4px', 'fontSize': '11px', 'borderBottom': '1px solid #eee'}),
                                html.Td(format_value(val), style={'padding': '4px', 'fontSize': '11px', 'textAlign': 'right',
                                                                   'fontFamily': 'monospace', 'borderBottom': '1px solid #eee'}),
                            ]))

                add_section("PRPD Features", prpd_feats, '#fff3e0')
                add_section("Mean Waveform Features", mean_feats, '#e8f5e9')
                add_section("Trimmed Mean Waveform Features", trimmed_feats, '#e3f2fd')

                selected_feat_table = html.Table(
                    feat_rows,
                    style={
                        'width': '100%',
                        'borderCollapse': 'collapse',
                        'fontSize': '11px',
                        'backgroundColor': '#fff'
                    }
                )
                content.append(html.Div(selected_feat_table, style={
                    'maxHeight': '250px', 'overflowY': 'auto',
                    'border': '1px solid #ddd', 'borderRadius': '4px'
                }))

            return html.Div(content), {
                'display': 'block',
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'border': '1px solid #dee2e6',
                'borderRadius': '5px',
                'maxHeight': '400px',
                'overflowY': 'auto'
            }

        except Exception as e:
            return html.Div([
                html.P(f"Error loading cluster details: {str(e)}", style={'color': '#721c24'})
            ]), {'display': 'block', 'padding': '15px', 'backgroundColor': '#f8d7da',
                 'border': '1px solid #f5c6cb', 'borderRadius': '5px'}

    @app.callback(
        Output('feature-toggles', 'options'),
        [Input('current-data-store', 'data')]
    )
    def update_feature_options(prefix):
        if not prefix:
            return []
        data = loader.load_all(prefix)
        if data['feature_names'] is None:
            return []
        # Create options for all features
        return [{'label': name.replace('_', ' ').title(), 'value': name}
                for name in data['feature_names']]

    @app.callback(
        Output('feature-display', 'children'),
        [Input('selected-waveform-idx', 'data'),
         Input('feature-toggles', 'value'),
         Input('polarity-method-dropdown', 'value')],
        [State('current-data-store', 'data')]
    )
    def update_feature_display(idx, visible_features, polarity_method, prefix):
        if idx is None or not prefix or not visible_features:
            return html.Div("Select a waveform to view features", style={'color': '#888', 'fontStyle': 'italic'})

        data = loader.load_all(prefix)
        if data['features'] is None or data['feature_names'] is None:
            return html.Div("No feature data available", style={'color': '#888'})

        features = data['features']
        feature_names = data['feature_names']
        waveforms = data['waveforms']
        sample_interval = data.get('sample_interval', 4e-9)

        if idx >= len(features):
            return html.Div("Invalid waveform index", style={'color': '#888'})

        # Get cluster and PD type info
        cluster_info = ""
        if data['cluster_labels'] is not None:
            cluster = data['cluster_labels'][idx]
            cluster_info = f"Cluster: {cluster}"
            if data['pd_types'] is not None:
                pd_type = data['pd_types'].get(cluster, 'UNKNOWN')
                cluster_info += f" | Type: {pd_type}"

        # Build feature display
        feature_items = []

        # Add cluster/type info at top
        if cluster_info:
            feature_items.append(
                html.Div(cluster_info, style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#333'})
            )

        # Add polarity comparison if waveform is available
        if waveforms is not None and idx < len(waveforms):
            wfm = waveforms[idx]
            polarity_results = compare_methods(wfm, sample_interval=sample_interval)

            # Show current/selected polarity
            stored_polarity = None
            if 'polarity' in feature_names:
                polarity_idx = feature_names.index('polarity')
                stored_polarity = features[idx, polarity_idx]

            polarity_items = []
            polarity_items.append(html.Span("Polarity: ", style={'fontWeight': 'bold', 'color': '#333'}))

            # Show stored value
            if stored_polarity is not None:
                stored_str = "+" if stored_polarity > 0 else "-"
                polarity_items.append(html.Span(f"Stored: {stored_str}", style={'marginRight': '10px', 'color': '#666'}))

            # Show all methods comparison
            method_strs = []
            for method, pol in polarity_results.items():
                pol_str = "+" if pol > 0 else "-"
                is_selected = (polarity_method == method)
                style = {'fontWeight': 'bold', 'color': '#007bff'} if is_selected else {'color': '#888'}
                method_short = method.replace('_', ' ').title()[:10]
                method_strs.append(html.Span(f"{method_short}: {pol_str}", style={**style, 'marginRight': '8px', 'fontSize': '10px'}))

            feature_items.append(
                html.Div([
                    *polarity_items,
                    html.Br(),
                    html.Span("Methods: ", style={'fontSize': '10px', 'color': '#999'}),
                    *method_strs
                ], style={'marginBottom': '8px', 'padding': '5px', 'backgroundColor': '#f0f0f0', 'borderRadius': '3px'})
            )

        # Group features by category
        displayed_count = 0
        for group_name, group_features in FEATURE_GROUPS.items():
            group_items = []
            for feat_name in group_features:
                if feat_name in visible_features and feat_name in feature_names:
                    feat_idx = feature_names.index(feat_name)
                    value = features[idx, feat_idx]
                    formatted = format_feature_value(feat_name, value)
                    display_name = feat_name.replace('_', ' ').title()
                    group_items.append(
                        html.Span([
                            html.Span(f"{display_name}: ", style={'color': '#666'}),
                            html.Span(formatted, style={'fontWeight': 'bold'})
                        ], style={'marginRight': '15px'})
                    )
                    displayed_count += 1

            if group_items:
                feature_items.append(
                    html.Div([
                        html.Span(f"{group_name}: ", style={'color': '#999', 'fontSize': '10px'}),
                        *group_items
                    ], style={'marginBottom': '3px'})
                )

        # Add any features not in groups
        ungrouped = []
        all_grouped = [f for group in FEATURE_GROUPS.values() for f in group]
        for feat_name in visible_features:
            if feat_name not in all_grouped and feat_name in feature_names:
                feat_idx = feature_names.index(feat_name)
                value = features[idx, feat_idx]
                formatted = format_feature_value(feat_name, value)
                display_name = feat_name.replace('_', ' ').title()
                ungrouped.append(
                    html.Span([
                        html.Span(f"{display_name}: ", style={'color': '#666'}),
                        html.Span(formatted, style={'fontWeight': 'bold'})
                    ], style={'marginRight': '15px'})
                )

        if ungrouped:
            feature_items.append(
                html.Div([
                    html.Span("Other: ", style={'color': '#999', 'fontSize': '10px'}),
                    *ungrouped
                ], style={'marginBottom': '3px'})
            )

        if displayed_count == 0:
            return html.Div("No features selected", style={'color': '#888', 'fontStyle': 'italic'})

        return html.Div(feature_items)

    # Manual Cluster Definition Callbacks

    # Toggle manual cluster mode
    @app.callback(
        [Output('manual-cluster-mode-container', 'style'),
         Output('manual-mode-active', 'data'),
         Output('manual-cluster-assignments', 'data', allow_duplicate=True)],
        [Input('enter-manual-mode-btn', 'n_clicks'),
         Input('exit-manual-mode-btn', 'n_clicks')],
        [State('manual-mode-active', 'data')],
        prevent_initial_call=True
    )
    def toggle_manual_mode(enter_clicks, exit_clicks, is_active):
        """Toggle manual cluster mode on/off."""
        triggered = ctx.triggered_id

        if triggered == 'enter-manual-mode-btn':
            return {'width': '95%', 'margin': '10px auto', 'display': 'block'}, True, {}
        elif triggered == 'exit-manual-mode-btn':
            return {'width': '95%', 'margin': '10px auto', 'display': 'none'}, False, {}

        raise PreventUpdate

    # Create/update the manual cluster plot
    @app.callback(
        Output('manual-cluster-plot', 'figure'),
        [Input('manual-mode-active', 'data'),
         Input('manual-cluster-assignments', 'data'),
         Input('dataset-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_manual_cluster_plot(is_active, assignments, prefix):
        """Update the manual cluster plot with current assignments."""
        if not is_active or not prefix:
            raise PreventUpdate

        data = loader.load_all(prefix)
        if data['features'] is None or data['feature_names'] is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig

        features = data['features']
        feature_names = list(data['feature_names'])

        # Get phase and amplitude
        phase_idx = feature_names.index('phase_angle') if 'phase_angle' in feature_names else 0
        amp_pos_idx = feature_names.index('peak_amplitude_positive') if 'peak_amplitude_positive' in feature_names else 1
        amp_neg_idx = feature_names.index('peak_amplitude_negative') if 'peak_amplitude_negative' in feature_names else 2
        polarity_idx = feature_names.index('polarity') if 'polarity' in feature_names else None

        phases = features[:, phase_idx]
        amp_pos = features[:, amp_pos_idx]
        amp_neg = features[:, amp_neg_idx]

        if polarity_idx is not None:
            polarity = features[:, polarity_idx]
            amplitudes = np.where(polarity > 0, amp_pos, amp_neg)
        else:
            amplitudes = np.where(amp_pos >= np.abs(amp_neg), amp_pos, amp_neg)

        fig = go.Figure()

        # Separate points by assignment
        assignments = assignments or {}
        cluster_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

        # Find unassigned points
        assigned_indices = set(int(k) for k in assignments.keys())
        unassigned_mask = np.array([i not in assigned_indices for i in range(len(phases))])

        # Plot unassigned points in gray
        if np.any(unassigned_mask):
            unassigned_indices = np.where(unassigned_mask)[0].tolist()
            fig.add_trace(go.Scatter(
                x=phases[unassigned_mask],
                y=amplitudes[unassigned_mask],
                mode='markers',
                marker=dict(size=4, color='#cccccc', opacity=0.6),
                name='Unassigned',
                customdata=unassigned_indices,
                hovertemplate='Phase: %{x:.1f}°<br>Amplitude: %{y:.4f} V<br>Index: %{customdata}<extra>Unassigned</extra>'
            ))

        # Plot assigned points by cluster
        for cluster_num in sorted(set(assignments.values())):
            cluster_indices = [int(k) for k, v in assignments.items() if v == cluster_num]
            if cluster_indices:
                mask = np.zeros(len(phases), dtype=bool)
                for idx in cluster_indices:
                    if 0 <= idx < len(phases):
                        mask[idx] = True

                if np.any(mask):
                    cluster_idx_list = np.where(mask)[0].tolist()
                    fig.add_trace(go.Scatter(
                        x=phases[mask],
                        y=amplitudes[mask],
                        mode='markers',
                        marker=dict(size=5, color=cluster_colors.get(cluster_num, '#000'), opacity=0.8),
                        name=f'Cluster {cluster_num}',
                        customdata=cluster_idx_list,
                        hovertemplate=f'Phase: %{{x:.1f}}°<br>Amplitude: %{{y:.4f}} V<br>Index: %{{customdata}}<extra>Cluster {cluster_num}</extra>'
                    ))

        # Add reference lines
        for phase in [90, 180, 270]:
            fig.add_vline(x=phase, line_dash="dash", line_color="gray", opacity=0.3)
        fig.add_hline(y=0, line_color="gray", opacity=0.5)

        fig.update_layout(
            title="Manual Cluster Definition - Select points and assign to clusters",
            xaxis_title="Phase (degrees)",
            yaxis_title="Amplitude (V)",
            xaxis=dict(range=[0, 360]),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            margin=dict(r=150),
            dragmode='lasso'  # Default to lasso for easier selection
        )

        # Add modebar buttons for selection
        fig.update_layout(
            modebar_add=['select2d', 'lasso2d'],
            modebar_remove=['autoScale2d']
        )

        return fig

    # Capture selection from manual cluster plot
    @app.callback(
        [Output('current-selection-indices', 'data'),
         Output('manual-selection-count', 'children')],
        [Input('manual-cluster-plot', 'selectedData')],
        prevent_initial_call=True
    )
    def capture_selection(selected_data):
        """Capture selected points from the manual cluster plot."""
        if not selected_data or 'points' not in selected_data:
            return [], "0"

        # Extract indices from customdata
        indices = []
        for point in selected_data['points']:
            if 'customdata' in point:
                indices.append(point['customdata'])

        return indices, str(len(indices))

    @app.callback(
        [Output('manual-cluster-assignments', 'data'),
         Output('manual-cluster-status', 'children')],
        [Input('assign-cluster-1-btn', 'n_clicks'),
         Input('assign-cluster-2-btn', 'n_clicks'),
         Input('assign-cluster-3-btn', 'n_clicks'),
         Input('assign-cluster-4-btn', 'n_clicks'),
         Input('clear-manual-clusters-btn', 'n_clicks')],
        [State('current-selection-indices', 'data'),
         State('manual-cluster-assignments', 'data')],
        prevent_initial_call=True
    )
    def assign_to_cluster(c1, c2, c3, c4, clear, current_selection, assignments):
        """Assign selected points to a cluster."""
        triggered = ctx.triggered_id
        assignments = assignments or {}

        if triggered == 'clear-manual-clusters-btn':
            return {}, html.Div("All assignments cleared. Start selecting points!", style={'color': '#17a2b8'})

        if not current_selection:
            return assignments, html.Div("No points selected. Use lasso or box select on the plot.",
                                        style={'color': '#856404', 'backgroundColor': '#fff3cd', 'padding': '5px', 'borderRadius': '3px'})

        # Determine which cluster was clicked
        cluster_num = None
        if triggered == 'assign-cluster-1-btn':
            cluster_num = 1
        elif triggered == 'assign-cluster-2-btn':
            cluster_num = 2
        elif triggered == 'assign-cluster-3-btn':
            cluster_num = 3
        elif triggered == 'assign-cluster-4-btn':
            cluster_num = 4

        if cluster_num:
            for idx in current_selection:
                assignments[str(idx)] = cluster_num

        # Generate status summary
        cluster_counts = {}
        for idx, c in assignments.items():
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

        if cluster_counts:
            status_items = []
            colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}
            for c in sorted(cluster_counts.keys()):
                status_items.append(
                    html.Span([
                        html.Span(f"Cluster {c}: ", style={'fontWeight': 'bold', 'color': colors.get(c, '#000')}),
                        html.Span(f"{cluster_counts[c]} pts  ", style={'marginRight': '10px'})
                    ])
                )
            status = html.Div([
                html.Span(f"Assigned: {sum(cluster_counts.values())} pts | ", style={'fontWeight': 'bold'}),
                *status_items
            ])
        else:
            status = html.Div("No points assigned yet. Select points and click a cluster button.", style={'color': '#666'})

        return assignments, status

    @app.callback(
        Output('manual-cluster-analysis-result', 'children'),
        [Input('analyze-manual-clusters-btn', 'n_clicks')],
        [State('manual-cluster-assignments', 'data'),
         State('dataset-dropdown', 'value')],
        prevent_initial_call=True
    )
    def analyze_manual_clusters(n_clicks, assignments, prefix):
        """Analyze which features best separate the manually-defined clusters."""
        if not n_clicks:
            raise PreventUpdate

        if not assignments or len(assignments) == 0:
            return html.Div("No cluster assignments. Please select points and assign them to clusters first.",
                          style={'color': '#856404', 'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '4px'})

        # Check we have at least 2 clusters
        unique_clusters = set(assignments.values())
        if len(unique_clusters) < 2:
            return html.Div("Need at least 2 different clusters to analyze. Please assign points to at least 2 clusters.",
                          style={'color': '#856404', 'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '4px'})

        # Load feature data
        data = loader.load_all(prefix)
        if data['features'] is None or data['feature_names'] is None:
            return html.Div("No feature data available for this dataset.",
                          style={'color': '#721c24', 'backgroundColor': '#f8d7da', 'padding': '10px', 'borderRadius': '4px'})

        features = data['features']
        feature_names = list(data['feature_names'])

        # Get indices and labels for assigned points
        indices = []
        labels = []
        for idx_str, cluster in assignments.items():
            idx = int(idx_str)
            if 0 <= idx < len(features):
                indices.append(idx)
                labels.append(cluster)

        if len(indices) < 10:
            return html.Div(f"Only {len(indices)} valid points assigned. Please assign more points for reliable analysis.",
                          style={'color': '#856404', 'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '4px'})

        # Extract features for assigned points
        X = features[indices]
        y = np.array(labels)

        # Calculate feature importance using multiple methods
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_selection import f_classif, mutual_info_classif

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Method 1: Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_scaled, y)
            rf_importance = rf.feature_importances_

            # Method 2: ANOVA F-scores
            f_scores, _ = f_classif(X_scaled, y)
            f_scores = np.nan_to_num(f_scores, nan=0.0)

            # Method 3: Mutual Information
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

            # Combine scores (normalize each to 0-1 range and average)
            def normalize(arr):
                min_val, max_val = arr.min(), arr.max()
                if max_val > min_val:
                    return (arr - min_val) / (max_val - min_val)
                return np.zeros_like(arr)

            rf_norm = normalize(rf_importance)
            f_norm = normalize(f_scores)
            mi_norm = normalize(mi_scores)

            combined_score = (rf_norm + f_norm + mi_norm) / 3

            # Sort features by combined score
            sorted_indices = np.argsort(combined_score)[::-1]

            # Create results display
            results = []

            # Summary
            results.append(html.Div([
                html.H4("Feature Importance Analysis", style={'marginBottom': '10px', 'color': '#28a745'}),
                html.P(f"Analyzed {len(indices)} points across {len(unique_clusters)} clusters",
                      style={'color': '#666', 'marginBottom': '15px'})
            ]))

            # Top features table
            results.append(html.H5("Top Features for Separating Your Clusters:", style={'marginBottom': '10px'}))

            table_rows = [
                html.Tr([
                    html.Th("Rank", style={'padding': '8px', 'borderBottom': '2px solid #dee2e6'}),
                    html.Th("Feature", style={'padding': '8px', 'borderBottom': '2px solid #dee2e6'}),
                    html.Th("Combined Score", style={'padding': '8px', 'borderBottom': '2px solid #dee2e6'}),
                    html.Th("RF Importance", style={'padding': '8px', 'borderBottom': '2px solid #dee2e6'}),
                    html.Th("F-Score", style={'padding': '8px', 'borderBottom': '2px solid #dee2e6'}),
                    html.Th("Mutual Info", style={'padding': '8px', 'borderBottom': '2px solid #dee2e6'})
                ])
            ]

            # Show top 15 features
            for rank, idx in enumerate(sorted_indices[:15], 1):
                feat_name = feature_names[idx]
                bg_color = '#d4edda' if rank <= 5 else '#fff3cd' if rank <= 10 else '#fff'
                table_rows.append(
                    html.Tr([
                        html.Td(str(rank), style={'padding': '6px', 'textAlign': 'center'}),
                        html.Td(feat_name, style={'padding': '6px', 'fontWeight': 'bold' if rank <= 5 else 'normal'}),
                        html.Td(f"{combined_score[idx]:.3f}", style={'padding': '6px', 'textAlign': 'center'}),
                        html.Td(f"{rf_importance[idx]:.3f}", style={'padding': '6px', 'textAlign': 'center'}),
                        html.Td(f"{f_scores[idx]:.1f}", style={'padding': '6px', 'textAlign': 'center'}),
                        html.Td(f"{mi_scores[idx]:.3f}", style={'padding': '6px', 'textAlign': 'center'})
                    ], style={'backgroundColor': bg_color})
                )

            results.append(
                html.Table(table_rows, style={
                    'width': '100%',
                    'borderCollapse': 'collapse',
                    'fontSize': '13px',
                    'marginBottom': '15px'
                })
            )

            # Recommendation
            top_features = [feature_names[sorted_indices[i]] for i in range(min(5, len(sorted_indices)))]
            results.append(html.Div([
                html.H5("Recommended Features for Clustering:", style={'marginTop': '15px', 'marginBottom': '10px'}),
                html.P([
                    "Based on your manual cluster definitions, these features best separate your clusters: ",
                    html.B(", ".join(top_features))
                ], style={'backgroundColor': '#d4edda', 'padding': '10px', 'borderRadius': '4px'}),
                html.P([
                    "Try selecting these features in the ",
                    html.B("Pulse Features (for clustering)"),
                    " section above, then click ",
                    html.B("Recluster"),
                    " to see if the automatic clustering matches your expectations."
                ], style={'color': '#666', 'fontSize': '12px', 'marginTop': '10px'})
            ]))

            return html.Div(results, style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})

        except ImportError:
            return html.Div("sklearn is required for feature analysis. Install with: pip install scikit-learn",
                          style={'color': '#721c24', 'backgroundColor': '#f8d7da', 'padding': '10px', 'borderRadius': '4px'})
        except Exception as e:
            return html.Div(f"Error during analysis: {str(e)}",
                          style={'color': '#721c24', 'backgroundColor': '#f8d7da', 'padding': '10px', 'borderRadius': '4px'})

    return app


def main():
    parser = argparse.ArgumentParser(description="PD Visualization GUI")
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory containing data files')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    args = parser.parse_args()

    if not DASH_AVAILABLE:
        print("Error: Dash is not installed.")
        print("Install with: pip install dash plotly")
        return

    print(f"Starting PD Visualization GUI...")
    print(f"Data directory: {args.data_dir}")
    print(f"Open browser to: http://localhost:{args.port}")

    app = create_app(args.data_dir)
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
