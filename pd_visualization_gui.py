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

try:
    from dash import Dash, html, dcc, callback, Output, Input, State
    from dash import ctx  # For callback context
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

DATA_DIR = "Rugged Data Files"

# Clustering methods available
CLUSTERING_METHODS = ['dbscan', 'kmeans']
DEFAULT_CLUSTERING_METHOD = 'dbscan'

# Pulse features used for clustering (from extract_features.py)
PULSE_FEATURES = [
    'phase_angle',
    'peak_amplitude_positive',
    'peak_amplitude_negative',
    'polarity',
    'rise_time',
    'fall_time',
    'pulse_width',
    'slew_rate',
    'energy',
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
]

# Default pulse features for clustering (all features selected by default)
DEFAULT_CLUSTERING_FEATURES = PULSE_FEATURES.copy()

# Cluster-level aggregated features (from aggregate_cluster_features.py)
CLUSTER_FEATURES = [
    'pulses_per_positive_halfcycle',
    'pulses_per_negative_halfcycle',
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
    """Handles loading PD analysis data."""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.datasets = []
        self.find_datasets()

    def find_datasets(self):
        """Find all available datasets."""
        feature_files = glob.glob(os.path.join(self.data_dir, "*-features.csv"))
        self.datasets = []

        for f in sorted(feature_files):
            basename = os.path.basename(f)
            prefix = basename.replace("-features.csv", "")
            self.datasets.append(prefix)

        return self.datasets

    def load_features(self, prefix):
        """Load features from CSV."""
        filepath = os.path.join(self.data_dir, f"{prefix}-features.csv")
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
        filepath = os.path.join(self.data_dir, f"{prefix}-clusters-{method}.csv")
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
        filepath = os.path.join(self.data_dir, f"{prefix}-pd-types-{method}.csv")
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
        """Load raw waveforms from WFMs.txt file."""
        filepath = os.path.join(self.data_dir, f"{prefix}-WFMs.txt")
        if not os.path.exists(filepath):
            return None

        try:
            waveforms = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Filter empty strings (consistent with extract_features.py)
                        values = [float(v) for v in line.split('\t') if v.strip()]
                        waveforms.append(np.array(values))
            return waveforms if waveforms else None
        except Exception as e:
            print(f"Error loading waveforms: {e}")
            return None

    def load_settings(self, prefix):
        """Load settings from -SG.txt file."""
        filepath = os.path.join(self.data_dir, f"{prefix}-SG.txt")
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
        margin=dict(r=150)
    )

    return fig


def create_waveform_plot(waveforms, idx, features, feature_names, cluster_labels, pd_types):
    """Create waveform plot for selected point."""
    fig = go.Figure()

    if waveforms is None or idx is None or idx >= len(waveforms):
        fig.add_annotation(
            text="Click on a point in the PRPD plot to view waveform",
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
                    style={'width': '100%'}
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([
                html.Label("Polarity Method (for display):"),
                dcc.Dropdown(
                    id='polarity-method-dropdown',
                    options=[{'label': 'Stored (from features)', 'value': 'stored'}] +
                            [{'label': get_method_description(m), 'value': m} for m in POLARITY_METHODS],
                    value='stored',
                    style={'width': '100%'}
                ),
            ], style={'width': '35%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([
                html.Label("Re-analyze:", style={'visibility': 'hidden'}),
                html.Button(
                    'Re-analyze',
                    id='reanalyze-button',
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'padding': '8px',
                        'backgroundColor': '#007bff',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontSize': '12px'
                    }
                ),
            ], style={'width': '14%', 'display': 'inline-block', 'verticalAlign': 'bottom'}),
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
                                        labelStyle={'marginRight': '20px'}
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
                                               style={'padding': '5px 10px'}),
                                ], style={'marginBottom': '10px'}),
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
                                               style={'padding': '5px 10px'}),
                                ], style={'marginBottom': '10px'}),
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
            html.Div(id='cluster-details-display', style={
                'display': 'none',
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'border': '1px solid #dee2e6',
                'borderRadius': '5px',
                'maxHeight': '400px',
                'overflowY': 'auto'
            })
        ], style={'width': '95%', 'margin': '10px auto'}),

        # PD Type PRPD - full row
        html.Div([
            dcc.Graph(id='pdtype-prpd', style={'height': '600px'})
        ], style={'width': '95%', 'margin': 'auto'}),

        # Third row: Histogram (left) and Waveform+Features (right)
        html.Div([
            # Histogram (Phase Distribution)
            html.Div([
                dcc.Graph(id='histogram', style={'height': '525px'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            # Waveform viewer and features
            html.Div([
                dcc.Graph(id='waveform-plot', style={'height': '400px'}),
                # Feature display area
                html.Div([
                    html.Div([
                        html.Label("Show Features:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                        dcc.Checklist(
                            id='feature-toggles',
                            options=[],  # Will be populated by callback
                            value=DEFAULT_VISIBLE_FEATURES,
                            inline=True,
                            style={'fontSize': '11px'},
                            inputStyle={'marginRight': '3px'},
                            labelStyle={'marginRight': '12px'}
                        ),
                    ], style={'marginBottom': '8px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
                    html.Div(id='feature-display', style={
                        'fontSize': '12px',
                        'fontFamily': 'monospace',
                        'backgroundColor': '#f8f8f8',
                        'padding': '8px',
                        'borderRadius': '4px',
                        'maxHeight': '120px',
                        'overflowY': 'auto'
                    })
                ], style={'padding': '5px', 'backgroundColor': '#fff', 'border': '1px solid #ddd', 'borderRadius': '4px'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '95%', 'margin': 'auto'}),

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
            # Reload data for the waveform
            data = loader.load_all(prefix)
            return create_waveform_plot(
                data['waveforms'], idx,
                data['features'], data['feature_names'],
                data['cluster_labels'], data['pd_types']
            ), idx
        elif prefix:
            # No point clicked yet, show placeholder
            data = loader.load_all(prefix)
            return create_waveform_plot(
                data['waveforms'], None,
                data['features'], data['feature_names'],
                data['cluster_labels'], data['pd_types']
            ), None

        return create_waveform_plot(None, None, None, None, None, None), None

    @app.callback(
        [Output('cluster-details-display', 'children'),
         Output('cluster-details-display', 'style')],
        [Input('cluster-prpd', 'clickData'),
         Input('pdtype-prpd', 'clickData')],
        [State('show-cluster-details-checkbox', 'value'),
         State('current-data-store', 'data'),
         State('clustering-method-radio', 'value')],
        prevent_initial_call=True
    )
    def update_cluster_details(cluster_click, pdtype_click, show_details, prefix, clustering_method):
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
