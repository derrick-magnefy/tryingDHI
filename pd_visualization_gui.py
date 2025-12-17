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

try:
    from dash import Dash, html, dcc, callback, Output, Input, State
    from dash import ctx  # For callback context
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Dash not available. Install with: pip install dash plotly")

from polarity_methods import (
    calculate_polarity, compare_methods, POLARITY_METHODS,
    DEFAULT_POLARITY_METHOD, get_method_description
)

DATA_DIR = "Rugged Data Files"

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
                        values = [float(v) for v in line.split('\t')]
                        waveforms.append(values)
            return np.array(waveforms) if waveforms else None
        except Exception as e:
            print(f"Error loading waveforms: {e}")
            return None

    def load_all(self, prefix, method='dbscan'):
        """Load all data for a dataset."""
        features, feature_names = self.load_features(prefix)
        cluster_labels = self.load_cluster_labels(prefix, method)
        pd_types = self.load_pd_types(prefix, method)
        waveforms = self.load_waveforms(prefix)

        return {
            'features': features,
            'feature_names': feature_names,
            'cluster_labels': cluster_labels,
            'pd_types': pd_types,
            'waveforms': waveforms
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
            ], style={'width': '60%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([
                html.Label("Polarity Method:"),
                dcc.Dropdown(
                    id='polarity-method-dropdown',
                    options=[{'label': 'Stored (from features)', 'value': 'stored'}] +
                            [{'label': get_method_description(m), 'value': m} for m in POLARITY_METHODS],
                    value='stored',
                    style={'width': '100%'}
                ),
            ], style={'width': '38%', 'display': 'inline-block'}),
        ], style={'width': '90%', 'margin': '10px auto'}),

        # Statistics
        html.Div([
            dcc.Markdown(id='stats-text', style={'padding': '10px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})
        ], style={'width': '90%', 'margin': '10px auto'}),

        # Main plots row
        html.Div([
            # Cluster PRPD
            html.Div([
                dcc.Graph(id='cluster-prpd', style={'height': '600px'})
            ], style={'width': '48%', 'display': 'inline-block'}),

            # PD Type PRPD
            html.Div([
                dcc.Graph(id='pdtype-prpd', style={'height': '600px'})
            ], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '95%', 'margin': 'auto'}),

        # Second row
        html.Div([
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

            # Histogram
            html.Div([
                dcc.Graph(id='histogram', style={'height': '525px'})
            ], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '95%', 'margin': 'auto'}),

        # Hidden stores
        dcc.Store(id='selected-index'),
        dcc.Store(id='current-data-store'),
        dcc.Store(id='selected-waveform-idx'),
        dcc.Store(id='feature-toggle-store', data=DEFAULT_VISIBLE_FEATURES),
    ])

    @app.callback(
        [Output('cluster-prpd', 'figure'),
         Output('pdtype-prpd', 'figure'),
         Output('histogram', 'figure'),
         Output('stats-text', 'children'),
         Output('current-data-store', 'data')],
        [Input('dataset-dropdown', 'value'),
         Input('polarity-method-dropdown', 'value')]
    )
    def update_dataset(prefix, polarity_method):
        if not prefix:
            empty_fig = go.Figure()
            return empty_fig, empty_fig, empty_fig, "No dataset selected", None

        data = loader.load_all(prefix)

        # Determine polarity method to use
        pm = polarity_method if polarity_method and polarity_method != 'stored' else None

        cluster_fig = create_prpd_plot(
            data['features'], data['feature_names'],
            data['cluster_labels'], data['pd_types'],
            color_by='cluster',
            waveforms=data['waveforms'],
            polarity_method=pm
        )

        pdtype_fig = create_prpd_plot(
            data['features'], data['feature_names'],
            data['cluster_labels'], data['pd_types'],
            color_by='pdtype',
            waveforms=data['waveforms'],
            polarity_method=pm
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
            polarity_results = compare_methods(wfm)

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
