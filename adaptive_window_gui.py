#!/usr/bin/env python3
"""
Adaptive Window Sizing Comparison GUI

Compare different methods for dynamically sizing waveform extraction windows.
This helps identify the best approach to avoid false multi-pulse detection.

Methods:
1. Original (Fixed) - Full extraction window as-is
2. Rise-Fall Based - Window based on 10%-90%-10% timing
3. Energy Based - Window containing 90% of signal energy
4. Peak Centered - Fixed small window centered on main peak
5. Adaptive Shrink - Shrink to primary pulse if multi-pulse detected
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pre_middleware.loaders.mat_loader import MatLoader
from pre_middleware.waveletProc.dwt_detector import DWTDetector, BAND_WINDOWS
from pre_middleware.waveletProc.waveform_extractor import WaveletExtractor
from pdlib.features.pulse_detection import detect_pulses


# =============================================================================
# ADAPTIVE WINDOW METHODS
# =============================================================================

def method_original(waveform: np.ndarray, sample_interval: float) -> Dict:
    """Original fixed window - no modification."""
    return {
        'name': 'Original (Fixed)',
        'waveform': waveform,
        'start_idx': 0,
        'end_idx': len(waveform),
        'description': 'Full extraction window, no adaptation'
    }


def method_rise_fall_based(waveform: np.ndarray, sample_interval: float,
                           threshold_pct: float = 0.1, padding_factor: float = 1.5) -> Dict:
    """
    Window based on rise/fall time analysis.
    Find where signal rises above and falls below threshold, add padding.
    """
    abs_wfm = np.abs(waveform)
    max_amp = np.max(abs_wfm)

    if max_amp == 0:
        return method_original(waveform, sample_interval)

    threshold = threshold_pct * max_amp

    # Find where signal exceeds threshold
    above_threshold = abs_wfm > threshold
    if not np.any(above_threshold):
        return method_original(waveform, sample_interval)

    indices = np.where(above_threshold)[0]
    start_idx = indices[0]
    end_idx = indices[-1]

    # Add padding
    signal_width = end_idx - start_idx
    padding = int(signal_width * (padding_factor - 1) / 2)

    new_start = max(0, start_idx - padding)
    new_end = min(len(waveform), end_idx + padding)

    return {
        'name': 'Rise-Fall Based',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
        'description': f'10%-threshold crossing with {padding_factor}x padding'
    }


def method_energy_based(waveform: np.ndarray, sample_interval: float,
                        energy_fraction: float = 0.90) -> Dict:
    """
    Window containing specified fraction of total signal energy.
    Centers on the energy centroid.
    """
    energy = waveform ** 2
    total_energy = np.sum(energy)

    if total_energy == 0:
        return method_original(waveform, sample_interval)

    # Find cumulative energy
    cumsum = np.cumsum(energy)
    normalized = cumsum / total_energy

    # Find bounds containing energy_fraction of energy
    margin = (1 - energy_fraction) / 2
    start_idx = np.searchsorted(normalized, margin)
    end_idx = np.searchsorted(normalized, 1 - margin)

    # Add small buffer
    buffer = max(10, (end_idx - start_idx) // 10)
    new_start = max(0, start_idx - buffer)
    new_end = min(len(waveform), end_idx + buffer)

    return {
        'name': f'Energy Based ({int(energy_fraction*100)}%)',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
        'description': f'Window containing {int(energy_fraction*100)}% of signal energy'
    }


def method_peak_centered(waveform: np.ndarray, sample_interval: float,
                         window_us: float = 0.5) -> Dict:
    """
    Fixed small window centered on the main peak.
    """
    abs_wfm = np.abs(waveform)
    peak_idx = np.argmax(abs_wfm)

    # Convert window size to samples
    window_samples = int(window_us * 1e-6 / sample_interval)
    half_window = window_samples // 2

    new_start = max(0, peak_idx - half_window)
    new_end = min(len(waveform), peak_idx + half_window)

    return {
        'name': f'Peak Centered ({window_us}µs)',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
        'description': f'Fixed {window_us}µs window centered on main peak'
    }


def method_adaptive_shrink(waveform: np.ndarray, sample_interval: float,
                           min_separation_us: float = 0.5) -> Dict:
    """
    If multi-pulse detected, shrink to just the primary (largest) pulse.
    Otherwise, use rise-fall method.
    """
    # First check for multi-pulse
    pulse_info = detect_pulses(waveform, sample_interval,
                               amplitude_threshold_ratio=0.3,
                               min_pulse_separation_us=min_separation_us)

    if pulse_info['pulse_count'] <= 1:
        # Not multi-pulse, use rise-fall method
        result = method_rise_fall_based(waveform, sample_interval)
        result['name'] = 'Adaptive Shrink (single pulse)'
        result['description'] = 'Single pulse detected, using rise-fall window'
        return result

    # Multi-pulse detected - isolate the largest pulse
    abs_wfm = np.abs(waveform)
    pulse_indices = pulse_info['pulse_indices']
    pulse_amps = [abs_wfm[i] for i in pulse_indices]

    # Find the largest pulse
    main_pulse_idx = pulse_indices[np.argmax(pulse_amps)]

    # Determine isolation window - halfway to nearest other pulse
    min_dist = len(waveform)  # Initialize with max
    for idx in pulse_indices:
        if idx != main_pulse_idx:
            dist = abs(idx - main_pulse_idx)
            min_dist = min(min_dist, dist)

    # Use half the distance to nearest pulse, but at least 50 samples
    half_window = max(50, min_dist // 2)

    new_start = max(0, main_pulse_idx - half_window)
    new_end = min(len(waveform), main_pulse_idx + half_window)

    return {
        'name': f'Adaptive Shrink ({pulse_info["pulse_count"]} pulses → 1)',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
        'description': f'Multi-pulse detected, isolated main pulse from {pulse_info["pulse_count"]} pulses'
    }


def method_derivative_based(waveform: np.ndarray, sample_interval: float,
                            smoothing_sigma: float = 2.0) -> Dict:
    """
    Window based on where the derivative is significant.
    Good for finding the active portion of the signal.
    """
    # Smooth the waveform slightly
    smoothed = gaussian_filter1d(np.abs(waveform), sigma=smoothing_sigma)

    # Compute derivative
    deriv = np.abs(np.diff(smoothed))
    max_deriv = np.max(deriv)

    if max_deriv == 0:
        return method_original(waveform, sample_interval)

    # Find where derivative is significant (>5% of max)
    threshold = 0.05 * max_deriv
    significant = deriv > threshold

    if not np.any(significant):
        return method_original(waveform, sample_interval)

    indices = np.where(significant)[0]
    start_idx = max(0, indices[0] - 20)
    end_idx = min(len(waveform), indices[-1] + 20)

    return {
        'name': 'Derivative Based',
        'waveform': waveform[start_idx:end_idx],
        'start_idx': start_idx,
        'end_idx': end_idx,
        'description': 'Window where signal derivative is significant'
    }


# All available methods
ADAPTIVE_METHODS = {
    'original': method_original,
    'rise_fall': method_rise_fall_based,
    'energy_90': lambda w, s: method_energy_based(w, s, 0.90),
    'energy_95': lambda w, s: method_energy_based(w, s, 0.95),
    'peak_0.5us': lambda w, s: method_peak_centered(w, s, 0.5),
    'peak_1.0us': lambda w, s: method_peak_centered(w, s, 1.0),
    'peak_2.0us': lambda w, s: method_peak_centered(w, s, 2.0),
    'adaptive_shrink': method_adaptive_shrink,
    'derivative': method_derivative_based,
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_waveform(waveform: np.ndarray, sample_interval: float) -> Dict:
    """Analyze a waveform for multi-pulse and other characteristics."""
    pulse_info = detect_pulses(waveform, sample_interval)

    abs_wfm = np.abs(waveform)
    max_amp = np.max(abs_wfm)
    peak_idx = np.argmax(abs_wfm)

    # Calculate energy distribution
    energy = waveform ** 2
    total_energy = np.sum(energy)

    # Find signal duration (10% threshold crossing)
    threshold = 0.1 * max_amp
    above = abs_wfm > threshold
    if np.any(above):
        indices = np.where(above)[0]
        duration_samples = indices[-1] - indices[0]
        duration_us = duration_samples * sample_interval * 1e6
    else:
        duration_us = 0

    return {
        'pulse_count': pulse_info['pulse_count'],
        'pulse_indices': pulse_info['pulse_indices'],
        'is_multi_pulse': pulse_info['pulse_count'] > 1,
        'max_amplitude': max_amp,
        'peak_index': peak_idx,
        'duration_us': duration_us,
        'total_samples': len(waveform),
        'window_us': len(waveform) * sample_interval * 1e6,
    }


# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, title='Adaptive Window Comparison')

app.layout = html.Div([
    html.H1('Adaptive Window Sizing Comparison', style={'textAlign': 'center'}),

    html.Div([
        # File selection
        html.Div([
            html.Label('MAT File:'),
            dcc.Input(
                id='file-path',
                type='text',
                value='Example Data/Surface_PD_lab.mat',
                style={'width': '400px'}
            ),
            html.Button('Load', id='load-btn', n_clicks=0),
        ], style={'margin': '10px'}),

        # Detection settings
        html.Div([
            html.Label('K-Threshold:'),
            dcc.Input(id='k-threshold', type='number', value=6, step=0.5, style={'width': '80px'}),
            html.Label('Band:', style={'marginLeft': '20px'}),
            dcc.Dropdown(
                id='band-select',
                options=[
                    {'label': 'D1 (1.0µs window)', 'value': 'D1'},
                    {'label': 'D2 (2.0µs window)', 'value': 'D2'},
                    {'label': 'D3 (5.0µs window)', 'value': 'D3'},
                ],
                value='D3',
                style={'width': '200px', 'display': 'inline-block'}
            ),
            html.Label('Waveform Index:', style={'marginLeft': '20px'}),
            dcc.Input(id='waveform-idx', type='number', value=0, min=0, style={'width': '80px'}),
            html.Button('Prev', id='prev-btn', n_clicks=0, style={'marginLeft': '10px'}),
            html.Button('Next', id='next-btn', n_clicks=0),
        ], style={'margin': '10px'}),

        # Methods to compare
        html.Div([
            html.Label('Methods to Compare:'),
            dcc.Checklist(
                id='methods-checklist',
                options=[
                    {'label': 'Original (Fixed)', 'value': 'original'},
                    {'label': 'Rise-Fall Based', 'value': 'rise_fall'},
                    {'label': 'Energy 90%', 'value': 'energy_90'},
                    {'label': 'Energy 95%', 'value': 'energy_95'},
                    {'label': 'Peak Centered 0.5µs', 'value': 'peak_0.5us'},
                    {'label': 'Peak Centered 1.0µs', 'value': 'peak_1.0us'},
                    {'label': 'Peak Centered 2.0µs', 'value': 'peak_2.0us'},
                    {'label': 'Adaptive Shrink', 'value': 'adaptive_shrink'},
                    {'label': 'Derivative Based', 'value': 'derivative'},
                ],
                value=['original', 'rise_fall', 'energy_90', 'adaptive_shrink'],
                inline=True,
                style={'marginTop': '5px'}
            ),
        ], style={'margin': '10px'}),

        html.Button('Analyze', id='analyze-btn', n_clicks=0,
                   style={'margin': '10px', 'fontSize': '16px', 'padding': '10px 20px'}),
    ], style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'}),

    # Status
    html.Div(id='status-text', style={'margin': '10px', 'fontWeight': 'bold'}),

    # Main comparison plot
    dcc.Graph(id='comparison-plot', style={'height': '600px'}),

    # Analysis results table
    html.Div([
        html.H3('Analysis Results'),
        html.Div(id='analysis-table'),
    ], style={'margin': '20px'}),

    # Store for loaded data
    dcc.Store(id='waveforms-store'),
    dcc.Store(id='sample-interval-store'),
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto'})


@app.callback(
    [Output('waveforms-store', 'data'),
     Output('sample-interval-store', 'data'),
     Output('status-text', 'children')],
    [Input('load-btn', 'n_clicks')],
    [State('file-path', 'value'),
     State('k-threshold', 'value'),
     State('band-select', 'value')]
)
def load_data(n_clicks, file_path, k_threshold, band):
    if n_clicks == 0:
        return None, None, 'Click Load to load data'

    try:
        path = Path(file_path)
        if not path.exists():
            return None, None, f'File not found: {file_path}'

        # Load MAT file
        loader = MatLoader(str(path))
        channels = loader.list_channels()

        # Find data channel (skip reference)
        data_channel = None
        for ch in channels:
            if 'Ch2' not in ch:  # Skip reference
                data_channel = ch
                break

        if data_channel is None:
            return None, None, 'No data channel found'

        # Get signal info
        info = loader.get_channel_info(data_channel)
        sample_rate = info.get('sample_rate', 250e6)
        sample_interval = 1.0 / sample_rate

        # Load raw signal
        raw_signal = loader.load_channel(data_channel)

        # Run DWT detection
        detector = DWTDetector(sample_rate=sample_rate)
        events = detector.detect(raw_signal, k_factor=float(k_threshold))

        # Filter by band
        band_events = [e for e in events if e.band == band]

        if len(band_events) == 0:
            return None, None, f'No events found in band {band}'

        # Extract waveforms
        extractor = WaveletExtractor(sample_rate=sample_rate)
        limited_events = band_events[:100]  # Limit to first 100

        result = extractor.extract_from_events(raw_signal, limited_events)

        if len(result.waveforms) == 0:
            return None, None, 'Failed to extract waveforms'

        waveforms = [wfm.samples.tolist() for wfm in result.waveforms]

        return waveforms, sample_interval, f'Loaded {len(waveforms)} waveforms from band {band}'

    except Exception as e:
        return None, None, f'Error: {str(e)}'


@app.callback(
    Output('waveform-idx', 'value'),
    [Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks')],
    [State('waveform-idx', 'value'),
     State('waveforms-store', 'data')]
)
def navigate_waveforms(prev_clicks, next_clicks, current_idx, waveforms):
    if waveforms is None:
        return 0

    ctx = callback_context
    if not ctx.triggered:
        return current_idx

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    max_idx = len(waveforms) - 1

    if button_id == 'prev-btn':
        return max(0, current_idx - 1)
    elif button_id == 'next-btn':
        return min(max_idx, current_idx + 1)

    return current_idx


@app.callback(
    [Output('comparison-plot', 'figure'),
     Output('analysis-table', 'children')],
    [Input('analyze-btn', 'n_clicks')],
    [State('waveforms-store', 'data'),
     State('sample-interval-store', 'data'),
     State('waveform-idx', 'value'),
     State('methods-checklist', 'value')]
)
def analyze_and_plot(n_clicks, waveforms, sample_interval, wfm_idx, selected_methods):
    if waveforms is None or len(waveforms) == 0:
        return go.Figure(), 'Load data first'

    if wfm_idx >= len(waveforms):
        wfm_idx = 0

    waveform = np.array(waveforms[wfm_idx])

    # Remove baseline
    baseline = np.mean(waveform[:50]) if len(waveform) > 50 else np.mean(waveform)
    waveform = waveform - baseline

    # Apply each selected method
    results = {}
    for method_name in selected_methods:
        if method_name in ADAPTIVE_METHODS:
            results[method_name] = ADAPTIVE_METHODS[method_name](waveform, sample_interval)

    # Create comparison plot
    n_methods = len(results)
    if n_methods == 0:
        return go.Figure(), 'Select at least one method'

    # Create subplots - original on top, methods below
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.4, 0.6],
        subplot_titles=['Original Waveform with Window Regions', 'Extracted Windows Comparison'],
        vertical_spacing=0.12
    )

    # Time axis for original waveform
    time_us = np.arange(len(waveform)) * sample_interval * 1e6

    # Plot original waveform
    fig.add_trace(
        go.Scatter(
            x=time_us, y=waveform,
            mode='lines',
            name='Original',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )

    # Add colored regions showing each method's window
    colors = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)',
              'rgba(255,165,0,0.2)', 'rgba(128,0,128,0.2)', 'rgba(0,255,255,0.2)',
              'rgba(255,192,203,0.2)', 'rgba(165,42,42,0.2)', 'rgba(128,128,0,0.2)']

    for i, (method_name, result) in enumerate(results.items()):
        color = colors[i % len(colors)]
        start_us = result['start_idx'] * sample_interval * 1e6
        end_us = result['end_idx'] * sample_interval * 1e6

        fig.add_vrect(
            x0=start_us, x1=end_us,
            fillcolor=color,
            layer='below',
            line_width=0,
            row=1, col=1
        )

    # Mark detected pulses on original
    pulse_info = detect_pulses(waveform, sample_interval)
    for idx in pulse_info['pulse_indices']:
        fig.add_vline(
            x=idx * sample_interval * 1e6,
            line=dict(color='red', width=2, dash='dash'),
            row=1, col=1
        )

    # Plot extracted windows (normalized for comparison)
    for i, (method_name, result) in enumerate(results.items()):
        extracted = result['waveform']
        # Create time axis for extracted waveform
        extracted_time = np.arange(len(extracted)) * sample_interval * 1e6

        # Normalize amplitude for comparison
        max_amp = np.max(np.abs(extracted)) if len(extracted) > 0 else 1
        normalized = extracted / max_amp if max_amp > 0 else extracted

        # Offset for visibility
        offset = i * 2.5

        fig.add_trace(
            go.Scatter(
                x=extracted_time,
                y=normalized + offset,
                mode='lines',
                name=result['name'],
                line=dict(width=1.5)
            ),
            row=2, col=1
        )

        # Add method label
        fig.add_annotation(
            x=0,
            y=offset + 0.5,
            text=result['name'],
            showarrow=False,
            xanchor='left',
            font=dict(size=10),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        title=f'Waveform {wfm_idx + 1}/{len(waveforms)} - {pulse_info["pulse_count"]} pulse(s) detected'
    )

    fig.update_xaxes(title_text='Time (µs)', row=1, col=1)
    fig.update_xaxes(title_text='Time (µs)', row=2, col=1)
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Normalized (offset)', row=2, col=1)

    # Create analysis table
    table_rows = []
    table_rows.append(html.Tr([
        html.Th('Method'),
        html.Th('Samples'),
        html.Th('Duration (µs)'),
        html.Th('Pulses Detected'),
        html.Th('Is Multi-Pulse'),
        html.Th('Description'),
    ]))

    for method_name, result in results.items():
        extracted = result['waveform']
        analysis = analyze_waveform(extracted, sample_interval)

        # Color code multi-pulse status
        mp_style = {'backgroundColor': '#ffcccc'} if analysis['is_multi_pulse'] else {'backgroundColor': '#ccffcc'}

        table_rows.append(html.Tr([
            html.Td(result['name']),
            html.Td(len(extracted)),
            html.Td(f"{len(extracted) * sample_interval * 1e6:.2f}"),
            html.Td(analysis['pulse_count']),
            html.Td('YES' if analysis['is_multi_pulse'] else 'NO', style=mp_style),
            html.Td(result['description']),
        ]))

    table = html.Table(table_rows, style={
        'borderCollapse': 'collapse',
        'width': '100%',
        'border': '1px solid #ddd'
    })

    # Add CSS for table cells
    for row in table_rows:
        for cell in row.children:
            cell.style = {**cell.style, 'border': '1px solid #ddd', 'padding': '8px'} if hasattr(cell, 'style') and cell.style else {'border': '1px solid #ddd', 'padding': '8px'}

    return fig, table


if __name__ == '__main__':
    print("=" * 60)
    print("ADAPTIVE WINDOW SIZING COMPARISON GUI")
    print("=" * 60)
    print("\nStarting server at http://localhost:8051")
    print("\nMethods available:")
    for name, _ in ADAPTIVE_METHODS.items():
        print(f"  - {name}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)

    app.run_server(debug=True, port=8051)
