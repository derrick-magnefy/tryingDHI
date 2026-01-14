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
from scipy.signal import find_peaks, hilbert
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


def method_smart_bounds(waveform: np.ndarray, sample_interval: float,
                        noise_window: int = 50, snr_threshold: float = 3.0,
                        min_separation_us: float = 0.5) -> Dict:
    """
    Smart boundary detection that:
    1. Estimates noise floor from signal edges
    2. Uses envelope (Hilbert) to find true signal boundaries
    3. For multi-pulse: isolates main pulse with smart boundaries
    4. For single-pulse: finds where signal actually returns to noise

    Parameters:
        noise_window: samples at edges to estimate noise floor
        snr_threshold: multiplier above noise floor to define signal
        min_separation_us: minimum pulse separation for multi-pulse detection
    """
    n = len(waveform)
    if n < noise_window * 2:
        return method_original(waveform, sample_interval)

    # Step 1: Estimate noise floor from edges of waveform
    # Use both start and end, take the minimum RMS (cleaner estimate)
    noise_start = waveform[:noise_window]
    noise_end = waveform[-noise_window:]
    noise_rms_start = np.sqrt(np.mean(noise_start ** 2))
    noise_rms_end = np.sqrt(np.mean(noise_end ** 2))
    noise_floor = min(noise_rms_start, noise_rms_end)

    # If noise floor is essentially zero, use a small fraction of max
    if noise_floor < 1e-10:
        noise_floor = np.max(np.abs(waveform)) * 0.01

    # Step 2: Compute envelope using Hilbert transform
    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope to reduce noise
    envelope_smooth = gaussian_filter1d(envelope, sigma=3)

    # Step 3: Define signal threshold as multiple of noise floor
    signal_threshold = snr_threshold * noise_floor

    # Find where envelope exceeds threshold
    above_threshold = envelope_smooth > signal_threshold

    if not np.any(above_threshold):
        # No signal above threshold - return minimal window around peak
        peak_idx = np.argmax(envelope_smooth)
        half_win = max(25, int(0.2e-6 / sample_interval))  # At least 0.2µs
        start_idx = max(0, peak_idx - half_win)
        end_idx = min(n, peak_idx + half_win)
        return {
            'name': f'SmartBounds {snr_threshold:.0f}x (low SNR)',
            'waveform': waveform[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'description': f'Low SNR, minimal window around peak'
        }

    # Step 4: Check for multi-pulse
    pulse_info = detect_pulses(waveform, sample_interval,
                               amplitude_threshold_ratio=0.3,
                               min_pulse_separation_us=min_separation_us)

    if pulse_info['pulse_count'] > 1:
        # Multi-pulse: isolate the main pulse using envelope
        pulse_indices = pulse_info['pulse_indices']
        pulse_amps = [envelope_smooth[i] for i in pulse_indices]
        main_pulse_idx = pulse_indices[np.argmax(pulse_amps)]

        # Find the boundaries of just the main pulse using envelope
        # Search backward from main pulse to find where envelope drops
        start_idx = main_pulse_idx
        for i in range(main_pulse_idx, -1, -1):
            if envelope_smooth[i] < signal_threshold:
                start_idx = i
                break
            # Also stop if we're getting close to another pulse
            for other_idx in pulse_indices:
                if other_idx < main_pulse_idx and i <= other_idx:
                    start_idx = max(i, other_idx + 10)
                    break

        # Search forward from main pulse
        end_idx = main_pulse_idx
        for i in range(main_pulse_idx, n):
            if envelope_smooth[i] < signal_threshold:
                end_idx = i
                break
            # Also stop if we're getting close to another pulse
            for other_idx in pulse_indices:
                if other_idx > main_pulse_idx and i >= other_idx - 10:
                    end_idx = min(i, other_idx - 10)
                    break

        # Add small padding
        padding = max(10, int(0.1e-6 / sample_interval))
        start_idx = max(0, start_idx - padding)
        end_idx = min(n, end_idx + padding)

        return {
            'name': f'SmartBounds {snr_threshold:.0f}x ({pulse_info["pulse_count"]}→1 pulse)',
            'waveform': waveform[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'description': f'Multi-pulse isolated using envelope, SNR threshold={snr_threshold:.1f}x noise'
        }

    # Step 5: Single pulse - find true boundaries where envelope returns to noise
    indices = np.where(above_threshold)[0]

    # Find continuous region around peak
    peak_idx = np.argmax(envelope_smooth)

    # Search backward from peak for true start
    start_idx = indices[0]
    for i in range(peak_idx, -1, -1):
        if envelope_smooth[i] < signal_threshold:
            start_idx = i
            break

    # Search forward from peak for true end (where signal returns to noise)
    end_idx = indices[-1]
    for i in range(peak_idx, n):
        if envelope_smooth[i] < signal_threshold:
            # Found where envelope drops below threshold
            # But check if signal rises again (might be ringing)
            remaining = envelope_smooth[i:min(i+50, n)]
            if np.max(remaining) < signal_threshold * 1.5:
                # Signal stays low, this is the true end
                end_idx = i
                break

    # Add padding proportional to signal duration
    signal_duration = end_idx - start_idx
    padding = max(10, signal_duration // 10)
    start_idx = max(0, start_idx - padding)
    end_idx = min(n, end_idx + padding)

    return {
        'name': f'SmartBounds {snr_threshold:.0f}x (single pulse)',
        'waveform': waveform[start_idx:end_idx],
        'start_idx': start_idx,
        'end_idx': end_idx,
        'description': f'Envelope-based bounds, {snr_threshold:.1f}x noise threshold'
    }


def method_smart_bounds_conservative(waveform: np.ndarray, sample_interval: float) -> Dict:
    """SmartBounds with conservative (higher) SNR threshold."""
    return method_smart_bounds(waveform, sample_interval, snr_threshold=5.0)


def method_smart_bounds_aggressive(waveform: np.ndarray, sample_interval: float) -> Dict:
    """SmartBounds with aggressive (lower) SNR threshold - captures more of the tail."""
    return method_smart_bounds(waveform, sample_interval, snr_threshold=2.0)


def method_smart_bounds_padded(waveform: np.ndarray, sample_interval: float,
                                min_window_us: float = 0.5,
                                snr_threshold: float = 3.0) -> Dict:
    """
    SmartBounds with minimum window guarantee.
    Finds smart boundaries but ensures at least min_window_us around the peak.
    Best of both worlds: isolates multi-pulse, keeps enough signal for classification.

    Parameters:
        min_window_us: minimum window size in microseconds
        snr_threshold: SNR multiplier for boundary detection (2.0=aggressive, 3.0=default, 5.0=conservative)
    """
    # First get smart bounds result with specified threshold
    result = method_smart_bounds(waveform, sample_interval, snr_threshold=snr_threshold)

    # Calculate minimum window in samples
    min_samples = int(min_window_us * 1e-6 / sample_interval)

    # Check if result is too small
    current_samples = len(result['waveform'])

    if current_samples >= min_samples:
        result['name'] = f'SmartBounds {snr_threshold:.0f}x+ (min {min_window_us}µs)'
        result['description'] = f'Envelope bounds ({snr_threshold:.0f}x), kept {current_samples} samples (>= {min_samples} min)'
        return result

    # Need to expand - center around peak of the extracted region
    abs_wfm = np.abs(result['waveform'])
    local_peak = np.argmax(abs_wfm)

    # Convert back to original indices
    orig_start = result['start_idx']
    peak_in_original = orig_start + local_peak

    # Expand to minimum window
    half_window = min_samples // 2
    new_start = max(0, peak_in_original - half_window)
    new_end = min(len(waveform), peak_in_original + half_window)

    return {
        'name': f'SmartBounds {snr_threshold:.0f}x+ (min {min_window_us}µs)',
        'waveform': waveform[new_start:new_end],
        'start_idx': new_start,
        'end_idx': new_end,
        'description': f'Expanded to {min_window_us}µs minimum ({snr_threshold:.0f}x threshold)'
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
    'smart_bounds': method_smart_bounds,
    'smart_bounds_conservative': method_smart_bounds_conservative,
    'smart_bounds_aggressive': method_smart_bounds_aggressive,
    'smart_bounds_3x_0.5us': lambda w, s: method_smart_bounds_padded(w, s, 0.5, 3.0),
    'smart_bounds_3x_1.0us': lambda w, s: method_smart_bounds_padded(w, s, 1.0, 3.0),
    'smart_bounds_2x_0.5us': lambda w, s: method_smart_bounds_padded(w, s, 0.5, 2.0),
    'smart_bounds_2x_1.0us': lambda w, s: method_smart_bounds_padded(w, s, 1.0, 2.0),
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
                    {'label': 'SmartBounds 3x', 'value': 'smart_bounds'},
                    {'label': 'SmartBounds 5x (conservative)', 'value': 'smart_bounds_conservative'},
                    {'label': 'SmartBounds 2x (aggressive)', 'value': 'smart_bounds_aggressive'},
                    {'label': 'SmartBounds 3x+ (min 0.5µs)', 'value': 'smart_bounds_3x_0.5us'},
                    {'label': 'SmartBounds 3x+ (min 1.0µs)', 'value': 'smart_bounds_3x_1.0us'},
                    {'label': 'SmartBounds 2x+ (min 0.5µs)', 'value': 'smart_bounds_2x_0.5us'},
                    {'label': 'SmartBounds 2x+ (min 1.0µs)', 'value': 'smart_bounds_2x_1.0us'},
                ],
                value=['original', 'smart_bounds', 'smart_bounds_2x_1.0us'],
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
        info = loader.get_info()
        sample_rate = info.get('sample_rate', 250e6)
        sample_interval = 1.0 / sample_rate

        # Load raw signal
        load_result = loader.load_channel(data_channel)
        raw_signal = load_result['signal']
        sample_rate = load_result.get('sample_rate', sample_rate)
        sample_interval = 1.0 / sample_rate

        # Run DWT detection
        detector = DWTDetector(sample_rate=sample_rate)
        detection_result = detector.detect(raw_signal)
        events = detection_result.events

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

        waveforms = [wfm.waveform.tolist() for wfm in result.waveforms]

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
    print("\nStarting server at http://localhost:8052")
    print("\nMethods available:")
    for name, _ in ADAPTIVE_METHODS.items():
        print(f"  - {name}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)

    app.run(debug=True, port=8052)
