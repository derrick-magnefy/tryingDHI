#!/usr/bin/env python3
"""
Preprocessing Test GUI

A GUI for testing and validating SyncAvg and Wavelet processing approaches
for IEEE raw data files.

Features:
- SyncAvg Tab: Phase interpolation, synchronous averaging, heat maps
- Wavelet Tab: DWT band analysis, detection visualization, waveform comparison

Usage:
    python preprocessing_test_gui.py [--port PORT] [--data-dir PATH]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Dash imports
try:
    from dash import Dash, html, dcc, Input, Output, State, callback, no_update
    from dash.exceptions import PreventUpdate
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Dash not available. Install with: pip install dash plotly")

# Import preprocessing modules
try:
    from pre_middleware.loaders import MatLoader, load_mat_file
    from pre_middleware.syncAvg import PhaseInterpolator, SyncAverager
    from pre_middleware.waveletProc import (
        DWTDetector, WaveletExtractor, WAVELETS, DEFAULT_WAVELET,
        compute_kurtosis, compute_kurtosis_per_quadrant, BAND_WINDOWS,
    )
    from pre_middleware.triggerProc import TriggerDetector, WaveformExtractor
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    print(f"Preprocessing modules not available: {e}")

# Default data directory
DEFAULT_DATA_DIR = "IEEE Data"


def find_mat_files(data_dir: str) -> List[str]:
    """Find all .mat files in the data directory."""
    mat_files = []
    data_path = Path(data_dir)
    if data_path.exists():
        for mat_file in data_path.glob("**/*.mat"):
            mat_files.append(str(mat_file))
    return sorted(mat_files)


def create_app(data_dir: str = DEFAULT_DATA_DIR):
    """Create the Dash application."""
    if not DASH_AVAILABLE:
        raise ImportError("Dash is required. Install with: pip install dash plotly")

    app = Dash(__name__, suppress_callback_exceptions=True)

    # Find available .mat files
    mat_files = find_mat_files(data_dir)

    # App layout
    app.layout = html.Div([
        # Header
        html.H1("Preprocessing Test GUI", style={'textAlign': 'center', 'marginBottom': '20px'}),

        # File selection
        html.Div([
            html.Label("Select IEEE Data File:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='file-selector',
                options=[{'label': os.path.basename(f), 'value': f} for f in mat_files],
                value=mat_files[0] if mat_files else None,
                style={'width': '100%'}
            ),
            html.Div(id='file-info', style={'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),
        ], style={'padding': '20px', 'marginBottom': '20px'}),

        # Tabs
        dcc.Tabs(id='main-tabs', value='syncavg-tab', children=[
            dcc.Tab(label='SyncAvg Analysis', value='syncavg-tab'),
            dcc.Tab(label='Wavelet Analysis', value='wavelet-tab'),
            dcc.Tab(label='Comparison', value='comparison-tab'),
        ]),

        # Tab content
        html.Div(id='tab-content', style={'padding': '20px'}),

        # Storage for loaded data
        dcc.Store(id='loaded-data-store'),
        dcc.Store(id='syncavg-results-store'),
        dcc.Store(id='wavelet-results-store'),
        dcc.Store(id='trigger-results-store'),
    ])

    # Callback to load file info
    @app.callback(
        [Output('file-info', 'children'),
         Output('loaded-data-store', 'data')],
        Input('file-selector', 'value')
    )
    def load_file_info(filepath):
        if not filepath or not os.path.exists(filepath):
            return html.Div("No file selected"), None

        try:
            loader = MatLoader(filepath)
            info = loader.get_info()
            channels = loader.list_channels(include_reference=True)

            # Get sample rate
            data = loader.load()
            sample_rate = data.get('sample_rate', 0)
            ac_freq = data.get('ac_frequency', 60)
            signal = data.get('signal')
            n_samples = len(signal) if signal is not None else 0
            duration_ms = (n_samples / sample_rate * 1000) if sample_rate > 0 else 0

            info_div = html.Div([
                html.P([html.Strong("File: "), os.path.basename(filepath)]),
                html.P([html.Strong("Channels: "), ", ".join(channels)]),
                html.P([html.Strong("Sample Rate: "), f"{sample_rate/1e6:.2f} MSPS"]),
                html.P([html.Strong("Samples: "), f"{n_samples:,}"]),
                html.P([html.Strong("Duration: "), f"{duration_ms:.2f} ms ({duration_ms/1000*ac_freq:.2f} AC cycles)"]),
                html.P([html.Strong("AC Frequency: "), f"{ac_freq} Hz"]),
            ])

            store_data = {
                'filepath': filepath,
                'channels': channels,
                'sample_rate': sample_rate,
                'ac_frequency': ac_freq,
                'n_samples': n_samples,
            }

            return info_div, store_data

        except Exception as e:
            return html.Div(f"Error loading file: {str(e)}", style={'color': 'red'}), None

    # Callback to render tab content
    @app.callback(
        Output('tab-content', 'children'),
        [Input('main-tabs', 'value'),
         Input('loaded-data-store', 'data')]
    )
    def render_tab(tab, loaded_data):
        if tab == 'syncavg-tab':
            return create_syncavg_tab(loaded_data)
        elif tab == 'wavelet-tab':
            return create_wavelet_tab(loaded_data)
        elif tab == 'comparison-tab':
            return create_comparison_tab(loaded_data)
        return html.Div("Select a tab")

    # ========== SyncAvg Tab Callbacks ==========

    @app.callback(
        [Output('syncavg-results', 'children'),
         Output('syncavg-heatmap', 'figure'),
         Output('syncavg-phase-plot', 'figure'),
         Output('syncavg-results-store', 'data')],
        [Input('run-syncavg-btn', 'n_clicks')],
        [State('loaded-data-store', 'data'),
         State('syncavg-channel', 'value'),
         State('syncavg-ref-channel', 'value'),
         State('syncavg-phase-method', 'value'),
         State('syncavg-num-bins', 'value')]
    )
    def run_syncavg_analysis(n_clicks, loaded_data, channel, ref_channel, phase_method, num_bins):
        if not n_clicks or not loaded_data:
            raise PreventUpdate

        try:
            filepath = loaded_data['filepath']
            sample_rate = loaded_data['sample_rate']
            ac_freq = loaded_data['ac_frequency']

            # Load the signal
            loader = MatLoader(filepath)
            data = loader.load_channel(channel)
            signal = data['signal']

            # Load reference if specified
            if ref_channel and ref_channel != channel:
                ref_data = loader.load_channel(ref_channel)
                reference = ref_data['signal']
            else:
                reference = None

            # Phase interpolation
            interpolator = PhaseInterpolator(ac_frequency=ac_freq, method=phase_method)

            if reference is not None:
                phase_result = interpolator.interpolate(reference, sample_rate)
            else:
                # Use signal itself or generate synthetic phase
                samples_per_cycle = sample_rate / ac_freq
                phases = (np.arange(len(signal)) % samples_per_cycle) / samples_per_cycle * 360
                phase_result = type('PhaseResult', (), {
                    'phases': phases,
                    'estimated_frequency': ac_freq,
                    'cycles_detected': int(len(signal) / samples_per_cycle),
                    'quality': 0.5,
                    'method': 'synthetic'
                })()

            # Sync averaging
            averager = SyncAverager(num_bins=num_bins, use_absolute=True)
            avg_result = averager.compute(signal, phase_result.phases)

            # Find hotspots
            hotspots = averager.find_hotspots(avg_result, threshold_sigma=3.0)

            # Create results display
            results_div = html.Div([
                html.H4("SyncAvg Results"),
                html.P([html.Strong("Phase Method: "), phase_method]),
                html.P([html.Strong("Estimated Frequency: "), f"{phase_result.estimated_frequency:.2f} Hz"]),
                html.P([html.Strong("Cycles Detected: "), str(phase_result.cycles_detected)]),
                html.P([html.Strong("Phase Quality: "), f"{phase_result.quality:.2f}"]),
                html.P([html.Strong("Number of Bins: "), str(num_bins)]),
                html.P([html.Strong("Hotspots Found: "), str(len(hotspots))]),
                html.Hr(),
                html.H5("Hotspots:"),
                html.Ul([
                    html.Li(f"Phase {hs.center_phase:.1f}° (range: {hs.phase_range[0]:.1f}°-{hs.phase_range[1]:.1f}°), "
                           f"Mean: {hs.mean_amplitude:.2e}, Peak: {hs.peak_amplitude:.2e}, Significance: {hs.significance:.1f}σ")
                    for hs in hotspots
                ]) if hotspots else html.P("No hotspots found above threshold"),
            ])

            # Create heatmap (amplitude vs phase)
            heatmap_fig = go.Figure()

            # Create a 2D representation by reshaping data into cycles
            samples_per_cycle = int(sample_rate / ac_freq)
            n_complete_cycles = len(signal) // samples_per_cycle

            if n_complete_cycles > 0:
                # Reshape into cycles x samples_per_cycle
                cycle_data = signal[:n_complete_cycles * samples_per_cycle].reshape(n_complete_cycles, samples_per_cycle)
                phase_axis = np.linspace(0, 360, samples_per_cycle)

                heatmap_fig.add_trace(go.Heatmap(
                    z=np.abs(cycle_data),
                    x=phase_axis,
                    y=np.arange(n_complete_cycles),
                    colorscale='Viridis',
                    colorbar=dict(title='|Amplitude|'),
                ))
                heatmap_fig.update_layout(
                    title='Signal Amplitude vs Phase (by Cycle)',
                    xaxis_title='Phase (degrees)',
                    yaxis_title='Cycle Number',
                )
            else:
                # Single cycle - show as line plot
                heatmap_fig.add_trace(go.Scatter(
                    x=phase_result.phases,
                    y=np.abs(signal),
                    mode='markers',
                    marker=dict(size=2),
                ))
                heatmap_fig.update_layout(
                    title='Signal Amplitude vs Phase',
                    xaxis_title='Phase (degrees)',
                    yaxis_title='|Amplitude|',
                )

            # Create phase average plot
            phase_fig = go.Figure()

            # Mean amplitude
            phase_fig.add_trace(go.Scatter(
                x=avg_result.phase_bins,
                y=avg_result.mean_amplitude,
                mode='lines',
                name='Mean',
                line=dict(color='blue'),
            ))

            # Max amplitude
            phase_fig.add_trace(go.Scatter(
                x=avg_result.phase_bins,
                y=avg_result.max_amplitude,
                mode='lines',
                name='Max',
                line=dict(color='red', dash='dash'),
            ))

            # RMS amplitude
            phase_fig.add_trace(go.Scatter(
                x=avg_result.phase_bins,
                y=avg_result.rms_amplitude,
                mode='lines',
                name='RMS',
                line=dict(color='green', dash='dot'),
            ))

            # Highlight hotspots
            for hs in hotspots:
                phase_fig.add_vrect(
                    x0=hs.phase_range[0], x1=hs.phase_range[1],
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line_width=0,
                )

            phase_fig.update_layout(
                title='Synchronous Average by Phase',
                xaxis_title='Phase (degrees)',
                yaxis_title='Amplitude',
                xaxis=dict(range=[0, 360]),
                legend=dict(x=0.02, y=0.98),
            )

            store_data = {
                'mean_amplitude': avg_result.mean_amplitude.tolist(),
                'phase_bins': avg_result.phase_bins.tolist(),
                'num_hotspots': len(hotspots),
            }

            return results_div, heatmap_fig, phase_fig, store_data

        except Exception as e:
            import traceback
            error_div = html.Div([
                html.P(f"Error: {str(e)}", style={'color': 'red'}),
                html.Pre(traceback.format_exc(), style={'fontSize': '10px'}),
            ])
            empty_fig = go.Figure()
            return error_div, empty_fig, empty_fig, None

    # ========== Wavelet Tab Callbacks ==========

    @app.callback(
        [Output('wavelet-results', 'children'),
         Output('wavelet-bands-plot', 'figure'),
         Output('wavelet-detections-plot', 'figure'),
         Output('wavelet-results-store', 'data')],
        [Input('run-wavelet-btn', 'n_clicks')],
        [State('loaded-data-store', 'data'),
         State('wavelet-channel', 'value'),
         State('wavelet-type', 'value'),
         State('wavelet-bands', 'value'),
         State('wavelet-threshold', 'value')]
    )
    def run_wavelet_analysis(n_clicks, loaded_data, channel, wavelet_type, bands, threshold_pct):
        if not n_clicks or not loaded_data:
            raise PreventUpdate

        try:
            filepath = loaded_data['filepath']
            sample_rate = loaded_data['sample_rate']
            ac_freq = loaded_data['ac_frequency']

            # Load the signal
            loader = MatLoader(filepath)
            data = loader.load_channel(channel)
            signal = data['signal']

            # Compute kurtosis
            kurtosis_overall = compute_kurtosis(signal)

            # Compute phases
            samples_per_cycle = sample_rate / ac_freq
            phases = (np.arange(len(signal)) % samples_per_cycle) / samples_per_cycle * 360

            # Per-quadrant kurtosis
            quadrant_result = compute_kurtosis_per_quadrant(signal, phases)

            # Run DWT detection
            detector = DWTDetector(
                sample_rate=sample_rate,
                wavelet=wavelet_type,
                bands=bands,
                threshold_percentile=threshold_pct,
            )
            detection_result = detector.detect(signal, phases, ac_freq)

            # Extract waveforms
            extractor = WaveletExtractor(sample_rate=sample_rate)
            extraction_result = extractor.extract(signal, detection_result)

            # Create results display
            results_div = html.Div([
                html.H4("Wavelet Detection Results"),
                html.P([html.Strong("Wavelet: "), wavelet_type]),
                html.P([html.Strong("Bands: "), ", ".join(bands)]),
                html.P([html.Strong("Threshold Percentile: "), f"{threshold_pct}%"]),
                html.Hr(),
                html.H5("Kurtosis Analysis:"),
                html.P([html.Strong("Overall Kurtosis: "), f"{kurtosis_overall:.2f}"]),
                html.P([html.Strong("Per-Quadrant: "),
                       f"Q1={quadrant_result.quadrant_kurtosis['Q1']:.2f}, "
                       f"Q2={quadrant_result.quadrant_kurtosis['Q2']:.2f}, "
                       f"Q3={quadrant_result.quadrant_kurtosis['Q3']:.2f}, "
                       f"Q4={quadrant_result.quadrant_kurtosis['Q4']:.2f}"]),
                html.P([html.Strong("Max Quadrant: "),
                       f"{quadrant_result.max_quadrant} ({quadrant_result.max_kurtosis:.2f})"]),
                html.Hr(),
                html.H5("Detection Results:"),
                html.P([html.Strong("Total Events Detected: "), str(len(detection_result.events))]),
                html.P([html.Strong("Waveforms Extracted: "), str(extraction_result.num_waveforms)]),
                html.Ul([
                    html.Li(f"{band}: {detection_result.band_stats[band]['num_detections']} detections, "
                           f"threshold={detection_result.thresholds[band]:.2e}")
                    for band in bands if band in detection_result.band_stats
                ]),
            ])

            # Create bands plot
            import pywt
            coeffs = pywt.wavedec(signal, wavelet_type, level=3)

            bands_fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original Signal', 'D1 Coefficients', 'D2 Coefficients', 'D3 Coefficients'],
                vertical_spacing=0.08,
            )

            # Original signal (subsampled for performance)
            subsample = max(1, len(signal) // 10000)
            bands_fig.add_trace(
                go.Scatter(y=signal[::subsample], mode='lines', name='Signal', line=dict(width=0.5)),
                row=1, col=1
            )

            # D1, D2, D3 coefficients
            for i, (coeff, band) in enumerate([(coeffs[-1], 'D1'), (coeffs[-2], 'D2'), (coeffs[-3], 'D3')]):
                if band in bands:
                    threshold = detection_result.thresholds.get(band, 0)
                    bands_fig.add_trace(
                        go.Scatter(y=np.abs(coeff), mode='lines', name=f'{band}', line=dict(width=0.5)),
                        row=i+2, col=1
                    )
                    # Add threshold line
                    bands_fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=i+2, col=1)

            bands_fig.update_layout(height=800, showlegend=False, title_text="DWT Band Coefficients")

            # Create detections plot (phase resolved)
            detections_fig = go.Figure()

            # Plot all detections by band
            colors = {'D1': 'red', 'D2': 'blue', 'D3': 'green', 'D4': 'orange', 'D5': 'purple'}
            for band in bands:
                band_events = [e for e in detection_result.events if e.band == band]
                if band_events:
                    detections_fig.add_trace(go.Scatter(
                        x=[e.phase_degrees for e in band_events],
                        y=[e.amplitude for e in band_events],
                        mode='markers',
                        name=band,
                        marker=dict(color=colors.get(band, 'gray'), size=8),
                    ))

            detections_fig.update_layout(
                title='Detections by Phase and Band',
                xaxis_title='Phase (degrees)',
                yaxis_title='Wavelet Coefficient Amplitude',
                xaxis=dict(range=[0, 360]),
            )

            store_data = {
                'num_events': len(detection_result.events),
                'num_waveforms': extraction_result.num_waveforms,
                'kurtosis': kurtosis_overall,
            }

            return results_div, bands_fig, detections_fig, store_data

        except Exception as e:
            import traceback
            error_div = html.Div([
                html.P(f"Error: {str(e)}", style={'color': 'red'}),
                html.Pre(traceback.format_exc(), style={'fontSize': '10px'}),
            ])
            empty_fig = go.Figure()
            return error_div, empty_fig, empty_fig, None

    # ========== Comparison Tab Callbacks ==========

    @app.callback(
        [Output('comparison-results', 'children'),
         Output('comparison-plot', 'figure'),
         Output('waveform-comparison-plot', 'figure'),
         Output('comparison-detections-store', 'data')],
        [Input('run-comparison-btn', 'n_clicks')],
        [State('loaded-data-store', 'data'),
         State('comparison-channel', 'value'),
         State('trigger-threshold', 'value'),
         State('comparison-wavelet', 'value')]
    )
    def run_comparison(n_clicks, loaded_data, channel, trigger_threshold, wavelet_type):
        if not n_clicks or not loaded_data:
            raise PreventUpdate

        try:
            filepath = loaded_data['filepath']
            sample_rate = loaded_data['sample_rate']
            ac_freq = loaded_data['ac_frequency']

            # Load the signal
            loader = MatLoader(filepath)
            data = loader.load_channel(channel)
            signal = data['signal']

            # Compute phases
            samples_per_cycle = sample_rate / ac_freq
            phases = (np.arange(len(signal)) % samples_per_cycle) / samples_per_cycle * 360

            # Run trigger-based detection using stdev method with k_sigma based on threshold
            # Lower k_sigma = more sensitive (more detections)
            # refine_to_onset moves trigger back to where pulse actually starts
            trigger_detector = TriggerDetector(
                method='stdev',
                polarity='both',
                min_separation=100,
                k_sigma=trigger_threshold,  # Use threshold as k_sigma multiplier
                refine_to_onset=True,  # Adjust trigger backward to pulse onset
            )
            trigger_result = trigger_detector.detect(signal, sample_rate, ac_freq)

            # Run wavelet detection
            wavelet_detector = DWTDetector(
                sample_rate=sample_rate,
                wavelet=wavelet_type,
                bands=['D1', 'D2', 'D3'],
            )
            wavelet_result = wavelet_detector.detect(signal, phases, ac_freq)

            # Common extraction parameters for fair comparison
            pre_samples = 100
            post_samples = 400

            # Helper function to find peak and extract waveform centered on it
            def extract_peak_centered(detection_idx, pre=100, post=400, search_window=100):
                """Find peak within search window and extract waveform centered on it."""
                # Search for peak around detection point
                search_start = max(0, detection_idx - search_window // 4)
                search_end = min(len(signal), detection_idx + search_window)
                search_region = signal[search_start:search_end]

                # Find peak (max absolute value)
                abs_vals = np.abs(search_region)
                peak_offset = np.argmax(abs_vals)
                peak_idx = search_start + peak_offset

                # Extract waveform centered on peak
                start = max(0, peak_idx - pre)
                end = min(len(signal), peak_idx + post)
                waveform = signal[start:end]

                # Pad if necessary to maintain consistent length
                total_len = pre + post
                if len(waveform) < total_len:
                    padded = np.zeros(total_len)
                    offset = pre - (peak_idx - start)
                    padded[offset:offset+len(waveform)] = waveform
                    waveform = padded

                return waveform, peak_idx

            # Extract waveforms using trigger method (with peak centering)
            trigger_idx_list = list(trigger_result.triggers) if len(trigger_result.triggers) > 0 else []
            trigger_waveform_list = []
            trigger_peak_indices = []
            for idx in trigger_idx_list:
                wfm, peak_idx = extract_peak_centered(idx, pre_samples, post_samples)
                trigger_waveform_list.append(wfm)
                trigger_peak_indices.append(peak_idx)

            # Extract waveforms using wavelet method (same window as trigger, peak centered)
            wavelet_waveform_list = []
            wavelet_peak_indices = []
            for e in wavelet_result.events:
                wfm, peak_idx = extract_peak_centered(e.sample_index, pre_samples, post_samples)
                wavelet_waveform_list.append(wfm)
                wavelet_peak_indices.append(peak_idx)

            # Compare detection locations and categorize
            trigger_indices = set(trigger_idx_list)
            wavelet_indices = set(e.sample_index for e in wavelet_result.events)

            # Find matches (within tolerance) and categorize each detection
            tolerance = 100
            matched_trigger = set()  # Trigger indices that have a wavelet match
            matched_wavelet = set()  # Wavelet indices that have a trigger match

            for t_idx in trigger_idx_list:
                for w_idx, e in zip([ev.sample_index for ev in wavelet_result.events], wavelet_result.events):
                    if abs(t_idx - w_idx) <= tolerance:
                        matched_trigger.add(t_idx)
                        matched_wavelet.add(w_idx)
                        break

            trigger_only = [idx for idx in trigger_idx_list if idx not in matched_trigger]
            wavelet_only = [(e.sample_index, e) for e in wavelet_result.events if e.sample_index not in matched_wavelet]
            matched_pairs = [(t_idx, t_peak) for t_idx, t_peak in zip(trigger_idx_list, trigger_peak_indices) if t_idx in matched_trigger]

            # Results - count per band
            num_trigger_wfms = len(trigger_waveform_list)
            num_wavelet_wfms = len(wavelet_waveform_list)
            d1_count = sum(1 for e in wavelet_result.events if e.band == 'D1')
            d2_count = sum(1 for e in wavelet_result.events if e.band == 'D2')
            d3_count = sum(1 for e in wavelet_result.events if e.band == 'D3')

            results_div = html.Div([
                html.H4("Detection Comparison"),
                html.Hr(),
                html.Div([
                    html.Div([
                        html.H5("Trigger-Based Detection"),
                        html.P([html.Strong("Method: "), "stdev"]),
                        html.P([html.Strong("K-Sigma: "), f"{trigger_threshold}"]),
                        html.P([html.Strong("Threshold Used: "), f"{trigger_result.threshold:.2e}"]),
                        html.P([html.Strong("Events Detected: "), str(len(trigger_indices))]),
                        html.P([html.Strong("Waveforms Extracted: "), str(num_trigger_wfms)]),
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([
                        html.H5("Wavelet Detection"),
                        html.P([html.Strong("Wavelet: "), wavelet_type]),
                        html.P([html.Strong("Total Events: "), str(len(wavelet_indices))]),
                        html.P([html.Strong("  D1 (fast): "), f"{d1_count} events"]),
                        html.P([html.Strong("  D2 (medium): "), f"{d2_count} events"]),
                        html.P([html.Strong("  D3 (slow): "), f"{d3_count} events"]),
                        html.P([html.Strong("Waveforms Extracted: "), str(num_wavelet_wfms)]),
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'}),
                ]),
                html.Hr(),
                html.H5("Comparison Metrics"),
                html.P([html.Strong("Matched (▲ green): "), f"{len(matched_trigger)} detections"]),
                html.P([html.Strong("Trigger-only (● red): "), f"{len(trigger_only)} detections"]),
                html.P([html.Strong("Wavelet-only: "), f"{len(wavelet_only)} total"]),
                html.P([html.Strong("  ◆ D1 (cyan): "), f"{sum(1 for _, e in wavelet_only if e.band == 'D1')}"]),
                html.P([html.Strong("  ✕ D2 (blue): "), f"{sum(1 for _, e in wavelet_only if e.band == 'D2')}"]),
                html.P([html.Strong("  ■ D3 (purple): "), f"{sum(1 for _, e in wavelet_only if e.band == 'D3')}"]),
            ])

            # Detection comparison plot - Phase-resolved (sinusoidal format)
            comparison_fig = go.Figure()

            # Helper function to get peak amplitude within window around detection
            def get_peak_amplitude(idx, window=100):
                start = max(0, idx - window // 4)
                end = min(len(signal), idx + window)
                window_data = signal[start:end]
                abs_vals = np.abs(window_data)
                peak_idx = np.argmax(abs_vals)
                return window_data[peak_idx]

            # Collect all amplitudes for sine wave scaling
            all_amps = []
            for idx in trigger_idx_list:
                all_amps.append(get_peak_amplitude(idx))
            for e in wavelet_result.events:
                all_amps.append(get_peak_amplitude(e.sample_index))

            if all_amps:
                amp_range = max(abs(min(all_amps)), abs(max(all_amps)))
                sine_scale = amp_range * 0.3
            else:
                sine_scale = 1.0

            # Add reference sine wave
            phase_axis = np.linspace(0, 360, 361)
            sine_wave = sine_scale * np.sin(np.radians(phase_axis))
            comparison_fig.add_trace(go.Scatter(
                x=phase_axis,
                y=sine_wave,
                mode='lines',
                name='AC Reference',
                line=dict(color='lightgray', width=2, dash='dash'),
            ))

            # Plot MATCHED detections (green triangles) - these are the same event
            if matched_pairs:
                matched_phases = [phases[idx] for idx, _ in matched_pairs]
                matched_amps = [get_peak_amplitude(idx) for idx, _ in matched_pairs]
                comparison_fig.add_trace(go.Scatter(
                    x=matched_phases,
                    y=matched_amps,
                    mode='markers',
                    name='Matched (Both)',
                    marker=dict(color='green', size=6, symbol='triangle-up'),
                ))

            # Plot TRIGGER-ONLY detections (red circles)
            if trigger_only:
                trigger_only_phases = [phases[idx] for idx in trigger_only]
                trigger_only_amps = [get_peak_amplitude(idx) for idx in trigger_only]
                comparison_fig.add_trace(go.Scatter(
                    x=trigger_only_phases,
                    y=trigger_only_amps,
                    mode='markers',
                    name='Trigger Only',
                    marker=dict(color='red', size=4, symbol='circle'),
                ))

            # Split wavelet-only by band
            wavelet_only_d1 = [(idx, e) for idx, e in wavelet_only if e.band == 'D1']
            wavelet_only_d2 = [(idx, e) for idx, e in wavelet_only if e.band == 'D2']
            wavelet_only_d3 = [(idx, e) for idx, e in wavelet_only if e.band == 'D3']

            # Plot WAVELET-ONLY D1 detections (cyan diamond)
            if wavelet_only_d1:
                d1_phases = [phases[idx] for idx, _ in wavelet_only_d1]
                d1_amps = [get_peak_amplitude(idx) for idx, _ in wavelet_only_d1]
                comparison_fig.add_trace(go.Scatter(
                    x=d1_phases,
                    y=d1_amps,
                    mode='markers',
                    name='Wavelet D1 Only',
                    marker=dict(color='cyan', size=5, symbol='diamond'),
                ))

            # Plot WAVELET-ONLY D2 detections (blue X)
            if wavelet_only_d2:
                d2_phases = [phases[idx] for idx, _ in wavelet_only_d2]
                d2_amps = [get_peak_amplitude(idx) for idx, _ in wavelet_only_d2]
                comparison_fig.add_trace(go.Scatter(
                    x=d2_phases,
                    y=d2_amps,
                    mode='markers',
                    name='Wavelet D2 Only',
                    marker=dict(color='blue', size=5, symbol='x'),
                ))

            # Plot WAVELET-ONLY D3 detections (purple square)
            if wavelet_only_d3:
                d3_phases = [phases[idx] for idx, _ in wavelet_only_d3]
                d3_amps = [get_peak_amplitude(idx) for idx, _ in wavelet_only_d3]
                comparison_fig.add_trace(go.Scatter(
                    x=d3_phases,
                    y=d3_amps,
                    mode='markers',
                    name='Wavelet D3 Only',
                    marker=dict(color='purple', size=5, symbol='square'),
                ))

            # Add vertical lines at quadrant boundaries
            for phase in [90, 180, 270]:
                comparison_fig.add_vline(x=phase, line_dash="dot", line_color="lightgray", opacity=0.5)

            comparison_fig.update_layout(
                title='Phase-Resolved Detection Comparison (Trigger vs Wavelet)',
                xaxis_title='Phase (degrees)',
                yaxis_title='Amplitude',
                xaxis=dict(range=[0, 360], dtick=45),
            )

            # Waveform comparison plot - split by detection category
            # Group wavelet waveforms by band for display
            d1_indices = [i for i, e in enumerate(wavelet_result.events) if e.band == 'D1']
            d2_indices = [i for i, e in enumerate(wavelet_result.events) if e.band == 'D2']
            d3_indices = [i for i, e in enumerate(wavelet_result.events) if e.band == 'D3']

            waveform_fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=[
                    f'Trigger-Extracted ({num_trigger_wfms} waveforms) - peak-centered',
                    f'Wavelet D1 ({len(d1_indices)} waveforms) - peak-centered, same window',
                    f'Wavelet D2 ({len(d2_indices)} waveforms) - peak-centered, same window',
                    f'Wavelet D3 ({len(d3_indices)} waveforms) - peak-centered, same window',
                ],
                vertical_spacing=0.08,
            )

            # Show first N waveforms from each method
            max_waveforms = 20

            # Trigger waveforms (from our peak-centered extraction)
            for i, wfm in enumerate(trigger_waveform_list[:max_waveforms]):
                waveform_fig.add_trace(
                    go.Scatter(y=wfm, mode='lines', line=dict(width=0.5),
                              showlegend=False, opacity=0.5),
                    row=1, col=1
                )

            # Wavelet D1 waveforms (from peak-centered extraction)
            for i in d1_indices[:max_waveforms]:
                waveform_fig.add_trace(
                    go.Scatter(y=wavelet_waveform_list[i], mode='lines', line=dict(width=0.5, color='red'),
                              showlegend=False, opacity=0.5),
                    row=2, col=1
                )

            # Wavelet D2 waveforms
            for i in d2_indices[:max_waveforms]:
                waveform_fig.add_trace(
                    go.Scatter(y=wavelet_waveform_list[i], mode='lines', line=dict(width=0.5, color='blue'),
                              showlegend=False, opacity=0.5),
                    row=3, col=1
                )

            # Wavelet D3 waveforms
            for i in d3_indices[:max_waveforms]:
                waveform_fig.add_trace(
                    go.Scatter(y=wavelet_waveform_list[i], mode='lines', line=dict(width=0.5, color='green'),
                              showlegend=False, opacity=0.5),
                    row=4, col=1
                )

            waveform_fig.update_layout(height=900, title_text="Extracted Waveform Comparison (All Peak-Centered, Same Window)")

            # Build detection store for click handling
            # Store detection info categorized by matched/trigger-only/wavelet-only (split by band)
            # Curve order: 0=AC ref, 1=Matched, 2=Trigger-only, 3=D1-only, 4=D2-only, 5=D3-only
            detection_store = {
                'filepath': filepath,
                'channel': channel,
                'sample_rate': sample_rate,
                'pre_samples': pre_samples,
                'post_samples': post_samples,
                'matched_detections': [
                    {'index': int(idx), 'phase': float(phases[idx]), 'type': 'matched'}
                    for idx, _ in matched_pairs
                ],
                'trigger_only_detections': [
                    {'index': int(idx), 'phase': float(phases[idx]), 'type': 'trigger_only'}
                    for idx in trigger_only
                ],
                'wavelet_d1_detections': [
                    {'index': int(idx), 'phase': float(phases[idx]), 'band': 'D1', 'type': 'wavelet_d1'}
                    for idx, e in wavelet_only_d1
                ],
                'wavelet_d2_detections': [
                    {'index': int(idx), 'phase': float(phases[idx]), 'band': 'D2', 'type': 'wavelet_d2'}
                    for idx, e in wavelet_only_d2
                ],
                'wavelet_d3_detections': [
                    {'index': int(idx), 'phase': float(phases[idx]), 'band': 'D3', 'type': 'wavelet_d3'}
                    for idx, e in wavelet_only_d3
                ],
            }

            return results_div, comparison_fig, waveform_fig, detection_store

        except Exception as e:
            import traceback
            error_div = html.Div([
                html.P(f"Error: {str(e)}", style={'color': 'red'}),
                html.Pre(traceback.format_exc(), style={'fontSize': '10px'}),
            ])
            empty_fig = go.Figure()
            return error_div, empty_fig, empty_fig, None

    # Callback for clicking on detection points
    @app.callback(
        [Output('clicked-waveform-plot', 'figure'),
         Output('clicked-waveform-info', 'children')],
        [Input('comparison-plot', 'clickData')],
        [State('comparison-detections-store', 'data'),
         State('loaded-data-store', 'data')]
    )
    def display_clicked_waveform(clickData, detection_store, loaded_data):
        if not clickData or not detection_store or not loaded_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Click a detection point to view waveform",
                xaxis_title="Sample",
                yaxis_title="Amplitude"
            )
            return empty_fig, "Click a detection point in the plot above to view its waveform."

        try:
            # Get click info
            point = clickData['points'][0]
            curve_num = point.get('curveNumber', 0)
            point_index = point.get('pointIndex', 0)
            clicked_phase = point.get('x', 0)
            clicked_amp = point.get('y', 0)

            # Curve order: 0=AC ref, 1=Matched, 2=Trigger-only, 3=D1-only, 4=D2-only, 5=D3-only
            if curve_num == 0:
                # Clicked on AC reference line, ignore
                return no_update, no_update

            # Map curve number to detection category
            if curve_num == 1:
                detections = detection_store.get('matched_detections', [])
                detection_type = 'Matched (Both Methods)'
                color = 'green'
            elif curve_num == 2:
                detections = detection_store.get('trigger_only_detections', [])
                detection_type = 'Trigger Only'
                color = 'red'
            elif curve_num == 3:
                detections = detection_store.get('wavelet_d1_detections', [])
                detection_type = 'Wavelet D1 Only'
                color = 'cyan'
            elif curve_num == 4:
                detections = detection_store.get('wavelet_d2_detections', [])
                detection_type = 'Wavelet D2 Only'
                color = 'blue'
            elif curve_num == 5:
                detections = detection_store.get('wavelet_d3_detections', [])
                detection_type = 'Wavelet D3 Only'
                color = 'purple'
            else:
                return no_update, "Unknown curve"

            if point_index >= len(detections):
                return no_update, f"Detection index {point_index} out of range"

            detection = detections[point_index]
            sample_index = detection['index']
            phase = detection['phase']
            band = detection.get('band', 'N/A')

            # Load signal and extract waveform (peak-centered)
            filepath = detection_store['filepath']
            channel = detection_store.get('channel', 'Ch1')

            loader = MatLoader(filepath)
            data = loader.load_channel(channel)
            signal = data['signal']
            sample_rate = detection_store['sample_rate']
            pre_samples = detection_store.get('pre_samples', 100)
            post_samples = detection_store.get('post_samples', 400)

            # Find peak and extract centered waveform
            search_start = max(0, sample_index - 25)
            search_end = min(len(signal), sample_index + 100)
            search_region = signal[search_start:search_end]
            peak_offset = np.argmax(np.abs(search_region))
            peak_idx = search_start + peak_offset

            start = max(0, peak_idx - pre_samples)
            end = min(len(signal), peak_idx + post_samples)
            waveform = signal[start:end]

            # ========== FEATURE EXTRACTION ==========
            # These features help understand why wavelet might miss a trigger detection

            # 1. Basic amplitude metrics
            peak_amp = np.max(np.abs(waveform))
            rms_amp = np.sqrt(np.mean(waveform**2))
            crest_factor = peak_amp / rms_amp if rms_amp > 0 else 0

            # 2. Pulse width (FWHM - Full Width at Half Maximum)
            abs_wfm = np.abs(waveform)
            half_max = peak_amp / 2
            above_half = abs_wfm >= half_max
            if np.any(above_half):
                first_above = np.argmax(above_half)
                last_above = len(above_half) - 1 - np.argmax(above_half[::-1])
                fwhm_samples = last_above - first_above
                fwhm_us = fwhm_samples / sample_rate * 1e6
            else:
                fwhm_samples = 0
                fwhm_us = 0

            # 3. Rise time (10% to 90% of peak)
            thresh_10 = peak_amp * 0.1
            thresh_90 = peak_amp * 0.9
            peak_idx_local = np.argmax(abs_wfm)
            # Search backward from peak for 10% crossing
            rise_start = 0
            for i in range(peak_idx_local, -1, -1):
                if abs_wfm[i] <= thresh_10:
                    rise_start = i
                    break
            # Search backward from peak for 90% crossing
            rise_end = peak_idx_local
            for i in range(peak_idx_local, -1, -1):
                if abs_wfm[i] <= thresh_90:
                    rise_end = i + 1
                    break
            rise_time_samples = rise_end - rise_start
            rise_time_ns = rise_time_samples / sample_rate * 1e9

            # 4. Dominant frequency (FFT-based)
            try:
                # Use FFT to find dominant frequency
                n_fft = len(waveform)
                fft_result = np.fft.rfft(waveform)
                fft_freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
                fft_magnitude = np.abs(fft_result)
                # Ignore DC component
                fft_magnitude[0] = 0
                dominant_freq_idx = np.argmax(fft_magnitude)
                dominant_freq_mhz = fft_freqs[dominant_freq_idx] / 1e6
            except:
                dominant_freq_mhz = 0

            # 5. Band energy distribution (what % of energy in D1, D2, D3 frequency ranges)
            # D1: highest freq (fs/4 to fs/2), D2: (fs/8 to fs/4), D3: (fs/16 to fs/8)
            try:
                total_energy = np.sum(fft_magnitude**2)
                if total_energy > 0:
                    # Define frequency bands (approximate for 125 MSPS)
                    # D1: 31.25-62.5 MHz, D2: 15.625-31.25 MHz, D3: 7.8125-15.625 MHz
                    d1_mask = (fft_freqs >= sample_rate/4) & (fft_freqs < sample_rate/2)
                    d2_mask = (fft_freqs >= sample_rate/8) & (fft_freqs < sample_rate/4)
                    d3_mask = (fft_freqs >= sample_rate/16) & (fft_freqs < sample_rate/8)
                    low_mask = fft_freqs < sample_rate/16

                    d1_energy_pct = np.sum(fft_magnitude[d1_mask]**2) / total_energy * 100
                    d2_energy_pct = np.sum(fft_magnitude[d2_mask]**2) / total_energy * 100
                    d3_energy_pct = np.sum(fft_magnitude[d3_mask]**2) / total_energy * 100
                    low_energy_pct = np.sum(fft_magnitude[low_mask]**2) / total_energy * 100
                else:
                    d1_energy_pct = d2_energy_pct = d3_energy_pct = low_energy_pct = 0
            except:
                d1_energy_pct = d2_energy_pct = d3_energy_pct = low_energy_pct = 0

            # 6. Actual wavelet coefficients at this location
            try:
                import pywt
                # Compute DWT of waveform
                coeffs = pywt.wavedec(waveform, 'db4', level=3)
                # Get max coefficient in each band
                d1_coeff_max = np.max(np.abs(coeffs[-1])) if len(coeffs) > 1 else 0
                d2_coeff_max = np.max(np.abs(coeffs[-2])) if len(coeffs) > 2 else 0
                d3_coeff_max = np.max(np.abs(coeffs[-3])) if len(coeffs) > 3 else 0
            except:
                d1_coeff_max = d2_coeff_max = d3_coeff_max = 0

            # 7. Kurtosis (measure of impulsiveness)
            try:
                from scipy.stats import kurtosis as scipy_kurtosis
                wfm_kurtosis = scipy_kurtosis(waveform, fisher=True)
            except:
                # Manual calculation if scipy not available
                mean_val = np.mean(waveform)
                std_val = np.std(waveform)
                if std_val > 0:
                    wfm_kurtosis = np.mean(((waveform - mean_val) / std_val)**4) - 3
                else:
                    wfm_kurtosis = 0

            # 8. SNR estimate (peak vs noise floor)
            try:
                # Estimate noise from edges of waveform (before pulse)
                noise_region = waveform[:pre_samples//2]
                noise_std = np.std(noise_region) if len(noise_region) > 10 else np.std(waveform)
                snr_db = 20 * np.log10(peak_amp / noise_std) if noise_std > 0 else 0
            except:
                snr_db = 0

            # Create time axis in microseconds (centered on peak)
            time_us = (np.arange(len(waveform)) - pre_samples) / sample_rate * 1e6

            # Create waveform plot
            wfm_fig = go.Figure()
            wfm_fig.add_trace(go.Scatter(
                x=time_us,
                y=waveform,
                mode='lines',
                name='Waveform',
                line=dict(color=color, width=1.5),
            ))

            # Add vertical line at peak point
            wfm_fig.add_vline(x=0, line_dash="dash", line_color="gray",
                             annotation_text="Peak", annotation_position="top")

            wfm_fig.update_layout(
                title=f'{detection_type} - Phase: {phase:.1f}°',
                xaxis_title='Time (µs)',
                yaxis_title='Amplitude',
            )

            # Build comprehensive info text with features
            info_items = [
                html.Strong(f"{detection_type}"),
                html.P(f"Sample Index: {sample_index:,}"),
                html.P(f"Phase: {phase:.1f}°"),
            ]
            if band != 'N/A':
                info_items.append(html.P(f"Band: {band}"))

            # Add feature analysis section
            info_items.extend([
                html.Hr(),
                html.Strong("Pulse Characteristics:"),
                html.P(f"Peak Amplitude: {peak_amp:.4e}"),
                html.P(f"Pulse Width (FWHM): {fwhm_us:.2f} µs ({fwhm_samples} samples)"),
                html.P(f"Rise Time (10-90%): {rise_time_ns:.1f} ns"),
                html.P(f"Dominant Frequency: {dominant_freq_mhz:.2f} MHz"),
                html.Hr(),
                html.Strong("Quality Metrics:"),
                html.P(f"Crest Factor: {crest_factor:.1f}"),
                html.P(f"Kurtosis: {wfm_kurtosis:.1f}"),
                html.P(f"SNR: {snr_db:.1f} dB"),
                html.Hr(),
                html.Strong("Band Energy Distribution:"),
                html.P(f"D1 (high freq): {d1_energy_pct:.1f}%"),
                html.P(f"D2 (medium): {d2_energy_pct:.1f}%"),
                html.P(f"D3 (low freq): {d3_energy_pct:.1f}%"),
                html.P(f"Below D3: {low_energy_pct:.1f}%"),
                html.Hr(),
                html.Strong("Wavelet Coefficients (max):"),
                html.P(f"D1 coeff: {d1_coeff_max:.4e}"),
                html.P(f"D2 coeff: {d2_coeff_max:.4e}"),
                html.P(f"D3 coeff: {d3_coeff_max:.4e}"),
            ])

            # Add explanation for trigger-only detections
            if detection_type == 'Trigger Only':
                info_items.extend([
                    html.Hr(),
                    html.Strong("Why Wavelet Missed:", style={'color': 'red'}),
                ])
                # Analyze why wavelet might have missed this
                reasons = []
                if d1_energy_pct < 10 and d2_energy_pct < 10 and d3_energy_pct < 10:
                    reasons.append("Most energy below D3 band (too slow)")
                if fwhm_us > 5:
                    reasons.append(f"Pulse too wide ({fwhm_us:.1f}µs > 5µs)")
                if crest_factor < 3:
                    reasons.append(f"Low crest factor ({crest_factor:.1f} < 3)")
                if snr_db < 10:
                    reasons.append(f"Low SNR ({snr_db:.1f}dB < 10dB)")
                if max(d1_coeff_max, d2_coeff_max, d3_coeff_max) < peak_amp * 0.1:
                    reasons.append("Wavelet coefficients below threshold")

                if reasons:
                    for reason in reasons:
                        info_items.append(html.P(f"• {reason}", style={'color': 'red'}))
                else:
                    info_items.append(html.P("• Coefficients likely just below detection threshold"))

            info_text = html.Div(info_items)

            return wfm_fig, info_text

        except Exception as e:
            import traceback
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5,
                                    xref="paper", yref="paper", showarrow=False)
            return error_fig, f"Error loading waveform: {str(e)}"

    return app


def create_syncavg_tab(loaded_data):
    """Create the SyncAvg analysis tab content."""
    channels = loaded_data.get('channels', []) if loaded_data else []

    return html.Div([
        html.H3("Synchronous Averaging Analysis"),
        html.P("Analyze phase-locked patterns in the raw data using synchronous averaging."),

        # Controls
        html.Div([
            html.Div([
                html.Label("Signal Channel:"),
                dcc.Dropdown(
                    id='syncavg-channel',
                    options=[{'label': ch, 'value': ch} for ch in channels],
                    value=channels[0] if channels else None,
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Reference Channel (for phase):"),
                dcc.Dropdown(
                    id='syncavg-ref-channel',
                    options=[{'label': 'None (synthetic)', 'value': ''}] +
                           [{'label': ch, 'value': ch} for ch in channels],
                    value='',
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Phase Method:"),
                dcc.Dropdown(
                    id='syncavg-phase-method',
                    options=[
                        {'label': 'Zero Crossing', 'value': 'zero_crossing'},
                        {'label': 'Hilbert Transform', 'value': 'hilbert'},
                        {'label': 'Sine Fit', 'value': 'fit'},
                    ],
                    value='zero_crossing',
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Number of Phase Bins:"),
                dcc.Input(
                    id='syncavg-num-bins',
                    type='number',
                    value=360,
                    min=36,
                    max=3600,
                    step=36,
                ),
            ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Button("Run SyncAvg", id='run-syncavg-btn', n_clicks=0,
                           style={'marginTop': '20px', 'padding': '10px 20px'}),
            ], style={'width': '15%', 'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),

        # Results
        html.Div(id='syncavg-results', style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'}),

        # Plots
        html.Div([
            html.Div([
                html.H4("Phase Heat Map"),
                dcc.Graph(id='syncavg-heatmap', figure=go.Figure()),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H4("Synchronous Average"),
                dcc.Graph(id='syncavg-phase-plot', figure=go.Figure()),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
        ]),
    ])


def create_wavelet_tab(loaded_data):
    """Create the Wavelet analysis tab content."""
    channels = loaded_data.get('channels', []) if loaded_data else []

    wavelet_options = [{'label': f"{k} - {v['description']}", 'value': k} for k, v in WAVELETS.items()]

    return html.Div([
        html.H3("Wavelet Detection Analysis"),
        html.P("Analyze raw data using DWT-based multi-band detection."),

        # Controls
        html.Div([
            html.Div([
                html.Label("Signal Channel:"),
                dcc.Dropdown(
                    id='wavelet-channel',
                    options=[{'label': ch, 'value': ch} for ch in channels],
                    value=channels[0] if channels else None,
                ),
            ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Wavelet Type:"),
                dcc.Dropdown(
                    id='wavelet-type',
                    options=wavelet_options,
                    value=DEFAULT_WAVELET,
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Bands:"),
                dcc.Checklist(
                    id='wavelet-bands',
                    options=[
                        {'label': 'D1', 'value': 'D1'},
                        {'label': 'D2', 'value': 'D2'},
                        {'label': 'D3', 'value': 'D3'},
                    ],
                    value=['D1', 'D2', 'D3'],
                    inline=True,
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Threshold Percentile:"),
                dcc.Input(
                    id='wavelet-threshold',
                    type='number',
                    value=99.5,
                    min=90,
                    max=99.99,
                    step=0.1,
                ),
            ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Button("Run Wavelet", id='run-wavelet-btn', n_clicks=0,
                           style={'marginTop': '20px', 'padding': '10px 20px'}),
            ], style={'width': '10%', 'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),

        # Results
        html.Div(id='wavelet-results', style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'}),

        # Plots
        html.Div([
            html.H4("DWT Band Coefficients"),
            dcc.Graph(id='wavelet-bands-plot', figure=go.Figure()),
        ]),

        html.Div([
            html.H4("Detections by Phase"),
            dcc.Graph(id='wavelet-detections-plot', figure=go.Figure()),
        ]),
    ])


def create_comparison_tab(loaded_data):
    """Create the Comparison tab content."""
    channels = loaded_data.get('channels', []) if loaded_data else []

    wavelet_options = [{'label': k, 'value': k} for k in WAVELETS.keys()]

    return html.Div([
        html.H3("Detection Method Comparison"),
        html.P("Compare trigger-based detection with wavelet-based detection."),

        # Controls
        html.Div([
            html.Div([
                html.Label("Signal Channel:"),
                dcc.Dropdown(
                    id='comparison-channel',
                    options=[{'label': ch, 'value': ch} for ch in channels],
                    value=channels[0] if channels else None,
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("K-Sigma (trigger sensitivity):"),
                dcc.Input(
                    id='trigger-threshold',
                    type='number',
                    value=5.0,
                    min=1.0,
                    max=20.0,
                    step=0.5,
                ),
            ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Wavelet Type:"),
                dcc.Dropdown(
                    id='comparison-wavelet',
                    options=wavelet_options,
                    value=DEFAULT_WAVELET,
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Button("Run Comparison", id='run-comparison-btn', n_clicks=0,
                           style={'marginTop': '20px', 'padding': '10px 20px'}),
            ], style={'width': '15%', 'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),

        # Results
        html.Div(id='comparison-results', style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'}),

        # Plots
        html.Div([
            html.H4("Detection Locations (click a point to view waveform)"),
            dcc.Graph(id='comparison-plot', figure=go.Figure()),
        ]),

        # Clicked waveform display
        html.Div([
            html.H4("Selected Waveform"),
            html.Div(id='clicked-waveform-info', style={'marginBottom': '10px'}),
            dcc.Graph(id='clicked-waveform-plot', figure=go.Figure()),
        ], style={'backgroundColor': '#f0f8ff', 'padding': '10px', 'marginBottom': '20px'}),

        # Store for detection data (to look up waveforms on click)
        dcc.Store(id='comparison-detections-store'),

        html.Div([
            html.H4("Waveform Comparison"),
            dcc.Graph(id='waveform-comparison-plot', figure=go.Figure()),
        ]),
    ])


def main():
    parser = argparse.ArgumentParser(description='Preprocessing Test GUI')
    parser.add_argument('--port', type=int, default=8051, help='Port to run the server on')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Directory containing IEEE .mat files')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    if not DASH_AVAILABLE:
        print("Error: Dash is required. Install with: pip install dash plotly")
        sys.exit(1)

    if not PREPROCESSING_AVAILABLE:
        print("Error: Preprocessing modules not available")
        sys.exit(1)

    print(f"Starting Preprocessing Test GUI on port {args.port}")
    print(f"Data directory: {args.data_dir}")

    app = create_app(data_dir=args.data_dir)
    app.run(debug=args.debug, port=args.port)


if __name__ == '__main__':
    main()
