"""
syncAvg - Synchronous Averaging for PD Detection

Performs phase-locked averaging of raw data across multiple AC cycles
to reveal consistent PD activity patterns buried in noise.

Key Features:
- Interpolates reference signal to get phase for every sample
- Builds phase-binned average of signal amplitude over many cycles
- Reveals consistent PD hotspots regardless of random noise
- Provides phase regions for deeper monitoring

Usage:
    from pre_middleware.syncAvg import PhaseInterpolator, SyncAverager
    from pre_middleware.loaders import MatLoader

    # Load raw data with reference channel
    loader = MatLoader('data.mat')
    data = loader.load(channel='Ch1', include_reference=True)

    # Get phase for every sample
    interpolator = PhaseInterpolator(ac_frequency=60.0)
    phases = interpolator.interpolate(reference_signal, sample_rate)

    # Build synchronous average
    averager = SyncAverager(num_bins=360)
    result = averager.compute(signal, phases)

    # Find phase regions with elevated activity
    hotspots = averager.find_hotspots(result, threshold_sigma=3.0)
"""

from .phase_interpolator import PhaseInterpolator, interpolate_phase
from .sync_averager import SyncAverager, SyncAverageResult

__all__ = [
    'PhaseInterpolator',
    'interpolate_phase',
    'SyncAverager',
    'SyncAverageResult',
]
