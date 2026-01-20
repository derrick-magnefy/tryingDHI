"""
DHI (Diagnostic Health Index) Calculation Module

Based on MATLAB reference implementation V1.1 with bug fixes.

Components:
- compute_rpd: PD-based reliability calculation
- compute_rgas: DGA-based reliability calculation
- compute_dhi: Global DHI calculation

IMPORTANT - Sign Logic Fix:
    When BOTH A < A0 AND RR < RR0 (PD below baseline), DR is set to 1.0.
    This fixes the original MATLAB bug where lower PD caused higher DR.

IMPORTANT - Baseline Values:
    A0, RR0 should be the FIRST measurement values from the transformer.
    Do not use the Python defaults (0.02 V, 5000 pulses/sec) in production.

IMPORTANT - T0 (Expected Asset Lifespan):
    T0_seconds represents the expected lifespan of the transformer/asset.
    This determines cumulative damage rate: damage_per_sample = dt/T0 × S

    T0 constants by expected lifespan:
    - T0_20_YEARS: For 20-year lifespan assets
    - T0_30_YEARS: For 30-year lifespan assets (typical MV)
    - T0_40_YEARS: For 40-year lifespan assets (default, typical HV)
    - T0_50_YEARS: For 50-year lifespan assets

    Example: Daily sampling on 40-year transformer
        dt/T0 = 1 day / 40 years ≈ 0.0000685 per sample

Usage:
    from dhi import compute_dhi_full, RpdConfig, RgasConfig, T0_40_YEARS

    # Full DHI with PD and DGA
    rpd_config = RpdConfig(
        A0=0.04,          # Use first measurement!
        RR0=5000,         # Use first measurement!
        mode='cumulative',
        T0_seconds=T0_40_YEARS,  # 40-year expected lifespan
    )
    result, state = compute_dhi_full(
        A=0.08, RR=8000, Pk=[0.3, 0.7],
        rpd_config=rpd_config,
        gas_names=['H2', 'CH4', 'C2H4', 'C2H2', 'CO', 'CO2', 'C2H6'],
        conc=[150, 120, 80, 0, 500, 3500, 90],
    )
    print(f"DHI: {result.DHI:.3f}, State: {result.health_state}")
"""

# Rpd (PD Reliability)
from .compute_rpd import (
    RpdConfig,
    RpdState,
    RpdResult,
    compute_rpd,
    compute_rpd_simple,
    compute_rpd_from_pd_types,
    compute_phase_severity,
    # T0 constants (asset lifespan)
    T0_20_YEARS,
    T0_30_YEARS,
    T0_40_YEARS,
    T0_50_YEARS,
    T0_DEFAULT,
    # Legacy T0 constants (for backwards compatibility)
    T0_HOURLY,
    T0_15MIN,
    T0_DAILY,
    T0_WEEKLY,
)

# Rgas (DGA Reliability)
from .compute_rgas import (
    RgasConfig,
    RgasResult,
    compute_rgas,
    compute_rgas_from_dict,
    compute_rgas_legacy,
    score_discrete,
    score_continuous,
    DEFAULT_BC,
    DEFAULT_BR,
    DEFAULT_WEIGHTS,
)

# Global DHI
from .compute_dhi import (
    DHIResult,
    EMHealthScore,
    PipelineDHIOutput,
    compute_dhi,
    compute_dhi_full,
    compute_dhi_from_pd_types,
    compute_dhi_from_pipeline_result,
    compute_dhi_metrics_standalone,
    compute_em_health_score,
    get_health_state,
    get_maintenance_state,
    get_worst_health_state,
    MAINTENANCE_THRESHOLDS,
    HEALTH_STATE_SEVERITY,
    # Multi-sensor (3-phase + tank)
    SENSOR_NAMES,
    SensorDHIInput,
    SensorDHIResult,
    MultiSensorDHIResult,
    compute_sensor_rpd,
    compute_multisensor_dhi,
)

__all__ = [
    # Rpd
    'RpdConfig',
    'RpdState',
    'RpdResult',
    'compute_rpd',
    'compute_rpd_simple',
    'compute_rpd_from_pd_types',
    'compute_phase_severity',
    # T0 constants (asset lifespan)
    'T0_20_YEARS',
    'T0_30_YEARS',
    'T0_40_YEARS',
    'T0_50_YEARS',
    'T0_DEFAULT',
    # Legacy T0 constants (for backwards compatibility)
    'T0_HOURLY',
    'T0_15MIN',
    'T0_DAILY',
    'T0_WEEKLY',

    # Rgas
    'RgasConfig',
    'RgasResult',
    'compute_rgas',
    'compute_rgas_from_dict',
    'compute_rgas_legacy',
    'score_discrete',
    'score_continuous',
    'DEFAULT_BC',
    'DEFAULT_BR',
    'DEFAULT_WEIGHTS',

    # DHI
    'DHIResult',
    'EMHealthScore',
    'PipelineDHIOutput',
    'compute_dhi',
    'compute_dhi_full',
    'compute_dhi_from_pd_types',
    'compute_dhi_from_pipeline_result',
    'compute_dhi_metrics_standalone',
    'compute_em_health_score',
    'get_health_state',
    'get_maintenance_state',
    'get_worst_health_state',
    'MAINTENANCE_THRESHOLDS',
    'HEALTH_STATE_SEVERITY',

    # Multi-sensor (3-phase + tank)
    'SENSOR_NAMES',
    'SensorDHIInput',
    'SensorDHIResult',
    'MultiSensorDHIResult',
    'compute_sensor_rpd',
    'compute_multisensor_dhi',
]
