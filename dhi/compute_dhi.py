"""
Global DHI (Diagnostic Health Index) Calculation
Based on MATLAB reference: DHI_computeDHI.m (V1.1)

DHI = Rpd × Rgas

Health state = WORST of Rpd and Rgas states (DGA is golden standard)

Maintenance Thresholds (MATLAB V1.1):
    - Safe:      DHI ≥ 0.6
    - Alert:     0.3 ≤ DHI < 0.6
    - Immediate: DHI < 0.3
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime

from .compute_rpd import (
    RpdConfig, RpdState, RpdResult,
    compute_rpd, compute_rpd_from_pd_types, compute_rpd_simple
)
from .compute_rgas import (
    RgasConfig, RgasResult,
    compute_rgas, compute_rgas_from_dict
)


# ============================================================================
# Health State Definitions (MATLAB V1.1)
# ============================================================================

# Maintenance thresholds
MAINTENANCE_THRESHOLDS = {
    'safe': 0.6,       # DHI ≥ 0.6: Safe operation
    'alert': 0.3,      # 0.3 ≤ DHI < 0.6: Alert
    'immediate': 0.0,  # DHI < 0.3: Immediate maintenance
}

# Health state severity (lower = worse)
HEALTH_STATE_SEVERITY = {
    "immediate": 0,
    "critical": 0,
    "poor": 1,
    "alert": 1,
    "fair": 2,
    "good": 3,
    "safe": 3,
}


def get_health_state(value: float) -> str:
    """
    Get health state from DHI/Rpd/Rgas value.

    Uses MATLAB V1.1 thresholds:
        ≥ 0.6: good (safe)
        0.3-0.6: fair (alert)
        < 0.3: poor (immediate)
    """
    if value >= 0.6:
        return "good"
    elif value >= 0.3:
        return "fair"
    else:
        return "poor"


def get_maintenance_state(value: float) -> str:
    """
    Get maintenance recommendation from DHI value.

    Returns:
        'safe': Normal operation, continue monitoring
        'alert': Schedule maintenance, increase monitoring frequency
        'immediate': Requires immediate attention
    """
    if value >= 0.6:
        return "safe"
    elif value >= 0.3:
        return "alert"
    else:
        return "immediate"


def get_worst_health_state(*states: str) -> str:
    """
    Get worst health state from multiple states.

    DGA is golden standard - worst state wins.

    Args:
        *states: Variable number of health state strings

    Returns:
        Worst (lowest severity) health state
    """
    worst = "good"
    worst_severity = HEALTH_STATE_SEVERITY.get("good", 3)

    for state in states:
        if state is None:
            continue
        state_lower = state.lower()
        severity = HEALTH_STATE_SEVERITY.get(state_lower, 3)
        if severity < worst_severity:
            worst = state_lower
            worst_severity = severity

    return worst


@dataclass
class DHIResult:
    """Complete DHI calculation result."""
    # Global DHI
    DHI: float
    health_state: str
    maintenance_state: str

    # Components
    Rpd: float
    Rgas: Optional[float]
    rpd_state: str
    rgas_state: Optional[str]

    # Detailed results
    rpd_result: Optional[RpdResult] = None
    rgas_result: Optional[RgasResult] = None

    # Cumulative state (for persistence)
    rpd_state_data: Optional[Dict[str, Any]] = None

    # Additional metrics
    components: Dict[str, float] = field(default_factory=dict)

    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'DHI': self.DHI,
            'health_state': self.health_state,
            'maintenance_state': self.maintenance_state,
            'Rpd': self.Rpd,
            'Rgas': self.Rgas,
            'rpd_state': self.rpd_state,
            'rgas_state': self.rgas_state,
            'rpd_result': self.rpd_result.to_dict() if self.rpd_result else None,
            'rgas_result': self.rgas_result.to_dict() if self.rgas_result else None,
            'rpd_state_data': self.rpd_state_data,
            'components': self.components,
            'timestamp': self.timestamp,
        }


def compute_dhi(
    Rpd: float,
    Rgas: Optional[float] = None,
    additional_components: Optional[Dict[str, float]] = None,
) -> DHIResult:
    """
    Compute global DHI from component reliabilities.

    Args:
        Rpd: PD reliability (0-1)
        Rgas: DGA reliability (0-1), optional
        additional_components: Optional dict of additional factors

    Returns:
        DHIResult
    """
    components = {'Rpd': Rpd}
    DHI = Rpd

    if Rgas is not None:
        DHI *= Rgas
        components['Rgas'] = Rgas

    if additional_components:
        for name, value in additional_components.items():
            if value is not None:
                DHI *= value
                components[name] = value

    DHI = max(0.0, min(1.0, DHI))

    rpd_state = get_health_state(Rpd)
    rgas_state = get_health_state(Rgas) if Rgas is not None else None
    health_state = get_worst_health_state(rpd_state, rgas_state)
    maintenance_state = get_maintenance_state(DHI)

    return DHIResult(
        DHI=DHI,
        health_state=health_state,
        maintenance_state=maintenance_state,
        Rpd=Rpd,
        Rgas=Rgas,
        rpd_state=rpd_state,
        rgas_state=rgas_state,
        components=components,
        timestamp=datetime.utcnow().isoformat(),
    )


def compute_dhi_full(
    # PD inputs
    A: float,
    RR: float,
    Pk: List[float],
    rpd_config: Optional[RpdConfig] = None,
    rpd_state: Optional[RpdState] = None,
    t: Optional[datetime] = None,
    reset_rpd: bool = False,
    # DGA inputs
    gas_names: Optional[List[str]] = None,
    conc: Optional[List[float]] = None,
    rate: Optional[List[float]] = None,
    rgas_config: Optional[RgasConfig] = None,
) -> Tuple[DHIResult, RpdState]:
    """
    Compute full DHI with both PD and DGA inputs.

    Args:
        A: Current PD amplitude (V)
        RR: Current PD repetition rate (pulses/sec)
        Pk: PD type probabilities [Psurf, Pint]
        rpd_config: RpdConfig (uses defaults if None)
        rpd_state: Previous RpdState for cumulative mode
        t: Timestamp for cumulative mode
        reset_rpd: Reset cumulative PD state

        gas_names: List of gas names for DGA
        conc: Gas concentrations (ppm)
        rate: Gas rates (ppm/month), optional
        rgas_config: RgasConfig (uses defaults if None)

    Returns:
        Tuple of (DHIResult, updated RpdState)
    """
    # ===== Compute Rpd =====
    if rpd_config is None:
        rpd_config = RpdConfig()

    rpd_result, new_rpd_state = compute_rpd(
        A=A, RR=RR, Pk=Pk, config=rpd_config,
        t=t, state=rpd_state, reset=reset_rpd
    )

    # ===== Compute Rgas (if DGA data provided) =====
    rgas_result = None
    if gas_names is not None and conc is not None:
        if rgas_config is None:
            rgas_config = RgasConfig()

        rgas_result = compute_rgas(
            gas_names=gas_names,
            conc=conc,
            rate=rate,
            config=rgas_config
        )

    # ===== Compute global DHI =====
    Rpd = rpd_result.Rpd
    Rgas = rgas_result.Rgas if rgas_result else None

    dhi_result = compute_dhi(Rpd=Rpd, Rgas=Rgas)

    # Attach detailed results
    dhi_result.rpd_result = rpd_result
    dhi_result.rgas_result = rgas_result
    dhi_result.rpd_state_data = new_rpd_state.to_dict()

    return dhi_result, new_rpd_state


def compute_dhi_from_pd_types(
    pd_types: Dict[str, Dict[str, Any]],
    A0: float,
    RR0: float,
    Rgas: Optional[float] = None,
    rpd_config: Optional[RpdConfig] = None,
    rpd_state: Optional[RpdState] = None,
    t: Optional[datetime] = None,
    reset: bool = False,
) -> Tuple[DHIResult, RpdState]:
    """
    Compute DHI from PD classification results.

    Args:
        pd_types: Dict with 'internal', 'surface', 'corona' containing
                  'count', 'amplitude', 'pulses_per_second', etc.
        A0: Baseline amplitude
        RR0: Baseline repetition rate
        Rgas: Optional pre-computed Rgas value
        rpd_config: Optional RpdConfig
        rpd_state: Optional RpdState for cumulative mode
        t: Timestamp for cumulative mode
        reset: Reset cumulative state

    Returns:
        Tuple of (DHIResult, RpdState)
    """
    if rpd_config is None:
        rpd_config = RpdConfig(A0=A0, RR0=RR0)
    else:
        rpd_config.A0 = A0
        rpd_config.RR0 = RR0

    rpd_result, new_state = compute_rpd_from_pd_types(
        pd_types=pd_types,
        A0=A0,
        RR0=RR0,
        config=rpd_config,
        state=rpd_state,
        t=t,
        reset=reset
    )

    dhi_result = compute_dhi(Rpd=rpd_result.Rpd, Rgas=Rgas)
    dhi_result.rpd_result = rpd_result
    dhi_result.rpd_state_data = new_state.to_dict()

    return dhi_result, new_state


# ============================================================================
# EM Health Score (for dashboard integration)
# ============================================================================

@dataclass
class EMHealthScore:
    """EM-based health score for dashboard display."""
    score: float              # 0-100 scale
    health_state: str         # good/fair/poor
    maintenance_state: str    # safe/alert/immediate
    Rpd: float
    defect_type: str
    pd_types: Dict[str, Dict]
    trend: str
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'health_state': self.health_state,
            'maintenance_state': self.maintenance_state,
            'Rpd': self.Rpd,
            'defect_type': self.defect_type,
            'pd_types': self.pd_types,
            'trend': self.trend,
            'last_updated': self.last_updated,
        }


def compute_em_health_score(
    pd_types: Dict[str, Dict[str, Any]],
    A0: float,
    RR0: float,
    previous_Rpd: Optional[float] = None,
    rpd_config: Optional[RpdConfig] = None,
    rpd_state: Optional[RpdState] = None,
    t: Optional[datetime] = None,
) -> Tuple[EMHealthScore, RpdState]:
    """
    Compute EM-based health score for dashboard.

    Args:
        pd_types: PD classification results
        A0: Baseline amplitude
        RR0: Baseline repetition rate
        previous_Rpd: Previous Rpd for trend calculation
        rpd_config: Optional RpdConfig
        rpd_state: Optional RpdState for cumulative mode
        t: Timestamp

    Returns:
        Tuple of (EMHealthScore, RpdState)
    """
    rpd_result, new_state = compute_rpd_from_pd_types(
        pd_types=pd_types,
        A0=A0,
        RR0=RR0,
        config=rpd_config,
        state=rpd_state,
        t=t,
    )

    score = rpd_result.Rpd * 100
    health_state = get_health_state(rpd_result.Rpd)
    maintenance_state = get_maintenance_state(rpd_result.Rpd)

    # Determine trend
    if previous_Rpd is None:
        trend = "stable"
    elif rpd_result.Rpd > previous_Rpd + 0.05:
        trend = "improving"
    elif rpd_result.Rpd < previous_Rpd - 0.05:
        trend = "degrading"
    else:
        trend = "stable"

    em_score = EMHealthScore(
        score=round(score, 1),
        health_state=health_state,
        maintenance_state=maintenance_state,
        Rpd=rpd_result.Rpd,
        defect_type=rpd_result.defect_type,
        pd_types=pd_types,
        trend=trend,
        last_updated=datetime.utcnow().isoformat(),
    )

    return em_score, new_state


# ============================================================================
# Pipeline Integration
# ============================================================================

@dataclass
class PipelineDHIOutput:
    """
    DHI output formatted for pipeline integration and S3 storage.

    Contains both:
    - Computed DHI outputs (Rpd, Rgas, DHI, health states)
    - Full detailed results for storage
    """
    # Core DHI values
    dhi_value: float
    rpd_value: float
    rgas_value: Optional[float]

    # Health classification
    health_state: str          # good/fair/poor
    maintenance_state: str     # safe/alert/immediate

    # Rpd details
    dr_value: float            # Degradation rate
    s_instantaneous: float     # Instantaneous S score
    spd_cumulative: Optional[float]  # Cumulative SPD (if cumulative mode)
    smax_used: float
    smax_auto_updated: bool

    # PD topology
    pk_surface: float
    pk_internal: float
    defect_type: str

    # Rgas details (optional)
    rgas_s_score: Optional[float]
    rgas_worst_gas: Optional[str]
    rgas_health_state: Optional[str]

    # Mode
    calculation_mode: str      # instantaneous/cumulative

    # Detailed result objects
    rpd_result: Optional[RpdResult] = None
    rgas_result: Optional[RgasResult] = None
    rpd_state_data: Optional[Dict[str, Any]] = None

    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            # Core values
            'dhi_value': self.dhi_value,
            'rpd_value': self.rpd_value,
            'rgas_value': self.rgas_value,
            # Health states
            'health_state': self.health_state,
            'maintenance_state': self.maintenance_state,
            # Rpd details
            'dr_value': self.dr_value,
            's_instantaneous': self.s_instantaneous,
            'spd_cumulative': self.spd_cumulative,
            'smax_used': self.smax_used,
            'smax_auto_updated': self.smax_auto_updated,
            # PD topology
            'pk_surface': self.pk_surface,
            'pk_internal': self.pk_internal,
            'defect_type': self.defect_type,
            # Rgas details
            'rgas_s_score': self.rgas_s_score,
            'rgas_worst_gas': self.rgas_worst_gas,
            'rgas_health_state': self.rgas_health_state,
            # Mode
            'calculation_mode': self.calculation_mode,
            # Detailed results
            'rpd_result': self.rpd_result.to_dict() if self.rpd_result else None,
            'rgas_result': self.rgas_result.to_dict() if self.rgas_result else None,
            'rpd_state_data': self.rpd_state_data,
            'timestamp': self.timestamp,
        }


def compute_dhi_from_pipeline_result(
    pipeline_result: Any,  # ProcessingResult from pipeline.py
    A0: float,
    RR0: float,
    Rgas: Optional[float] = None,
    dga_values: Optional[Dict[str, float]] = None,
    gas_rates: Optional[Dict[str, float]] = None,
    previous_Rpd: Optional[float] = None,
    rpd_config: Optional[RpdConfig] = None,
    rpd_state: Optional[RpdState] = None,
    t: Optional[datetime] = None,
) -> PipelineDHIOutput:
    """
    Compute DHI from a pipeline ProcessingResult.

    Bridges the pipeline output to the DHI calculation system.

    Args:
        pipeline_result: ProcessingResult from process_raw_data or process_waveforms
        A0: Baseline amplitude for this transformer
        RR0: Baseline repetition rate (pulses/sec)
        Rgas: Optional pre-computed DGA reliability (0-1)
        dga_values: Optional DGA values to compute Rgas {'h2': ppm, ...}
        gas_rates: Optional gas generation rates {'h2': ppm/month, ...}
        previous_Rpd: Previous Rpd for trend calculation
        rpd_config: Optional RpdConfig
        rpd_state: Optional RpdState for cumulative mode
        t: Timestamp for cumulative mode

    Returns:
        PipelineDHIOutput with computed DHI values
    """
    # Extract PD type distribution from pipeline result
    pd_types = {}
    if pipeline_result.pd_type_results:
        pd_dist = pipeline_result.pd_type_results.pd_type_distribution
        cluster_classes = pipeline_result.pd_type_results.cluster_classifications

        # Build pd_types dict from classification results
        for pd_type in ['internal', 'surface', 'corona']:
            count = 0
            total_amplitude = 0.0

            for cluster_id, info in cluster_classes.items():
                if info.get('pd_type', '').lower() == pd_type:
                    count += info.get('count', 0)
                    total_amplitude += info.get('amplitude', 0) * info.get('count', 0)

            avg_amplitude = total_amplitude / count if count > 0 else 0.0

            pd_types[pd_type] = {
                'count': count,
                'amplitude': avg_amplitude,
                'fraction': pd_dist.get(pd_type, 0.0),
            }

    # Get amplitude and repetition rate from DHI metrics
    A = 0.0
    RR = 0.0
    if pipeline_result.dhi_metrics:
        A = pipeline_result.dhi_metrics.amplitude_avg
        RR = pipeline_result.dhi_metrics.repetition_rate

    # Compute Pk from pd_types
    total_count = sum(pd_types.get(t, {}).get('count', 0) for t in ['internal', 'surface'])
    if total_count > 0:
        Pk = [
            pd_types.get('surface', {}).get('count', 0) / total_count,
            pd_types.get('internal', {}).get('count', 0) / total_count,
        ]
    else:
        Pk = [0.5, 0.5]  # Default equal split

    # Initialize config
    if rpd_config is None:
        rpd_config = RpdConfig(A0=A0, RR0=RR0)
    else:
        rpd_config.A0 = A0
        rpd_config.RR0 = RR0

    # Compute Rpd
    rpd_result, new_rpd_state = compute_rpd(
        A=A, RR=RR, Pk=Pk, config=rpd_config,
        t=t, state=rpd_state
    )

    # Compute Rgas if DGA data provided
    rgas_result = None
    if dga_values is not None:
        gas_names = list(dga_values.keys())
        conc = list(dga_values.values())
        rate = list(gas_rates.values()) if gas_rates else None

        rgas_result = compute_rgas(gas_names=gas_names, conc=conc, rate=rate)
        Rgas = rgas_result.Rgas

    # Compute global DHI
    dhi_result = compute_dhi(Rpd=rpd_result.Rpd, Rgas=Rgas)

    # Build output
    output = PipelineDHIOutput(
        # Core values
        dhi_value=dhi_result.DHI,
        rpd_value=rpd_result.Rpd,
        rgas_value=Rgas,
        # Health states
        health_state=dhi_result.health_state,
        maintenance_state=dhi_result.maintenance_state,
        # Rpd details
        dr_value=rpd_result.DR,
        s_instantaneous=rpd_result.S,
        spd_cumulative=rpd_result.SPD if rpd_result.mode == 'cumulative' else None,
        smax_used=rpd_result.Smax_used,
        smax_auto_updated=rpd_result.did_auto_update,
        # PD topology
        pk_surface=Pk[0],
        pk_internal=Pk[1],
        defect_type=rpd_result.defect_type,
        # Rgas details
        rgas_s_score=rgas_result.S if rgas_result else None,
        rgas_worst_gas=rgas_result.worst_gas if rgas_result else None,
        rgas_health_state=rgas_result.health_state if rgas_result else None,
        # Mode
        calculation_mode=rpd_result.mode,
        # Detailed results
        rpd_result=rpd_result,
        rgas_result=rgas_result,
        rpd_state_data=new_rpd_state.to_dict(),
        timestamp=datetime.utcnow().isoformat(),
    )

    return output


def compute_dhi_metrics_standalone(
    A: float,
    RR: float,
    Pk: List[float],
    A0: float,
    RR0: float,
    Rgas: Optional[float] = None,
    rpd_config: Optional[RpdConfig] = None,
) -> PipelineDHIOutput:
    """
    Compute DHI metrics without a pipeline result.

    Useful for standalone testing or when processing data outside the pipeline.

    Args:
        A: Current amplitude
        RR: Current repetition rate
        Pk: [Psurf, Pint] probabilities
        A0: Baseline amplitude
        RR0: Baseline repetition rate
        Rgas: Optional Rgas value
        rpd_config: Optional RpdConfig

    Returns:
        PipelineDHIOutput
    """
    if rpd_config is None:
        rpd_config = RpdConfig(A0=A0, RR0=RR0)

    rpd_result, rpd_state = compute_rpd(
        A=A, RR=RR, Pk=Pk, config=rpd_config
    )

    dhi_result = compute_dhi(Rpd=rpd_result.Rpd, Rgas=Rgas)

    return PipelineDHIOutput(
        dhi_value=dhi_result.DHI,
        rpd_value=rpd_result.Rpd,
        rgas_value=Rgas,
        health_state=dhi_result.health_state,
        maintenance_state=dhi_result.maintenance_state,
        dr_value=rpd_result.DR,
        s_instantaneous=rpd_result.S,
        spd_cumulative=None,
        smax_used=rpd_result.Smax_used,
        smax_auto_updated=rpd_result.did_auto_update,
        pk_surface=Pk[0],
        pk_internal=Pk[1],
        defect_type=rpd_result.defect_type,
        rgas_s_score=None,
        rgas_worst_gas=None,
        rgas_health_state=None,
        calculation_mode='instantaneous',
        rpd_result=rpd_result,
        rpd_state_data=rpd_state.to_dict(),
        timestamp=datetime.utcnow().isoformat(),
    )


# ============================================================================
# Multi-Sensor DHI (3-Phase + Tank Ground)
# ============================================================================

# Valid sensor names
SENSOR_NAMES = ['phase_a', 'phase_b', 'phase_c', 'tank']


@dataclass
class SensorDHIInput:
    """Input data for a single sensor's DHI calculation."""
    sensor_name: str           # 'phase_a', 'phase_b', 'phase_c', 'tank'
    A: float                   # Current amplitude
    RR: float                  # Current repetition rate (pulses/sec)
    Pk: List[float]            # [Psurf, Pint] probabilities
    A0: float                  # Baseline amplitude for this sensor
    RR0: float                 # Baseline repetition rate for this sensor

    # Optional cumulative state
    state: Optional[RpdState] = None


@dataclass
class SensorDHIResult:
    """DHI result for a single sensor."""
    sensor_name: str
    Rpd: float
    DHI: float                 # Rpd × Rgas (Rgas is shared)
    health_state: str
    maintenance_state: str

    # Rpd details
    DR: float
    S: float
    SPD: Optional[float]
    defect_type: str
    Pk: List[float]

    # Full result for storage
    rpd_result: Optional[RpdResult] = None
    rpd_state_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sensor_name': self.sensor_name,
            'Rpd': self.Rpd,
            'DHI': self.DHI,
            'health_state': self.health_state,
            'maintenance_state': self.maintenance_state,
            'DR': self.DR,
            'S': self.S,
            'SPD': self.SPD,
            'defect_type': self.defect_type,
            'Pk': self.Pk,
            'rpd_result': self.rpd_result.to_dict() if self.rpd_result else None,
            'rpd_state_data': self.rpd_state_data,
        }


@dataclass
class MultiSensorDHIResult:
    """
    Complete DHI result for a 3-phase + tank ground transformer.

    Global DHI = min(Rpd_A, Rpd_B, Rpd_C, Rpd_Tank) × Rgas
    """
    # Per-sensor results
    phase_a: Optional[SensorDHIResult] = None
    phase_b: Optional[SensorDHIResult] = None
    phase_c: Optional[SensorDHIResult] = None
    tank: Optional[SensorDHIResult] = None

    # Global DHI (worst sensor × Rgas)
    global_Rpd: float = 1.0        # min of all sensor Rpds
    global_DHI: float = 1.0        # global_Rpd × Rgas
    worst_sensor: str = ""         # Which sensor has lowest Rpd
    global_health_state: str = "good"
    global_maintenance_state: str = "safe"

    # DGA (transformer-wide)
    Rgas: Optional[float] = None
    rgas_result: Optional[RgasResult] = None
    rgas_health_state: Optional[str] = None

    # Metadata
    sensors_computed: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'phase_a': self.phase_a.to_dict() if self.phase_a else None,
            'phase_b': self.phase_b.to_dict() if self.phase_b else None,
            'phase_c': self.phase_c.to_dict() if self.phase_c else None,
            'tank': self.tank.to_dict() if self.tank else None,
            'global_Rpd': self.global_Rpd,
            'global_DHI': self.global_DHI,
            'worst_sensor': self.worst_sensor,
            'global_health_state': self.global_health_state,
            'global_maintenance_state': self.global_maintenance_state,
            'Rgas': self.Rgas,
            'rgas_result': self.rgas_result.to_dict() if self.rgas_result else None,
            'rgas_health_state': self.rgas_health_state,
            'sensors_computed': self.sensors_computed,
            'timestamp': self.timestamp,
        }

    def get_sensor_result(self, sensor_name: str) -> Optional[SensorDHIResult]:
        """Get result for a specific sensor."""
        return getattr(self, sensor_name, None)


def compute_sensor_rpd(
    sensor_input: SensorDHIInput,
    rpd_config: Optional[RpdConfig] = None,
    t: Optional[datetime] = None,
    reset: bool = False,
) -> Tuple[SensorDHIResult, RpdState]:
    """
    Compute Rpd for a single sensor.

    Args:
        sensor_input: SensorDHIInput with sensor data
        rpd_config: Optional RpdConfig (will set A0/RR0 from input)
        t: Timestamp for cumulative mode
        reset: Reset cumulative state

    Returns:
        Tuple of (SensorDHIResult, new RpdState)
    """
    if rpd_config is None:
        rpd_config = RpdConfig()

    # Set baselines from sensor input
    rpd_config.A0 = sensor_input.A0
    rpd_config.RR0 = sensor_input.RR0

    # Compute Rpd
    rpd_result, new_state = compute_rpd(
        A=sensor_input.A,
        RR=sensor_input.RR,
        Pk=sensor_input.Pk,
        config=rpd_config,
        t=t,
        state=sensor_input.state,
        reset=reset,
    )

    # Build sensor result (DHI will be set later when Rgas is known)
    sensor_result = SensorDHIResult(
        sensor_name=sensor_input.sensor_name,
        Rpd=rpd_result.Rpd,
        DHI=rpd_result.Rpd,  # Will be multiplied by Rgas later
        health_state=get_health_state(rpd_result.Rpd),
        maintenance_state=get_maintenance_state(rpd_result.Rpd),
        DR=rpd_result.DR,
        S=rpd_result.S,
        SPD=rpd_result.SPD if rpd_result.mode == 'cumulative' else None,
        defect_type=rpd_result.defect_type,
        Pk=sensor_input.Pk,
        rpd_result=rpd_result,
        rpd_state_data=new_state.to_dict(),
    )

    return sensor_result, new_state


def compute_multisensor_dhi(
    sensor_inputs: List[SensorDHIInput],
    Rgas: Optional[float] = None,
    dga_values: Optional[Dict[str, float]] = None,
    gas_rates: Optional[Dict[str, float]] = None,
    rgas_config: Optional[RgasConfig] = None,
    rpd_config: Optional[RpdConfig] = None,
    t: Optional[datetime] = None,
    reset: bool = False,
) -> Tuple[MultiSensorDHIResult, Dict[str, RpdState]]:
    """
    Compute DHI for multiple sensors (3-phase + tank ground).

    Global DHI = min(Rpd_A, Rpd_B, Rpd_C, Rpd_Tank) × Rgas

    Args:
        sensor_inputs: List of SensorDHIInput for each sensor
        Rgas: Optional pre-computed Rgas value
        dga_values: Optional DGA concentrations to compute Rgas
        gas_rates: Optional DGA rates
        rgas_config: Optional RgasConfig
        rpd_config: Optional RpdConfig (shared config, baselines per sensor)
        t: Timestamp for cumulative mode
        reset: Reset all cumulative states

    Returns:
        Tuple of (MultiSensorDHIResult, dict of sensor_name -> RpdState)
    """
    result = MultiSensorDHIResult(timestamp=datetime.utcnow().isoformat())
    new_states: Dict[str, RpdState] = {}

    # Compute Rpd for each sensor
    sensor_rpds: Dict[str, float] = {}

    for sensor_input in sensor_inputs:
        sensor_result, new_state = compute_sensor_rpd(
            sensor_input=sensor_input,
            rpd_config=rpd_config,
            t=t,
            reset=reset,
        )

        # Store result
        setattr(result, sensor_input.sensor_name, sensor_result)
        new_states[sensor_input.sensor_name] = new_state
        sensor_rpds[sensor_input.sensor_name] = sensor_result.Rpd
        result.sensors_computed.append(sensor_input.sensor_name)

    # Compute Rgas if DGA data provided
    rgas_result = None
    if dga_values is not None:
        gas_names = list(dga_values.keys())
        conc = list(dga_values.values())
        rate = list(gas_rates.values()) if gas_rates else None

        if rgas_config is None:
            rgas_config = RgasConfig()

        rgas_result = compute_rgas(
            gas_names=gas_names,
            conc=conc,
            rate=rate,
            config=rgas_config,
        )
        Rgas = rgas_result.Rgas

    # Store Rgas results
    result.Rgas = Rgas
    result.rgas_result = rgas_result
    result.rgas_health_state = get_health_state(Rgas) if Rgas else None

    # Find worst sensor (lowest Rpd)
    if sensor_rpds:
        worst_sensor = min(sensor_rpds, key=sensor_rpds.get)
        result.worst_sensor = worst_sensor
        result.global_Rpd = sensor_rpds[worst_sensor]
    else:
        result.global_Rpd = 1.0
        result.worst_sensor = ""

    # Compute global DHI
    if Rgas is not None:
        result.global_DHI = result.global_Rpd * Rgas
    else:
        result.global_DHI = result.global_Rpd

    # Update per-sensor DHI with Rgas
    if Rgas is not None:
        for sensor_name in result.sensors_computed:
            sensor_result = getattr(result, sensor_name)
            if sensor_result:
                sensor_result.DHI = sensor_result.Rpd * Rgas
                # Update health/maintenance states based on full DHI
                sensor_result.health_state = get_health_state(sensor_result.DHI)
                sensor_result.maintenance_state = get_maintenance_state(sensor_result.DHI)

    # Global health state (worst of all)
    all_states = [result.rgas_health_state] if result.rgas_health_state else []
    for sensor_name in result.sensors_computed:
        sensor_result = getattr(result, sensor_name)
        if sensor_result:
            all_states.append(sensor_result.health_state)

    result.global_health_state = get_worst_health_state(*all_states) if all_states else "good"
    result.global_maintenance_state = get_maintenance_state(result.global_DHI)

    return result, new_states
