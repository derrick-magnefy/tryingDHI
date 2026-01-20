"""
Rpd (PD Reliability) Calculation
Based on MATLAB reference: compute_Rpd.m (V1.1)

Supports:
- Instantaneous mode: Single-point Rpd calculation
- Cumulative mode: Time-aware integration with trapezoidal rule
- Auto Smax adjustment to prevent DHI saturation
- PD type weighting (internal weighted higher than surface)

Formula:
    DR = (A/A0)^signA × (RR/RR0)^signRR
    Sk = Pk × DR  (per topology)
    S = Σ(Wk × Sk) / Σ(Wk)  (weighted average)

    Instantaneous: Rpd = 1 - S/(Smax-1)
    Cumulative: SPD += 0.5×(S + S_prev)×dt/T0, then Rpd = 1 - SPD/(Smax-1)

IMPORTANT - Sign Logic Fix:
    When BOTH A < A0 AND RR < RR0 (PD below baseline), DR is set to 1.0.
    This treats "below baseline" as equivalent to "at baseline" (no extra damage).
    The original MATLAB sign logic had a bug where lower PD caused higher DR.

IMPORTANT - Baseline Values (A0, RR0):
    A0 and RR0 should be the FIRST measurement values from the transformer.
    Per the DHI algorithm specification, baselines are "initial values of the test"
    - the measurements taken when monitoring begins (assumed healthy state).

    The Python defaults (0.02 V, 5000 pulses/sec) are PLACEHOLDERS only.
    For production use, always provide actual first-measurement baselines.

IMPORTANT - T0 (Expected Asset Lifespan):
    T0 represents the expected lifespan of the transformer/asset.
    It determines how much each sample contributes to cumulative damage:

        damage_per_sample = dt/T0 × S

    For a daily reading on a 40-year transformer:
        dt/T0 = 1 day / 40 years ≈ 0.0000685

    This means each day at baseline contributes ~0.007% of total lifespan damage.

    T0 constants by transformer type:
    - T0_20_YEARS: For assets with 20-year expected lifespan
    - T0_30_YEARS: For assets with 30-year expected lifespan
    - T0_40_YEARS: For assets with 40-year expected lifespan (default)
    - T0_50_YEARS: For assets with 50-year expected lifespan

    T0 should be configured per asset based on expected service life.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import math


# =============================================================================
# T0 Constants (Expected Asset Lifespan)
# =============================================================================
# T0 represents the expected lifespan of the transformer/asset.
# It determines how quickly cumulative damage accumulates:
#   damage_per_sample = dt/T0 × S
#
# Example: Daily sampling on 40-year transformer
#   dt/T0 = 1 day / 40 years = 1/(40×365) ≈ 0.0000685 per sample

SECONDS_PER_YEAR = 365.25 * 24 * 3600  # ~31,557,600 seconds

T0_20_YEARS = 20 * SECONDS_PER_YEAR    # ~631M seconds - shorter lifespan assets
T0_30_YEARS = 30 * SECONDS_PER_YEAR    # ~946M seconds - typical MV transformers
T0_40_YEARS = 40 * SECONDS_PER_YEAR    # ~1.26B seconds - typical HV transformers
T0_50_YEARS = 50 * SECONDS_PER_YEAR    # ~1.58B seconds - longer lifespan assets

# Legacy constants (for backwards compatibility)
T0_HOURLY = 3600.0       # Legacy: hourly interval
T0_15MIN = 900.0         # Legacy: 15-minute interval
T0_DAILY = 86400.0       # Legacy: daily interval
T0_WEEKLY = 604800.0     # Legacy: weekly interval

# Default: 40-year expected lifespan (typical for HV transformers)
T0_DEFAULT = T0_40_YEARS


@dataclass
class RpdConfig:
    """
    Configuration for Rpd calculation.

    IMPORTANT:
        A0, RR0 should be the FIRST measurement values from this transformer.
        The defaults are placeholders; always provide actual baselines.

        T0_seconds is the expected asset lifespan in seconds.
        Default is 40 years (~1.26 billion seconds) for typical HV transformers.
        This should be configured per asset based on expected service life.
    """
    # Baselines - SHOULD BE FIRST MEASUREMENT (defaults are placeholders)
    A0: float = 0.02          # Baseline amplitude (V) - USE FIRST MEASUREMENT
    RR0: float = 5000.0       # Baseline repetition rate (pulses/sec) - USE FIRST MEASUREMENT

    # PD type weights: [surface, internal] - internal weighted 3x (more severe)
    W: List[float] = field(default_factory=lambda: [1.0, 3.0])

    # Smax for score normalization (auto-adjusts in cumulative mode)
    Smax: float = 12.0

    # Clip Rpd to [0, 1]
    clip01: bool = True

    # Mode: 'instantaneous' or 'cumulative'
    mode: str = 'instantaneous'

    # Cumulative mode settings
    # T0 = expected asset lifespan in seconds
    # Default: 40 years for typical HV transformers
    T0_seconds: float = T0_DEFAULT

    # Phase severity settings
    zero_crossing_threshold: float = 15.0  # degrees from 0/180


@dataclass
class RpdState:
    """Persistent state for cumulative mode."""
    SPD: float = 0.0              # Cumulative score ΣS
    S_prev: float = 0.0           # Previous instantaneous S
    t_prev: Optional[datetime] = None
    Smax_used: float = 12.0       # Current Smax (may auto-adjust)
    auto_smax_initialized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'SPD': self.SPD,
            'S_prev': self.S_prev,
            't_prev': self.t_prev.isoformat() if self.t_prev else None,
            'Smax_used': self.Smax_used,
            'auto_smax_initialized': self.auto_smax_initialized,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RpdState':
        t_prev = None
        if data.get('t_prev'):
            try:
                t_prev = datetime.fromisoformat(data['t_prev'])
            except (ValueError, TypeError):
                pass
        return cls(
            SPD=data.get('SPD', 0.0),
            S_prev=data.get('S_prev', 0.0),
            t_prev=t_prev,
            Smax_used=data.get('Smax_used', 12.0),
            auto_smax_initialized=data.get('auto_smax_initialized', False),
        )


@dataclass
class RpdResult:
    """Result of Rpd calculation."""
    Rpd: float
    S: float                    # Instantaneous weighted score
    DR: float                   # Degradation rate

    # Inputs used
    A: float
    RR: float
    A0: float
    RR0: float
    Pk: List[float]            # PD type probabilities [Psurf, Pint]

    # Per-topology details
    Sk: List[float]            # Per-topology scores
    W: List[float]             # Weights used

    # Sign logic
    sign_A: int
    sign_RR: int

    # Mode and cumulative state
    mode: str
    SPD: Optional[float] = None          # Cumulative score (if cumulative mode)
    dt_seconds: Optional[float] = None   # Time delta (if cumulative mode)
    dt_factor: Optional[float] = None    # dt/T0 factor

    # Smax tracking
    Smax_input: float = 12.0
    Smax_used: float = 12.0
    did_auto_update: bool = False

    # Defect analysis
    defect_type: str = "unknown"
    pd_counts: Dict[str, int] = field(default_factory=dict)

    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'Rpd': self.Rpd,
            'S': self.S,
            'DR': self.DR,
            'A': self.A,
            'RR': self.RR,
            'A0': self.A0,
            'RR0': self.RR0,
            'Pk': self.Pk,
            'Sk': self.Sk,
            'W': self.W,
            'sign_A': self.sign_A,
            'sign_RR': self.sign_RR,
            'mode': self.mode,
            'SPD': self.SPD,
            'dt_seconds': self.dt_seconds,
            'dt_factor': self.dt_factor,
            'Smax_input': self.Smax_input,
            'Smax_used': self.Smax_used,
            'did_auto_update': self.did_auto_update,
            'defect_type': self.defect_type,
            'pd_counts': self.pd_counts,
            'timestamp': self.timestamp,
        }


def compute_rpd(
    A: float,
    RR: float,
    Pk: List[float],
    config: RpdConfig,
    t: Optional[datetime] = None,
    state: Optional[RpdState] = None,
    reset: bool = False,
) -> Tuple[RpdResult, RpdState]:
    """
    Compute PD-based reliability at a single time point.

    Supports instantaneous and cumulative (time-aware) modes.

    Args:
        A: Current amplitude (V)
        RR: Current repetition rate (pulses/sec)
        Pk: PD type probabilities [Psurf, Pint] - must sum to ~1
        config: RpdConfig with baselines and settings
        t: Current timestamp (required for cumulative mode)
        state: Previous RpdState (for cumulative mode continuity)
        reset: If True, reset cumulative state

    Returns:
        Tuple of (RpdResult, updated RpdState)

    Example:
        config = RpdConfig(A0=0.02, RR0=5000, mode='cumulative')
        result, state = compute_rpd(A=0.04, RR=8000, Pk=[0.3, 0.7], config=config, t=datetime.now())
    """
    # Initialize or restore state
    if state is None:
        state = RpdState(Smax_used=config.Smax)

    # Ensure Pk is a list
    Pk = list(Pk) if not isinstance(Pk, list) else Pk
    K = len(Pk)

    # Weights - match length to Pk
    W = list(config.W)
    if len(W) != K:
        if K == 2:
            W = [1.0, 3.0]  # Default: [surface, internal]
        else:
            W = [1.0] * K

    # -------- Compute DR (Degradation Rate) --------
    # Compute ratios (avoid division by zero)
    eps = 1e-10
    ratio_A = max(A, eps) / max(config.A0, eps)
    ratio_RR = max(RR, eps) / max(config.RR0, eps)

    # Sign logic fix: When BOTH A < A0 AND RR < RR0, use DR = 1.0 (baseline)
    # This fixes the bug where lower PD caused higher DR due to sign inversion.
    # "Below baseline" is treated as "at baseline" - no extra damage.
    if A < config.A0 and RR < config.RR0:
        # Both below baseline - use DR = 1 (no extra damage beyond baseline)
        sign_A = 1
        sign_RR = 1
        DR = 1.0
    else:
        # Original sign logic for other cases
        sign_A = 1 if A >= config.A0 else -1
        sign_RR = 1 if RR >= config.RR0 else -1
        DR = (ratio_A ** sign_A) * (ratio_RR ** sign_RR)

    # -------- Compute per-topology scores Sk and instantaneous S --------
    Sk = [pk * DR for pk in Pk]
    W_sum = sum(W)
    S = sum(w * sk for w, sk in zip(W, Sk)) / W_sum if W_sum > 0 else 0.0

    # -------- Cumulative mode: time-aware integration --------
    mode = config.mode.lower()
    use_cumulative = (mode == 'cumulative')

    dt_seconds = None
    dt_factor = 0.0

    if use_cumulative:
        if reset:
            # Reset cumulative state
            state.SPD = 0.0
            state.S_prev = S
            state.t_prev = t
            state.auto_smax_initialized = False
        else:
            if state.t_prev is None:
                # First point - initialize
                state.SPD = 0.0
                state.S_prev = S
                state.t_prev = t
            else:
                # Compute time delta
                if t is not None and state.t_prev is not None:
                    dt = t - state.t_prev
                    dt_seconds = dt.total_seconds()

                    if dt_seconds > 0 and config.T0_seconds > 0:
                        dt_factor = dt_seconds / config.T0_seconds

                        # Trapezoidal integration: SPD += 0.5×(S + S_prev)×dt/T0
                        state.SPD = state.SPD + 0.5 * (S + state.S_prev) * dt_factor

                # Update previous values
                state.S_prev = S
                state.t_prev = t

    # -------- Auto Smax update (cumulative mode only) --------
    Smax_used = state.Smax_used if use_cumulative else config.Smax
    did_auto_update = False

    if use_cumulative:
        # Rule: if SPD > (Smax - 1), update Smax = 2×SPD + 1
        if math.isfinite(state.SPD) and state.SPD > (Smax_used - 1):
            Smax_used = 2 * state.SPD + 1
            state.Smax_used = Smax_used
            state.auto_smax_initialized = True
            did_auto_update = True

    # -------- Map score to Rpd --------
    if use_cumulative:
        score_for_map = state.SPD
    else:
        score_for_map = S

    # Rpd = 1 - score / (Smax - 1)
    denom = Smax_used - 1
    if denom <= 0:
        denom = max(eps, denom)

    Rpd = 1.0 - score_for_map / denom

    if config.clip01:
        Rpd = max(0.0, min(1.0, Rpd))

    # -------- Determine defect type --------
    # Pk[0] = surface, Pk[1] = internal
    if len(Pk) >= 2:
        if Pk[1] > Pk[0]:
            defect_type = "internal"
        elif Pk[0] > 0:
            defect_type = "surface"
        else:
            defect_type = "none"
    else:
        defect_type = "unknown"

    # -------- Build result --------
    result = RpdResult(
        Rpd=Rpd,
        S=S,
        DR=DR,
        A=A,
        RR=RR,
        A0=config.A0,
        RR0=config.RR0,
        Pk=Pk,
        Sk=Sk,
        W=W,
        sign_A=sign_A,
        sign_RR=sign_RR,
        mode=mode,
        SPD=state.SPD if use_cumulative else None,
        dt_seconds=dt_seconds,
        dt_factor=dt_factor if use_cumulative else None,
        Smax_input=config.Smax,
        Smax_used=Smax_used,
        did_auto_update=did_auto_update,
        defect_type=defect_type,
        pd_counts={},
        timestamp=datetime.utcnow().isoformat(),
    )

    return result, state


def compute_rpd_simple(
    A: float,
    RR: float,
    A0: float,
    RR0: float,
    Pk: Optional[List[float]] = None,
    Smax: float = 12.0,
) -> RpdResult:
    """
    Simple instantaneous Rpd calculation (no cumulative state).

    Args:
        A: Current amplitude
        RR: Current repetition rate
        A0: Baseline amplitude
        RR0: Baseline repetition rate
        Pk: PD type probabilities [Psurf, Pint]. Default [0.5, 0.5]
        Smax: Maximum score for normalization

    Returns:
        RpdResult
    """
    if Pk is None:
        Pk = [0.5, 0.5]

    config = RpdConfig(A0=A0, RR0=RR0, Smax=Smax, mode='instantaneous')
    result, _ = compute_rpd(A, RR, Pk, config)
    return result


def compute_rpd_from_pd_types(
    pd_types: Dict[str, Dict[str, Any]],
    A0: float,
    RR0: float,
    config: Optional[RpdConfig] = None,
    state: Optional[RpdState] = None,
    t: Optional[datetime] = None,
    reset: bool = False,
) -> Tuple[RpdResult, RpdState]:
    """
    Compute Rpd from PD classification results.

    Args:
        pd_types: Dict with keys 'internal', 'surface', 'corona' containing
                  'count', 'amplitude', 'pulses_per_second', etc.
        A0: Baseline amplitude
        RR0: Baseline repetition rate
        config: Optional RpdConfig (created with defaults if None)
        state: Optional RpdState for cumulative mode
        t: Timestamp for cumulative mode
        reset: Reset cumulative state

    Returns:
        Tuple of (RpdResult, RpdState)
    """
    # Extract counts
    internal_count = pd_types.get('internal', {}).get('count', 0)
    surface_count = pd_types.get('surface', {}).get('count', 0)
    corona_count = pd_types.get('corona', {}).get('count', 0)

    total_pd = internal_count + surface_count
    pd_counts = {
        'internal': internal_count,
        'surface': surface_count,
        'corona': corona_count,
    }

    # Calculate Pk (probabilities)
    if total_pd > 0:
        Psurf = surface_count / total_pd
        Pint = internal_count / total_pd
    else:
        Psurf = 0.5
        Pint = 0.5
    Pk = [Psurf, Pint]

    # Extract amplitude (max of internal and surface)
    A = 0.0
    for pd_type in ['internal', 'surface']:
        type_amp = pd_types.get(pd_type, {}).get('amplitude', 0)
        if type_amp and type_amp > A:
            A = type_amp

    # Extract repetition rate (sum of internal and surface)
    RR = 0.0
    for pd_type in ['internal', 'surface']:
        type_rr = pd_types.get(pd_type, {}).get('pulses_per_second', 0)
        if type_rr:
            RR += type_rr

    # Create config if not provided
    if config is None:
        config = RpdConfig(A0=A0, RR0=RR0)
    else:
        config.A0 = A0
        config.RR0 = RR0

    # Compute Rpd
    result, new_state = compute_rpd(
        A=A, RR=RR, Pk=Pk, config=config,
        t=t, state=state, reset=reset
    )

    # Add PD counts to result
    result.pd_counts = pd_counts

    return result, new_state


# Legacy function for backward compatibility
def compute_phase_severity(phase_range: Tuple[float, float]) -> float:
    """
    Compute phase severity based on proximity to zero crossings.

    Internal PD near zero crossings (0°, 180°) is more severe.

    Args:
        phase_range: Tuple of (start, end) phase angles in degrees

    Returns:
        Severity factor 0.5-1.0 (1.0 = most severe)
    """
    if phase_range is None:
        return 1.0

    start, end = phase_range
    zero_crossings = [0, 180, 360]
    min_distance = float('inf')

    midpoint = (start + end) / 2
    for zc in zero_crossings:
        distance = min(
            abs(midpoint - zc),
            abs(midpoint - zc + 360),
            abs(midpoint - zc - 360)
        )
        min_distance = min(min_distance, distance)

    # Closer to zero crossing = higher severity
    if min_distance >= 45:
        return 1.0
    else:
        return 0.5 + 0.5 * (min_distance / 45.0)
