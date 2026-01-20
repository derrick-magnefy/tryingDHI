"""
Rgas (DGA Reliability) Calculation
Based on MATLAB reference: compute_Rgas.m (V1.1)

DGA is the golden standard for transformer health assessment.

Supports:
- Both concentration and rate scoring
- Discrete and continuous scoring modes
- Weight capping to prevent instability
- Optional Duval multiplier for fault-type weighting

Formula:
    sC = score_from_concentration(conc, Bc)
    sR = score_from_rate(rate, Br)
    q = α×sC + (1-α)×sR   (per gas combined score)
    S = Σ(wFinal × q) / Σ(wFinal)   (weighted average)
    Rgas = 1 - S/(SCmax-1)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime


# ============================================================================
# Default Thresholds (IEEE C57.104 based)
# ============================================================================

# Concentration thresholds [b1, b2, b3, b4] for each gas
# Levels: Good (≤b1), Fair (b1-b2), Poor (b2-b3), Action (b3-b4), Critical (>b4)
DEFAULT_BC = {
    'H2':    [100,  300,  700, 2000],
    'CH4':   [120,  300,  600, 1200],
    'C2H4':  [ 60,  150,  300,  600],
    'C2H2':  [  1,    5,   20,   50],
    'CO':    [350,  700, 1000, 2000],
    'CO2':   [2500, 4000, 6000, 10000],
    'C2H6':  [ 60,  150,  300,  600],
}

# Rate thresholds [b1, b2, b3, b4] in ppm/month
DEFAULT_BR = {
    'H2':    [  5,   15,   30,   50],
    'CH4':   [  4,   12,   24,   37],
    'C2H4':  [  3,    9,   18,   30],
    'C2H2':  [  1,    3,    6,   10],
    'CO':    [ 10,   20,   35,   50],
    'CO2':   [ 50,  100,  180,  250],
    'C2H6':  [  2,    4,    8,   12],
}

# Default weights per gas
DEFAULT_WEIGHTS = {
    'H2':   5.0,
    'CH4':  4.0,
    'C2H4': 5.0,
    'C2H2': 5.0,
    'CO':   3.0,
    'CO2':  3.0,
    'C2H6': 4.0,
}


@dataclass
class RgasConfig:
    """Configuration for Rgas calculation."""
    # Blending factor: α=1 uses only concentration, α=0 uses only rate
    alpha: float = 0.8

    # Scoring mode: 'discrete' (step function) or 'continuous' (interpolated)
    mode: str = 'continuous'

    # Maximum score (Action level = 4)
    SCmax: float = 5.0

    # Weight cap to prevent instability
    capW: float = 5.5

    # Duval multiplier function (optional)
    # Should take gas_names list and return multipliers list
    duval_multiplier: Optional[Callable[[List[str]], List[float]]] = None


@dataclass
class RgasResult:
    """Result of Rgas calculation."""
    Rgas: float
    S: float                          # Weighted average score
    health_state: str

    # Per-gas details
    q: Dict[str, float]               # Combined score per gas
    sC: Dict[str, float]              # Concentration score per gas
    sR: Dict[str, float]              # Rate score per gas
    w_final: Dict[str, float]         # Final weight per gas

    # Analysis
    worst_gas: str
    worst_score: float
    fault_share: Dict[str, float]     # Contribution of each gas to total score

    # Config used
    alpha: float
    mode: str
    SCmax: float
    capW: float

    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'Rgas': self.Rgas,
            'S': self.S,
            'health_state': self.health_state,
            'q': self.q,
            'sC': self.sC,
            'sR': self.sR,
            'w_final': self.w_final,
            'worst_gas': self.worst_gas,
            'worst_score': self.worst_score,
            'fault_share': self.fault_share,
            'alpha': self.alpha,
            'mode': self.mode,
            'SCmax': self.SCmax,
            'capW': self.capW,
            'timestamp': self.timestamp,
        }


def score_discrete(x: float, thresholds: List[float]) -> float:
    """
    Discrete scoring: step function (0, 1, 2, 3, 4).

    Args:
        x: Value to score
        thresholds: [b1, b2, b3, b4]

    Returns:
        Score 0-4
    """
    b1, b2, b3, b4 = thresholds
    if x <= b1:
        return 0.0
    elif x <= b2:
        return 1.0
    elif x <= b3:
        return 2.0
    elif x <= b4:
        return 3.0
    else:
        return 4.0


def score_continuous(x: float, thresholds: List[float]) -> float:
    """
    Continuous scoring: linear interpolation between levels.

    MATLAB reference starts at 1 for values below first threshold.
    Updated to start at 0 to match expected behavior.

    Args:
        x: Value to score
        thresholds: [b1, b2, b3, b4]

    Returns:
        Score 0-4+ (can exceed 4 if value >> b4)
    """
    b1, b2, b3, b4 = thresholds
    eps = 1e-10

    if x <= b1:
        return 0.0
    elif x <= b2:
        return 0.0 + (x - b1) / max(eps, b2 - b1)  # 0 to 1
    elif x <= b3:
        return 1.0 + (x - b2) / max(eps, b3 - b2)  # 1 to 2
    elif x <= b4:
        return 2.0 + (x - b3) / max(eps, b4 - b3)  # 2 to 3
    else:
        # Beyond b4: continue linear extrapolation, cap at 4
        overshoot = (x - b4) / max(eps, b4 - b3)
        return min(4.0, 3.0 + overshoot)


def compute_rgas(
    gas_names: List[str],
    conc: List[float],
    rate: Optional[List[float]] = None,
    Bc: Optional[Dict[str, List[float]]] = None,
    Br: Optional[Dict[str, List[float]]] = None,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[RgasConfig] = None,
) -> RgasResult:
    """
    Compute DGA reliability from gas concentrations and rates.

    Args:
        gas_names: List of gas names, e.g., ['H2', 'CH4', 'C2H4', ...]
        conc: Concentration values (ppm) for each gas
        rate: Rate values (ppm/month) for each gas. If None, uses conc only (alpha=1)
        Bc: Concentration thresholds per gas. Uses DEFAULT_BC if None.
        Br: Rate thresholds per gas. Uses DEFAULT_BR if None.
        weights: Base weights per gas. Uses DEFAULT_WEIGHTS if None.
        config: RgasConfig with settings. Uses defaults if None.

    Returns:
        RgasResult with Rgas and per-gas details

    Example:
        gas_names = ['H2', 'CH4', 'C2H4', 'C2H2', 'CO', 'CO2', 'C2H6']
        conc = [150, 120, 80, 0, 500, 3500, 90]
        rate = [12, 5, 4, 0, 30, 120, 3]

        result = compute_rgas(gas_names, conc, rate)
        print(f"Rgas: {result.Rgas:.3f}, State: {result.health_state}")
    """
    # Use defaults if not provided
    if config is None:
        config = RgasConfig()

    if Bc is None:
        Bc = DEFAULT_BC
    if Br is None:
        Br = DEFAULT_BR
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # If no rate data, use concentration only
    if rate is None:
        rate = [0.0] * len(gas_names)
        effective_alpha = 1.0  # Concentration only
    else:
        effective_alpha = config.alpha

    G = len(gas_names)
    assert len(conc) == G, f"conc length ({len(conc)}) must match gas_names ({G})"
    assert len(rate) == G, f"rate length ({len(rate)}) must match gas_names ({G})"

    # Score function based on mode
    if config.mode.lower() == 'discrete':
        score_func = score_discrete
    else:
        score_func = score_continuous

    # ===== 1) Score concentration & rate per gas =====
    sC_dict = {}
    sR_dict = {}

    for i, gas in enumerate(gas_names):
        # Get thresholds for this gas
        bc = Bc.get(gas, [100, 300, 700, 2000])
        br = Br.get(gas, [5, 15, 30, 50])

        sC_dict[gas] = score_func(conc[i], bc)
        sR_dict[gas] = score_func(rate[i], br)

    # ===== 2) Combined per-gas score q =====
    q_dict = {}
    for gas in gas_names:
        q_dict[gas] = effective_alpha * sC_dict[gas] + (1 - effective_alpha) * sR_dict[gas]

    # ===== 3) Apply Duval multiplier if provided =====
    mult = {gas: 1.0 for gas in gas_names}
    if config.duval_multiplier is not None:
        try:
            mult_list = config.duval_multiplier(gas_names)
            for i, gas in enumerate(gas_names):
                mult[gas] = mult_list[i] if i < len(mult_list) else 1.0
        except Exception:
            pass  # Ignore multiplier errors

    # ===== 4) Final weights with capping =====
    w_final = {}
    for gas in gas_names:
        base_w = weights.get(gas, 1.0)
        w_final[gas] = min(config.capW, base_w * mult[gas])

    # ===== 5) Weighted average score S =====
    total_weight = sum(w_final.values())
    if total_weight > 0:
        S = sum(w_final[gas] * q_dict[gas] for gas in gas_names) / total_weight
    else:
        S = 0.0

    # ===== 6) Calculate fault share (contribution of each gas) =====
    fault_share = {}
    if total_weight > 0 and S > 0:
        for gas in gas_names:
            contribution = w_final[gas] * q_dict[gas]
            fault_share[gas] = (contribution / (S * total_weight)) * 100
    else:
        fault_share = {gas: 0.0 for gas in gas_names}

    # ===== 7) Reliability Rgas =====
    SCmax = config.SCmax
    if SCmax > 1:
        Rgas = 1.0 - S / (SCmax - 1)
    else:
        Rgas = 1.0

    Rgas = max(0.0, min(1.0, Rgas))

    # ===== 8) Health state (MATLAB V1.1 thresholds) =====
    if Rgas >= 0.6:
        health_state = "good"
    elif Rgas >= 0.3:
        health_state = "fair"
    else:
        health_state = "poor"

    # ===== 9) Find worst gas =====
    worst_gas = max(gas_names, key=lambda g: q_dict.get(g, 0))
    worst_score = q_dict.get(worst_gas, 0)

    return RgasResult(
        Rgas=Rgas,
        S=S,
        health_state=health_state,
        q=q_dict,
        sC=sC_dict,
        sR=sR_dict,
        w_final=w_final,
        worst_gas=worst_gas,
        worst_score=worst_score,
        fault_share=fault_share,
        alpha=effective_alpha,
        mode=config.mode,
        SCmax=SCmax,
        capW=config.capW,
        timestamp=datetime.utcnow().isoformat(),
    )


def compute_rgas_from_dict(
    gas_values: Dict[str, float],
    gas_rates: Optional[Dict[str, float]] = None,
    config: Optional[RgasConfig] = None,
) -> RgasResult:
    """
    Convenience function to compute Rgas from dict of gas values.

    Args:
        gas_values: Dict mapping gas name to concentration (ppm)
        gas_rates: Optional dict mapping gas name to rate (ppm/month)
        config: Optional RgasConfig

    Returns:
        RgasResult
    """
    gas_names = list(gas_values.keys())
    conc = [gas_values[g] for g in gas_names]

    if gas_rates:
        rate = [gas_rates.get(g, 0.0) for g in gas_names]
    else:
        rate = None

    return compute_rgas(
        gas_names=gas_names,
        conc=conc,
        rate=rate,
        config=config,
    )


# ============================================================================
# Legacy compatibility function
# ============================================================================

def compute_rgas_legacy(
    gas_values: Dict[str, float],
    alpha: float = 0.8,
    SCmax: float = 5.0,
    weights: Optional[Dict[str, float]] = None,
) -> RgasResult:
    """
    Legacy interface for backward compatibility.

    Uses concentration only (no rate data).
    """
    config = RgasConfig(alpha=alpha, SCmax=SCmax, mode='continuous')
    return compute_rgas_from_dict(gas_values, config=config)
