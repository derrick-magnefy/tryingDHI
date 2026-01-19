"""
Rgas (DGA Reliability) Calculation
Based on MATLAB reference: DHI_computeRgas.m
DGA is the golden standard for transformer health.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

IEEE_THRESHOLDS = {
    'H2': [100, 200, 500, 700],
    'CH4': [75, 125, 200, 400],
    'C2H6': [65, 100, 150, 200],
    'C2H4': [50, 100, 200, 300],
    'C2H2': [3, 7, 35, 50],
    'CO': [350, 700, 1000, 1400],
    'CO2': [2500, 4000, 6000, 10000],
}


@dataclass
class RgasResult:
    Rgas: float
    health_state: str
    gas_scores: Dict[str, float]
    worst_gas: str
    worst_level: int
    composite_score: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'Rgas': self.Rgas, 'health_state': self.health_state,
            'gas_scores': self.gas_scores, 'worst_gas': self.worst_gas,
            'worst_level': self.worst_level, 'composite_score': self.composite_score,
            'timestamp': self.timestamp,
        }


def score_continuous(value: float, thresholds: List[float]) -> float:
    """MATLAB reference: starts at 1 for values below first threshold."""
    b1, b2, b3, b4 = thresholds
    eps = 1e-10
    if value <= b1:
        return 1.0
    elif value <= b2:
        return 1.0 + (value - b1) / max(eps, b2 - b1)
    elif value <= b3:
        return 2.0 + (value - b2) / max(eps, b3 - b2)
    elif value <= b4:
        return 3.0 + (value - b3) / max(eps, b4 - b3)
    else:
        overshoot = (value - b4) / max(eps, b4 - b3)
        return min(5.0, 4.0 + overshoot)


def compute_rgas(
    gas_values: Dict[str, float],
    alpha: float = 0.8,  # MATLAB default
    SCmax: float = 5.0,
    weights: Optional[Dict[str, float]] = None,
) -> RgasResult:
    if weights is None:
        weights = {'H2': 1.0, 'CH4': 1.0, 'C2H6': 1.0, 'C2H4': 1.0,
                   'C2H2': 1.2, 'CO': 1.0, 'CO2': 0.8}

    gas_scores = {}
    gas_levels = {}
    total_weight = 0.0
    weighted_score = 0.0

    for gas, value in gas_values.items():
        if gas not in IEEE_THRESHOLDS:
            continue
        thresholds = IEEE_THRESHOLDS[gas]
        weight = weights.get(gas, 1.0)
        score = score_continuous(value, thresholds)
        level = int(min(4, max(1, round(score))))
        gas_scores[gas] = score
        gas_levels[gas] = level
        weighted_score += score * weight
        total_weight += weight

    SC = weighted_score / total_weight if total_weight > 0 else 1.0
    worst_gas = max(gas_levels.keys(), key=lambda g: gas_levels[g]) if gas_levels else "none"
    worst_level = gas_levels.get(worst_gas, 1)

    Rgas = 1.0 - alpha * (SC - 1.0) / (SCmax - 1.0) if SCmax > 1 else 1.0
    Rgas = max(0.0, min(1.0, Rgas))

    if Rgas >= 0.85:
        health_state = "good"
    elif Rgas >= 0.70:
        health_state = "fair"
    elif Rgas >= 0.50:
        health_state = "poor"
    else:
        health_state = "critical"

    return RgasResult(
        Rgas=Rgas, health_state=health_state, gas_scores=gas_scores,
        worst_gas=worst_gas, worst_level=worst_level, composite_score=SC,
        timestamp=datetime.utcnow().isoformat(),
    )
