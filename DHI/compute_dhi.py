"""
Global DHI Calculation
DHI = Rpd * Rgas
Health state = WORST of Rpd and Rgas states (DGA is golden standard)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from .compute_rpd import RpdResult, compute_rpd_from_pd_types


@dataclass
class DHIResult:
    DHI: float
    Rpd: float
    Rgas: Optional[float]
    health_state: str
    rpd_state: str
    rgas_state: Optional[str]
    components: Dict[str, float]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'DHI': self.DHI, 'Rpd': self.Rpd, 'Rgas': self.Rgas,
            'health_state': self.health_state, 'rpd_state': self.rpd_state,
            'rgas_state': self.rgas_state, 'components': self.components,
            'timestamp': self.timestamp,
        }


def get_health_state(value: float) -> str:
    if value >= 0.85:
        return "good"
    elif value >= 0.70:
        return "fair"
    elif value >= 0.50:
        return "poor"
    else:
        return "critical"


HEALTH_STATE_SEVERITY = {"critical": 0, "poor": 1, "fair": 2, "good": 3}


def get_worst_health_state(*states: str) -> str:
    """DGA is golden standard - worst state wins."""
    worst = "good"
    worst_severity = 3
    for state in states:
        if state is None:
            continue
        severity = HEALTH_STATE_SEVERITY.get(state.lower(), 3)
        if severity < worst_severity:
            worst = state.lower()
            worst_severity = severity
    return worst


def compute_dhi(
    Rpd: float,
    Rgas: Optional[float] = None,
    additional_components: Optional[Dict[str, float]] = None,
) -> DHIResult:
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

    return DHIResult(
        DHI=DHI, Rpd=Rpd, Rgas=Rgas, health_state=health_state,
        rpd_state=rpd_state, rgas_state=rgas_state, components=components,
        timestamp=datetime.utcnow().isoformat(),
    )


def compute_dhi_from_pd_types(
    pd_types: Dict[str, Dict[str, Any]], A0: float, RR0: float,
    Rgas: Optional[float] = None,
) -> DHIResult:
    rpd_result = compute_rpd_from_pd_types(pd_types, A0, RR0)
    return compute_dhi(Rpd=rpd_result.Rpd, Rgas=Rgas)


@dataclass
class EMHealthScore:
    score: float
    health_state: str
    Rpd: float
    defect_type: str
    pd_types: Dict[str, Dict]
    trend: str
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score, 'health_state': self.health_state,
            'Rpd': self.Rpd, 'defect_type': self.defect_type,
            'pd_types': self.pd_types, 'trend': self.trend,
            'last_updated': self.last_updated,
        }


def compute_em_health_score(
    pd_types: Dict[str, Dict[str, Any]], A0: float, RR0: float,
    previous_Rpd: Optional[float] = None,
) -> EMHealthScore:
    rpd_result = compute_rpd_from_pd_types(pd_types, A0, RR0)
    score = rpd_result.Rpd * 100
    if previous_Rpd is None:
        trend = "stable"
    elif rpd_result.Rpd > previous_Rpd + 0.05:
        trend = "improving"
    elif rpd_result.Rpd < previous_Rpd - 0.05:
        trend = "degrading"
    else:
        trend = "stable"
    return EMHealthScore(
        score=round(score, 1), health_state=get_health_state(rpd_result.Rpd),
        Rpd=rpd_result.Rpd, defect_type=rpd_result.defect_type,
        pd_types=pd_types, trend=trend, last_updated=datetime.utcnow().isoformat(),
    )
