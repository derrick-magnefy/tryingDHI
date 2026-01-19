"""
Rpd (PD Reliability) Calculation
Based on MATLAB reference: DHI_computeRpd.m
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


@dataclass
class RpdResult:
    Rpd: float
    A: float
    RR: float
    A0: float
    RR0: float
    Ps: float
    defect_type: str
    pd_counts: Dict[str, int]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'Rpd': self.Rpd, 'A': self.A, 'RR': self.RR,
            'A0': self.A0, 'RR0': self.RR0, 'Ps': self.Ps,
            'defect_type': self.defect_type, 'pd_counts': self.pd_counts,
            'timestamp': self.timestamp,
        }


def compute_phase_severity(phase_range: Tuple[float, float]) -> float:
    if phase_range is None:
        return 1.0
    start, end = phase_range
    zero_crossings = [0, 180, 360]
    min_distance = float('inf')
    for zc in zero_crossings:
        midpoint = (start + end) / 2
        distance = min(abs(midpoint - zc), abs(midpoint - zc + 360), abs(midpoint - zc - 360))
        min_distance = min(min_distance, distance)
    if min_distance >= 45:
        return 1.0
    else:
        return 0.5 + 0.5 * (min_distance / 45.0)


def compute_rpd(
    A: float, RR: float, A0: float, RR0: float,
    phase_range: Optional[Tuple[float, float]] = None,
    defect_type: str = "unknown",
    pd_counts: Optional[Dict[str, int]] = None,
    Amax_factor: float = 10.0,
    RRmax_factor: float = 10.0,
) -> RpdResult:
    Amax = A0 * Amax_factor
    RRmax = RR0 * RRmax_factor
    A_clamped = min(A, Amax)
    RR_clamped = min(RR, RRmax)
    amp_factor = 1.0 - (A_clamped / Amax) if Amax > 0 else 1.0
    rr_factor = 1.0 - (RR_clamped / RRmax) if RRmax > 0 else 1.0
    Ps = compute_phase_severity(phase_range) if defect_type == "internal" and phase_range else 1.0
    Rpd = max(0.0, min(1.0, amp_factor * rr_factor * Ps))
    return RpdResult(
        Rpd=Rpd, A=A, RR=RR, A0=A0, RR0=RR0, Ps=Ps,
        defect_type=defect_type, pd_counts=pd_counts or {},
        timestamp=datetime.utcnow().isoformat(),
    )


def compute_rpd_from_pd_types(
    pd_types: Dict[str, Dict[str, Any]], A0: float, RR0: float,
) -> RpdResult:
    internal_count = pd_types.get('internal', {}).get('count', 0)
    surface_count = pd_types.get('surface', {}).get('count', 0)
    corona_count = pd_types.get('corona', {}).get('count', 0)
    pd_counts = {'internal': internal_count, 'surface': surface_count, 'corona': corona_count}
    
    if internal_count > surface_count:
        defect_type = "internal"
        primary_data = pd_types.get('internal', {})
    elif surface_count > 0:
        defect_type = "surface"
        primary_data = pd_types.get('surface', {})
    elif internal_count > 0:
        defect_type = "internal"
        primary_data = pd_types.get('internal', {})
    else:
        defect_type = "none"
        primary_data = {}

    A = 0.0
    for pd_type in ['internal', 'surface']:
        type_amp = pd_types.get(pd_type, {}).get('amplitude', 0)
        if type_amp and type_amp > A:
            A = type_amp

    RR = 0.0
    for pd_type in ['internal', 'surface']:
        type_rr = pd_types.get(pd_type, {}).get('pulses_per_second', 0)
        if type_rr:
            RR += type_rr

    phase_range = primary_data.get('phase_range')
    if isinstance(phase_range, (list, tuple)) and len(phase_range) >= 2:
        phase_range = (phase_range[0], phase_range[1])
    else:
        phase_range = None

    return compute_rpd(A=A, RR=RR, A0=A0, RR0=RR0, phase_range=phase_range,
                       defect_type=defect_type, pd_counts=pd_counts)
