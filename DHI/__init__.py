"""
DHI (Diagnostic Health Index) Calculation Module
"""

from .compute_rpd import compute_rpd, compute_rpd_from_pd_types, RpdResult
from .compute_rgas import compute_rgas, RgasResult
from .compute_dhi import (
    compute_dhi,
    compute_dhi_from_pd_types,
    compute_em_health_score,
    get_health_state,
    get_worst_health_state,
    DHIResult,
    EMHealthScore,
)

__all__ = [
    'compute_rpd', 'compute_rpd_from_pd_types', 'RpdResult',
    'compute_rgas', 'RgasResult',
    'compute_dhi', 'compute_dhi_from_pd_types', 'compute_em_health_score',
    'get_health_state', 'get_worst_health_state', 'DHIResult', 'EMHealthScore',
]
