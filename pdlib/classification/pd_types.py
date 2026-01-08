"""
PD Type definitions.

Defines the partial discharge types and their characteristics.
"""

from typing import Dict, Any

# PD Type definitions with codes and descriptions
PD_TYPES: Dict[str, Dict[str, Any]] = {
    'NOISE': {
        'code': 0,
        'description': 'Non-PD or random noise signals',
        'characteristics': [
            'Identified as noise by DBSCAN clustering',
            'Low pulse count or erratic distribution',
            'High coefficient of variation',
            'No clear phase correlation',
        ]
    },
    'NOISE_MULTIPULSE': {
        'code': 5,
        'description': 'Multi-pulse waveform (multiple PD events in single acquisition)',
        'characteristics': [
            'Multiple distinct pulses detected in waveform',
            'Pulse count >= 2 per waveform window',
            'May indicate high PD activity or overlapping events',
            'Requires separation before individual classification',
        ]
    },
    'CORONA': {
        'code': 1,
        'description': 'Corona discharge (surface ionization in gas/air)',
        'characteristics': [
            'Highly asymmetric - predominantly in one half-cycle',
            'Phase concentrated near voltage peaks (0-180deg or 180-360deg)',
            'Fast rise times (<20ns typical)',
            'Higher amplitude variability',
            '"Rabbit ear" or "wing" pattern in PRPD',
        ]
    },
    'INTERNAL': {
        'code': 2,
        'description': 'Internal/void discharge (cavities in solid insulation)',
        'characteristics': [
            'Symmetric discharge in both half-cycles',
            'High cross-correlation between half-cycles (>0.7)',
            'Phase peaks near 90deg and 270deg (voltage peaks)',
            'Uniform amplitude distribution (Weibull beta > 2)',
            'Moderate rise times',
        ]
    },
    'SURFACE': {
        'code': 3,
        'description': 'Surface discharge (tracking/creeping discharge)',
        'characteristics': [
            'Activity near zero-crossings (0deg, 180deg)',
            'Moderate asymmetry',
            'May show tracking patterns',
            'Variable rise times',
            'Can transition to flashover',
        ]
    },
    'UNKNOWN': {
        'code': 4,
        'description': 'Unclassified PD pattern',
        'characteristics': [
            'Does not match known PD signatures',
            'May be mixed or transitional pattern',
            'Requires manual inspection',
        ]
    }
}

# PD type codes for quick lookup
PD_TYPE_CODES: Dict[str, int] = {
    name: info['code'] for name, info in PD_TYPES.items()
}

# Reverse lookup: code to name
PD_CODE_NAMES: Dict[int, str] = {
    info['code']: name for name, info in PD_TYPES.items()
}


def get_pd_type_info(pd_type: str) -> Dict[str, Any]:
    """Get information about a PD type."""
    return PD_TYPES.get(pd_type, PD_TYPES['UNKNOWN'])


def get_pd_type_code(pd_type: str) -> int:
    """Get the numeric code for a PD type."""
    return PD_TYPE_CODES.get(pd_type, PD_TYPE_CODES['UNKNOWN'])


def get_pd_type_name(code: int) -> str:
    """Get the PD type name from its numeric code."""
    return PD_CODE_NAMES.get(code, 'UNKNOWN')
