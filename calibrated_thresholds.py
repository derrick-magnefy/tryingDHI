# Optimized thresholds from auto-calibration
# Accuracy: 9.38%
# K-Threshold: 6.0
# Generated: 2026-01-13T07:55:33.582210

CALIBRATED_THRESHOLDS = {
    'corona_internal': {
        'min_corona_score': 10,
        'min_internal_score': 10,
    },
    'phase_spread': {
        'surface_phase_spread_min': 20.000000,
    },
    'surface_detection': {
        'min_surface_score': 6,
    },
}

# Usage:
# classifier = PDTypeClassifier(thresholds=CALIBRATED_THRESHOLDS)