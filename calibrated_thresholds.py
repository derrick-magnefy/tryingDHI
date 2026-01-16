# Optimized thresholds from auto-calibration
# Accuracy: 99.39%
# K-Threshold: 6.0
# Generated: 2026-01-13T08:18:41.993442

CALIBRATED_THRESHOLDS = {
    'corona_internal': {
        'corona_max_amp_phase_corr': 0.200000,
        'corona_max_spectral_power_low': 0.650000,
        'corona_min_slew_rate': 50000000,
        'corona_neg_max_asymmetry': -0.750000,
        'corona_pos_min_asymmetry': 0.650000,
        'internal_max_asymmetry': 0.650000,
        'internal_max_slew_rate': 55000000,
        'internal_min_amp_phase_corr': 0.500000,
        'internal_min_asymmetry': -0.850000,
        'internal_min_slew_rate': 10000000,
        'internal_min_spectral_power_low': 0.900000,
        'min_corona_score': 9,
        'min_internal_score': 8,
    },
    'noise_detection': {
        'max_amplitude_cv': 2.750000,
        'max_pulses_per_cycle': 10,
        'min_crest_factor': 2.500000,
        'min_mean_snr_for_pd': 6.500000,
        'min_phase_entropy': 0.990000,
        'min_pulses_for_multipulse': 6,
        'min_slew_rate': 700000,
        'min_spectral_flatness': 0.700000,
        'noise_score_threshold': 0.450000,
    },
    'phase_spread': {
        'surface_phase_spread_min': 110,
    },
    'surface_detection': {
        'min_surface_score': 4,
        'surface_max_phase_entropy': 0.800000,
        'surface_max_slew_rate': 3000000,
        'surface_min_snr_for_focused': 4.500000,
        'surface_phase_spread': 150,
    },
}

# Usage:
# classifier = PDTypeClassifier(thresholds=CALIBRATED_THRESHOLDS)