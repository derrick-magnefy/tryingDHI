"""
PD Type Classifier using decision tree.

Classifies partial discharge clusters into types:
- Noise: Non-PD signals identified by clustering or failing tests
- Corona: Asymmetric discharge in one half-cycle
- Internal: Symmetric discharge in both half-cycles
- Surface: Discharge near zero-crossings or with tracking patterns
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from .pd_types import PD_TYPES


class PDTypeClassifier:
    """
    Decision tree classifier for partial discharge types.

    The classifier uses a hierarchical decision tree with the following branches:

    Branch 1: Noise Detection
        - DBSCAN noise label (-1)
        - Multi-pulse detection
        - Score-based noise indicators

    Branch 2: Phase Spread Check
        - High phase spread indicates Surface PD

    Branch 3: Surface Detection
        - 8-feature weighted scoring

    Branch 4: Corona vs Internal
        - Score-based comparison of characteristics

    Branch 5: Fallback heuristics
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        selected_features: Optional[List[str]] = None
    ):
        """
        Initialize the classifier.

        Args:
            thresholds: Optional custom threshold configuration.
                       If None, uses default values.
            verbose: Whether to print detailed classification info
            selected_features: List of feature names to use (None = all)
        """
        self.verbose = verbose
        self.selected_features = selected_features
        self.classification_log = []

        # Load default thresholds
        self.thresholds = self._get_default_thresholds()

        # Override with custom thresholds if provided
        if thresholds:
            self._apply_custom_thresholds(thresholds)

    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Get default threshold values."""
        return {
            'noise_detection': {
                'dbscan_noise_label': -1,
                'min_spectral_flatness': 0.6,
                'max_bandwidth_3db': 2.0e6,
                'max_dominant_frequency': 1.0e6,
                'min_slew_rate': 1.0e6,
                'min_crest_factor': 3.0,
                'max_oscillation_count': 20,
                'min_signal_to_noise_ratio': 3.0,
                'min_cross_correlation': 0.3,
                'max_coefficient_of_variation': 2.0,
                'min_pulses_for_multipulse': 2,
                # New noise indicators for sub-threshold detection
                # NOTE: Tuned to avoid false positives on Surface PD (which has wide phase spread)
                'min_phase_entropy': 0.95,  # Very high entropy = nearly uniform = likely noise
                'max_amplitude_cv': 1.5,  # High CV = very inconsistent = likely noise
                'max_pulses_per_cycle': 20.0,  # Many pulses per cycle = likely noise
                'noise_score_threshold': 0.35,  # Balanced threshold
                # Amplitude/SNR based noise detection (primary differentiator for sub-threshold)
                'min_mean_snr_for_pd': 6.0,  # Real PD should have SNR > 6 dB
                'min_mean_amplitude_for_pd': 0.001,  # Real PD should have decent amplitude
            },
            'phase_spread': {
                'surface_phase_spread_min': 30.0,  # Lowered from 120 - Surface can have narrow phase spread
            },
            'surface_detection': {
                'weights': {'primary': 4, 'secondary': 3, 'mid': 2, 'supporting': 1},
                'min_surface_score': 8,
                'surface_phase_spread': 30.0,  # Lowered from 120 - Surface can have narrow phase spread
                'corona_phase_spread': 100.0,
                'surface_max_slew_rate': 5.0e6,
                'corona_min_slew_rate': 1.0e7,
                'surface_max_spectral_power_ratio': 0.5,
                'corona_min_spectral_power_ratio': 0.8,
                'surface_min_cv': 0.4,
                'corona_max_cv': 0.3,
                'surface_min_crest_factor': 4.0,
                'surface_max_crest_factor': 10.0,  # Expanded from 6.0 - Surface can have high crest factor
                'corona_min_crest_factor': 6.0,
                'surface_min_cross_corr': 0.4,
                'surface_max_cross_corr': 0.6,
                'corona_min_cross_corr': 0.7,
                'surface_min_spectral_flatness': 0.4,
                'surface_max_spectral_flatness': 0.5,
                'corona_max_spectral_flatness': 0.35,
                'surface_min_rep_rate_var': 0.5,
                'corona_max_rep_rate_var': 0.3,
                'surface_min_dominant_freq': 1.0e6,
                'surface_max_dominant_freq': 5.0e6,
            },
            'corona_internal': {
                'weights': {'primary': 4, 'secondary': 2, 'supporting': 1},
                'min_corona_score': 15,  # Raised from 8 - harder to classify as Corona
                'min_internal_score': 15,  # Raised from 8 - harder to classify as Internal
                'corona_neg_max_asymmetry': -0.6,
                'corona_pos_min_asymmetry': 0.6,
                'internal_min_asymmetry': -0.9,
                'internal_max_asymmetry': 0.9,
                'corona_neg_phase_min': 180,
                'corona_neg_phase_max': 270,
                'corona_pos_phase_q1_min': 0,
                'corona_pos_phase_q1_max': 90,
                'corona_pos_phase_q4_min': 270,
                'corona_pos_phase_q4_max': 360,
                'internal_phase_q1_min': 45,
                'internal_phase_q1_max': 90,
                'internal_phase_q3_min': 225,
                'internal_phase_q3_max': 270,
                'internal_min_amp_phase_corr': 0.5,
                'corona_max_amp_phase_corr': 0.3,
                'internal_min_spectral_power_low': 0.85,
                'corona_max_spectral_power_low': 0.60,
                'corona_min_slew_rate': 5.0e7,
                'internal_min_slew_rate': 1.0e7,
                'internal_max_slew_rate': 5.0e7,
                'corona_min_norm_slew_rate': 8.0,
                'internal_max_norm_slew_rate': 5.0,
                'corona_min_spectral_ratio': 1.5,
                'internal_min_spectral_ratio': 0.8,
                'internal_max_spectral_ratio': 1.5,
                'corona_min_oscillation': 90,
                'internal_max_oscillation': 90,
                'corona_min_crest_factor': 7.0,
                'internal_min_crest_factor': 4.0,
                'internal_max_crest_factor': 6.5,
                'corona_neg_min_dominant_freq': 1.5e7,
                'corona_pos_min_dominant_freq': 5.0e6,
                'corona_pos_max_dominant_freq': 1.5e7,
                'internal_min_dominant_freq': 5.0e6,
                'internal_max_dominant_freq': 3.0e7,
                'corona_max_cv': 0.15,
                'internal_min_cv': 0.15,
                'internal_max_cv': 0.35,
                'corona_neg_min_q3_pct': 55,
                'internal_min_q3_pct': 35,
                'internal_max_q3_pct': 50,
                'corona_min_rep_rate': 100,
                'internal_min_rep_rate': 20,
                'internal_max_rep_rate': 100,
            },
        }

    def _apply_custom_thresholds(self, custom: Dict[str, Any]) -> None:
        """Apply custom threshold overrides."""
        for section, values in custom.items():
            if section in self.thresholds:
                if isinstance(values, dict):
                    self.thresholds[section].update(values)
                else:
                    self.thresholds[section] = values

    def _is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature should be used in classification."""
        if self.selected_features is None:
            return True
        return feature_name in self.selected_features

    def _get_feature(
        self,
        cluster_features: Dict[str, float],
        feature_name: str,
        default: float = 0.0
    ) -> float:
        """Get a feature value if enabled, otherwise return default."""
        if not self._is_feature_enabled(feature_name):
            return default
        return cluster_features.get(feature_name, default)

    def classify(
        self,
        cluster_features: Dict[str, float],
        cluster_label: int,
        pulse_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Classify a cluster into a PD type.

        Args:
            cluster_features: Dict of aggregated cluster features
            cluster_label: Original cluster label (e.g., -1 for noise)
            pulse_features: Optional array of individual pulse features

        Returns:
            Classification result with type, confidence, and reasoning
        """
        result = {
            'cluster_label': cluster_label,
            'pd_type': 'UNKNOWN',
            'pd_type_code': PD_TYPES['UNKNOWN']['code'],
            'confidence': 0.0,
            'branch_path': [],
            'reasoning': [],
            'warnings': [],
        }

        noise_cfg = self.thresholds['noise_detection']
        phase_cfg = self.thresholds['phase_spread']
        surface_cfg = self.thresholds['surface_detection']
        ci_cfg = self.thresholds['corona_internal']

        # Calculate total pulse count
        n_pulses = (
            self._get_feature(cluster_features, 'pulses_per_positive_halfcycle', 0) +
            self._get_feature(cluster_features, 'pulses_per_negative_halfcycle', 0)
        )

        # ===== BRANCH 1: NOISE DETECTION =====
        result['branch_path'].append('Branch 1: Noise Detection')

        # Check DBSCAN noise label
        if cluster_label == noise_cfg['dbscan_noise_label']:
            result['pd_type'] = 'NOISE'
            result['pd_type_code'] = PD_TYPES['NOISE']['code']
            result['confidence'] = 0.95
            result['reasoning'].append(f"DBSCAN noise label={cluster_label}")
            return result

        # Check for multi-pulse waveforms
        pulses_per_waveform = self._get_feature(
            cluster_features, 'mean_pulse_count',
            self._get_feature(cluster_features, 'pulses_per_waveform', 1)
        )
        is_multi_pulse = self._get_feature(cluster_features, 'mean_is_multi_pulse', 0)

        if is_multi_pulse > 0.5 or pulses_per_waveform >= noise_cfg['min_pulses_for_multipulse']:
            result['pd_type'] = 'NOISE_MULTIPULSE'
            result['pd_type_code'] = PD_TYPES['NOISE_MULTIPULSE']['code']
            result['confidence'] = 0.90
            result['reasoning'].append(
                f"Multi-pulse: {pulses_per_waveform:.1f} pulses/waveform"
            )
            return result

        # Score-based noise detection
        noise_score, noise_reasons = self._compute_noise_score(cluster_features, noise_cfg)
        noise_threshold = noise_cfg.get('noise_score_threshold', 0.30)

        if noise_score >= noise_threshold:
            result['pd_type'] = 'NOISE'
            result['pd_type_code'] = PD_TYPES['NOISE']['code']
            result['confidence'] = min(0.5 + noise_score, 0.95)
            result['reasoning'].append(f"Noise indicators: {', '.join(noise_reasons)}")
            return result

        if noise_score > noise_threshold * 0.5:
            result['warnings'].append(f"Partial noise indicators (score={noise_score:.2f})")

        result['reasoning'].append(f"Passed noise detection (score={noise_score:.2f})")

        # ===== BRANCH 2: PHASE SPREAD CHECK =====
        result['branch_path'].append('Branch 2: Phase Spread')

        phase_spread = self._get_feature(cluster_features, 'phase_spread', 0)
        result['reasoning'].append(f"Phase spread: {phase_spread:.1f}deg")

        if phase_spread > phase_cfg['surface_phase_spread_min']:
            result['pd_type'] = 'SURFACE'
            result['pd_type_code'] = PD_TYPES['SURFACE']['code']
            result['confidence'] = 0.85
            result['reasoning'].append(
                f"SURFACE: phase_spread={phase_spread:.1f}° > {phase_cfg['surface_phase_spread_min']}°"
            )
            return result

        # ===== BRANCH 3: SURFACE DETECTION =====
        result['branch_path'].append('Branch 3: Surface Detection')

        surface_score, surface_indicators, max_surface_score = self._compute_surface_score(
            cluster_features, phase_spread, surface_cfg
        )

        min_surface = surface_cfg['min_surface_score']
        result['reasoning'].append(
            f"Surface score: {surface_score}/{max_surface_score} (need {min_surface})"
        )

        if surface_score >= min_surface:
            result['pd_type'] = 'SURFACE'
            result['pd_type_code'] = PD_TYPES['SURFACE']['code']
            result['confidence'] = min(0.5 + (surface_score / max_surface_score) * 0.4, 0.90)
            result['reasoning'].append(f"SURFACE: {', '.join(surface_indicators[:3])}...")
            return result

        # ===== BRANCH 4: CORONA VS INTERNAL =====
        result['branch_path'].append('Branch 4: Corona vs Internal')

        corona_score, internal_score, corona_ind, internal_ind, max_ci_score = (
            self._compute_corona_internal_scores(cluster_features, ci_cfg)
        )

        min_corona = ci_cfg['min_corona_score']
        min_internal = ci_cfg['min_internal_score']

        result['reasoning'].append(f"Corona: {corona_score}/{max_ci_score} (need {min_corona})")
        result['reasoning'].append(f"Internal: {internal_score}/{max_ci_score} (need {min_internal})")

        # Classification based on scores
        if corona_score >= min_corona and corona_score > internal_score:
            result['pd_type'] = 'CORONA'
            result['pd_type_code'] = PD_TYPES['CORONA']['code']
            result['confidence'] = min(0.5 + (corona_score / max_ci_score) * 0.4, 0.95)
            result['reasoning'].append(f"CORONA: {', '.join(corona_ind[:3])}...")
            return result

        if internal_score >= min_internal and internal_score > corona_score:
            result['pd_type'] = 'INTERNAL'
            result['pd_type_code'] = PD_TYPES['INTERNAL']['code']
            result['confidence'] = min(0.5 + (internal_score / max_ci_score) * 0.4, 0.95)
            result['reasoning'].append(f"INTERNAL: {', '.join(internal_ind[:3])}...")
            return result

        # Tie-breaker
        if corona_score >= min_corona and internal_score >= min_internal:
            asymmetry = self._get_feature(cluster_features, 'discharge_asymmetry', 0)
            if asymmetry < -0.2:
                result['pd_type'] = 'CORONA'
                result['pd_type_code'] = PD_TYPES['CORONA']['code']
                result['confidence'] = 0.60
            else:
                result['pd_type'] = 'INTERNAL'
                result['pd_type_code'] = PD_TYPES['INTERNAL']['code']
                result['confidence'] = 0.60
            result['reasoning'].append(f"Tie-breaker: asymmetry={asymmetry:.2f}")
            return result

        # Fallback: weak classification
        if corona_score > internal_score and corona_score >= min_corona / 2:
            result['pd_type'] = 'CORONA'
            result['pd_type_code'] = PD_TYPES['CORONA']['code']
            result['confidence'] = 0.45
            result['reasoning'].append(f"Weak CORONA: {corona_score}/{max_ci_score}")
            return result

        if internal_score > corona_score and internal_score >= min_internal / 2:
            result['pd_type'] = 'INTERNAL'
            result['pd_type_code'] = PD_TYPES['INTERNAL']['code']
            result['confidence'] = 0.45
            result['reasoning'].append(f"Weak INTERNAL: {internal_score}/{max_ci_score}")
            return result

        # Unknown
        result['pd_type'] = 'UNKNOWN'
        result['pd_type_code'] = PD_TYPES['UNKNOWN']['code']
        result['confidence'] = 0.3
        result['reasoning'].append(f"Unclear pattern: C={corona_score}, I={internal_score}")
        result['warnings'].append("Manual review recommended")

        return result

    def _compute_noise_score(
        self,
        features: Dict[str, float],
        cfg: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Compute noise detection score."""
        score = 0.0
        reasons = []

        spectral_flatness = self._get_feature(features, 'mean_spectral_flatness',
                                              self._get_feature(features, 'spectral_flatness', 0))
        if spectral_flatness > cfg['min_spectral_flatness']:
            score += 0.15
            reasons.append(f"spectral_flatness={spectral_flatness:.2f}")

        slew_rate = self._get_feature(features, 'mean_slew_rate',
                                      self._get_feature(features, 'slew_rate', 1e9))
        if slew_rate < cfg['min_slew_rate']:
            score += 0.15
            reasons.append(f"slow_slew={slew_rate:.2e}")

        crest_factor = self._get_feature(features, 'mean_crest_factor',
                                         self._get_feature(features, 'crest_factor', 10))
        if crest_factor < cfg['min_crest_factor']:
            score += 0.15
            reasons.append(f"low_crest={crest_factor:.2f}")

        cross_corr = self._get_feature(features, 'cross_correlation', 0.5)
        if cross_corr < cfg['min_cross_correlation']:
            score += 0.10
            reasons.append(f"low_corr={cross_corr:.2f}")

        oscillation = self._get_feature(features, 'mean_oscillation_count',
                                        self._get_feature(features, 'oscillation_count', 5))
        if oscillation > cfg['max_oscillation_count']:
            score += 0.10
            reasons.append(f"oscillations={oscillation}")

        snr = self._get_feature(features, 'mean_signal_to_noise_ratio',
                                self._get_feature(features, 'signal_to_noise_ratio', 10))
        if snr < cfg['min_signal_to_noise_ratio']:
            score += 0.15
            reasons.append(f"low_snr={snr:.2f}")

        cv = self._get_feature(features, 'coefficient_of_variation', 0)
        if cv > cfg['max_coefficient_of_variation']:
            score += 0.10
            reasons.append(f"high_cv={cv:.2f}")

        bandwidth = self._get_feature(features, 'mean_bandwidth_3db',
                                      self._get_feature(features, 'bandwidth_3db', 1e9))
        if bandwidth < cfg['max_bandwidth_3db']:
            score += 0.05
            reasons.append(f"narrowband={bandwidth:.2e}")

        dom_freq = self._get_feature(features, 'mean_dominant_frequency',
                                     self._get_feature(features, 'dominant_frequency', 1e6))
        if dom_freq < cfg['max_dominant_frequency']:
            score += 0.10
            reasons.append(f"low_freq={dom_freq:.0f}")

        # === NEW NOISE INDICATORS FOR SUB-THRESHOLD DETECTION ===
        # NOTE: Lower weights to avoid false positives on Surface PD

        # 1. Phase distribution uniformity (high entropy = random/uniform = likely noise)
        # Real PD has characteristic phase patterns; noise is uniformly distributed
        # Surface PD has wide spread but NOT perfectly uniform - threshold set high
        phase_entropy = self._get_feature(features, 'phase_entropy', 0)
        if phase_entropy > cfg.get('min_phase_entropy', 0.95):
            score += 0.10  # Reduced from 0.15
            reasons.append(f"uniform_phase={phase_entropy:.2f}")

        # 2. Amplitude consistency (high CV in amplitudes = inconsistent = likely noise)
        # Real PD clusters have relatively consistent amplitudes; noise is random
        # Surface PD can have variable amplitudes - threshold set higher
        amp_cv = self._get_feature(features, 'amplitude_coefficient_of_variation',
                                   self._get_feature(features, 'amp_cv', 0))
        if amp_cv > cfg.get('max_amplitude_cv', 1.5):
            score += 0.08  # Reduced from 0.10
            reasons.append(f"amp_inconsistent={amp_cv:.2f}")

        # 3. Pulses per cycle (too many pulses per AC cycle = likely noise)
        # Real PD typically has limited pulses per cycle; excessive count suggests noise
        pulses_per_cycle = self._get_feature(features, 'pulses_per_cycle', 0)
        if pulses_per_cycle > cfg.get('max_pulses_per_cycle', 20.0):
            score += 0.10  # Reduced from 0.15
            reasons.append(f"high_pulse_rate={pulses_per_cycle:.1f}/cycle")

        # 4. Amplitude/SNR based noise detection (PRIMARY for sub-threshold)
        # Real PD should have decent amplitude and SNR even if below trigger threshold
        # This is a key differentiator: Surface PD has HIGH amplitude, noise has LOW amplitude
        mean_snr = self._get_feature(features, 'mean_signal_to_noise_ratio',
                                     self._get_feature(features, 'signal_to_noise_ratio', 10))
        mean_amp = self._get_feature(features, 'mean_absolute_amplitude',
                                     self._get_feature(features, 'absolute_amplitude', 0.01))

        min_snr_for_pd = cfg.get('min_mean_snr_for_pd', 6.0)
        min_amp_for_pd = cfg.get('min_mean_amplitude_for_pd', 0.001)

        # Low SNR is a strong noise indicator
        if mean_snr < min_snr_for_pd:
            score += 0.15
            reasons.append(f"low_mean_snr={mean_snr:.1f}dB")

        # Very low amplitude combined with other noise indicators
        if mean_amp < min_amp_for_pd:
            score += 0.10
            reasons.append(f"low_mean_amp={mean_amp:.2e}")

        return score, reasons

    def _compute_surface_score(
        self,
        features: Dict[str, float],
        phase_spread: float,
        cfg: Dict[str, Any]
    ) -> Tuple[int, List[str], int]:
        """Compute surface PD detection score."""
        weights = cfg['weights']
        score = 0
        indicators = []

        # Primary: phase spread (activity across phases)
        if phase_spread > cfg['surface_phase_spread']:
            score += weights['primary']
            indicators.append(f"phase_spread={phase_spread:.1f}°")

        # Secondary features
        slew_rate = self._get_feature(features, 'mean_slew_rate',
                                      self._get_feature(features, 'slew_rate', 1e7))
        if slew_rate < cfg['surface_max_slew_rate']:
            score += weights['secondary']
            indicators.append(f"low_slew={slew_rate:.2e}")

        spectral_ratio = self._get_feature(features, 'spectral_power_ratio',
                                           self._get_feature(features, 'mean_spectral_power_ratio', 0.5))
        if spectral_ratio < cfg['surface_max_spectral_power_ratio']:
            score += weights['secondary']
            indicators.append(f"low_ratio={spectral_ratio:.2f}")

        cv = self._get_feature(features, 'coefficient_of_variation', 0.3)
        if cv > cfg['surface_min_cv']:
            score += weights['secondary']
            indicators.append(f"high_cv={cv:.2f}")

        # Mid features
        crest_factor = self._get_feature(features, 'mean_crest_factor',
                                         self._get_feature(features, 'crest_factor', 5))
        if cfg['surface_min_crest_factor'] <= crest_factor <= cfg['surface_max_crest_factor']:
            score += weights['mid']
            indicators.append(f"mod_crest={crest_factor:.1f}")

        cross_corr = self._get_feature(features, 'cross_correlation', 0.5)
        if cfg['surface_min_cross_corr'] <= cross_corr <= cfg['surface_max_cross_corr']:
            score += weights['mid']
            indicators.append(f"lower_corr={cross_corr:.2f}")

        # Supporting features
        spectral_flat = self._get_feature(features, 'mean_spectral_flatness',
                                          self._get_feature(features, 'spectral_flatness', 0.4))
        if cfg['surface_min_spectral_flatness'] <= spectral_flat <= cfg['surface_max_spectral_flatness']:
            score += weights['supporting']
            indicators.append(f"flatness={spectral_flat:.2f}")

        rep_var = self._get_feature(features, 'repetition_rate_variance', 0.4)
        if rep_var > cfg['surface_min_rep_rate_var']:
            score += weights['supporting']
            indicators.append(f"rep_var={rep_var:.2f}")

        dom_freq = self._get_feature(features, 'mean_dominant_frequency',
                                     self._get_feature(features, 'dominant_frequency', 3e6))
        if cfg['surface_min_dominant_freq'] <= dom_freq <= cfg['surface_max_dominant_freq']:
            score += weights['supporting']
            indicators.append(f"freq={dom_freq/1e6:.1f}MHz")

        max_score = (
            1 * weights['primary'] +  # phase_spread only
            3 * weights['secondary'] +
            2 * weights['mid'] +
            3 * weights['supporting']
        )

        return score, indicators, max_score

    def _compute_corona_internal_scores(
        self,
        features: Dict[str, float],
        cfg: Dict[str, Any]
    ) -> Tuple[int, int, List[str], List[str], int]:
        """Compute Corona and Internal detection scores."""
        weights = cfg['weights']
        corona_score = 0
        internal_score = 0
        corona_ind = []
        internal_ind = []

        # Primary: asymmetry
        asymmetry = self._get_feature(features, 'discharge_asymmetry', 0)
        if asymmetry < cfg['corona_neg_max_asymmetry']:
            corona_score += weights['primary']
            corona_ind.append(f"asym={asymmetry:.2f}")
        elif asymmetry > cfg['corona_pos_min_asymmetry']:
            corona_score += weights['primary']
            corona_ind.append(f"asym={asymmetry:.2f}")
        if cfg['internal_min_asymmetry'] <= asymmetry <= cfg['internal_max_asymmetry']:
            internal_score += weights['primary']
            internal_ind.append(f"asym={asymmetry:.2f}")

        # Primary: phase of max activity
        phase_max = self._get_feature(features, 'phase_of_max_activity', 0)
        if cfg['corona_neg_phase_min'] <= phase_max <= cfg['corona_neg_phase_max']:
            corona_score += weights['primary']
            corona_ind.append(f"phase={phase_max:.0f}°")
        elif ((cfg['corona_pos_phase_q1_min'] <= phase_max <= cfg['corona_pos_phase_q1_max']) or
              (cfg['corona_pos_phase_q4_min'] <= phase_max <= cfg['corona_pos_phase_q4_max'])):
            corona_score += weights['primary']
            corona_ind.append(f"phase={phase_max:.0f}°")

        if ((cfg['internal_phase_q1_min'] <= phase_max <= cfg['internal_phase_q1_max']) or
            (cfg['internal_phase_q3_min'] <= phase_max <= cfg['internal_phase_q3_max'])):
            internal_score += weights['primary']
            internal_ind.append(f"phase={phase_max:.0f}°")

        # Primary: amplitude-phase correlation
        amp_corr = self._get_feature(features, 'amplitude_phase_correlation', 0)
        if amp_corr >= cfg['internal_min_amp_phase_corr']:
            internal_score += weights['primary']
            internal_ind.append(f"amp_corr={amp_corr:.2f}")
        if amp_corr <= cfg['corona_max_amp_phase_corr']:
            corona_score += weights['primary']
            corona_ind.append(f"amp_corr={amp_corr:.2f}")

        # Primary: spectral power low
        spec_low = self._get_feature(features, 'mean_spectral_power_low',
                                     self._get_feature(features, 'spectral_power_low', 0.5))
        if spec_low >= cfg['internal_min_spectral_power_low']:
            internal_score += weights['primary']
            internal_ind.append(f"spec_low={spec_low:.2f}")
        if spec_low <= cfg['corona_max_spectral_power_low']:
            corona_score += weights['primary']
            corona_ind.append(f"spec_low={spec_low:.2f}")

        # Secondary: slew rate
        slew = self._get_feature(features, 'mean_slew_rate',
                                 self._get_feature(features, 'slew_rate', 1e7))
        if slew > cfg['corona_min_slew_rate']:
            corona_score += weights['secondary']
            corona_ind.append(f"slew={slew:.1e}")
        if cfg['internal_min_slew_rate'] <= slew <= cfg['internal_max_slew_rate']:
            internal_score += weights['secondary']
            internal_ind.append(f"slew={slew:.1e}")

        # Secondary: oscillation count
        osc = self._get_feature(features, 'mean_oscillation_count',
                                self._get_feature(features, 'oscillation_count', 5))
        if osc >= cfg['corona_min_oscillation']:
            corona_score += weights['secondary']
            corona_ind.append(f"osc={osc:.0f}")
        if osc < cfg['internal_max_oscillation']:
            internal_score += weights['secondary']
            internal_ind.append(f"osc={osc:.0f}")

        # Secondary: crest factor
        crest = self._get_feature(features, 'mean_crest_factor',
                                  self._get_feature(features, 'crest_factor', 5))
        if crest >= cfg['corona_min_crest_factor']:
            corona_score += weights['secondary']
            corona_ind.append(f"crest={crest:.1f}")
        if cfg['internal_min_crest_factor'] <= crest <= cfg['internal_max_crest_factor']:
            internal_score += weights['secondary']
            internal_ind.append(f"crest={crest:.1f}")

        # Secondary: spectral power ratio
        spec_ratio = self._get_feature(features, 'spectral_power_ratio',
                                       self._get_feature(features, 'mean_spectral_power_ratio', 1.0))
        if spec_ratio > cfg.get('corona_min_spectral_ratio', 1.5):
            corona_score += weights['secondary']
            corona_ind.append(f"spec_ratio={spec_ratio:.2f}")
        if cfg.get('internal_min_spectral_ratio', 0.8) <= spec_ratio <= cfg.get('internal_max_spectral_ratio', 1.5):
            internal_score += weights['secondary']
            internal_ind.append(f"spec_ratio={spec_ratio:.2f}")

        # Secondary: normalized slew rate
        norm_slew = self._get_feature(features, 'mean_norm_slew_rate',
                                      self._get_feature(features, 'norm_slew_rate', 3.0))
        if norm_slew >= cfg.get('corona_min_norm_slew_rate', 8.0):
            corona_score += weights['secondary']
            corona_ind.append(f"norm_slew={norm_slew:.1f}")
        if norm_slew <= cfg.get('internal_max_norm_slew_rate', 5.0):
            internal_score += weights['secondary']
            internal_ind.append(f"norm_slew={norm_slew:.1f}")

        # Secondary: dominant frequency
        dom_freq = self._get_feature(features, 'mean_dominant_frequency',
                                     self._get_feature(features, 'dominant_frequency', 8e6))
        # Check for negative corona high frequency
        if dom_freq >= cfg.get('corona_neg_min_dominant_freq', 1.5e7):
            corona_score += weights['secondary']
            corona_ind.append(f"freq={dom_freq/1e6:.1f}MHz")
        # Check for internal frequency range
        if cfg.get('internal_min_dominant_freq', 5e6) <= dom_freq <= cfg.get('internal_max_dominant_freq', 3e7):
            internal_score += weights['secondary']
            internal_ind.append(f"freq={dom_freq/1e6:.1f}MHz")

        # Supporting: CV
        cv = self._get_feature(features, 'coefficient_of_variation', 0.2)
        if cv < cfg['corona_max_cv']:
            corona_score += weights['supporting']
            corona_ind.append(f"cv={cv:.2f}")
        if cfg['internal_min_cv'] <= cv <= cfg['internal_max_cv']:
            internal_score += weights['supporting']
            internal_ind.append(f"cv={cv:.2f}")

        # Supporting: Q3 percentage
        q3 = self._get_feature(features, 'quadrant_3_percentage', 0)
        if q3 > cfg['corona_neg_min_q3_pct']:
            corona_score += weights['supporting']
            corona_ind.append(f"q3={q3:.1f}%")
        if cfg['internal_min_q3_pct'] <= q3 <= cfg['internal_max_q3_pct']:
            internal_score += weights['supporting']
            internal_ind.append(f"q3={q3:.1f}%")

        # Supporting: repetition rate
        rep_rate = self._get_feature(features, 'repetition_rate',
                                     self._get_feature(features, 'pulses_per_cycle', 50))
        if rep_rate > cfg.get('corona_min_rep_rate', 100):
            corona_score += weights['supporting']
            corona_ind.append(f"rep={rep_rate:.0f}")
        if cfg.get('internal_min_rep_rate', 20) <= rep_rate <= cfg.get('internal_max_rep_rate', 100):
            internal_score += weights['supporting']
            internal_ind.append(f"rep={rep_rate:.0f}")

        # Max score calculation: 4 primary + 6 secondary + 3 supporting features
        max_score = 4 * weights['primary'] + 6 * weights['secondary'] + 3 * weights['supporting']

        return corona_score, internal_score, corona_ind, internal_ind, max_score

    def classify_all(
        self,
        cluster_features_dict: Dict[int, Dict[str, float]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Classify all clusters in a dataset.

        Args:
            cluster_features_dict: Dict mapping cluster labels to feature dicts

        Returns:
            Dict mapping cluster labels to classification results
        """
        results = {}
        for label, features in cluster_features_dict.items():
            results[label] = self.classify(features, label)
        return results
