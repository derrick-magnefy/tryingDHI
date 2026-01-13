"""
Feature Extraction Pipeline

Extracts features from labeled datasets using wavelet detection and PDFeatureExtractor.
Applies noise floor threshold (K-value) to separate real PD from noise.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pre_middleware.wavelet_processing import WaveletProcessor
from pdlib.features import PDFeatureExtractor, FEATURE_NAMES
from pdlib.clustering import cluster_pulses, compute_cluster_features

from .labeled_loader import LabeledDataset


@dataclass
class ExtractedPulse:
    """Container for an extracted pulse with features."""
    index: int  # Sample index in original signal
    phase: float  # Phase angle (degrees)
    amplitude: float  # Signed peak amplitude
    snr_db: float  # Signal-to-noise ratio in dB
    above_noise_floor: bool  # True if amplitude > K * noise_std
    features: Dict[str, float]  # All extracted features
    waveform: np.ndarray  # Raw waveform data


@dataclass
class DatasetFeatures:
    """Container for all features extracted from a dataset."""
    dataset: LabeledDataset
    pulses: List[ExtractedPulse]
    noise_std: float
    k_threshold: float
    n_above_threshold: int
    n_below_threshold: int
    feature_matrix: np.ndarray  # Shape: (n_pulses, n_features)
    feature_names: List[str]


class FeaturePipeline:
    """
    Extracts features from labeled datasets.

    Pipeline:
    1. Estimate noise floor from signal
    2. Run wavelet detection to find pulses
    3. Apply K-threshold to identify above/below noise floor
    4. Extract features using PDFeatureExtractor
    5. Return structured feature data for calibration

    Usage:
        pipeline = FeaturePipeline(k_threshold=6.0)
        features = pipeline.extract(dataset)

        # Get only above-threshold pulses (likely real PD)
        real_pd_pulses = [p for p in features.pulses if p.above_noise_floor]
    """

    def __init__(
        self,
        k_threshold: float = 6.0,
        wavelet: str = 'db4',
        level: int = 3,
        pre_samples: int = 40,
        post_samples: int = 85,
        min_separation: int = 100,
    ):
        """
        Initialize the feature pipeline.

        Args:
            k_threshold: Noise floor multiplier (K-value). Pulses with
                        amplitude > K * noise_std are considered above noise floor.
                        Recommended: K=6 (conservative), K=7 (very conservative)
            wavelet: Wavelet type for detection
            level: Decomposition level
            pre_samples: Samples before peak for waveform extraction
            post_samples: Samples after peak for waveform extraction
            min_separation: Minimum samples between detections
        """
        self.k_threshold = k_threshold
        self.wavelet = wavelet
        self.level = level
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.min_separation = min_separation

    def estimate_noise_floor(self, signal: np.ndarray) -> float:
        """
        Estimate noise standard deviation using MAD (Median Absolute Deviation).

        This is robust to outliers (i.e., actual PD pulses).
        """
        # Use MAD for robust noise estimation
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        # Convert MAD to standard deviation (for normal distribution)
        noise_std = mad * 1.4826
        return noise_std

    def compute_phase(
        self,
        sample_index: int,
        sample_rate: float,
        ac_frequency: float
    ) -> float:
        """Compute phase angle in degrees for a sample index."""
        time = sample_index / sample_rate
        cycles = time * ac_frequency
        phase = (cycles % 1.0) * 360.0
        return phase

    def extract(self, dataset: LabeledDataset) -> DatasetFeatures:
        """
        Extract features from a dataset.

        Args:
            dataset: LabeledDataset to process

        Returns:
            DatasetFeatures containing all extracted pulse features
        """
        signal = dataset.signal
        sample_rate = dataset.sample_rate
        ac_frequency = dataset.ac_frequency

        # 1. Estimate noise floor
        noise_std = self.estimate_noise_floor(signal)
        amplitude_threshold = self.k_threshold * noise_std

        # 2. Run wavelet detection
        processor = WaveletProcessor(
            wavelet=self.wavelet,
            level=self.level,
            threshold_method='mad',
            k_factor=3.0,  # Use lower K for detection, filter later
        )

        detection_result = processor.process(signal, sample_rate)

        # Combine detections from all bands
        all_detections = []
        for band in ['D1', 'D2', 'D3']:
            band_dets = detection_result.get(f'{band}_detections', [])
            for det in band_dets:
                det['band'] = band
                all_detections.append(det)

        # Sort by index and remove duplicates (within min_separation)
        all_detections.sort(key=lambda x: x['index'])
        filtered_detections = []
        last_idx = -self.min_separation

        for det in all_detections:
            if det['index'] - last_idx >= self.min_separation:
                filtered_detections.append(det)
                last_idx = det['index']

        # 3. Extract waveforms and compute features
        sample_interval = 1.0 / sample_rate
        extractor = PDFeatureExtractor(
            sample_interval=sample_interval,
            ac_frequency=ac_frequency
        )

        pulses = []
        waveforms = []
        phases = []

        for det in filtered_detections:
            idx = det['index']

            # Find peak in search region
            search_start = max(0, idx - 25)
            search_end = min(len(signal), idx + 100)
            search_region = signal[search_start:search_end]

            if len(search_region) == 0:
                continue

            peak_offset = np.argmax(np.abs(search_region))
            peak_idx = search_start + peak_offset

            # Extract waveform
            start = max(0, peak_idx - self.pre_samples)
            end = min(len(signal), peak_idx + self.post_samples)
            waveform = signal[start:end]

            if len(waveform) < 20:
                continue

            # Get signed amplitude
            abs_wfm = np.abs(waveform)
            local_peak_idx = np.argmax(abs_wfm)
            signed_amplitude = waveform[local_peak_idx]
            abs_amplitude = abs(signed_amplitude)

            # Compute phase
            phase = self.compute_phase(peak_idx, sample_rate, ac_frequency)

            # Check noise floor
            above_threshold = abs_amplitude > amplitude_threshold

            # Compute SNR
            snr_db = 20 * np.log10(abs_amplitude / noise_std) if noise_std > 0 else 0

            waveforms.append(waveform)
            phases.append(phase)

            pulses.append(ExtractedPulse(
                index=peak_idx,
                phase=phase,
                amplitude=signed_amplitude,
                snr_db=snr_db,
                above_noise_floor=above_threshold,
                features={},  # Will be filled below
                waveform=waveform,
            ))

        # 4. Extract features using PDFeatureExtractor
        if waveforms:
            all_features = extractor.extract_all(
                waveforms,
                phase_angles=phases,
                normalize=True
            )

            # Build feature matrix and update pulse features
            feature_matrix = np.zeros((len(pulses), len(FEATURE_NAMES)))

            for i, (pulse, feat_dict) in enumerate(zip(pulses, all_features)):
                pulse.features = feat_dict
                for j, name in enumerate(FEATURE_NAMES):
                    feature_matrix[i, j] = feat_dict.get(name, 0.0)
        else:
            feature_matrix = np.array([]).reshape(0, len(FEATURE_NAMES))

        # Count above/below threshold
        n_above = sum(1 for p in pulses if p.above_noise_floor)
        n_below = len(pulses) - n_above

        return DatasetFeatures(
            dataset=dataset,
            pulses=pulses,
            noise_std=noise_std,
            k_threshold=self.k_threshold,
            n_above_threshold=n_above,
            n_below_threshold=n_below,
            feature_matrix=feature_matrix,
            feature_names=list(FEATURE_NAMES),
        )

    def extract_all(
        self,
        datasets: List[LabeledDataset],
        verbose: bool = True
    ) -> List[DatasetFeatures]:
        """
        Extract features from multiple datasets.

        Args:
            datasets: List of LabeledDataset objects
            verbose: Print progress

        Returns:
            List of DatasetFeatures
        """
        results = []

        for i, ds in enumerate(datasets):
            if verbose:
                print(f"[{i+1}/{len(datasets)}] Processing {ds.filename} ({ds.expected_type})...")

            features = self.extract(ds)
            results.append(features)

            if verbose:
                print(f"  Found {len(features.pulses)} pulses: "
                      f"{features.n_above_threshold} above K={self.k_threshold}, "
                      f"{features.n_below_threshold} below")

        return results


def prepare_training_data(
    dataset_features: List[DatasetFeatures],
    only_above_threshold: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Prepare training data from extracted features.

    Args:
        dataset_features: List of DatasetFeatures from pipeline
        only_above_threshold: If True, only use pulses above noise floor

    Returns:
        Tuple of:
        - feature_matrix: (n_samples, n_features) array
        - labels: (n_samples,) array of PD type labels
        - feature_names: List of feature names
        - unique_labels: List of unique PD types
    """
    all_features = []
    all_labels = []

    for df in dataset_features:
        expected_type = df.dataset.expected_type

        for i, pulse in enumerate(df.pulses):
            if only_above_threshold and not pulse.above_noise_floor:
                continue

            all_features.append(df.feature_matrix[i])
            all_labels.append(expected_type)

    if not all_features:
        return np.array([]), np.array([]), [], []

    feature_matrix = np.array(all_features)
    labels = np.array(all_labels)
    feature_names = dataset_features[0].feature_names if dataset_features else []
    unique_labels = sorted(set(all_labels))

    return feature_matrix, labels, feature_names, unique_labels


if __name__ == "__main__":
    # Test the pipeline
    from .labeled_loader import LabeledDatasetLoader

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "IEEE_Example_Data"

    # Load datasets
    loader = LabeledDatasetLoader(data_dir)
    datasets = loader.load_all()

    if not datasets:
        print("No datasets found!")
        sys.exit(1)

    # Extract features
    pipeline = FeaturePipeline(k_threshold=6.0)
    features = pipeline.extract_all(datasets)

    # Prepare training data
    X, y, feature_names, labels = prepare_training_data(features)
    print(f"\nTraining data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels: {labels}")
    for label in labels:
        print(f"  {label}: {np.sum(y == label)} samples")
