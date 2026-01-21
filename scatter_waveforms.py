"""
Scatter Waveforms Storage

Stores waveform data for scatter plot visualization with click-to-view functionality.
Each waveform includes:
- Phase angle (x-axis)
- Amplitude (y-axis)
- PD type classification
- Cluster ID
- Actual waveform samples for detailed view

Designed to be cached in browser IndexedDB for fast access.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import gzip


@dataclass
class ScatterWaveform:
    """Single waveform for scatter plot."""
    id: int                      # Unique ID within dataset
    phase: float                 # Phase angle in degrees (0-360)
    amplitude: float             # Peak amplitude
    pd_type: str                 # Classification: internal, surface, corona, noise
    cluster_id: int              # Cluster assignment
    samples: List[float]         # Actual waveform samples
    file_key: Optional[str]      # Source file S3 key
    file_index: Optional[int]    # Index within source file


@dataclass
class ScatterDataset:
    """Collection of waveforms for one transformer/file."""
    transformer_id: str
    file_key: Optional[str]      # Source file (None for aggregated)
    timestamp: str               # When data was recorded
    total_waveforms: int
    waveform_length: int         # Samples per waveform
    sample_rate: float           # Sample rate in Hz
    waveforms: List[ScatterWaveform]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'transformer_id': self.transformer_id,
            'file_key': self.file_key,
            'timestamp': self.timestamp,
            'metadata': {
                'total_waveforms': self.total_waveforms,
                'waveform_length': self.waveform_length,
                'sample_rate': self.sample_rate,
            },
            'waveforms': [
                {
                    'id': w.id,
                    'phase': w.phase,
                    'amplitude': w.amplitude,
                    'pd_type': w.pd_type,
                    'cluster_id': w.cluster_id,
                    'samples': w.samples,
                    'file_key': w.file_key,
                    'file_index': w.file_index,
                }
                for w in self.waveforms
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScatterDataset':
        """Create from dictionary."""
        metadata = data.get('metadata', {})
        waveforms = [
            ScatterWaveform(
                id=w['id'],
                phase=w['phase'],
                amplitude=w['amplitude'],
                pd_type=w.get('pd_type', 'unknown'),
                cluster_id=w.get('cluster_id', 0),
                samples=w.get('samples', []),
                file_key=w.get('file_key'),
                file_index=w.get('file_index'),
            )
            for w in data.get('waveforms', [])
        ]

        return cls(
            transformer_id=data.get('transformer_id', ''),
            file_key=data.get('file_key'),
            timestamp=data.get('timestamp', ''),
            total_waveforms=metadata.get('total_waveforms', len(waveforms)),
            waveform_length=metadata.get('waveform_length', 0),
            sample_rate=metadata.get('sample_rate', 250e6),
            waveforms=waveforms,
        )

    def to_json(self, compress: bool = True) -> bytes:
        """Serialize to JSON bytes."""
        json_str = json.dumps(self.to_dict())
        data = json_str.encode('utf-8')

        if compress:
            data = gzip.compress(data)

        return data

    @classmethod
    def from_json(cls, data: bytes, compressed: bool = True) -> 'ScatterDataset':
        """Deserialize from JSON bytes."""
        if compressed:
            data = gzip.decompress(data)

        json_data = json.loads(data.decode('utf-8'))
        return cls.from_dict(json_data)


class ScatterWaveformBuilder:
    """
    Builder for creating scatter waveform datasets from processing results.

    Extracts waveforms from ProcessingResult objects and builds
    ScatterDataset for visualization.
    """

    def __init__(
        self,
        transformer_id: str,
        waveform_length: int = 1000,
        sample_rate: float = 250e6,
    ):
        """
        Initialize builder.

        Args:
            transformer_id: Transformer ID
            waveform_length: Number of samples per waveform
            sample_rate: Sample rate in Hz
        """
        self.transformer_id = transformer_id
        self.waveform_length = waveform_length
        self.sample_rate = sample_rate
        self.waveforms: List[ScatterWaveform] = []
        self.next_id = 0

    def add_waveform(
        self,
        phase: float,
        amplitude: float,
        samples: np.ndarray,
        pd_type: str = 'unknown',
        cluster_id: int = 0,
        file_key: Optional[str] = None,
        file_index: Optional[int] = None,
    ):
        """
        Add a waveform to the dataset.

        Args:
            phase: Phase angle in degrees
            amplitude: Peak amplitude
            samples: Waveform sample array
            pd_type: PD classification
            cluster_id: Cluster assignment
            file_key: Source file S3 key
            file_index: Index within source file
        """
        # Truncate or pad samples to waveform_length
        if len(samples) > self.waveform_length:
            samples = samples[:self.waveform_length]
        elif len(samples) < self.waveform_length:
            samples = np.pad(samples, (0, self.waveform_length - len(samples)))

        waveform = ScatterWaveform(
            id=self.next_id,
            phase=float(phase),
            amplitude=float(amplitude),
            pd_type=pd_type,
            cluster_id=cluster_id,
            samples=samples.tolist(),
            file_key=file_key,
            file_index=file_index,
        )

        self.waveforms.append(waveform)
        self.next_id += 1

    def add_from_processing_result(
        self,
        result: Any,  # ProcessingResult
        file_key: Optional[str] = None,
    ):
        """
        Add waveforms from a ProcessingResult object.

        Args:
            result: ProcessingResult from pipeline
            file_key: Source file S3 key
        """
        # Get waveforms from result
        if hasattr(result, 'extracted_waveforms') and result.extracted_waveforms:
            waveforms = result.extracted_waveforms
        elif hasattr(result, 'waveforms') and result.waveforms:
            waveforms = result.waveforms
        else:
            return

        # Get phase/amplitude data
        phase_amp = getattr(result, 'phase_amplitude_data', [])

        # Get PD type info
        pd_types = []
        if hasattr(result, 'pd_type_results') and result.pd_type_results:
            pd_types = getattr(result.pd_type_results, 'per_waveform_types', [])

        # Get cluster assignments
        clusters = getattr(result, 'cluster_assignments', [])

        # Add each waveform
        for i, wf in enumerate(waveforms):
            if isinstance(wf, np.ndarray):
                samples = wf
            elif hasattr(wf, 'samples'):
                samples = np.array(wf.samples)
            else:
                continue

            # Get phase/amplitude
            if i < len(phase_amp):
                phase = phase_amp[i].get('phase', 0)
                amplitude = phase_amp[i].get('amplitude', np.max(np.abs(samples)))
            else:
                phase = 0
                amplitude = float(np.max(np.abs(samples)))

            # Get PD type
            pd_type = pd_types[i] if i < len(pd_types) else 'unknown'

            # Get cluster
            cluster_id = clusters[i] if i < len(clusters) else 0

            self.add_waveform(
                phase=phase,
                amplitude=amplitude,
                samples=samples,
                pd_type=pd_type,
                cluster_id=cluster_id,
                file_key=file_key,
                file_index=i,
            )

    def build(self, timestamp: str, file_key: Optional[str] = None) -> ScatterDataset:
        """
        Build the ScatterDataset.

        Args:
            timestamp: Timestamp for the dataset
            file_key: Source file key (for single-file datasets)

        Returns:
            ScatterDataset
        """
        return ScatterDataset(
            transformer_id=self.transformer_id,
            file_key=file_key,
            timestamp=timestamp,
            total_waveforms=len(self.waveforms),
            waveform_length=self.waveform_length,
            sample_rate=self.sample_rate,
            waveforms=self.waveforms,
        )

    def clear(self):
        """Clear all waveforms."""
        self.waveforms = []
        self.next_id = 0


def estimate_dataset_size(
    num_waveforms: int,
    waveform_length: int = 1000,
    bytes_per_sample: int = 8,
) -> int:
    """
    Estimate size of scatter dataset in bytes.

    Args:
        num_waveforms: Number of waveforms
        waveform_length: Samples per waveform
        bytes_per_sample: Bytes per sample (8 for float64 in JSON)

    Returns:
        Estimated size in bytes
    """
    # Metadata per waveform: ~100 bytes (id, phase, amp, type, cluster)
    metadata_size = num_waveforms * 100

    # Samples: each float in JSON is ~10-15 characters
    samples_size = num_waveforms * waveform_length * 12

    return metadata_size + samples_size


def estimate_compressed_size(uncompressed_size: int) -> int:
    """
    Estimate compressed size (gzip typically achieves 4-6x compression on JSON).

    Args:
        uncompressed_size: Uncompressed size in bytes

    Returns:
        Estimated compressed size
    """
    return uncompressed_size // 5  # Assume 5x compression ratio
