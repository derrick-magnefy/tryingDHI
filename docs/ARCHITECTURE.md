# PD Analysis System Architecture

## Overview

This system analyzes Partial Discharge (PD) waveforms through a multi-stage pipeline:
1. **Feature Extraction** - Extract 53 pulse-level features from waveforms
2. **Clustering** - Group similar pulses using DBSCAN, HDBSCAN, or K-means
3. **Aggregation** - Compute cluster-level statistics (PRPD features)
4. **Classification** - Classify clusters into PD types (Corona, Internal, Surface, Noise)

## Directory Structure

```
tryingDHI/
├── pdlib/                    # Core library (reusable modules)
│   ├── features/             # Feature extraction
│   │   ├── extractor.py      # PDFeatureExtractor class
│   │   ├── definitions.py    # Feature names and groups
│   │   ├── polarity.py       # Polarity calculation methods
│   │   └── pulse_detection.py # Multi-pulse detection
│   ├── clustering/           # Clustering algorithms
│   │   ├── algorithms.py     # DBSCAN, HDBSCAN, K-means
│   │   ├── cluster_features.py # Cluster aggregation
│   │   └── definitions.py    # Cluster feature names
│   └── classification/       # PD type classification
│       ├── classifier.py     # PDTypeClassifier class
│       └── pd_types.py       # PD type definitions
│
├── middleware/               # Data format handlers
│   └── formats/
│       ├── base.py           # BaseLoader interface
│       ├── rugged.py         # Rugged format loader
│       └── detection.py      # Auto-detect format
│
├── config/                   # Configuration system
│   ├── loader.py             # ConfigLoader class
│   └── defaults/
│       ├── features.yaml     # Default clustering features
│       └── thresholds.yaml   # Classification thresholds
│
├── scripts/                  # CLI tools
│   └── run_pipeline.py       # Main pipeline script
│
├── tests/                    # Test suite
│   ├── test_pdlib_comparison.py    # Validate pdlib vs old impl
│   └── test_pipeline_comparison.py # Compare pipeline outputs
│
├── docs/                     # Documentation
│   └── ARCHITECTURE.md       # This file
│
├── pd_visualization_gui.py   # Web-based GUI (Dash/Plotly)
├── run_analysis_pipeline.py  # Legacy subprocess-based pipeline
└── run_pipeline_integrated.py # New pdlib-based pipeline
```

## Core Modules

### pdlib.features

Extract features from PD waveforms.

```python
from pdlib.features import PDFeatureExtractor

extractor = PDFeatureExtractor(sample_interval=4e-9)

# Single waveform
features = extractor.extract_features(waveform, phase_angle=45.0)

# Batch (includes normalized features)
all_features = extractor.extract_all(waveforms, phase_angles, normalize=True)
```

**53 Features Extracted:**
- Amplitude: peak_amplitude_positive/negative, absolute_amplitude, rms_amplitude
- Timing: rise_time, fall_time, pulse_width, slew_rate
- Energy: energy, charge, cumulative_energy_*
- Spectral: dominant_frequency, bandwidth_3db, spectral_power_low/high
- Shape: crest_factor, oscillation_count, zero_crossing_count
- Normalized: norm_slew_rate, norm_energy, norm_rise_time, etc.

### pdlib.clustering

Cluster pulses and compute cluster-level features.

```python
from pdlib.clustering import cluster_pulses, compute_cluster_features

# Cluster using HDBSCAN (default)
labels, info = cluster_pulses(features, feature_names, method='hdbscan')

# Or DBSCAN with auto-eps
labels, info = cluster_pulses(features, feature_names, method='dbscan')

# Compute cluster features (PRPD + aggregates)
cluster_features = compute_cluster_features(
    features_matrix, feature_names, labels, ac_frequency=60.0
)
```

**Clustering Methods:**
- `hdbscan` - Hierarchical DBSCAN, auto-determines clusters (default)
- `dbscan` - Density-based, auto-estimates epsilon
- `kmeans` - Fixed number of clusters

**140 Cluster Features:**
- PRPD: phase distributions, discharge asymmetry, quadrant percentages
- Waveform aggregates: mean and trimmed-mean of all pulse features

### pdlib.classification

Classify clusters into PD types.

```python
from pdlib.classification import PDTypeClassifier

classifier = PDTypeClassifier()
result = classifier.classify(cluster_features, cluster_label)

# Result contains:
# - pd_type: 'CORONA', 'INTERNAL', 'SURFACE', 'NOISE', 'UNKNOWN'
# - confidence: 0.0 - 1.0
# - reasoning: list of decision factors
```

**Classification Decision Tree:**
1. Noise Detection - DBSCAN label, spectral characteristics
2. Phase Spread - Surface PD if spread > 120°
3. Surface Detection - 10-feature weighted scoring
4. Corona vs Internal - 13-feature weighted scoring

### middleware.formats

Load data from various formats.

```python
from middleware.formats import RuggedLoader, AutoLoader

# Rugged format (tab-separated text files)
loader = RuggedLoader('Rugged Data Files')
waveforms = loader.load_waveforms('dataset_prefix')
settings = loader.load_settings('dataset_prefix')
phases = loader.load_phase_angles('dataset_prefix')

# Auto-detect format
loader = AutoLoader('data_directory')
data = loader.load('dataset_prefix')
```

**Supported Formats:**
- Rugged: -WFMs.txt, -SG.txt, -Ph.txt, -Ti.txt

### config

YAML-based configuration.

```python
from config.loader import ConfigLoader

config = ConfigLoader()

# Get default clustering features
features = config.get_features()
default_clustering = features['pulse_features']['default_clustering']

# Get classification thresholds
thresholds = config.get_thresholds()
```

**Configuration Files:**
- `config/defaults/features.yaml` - Default feature selection
- `config/defaults/thresholds.yaml` - Classification thresholds

## Pipelines

### Integrated Pipeline (Recommended)

Uses pdlib modules directly - faster, better error handling.

```bash
python run_pipeline_integrated.py --input-dir "Rugged Data Files" --clustering-method hdbscan
```

### Legacy Pipeline

Uses subprocess calls to standalone scripts.

```bash
python run_analysis_pipeline.py --input-dir "Rugged Data Files" --clustering-method dbscan
```

### GUI

Web-based visualization with interactive clustering.

```bash
pip install dash plotly
python pd_visualization_gui.py --port 8050
```

## Data Flow

```
Raw Waveforms (-WFMs.txt)
       │
       ▼
┌─────────────────────────────────┐
│  Feature Extraction (pdlib)     │
│  PDFeatureExtractor.extract_all │
└─────────────────────────────────┘
       │
       ▼
   {prefix}-features.csv (53 features per pulse)
       │
       ▼
┌─────────────────────────────────┐
│  Clustering (pdlib)             │
│  cluster_pulses()               │
└─────────────────────────────────┘
       │
       ▼
   {prefix}-clusters-{method}.csv (cluster labels)
       │
       ▼
┌─────────────────────────────────┐
│  Aggregation (pdlib)            │
│  compute_cluster_features()     │
└─────────────────────────────────┘
       │
       ▼
   {prefix}-cluster-features-{method}.csv (140 features per cluster)
       │
       ▼
┌─────────────────────────────────┐
│  Classification (pdlib)         │
│  PDTypeClassifier.classify()    │
└─────────────────────────────────┘
       │
       ▼
   {prefix}-pd-types-{method}.csv (PD type + confidence)
```

## Testing

```bash
# Validate pdlib modules match original implementations
python tests/test_pdlib_comparison.py

# Compare old vs new pipeline outputs
python tests/test_pipeline_comparison.py --file "dataset_prefix"
```

## Configuration

### Default Clustering Features

Edit `config/defaults/features.yaml`:

```yaml
pulse_features:
  default_clustering:
    - absolute_amplitude
    - rise_time
    - fall_time
    - pulse_width
    - slew_rate
    - energy
    - dominant_frequency
    - spectral_power_low
    - spectral_power_high
    - crest_factor
    - oscillation_count
    - norm_slew_rate
    - norm_energy
    - norm_rise_time
```

### Classification Thresholds

Edit `config/defaults/thresholds.yaml` to adjust:
- Noise detection thresholds
- Surface PD scoring weights
- Corona vs Internal scoring weights
