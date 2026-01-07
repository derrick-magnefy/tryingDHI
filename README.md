# Partial Discharge Analysis Pipeline

A comprehensive Python-based analysis pipeline for partial discharge (PD) data, including feature extraction, clustering, and automated PD type classification.

## Overview

This pipeline processes partial discharge waveform data and performs:
1. **Feature Extraction**: Extracts 53 pulse-level features from waveforms (time-domain, spectral, energy, normalized)
2. **Clustering**: Groups similar pulses using DBSCAN, HDBSCAN, or K-means algorithms
3. **Cluster Aggregation**: Computes 140 statistical features for each cluster (PRPD + waveform aggregates)
4. **PD Type Classification**: Classifies clusters as Noise, Corona, Internal, Surface, or Noise-Multipulse

## Quick Start

```bash
# Run the complete pipeline (recommended - uses pdlib modules)
python run_pipeline_integrated.py --input-dir "Rugged Data Files" --clustering-method hdbscan

# Or use the legacy subprocess-based pipeline
python run_analysis_pipeline.py --clustering-method dbscan

# Launch the web GUI for interactive visualization
python pd_visualization_gui.py --port 8050
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed module documentation.

---

## Pipeline Process Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA FILES                            │
├─────────────────────────────────────────────────────────────────────┤
│  *-WFMs.txt    Raw waveform data (2000 samples x N waveforms)      │
│  *-SG.txt      Settings (sample interval, AC frequency, etc.)      │
│  *-Ti.txt      Trigger times for each waveform                     │
│  *-P.txt       Phase angles                                        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: FEATURE EXTRACTION                       │
│                      (pdlib.features module)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Extracts 53 features per pulse:                                   │
│  - Amplitude: peak +/-, absolute, RMS, crest factor                │
│  - Timing: rise_time, fall_time, pulse_width, slew_rate            │
│  - Energy: energy, charge, cumulative_energy metrics               │
│  - Spectral: dominant_freq, bandwidth_3db, spectral_power_*        │
│  - Normalized: norm_slew_rate, norm_energy, norm_rise_time, etc.   │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-features.csv                                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STEP 2: CLUSTERING                             │
│                   (pdlib.clustering module)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Groups similar pulses using:                                       │
│  - HDBSCAN: Hierarchical density, auto-determines clusters         │
│  - DBSCAN: Density-based, auto-estimates epsilon, noise=-1         │
│  - K-means: Partitional, fixed number of clusters                  │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-clusters-{method}.csv                                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STEP 3: CLUSTER AGGREGATION                        │
│                   (pdlib.clustering module)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Computes 140 statistical features per cluster:                    │
│  - PRPD: phase distributions, Hn/Hqn skewness/kurtosis            │
│  - Amplitude: statistics by polarity, quadrant percentages        │
│  - Waveform: mean and trimmed-mean of all pulse features          │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-cluster-features-{method}.csv                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STEP 4: PD TYPE CLASSIFICATION                      │
│                 (pdlib.classification module)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Decision tree classifies each cluster as:                         │
│  - NOISE: Non-PD signals (DBSCAN label=-1 or high variance)        │
│  - CORONA: Asymmetric, single half-cycle discharge                 │
│  - INTERNAL: Symmetric void discharge                              │
│  - SURFACE: Discharge near zero-crossings (phase spread >120°)     │
│  - NOISE_MULTIPULSE: Multi-pulse waveform noise                    │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-pd-types-{method}.csv, summary report                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Reference

### Step 1: Pulse-Level Features (53 features)

| Category | Features | Description |
|----------|----------|-------------|
| **Phase** | phase_angle | AC phase when pulse triggered (degrees) |
| **Amplitude** | peak_amplitude_positive, peak_amplitude_negative | Max positive/negative amplitude (V) |
| | absolute_amplitude, rms_amplitude | Absolute peak, RMS amplitude (V) |
| | polarity, peak_to_peak_amplitude, crest_factor | Polarity, range, peak/RMS ratio |
| **Timing** | rise_time, fall_time, pulse_width | 10-90% rise, 90-10% fall, 50% width (s) |
| | slew_rate, rise_fall_ratio | Max rate of change (V/s), timing ratio |
| | zero_crossing_count, oscillation_count | Waveform shape characteristics |
| **Energy** | energy, charge | Integral of V² (V²·s), integral of V (V·s) |
| | equivalent_time, equivalent_bandwidth | Energy-weighted centroid, spread (s) |
| | cumulative_energy_* | peak, rise_time, shape_factor, area_ratio |
| | energy_charge_ratio | Energy / Charge (V) |
| **Spectral** | dominant_frequency, center_frequency | Max power freq, power-weighted centroid (Hz) |
| | bandwidth_3db | 3dB bandwidth (Hz) |
| | spectral_power_low, spectral_power_high | Power in low/high bands (normalized) |
| | spectral_power_ratio | Low/high ratio |
| | spectral_flatness, spectral_entropy | Flatness, entropy (bits) |
| **Normalized** | norm_slew_rate, norm_energy | Normalized by amplitude |
| | norm_rise_time, norm_fall_time | Normalized by amplitude |
| | norm_pulse_width, norm_bandwidth_3db | Normalized by amplitude |
| | norm_spectral_power_low/high | Normalized spectral power |
| | norm_energy_charge_ratio, norm_center_frequency | Normalized ratios |

*See `pdlib/features/definitions.py` for complete feature list.*

### Step 3: Cluster-Level Features (140 features)

**PRPD Features (34):**

| Category | Features | Description |
|----------|----------|-------------|
| **Pulse Count** | pulses_per_positive/negative_halfcycle | Pulses per half-cycle |
| **Correlation** | cross_correlation, discharge_asymmetry | Half-cycle correlation, asymmetry ratio |
| **Hn Distribution** | skewness/kurtosis_Hn_positive/negative | Pulse count distribution statistics |
| **Hqn Distribution** | skewness/kurtosis_Hqn_positive/negative | Charge distribution statistics |
| **Amplitude** | mean/max/variance_amplitude_positive/negative | Amplitude statistics by polarity |
| | coefficient_of_variation | Overall amplitude CV |
| **Peaks** | number_of_peaks_Hn_positive/negative | Peak count in distributions |
| **Phase** | phase_of_max_activity, phase_spread | Max activity phase, circular std dev |
| | inception_phase, extinction_phase | First/last activity phase |
| **Quadrants** | quadrant_1/2/3/4_percentage | % in each 90° quadrant |
| **Weibull** | weibull_alpha, weibull_beta | Scale, shape parameters |
| **Rate** | repetition_rate | Pulses per second |

**Waveform Aggregates (106):** Mean and trimmed-mean of all 53 pulse features.

*See `pdlib/clustering/definitions.py` for complete feature list.*

---

## Decision Tree Classification

### Tree Structure

```
                              ┌─────────────────┐
                              │  Input Cluster  │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      │
         ┌─────────────────────┐                          │
         │  BRANCH 1: NOISE    │                          │
         │  DETECTION          │                          │
         └─────────┬───────────┘                          │
                   │                                       │
    ┌──────────────┼──────────────┐                       │
    ▼              ▼              ▼                       │
┌────────┐   ┌──────────┐   ┌──────────┐                 │
│DBSCAN  │   │Pulse     │   │High CV   │                 │
│label=-1│   │count<10  │   │(>2.0)    │                 │
└────┬───┘   └────┬─────┘   └────┬─────┘                 │
     │            │              │                        │
     └────────────┴──────────────┘                       │
                   │                                      │
                   ▼                                      │
            ┌──────────┐                                  │
            │  NOISE   │                                  │
            │  (exit)  │                                  │
            └──────────┘                                  │
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
         ┌─────────────────────┐
         │  BRANCH 2: PHASE    │
         │  CORRELATION        │
         └─────────┬───────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
┌─────────────────┐      ┌─────────────────┐
│ SYMMETRIC       │      │ ASYMMETRIC      │
│ cross_corr>0.7  │      │ |asymmetry|>0.6 │
│ |asymmetry|<0.35│      │                 │
└────────┬────────┘      └────────┬────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│  BRANCH 3A:     │      │  BRANCH 3B:     │
│  INTERNAL CHECK │      │  CORONA CHECK   │
└────────┬────────┘      └────────┬────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ Balanced quads? │      │ Single half-    │
│ (15-35% each)   │      │ cycle >80%?     │
│ Weibull B: 2-15?│      │ Phase spread    │
│ Peak phase?     │      │ <60 degrees?    │
└────────┬────────┘      └────────┬────────┘
         │                        │
    ┌────┴────┐              ┌────┴────┐
    ▼         ▼              ▼         ▼
┌────────┐ ┌──────┐     ┌────────┐ ┌──────┐
│INTERNAL│ │Branch│     │ CORONA │ │Branch│
│ (exit) │ │  4   │     │ (exit) │ │  4   │
└────────┘ └──────┘     └────────┘ └──────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  BRANCH 4: PHASE    │
                   │  LOCATION           │
                   └─────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │ Near zero-cross │           │ Near voltage    │
     │ (0, 180, 360 deg│           │ peaks (90, 270) │
     └────────┬────────┘           └────────┬────────┘
              │                             │
              ▼                             ▼
       ┌──────────┐                 ┌───────────────┐
       │ SURFACE  │                 │ BRANCH 5:     │
       │ (exit)   │                 │ AMPLITUDE     │
       └──────────┘                 └───────┬───────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ Amplitude ratio │
                                   │ Weibull params  │
                                   │ -> Final type   │
                                   └─────────────────┘
```

### Branch Details and Thresholds

#### Branch 1: Noise Detection
| Criterion | Threshold | Result if True |
|-----------|-----------|----------------|
| DBSCAN label | = -1 | NOISE (95% conf) |
| Pulse count | < 10 | NOISE (80% conf) |
| Coefficient of variation | > 2.0 | Warning flag |

#### Branch 2: Phase Correlation
| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Cross-correlation | > 0.7 | Symmetric pattern |
| Discharge asymmetry | < 0.35 | Balanced half-cycles |
| Asymmetry | > 0.6 | Corona-like asymmetric |

#### Branch 3: Quadrant Distribution
| Pattern | Q1 | Q2 | Q3 | Q4 | Type |
|---------|----|----|----|----|------|
| Corona (+) | 40-100% | 0-60% | ~0% | ~0% | CORONA |
| Corona (-) | ~0% | ~0% | 40-100% | 0-60% | CORONA |
| Internal | 15-35% | 15-35% | 15-35% | 15-35% | INTERNAL |
| Surface | varies | varies | varies | varies | check phase |

#### Branch 4: Phase Location
| Phase of Max Activity | Interpretation |
|-----------------------|----------------|
| 0-45 deg or 135-180 deg | Near zero-crossing -> SURFACE |
| 45-135 deg | Near positive peak -> CORONA/INTERNAL |
| 180-225 deg or 315-360 deg | Near zero-crossing -> SURFACE |
| 225-315 deg | Near negative peak -> CORONA/INTERNAL |

#### Branch 5: Amplitude Characteristics
| Feature | Corona | Internal | Surface |
|---------|--------|----------|---------|
| Weibull beta | Variable | 2-15 | Variable |
| Amplitude ratio | > 3 | < 3 | Variable |
| Rise time | < 30 ns | 10-100 ns | Variable |

---

## PD Type Characteristics

### NOISE (Code: 0)
- **Description**: Non-PD or random noise signals
- **Indicators**:
  - DBSCAN label = -1
  - Pulse count < 10
  - High coefficient of variation (>2.0)
  - No clear phase pattern

### CORONA (Code: 1)
- **Description**: Surface ionization in gas/air around conductors
- **PRPD Pattern**: "Rabbit ear" or "wing" shape in one half-cycle
- **Key Features**:
  - Highly asymmetric (|asymmetry| > 0.6)
  - Single half-cycle dominance (>80%)
  - Concentrated phase spread (<60 deg)
  - Fast rise times (<30 ns typical)
  - High frequency content
- **Typical Locations**: Corona rings, sharp edges, conductor surfaces

### INTERNAL (Code: 2)
- **Description**: Void discharge in solid insulation cavities
- **PRPD Pattern**: Symmetric "turtle shell" pattern at voltage peaks
- **Key Features**:
  - Symmetric (|asymmetry| < 0.35)
  - High cross-correlation (>0.7)
  - Balanced quadrant distribution (15-35% each)
  - Phase peaks at 90 deg and 270 deg
  - Weibull beta: 2-15 (uniform amplitude)
- **Typical Locations**: Cable joints, motor windings, transformer insulation

### SURFACE (Code: 3)
- **Description**: Tracking/creeping discharge along insulation surfaces
- **PRPD Pattern**: Activity near zero-crossings (0 deg, 180 deg)
- **Key Features**:
  - Moderate asymmetry (0.2-0.7)
  - Phase activity near 0 deg, 180 deg, 360 deg
  - Wide phase spread (>30 deg)
  - Adjacent quadrant concentration
- **Typical Locations**: Bushings, contaminated surfaces, cable terminations

---

## Validation Results

The classifier was validated against the dataset with the following results:

| Dataset | Main Cluster Type | Confidence | Validation |
|---------|------------------|------------|------------|
| AC Corona Ch1 | CORONA | 75-90% | Rise time 15-35ns, consistent |
| AC Corona Ch2 | Mixed | - | Some SURFACE near zero-crossings |
| AC Corona Ch3 | CORONA/SURFACE | 65-90% | Mixed patterns detected |
| AC Motor Ch1 | INTERNAL | 95% | Symmetric, balanced quadrants |
| AC Motor Ch2 | INTERNAL | 50% | Symmetric but low Weibull |
| AC Motor Ch3 | INTERNAL | 55% | Symmetric pattern |

---

## Usage Examples

### Integrated Pipeline (Recommended)
```bash
# Run full pipeline with HDBSCAN (default)
python run_pipeline_integrated.py --input-dir "Rugged Data Files"

# Run with DBSCAN clustering
python run_pipeline_integrated.py --clustering-method dbscan

# Run with K-means (5 clusters)
python run_pipeline_integrated.py --clustering-method kmeans --n-clusters 5

# Process a specific file
python run_pipeline_integrated.py --file "Motor Ch1 - 04.06.25, 12.49 PM"

# Use custom pulse features for clustering
python run_pipeline_integrated.py --pulse-features absolute_amplitude rise_time energy
```

### Web GUI
```bash
# Launch interactive visualization dashboard
python pd_visualization_gui.py --port 8050

# Access at http://localhost:8050
# Features: PRPD plots, waveform viewer, interactive clustering, PD classification
```

### Legacy Pipeline (Subprocess-based)
```bash
# Run full pipeline with DBSCAN
python run_analysis_pipeline.py --clustering-method dbscan

# Run with both clustering methods
python run_analysis_pipeline.py --clustering-method both
```

### Python API
```python
from pdlib.features import PDFeatureExtractor
from pdlib.clustering import cluster_pulses, compute_cluster_features
from pdlib.classification import PDTypeClassifier
from middleware.formats import RuggedLoader

# Load data
loader = RuggedLoader('Rugged Data Files')
waveforms = loader.load_waveforms('dataset_prefix')
phases = loader.load_phase_angles('dataset_prefix')

# Extract features
extractor = PDFeatureExtractor(sample_interval=4e-9)
features = extractor.extract_all(waveforms, phases, normalize=True)

# Cluster pulses
labels, info = cluster_pulses(features.values, features.columns.tolist())

# Compute cluster features and classify
cluster_feats = compute_cluster_features(features.values, features.columns.tolist(), labels)
classifier = PDTypeClassifier()
result = classifier.classify(cluster_feats[0], cluster_label=0)
```

---

## Output Files

| File Pattern | Description |
|--------------|-------------|
| `*-features.csv` | Pulse-level features (53 per pulse) |
| `*-clusters-{method}.csv` | Cluster labels for each pulse |
| `*-cluster-features-{method}.csv` | Aggregated cluster features (140 per cluster) |
| `*-pd-types-{method}.csv` | PD type classification results |
| `pd-classification-summary-{method}.txt` | Overall classification summary |

---

## Dependencies

```bash
pip install numpy scipy scikit-learn pandas pyyaml

# Optional for enhanced clustering
pip install hdbscan

# Optional for web GUI
pip install dash plotly
```

---

## Project Structure

```
tryingDHI/
├── pdlib/                        # Core library (reusable modules)
│   ├── features/                 # Feature extraction
│   │   ├── extractor.py          # PDFeatureExtractor class
│   │   ├── definitions.py        # Feature names and groups
│   │   ├── polarity.py           # Polarity calculation methods
│   │   └── pulse_detection.py    # Multi-pulse detection
│   ├── clustering/               # Clustering algorithms
│   │   ├── algorithms.py         # DBSCAN, HDBSCAN, K-means
│   │   ├── cluster_features.py   # Cluster aggregation
│   │   └── definitions.py        # Cluster feature names
│   ├── classification/           # PD type classification
│   │   ├── classifier.py         # PDTypeClassifier class
│   │   └── pd_types.py           # PD type definitions
│   └── config/                   # Configuration (bundled for portability)
│       ├── loader.py             # ConfigLoader class
│       └── defaults/
│           ├── features.yaml     # Default clustering features
│           └── thresholds.yaml   # Classification thresholds
│
├── middleware/                   # Data format handlers
│   └── formats/
│       ├── base.py               # BaseLoader interface
│       ├── rugged.py             # Rugged format loader
│       └── detection.py          # Auto-detect format
│
├── tests/                        # Test suite
│   ├── test_pdlib_comparison.py  # Validate pdlib vs old impl
│   └── test_pipeline_comparison.py # Compare pipeline outputs
│
├── docs/                         # Documentation
│   └── ARCHITECTURE.md           # Detailed module documentation
│
├── run_pipeline_integrated.py    # Main pipeline (uses pdlib)
├── run_analysis_pipeline.py      # Legacy pipeline (subprocess)
├── pd_visualization_gui.py       # Web-based GUI (Dash/Plotly)
│
├── extract_features.py           # Standalone feature extraction
├── cluster_pulses.py             # Standalone clustering
├── aggregate_cluster_features.py # Standalone aggregation
├── classify_pd_type.py           # Standalone classification
│
└── Rugged Data Files/            # Data directory
    ├── *-WFMs.txt                # Raw waveform data
    ├── *-SG.txt                  # Settings
    ├── *-P.txt or *-Ph.txt       # Phase data
    └── *-Ti.txt                  # Trigger times
```

---

## References

The classification thresholds and PD type characteristics are based on:

1. IEC 60270: High-voltage test techniques - Partial discharge measurements
2. IEEE Std 400.3: Guide for PD testing of shielded power cable systems
3. CIGRE Technical Brochure 366: Guide for PD detection in HV equipment
4. Extensive literature on PRPD pattern recognition
