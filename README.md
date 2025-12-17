# Partial Discharge Analysis Pipeline

A comprehensive Python-based analysis pipeline for partial discharge (PD) data, including feature extraction, clustering, and automated PD type classification.

## Overview

This pipeline processes partial discharge waveform data and performs:
1. **Feature Extraction**: Extracts 29 time-domain, frequency-domain, and energy features from each PD pulse
2. **Clustering**: Groups similar pulses using DBSCAN or K-means algorithms
3. **Cluster Aggregation**: Computes 32 statistical features for each cluster
4. **PD Type Classification**: Classifies clusters as Noise, Corona, Internal, or Surface discharge

## Quick Start

```bash
# Run the complete pipeline
python run_analysis_pipeline.py

# Or run individual steps
python extract_features.py
python cluster_pulses.py --method dbscan
python aggregate_cluster_features.py
python classify_pd_type.py
```

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
│                      (extract_features.py)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Extracts 29 features per pulse:                                   │
│  - Time-domain: rise_time, fall_time, pulse_width, slew_rate       │
│  - Amplitude: peak positive/negative, RMS, crest factor            │
│  - Energy: energy, equivalent_time, cumulative_energy metrics      │
│  - Spectral: dominant_freq, center_freq, bandwidth, entropy        │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-features.csv                                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STEP 2: CLUSTERING                             │
│                     (cluster_pulses.py)                             │
├─────────────────────────────────────────────────────────────────────┤
│  Groups similar pulses using:                                       │
│  - DBSCAN: Density-based, auto-detects noise (label=-1)            │
│  - K-means: Partitional, fixed number of clusters                  │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-clusters-{method}.csv                                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STEP 3: CLUSTER AGGREGATION                        │
│                (aggregate_cluster_features.py)                      │
├─────────────────────────────────────────────────────────────────────┤
│  Computes 32 statistical features per cluster:                     │
│  - Phase distribution (Hn, Hqn skewness/kurtosis)                  │
│  - Amplitude statistics by polarity                                │
│  - Quadrant percentages, Weibull parameters                        │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-cluster-features-{method}.csv                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STEP 4: PD TYPE CLASSIFICATION                      │
│                    (classify_pd_type.py)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Decision tree classifies each cluster as:                         │
│  - NOISE: Non-PD signals                                           │
│  - CORONA: Asymmetric, single half-cycle discharge                 │
│  - INTERNAL: Symmetric void discharge                              │
│  - SURFACE: Discharge near zero-crossings                          │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT: *-pd-types-{method}.csv, summary report                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Reference

### Step 1: Pulse-Level Features (29 features)

| Category | Feature | Description | Units |
|----------|---------|-------------|-------|
| **Phase** | phase_angle | AC phase when pulse triggered | degrees |
| **Amplitude** | peak_amplitude_positive | Maximum positive amplitude | V |
| | peak_amplitude_negative | Maximum negative amplitude | V |
| | polarity | Dominant polarity (+1 or -1) | - |
| | peak_to_peak_amplitude | Total amplitude range | V |
| | rms_amplitude | Root-mean-square amplitude | V |
| | crest_factor | Peak / RMS ratio | - |
| **Time** | rise_time | 10% to 90% rise time | s |
| | fall_time | 90% to 10% fall time | s |
| | pulse_width | Width at 50% amplitude | s |
| | slew_rate | Maximum rate of change | V/s |
| | rise_fall_ratio | rise_time / fall_time | - |
| | zero_crossing_count | Number of zero crossings | count |
| | oscillation_count | Number of local extrema | count |
| **Energy** | energy | Integral of squared signal | V²·s |
| | equivalent_time | Energy-weighted time centroid | s |
| | equivalent_bandwidth | Time spread of energy | s |
| | cumulative_energy_peak | Normalized energy at peak | - |
| | cumulative_energy_rise_time | 10%-90% energy rise time | s |
| | cumulative_energy_shape_factor | First/second half ratio | - |
| | cumulative_energy_area_ratio | Positive/negative ratio | - |
| | energy_charge_ratio | Energy / Charge ratio | V |
| **Spectral** | dominant_frequency | Frequency of max power | Hz |
| | center_frequency | Power-weighted centroid | Hz |
| | bandwidth_3db | 3dB bandwidth | Hz |
| | spectral_power_low | Power in low band (0-25%) | - |
| | spectral_power_high | Power in high band (75-100%) | - |
| | spectral_flatness | Geometric/arithmetic mean | - |
| | spectral_entropy | Spectral distribution entropy | bits |

### Step 3: Cluster-Level Features (32 features)

| Category | Feature | Description |
|----------|---------|-------------|
| **Pulse Count** | pulses_per_positive_halfcycle | Pulses in 0-180° |
| | pulses_per_negative_halfcycle | Pulses in 180-360° |
| **Correlation** | cross_correlation | Correlation between half-cycles |
| | discharge_asymmetry | (pos - neg) / total |
| **Hn Distribution** | skewness_Hn_positive | Skewness of positive Hn |
| | skewness_Hn_negative | Skewness of negative Hn |
| | kurtosis_Hn_positive | Kurtosis of positive Hn |
| | kurtosis_Hn_negative | Kurtosis of negative Hn |
| **Hqn Distribution** | skewness_Hqn_positive | Skewness of positive Hqn |
| | skewness_Hqn_negative | Skewness of negative Hqn |
| | kurtosis_Hqn_positive | Kurtosis of positive Hqn |
| | kurtosis_Hqn_negative | Kurtosis of negative Hqn |
| **Amplitude** | mean_amplitude_positive | Mean |amplitude| in positive |
| | mean_amplitude_negative | Mean |amplitude| in negative |
| | max_amplitude_positive | Max |amplitude| in positive |
| | max_amplitude_negative | Max |amplitude| in negative |
| | variance_amplitude_positive | Variance in positive |
| | variance_amplitude_negative | Variance in negative |
| | coefficient_of_variation | Overall amplitude CV |
| **Peaks** | number_of_peaks_Hn_positive | Peak count in positive Hn |
| | number_of_peaks_Hn_negative | Peak count in negative Hn |
| **Phase** | phase_of_max_activity | Phase bin with most pulses |
| | phase_spread | Circular std deviation |
| | inception_phase | First significant activity |
| | extinction_phase | Last significant activity |
| **Quadrants** | quadrant_1_percentage | % in 0-90° |
| | quadrant_2_percentage | % in 90-180° |
| | quadrant_3_percentage | % in 180-270° |
| | quadrant_4_percentage | % in 270-360° |
| **Weibull** | weibull_alpha | Scale parameter |
| | weibull_beta | Shape parameter |
| **Rate** | repetition_rate | Pulses per second |

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

### Basic Pipeline Run
```bash
# Run full pipeline with DBSCAN clustering
python run_analysis_pipeline.py

# Run with both clustering methods
python run_analysis_pipeline.py --clustering-method both

# Run with K-means (5 clusters)
python run_analysis_pipeline.py --clustering-method kmeans --n-clusters 5
```

### Individual Scripts
```bash
# Extract features only
python extract_features.py --input-dir "Rugged Data Files"

# Cluster with custom DBSCAN parameters
python cluster_pulses.py --method dbscan --min-samples 10 --eps 2.5

# Aggregate features for K-means clusters
python aggregate_cluster_features.py --method kmeans

# Classify PD types
python classify_pd_type.py --method dbscan
```

### Process Specific File
```bash
python run_analysis_pipeline.py --file "Motor Ch1 - 04.06.25, 12.49 PM"
```

---

## Output Files

| File Pattern | Description |
|--------------|-------------|
| `*-features.csv` | Pulse-level features (29 per pulse) |
| `*-clusters-{method}.csv` | Cluster labels for each pulse |
| `*-cluster-features-{method}.csv` | Aggregated cluster features (32 per cluster) |
| `*-pd-types-{method}.csv` | PD type classification results |
| `pd-classification-summary-{method}.txt` | Overall classification summary |

---

## Dependencies

```bash
pip install numpy scipy scikit-learn
```

---

## File Structure

```
tryingDHI/
├── extract_features.py          # Step 1: Pulse feature extraction
├── cluster_pulses.py            # Step 2: DBSCAN/K-means clustering
├── aggregate_cluster_features.py # Step 3: Cluster feature aggregation
├── classify_pd_type.py          # Step 4: PD type classification
├── run_analysis_pipeline.py     # Master pipeline runner
├── regenerate_amplitude.py      # Amplitude reconstruction utility
├── README.md                    # This documentation
└── Rugged Data Files/           # Data directory
    ├── *-WFMs.txt               # Raw waveform data
    ├── *-A.txt                  # Amplitude data
    ├── *-P.txt                  # Phase data
    ├── *-SG.txt                 # Settings
    ├── *-Ti.txt                 # Trigger times
    └── *.pdn                    # Binary waveform files
```

---

## References

The classification thresholds and PD type characteristics are based on:

1. IEC 60270: High-voltage test techniques - Partial discharge measurements
2. IEEE Std 400.3: Guide for PD testing of shielded power cable systems
3. CIGRE Technical Brochure 366: Guide for PD detection in HV equipment
4. Extensive literature on PRPD pattern recognition
