# Dynamic-Grating-SBL

**A robust Sparse Bayesian Learning framework for tracking non-standard Gaussian Fiber Bragg Grating (FBG) spectra.**

This project (formerly DG-SBL Framework) is designed to handle complex spectral scenarios where traditional Gaussian peak assumptions fail. It integrates Dictionary Learning and SBL to achieve high-precision tracking of deformed, multi-peak, and overlapping grating signals.

## Key Features

- **Non-Standard Peak Handling**: Robustly tracks distorted, flat-top, and asymmetric peaks.
- **Dynamic Grating (DG) Processing**: Optimized for time-varying spectral shifts.


1. **Code Structure Optimization**: Clear directory layering, separation of core algorithms and utility modules
2. **Significant Performance Boost**: Stage2 reaches 7500+ FPS, Stage3 reaches 1000+ FPS
3. **GPU Acceleration**: GPU parallel computing implemented using PyTorch
4. **Memory Optimization**: Pre-allocated memory, reduced data transfer overhead
5. **Algorithm Optimization**: EM iteration optimization, diagonal approximation, fast convergence strategies

## Performance Metrics

- **Stage2**: 7,507 FPS (0.1ms/frame) - GPU optimized version
- **Stage3**: 1,000+ FPS - Ultra-fast tracking algorithm
- **Overall System**: Supports real-time processing (>30 FPS)

## System Architecture

```
DG-SBL Complete System
├── Data Input (data_reader.py)
├── Stage1: Dictionary Learning (stage1_main.py + dictionary_learning.py)
├── Stage2: SBL Tracking (stage2_main.py + sparse_bayesian_learning.py)
├── Stage3: Intelligent Tracking (stage3_main.py)
│   ├── Peak Detection (peak_detection.py)
│   ├── Signal Tracking (signal_tracker.py)
│   ├── Direction Prediction (direction_prediction.py)
│   └── Signal Separation (signal_separation.py)
├── System Support
│   ├── Memory Management (memory_manager.py)
│   └── Atom Management (atom_set_manager.py)
└── Result Output
    ├── Offset Data in CSV Format
    ├── Detailed Results in JSON Format
    └── Visualization Charts
```

## Directory Structure

```
src/
├── core/                    # Core algorithm implementation
│   ├── stage1_main.py      # Stage1: Dictionary learning
│   ├── optimized_pytorch_sbl.py    # Stage2: GPU optimized SBL algorithm
│   ├── optimized_stage2_main.py    # Stage2: Main program (7500+ FPS)
│   ├── ultra_fast_stage3.py        # Stage3: Ultra-fast signal tracking (1000+ FPS)
│   └── dictionary_learning.py      # Dictionary learning core algorithm
├── modules/                 # Functional modules
│   ├── data_reader.py      # Data reading
│   ├── peak_detection.py   # Peak detection
│   ├── signal_tracker.py   # Signal tracking
│   ├── memory_manager.py   # Memory management
│   └── ...                 # Other utility modules
├── config/                  # Configuration files
│   ├── config_fast_test.json       # Fast test configuration
│   └── config_high_precision.json  # High precision configuration
├── main.py                  # Complete system main program
└── integration_test.py      # Integration test program
```

## Quick Start

### Method 1: Run Main Script (Recommended)
```bash
# Enter src directory
cd "src"

# Run complete DG-SBL algorithm (using default configuration)
python main.py
```

### Method 1b: Use Version with Command Line Arguments
```bash
# Enter src directory
cd "src"

# Use default configuration
python main_with_args.py

# Use specific configuration file
python main_with_args.py config/config_fast_test.json
python main_with_args.py config/config_high_precision.json

# Enable reconstructed waveform data saving
python main_with_args.py config/config_fast_test.json --save_reconstructed_waveforms

# Specify sparsity level
python main_with_args.py --sparsity_level 3
```

### Method 2: Use Preset Configurations

The system provides four preset configuration files for reference:

1.  **Fast Test Configuration (`config/config_fast_test.json`)**:
    *   Applicable: Functional verification, quick debugging.
    *   Run Time: 30s - 1 min.
    *   Usage: `python main_with_args.py config/config_fast_test.json`

2.  **High Precision Configuration (`config/config_high_precision.json`)**:
    *   Applicable: Research analysis, paper experiments.
    *   Run Time: 5-10 mins.
    *   Usage: `python main_with_args.py config/config_high_precision.json`

3.  **Full Data Configuration (`config/config_full_data.json`)**:
    *   Applicable: Full dataset processing, long-term monitoring.
    *   Run Time: 30-60 mins.
    *   Usage: `python main_with_args.py config/config_full_data.json`

4.  **Sampled Test Configuration (`config/config_sampled_10.json`)**:
    *   Applicable: Auto-detection mode testing.
    *   Run Time: 2-3 mins.
    *   Usage: `python main_with_args.py config/config_sampled_10.json`

## Detailed Parameter Configuration Guide

### 1. Data Configuration (`data`)
*   `training_frames`: Number of frames for dictionary learning (Rec: 20-100).
*   `test_frames`: Number of frames tracked in Stage 2.
*   `start_frame`: Start frame index.
*   `signal_dimension`: Fixed at 4101 wavelength points.

### 2. Stage1 Dictionary Learning (`stage1`)
*   `dictionary_size`: Number of atoms (Rec: 512-1024).
*   `sparsity_level`: Max atoms per signal (Rec: 2-5).
*   `max_iterations`: K-SVD iterations (Rec: 5-30).

### 3. Stage2 SBL Tracking (`stage2`)
*   `max_sbl_iterations`: SBL max iterations (Rec: 30-50).
*   `sbl_tolerance`: Convergence threshold (Rec: 1e-3 to 1e-5).

### 4. Stage3 Intelligent Tracking (`stage3`)
*   `signals`: Define FBG wavelength ranges and expected peak counts.
*   `quality_threshold`: Waveform fitting quality threshold (0.5-0.9).
*   `enable_signal_separation`: Enable multi-signal separation.
*   `min_signal_distance`: Minimum distance between signals (nm).

### 5. Output Configuration (`output`)
*   `save_results`: Save result files.
*   `create_plots`: Generate visualization charts.
*   `save_wavelength_offsets`: Save detailed offset data (CSV/JSON).
*   `save_reconstructed_waveforms`: Save reconstructed waveform data (NPZ).

## Algorithm Process Details

### Stage 1: Dictionary Learning and Global Parameter Estimation
*   **Input**: FBG data frames (4101 × N)
*   **Process**: K-SVD Dictionary Learning
*   **Output**: Dictionary matrix Φ (4101 × 512) and Global noise precision β

### Stage 2: Covariance-free SBL Online Dynamic Tracking
*   **Input**: Dictionary matrix Φ, noise precision β, new data frame
*   **Process**: Sparse Bayesian Learning
*   **Output**: Sparse representation matrix x̂, Tracking performance indicators

### Stage 3: Direction Prediction Guided Multi-signal Intelligent Tracking
*   **Input**: Sparse representation, new data frame
*   **Process**: Multi-signal Separation -> Direction Prediction -> Peak Detection
*   **Output**: Wavelength offsets, Tracking quality indicators

## Output Results

1.  **Complete Result File (JSON)**: `dgsbl_complete_results_YYYYMMDD_HHMMSS.json`
2.  **Spectral Shift Data File (CSV)**: `wavelength_offsets_YYYYMMDD_HHMMSS.csv`
3.  **Detailed Offset Analysis (JSON)**: `wavelength_offsets_detailed_YYYYMMDD_HHMMSS.json`
4.  **Reconstructed Waveform Data (NPZ)**: `data/reconstructed_waveforms.npz` (if enabled)
5.  **Visualization Charts**: Error plots, tracking plots, timing comparison.

## Troubleshooting

### Common Issues

1.  **Data File Not Found**: Ensure data is at `data/10s_1.csv`.
2.  **Out of Memory**: Reduce `training_frames` or `dictionary_size`.
3.  **Reconstructed Waveform Data Not Found**: Ensure `"save_reconstructed_waveforms": true` is set or use command line flag.

## Version Information

- **Current Version**: DG-SBL v3.1
- **Main Dependencies**: numpy, scipy, matplotlib, torch (for GPU acceleration)

## Contact and Support

If you need technical support, please provide:
1. Error message and stack trace
2. System configuration parameters
3. Input data sample
4. Runtime environment information
