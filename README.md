# Dynamic Grating SBL (DG-SBL)

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/Language-中文-red.svg)](README_zh-CN.md)

This project proposes a two-stage algorithm based on Sparse Bayesian Learning (SBL) to address non-standard peak shapes and multi-peak overlap issues in spectral signals. The system achieves high-precision tracking and sparse reconstruction of continuous dynamic spectral signals.

> **Note**: For detailed mathematical principles, formula derivations, and pseudocode, please refer to the [Algorithm Guide](ALGORITHM_GUIDE.md).

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/Dynamic-Grating-SBL.git
    cd Dynamic-Grating-SBL
    ```

2.  Install dependencies:
    Ensure you have Python 3.8+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Quick Start

### Run Main Program
There are two main ways to run the project:

1.  **Default Run** (Using default configuration):
    ```bash
    python src/main.py
    ```

2.  **Run with Arguments** (Custom configuration):
    ```bash
    python src/main_with_args.py --config src/config/config_full_data.json
    ```

### Run Visualization Scripts
The `scripts/` directory contains scripts to visualize algorithm performance:

```bash
# Run two-stage algorithm visualization
python scripts/two_stage_visualization.py

# Run three-stage algorithm visualization
python scripts/three_stage_visualization.py
```

## 📂 Project Structure

```
d:\Dynamic-Grating-SBL\
├── src\
│   ├── config\          # Configuration files (JSON)
│   ├── core\            # Core algorithm implementations
│   │   ├── stage1_main.py           # Stage 1: Dictionary Learning & Global Param Estimation
│   │   ├── optimized_stage2_main.py # Stage 2: Online Tracking (SBL)
│   │   ├── ultra_fast_stage3.py     # Stage 3: High-speed Tracking
│   │   └── optimized_pytorch_sbl.py # PyTorch implementation of SBL
│   ├── modules\         # Helper modules and components
│   │   ├── data_reader.py           # Data ingestion
│   │   ├── dictionary_learning.py   # Dictionary learning logic
│   │   ├── direction_prediction.py  # Drift prediction
│   │   ├── peak_detection.py        # Peak finding
│   │   ├── signal_separation.py     # Signal separation logic
│   │   ├── signal_tracker.py        # Tracking logic
│   │   ├── waveform_reconstruction.py # Waveform reconstruction
│   │   ├── atom_set_manager.py      # Atom set management
│   ├── main.py          # Main entry point
│   └── main_with_args.py# Entry point with command line arguments
├── scripts\             # Utility scripts and visualizations
├── tests\               # Unit and integration tests
├── data\                # Input data directory (Place your .npz or .csv data here)
├── output\              # Output results directory (Simulation results, logs)
├── ALGORITHM_GUIDE.md   # Detailed Algorithm Principles
└── requirements.txt     # Project dependencies
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
