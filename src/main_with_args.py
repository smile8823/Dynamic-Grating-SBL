"""
DG-SBL Complete System Main Script (Command Line Arguments Supported)
Full process from data reading to Stage 1, Stage 2, Stage 3, until outputting final wavelength offsets

Usage:
    python main_with_args.py                           # Use default configuration
    python main_with_args.py config/config_fast_test.json  # Use specified configuration file

Complete Workflow:
1. Data reading and preprocessing
2. Stage 1: Dictionary learning and global parameter estimation
3. Stage 2: Covariance-free SBL online dynamic tracking
4. Stage 3: Direction prediction guided multi-signal intelligent tracking
5. Result output and visualization
"""

import sys
import os
from main import DGSBLCompleteSystem


def main():
    """Main function - supports command line arguments"""
    print("DG-SBL Complete System Startup (Command Line Arguments Supported)")
    print("Author: DG-SBL Team")
    print("Version: 3.0")
    
    # Handle command line arguments
    config_file = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"Error: Configuration file does not exist: {config_file}")
            print("Usage:")
            print("  python main_with_args.py                           # Use default configuration")
            print("  python main_with_args.py config/config_fast_test.json  # Use specified configuration")
            print("  python main_with_args.py --sparsity_level 1        # Set sparsity level to 1")
            print("  python main_with_args.py --save_reconstructed_waveforms  # Save reconstructed waveforms")
            return 1
        print(f"Using configuration file: {config_file}")
    else:
        print("Using default configuration")
    
    # Create system
    system = DGSBLCompleteSystem(config_file)
    
    # Handle command line arguments overriding configuration
    if '--save_reconstructed_waveforms' in sys.argv:
        system.config['output']['save_reconstructed_waveforms'] = True
        print("Command line argument: Enable reconstructed waveform data saving")
    
    # Handle sparsity level argument
    if '--sparsity_level' in sys.argv:
        try:
            sparsity_idx = sys.argv.index('--sparsity_level')
            if sparsity_idx + 1 < len(sys.argv):
                sparsity_level = int(sys.argv[sparsity_idx + 1])
                system.config['stage1']['sparsity_level'] = sparsity_level
                print(f"Command line argument: Set sparsity level to {sparsity_level}")
            else:
                print("Error: --sparsity_level argument requires a value")
                return 1
        except (ValueError, IndexError):
            print("Error: --sparsity_level argument must be an integer")
            return 1
    
    # Run complete pipeline
    results = system.run_complete_pipeline()
    
    if 'error' not in results:
        print("\nSystem run completed successfully!")
        return 0
    else:
        print(f"\nSystem run failed: {results['error']}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
