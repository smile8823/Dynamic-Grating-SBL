import numpy as np
from typing import Tuple
import numpy as np
import time
import os
from datetime import datetime
from modules.dictionary_learning import DictionaryLearning
from modules.data_reader import get_fbg_training_data, normalize_data, denormalize_data

def load_fbg_data(num_frames: int = 100, use_sampling: bool = False, filename: str = None) -> np.ndarray:
    """
    Load real FBG data for training
    
    Args:
        num_frames: Number of training frames
        use_sampling: Whether to use grouped sampling mode
        filename: Data filename
        
    Returns:
        np.ndarray: Raw training data matrix with shape (num_wavelengths, num_frames)
    """
    print("Loading real FBG data from CSV file...")
    
    if filename is None:
         # Fallback or error, but since we want to avoid hardcoding, we should probably require it.
         # However, to be safe with existing calls, maybe default to the one known, or better, error out.
         # Given the user instruction, I'll error out if not provided to enforce config usage.
         raise ValueError("Filename must be provided to load_fbg_data")

    if use_sampling:
        # Use grouped sampling mode
        from modules.data_reader import get_fbg_sampled_data
        Y = get_fbg_sampled_data(num_samples=num_frames, filename=filename)
        print(f"Loading data using grouped sampling mode")
    else:
        # Use normal mode
        Y = get_fbg_training_data(num_frames=num_frames, filename=filename)
        print(f"Loading data using normal mode")
    
    print(f"Loaded training data with shape: {Y.shape}")
    print(f"Data range: [{np.min(Y):.6f}, {np.max(Y):.6f}]")
    return Y


def generate_synthetic_data(M: int, N: int, K: int, noise_level: float = 0.1) -> np.ndarray:
    """
    Generate synthetic data (Keep original function for testing purposes)
    """
    Y = np.zeros((M, N))
    
    for i in range(N):
        signal = np.zeros(M)
        peak_positions = np.random.choice(M, K, replace=False)
        
        for pos in peak_positions:
            width = np.random.uniform(5, 15)
            amplitude = np.random.uniform(0.5, 2.0)
            x = np.arange(M)
            signal += amplitude * np.exp(-0.5 * ((x - pos) / width) ** 2)
        
        noise = noise_level * np.random.randn(M)
        Y[:, i] = signal + noise
    
    return Y

def save_stage1_results(phi_final: np.ndarray, beta_global: float, Y: np.ndarray, 
                        x_final: np.ndarray, wavelengths: np.ndarray) -> str:
    """
    Save Stage 1 results to output folder, only save necessary data
    
    Args:
        phi_final: Final dictionary matrix
        beta_global: Global noise precision
        Y: Original training data
        x_final: Final sparse representation
        wavelengths: Wavelength array
        
    Returns:
        str: Saved directory path
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("..", "output", f"stage1_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate reconstructed waveforms
    Y_reconstructed = phi_final @ x_final
    
    # Save reconstructed waveform data (same format as original data)
    np.savez_compressed(
        os.path.join(output_dir, "reconstructed_waveforms.npz"),
        wavelengths=wavelengths,
        Y_reconstructed=Y_reconstructed,
        Y_original=Y
    )
    
    # Save dictionary for subsequent stages
    np.savez_compressed(
        os.path.join(output_dir, "dictionary.npz"),
        phi_final=phi_final,
        beta_global=beta_global
    )
    
    print(f"Stage 1 results saved to: {output_dir}")
    return output_dir


def run_stage_one(Y: np.ndarray, D: int, K: int, max_iterations: int = 50, 
                  save_results: bool = True) -> Tuple[np.ndarray, float, str]:
    M = Y.shape[0]
    
    print("=== Stage 1: Offline Training and Global Parameter Estimation ===")
    print(f"Input parameters:")
    print(f"  - Data shape: {Y.shape}")
    print(f"  - Dictionary size (D): {D}")
    print(f"  - Sparsity (K): {K}")
    print(f"  - Max iterations: {max_iterations}")
    
    dl = DictionaryLearning(M, D, K, max_iterations)
    
    print("\nStarting K-SVD to learn the dictionary...")
    phi_final, x_final = dl.fit(Y)
    print("Dictionary learning complete.")
    
    print("\nEstimating global noise precision...")
    beta_global = dl.estimate_noise_precision(Y)
    print(f"Global noise precision β estimated as: {beta_global:.6f}")
    
    # Save results
    output_dir = ""
    if save_results:
        # Get real wavelength array
        from modules.data_reader import read_csv_data
        try:
            wavelengths, _ = read_csv_data()
        except:
            # If unable to read real wavelengths, use default range
            wavelengths = np.linspace(1527.0, 1568.0, M)
            
        output_dir = save_stage1_results(phi_final, beta_global, Y, x_final, wavelengths)
    
    return phi_final, beta_global, output_dir

def main():
    # Real data parameters
    N = 50       # Number of training frames
    K = 3        # Sparsity
    D = 1024     # Dictionary size
    max_iterations = 10
    
    print("=== DG-SBL Stage 1: Loading Real FBG Data ===")
    
    # Load real FBG data (no normalization)
    Y = load_fbg_data(num_frames=N)
    
    # Get real data dimension
    M = Y.shape[0]  # Number of wavelength points
    
    print(f"\nData parameters:")
    print(f"  - Signal dimension (M): {M}")
    print(f"  - Training frames (N): {N}")
    print(f"  - Dictionary size (D): {D}")
    print(f"  - Sparsity level (K): {K}")
    print(f"  - Max iterations: {max_iterations}")
    
    # Run Stage 1: Offline dictionary learning and global parameter estimation
    phi_final, beta_global, output_dir = run_stage_one(Y, D, K, max_iterations)
    
    print(f"\n=== Results ===")
    print(f"Final dictionary shape: {phi_final.shape}")
    print(f"Global noise precision β: {beta_global:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return phi_final, beta_global, output_dir

if __name__ == "__main__":
    main()