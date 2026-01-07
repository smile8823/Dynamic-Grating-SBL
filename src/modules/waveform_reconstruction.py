"""
Simplified Waveform Reconstruction Module
Provides basic waveform reconstruction functionality for the DG-SBL framework

Core Functions:
1. Reconstruct waveforms based on sparse coefficients and dictionary
2. Support multi-frame reconstruction
3. Save waveform data consistent with original data format
"""

import numpy as np
from typing import List


def reconstruct_waveforms(phi_matrix: np.ndarray, 
                         sparse_coefficients_stream: List[np.ndarray],
                         wavelengths: np.ndarray) -> np.ndarray:
    """
    Reconstruct multi-frame waveforms
    
    Args:
        phi_matrix: Dictionary matrix (M x D)
        sparse_coefficients_stream: List of sparse coefficients stream
        wavelengths: Wavelength array (M,)
        
    Returns:
        np.ndarray: Reconstructed waveform matrix (M x N)
    """
    if not sparse_coefficients_stream:
        return np.zeros((len(wavelengths), 0))
        
    num_frames = len(sparse_coefficients_stream)
    signal_dimension = phi_matrix.shape[0]
    reconstructed_waveforms = np.zeros((signal_dimension, num_frames))
    
    for i, x_sparse in enumerate(sparse_coefficients_stream):
        reconstructed_waveforms[:, i] = phi_matrix @ x_sparse
        
    return reconstructed_waveforms


def create_gaussian_waveform_from_offset(wavelengths: np.ndarray, 
                                       center_wavelength: float, 
                                       width: float = 1.0,
                                       amplitude: float = 1.0) -> np.ndarray:
    """
    Create Gaussian waveform based on offset
    
    Args:
        wavelengths: Wavelength array
        center_wavelength: Center wavelength
        width: Waveform width
        amplitude: Amplitude
        
    Returns:
        np.ndarray: Gaussian waveform
    """
    return amplitude * np.exp(-((wavelengths - center_wavelength) ** 2) / (2 * width ** 2))


if __name__ == "__main__":
    # Simple test
    print("=== Test Simplified Waveform Reconstruction Module ===")
    
    # Create simulated data
    M, D, N = 100, 50, 10
    phi_matrix = np.random.randn(M, D)
    wavelengths = np.linspace(1527.0, 1568.0, M)
    
    # Create simulated sparse coefficients
    sparse_coeffs = []
    for _ in range(N):
        coeff = np.zeros(D)
        active_atoms = np.random.choice(D, 3, replace=False)
        coeff[active_atoms] = np.random.randn(3)
        sparse_coeffs.append(coeff)
    
    # Reconstruct waveforms
    reconstructed = reconstruct_waveforms(phi_matrix, sparse_coeffs, wavelengths)
    print(f"Reconstructed waveform shape: {reconstructed.shape}")
    print("Test completed")