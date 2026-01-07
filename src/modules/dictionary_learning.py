import numpy as np
from typing import Tuple, List, Optional
import time

class DictionaryLearning:
    def __init__(self, M: int, D: int, K: int, max_iterations: int = 50):
        self.M = M
        self.D = D
        self.K = K
        self.max_iterations = max_iterations
        self.dictionary = None
        self.sparse_representation = None
        
    def _initialize_dictionary(self, Y: np.ndarray) -> np.ndarray:
        N = Y.shape[1]
        
        if self.D > N:
            print(f"Warning: Dictionary size D={self.D} > Number of samples N={N}. Sampling with replacement.")
            selected_indices = np.random.choice(N, self.D, replace=True)
        else:
            selected_indices = np.random.choice(N, self.D, replace=False)
        
        phi_initial = Y[:, selected_indices]
        
        # For single-frame learning, add random perturbation to increase diversity
        if N == 1:
            # Single-frame case: Create multiple perturbed versions
            original_signal = Y[:, 0]
            phi_initial = np.zeros((self.M, self.D))
            
            # The first atom is the original signal
            phi_initial[:, 0] = original_signal / np.linalg.norm(original_signal)
            
            # Remaining atoms are versions with small random perturbations
            for k in range(1, self.D):
                noise_level = 0.01 * np.std(original_signal)  # 1% noise level
                perturbed_signal = original_signal + np.random.normal(0, noise_level, self.M)
                phi_initial[:, k] = perturbed_signal / np.linalg.norm(perturbed_signal)
        else:
            # Multi-frame case: Normal initialization
            for k in range(self.D):
                phi_k = phi_initial[:, k]
                norm_phi_k = np.linalg.norm(phi_k)
                if norm_phi_k > 0:
                    phi_initial[:, k] = phi_k / norm_phi_k
                else:
                    phi_initial[:, k] = np.random.randn(self.M)
                    phi_initial[:, k] = phi_initial[:, k] / np.linalg.norm(phi_initial[:, k])
                
        return phi_initial
    
    def _orthogonal_matching_pursuit(self, y: np.ndarray, phi: np.ndarray) -> np.ndarray:
        x = np.zeros(self.D)
        residual = y.copy()
        support_set = []
        
        for _ in range(self.K):
            correlations = np.abs(phi.T @ residual)
            
            # Find the atom with maximum correlation that is not already selected
            sorted_indices = np.argsort(correlations)[::-1]
            best_atom_index = None
            for idx in sorted_indices:
                if idx not in support_set:
                    best_atom_index = idx
                    break
            
            # If no more atoms can be selected, break
            if best_atom_index is None:
                break
                
            support_set.append(best_atom_index)
            
            # Orthogonal projection onto the selected atoms
            phi_support = phi[:, support_set]
            
            # Use least squares to find coefficients
            try:
                a = np.linalg.lstsq(phi_support, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                # If least squares fails, use pseudo-inverse
                a = np.linalg.pinv(phi_support) @ y
            
            # Update residual
            residual = y - phi_support @ a
            
        # Assign the coefficients to the solution vector
        if len(support_set) > 0:
            x[support_set] = a
            
        return x
    
    def _sparse_coding(self, Y: np.ndarray, phi: np.ndarray) -> np.ndarray:
        N = Y.shape[1]
        X = np.zeros((self.D, N))
        
        for i in range(N):
            y_i = Y[:, i]
            x_i = self._orthogonal_matching_pursuit(y_i, phi)
            X[:, i] = x_i
            
        return X
    
    def _update_dictionary_atom(self, Y: np.ndarray, X: np.ndarray, phi: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        omega_k = np.where(X[k, :] != 0)[0]
        
        if len(omega_k) == 0:
            phi[:, k] = np.random.randn(self.M)
            phi[:, k] = phi[:, k] / np.linalg.norm(phi[:, k])
            return phi, X
            
        Y_current_residual = Y[:, omega_k].copy()
        
        for j in range(self.D):
            if j == k:
                continue
            if len(omega_k) > 0:
                Y_current_residual -= np.outer(phi[:, j], X[j, omega_k])
        
        if Y_current_residual.size == 0:
            return phi, X
            
        try:
            U, S, Vt = np.linalg.svd(Y_current_residual, full_matrices=False)
            
            if len(S) > 0 and U.shape[1] > 0:
                phi[:, k] = U[:, 0]
                X[k, omega_k] = S[0] * Vt[0, :]
        except (np.linalg.LinAlgError, ValueError, IndexError):
            # If SVD fails, replace with random atom
            phi[:, k] = np.random.randn(self.M)
            phi[:, k] = phi[:, k] / np.linalg.norm(phi[:, k])
            
        return phi, X
    
    def _dictionary_update(self, Y: np.ndarray, X: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for k in range(self.D):
            phi, X = self._update_dictionary_atom(Y, X, phi, k)
        return phi, X
    
    def fit(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print("Starting K-SVD dictionary learning...")
        start_time = time.time()
        
        phi = self._initialize_dictionary(Y)
        N = Y.shape[1]
        
        # For single-frame learning, reduce iteration count and add regularization
        if N == 1:
            effective_iterations = min(self.max_iterations, 10)  # Maximum 10 iterations for single frame
            print(f"Single-frame learning mode: Limiting iterations to {effective_iterations}")
        else:
            effective_iterations = self.max_iterations
        
        prev_reconstruction_error = float('inf')
        
        for iteration in range(effective_iterations):
            print(f"Iteration {iteration + 1}/{effective_iterations}")
            
            X = self._sparse_coding(Y, phi)
            
            # Calculate reconstruction error
            reconstruction = phi @ X
            current_error = np.linalg.norm(Y - reconstruction, 'fro')
            
            # For single-frame learning, stop early if error is already very small
            if N == 1 and current_error < 1e-6:
                print(f"  Single-frame learning converged early, reconstruction error: {current_error:.2e}")
                break
            
            # Early stopping mechanism: If error no longer decreases significantly
            if abs(prev_reconstruction_error - current_error) < 1e-8:
                print(f"  Convergence check: Error change < 1e-8, stopping early")
                break
            
            prev_reconstruction_error = current_error
            
            phi, X = self._dictionary_update(Y, X, phi)
            
            elapsed_time = time.time() - start_time
            print(f"  Completed in {elapsed_time:.2f}s, reconstruction error: {current_error:.2e}")
        
        self.dictionary = phi
        self.sparse_representation = X
        
        print("Dictionary learning complete.")
        return phi, X
    
    def estimate_noise_precision(self, Y: np.ndarray) -> float:
        if self.dictionary is None or self.sparse_representation is None:
            raise ValueError("Must call fit() before estimating noise precision")
        
        N_est = Y - self.dictionary @ self.sparse_representation
        SSE = np.sum(N_est ** 2)
        
        M, N = Y.shape
        sigma_squared = SSE / (M * N)
        
        # Prevent division by zero and numerical instability
        if sigma_squared <= 1e-12:
            # When reconstruction error is too small, use reasonable estimation based on signal amplitude
            signal_variance = np.var(Y)
            sigma_squared = max(1e-6, signal_variance * 1e-4)  # Use 1/10000 of signal variance as noise variance
            print(f"    Noise variance too small, using signal variance estimation: σ²={sigma_squared:.6e}")
            
        beta_global = 1 / sigma_squared
        
        # Limit noise precision range to prevent outliers
        beta_global = min(beta_global, 1e6)  # Maximum precision limit
        beta_global = max(beta_global, 1e-3)  # Minimum precision limit
        
        return beta_global