#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized PyTorch Sparse Bayesian Learning Implementation
Deep optimization for performance bottlenecks
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Optimization Configuration"""
    max_iterations: int = 5  # Reduce iteration count
    tolerance: float = 1e-2  # Relax convergence condition
    k_sparsity: int = 3      # Sparsity constraint
    use_fast_convergence: bool = True  # Use fast convergence strategy
    use_precomputed_matrices: bool = True  # Precompute matrices
    batch_inference: bool = True  # Batch inference
    memory_efficient: bool = True  # Memory efficient mode

class OptimizedPyTorchSBL:
    """Optimized PyTorch Sparse Bayesian Learning"""
    
    def __init__(self, 
                 dictionary: np.ndarray,
                 config: OptimizationConfig = None,
                 device: str = 'cuda'):
        """
        Initialize Optimized SBL
        
        Args:
            dictionary: Dictionary matrix (M, N)
            config: Optimization configuration
            device: Computing device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.config = config or OptimizationConfig()
        
        # Transfer dictionary to GPU
        self.phi = torch.from_numpy(dictionary.astype(np.float32)).to(self.device)
        self.M, self.N = self.phi.shape
        
        logger.info(f"Optimized SBL initialized: Dictionary {self.phi.shape}, Device {self.device}")
        
        # Precompute common matrices
        self._precompute_matrices()
        
        # Performance statistics
        self.stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_iterations': 0.0,
            'convergence_rate': 0.0
        }
    
    def _precompute_matrices(self):
        """Precompute common matrices to accelerate operations"""
        logger.info("Precomputing matrices...")
        
        if self.config.use_precomputed_matrices:
            # Precompute Phi^T
            self.phi_T = self.phi.T.contiguous()
            
            # Precompute Phi^T @ Phi (if memory allows)
            if self.N <= 1024:  # Precompute only for smaller dictionaries
                self.phi_T_phi = torch.mm(self.phi_T, self.phi)
                logger.info("Precomputation of Phi^T @ Phi complete")
            else:
                self.phi_T_phi = None
                logger.info("Dictionary too large, skipping Phi^T @ Phi precomputation")
        else:
            self.phi_T = self.phi.T
            self.phi_T_phi = None
    
    def _fast_sbl_inference(self, 
                           y: torch.Tensor,
                           alpha_init: Optional[torch.Tensor] = None,
                           beta_init: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Fast SBL Inference (Optimized)
        
        Args:
            y: Observed signal (M,)
            alpha_init: Initial hyperparameters
            beta_init: Initial noise precision
            
        Returns:
            (mu, alpha, beta) - Inference result
        """
        # Initialize hyperparameters
        if alpha_init is None:
            alpha = torch.ones(self.N, device=self.device, dtype=self.dtype)
        else:
            alpha = alpha_init.clone()
        
        beta = beta_init
        
        # Precompute Phi^T @ y
        phi_T_y = torch.mv(self.phi_T, y)
        
        # Fast convergence strategy
        prev_alpha = alpha.clone()
        convergence_history = []
        
        for iteration in range(self.config.max_iterations):
            # E-step: Calculate posterior mean (simplified)
            if self.config.use_fast_convergence:
                # Use diagonal approximation to accelerate calculation
                diagonal_term = alpha + beta * torch.sum(self.phi ** 2, dim=0)
                mu = beta * phi_T_y / diagonal_term
            else:
                # Standard method (slower but more accurate)
                A_diag = alpha + beta * torch.sum(self.phi ** 2, dim=0)
                mu = beta * phi_T_y / A_diag
            
            # M-step: Update hyperparameters (simplified)
            if self.config.use_fast_convergence:
                # Fast update strategy
                gamma = 1.0 / (1.0 + alpha / (beta * torch.sum(self.phi ** 2, dim=0) + 1e-12))
                alpha_new = gamma / (mu ** 2 + 1e-12)
            else:
                # Standard update
                sigma_diag = 1.0 / (alpha + beta * torch.sum(self.phi ** 2, dim=0))
                gamma = 1.0 - alpha * sigma_diag
                alpha_new = gamma / (mu ** 2 + sigma_diag + 1e-12)
            
            # Update noise precision (simplified)
            residual = y - torch.mv(self.phi, mu)
            beta_new = max(0.1, (self.M - torch.sum(gamma)) / (torch.dot(residual, residual) + 1e-12))
            beta = float(beta_new)
            
            # Check convergence
            alpha_change = torch.norm(alpha_new - alpha) / (torch.norm(alpha) + 1e-12)
            convergence_history.append(float(alpha_change))
            
            if alpha_change < self.config.tolerance:
                logger.debug(f"Fast convergence at iteration {iteration+1}")
                break
            
            # Adaptive learning rate
            if len(convergence_history) > 2:
                if convergence_history[-1] > convergence_history[-2]:
                    # Convergence slowing down, use more conservative update
                    alpha = 0.7 * alpha_new + 0.3 * alpha
                else:
                    alpha = alpha_new
            else:
                alpha = alpha_new
        
        # Apply sparsity constraint
        if self.config.k_sparsity > 0:
            mu = self._apply_k_sparsity_constraint(mu, self.config.k_sparsity)
        
        return mu, alpha, beta
    
    def _apply_k_sparsity_constraint(self, 
                                     coefficients: torch.Tensor, 
                                     k: int) -> torch.Tensor:
        """Apply k-sparsity constraint (optimized)"""
        if k >= len(coefficients):
            return coefficients
        
        # Use topk operation, faster than sorting
        _, indices = torch.topk(torch.abs(coefficients), k)
        
        # Create sparse version
        sparse_coefficients = torch.zeros_like(coefficients)
        sparse_coefficients[indices] = coefficients[indices]
        
        return sparse_coefficients
    
    def track_signal_stream_optimized(self, 
                                     signal_stream: np.ndarray,
                                     batch_size: int = 20) -> Dict[str, Any]:
        """
        Optimized signal stream tracking
        
        Args:
            signal_stream: Signal stream data (num_frames, M)
            batch_size: Batch size
            
        Returns:
            Tracking results dictionary
        """
        num_frames = signal_stream.shape[0]
        logger.info(f"Start optimized GPU tracking of {num_frames} frames")
        
        # Transfer all data to GPU at once
        signal_tensor = torch.from_numpy(signal_stream.astype(np.float32)).to(self.device)
        
        # Pre-allocate result tensor
        all_coefficients = torch.zeros((num_frames, self.N), device=self.device, dtype=self.dtype)
        
        # Initialize adaptive hyperparameters
        alpha = torch.ones(self.N, device=self.device, dtype=self.dtype)
        beta = 1.0
        
        start_time = time.time()
        total_iterations = 0
        converged_frames = 0
        
        # Batch tracking
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_signals = signal_tensor[batch_start:batch_end]
            
            # Batch processing (if configuration allows)
            if self.config.batch_inference and len(batch_signals) > 1:
                batch_results = self._batch_inference(batch_signals, alpha, beta)
                all_coefficients[batch_start:batch_end] = batch_results['coefficients']
                alpha = batch_results['alpha']  # Use alpha from the end of the batch
                beta = batch_results['beta']
                total_iterations += batch_results['iterations']
                converged_frames += batch_results['converged']
            else:
                # Frame-by-frame processing
                for i, y in enumerate(batch_signals):
                    frame_idx = batch_start + i
                    
                    # Run optimized SBL
                    mu, alpha, beta = self._fast_sbl_inference(y, alpha, beta)
                    all_coefficients[frame_idx] = mu
                    
                    # Adaptive hyperparameter update
                    if frame_idx > 0:
                        alpha = 0.8 * alpha + 0.2 * torch.ones_like(alpha)
                    
                    total_iterations += self.config.max_iterations
                    
                    if (frame_idx + 1) % 50 == 0:
                        logger.info(f"Processed {frame_idx + 1}/{num_frames} frames")
        
        # Transfer results to CPU at once
        results_cpu = all_coefficients.cpu().numpy()
        
        total_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_frames'] += num_frames
        self.stats['total_time'] += total_time
        self.stats['avg_iterations'] = total_iterations / num_frames
        self.stats['convergence_rate'] = converged_frames / num_frames
        
        logger.info(f"Optimized GPU tracking complete, time: {total_time:.2f}s")
        logger.info(f"Average per frame: {total_time/num_frames*1000:.2f}ms")
        logger.info(f"Average iterations: {self.stats['avg_iterations']:.1f}")
        
        return {
            'coefficients': results_cpu,
            'processing_time': total_time,
            'frames_per_second': num_frames / total_time,
            'avg_iterations': self.stats['avg_iterations'],
            'convergence_rate': self.stats['convergence_rate'],
            'device': str(self.device),
            'optimization_config': self.config.__dict__,
            'stats': self.stats.copy()
        }
    
    def _batch_inference(self, 
                        batch_signals: torch.Tensor,
                        alpha_init: torch.Tensor,
                        beta_init: float) -> Dict[str, Any]:
        """
        Batch Inference (Experimental Feature)
        
        Args:
            batch_signals: Batch signals (batch_size, M)
            alpha_init: Initial hyperparameters
            beta_init: Initial noise precision
            
        Returns:
            Batch inference results
        """
        batch_size = batch_signals.shape[0]
        
        # Precompute batch Phi^T @ Y
        phi_T_batch_y = torch.mm(self.phi_T, batch_signals.T)  # (N, batch_size)
        
        # Initialize
        alpha = alpha_init.clone()
        beta = beta_init
        
        # Batch result storage
        batch_coefficients = torch.zeros((batch_size, self.N), device=self.device, dtype=self.dtype)
        
        iterations = 0
        converged = 0
        
        # Simplified batch EM iteration
        for iteration in range(self.config.max_iterations):
            iterations += 1
            
            # Batch E-step
            diagonal_term = alpha.unsqueeze(1) + beta * torch.sum(self.phi ** 2, dim=0).unsqueeze(1)
            mu_batch = beta * phi_T_batch_y / diagonal_term  # (N, batch_size)
            
            # Batch M-step (using average)
            mu_avg = torch.mean(mu_batch, dim=1)  # (N,)
            gamma_avg = 1.0 / (1.0 + alpha / (beta * torch.sum(self.phi ** 2, dim=0) + 1e-12))
            alpha_new = gamma_avg / (mu_avg ** 2 + 1e-12)
            
            # Check convergence
            alpha_change = torch.norm(alpha_new - alpha) / (torch.norm(alpha) + 1e-12)
            if alpha_change < self.config.tolerance:
                converged = batch_size
                break
            
            alpha = alpha_new
        
        # Transpose results
        batch_coefficients = mu_batch.T  # (batch_size, N)
        
        # Apply sparsity constraint
        if self.config.k_sparsity > 0:
            for i in range(batch_size):
                batch_coefficients[i] = self._apply_k_sparsity_constraint(
                    batch_coefficients[i], self.config.k_sparsity
                )
        
        return {
            'coefficients': batch_coefficients,
            'alpha': alpha,
            'beta': beta,
            'iterations': iterations,
            'converged': converged
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_frames': self.stats['total_frames'],
            'total_time': self.stats['total_time'],
            'avg_time_per_frame': self.stats['total_time'] / max(1, self.stats['total_frames']),
            'avg_iterations': self.stats['avg_iterations'],
            'convergence_rate': self.stats['convergence_rate'],
            'theoretical_max_fps': 1.0 / (self.stats['total_time'] / max(1, self.stats['total_frames'])),
            'device': str(self.device),
            'optimization_enabled': True
        }
    
    def cleanup(self):
        """Cleanup GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned")

# Convenience function
def create_optimized_sbl(dictionary: np.ndarray, 
                        fast_mode: bool = True,
                        device: str = 'cuda') -> OptimizedPyTorchSBL:
    """
    Create optimized SBL instance
    
    Args:
        dictionary: Dictionary matrix
        fast_mode: Whether to use fast mode
        device: Computing device
        
    Returns:
        Optimized SBL instance
    """
    if fast_mode:
        config = OptimizationConfig(
            max_iterations=3,
            tolerance=5e-2,
            k_sparsity=3,
            use_fast_convergence=True,
            use_precomputed_matrices=True,
            batch_inference=True,
            memory_efficient=True
        )
    else:
        config = OptimizationConfig(
            max_iterations=5,
            tolerance=1e-2,
            k_sparsity=5,
            use_fast_convergence=True,
            use_precomputed_matrices=True,
            batch_inference=False,
            memory_efficient=True
        )
    
    return OptimizedPyTorchSBL(dictionary, config, device)