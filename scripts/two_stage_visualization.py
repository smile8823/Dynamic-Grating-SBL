#!/usr/bin/env python3
"""
DG-SBL Two-Stage Independent Learning Reconstruction Script

Functionality:
1. Load data from data folder, perform n samplings
2. Perform independent Stage 1 (Dictionary Learning) and Stage 2 (Waveform Reconstruction) for each sampled frame
3. Plot comparison between original and reconstructed waveforms
4. Learning and reconstruction for each sample are independent, without reference to previous frames

Usage:
==========

Basic Usage:
python two_stage_visualization.py

Run with default parameters (sampling count 5, dictionary size 512, sparsity 3)

Custom Parameters:
python two_stage_visualization.py --sampling_count 10 --dict_size 256 --sparsity 2

Arguments:
==========

Optional Arguments:
  --sampling_count, -n     Sampling count (default: 5)
                          Suggested range: 2-20, more samples make comparison more obvious
  
  --dict_size, -d          Dictionary size/Number of atoms (default: 512)
                          Suggested range: 50-1024, larger dictionary means stronger capability but more calculation
  
  --sparsity, -s           Sparsity constraint (default: 3)
                          Suggested range: 1-10, smaller value means stronger sparsity
  
  --max_iterations         K-SVD max iterations (default: 20)
                          Suggested range: 10-50, more iterations mean better convergence but longer time
  
  --output_dir             Output directory (default: output in project root)
                          Specify location to save generated images and metadata
  
  --save_metadata          Whether to save metadata (default: True)
                          Default is on; use --no_save_metadata to turn off

  --xlim                   X-axis range for waveform reconstruction comparison plot, format m,n (e.g. 1538,1542)

Usage Examples:
==========

1. Quick Test (Small Parameters):
   python two_stage_visualization.py -n 3 -d 50 -s 5

2. Standard Configuration:
   python two_stage_visualization.py --sampling_count 10

3. High Precision Configuration:
   python two_stage_visualization.py -n 15 -d 1024 -s 2 --max_iterations 30

Output Files:
==========
- independent_waveform_reconstruction_YYYYMMDD_HHMMSS.png  # Independent reconstruction comparison plot
- plot_metadata_YYYYMMDD_HHMMSS.pkl                       # Complete metadata (saved by default)
- plot_summary_YYYYMMDD_HHMMSS.json                       # Result summary (saved by default)

Notes:
==========
1. Ensure at least one CSV file exists in data directory
2. Script will automatically create output directory (if not exists)
3. Dictionary learning is performed independently for each sample, dictionary files are not saved
4. Suggest testing with small parameters first when processing large datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import argparse
import time
import pickle
import json
from datetime import datetime
from typing import Tuple, Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))

# Import required modules
from modules.data_reader import read_csv_data, get_fbg_training_data
from modules.dictionary_learning import DictionaryLearning
from core.optimized_pytorch_sbl import OptimizedPyTorchSBL
import torch

# Helper functions for dictionary shifting and amplitude matching
import numpy as _np

def shift_dictionary(phi: _np.ndarray, wavelengths: _np.ndarray, delta_nm: float) -> _np.ndarray:
    """Dictionary shift function"""
    M, D = phi.shape
    shifted = _np.zeros_like(phi)
    for j in range(D):
        shifted[:, j] = _np.interp(wavelengths - delta_nm, wavelengths, phi[:, j], left=0.0, right=0.0)
    return shifted


def amplitude_match(y: _np.ndarray, yhat: _np.ndarray):
    """Amplitude adjustment based on highest point matching: match the highest point of reconstruction with original"""
    y_max = float(_np.max(y))
    yhat_max = float(_np.max(yhat))
    
    if yhat_max <= 1e-12:
        return yhat, 1.0
    
    # Calculate scaling factor: original max / reconstruction max
    alpha = y_max / yhat_max
    return alpha * yhat, alpha


def find_best_shift_sbl(y: _np.ndarray, phi: _np.ndarray, wavelengths: _np.ndarray,
                        k_sparsity: int = 3, beta_init: float = 1.0,
                        search_max_nm: float = 0.5):
    """DG-SBL Reconstruction: Based on dictionary shift matching"""
    from core.optimized_pytorch_sbl import OptimizedPyTorchSBL as _SBL
    dx = float(wavelengths[1] - wavelengths[0]) if len(wavelengths) > 1 else 0.001
    step = max(dx, 0.02)
    deltas = _np.arange(-search_max_nm, search_max_nm + step/2, step)
    best_rmse = float('inf')
    best_delta = 0.0
    best_recon = _np.zeros_like(y)
    device = 'cpu'
    
    for delta in deltas:
        phi_shift = shift_dictionary(phi, wavelengths, delta)
        try:
            sbl = _SBL(dictionary=phi_shift, device=device)
            sbl.config.k_sparsity = k_sparsity
            mu, _, _ = sbl._fast_sbl_inference(y=sbl.phi.new_tensor(y.astype(_np.float32)), beta_init=beta_init)
            yhat = (phi_shift @ mu.cpu().numpy()).astype(_np.float64)
        except Exception:
            from modules.dictionary_learning import DictionaryLearning as _DL
            dl_tmp = _DL(M=phi.shape[0], D=phi.shape[1], K=k_sparsity, max_iterations=1)
            X_tmp = dl_tmp._sparse_coding(y.reshape(-1, 1), phi_shift)
            yhat = (phi_shift @ X_tmp[:, 0]).astype(_np.float64)
        
        yhat_matched, _ = amplitude_match(y, yhat)
        rmse = float(_np.sqrt(_np.mean((y - yhat_matched) ** 2)))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_delta = float(delta)
            best_recon = yhat_matched
    
    return best_recon, best_delta, best_rmse


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DG-SBL Two-Stage Independent Learning Reconstruction')
    
    # Basic parameters
    parser.add_argument('--sampling_count', '-n', type=int, default=5,
                       help='Sampling count (default: 5)')
    
    # Dictionary learning parameters
    parser.add_argument('--dict_size', '-d', type=int, default=512,
                       help='Dictionary size (default: 512)')
    parser.add_argument('--sparsity', '-k', type=int, default=3,
                       help='Sparsity (default: 3)')
    parser.add_argument('--max_iterations', '-i', type=int, default=20,
                       help='K-SVD max iterations (default: 20)')
    
    # Output parameters
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory (default: output in project root)')
    parser.add_argument('--save_metadata', dest='save_metadata', action='store_true', default=True,
                       help='Save image metadata for later optimization (default: True)')
    parser.add_argument('--no_save_metadata', dest='save_metadata', action='store_false',
                       help='Do not save image metadata')
    parser.add_argument('--xlim', type=str, default=None,
                       help='X-axis range for waveform reconstruction comparison plot, format m,n (e.g. 1538,1542)')
    
    return parser.parse_args()


# Font configuration
from matplotlib import font_manager

def configure_fonts():
    """Configure fonts"""
    # Placeholder for font configuration if needed
    pass


class TwoStageVisualizer:
    """Two-Stage Independent Learning Reconstruction Visualizer"""
    
    def __init__(self, args):
        self.args = args
        self.wavelengths = None
        self.full_data = None
        self.sampling_count = args.sampling_count
        
        # Result storage
        self.results = {}
        
        # Plot metadata storage
        self.plot_metadata = {}
        
        # Parse and normalize output directory (default to project root's output)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        default_output_dir = os.path.join(project_root, 'output')
        resolved_output_dir = os.path.abspath(args.output_dir or default_output_dir)
        self.args.output_dir = resolved_output_dir
        
        # Ensure output directory exists
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        print("="*60)
        print("DG-SBL Two-Stage Independent Learning Reconstruction System")
        print("="*60)
        print(f"Sampling Count: {self.sampling_count}")
        print(f"Dictionary Size: {args.dict_size}")
        print(f"Sparsity: {args.sparsity}")
        print(f"Output Directory: {self.args.output_dir}")
        
        # Configure fonts
        configure_fonts()
        
        # Parse x-axis range parameter (optional)
        self.xlim = (1546.0, 1555.0)  # Modified to match actual data wavelength range
        try:
            xlim_arg = getattr(args, 'xlim', None)
            if xlim_arg:
                parts = xlim_arg.split(',')
                if len(parts) == 2:
                    self.xlim = (float(parts[0].strip()), float(parts[1].strip()))
                else:
                    print("Warning: --xlim format should be m,n, e.g. 1546,1555; parameter ignored")
        except Exception:
            print("Warning: Failed to parse --xlim; parameter ignored")

    def load_data(self):
        """Load CSV data and initialize internal state"""
        try:
            # Try to read CSV files from data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in data directory")
            
            # Use the first CSV file
            csv_file = csv_files[0]
            csv_path = os.path.join(data_dir, csv_file)
            print(f"Loading data file: {csv_file}")
            
            self.wavelengths, self.full_data = read_csv_data(csv_path)
            print(f"Data loaded: Wavelength points={len(self.wavelengths)}, Frames={self.full_data.shape[1]}")
        except Exception as e:
            print(f"Error: Data loading failed: {e}")
            # Fallback to synthetic data to ensure flow runs
            M = 400
            T = 120
            wl_start, wl_end = 1528.0, 1532.0
            self.wavelengths = np.linspace(wl_start, wl_end, M)
            centers = 1530.0 + 0.2 * np.sin(np.linspace(0, 6*np.pi, T))
            self.full_data = np.zeros((M, T))
            for t in range(T):
                center = centers[t]
                x = self.wavelengths
                peak = np.exp(-((x - center) ** 2) / (2 * (0.08 ** 2)))
                oscill = 0.05 * np.sin(2 * np.pi * (x - 1528.0))
                noise = 0.02 * np.random.randn(M)
                self.full_data[:, t] = 1.0 * peak + 0.2 * oscill + noise
            print(f"Synthetic data generated: Wavelength points={M}, Frames={T}")

    def independent_learning_reconstruction(self):
        """Independent learning reconstruction: independent dictionary learning and reconstruction for each sample"""
        if self.full_data is None:
            raise RuntimeError("Data not loaded. Please call load_data() first.")
        
        total_frames = self.full_data.shape[1]
        n = min(self.sampling_count, total_frames)
        
        # Uniform sampling across all frames
        sample_indices = np.linspace(0, total_frames - 1, num=n, dtype=int)
        
        original_signals = []
        reconstructed_signals = []
        best_shifts = []
        rmse_list = []
        learning_times = []
        
        print(f"Starting independent learning reconstruction, {n} samples in total...")
        
        for i in range(n):
            frame_idx = sample_indices[i]
            y = self.full_data[:, frame_idx]
            
            print(f"Processing sample {i+1}/{n} (frame {frame_idx})...")
            
            # Independent dictionary learning (using current frame only)
            start_time = time.time()
            M = len(y)
            dl = DictionaryLearning(M=M, D=self.args.dict_size, K=self.args.sparsity, 
                                  max_iterations=self.args.max_iterations)
            
            # Dictionary learning using current frame
            Y_train = y.reshape(-1, 1)  # Single frame data
            phi, X = dl.fit(Y_train)
            
            try:
                beta_global = dl.estimate_noise_precision(Y_train)
            except Exception:
                beta_global = 1.0
            
            learning_time = time.time() - start_time
            learning_times.append(learning_time)
            
            # Reconstruction using the learned dictionary
            yhat, delta, rmse = find_best_shift_sbl(
                y, phi, self.wavelengths,
                k_sparsity=self.args.sparsity,
                beta_init=beta_global
            )
            
            original_signals.append(y)
            reconstructed_signals.append(yhat)
            best_shifts.append(delta)
            rmse_list.append(rmse)
            
            print(f"  Dictionary learning time: {learning_time:.3f}s, RMSE: {rmse:.6f}")
            
            # Diagnostic info: check for poor reconstruction quality
            if rmse > 0.01:  # RMSE threshold, above which reconstruction is considered poor
                print(f"    Warning: Poor reconstruction for frame {frame_idx} (RMSE={rmse:.6f})")
                print(f"    Original signal: mean={y.mean():.3f}, std={y.std():.3f}, max={y.max():.3f}")
                print(f"    Reconstructed signal: mean={yhat.mean():.3f}, std={yhat.std():.3f}, max={yhat.max():.3f}")
                print(f"    Wavelength shift: {delta:.6f} nm")
                print(f"    Noise precision estimate: {beta_global:.6f}")
                
                # Detailed signal analysis
                residual = y - yhat
                print(f"    Residual analysis: mean={residual.mean():.6f}, std={residual.std():.6f}")
                print(f"    Residual range: [{residual.min():.6f}, {residual.max():.6f}]")
                
                # Signal quality metrics
                snr_original = 20 * np.log10(np.std(y) / (np.std(residual) + 1e-10))
                print(f"    SNR: {snr_original:.2f} dB")
                
                # Peak position analysis
                y_peak_idx = np.argmax(y)
                yhat_peak_idx = np.argmax(yhat)
                peak_shift_points = abs(y_peak_idx - yhat_peak_idx)
                print(f"    Peak position shift: {peak_shift_points} sample points")
                
                # Dictionary atom usage
                if hasattr(dl, 'sparse_representation') and dl.sparse_representation is not None:
                    active_atoms = np.sum(np.abs(dl.sparse_representation) > 1e-6)
                    print(f"    Active dictionary atoms: {active_atoms}/{self.args.dict_size}")
                    
                    # Sparse coefficient analysis
                    sparse_coeffs = dl.sparse_representation.flatten()
                    nonzero_coeffs = sparse_coeffs[np.abs(sparse_coeffs) > 1e-6]
                    if len(nonzero_coeffs) > 0:
                        print(f"    Sparse coefficient range: [{nonzero_coeffs.min():.6f}, {nonzero_coeffs.max():.6f}]")
                        print(f"    Sparse coefficient mean: {nonzero_coeffs.mean():.6f}")
                
                # Check dictionary learning convergence
                if hasattr(dl, 'convergence_history'):
                    print(f"    Dictionary learning convergence history: {dl.convergence_history[-3:]}")  # Show last 3 iterations
                    
                # Check for numerical anomalies
                if np.any(np.isnan(yhat)) or np.any(np.isinf(yhat)):
                    print(f"    ⚠️  Reconstructed signal contains NaN or Inf!")
                if np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
                    print(f"    ⚠️  Dictionary contains NaN or Inf!")
        
        # Calculate quality metrics
        mae_list = [float(np.mean(np.abs(original_signals[i] - reconstructed_signals[i]))) for i in range(n)]
        
        # Calculate percentage error (RMSE relative to signal amplitude)
        rmse_percentage_list = []
        for i in range(n):
            y = original_signals[i]
            signal_range = y.max() - y.min()  # Signal amplitude range
            if signal_range > 0:
                rmse_percentage = (rmse_list[i] / signal_range) * 100
            else:
                rmse_percentage = 0.0
            rmse_percentage_list.append(float(rmse_percentage))
        
        cosine_list = []
        for i in range(n):
            y = original_signals[i]
            yhat = reconstructed_signals[i]
            denom = np.linalg.norm(y) * np.linalg.norm(yhat)
            cosine = float(np.dot(y, yhat) / denom) if denom > 1e-12 else 0.0
            cosine_list.append(cosine)
        
        # Output detailed metrics per frame
        print("\nDetailed reconstruction quality metrics per frame:")
        print("Frame Index | RMSE(%) | Cosine Similarity | Learning Time(s)")
        print("-" * 45)
        for i in range(n):
            frame_idx = sample_indices[i]
            print(f"  {frame_idx:3d}   | {rmse_percentage_list[i]:6.3f}  | {cosine_list[i]:.8f} | {learning_times[i]:7.3f}")
        
        metrics = {
            'rmse_per_frame': rmse_list,
            'rmse_percentage_per_frame': rmse_percentage_list,
            'mae_per_frame': mae_list,
            'cosine_similarity_per_frame': cosine_list,
            'learning_times': learning_times,
            'sample_indices': sample_indices.tolist(),
            'rmse_avg': float(np.mean(rmse_list)) if rmse_list else None,
            'rmse_percentage_avg': float(np.mean(rmse_percentage_list)) if rmse_percentage_list else None,
            'mae_avg': float(np.mean(mae_list)) if mae_list else None,
            'cosine_similarity_avg': float(np.mean(cosine_list)) if cosine_list else None,
            'learning_time_avg': float(np.mean(learning_times)) if learning_times else None
        }
        
        self.results = {
            'original_signals': original_signals,
            'reconstructed_signals': reconstructed_signals,
            'sample_frame_indices': sample_indices.tolist(),
            'best_shifts_nm': best_shifts,
            'metrics': metrics
        }
        
        print(f"Independent learning reconstruction complete!")
        print(f"Average dictionary learning time: {metrics['learning_time_avg']:.3f}s")
        print(f"Reconstruction quality: RMSE percentage avg={metrics['rmse_percentage_avg']:.3f}%, Cosine similarity avg={metrics['cosine_similarity_avg']:.8f}")

    def create_waveform_comparison_plot(self):
        print("Creating independent learning reconstruction comparison plot...")
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 32
        tick_size = 32
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        original_signals = self.results['original_signals']
        reconstructed_signals = self.results['reconstructed_signals']
        sample_indices = self.results['sample_frame_indices']
        n_samples = len(original_signals)
        fig, axes = plt.subplots(2, 2, figsize=(24, 18), sharex=True, constrained_layout=True)
        axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
        max_plots = min(4, n_samples)
        for idx in range(4):
            ax = axes_flat[idx]
            ax.tick_params(labelsize=tick_size, labelbottom=True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.8)
            ax.spines['bottom'].set_linewidth(1.8)
            ax.grid(True, which='major', axis='both', linestyle='--', linewidth=1.8, color='#555555', alpha=0.6)
            if idx in (2, 3):
                ax.set_xlabel('Wavelength (nm)')
            else:
                ax.set_xlabel('')
            ax.set_ylabel('')
        for i in range(max_plots):
            ax = axes_flat[i]
            ax.plot(self.wavelengths, original_signals[i], color='#08306b', linewidth=3.0, linestyle='-')
            ax.plot(
                self.wavelengths,
                reconstructed_signals[i],
                color='#e31a1c',
                linestyle='None',
                marker='o',
                markersize=12,
                markerfacecolor='none',
                markeredgewidth=2.5,
                markeredgecolor='#e31a1c',
                markevery=30
            )
            ax.text(0.02, 0.98, f"Sample {sample_indices[i]}", transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='#000000', linewidth=0.8, pad=0.2))
            if self.xlim is not None:
                ax.set_xlim(self.xlim[0], self.xlim[1])
        for j in range(max_plots, 4):
            axes_flat[j].axis('off')
        handles = [
            plt.Line2D([], [], color='#08306b', linewidth=3.0, linestyle='-', label='Original'),
            plt.Line2D([], [], color='#e31a1c', linestyle='None', marker='o', markersize=12, markerfacecolor='none', markeredgewidth=2.5, markeredgecolor='#e31a1c', label='Reconstructed')
        ]
        legend_title = f"{max_plots} samples"
        leg = fig.legend(handles=handles, loc='lower center', ncol=2, frameon=True, framealpha=1.0, title=legend_title)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#000000')
        leg.get_frame().set_linewidth(1.2)
        info_text = f"Dict={self.args.dict_size}, Sparsity={self.args.sparsity}, Iter={self.args.max_iterations}"
        metrics = self.results.get('metrics', {})
        if metrics:
            info_text += f"; RMSE% Avg={metrics.get('rmse_percentage_avg', float('nan')):.3f}%; Cosine Avg={metrics.get('cosine_similarity_avg', float('nan')):.8f}; Avg Learning Time={metrics.get('learning_time_avg', float('nan')):.3f}s"
        print(info_text)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"independent_waveform_reconstruction_{timestamp}.png"
        filepath = os.path.join(self.args.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plot_metadata['waveform_comparison'] = {
            'figure': fig,
            'axes': axes_flat,
            'filename': filename,
            'filepath': filepath,
            'data': {
                'wavelengths': self.wavelengths,
                'original_signals': original_signals,
                'reconstructed_signals': reconstructed_signals,
                'sample_indices': sample_indices
            }
        }
        plt.close()
        print(f"Independent learning reconstruction comparison plot saved: {filepath}")
        return filepath

    def plot_first_raw_frame(self):
        if self.full_data is None or self.wavelengths is None:
            return None
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 32
        tick_size = 32
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        y = self.full_data[:, 0]
        fig, ax = plt.subplots(figsize=(32, 16), constrained_layout=True)
        ax.tick_params(labelsize=tick_size)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.plot(self.wavelengths, y, color='#08306b', linewidth=22.4)
        if self.xlim is not None:
            ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(7.2)
        ax.spines['bottom'].set_linewidth(7.2)
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=7.2, color='#555555', alpha=0.6)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"first_raw_frame_{timestamp}.png"
        filepath = os.path.join(self.args.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plot_metadata['first_raw_frame'] = {
            'figure': fig,
            'axes': [ax],
            'filename': filename,
            'filepath': filepath,
            'data': {
                'wavelengths': self.wavelengths,
                'raw_frame': y.tolist()
            }
        }
        plt.close()
        print(f"First raw frame plot saved: {filepath}")
        return filepath

    def save_plot_metadata(self):
        """Save plot metadata"""
        if not self.args.save_metadata:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full metadata (pickle format)
        metadata_filename = f"plot_metadata_{timestamp}.pkl"
        metadata_filepath = os.path.join(self.args.output_dir, metadata_filename)
        
        try:
            with open(metadata_filepath, 'wb') as f:
                pickle.dump(self.plot_metadata, f)
            print(f"Plot metadata saved: {metadata_filepath}")
        except Exception as e:
            print(f"Failed to save plot metadata: {e}")
        
        # Save result summary (JSON format)
        summary_filename = f"plot_summary_{timestamp}.json"
        summary_filepath = os.path.join(self.args.output_dir, summary_filename)
        
        try:
            summary = {
                'timestamp': timestamp,
                'parameters': {
                    'sampling_count': self.args.sampling_count,
                    'dict_size': self.args.dict_size,
                    'sparsity': self.args.sparsity,
                    'max_iterations': self.args.max_iterations
                },
                'data_info': {
                    'wavelength_points': len(self.wavelengths) if self.wavelengths is not None else 0,
                    'total_frames': self.full_data.shape[1] if self.full_data is not None else 0
                },
                'results': {
                    'sample_indices': self.results.get('sample_frame_indices', []),
                    'metrics': self.results.get('metrics', {})
                },
                'output_files': {
                    'waveform_comparison': self.plot_metadata.get('waveform_comparison', {}).get('filename', ''),
                    'first_raw_frame': self.plot_metadata.get('first_raw_frame', {}).get('filename', '')
                }
            }
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Result summary saved: {summary_filepath}")
        except Exception as e:
            print(f"Failed to save result summary: {e}")

    def run(self):
        """Run the full two-stage independent learning reconstruction process"""
        print("Starting two-stage independent learning reconstruction process...")
        
        # Load data
        self.load_data()
        
        # Plot first raw frame
        self.plot_first_raw_frame()
        
        # Independent learning reconstruction
        self.independent_learning_reconstruction()
        
        # Create visualization plots
        self.create_waveform_comparison_plot()
        
        # Save metadata
        self.save_plot_metadata()
        
        print("="*60)
        print("Two-stage independent learning reconstruction process complete!")
        print("="*60)


def main():
    """Main function"""
    try:
        args = parse_arguments()
        visualizer = TwoStageVisualizer(args)
        visualizer.run()
        return 0
    except KeyboardInterrupt:
        print("\nUser interrupted operation")
        return 1
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
