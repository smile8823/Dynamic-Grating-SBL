#!/usr/bin/env python3
"""
Stage 1 Single Peak Signal Dictionary Learning and Single Frame Reconstruction Comparison Script

Functionality:
- Read specified CSV from data directory (e.g., right_fbg5s.csv)
- Perform Stage 1 dictionary learning (K-SVD) using the first m frames, save dictionary to learned_dicts directory in project root
- Perform sparse reconstruction (OMP) on a specified frame using the learned dictionary, plot original vs reconstructed waveform comparison

Usage Example:
python scripts/stage1_single_peak_reconstruct.py --data_file data/right_fbg5s.csv -m 200 -d 64 -k 3 --frame_index -1

Arguments:
- --data_file/-f        Data file path (CSV), defaults to finding the first CSV in data directory automatically
- --learning_frames/-m  Number of frames for dictionary learning, default 100 (-1 means use all frames)
- --dict_size/-d        Dictionary size (number of atoms), default 64
- --sparsity/-k         Sparsity level K, default 3
- --max_iterations      Max K-SVD iterations, default 10 (usually sufficient for single peak/single frame scenarios)
- --frame_index         Frame index to reconstruct, default -1 (last frame)
- --output_dir          Output image directory, default output
"""

import os
import sys
import argparse
import datetime
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add project src directory to path for module imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src', 'modules'))

from modules.data_reader import read_csv_data
from modules.dictionary_learning import DictionaryLearning


def configure_fonts():
    """Configure fonts to ensure text displays correctly."""
    import matplotlib
    # Try common fonts that support necessary characters
    try_fonts = [
        'Arial',
        'DejaVu Sans',
        'Liberation Sans',
        'Segoe UI',
        'Helvetica',
    ]
    for fname in try_fonts:
        try:
            # Check if font is available
            # Note: matplotlib.font_manager.findfont might raise error or return default if not found
            # Here we just set it and let matplotlib handle fallback usually
            matplotlib.rcParams['font.sans-serif'] = [fname] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            # print(f"Font set: {fname}") # Optional: suppress font setting message
            break
        except Exception as e:
            print(f"Failed to set font {fname}: {e}")


def save_dictionary_npz(phi: np.ndarray, wavelengths: np.ndarray, dict_name: Optional[str] = None) -> str:
    os.makedirs(os.path.join(PROJECT_ROOT, 'learned_dicts'), exist_ok=True)
    if dict_name is None or len(dict_name.strip()) == 0:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dict_filename = f'learned_dictionary_{ts}.npz'
    else:
        dict_filename = dict_name if dict_name.lower().endswith('.npz') else f'{dict_name}.npz'
    dict_path = os.path.join(PROJECT_ROOT, 'learned_dicts', dict_filename)
    np.savez(dict_path, phi=phi, wavelengths=wavelengths)
    print(f"Dictionary saved: {dict_path}")
    return dict_path


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_comparison(wavelengths: np.ndarray,
                    y_original: np.ndarray,
                    y_reconstructed: np.ndarray,
                    frame_index: int,
                    dict_path: Optional[str],
                    output_path: str,
                    extra_text: Optional[str] = None) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(wavelengths, y_original, label='Original Waveform (Solid)', color='tab:blue', linewidth=1.8, linestyle='-')
    plt.plot(wavelengths, y_reconstructed, label='Dictionary Reconstruction (Dashed)', color='tab:red', linewidth=1.8, linestyle='--')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Amplitude')
    title = f'Single Peak Signal: Original vs Dictionary Reconstruction | Frame {frame_index}'
    if extra_text:
        title += f' | {extra_text}'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Image saved: {output_path}")


def main():
    # configure_fonts() removed as requested

    parser = argparse.ArgumentParser(description='Stage 1 Single Peak Signal Dictionary Learning and Single Frame Reconstruction Comparison')
    parser.add_argument('--data_file', '-f', type=str, default=None, help='CSV data file path, defaults to automatically selecting the first CSV in data directory')
    parser.add_argument('--learning_frames', '-m', type=int, default=100, help='Number of frames for dictionary learning (-1 means use all frames)')
    parser.add_argument('--dict_size', '-d', type=int, default=64, help='Dictionary size/Number of atoms')
    parser.add_argument('--sparsity', '-k', type=int, default=3, help='Sparsity level K')
    parser.add_argument('--max_iterations', type=int, default=10, help='Max K-SVD iterations')
    parser.add_argument('--frame_index', type=int, default=-1, help='Frame index for reconstruction (-1 for last frame)')
    parser.add_argument('--output_dir', type=str, default='output', help='Output image directory')
    parser.add_argument('--dict_name', type=str, default=None, help='Dictionary filename (without extension), e.g.: dict1')
    args = parser.parse_args()

    # Automatically select first CSV in data directory
    if args.data_file is None:
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        csv_file = None
        if os.path.isdir(data_dir):
            for name in os.listdir(data_dir):
                if name.lower().endswith('.csv'):
                    csv_file = os.path.join(data_dir, name)
                    break
        if csv_file is None:
            raise FileNotFoundError('No CSV file found in data directory, please specify using --data_file')
        args.data_file = csv_file

    print(f"Reading data file: {args.data_file}")
    wavelengths, data_matrix = read_csv_data(csv_filename=args.data_file)
    M, total_frames = data_matrix.shape
    print(f"Data dimensions M={M}, Frames N={total_frames}")

    # Select training frames
    if args.learning_frames is None or args.learning_frames < 0 or args.learning_frames > total_frames:
        m = total_frames
    else:
        m = args.learning_frames
    Y_train = data_matrix[:, :m]
    print(f"Frames used for dictionary learning: {m}")

    # Stage 1 Dictionary Learning
    dl = DictionaryLearning(M=M, D=args.dict_size, K=args.sparsity, max_iterations=args.max_iterations)
    print("Starting K-SVD dictionary learning...")
    phi, X = dl.fit(Y_train)
    dict_path = save_dictionary_npz(phi, wavelengths, dict_name=args.dict_name)

    # Select reconstruction frame
    frame_idx = args.frame_index
    if frame_idx < 0 or frame_idx >= total_frames:
        frame_idx = total_frames - 1
    y = data_matrix[:, frame_idx]
    print(f"Reconstruction frame index: {frame_idx}")

    # Sparse Reconstruction (OMP)
    x_rec = dl._orthogonal_matching_pursuit(y, phi)
    y_hat = phi @ x_rec
    err = rmse(y_hat, y)
    print(f"Reconstruction RMSE: {err:.6f}")

    # Plot and save
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f'stage1_single_peak_reconstruct_{ts}.png'
    out_path = os.path.join(args.output_dir, out_name)
    extra = f'RMSE={err:.3f} | D={args.dict_size} | K={args.sparsity} | m={m}'
    plot_comparison(wavelengths, y, y_hat, frame_idx, dict_path, out_path, extra_text=extra)


if __name__ == '__main__':
    main()
