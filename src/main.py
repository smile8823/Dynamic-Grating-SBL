"""
DG-SBL Complete System Main Script
From data reading to Stage 1, Stage 2, Stage 3 full process until outputting final wavelength offsets

Complete Workflow:
1. Data reading and preprocessing
2. Stage 1: Dictionary learning and global parameter estimation
3. Stage 2: Covariance-free SBL online dynamic tracking
4. Stage 3: Direction prediction guided multi-signal intelligent tracking
5. Result output and visualization
"""

import numpy as np
import csv
import sys
import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from core.stage1_main import run_stage_one, load_fbg_data
from core.optimized_stage2_main import OptimizedStage2Processor
from core.ultra_fast_stage3 import UltraFastStage3
from modules.data_reader import read_csv_data, get_fbg_training_data
from modules.memory_manager import MemoryManager
from modules.peak_detection import PeakDetectionSystem
from modules.memory_manager import get_global_memory_manager, register_for_cleanup


class DGSBLCompleteSystem:
    """
    DG-SBL Complete System
    
    Integrates all stages to achieve end-to-end wavelength offset tracking
    """
    
    def __init__(self, config_file: Optional[str] = None) -> None:
        """
        Initialize the complete system
        
        Args:
            config_file: Path to configuration file
        """
        self.start_time = time.time()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # System components
        self.phi_final = None
        self.beta_global = None
        self.stage3_controller = None
        self.output_dir = None
        
        # Memory manager
        self.memory_manager = get_global_memory_manager()
        
        # Result storage
        self.stage1_results = {}
        self.stage2_results = {}
        self.stage3_results = {}
        self.final_results = {}
        
        # Performance statistics
        self.system_performance = {
            'stage1_time': 0.0,
            'stage2_time': 0.0,
            'stage3_time': 0.0,
            'total_time': 0.0,
            'memory_usage': 0.0
        }
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration file"""
        default_config = {
            # Data configuration
            'data': {
                'training_frames': 30,
                'test_frames': 50,
                'start_frame': 100,
                'signal_dimension': 4101
            },
            
            # Stage 1 configuration
            'stage1': {
                'dictionary_size': 512,
                'sparsity_level': 3,
                'max_iterations': 5
            },
            
            # Stage 2 configuration
            'stage2': {
                'max_sbl_iterations': 20,
                'sbl_tolerance': 1e-4,
                'epsilon': 1e-9
            },
            
            # Stage 3 configuration
            'stage3': {
                'signals': {
                    'FBG1': {'wavelength_range': (1527, 1537), 'expected_peak_count': 1},
                    'FBG2': {'wavelength_range': (1540, 1550), 'expected_peak_count': 1},
                    'FBG3': {'wavelength_range': (1553, 1563), 'expected_peak_count': 1}
                },
                'quality_threshold': 0.7,
                'confidence_threshold': 0.6,
                'enable_signal_separation': True,
                'min_signal_distance': 0.5,
                'enable_parallel_processing': True,
                'output_format': 'json'
            },
            
            # Output configuration
            'output': {
                'save_results': True,
                'create_plots': True,
                'export_format': 'json',
                'plot_format': 'png'
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # Merge configurations
                self._merge_configs(default_config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
                print("Using default configuration")
                
        return default_config
        
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> None:
        """Recursively merge configurations"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
                
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run complete pipeline"""
        print("=" * 80)
        print("DG-SBL Complete System Started")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Stage 1: Dictionary Learning
            print("\n" + "=" * 50)
            print("Stage 1: Dictionary Learning and Global Parameter Estimation")
            print("=" * 50)
            
            self._run_stage1()
            
            # Stage 2: SBL Tracking
            print("\n" + "=" * 50)
            print("Stage 2: Covariance-free SBL Online Dynamic Tracking")
            print("=" * 50)
            
            self._run_stage2()
            
            # Stage 3: Intelligent Tracking
            print("\n" + "=" * 50)
            print("Stage 3: Direction Prediction Guided Multi-signal Intelligent Tracking")
            print("=" * 50)
            
            self._run_stage3()
            
            # Generate Final Results
            print("\n" + "=" * 50)
            print("Generating Final Results")
            print("=" * 50)
            
            self._generate_final_results()
            
            # Output Results
            self._output_results()
                
            print("\n" + "=" * 80)
            print("DG-SBL Complete System Finished")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Time: {self.system_performance['total_time']:.2f}s")
            print("=" * 80)
            
            return self.final_results
            
        except Exception as e:
            print(f"\nError: System run failed - {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
            
    def run_stage1(self, training_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run Stage 1: Dictionary learning and global parameter estimation
        
        Args:
            training_data: Optional training data, use default if None
            
        Returns:
            Stage 1 result dictionary
        """
        try:
            self._run_stage1()
            return {
                'success': True,
                'phi_final': self.phi_final,
                'beta_global': self.beta_global,
                'dictionary_shape': self.phi_final.shape,
                'processing_time': self.system_performance['stage1_time']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - self.start_time
            }
            
    def run_stage2(self, num_frames: Optional[int] = None, start_frame: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Stage 2: Covariance-free SBL online dynamic tracking
        
        Args:
            num_frames: Number of frames to process, use config value if None
            start_frame: Start frame, use config value if None
            
        Returns:
            Stage 2 result dictionary
        """
        try:
            self._run_stage2()
            return {
                'success': True,
                'x_estimated_stream': self.stage2_results['x_estimated_stream'],
                'y_stream': self.stage2_results['y_stream'],
                'performance_metrics': self.stage2_results['performance_metrics'],
                'num_frames': self.stage2_results['num_frames'],
                'processing_time': self.system_performance['stage2_time']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - self.start_time
            }
            
    def run_stage3(self, initial_frames: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Run Stage 3: Direction prediction guided multi-signal intelligent tracking
        
        Args:
            initial_frames: Initial frame data, use default frames if None
            
        Returns:
            Stage 3 result dictionary
        """
        try:
            self._run_stage3()
            return {
                'success': True,
                'outputs': self.stage3_results['outputs'],
                'num_frames': self.stage3_results['num_frames'],
                'controller': self.stage3_controller,
                'system_health': self.stage3_controller.get_system_report()['health_status'],
                'processing_time': self.system_performance['stage3_time']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - self.start_time
            }
            
    def _run_stage1(self) -> None:
        """Run Stage 1"""
        start_time = time.time()
        
        try:
            # Load training data
            print("Loading training data...")
            N = self.config['data']['training_frames']
            
            # Check if using sampling mode
            use_sampling = self.config['data'].get('use_sampling', False)
            Y = load_fbg_data(num_frames=N, use_sampling=use_sampling)
            print(f"Training data shape: {Y.shape}")
            
            # Run Stage 1
            print("Starting dictionary learning...")
            D = self.config['stage1']['dictionary_size']
            K = self.config['stage1']['sparsity_level']
            max_iterations = self.config['stage1']['max_iterations']
            
            self.phi_final, self.beta_global, self.output_dir = run_stage_one(Y, D, K, max_iterations)
            
            # Store results
            self.stage1_results = {
                'phi_final': self.phi_final,
                'beta_global': self.beta_global,
                'output_dir': self.output_dir,
                'dictionary_shape': self.phi_final.shape,
                'training_frames': N
            }
            
            # Register for cleanup
            if self.phi_final is not None:
                register_for_cleanup(self.phi_final, 'phi_final')
            if self.beta_global is not None:
                register_for_cleanup(self.beta_global, 'beta_global')
            
            self.system_performance['stage1_time'] = time.time() - start_time
            
            print(f"Stage 1 completed, time: {self.system_performance['stage1_time']:.2f}s")
            print(f"Dictionary matrix shape: {self.phi_final.shape}")
            print(f"Global noise precision: {self.beta_global:.6f}")
            
        except Exception as e:
            print(f"Stage 1 failed: {e}")
            raise
            
    def _run_stage2(self) -> None:
        """Run Stage 2"""
        start_time = time.time()
        
        try:
            # Create Stage 2 processor
            print("Initializing Stage 2 processor...")
            processor = OptimizedStage2Processor(device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load Stage 1 results
            stage1_output_dir = self.stage1_results['output_dir']
            dictionary, _ = processor.load_stage1_results(Path(stage1_output_dir))
            
            # Initialize SBL
            processor.initialize_optimized_sbl(dictionary, mode='fast')
            
            # Load test data
            data_file = self.config['data'].get('data_file', '10s_1.csv')
            # Try to find data file path
            if os.path.exists(data_file):
                data_path = data_file
            elif os.path.exists(os.path.join("..", "data", data_file)):
                data_path = os.path.join("..", "data", data_file)
            elif os.path.exists(os.path.join("data", data_file)):
                data_path = os.path.join("data", data_file)
            else:
                data_path = data_file # Let data_reader handle error
                
            wavelengths, signal_data = processor.load_test_data(data_path)
            
            # Process signal stream
            print("Starting SBL tracking...")
            num_frames = self.config['data']['test_frames']
            
            results = processor.process_signal_stream(
                signal_data=signal_data,
                batch_size=min(num_frames, 25),
                max_frames=num_frames
            )
            
            # Store results
            coefficients = results.get('coefficients', [])
            
            self.stage2_results = {
                'x_estimated_stream': coefficients,  # Sparse coefficients
                'coefficients': coefficients,  # Sparse coefficients
                'dictionary': dictionary,  # Dictionary matrix
                'output_dir': None,  # Not saving to file temporarily
                'performance_metrics': {
                    'frames_per_second': results['frames_per_second'],
                    'processing_time': results['processing_time'],
                    'mean_reconstruction_error': results.get('reconstruction_error', 0.0),
                    'mean_sparsity_level': results.get('sparsity_level', 0.0)
                },
                'num_frames': num_frames
            }
            
            self.system_performance['stage2_time'] = time.time() - start_time
            
            print(f"Stage 2 completed, time: {self.system_performance['stage2_time']:.2f}s")
            print(f"Processed frames: {num_frames}")
            print(f"FPS: {results['frames_per_second']:.2f}")
            
        except Exception as e:
            print(f"Stage 2 failed: {e}")
            raise
            
    def _run_stage3(self) -> None:
        """Run Ultra-fast Stage 3"""
        start_time = time.time()
        
        try:
            print("Initializing Ultra-fast Stage 3...")
            
            # Create Ultra-fast Stage 3 tracker
            wavelength_array = np.linspace(1527, 1568, self.config['data']['signal_dimension'])
            self.stage3_controller = UltraFastStage3(self.phi_final, wavelength_array)
            
            # Prepare all frame data
            print("Preparing Stage 3 frame data...")
            all_frames = []
            
            # Check stage2_results data structure
            coefficients = self.stage2_results.get('coefficients', [])
            if len(coefficients) == 0:
                print("Warning: No coefficients data in Stage 2 results")
                return
                
            # Determine number of frames
            if isinstance(coefficients, np.ndarray):
                if len(coefficients.shape) == 2:
                    # Determine frame dimension based on coefficient matrix shape
                    if coefficients.shape[0] > coefficients.shape[1]:
                        # (dict_size, num_frames) format
                        num_frames = coefficients.shape[1]
                        print(f"Detected coefficient matrix format: (dict_size={coefficients.shape[0]}, num_frames={coefficients.shape[1]})")
                    else:
                        # (num_frames, dict_size) format
                        num_frames = coefficients.shape[0]
                        print(f"Detected coefficient matrix format: (num_frames={coefficients.shape[0]}, dict_size={coefficients.shape[1]})")
                else:
                    num_frames = 1
            elif isinstance(coefficients, list):
                num_frames = len(coefficients)
            else:
                num_frames = 1
                
            # Remove frame limit, process all frames
            print(f"Preparing to process all {num_frames} frames data...")
            
            # Prepare frame data in batches, show progress every 100 frames
            batch_size = 100
            for frame_idx in range(num_frames):
                current_frames = self._prepare_stage3_frames(frame_idx)
                all_frames.append(current_frames)
                
                # Show progress
                if (frame_idx + 1) % batch_size == 0 or frame_idx == num_frames - 1:
                    progress = (frame_idx + 1) / num_frames * 100
                    print(f"  Preparing frame data progress: {frame_idx + 1}/{num_frames} ({progress:.1f}%)")
            
            print(f"Starting Ultra-fast Stage 3 tracking, total {len(all_frames)} frames...")
            
            # Batch process all frames (Ultra-fast)
            stage3_results = self.stage3_controller.track_multiple_frames(all_frames)
            
            # Convert result format to maintain compatibility
            stage3_outputs = []
            total_frames = len(stage3_results)
            print(f"Processing {total_frames} frames of Stage 3 results...")
            
            for i, frame_results in enumerate(stage3_results):
                # Simulate original Stage 3 output format
                output = type('Stage3Output', (), {
                    'results': [type('TrackingResult', (), {
                        'signal_id': result.signal_id,
                        'wavelength_offset': result.wavelength_offset,
                        'confidence': result.confidence,
                        'quality': result.confidence,  # Use confidence as quality
                        'status': 'success' if result.confidence > 0.6 else 'warning'
                    })() for result in frame_results],
                    'system_status': type('SystemStatus', (), {'value': 'running'})(),
                    'processing_time': max(r.processing_time for r in frame_results) if frame_results else 0.0,
                    'performance_metrics': {
                        'total_frames': i + 1,
                        'successful_tracks': sum(1 for r in frame_results if r.confidence > 0.6),
                        'failed_tracks': sum(1 for r in frame_results if r.confidence <= 0.6)
                    }
                })()
                stage3_outputs.append(output)
                
                # More detailed progress display
                if total_frames <= 100:
                    # Less than 100 frames, show every 10 frames
                    if (i + 1) % 10 == 0 or i < 5 or i == total_frames - 1:
                        progress = (i + 1) / total_frames * 100
                        if frame_results and all(hasattr(r, 'processing_time') and not np.isnan(r.processing_time) for r in frame_results):
                            avg_time = np.mean([r.processing_time for r in frame_results]) * 1000
                            print(f"  Frame {i+1}/{total_frames} ({progress:.1f}%): Avg processing time {avg_time:.2f}ms")
                        else:
                            print(f"  Frame {i+1}/{total_frames} ({progress:.1f}%)")
                else:
                    # More than 100 frames, show every 100 frames
                    if (i + 1) % 100 == 0 or i < 5 or i == total_frames - 1:
                        progress = (i + 1) / total_frames * 100
                        elapsed_time = time.time() - start_time
                        estimated_total = elapsed_time / (i + 1) * total_frames
                        remaining_time = estimated_total - elapsed_time
                        if frame_results and all(hasattr(r, 'processing_time') and not np.isnan(r.processing_time) for r in frame_results):
                            avg_time = np.mean([r.processing_time for r in frame_results]) * 1000
                            print(f"  Frame {i+1}/{total_frames} ({progress:.1f}%): Avg processing time {avg_time:.2f}ms, Est. remaining time {remaining_time:.1f}s")
                        else:
                            print(f"  Frame {i+1}/{total_frames} ({progress:.1f}%): Est. remaining time {remaining_time:.1f}s")
                    
            # Store results
            self.stage3_results = {
                'outputs': stage3_outputs,
                'num_frames': len(stage3_outputs),
                'controller': self.stage3_controller
            }
            
            self.system_performance['stage3_time'] = time.time() - start_time
            
            print(f"Ultra-fast Stage 3 completed, time: {self.system_performance['stage3_time']:.2f}s")
            print(f"Processed frames: {len(stage3_outputs)}")
            print(f"Avg time per frame: {self.system_performance['stage3_time']/len(stage3_outputs)*1000:.2f}ms")
            print(f"Processing speed: {len(stage3_outputs)/self.system_performance['stage3_time']:.1f} FPS")
            
        except Exception as e:
            print(f"Stage 3 failed: {e}")
            raise
            
    def _prepare_stage3_frames(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Prepare Stage 3 frame data"""
        frames = {}
        
        try:
            # Get sparse coefficients and dictionary from Stage 2 results
            coefficients = self.stage2_results.get('coefficients', [])
            dictionary = self.stage2_results.get('dictionary', None)
            
            if dictionary is None or len(coefficients) == 0:
                print("Warning: Missing dictionary or coefficients data")
                return self._create_default_frames()
            
            # Check coefficient data format and extract current frame coefficients
            if isinstance(coefficients, list):
                if frame_idx >= len(coefficients):
                    frame_idx = len(coefficients) - 1
                if frame_idx < 0:
                    frame_idx = 0
                x_frame = coefficients[frame_idx]
            elif isinstance(coefficients, np.ndarray):
                if len(coefficients.shape) == 1:
                    # 1D array, only one frame data
                    x_frame = coefficients
                elif len(coefficients.shape) == 2:
                    # 2D array, need to determine which dimension is frame dimension
                    # Improved dimension detection logic
                    if coefficients.shape[0] > coefficients.shape[1]:
                        # (dict_size, num_frames) format - dictionary size is usually larger than number of frames
                        if frame_idx >= coefficients.shape[1]:
                            frame_idx = coefficients.shape[1] - 1
                        x_frame = coefficients[:, frame_idx]
                        if frame_idx == 0:  # Show info only on first frame
                            print(f"Using coefficient matrix format: (dict_size={coefficients.shape[0]}, num_frames={coefficients.shape[1]})")
                    else:
                        # (num_frames, dict_size) format - number of frames is smaller than dictionary size
                        if frame_idx >= coefficients.shape[0]:
                            frame_idx = coefficients.shape[0] - 1
                        x_frame = coefficients[frame_idx, :]
                        if frame_idx == 0:  # Show info only on first frame
                            print(f"Using coefficient matrix format: (num_frames={coefficients.shape[0]}, dict_size={coefficients.shape[1]})")
                else:
                    print(f"Warning: Coefficient data dimension anomaly {coefficients.shape}")
                    return self._create_default_frames()
            else:
                print(f"Warning: Coefficient data type anomaly {type(coefficients)}")
                return self._create_default_frames()
            
            # Check coefficient vector dimension
            if len(x_frame) != dictionary.shape[1]:
                print(f"Warning: Coefficient dimension {len(x_frame)} does not match dictionary dimension {dictionary.shape[1]}")
                # Try to adjust dimension
                if len(x_frame) < dictionary.shape[1]:
                    # If coefficient dimension is smaller than dictionary dimension, pad with zeros
                    x_frame_padded = np.zeros(dictionary.shape[1])
                    x_frame_padded[:len(x_frame)] = x_frame
                    x_frame = x_frame_padded
                else:
                    # If coefficient dimension is larger than dictionary dimension, truncate
                    x_frame = x_frame[:dictionary.shape[1]]
            
            # Add debug output: Coefficient statistics
            if frame_idx < 3:  # Show detailed info only for first 3 frames
                print(f"🔍 Frame {frame_idx} debug info:")
                print(f"   Coefficient vector shape: {x_frame.shape}")
                print(f"   Non-zero elements: {np.count_nonzero(x_frame)}")
                print(f"   Max coefficient: {np.max(np.abs(x_frame)):.6f}")
                print(f"   Coefficient energy: {np.sum(x_frame**2):.6f}")
            
            # Reconstruct signal using dictionary: y = D * x
            try:
                y_frame = np.dot(dictionary, x_frame)
                
                # Add debug output: Reconstructed signal statistics
                if frame_idx < 3:
                    print(f"   Reconstructed signal shape: {y_frame.shape}")
                    print(f"   Reconstructed signal range: [{np.min(y_frame):.6f}, {np.max(y_frame):.6f}]")
                    print(f"   Reconstructed signal energy: {np.sum(y_frame**2):.6f}")
                    print(f"   Reconstructed signal mean: {np.mean(y_frame):.6f}")
                    print(f"   Reconstructed signal std: {np.std(y_frame):.6f}")
                    
            except Exception as e:
                print(f"Warning: Signal reconstruction failed {e}")
                print(f"Dictionary shape: {dictionary.shape}, Coefficient shape: {x_frame.shape}")
                return self._create_default_frames()
            
            # Check reconstructed signal
            if len(y_frame) == 0 or not np.isfinite(y_frame).any():
                print(f"Warning: Invalid reconstructed signal at frame {frame_idx}")
                return self._create_default_frames()
            
            # Create local signal for each signal config
            stage3_config = self.config['stage3']['signals']
            
            # Check if using auto detect mode
            if 'auto_detect' in stage3_config and stage3_config['auto_detect'].get('enable', False):
                # Auto detect mode: no segmentation, use entire spectrum
                print(f"🔍 Using auto detect mode, processing entire spectral range")
                
                # Use entire reconstructed signal as input
                frames['auto_signal'] = y_frame
                
                # Add debug output
                if frame_idx < 3:
                    print(f"   Auto detect signal:")
                    print(f"     Signal length: {len(y_frame)}")
                    print(f"     Signal range: [{np.min(y_frame):.6f}, {np.max(y_frame):.6f}]")
                    print(f"     Signal energy: {np.sum(np.abs(y_frame)):.6f}")
                    
            else:
                # Traditional segmentation mode (backward compatible)
                for signal_id, signal_config in stage3_config.items():
                    try:
                        wavelength_range = signal_config['wavelength_range']
                        wavelength_array = np.linspace(1527, 1568, self.config['data']['signal_dimension'])
                        
                        # Safe index calculation
                        min_idx = np.argmin(np.abs(wavelength_array - wavelength_range[0]))
                        max_idx = np.argmin(np.abs(wavelength_array - wavelength_range[1]))
                        
                        # Boundary check and dimension validation
                        min_idx = max(0, min_idx)
                        max_idx = min(len(wavelength_array), max_idx)
                        
                        # Add debug output: Signal extraction process
                        if frame_idx < 3:
                            print(f"   Signal {signal_id} extraction:")
                            print(f"     Wavelength range: {wavelength_range}")
                            print(f"     Index range: [{min_idx}, {max_idx}]")
                            print(f"     Extracted length: {max_idx - min_idx}")
                        
                        # Validate reconstructed signal dimension
                        if len(y_frame) != len(wavelength_array):
                            print(f"⚠️  Reconstructed signal dimension anomaly:")
                            print(f"   Expected dimension: {len(wavelength_array)} (configured signal_dimension)")
                            print(f"   Actual dimension: {len(y_frame)} (Stage 2 reconstruction result)")
                            print(f"   Dictionary dimension: {dictionary.shape[0]} x {dictionary.shape[1]}")
                            
                            # Analyze possible reasons for dimension mismatch
                            if len(y_frame) == dictionary.shape[0]:
                                print(f"   ✓ Reconstructed signal dimension matches dictionary rows, this is normal")
                            else:
                                print(f"   ❌ Reconstructed signal dimension anomaly, possible reasons:")
                                print(f"      - Dimension processing error during Stage 2 reconstruction")
                                print(f"      - Dictionary does not match configured signal_dimension")
                                print(f"      - Coefficient vector dimension error")
                        
                        # Ensure index range is valid
                        if min_idx < max_idx and min_idx < len(y_frame) and max_idx <= len(y_frame):
                            local_signal = y_frame[min_idx:max_idx]
                            
                            # Validate extracted local signal
                            if len(local_signal) == 0:
                                print(f"Warning: Signal {signal_id} extraction result is empty")
                                continue
                                
                            # Check signal quality
                            signal_energy = np.sum(np.abs(local_signal))
                            if signal_energy < 1e-10:
                                print(f"Warning: Signal {signal_id} energy too low ({signal_energy:.2e})")
                            
                            # Add debug output: Local signal statistics
                            if frame_idx < 3:
                                print(f"     Local signal shape: {local_signal.shape}")
                                print(f"     Local signal range: [{np.min(local_signal):.6f}, {np.max(local_signal):.6f}]")
                                print(f"     Local signal energy: {signal_energy:.6f}")
                                print(f"     Local signal mean: {np.mean(local_signal):.6f}")
                                
                        else:
                            print(f"Warning: Signal {signal_id} index range invalid: [{min_idx}, {max_idx}], Reconstructed signal length: {len(y_frame)}")
                            # Create default signal
                            expected_length = max_idx - min_idx
                            local_signal = np.zeros(max(expected_length, 10))  # At least 10 points
                            
                        # Check if local signal is valid
                        if len(local_signal) > 0 and not np.all(np.isnan(local_signal)):
                            frames[signal_id] = local_signal
                        else:
                            frames[signal_id] = self._create_default_signal(signal_id)
                            
                    except Exception as e:
                        print(f"Error processing signal {signal_id}: {e}")
                        frames[signal_id] = self._create_default_signal(signal_id)
                    
        except Exception as e:
            print(f"Error in _prepare_stage3_frames: {e}")
            return self._create_default_frames()
        
        # Add debug output: Final frame data statistics
        if frame_idx < 3:
            print(f"   Final frame data contains {len(frames)} signals")
            for signal_id, signal_data in frames.items():
                print(f"     {signal_id}: Length={len(signal_data)}, Energy={np.sum(np.abs(signal_data)):.6f}")
                
        return frames
        
    def _create_default_signal(self, signal_id: str) -> np.ndarray:
        """Create default signal"""
        # Create small random noise signal to avoid completely empty signal
        np.random.seed(hash(signal_id) % 2147483647)  # Random seed based on signal ID
        default_signal = np.random.randn(100) * 0.01
        return default_signal
        
    def _create_default_frames(self) -> Dict[str, np.ndarray]:
        """Create default frame data"""
        frames = {}
        for signal_id in self.config['stage3']['signals'].keys():
            frames[signal_id] = self._create_default_signal(signal_id)
        return frames
        
    def _find_output_directory(self) -> str:
        """Intelligently find output directory"""
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Possible output directory paths
        possible_paths = [
            os.path.join(current_dir, '..', 'output'),  # output in parent directory of src
            os.path.join(current_dir, 'output'),        # output in src directory
            os.path.join('..', 'output'),              # Relative path
            'output'                                    # Direct path
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # If not found, create a default output directory
        default_output_dir = os.path.join(current_dir, '..', 'output')
        os.makedirs(default_output_dir, exist_ok=True)
        return default_output_dir
        
    def _save_wavelength_offsets_to_csv(self, output_dir: str, timestamp: str) -> str:
        """
        Save wavelength offsets to CSV format
        
        Args:
            output_dir: Output directory path
            timestamp: Timestamp
            
        Returns:
            Saved file path
        """
        try:
            offsets = self.final_results.get('final_wavelength_offsets', {})
            if not offsets:
                print("Warning: No offset data found")
                return ""
                
            # Prepare CSV data
            csv_filename = os.path.join(output_dir, f"wavelength_offsets_{timestamp}.csv")
            
            # Get max frames
            max_frames = max(len(offsets) for offsets in offsets.values()) if offsets else 0
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                headers = ['Frame_Index'] + list(offsets.keys())
                writer.writerow(headers)
                
                # Write data
                for frame_idx in range(max_frames):
                    row = [frame_idx + 1]  # Frame index starts from 1
                    for signal_id in headers[1:]:
                        if frame_idx < len(offsets[signal_id]):
                            row.append(f"{offsets[signal_id][frame_idx]:.8f}")
                        else:
                            row.append("")
                    writer.writerow(row)
                    
                # Add statistics rows
                writer.writerow([])  # Empty row
                writer.writerow(['Statistics'])  # Statistics title
                
                for signal_id in headers[1:]:
                    signal_data = offsets[signal_id]
                    if signal_data:
                        mean_val = np.mean(signal_data)
                        std_val = np.std(signal_data)
                        min_val = np.min(signal_data)
                        max_val = np.max(signal_data)
                        writer.writerow([
                            f'{signal_id}_Mean', f'{mean_val:.8f}'
                        ])
                        writer.writerow([
                            f'{signal_id}_Std', f'{std_val:.8f}'
                        ])
                        writer.writerow([
                            f'{signal_id}_Min', f'{min_val:.8f}'
                        ])
                        writer.writerow([
                            f'{signal_id}_Max', f'{max_val:.8f}'
                        ])
                        
            print(f"Wavelength offsets CSV data saved to: {csv_filename}")
            return csv_filename
            
        except Exception as e:
            print(f"Failed to save offsets CSV data: {e}")
            return ""
            
    def _save_wavelength_offsets_to_json(self, output_dir: str, timestamp: str) -> str:
        """
        Save detailed wavelength offsets to JSON format
        
        Args:
            output_dir: Output directory path
            timestamp: Timestamp
            
        Returns:
            Saved file path
        """
        try:
            offsets = self.final_results.get('final_wavelength_offsets', {})
            if not offsets:
                print("Warning: No offset data found")
                return ""
                
            # Prepare detailed JSON data
            json_filename = os.path.join(output_dir, f"wavelength_offsets_detailed_{timestamp}.json")
            
            detailed_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'system_version': 'DG-SBL v3.0',
                    'data_description': 'Detailed wavelength offsets for each FBG signal',
                    'units': 'nanometers (nm)'
                },
                'signals': {}
            }
            
            # Add detailed data for each signal
            for signal_id, signal_offsets in offsets.items():
                if signal_offsets:
                    signal_data = {
                        'signal_id': signal_id,
                        'total_frames': len(signal_offsets),
                        'wavelength_offsets': signal_offsets,
                        'statistics': {
                            'mean_nm': float(np.mean(signal_offsets)),
                            'std_nm': float(np.std(signal_offsets)),
                            'min_nm': float(np.min(signal_offsets)),
                            'max_nm': float(np.max(signal_offsets)),
                            'range_nm': float(np.max(signal_offsets) - np.min(signal_offsets)),
                            'latest_offset_nm': float(signal_offsets[-1]) if signal_offsets else 0.0
                        },
                        'time_series': {
                            'frame_indices': list(range(1, len(signal_offsets) + 1)),
                            'offsets_nm': [float(x) for x in signal_offsets]
                        }
                    }
                    detailed_data['signals'][signal_id] = signal_data
                    
            # Add overall statistics
            all_offsets = []
            for signal_offsets in offsets.values():
                all_offsets.extend(signal_offsets)
                
            if all_offsets:
                detailed_data['overall_statistics'] = {
                    'total_measurements': len(all_offsets),
                    'overall_mean_nm': float(np.mean(all_offsets)),
                    'overall_std_nm': float(np.std(all_offsets)),
                    'overall_range_nm': float(np.max(all_offsets) - np.min(all_offsets))
                }
                
            # Save to file
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
                
            print(f"Detailed wavelength offsets JSON data saved to: {json_filename}")
            return json_filename
            
        except Exception as e:
            print(f"Failed to save detailed offsets JSON data: {e}")
            return ""
        
    def _generate_final_results(self) -> None:
        """Generate final results"""
        try:
            # Aggregate results from all stages
            final_results = {
                'metadata': {
                    'system_version': 'DG-SBL v3.0',
                    'timestamp': datetime.now().isoformat(),
                    'total_processing_time': time.time() - self.start_time,
                    'config': self.config
                },
                
                'stage1_results': {
                    'dictionary_shape': self.stage1_results['dictionary_shape'],
                    'beta_global': float(self.stage1_results['beta_global']),
                    'training_frames': self.stage1_results['training_frames'],
                    'processing_time': self.system_performance['stage1_time']
                },
                
                'stage2_results': {
                    'num_frames': self.stage2_results['num_frames'],
                    'performance_metrics': self.stage2_results['performance_metrics'],
                    'processing_time': self.system_performance['stage2_time']
                },
                
                'stage3_results': {
                    'num_frames': self.stage3_results['num_frames'],
                    'processing_time': self.system_performance['stage3_time'],
                    'system_health': self.stage3_controller.get_system_report()['health_status'] if self.stage3_controller else 'unknown'
                },
                
                'final_wavelength_offsets': self._extract_final_offsets(),
                
                'system_performance': {
                    'stage1_time': self.system_performance['stage1_time'],
                    'stage2_time': self.system_performance['stage2_time'],
                    'stage3_time': self.system_performance['stage3_time'],
                    'total_time': time.time() - self.start_time
                }
            }
            
            self.final_results = final_results
            
        except Exception as e:
            print(f"Failed to generate final results: {e}")
            self.final_results = {'error': str(e)}
            
    def _extract_final_offsets(self) -> Dict[str, List[float]]:
        """Extract final wavelength offsets"""
        offsets = {}
        
        if self.stage3_results and self.stage3_results['outputs']:
            # Collect all appearing signal IDs
            all_signal_ids = set()
            for output in self.stage3_results['outputs']:
                for result in output.results:
                    all_signal_ids.add(result.signal_id)
            
            # Extract offsets for each signal ID
            for signal_id in all_signal_ids:
                signal_offsets = []
                
                for output in self.stage3_results['outputs']:
                    for result in output.results:
                        if result.signal_id == signal_id:
                            signal_offsets.append(result.wavelength_offset)
                            break
                            
                offsets[signal_id] = signal_offsets
                
        return offsets
        
    def _output_results(self) -> None:
        """Output results - simplified version, only keep wavelength offset data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Intelligently find output directory
            output_dir = self._find_output_directory()
            
            # Decide which results to save based on config
            if self.config['output'].get('save_wavelength_offsets', True):
                # Save wavelength offset data to CSV format
                offsets_csv_file = self._save_wavelength_offsets_to_csv(output_dir, timestamp)
                
                # Save detailed offset data to JSON format
                offsets_json_file = self._save_wavelength_offsets_to_json(output_dir, timestamp)
            
            # Optional: Save reconstructed waveform data for animation comparison
            if self.config['output'].get('save_reconstructed_waveforms', False):
                self._save_reconstructed_waveforms(output_dir, timestamp)
                
            # Print key results
            print("\n=== DG-SBL Algorithm Run Completed ===")
            print(f"Total processing time: {time.time() - self.start_time:.2f}s")
            
            if hasattr(self, 'stage3_results') and self.stage3_results:
                num_frames = self.stage3_results.get('num_frames', 0)
                if num_frames > 0:
                    fps = num_frames / (time.time() - self.start_time)
                    print(f"Processed frames: {num_frames}")
                    print(f"Average FPS: {fps:.2f} FPS")
                    
        except Exception as e:
            print(f"Error occurred while outputting results: {e}")
            raise
                
    def _save_reconstructed_waveforms(self, output_dir: str, timestamp: str) -> None:
        """Save reconstructed waveform data, format compatible with animate_comparison.py"""
        try:
            # Get Stage 2 results
            coefficients = self.stage2_results.get('coefficients', [])
            dictionary = self.stage2_results.get('dictionary', None)
            
            if dictionary is None or len(coefficients) == 0:
                print("Warning: Missing dictionary or coefficient data, cannot generate reconstructed waveforms")
                return
            
            print("Generating reconstructed waveform data...")
            
            # Generate wavelength array
            wavelengths = np.linspace(1527, 1568, dictionary.shape[0])
            
            # Calculate reconstructed waveforms
            Y_reconstructed_list = []
            for coeff in coefficients:
                if isinstance(coeff, np.ndarray):
                    # Reconstruct waveform: Y = dictionary @ coefficients
                    y_reconstructed = dictionary @ coeff
                    Y_reconstructed_list.append(y_reconstructed)
            
            if not Y_reconstructed_list:
                print("Warning: No valid coefficient data")
                return
            
            # Convert to matrix format (frames, wavelength_points) - Fix dimension order
            Y_reconstructed = np.array(Y_reconstructed_list)
            
            # Try to load original data for comparison
            Y_original = None
            try:
                # Load original data from test data path
                data_path = "../data/10s_1.csv"
                if os.path.exists(data_path):
                    original_data = np.loadtxt(data_path, delimiter=',')
                    if original_data.ndim == 2:
                        # Take same number of frames as reconstructed data, and transpose to (frames, wavelength_points)
                        num_frames = min(original_data.shape[1], Y_reconstructed.shape[0])
                        Y_original = original_data[:, :num_frames].T  # Transpose to match dimension
                        Y_reconstructed = Y_reconstructed[:num_frames, :]
            except Exception as e:
                print(f"Failed to load original data: {e}")
            
            # Save reconstructed waveform data
            waveform_path = os.path.join(output_dir, "reconstructed_waveforms.npz")
            
            if Y_original is not None:
                np.savez_compressed(
                    waveform_path,
                    wavelengths=wavelengths,
                    Y_reconstructed=Y_reconstructed,
                    Y_original=Y_original
                )
                print(f"Reconstructed waveform data saved: {waveform_path}")
                print(f"  - Wavelength points: {len(wavelengths)}")
                print(f"  - Frames: {Y_reconstructed.shape[0]}")
                print(f"  - Includes original data comparison")
            else:
                np.savez_compressed(
                    waveform_path,
                    wavelengths=wavelengths,
                    Y_reconstructed=Y_reconstructed
                )
                print(f"Reconstructed waveform data saved: {waveform_path}")
                print(f"  - Wavelength points: {len(wavelengths)}")
                print(f"  - Frames: {Y_reconstructed.shape[0]}")
                print(f"  - Original data not included")
            
        except Exception as e:
            print(f"Failed to save reconstructed waveform data: {e}")
                    
            # Print key results
            print("\n" + "=" * 50)
            print("Final Results Summary")
            print("=" * 50)
            
            print(f"Total processing time: {self.final_results['system_performance']['total_time']:.2f}s")
            print(f"Stage 1 time: {self.final_results['stage1_results']['processing_time']:.2f}s")
            print(f"Stage 2 time: {self.final_results['stage2_results']['processing_time']:.2f}s")
            print(f"Stage 3 time: {self.final_results['stage3_results']['processing_time']:.2f}s")
            
            print(f"\nStage 2 Reconstruction Error: {self.final_results['stage2_results']['performance_metrics']['mean_reconstruction_error']:.6f}")
            print(f"Stage 3 System Health: {self.final_results['stage3_results']['system_health']}")
            
            # Show final offsets
            print("\nFinal Wavelength Offsets:")
            for signal_id, offsets in self.final_results['final_wavelength_offsets'].items():
                if offsets:
                    print(f"  {signal_id}: {offsets[-1]:.6f} nm (Latest)")
                    print(f"           Offset Range: [{min(offsets):.6f}, {max(offsets):.6f}] nm")
                    
            # Show saved file info
            print(f"\nData files saved to {output_dir} directory:")
            if self.config['output'].get('save_wavelength_offsets', True):
                if 'offsets_csv_file' in locals() and offsets_csv_file:
                    print(f"  - Offsets CSV: {os.path.basename(offsets_csv_file)}")
                if 'offsets_json_file' in locals() and offsets_json_file:
                    print(f"  - Detailed Offsets JSON: {os.path.basename(offsets_json_file)}")
            
            if self.config['output'].get('save_reconstructed_waveforms', False):
                print(f"  - Reconstructed Waveform Data: reconstructed_waveforms_{timestamp}.npz")
                    
        except Exception as e:
            print(f"Output results failed: {e}")
            
    def _create_visualizations(self) -> None:
        """Create visualization charts (disabled)"""
        pass


def main():
    """Main function"""
    print("DG-SBL Complete System Startup")
    print("Author: DG-SBL Team")
    print("Version: 3.0")
    
    # Create system
    system = DGSBLCompleteSystem()
    
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
