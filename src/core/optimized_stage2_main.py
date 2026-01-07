#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Stage2 Main Program
Using Optimized PyTorch SBL Implementation
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import logging

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from modules.data_reader import read_csv_data
from core.optimized_pytorch_sbl import create_optimized_sbl, OptimizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedStage2Processor:
    """Optimized Stage2 Processor"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.sbl = None
        self.dictionary = None
        
    def load_stage1_results(self, stage1_dir: Path) -> tuple:
        """Load Stage1 results"""
        logger.info(f"Loading Stage1 results: {stage1_dir}")
        
        # Load dictionary
        dict_path = stage1_dir / 'dictionary.npz'
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        
        dict_data = np.load(dict_path)
        
        # Try different key names
        if 'phi_final' in dict_data:
            dictionary = dict_data['phi_final']
        elif 'dictionary' in dict_data:
            dictionary = dict_data['dictionary']
        else:
            # Use the first available key
            first_key = list(dict_data.keys())[0]
            dictionary = dict_data[first_key]
            logger.warning(f"Using key '{first_key}' as dictionary")
        
        logger.info(f"Dictionary shape: {dictionary.shape}")
        
        # Load reconstructed waveforms
        waveform_path = stage1_dir / 'reconstructed_waveforms.npz'
        if waveform_path.exists():
            waveform_data = np.load(waveform_path)
            
            if 'Y_reconstructed' in waveform_data:
                reconstructed_waveforms = waveform_data['Y_reconstructed']
            elif 'waveforms' in waveform_data:
                reconstructed_waveforms = waveform_data['waveforms']
            else:
                first_key = list(waveform_data.keys())[0]
                reconstructed_waveforms = waveform_data[first_key]
                logger.warning(f"Using key '{first_key}' as reconstructed waveforms")
            
            logger.info(f"Reconstructed waveforms shape: {reconstructed_waveforms.shape}")
        else:
            reconstructed_waveforms = None
            logger.warning("Reconstructed waveforms file not found")
        
        return dictionary, reconstructed_waveforms
    
    def load_test_data(self, data_path: str) -> tuple:
        """Load test data"""
        logger.info(f"Loading test data: {data_path}")
        
        wavelengths, signal_data = read_csv_data(data_path)
        
        logger.info(f"Test data shape: {signal_data.shape}")
        logger.info(f"Wavelength range: {wavelengths[0]:.4f} - {wavelengths[-1]:.4f} nm")
        
        return wavelengths, signal_data
    
    def initialize_optimized_sbl(self, dictionary: np.ndarray, mode: str = 'fast') -> None:
        """Initialize optimized SBL"""
        logger.info(f"Initializing optimized SBL, mode: {mode}")
        
        self.dictionary = dictionary
        
        if mode == 'fast':
            # Fast mode: maximum performance
            self.sbl = create_optimized_sbl(
                dictionary=dictionary,
                fast_mode=True,
                device=self.device
            )
        elif mode == 'balanced':
            # Balanced mode: balance between performance and accuracy
            config = OptimizationConfig(
                max_iterations=4,
                tolerance=2e-2,
                k_sparsity=4,
                use_fast_convergence=True,
                use_precomputed_matrices=True,
                batch_inference=True,
                memory_efficient=True
            )
            from optimized_pytorch_sbl import OptimizedPyTorchSBL
            self.sbl = OptimizedPyTorchSBL(dictionary, config, self.device)
        else:
            # Accurate mode: higher accuracy
            self.sbl = create_optimized_sbl(
                dictionary=dictionary,
                fast_mode=False,
                device=self.device
            )
        
        logger.info("Optimized SBL initialization complete")
    
    def process_signal_stream(self, 
                             signal_data: np.ndarray,
                             batch_size: int = 50,
                             max_frames: int = None) -> dict:
        """Process signal stream"""
        if self.sbl is None:
            raise RuntimeError("SBL not initialized")
        
        # Limit number of frames (for testing)
        if max_frames is not None:
            signal_data = signal_data[:, :max_frames]
        
        num_frames = signal_data.shape[1]
        logger.info(f"Starting to process signal stream: {signal_data.shape}")
        logger.info(f"Batch size: {batch_size}, Total frames: {num_frames}")
        
        # Transpose data to (num_frames, M) format
        signal_stream = signal_data.T
        
        # Run optimized SBL
        start_time = time.time()
        
        results = self.sbl.track_signal_stream_optimized(
            signal_stream=signal_stream,
            batch_size=batch_size
        )
        
        total_time = time.time() - start_time
        
        logger.info(f"Signal stream processing complete")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Frame rate: {results['frames_per_second']:.2f} FPS")
        logger.info(f"Average per frame: {total_time/num_frames*1000:.2f}ms")
        
        return results
    
    def run_performance_test(self, 
                           stage1_dir: Path,
                           data_path: str,
                           test_configs: list) -> dict:
        """Run performance test"""
        logger.info("Starting optimized Stage2 performance test")
        
        # Load data
        dictionary, _ = self.load_stage1_results(stage1_dir)
        wavelengths, signal_data = self.load_test_data(data_path)
        
        test_results = {}
        
        for config in test_configs:
            mode = config['mode']
            frames = config['frames']
            batch_size = config['batch_size']
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Test config: {mode} mode, {frames} frames, batch size {batch_size}")
            logger.info(f"{'='*50}")
            
            try:
                # Initialize SBL
                self.initialize_optimized_sbl(dictionary, mode)
                
                # Process signal
                results = self.process_signal_stream(
                    signal_data=signal_data,
                    batch_size=batch_size,
                    max_frames=frames
                )
                
                # Record results
                test_results[f"{mode}_{frames}frames_batch{batch_size}"] = {
                    'mode': mode,
                    'frames': frames,
                    'batch_size': batch_size,
                    'fps': results['frames_per_second'],
                    'avg_time_per_frame_ms': results['processing_time'] / frames * 1000,
                    'total_time': results['processing_time'],
                    'avg_iterations': results.get('avg_iterations', 0),
                    'convergence_rate': results.get('convergence_rate', 0),
                    'success': True
                }
                
                # Clean up memory
                self.sbl.cleanup()
                
                logger.info(f"✅ {mode} mode test complete: {results['frames_per_second']:.2f} FPS")
                
            except Exception as e:
                logger.error(f"❌ {mode} mode test failed: {e}")
                test_results[f"{mode}_{frames}frames_batch{batch_size}"] = {
                    'mode': mode,
                    'frames': frames,
                    'batch_size': batch_size,
                    'error': str(e),
                    'success': False
                }
        
        return test_results

def main():
    """Main function"""
    logger.info("="*60)
    logger.info("Optimized Stage2 Performance Test")
    logger.info("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    try:
        # Find Stage1 results
        output_dir = Path("../output")
        stage1_dirs = [d for d in output_dir.glob('stage1_results_*') if d.is_dir()]
        
        if not stage1_dirs:
            raise FileNotFoundError("Stage1 results directory not found")
        
        stage1_dir = max(stage1_dirs, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using Stage1 results: {stage1_dir}")
        
        # Create processor
        processor = OptimizedStage2Processor(device='cuda')
        
        # Test configuration
        test_configs = [
            {'mode': 'fast', 'frames': 10, 'batch_size': 10},
            {'mode': 'fast', 'frames': 20, 'batch_size': 20},
            {'mode': 'fast', 'frames': 50, 'batch_size': 25},
            {'mode': 'balanced', 'frames': 20, 'batch_size': 20},
            {'mode': 'balanced', 'frames': 50, 'batch_size': 25},
        ]
        
        # Run tests
        results = processor.run_performance_test(
            stage1_dir=stage1_dir,
            data_path="../data/10s_1.csv",
            test_configs=test_configs
        )
        
        # Show results summary
        print(f"\n{'='*60}")
        print("Optimization Summary")
        print(f"{'='*60}")
        
        successful_tests = [r for r in results.values() if r.get('success', False)]
        
        if successful_tests:
            # Sort by FPS
            sorted_results = sorted(successful_tests, key=lambda x: x['fps'], reverse=True)
            
            print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
            print(f"\n🏆 Performance Ranking:")
            
            for i, result in enumerate(sorted_results[:3]):
                print(f"   {i+1}. {result['mode']} mode ({result['frames']} frames): "
                      f"{result['fps']:.2f} FPS, {result['avg_time_per_frame_ms']:.1f}ms/frame")
            
            # Best performance
            best_result = sorted_results[0]
            print(f"\n🚀 Best Performance:")
            print(f"   Mode: {best_result['mode']}")
            print(f"   Frame rate: {best_result['fps']:.2f} FPS")
            print(f"   Time per frame: {best_result['avg_time_per_frame_ms']:.1f}ms")
            print(f"   Avg iterations: {best_result['avg_iterations']:.1f}")
            
            # Comparison with original version
            original_fps = 4.2  # Based on previous test results
            improvement = best_result['fps'] / original_fps
            print(f"   Improvement over original: {improvement:.1f}x")
            
            # Real-time processing assessment
            if best_result['fps'] >= 30:
                print(f"   🎯 Real-time capability: Excellent (>30 FPS)")
            elif best_result['fps'] >= 15:
                print(f"   🎯 Real-time capability: Good (>15 FPS)")
            elif best_result['fps'] >= 10:
                print(f"   🎯 Real-time capability: Acceptable (>10 FPS)")
            else:
                print(f"   🎯 Real-time capability: Needs optimization (<10 FPS)")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"optimized_stage2_results_{timestamp}.json"
        
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()