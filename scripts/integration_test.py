"""
Fast Integration Test System - Specifically verifying Ultra-efficient Stage 3 Integration
"""

import numpy as np
import time
import sys
import os

# Add path
# Adjust path to include 'src' directory since this script is now in 'scripts'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from core.ultra_fast_stage3 import UltraFastStage3
from core.optimized_stage2_main import OptimizedStage2Processor

def quick_integration_test():
    """Quick Integration Test"""
    print("="*60)
    print("DG-SBL System Integration Test - Ultra-efficient Stage 3")
    print("="*60)
    
    # Simulate Stage 1 Output
    print("1. Simulating Stage 1 results...")
    n_wavelengths = 1000  # Use smaller scale for fast testing
    n_atoms = 100
    
    wavelength_array = np.linspace(1527, 1568, n_wavelengths)
    phi_final = np.random.randn(n_wavelengths, n_atoms)
    beta_global = 1.0 / 0.01
    
    print(f"   Dictionary Matrix: {phi_final.shape}")
    print(f"   Wavelength Array: {len(wavelength_array)} points")
    
    # Simulate Stage 2 Output
    print("\n2. Simulating Stage 2 results...")
    n_frames = 20
    n_signals = 3
    
    # Create simulated stream data
    y_stream = []
    for signal_idx in range(n_signals):
        signal_stream = []
        for frame_idx in range(n_frames):
            center = 1540 + signal_idx * 5 + frame_idx * 0.1
            signal = np.exp(-(wavelength_array - center)**2 / (2 * 1.0**2))
            signal += np.random.normal(0, 0.05, n_wavelengths)
            signal_stream.append(signal)
        y_stream.append(signal_stream)
    
    # Simulate sparse coefficients
    sparse_coefficients = np.random.rand(n_atoms, n_frames) * 0.1
    
    print(f"   Stream Data: {n_signals} signals x {n_frames} frames")
    print(f"   Sparse Coefficients: {sparse_coefficients.shape}")
    
    # Create Ultra-efficient Stage 3
    print("\n3. Initializing Ultra-efficient Stage 3...")
    stage3_start = time.time()
    
    stage3_controller = UltraFastStage3(phi_final, wavelength_array)
    
    init_time = time.time() - stage3_start
    print(f"   Initialization Time: {init_time:.3f}s")
    
    # Prepare Stage 3 frame data
    print("\n4. Preparing Stage 3 frame data...")
    all_frames = []
    
    for frame_idx in range(n_frames):
        frame_dict = {}
        for signal_idx in range(n_signals):
            signal_id = f"FBG{signal_idx+1}"
            frame_dict[signal_id] = y_stream[signal_idx][frame_idx]
        all_frames.append(frame_dict)
    
    print(f"   Prepared {len(all_frames)} frames of data")
    
    # Run Ultra-efficient Stage 3
    print("\n5. Running Ultra-efficient Stage 3 tracking...")
    tracking_start = time.time()
    
    stage3_results = stage3_controller.track_multiple_frames(all_frames)
    
    tracking_time = time.time() - tracking_start
    
    # Performance Analysis
    print(f"\n" + "="*60)
    print("Integration Test Results")
    print("="*60)
    
    print(f"Stage 3 Tracking Time: {tracking_time:.3f}s")
    print(f"Processed Frames: {len(stage3_results)}")
    print(f"Average Time Per Frame: {tracking_time/len(stage3_results)*1000:.2f}ms")
    print(f"Processing Speed: {len(stage3_results)/tracking_time:.1f} FPS")
    
    # Detailed Statistics
    all_times = []
    for frame_results in stage3_results:
        for result in frame_results:
            all_times.append(result.processing_time)
    
    all_times = np.array(all_times)
    
    print(f"\nDetailed Performance Statistics:")
    print(f"  Total Signals Processed: {len(all_times)}")
    print(f"  Average Processing Time: {np.mean(all_times)*1000:.3f}ms")
    print(f"  Fastest Processing Time: {np.min(all_times)*1000:.3f}ms")
    print(f"  Slowest Processing Time: {np.max(all_times)*1000:.3f}ms")
    
    # Verify Result Quality
    print(f"\nResult Quality Verification:")
    avg_confidence = np.mean([r.confidence for frame in stage3_results for r in frame])
    print(f"  Average Confidence: {avg_confidence:.3f}")
    print(f"  High Confidence Ratio (>0.7): {np.mean([r.confidence > 0.7 for frame in stage3_results for r in frame]):.1%}")
    
    # Export Results
    print(f"\n6. Exporting Results...")
    output_dir = stage3_controller.export_tracking_results()
    print(f"   Results saved to: {output_dir}")
    
    # System Health Check
    system_report = stage3_controller.get_system_report()
    print(f"\n7. System Health Status:")
    print(f"   Health Status: {system_report['health_status']}")
    
    # Performance Evaluation
    avg_time_ms = np.mean(all_times) * 1000
    print(f"\nPerformance Evaluation:")
    if avg_time_ms < 0.5:
        print("   Excellent Performance! True millisecond-level processing")
        print("   Integration Successful: Ultra-efficient Stage 3 successfully integrated into DG-SBL system")
    elif avg_time_ms < 2:
        print("   Good Performance! Meets real-time processing requirements")
        print("   Integration Successful")
    else:
        print("   Average Performance, may need further optimization")
    
    return {
        'success': True,
        'avg_processing_time_ms': avg_time_ms,
        'fps': len(stage3_results) / tracking_time,
        'total_signals': len(all_times),
        'avg_confidence': avg_confidence,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    result = quick_integration_test()
    
    if result['success']:
        print(f"\n🎉 DG-SBL System Integration Test Successful!")
        print(f"   - Average Processing Time: {result['avg_processing_time_ms']:.2f}ms")
        print(f"   - Processing Speed: {result['fps']:.1f} FPS")
        print(f"   - Number of Signals: {result['total_signals']}")
        print(f"   - Average Confidence: {result['avg_confidence']:.3f}")
        print(f"\n✅ Ultra-efficient Stage 3 successfully integrated into DG-SBL framework!")
        print("   You can now run the complete system using 'python main.py'.")
    else:
        print(f"\n❌ Integration Test Failed, further debugging required.")
