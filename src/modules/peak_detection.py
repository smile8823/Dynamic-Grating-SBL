"""
Peak Detection & Wavelength Extraction System
Stage3 Core Module: Handles peak detection for various signal forms, achieving sub-pixel accuracy

Core Features:
1. Multi-form peak detection (Gaussian, Flat-top, Multi-peak)
2. Sub-pixel accuracy optimization
3. Adaptive detection strategy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')


class PeakDetectionSystem:
    """
    Peak Detection System
    
    Handles peak detection for various signal forms, achieving sub-pixel accuracy
    """
    
    def __init__(self, min_peak_height: float = 0.1, min_peak_distance: int = 20) -> None:
        """
        Initialize peak detection system
        
        Args:
            min_peak_height: Minimum peak height (relative value)
            min_peak_distance: Minimum distance between peaks
        """
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        
        # Detection parameters
        self.detection_methods = {
            'standard': self._standard_peak_detection,
            'subpixel': self._subpixel_peak_detection,
            'flat_top': self._flat_top_peak_detection,
            'multi_peak': self._multi_peak_detection,
            'adaptive': self._adaptive_peak_detection
        }
        
        # Performance statistics
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_confidence': 0.0
        }
        
    def detect_peaks(self, y_signal: np.ndarray, wavelength_array: np.ndarray,
                    method: str = 'adaptive', **kwargs) -> Dict[str, Any]:
        """
        Detect peaks
        
        Args:
            y_signal: Input signal
            wavelength_array: Wavelength array
            method: Detection method
            **kwargs: Other parameters
            
        Returns:
            Detection result dictionary
        """
        if method not in self.detection_methods:
            method = 'adaptive'
            
        result = self.detection_methods[method](y_signal, wavelength_array, **kwargs)
        
        # Update statistics
        self.detection_stats['total_detections'] += 1
        if result['peak_count'] > 0:
            self.detection_stats['successful_detections'] += 1
            self.detection_stats['average_confidence'] = (
                (self.detection_stats['average_confidence'] * (self.detection_stats['successful_detections'] - 1) + 
                 np.mean(result['confidences'])) / self.detection_stats['successful_detections']
            )
            
        return result
        
    def _standard_peak_detection(self, y_signal: np.ndarray, wavelength_array: np.ndarray,
                               height_threshold: Optional[float] = None, distance_min: Optional[int] = None) -> Dict[str, Any]:
        """
        Standard peak detection - Suitable for single peak Gaussian FBG signals
        """
        # Set default parameters
        if height_threshold is None:
            height_threshold = np.max(y_signal) * self.min_peak_height
        if distance_min is None:
            distance_min = self.min_peak_distance
            
        # Use scipy's find_peaks
        peak_indices, properties = find_peaks(
            y_signal,
            height=height_threshold,
            distance=distance_min
        )
        
        if len(peak_indices) == 0:
            return {
                'peak_wavelengths': np.array([]),
                'peak_indices': np.array([], dtype=int),
                'peak_heights': np.array([]),
                'confidences': np.array([]),
                'peak_count': 0,
                'method': 'standard',
                'success': False
            }
            
        peak_wavelengths = wavelength_array[peak_indices]
        peak_heights = properties['peak_heights']
        
        # Optimize confidence calculation (vectorized operation)
        max_signal = np.max(y_signal)
        confidences = np.clip(peak_heights / max_signal, 0.0, 1.0)
            
        return {
            'peak_wavelengths': peak_wavelengths,
            'peak_indices': peak_indices,
            'peak_heights': peak_heights,
            'confidences': confidences,
            'peak_count': len(peak_wavelengths),
            'method': 'standard',
            'success': True
        }
        
    def _subpixel_peak_detection(self, y_signal: np.ndarray, wavelength_array: np.ndarray,
                                peak_indices: Optional[np.ndarray] = None, window_size: int = 5) -> Dict[str, Any]:
        """
        Subpixel accuracy peak detection - Improve accuracy through curve fitting
        """
        # Input validation
        if y_signal is None or len(y_signal) == 0:
            return self._create_empty_result("subpixel")
            
        if wavelength_array is None or len(wavelength_array) == 0:
            return self._create_empty_result("subpixel")
            
        if len(y_signal) != len(wavelength_array):
            print(f"Warning: Signal length {len(y_signal)} doesn't match wavelength array length {len(wavelength_array)}")
            min_len = min(len(y_signal), len(wavelength_array))
            if min_len == 0:
                return self._create_empty_result("subpixel")
            y_signal = y_signal[:min_len]
            wavelength_array = wavelength_array[:min_len]
            
        # If no peak positions provided, perform rough localization first
        if peak_indices is None or len(peak_indices) == 0:
            standard_result = self._standard_peak_detection(y_signal, wavelength_array)
            peak_indices = standard_result['peak_indices']
            
        if len(peak_indices) == 0:
            return standard_result
            
        # Validate peak indices
        peak_indices = np.array(peak_indices)
        peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(y_signal))]
        
        if len(peak_indices) == 0:
            return self._create_empty_result("subpixel")
            
        precise_wavelengths = []
        precise_heights = []
        confidences = []
        
        for peak_idx in peak_indices:
            # Safe boundary check
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(len(y_signal), peak_idx + window_size + 1)
            
            # Ensure window is valid
            if end_idx <= start_idx:
                print(f"Warning: Invalid window for peak at index {peak_idx}")
                continue
                
            # Extract data
            window_wavelengths = wavelength_array[start_idx:end_idx]
            window_signal = y_signal[start_idx:end_idx]
            
            # Check window data validity
            if len(window_wavelengths) < 3 or len(window_signal) < 3:
                print(f"Warning: Window too small for peak at index {peak_idx}")
                continue
                
            # Check data validity
            if not np.isfinite(window_wavelengths).all() or not np.isfinite(window_signal).all():
                print(f"Warning: Invalid values in window for peak at index {peak_idx}")
                continue
            
            try:
                # Gaussian fitting
                def gaussian(x, a, b, c, d):
                    return a * np.exp(-(x-b)**2/(2*c**2)) + d
                
                # Initial parameter estimation
                a_init = window_signal.max() - window_signal.min()
                b_init = window_wavelengths[np.argmax(window_signal)]
                c_init = (window_wavelengths[-1] - window_wavelengths[0]) / 4
                d_init = window_signal.min()
                
                popt, pcov = curve_fit(
                    gaussian, window_wavelengths, window_signal,
                    p0=[a_init, b_init, c_init, d_init],
                    maxfev=1000
                )
                
                # Calculate fit quality (vectorized operation)
                y_fit = gaussian(window_wavelengths, *popt)
                residuals = window_signal - y_fit
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((window_signal - np.mean(window_signal)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                precise_wavelengths.append(popt[1])  # Center wavelength b
                precise_heights.append(popt[0])     # Amplitude a
                confidences.append(max(0.0, min(1.0, r_squared)))
                
            except:
                # If fitting fails, use original position
                precise_wavelengths.append(wavelength_array[peak_idx])
                precise_heights.append(y_signal[peak_idx])
                confidences.append(0.5)  # Medium confidence
                
        return {
            'peak_wavelengths': np.array(precise_wavelengths),
            'peak_indices': peak_indices,
            'peak_heights': np.array(precise_heights),
            'confidences': np.array(confidences),
            'peak_count': len(precise_wavelengths),
            'method': 'subpixel',
            'success': True
        }
        
    def _flat_top_peak_detection(self, y_signal: np.ndarray, wavelength_array: np.ndarray,
                               flatness_threshold: float = 0.90, min_width: int = 3) -> Dict[str, Any]:
        """
        Flat top signal detection - Handle flat top phenomenon of FBG signals
        """
        signal_max = np.max(y_signal)
        threshold = signal_max * flatness_threshold
        
        # Mark positions above threshold
        high_regions = y_signal >= threshold
        
        # Find continuous regions
        peak_wavelengths = []
        peak_indices = []
        peak_heights = []
        confidences = []
        
        i = 0
        while i < len(y_signal):
            if high_regions[i]:
                # Find start and end of flat top region
                start_idx = i
                while i < len(y_signal) and high_regions[i]:
                    i += 1
                end_idx = i - 1
                
                region_width = end_idx - start_idx + 1
                
                if region_width >= min_width:
                    # Calculate center position of flat top region
                    region_signal = y_signal[start_idx:end_idx+1]
                    region_wavelengths = wavelength_array[start_idx:end_idx+1]
                    
                    # Use weighted average to calculate center position
                    center_wavelength = np.sum(region_wavelengths * region_signal) / np.sum(region_signal)
                    center_idx = start_idx + np.argmax(region_signal)
                    
                    peak_wavelengths.append(center_wavelength)
                    peak_indices.append(center_idx)
                    peak_heights.append(signal_max)
                    
                    # Confidence of flat top signal based on flatness
                    flatness = 1.0 - np.std(region_signal) / (np.mean(region_signal) + 1e-6)
                    confidences.append(min(1.0, flatness))
            else:
                i += 1
                
        return {
            'peak_wavelengths': np.array(peak_wavelengths),
            'peak_indices': np.array(peak_indices, dtype=int),
            'peak_heights': np.array(peak_heights),
            'confidences': np.array(confidences),
            'peak_count': len(peak_wavelengths),
            'method': 'flat_top',
            'success': len(peak_wavelengths) > 0
        }
        
    def _multi_peak_detection(self, y_signal: np.ndarray, wavelength_array: np.ndarray,
                            max_peaks: int = 5, min_peak_height_ratio: float = 0.1,
                            min_peak_distance_ratio: float = 0.05) -> Dict[str, Any]:
        """
        Multi-peak signal detection - Handle complex multi-FBG sensor signals
        """
        signal_length = len(y_signal)
        signal_max = np.max(y_signal)
        
        # Set parameters
        height_threshold = signal_max * min_peak_height_ratio
        distance_min = int(signal_length * min_peak_distance_ratio)
        
        # Find all peaks
        peak_indices, properties = find_peaks(
            y_signal,
            height=height_threshold,
            distance=distance_min
        )
        
        # Limit number of peaks
        if len(peak_indices) > max_peaks:
            # Sort by height, take top max_peaks
            sorted_indices = np.argsort(properties['peak_heights'])[::-1]
            top_indices = sorted_indices[:max_peaks]
            peak_indices = peak_indices[top_indices]
            properties['peak_heights'] = properties['peak_heights'][top_indices]
            
        # Subpixel accuracy optimization
        subpixel_result = self._subpixel_peak_detection(y_signal, wavelength_array, peak_indices)
        
        # Optimize SNR calculation (vectorized operation)
        snr_values = np.array([self._calculate_snr(y_signal, idx) for idx in peak_indices])
            
        # Optimize confidence update (vectorized operation)
        snr_weights = np.clip(snr_values / 10.0, 0.0, 1.0)  # SNR > 10 is considered good
        enhanced_confidences = (subpixel_result['confidences'] * 0.7 + snr_weights * 0.3)
            
        # Optimize peak info construction (vectorized operation)
        relative_heights = subpixel_result['peak_heights'] / signal_max
        all_peaks = [
            {
                'index': idx,
                'wavelength': wavelength,
                'height': height,
                'relative_height': rel_height,
                'snr': snr_val,
                'confidence': conf
            }
            for idx, wavelength, height, rel_height, snr_val, conf in zip(
                peak_indices, subpixel_result['peak_wavelengths'], 
                subpixel_result['peak_heights'], relative_heights, 
                snr_values, enhanced_confidences
            )
        ]
            
        # Sort by wavelength
        all_peaks.sort(key=lambda x: x['wavelength'])
        
        # Optimize array extraction (vectorized operation)
        sorted_wavelengths = np.array([p['wavelength'] for p in all_peaks], dtype=float)
        sorted_heights = np.array([p['height'] for p in all_peaks], dtype=float)
        sorted_confidences = np.array([p['confidence'] for p in all_peaks], dtype=float)
        
        return {
            'peak_wavelengths': sorted_wavelengths,
            'peak_indices': np.array([p['index'] for p in all_peaks], dtype=int),
            'peak_heights': sorted_heights,
            'confidences': sorted_confidences,
            'peak_count': len(all_peaks),
            'method': 'multi_peak',
            'success': len(all_peaks) > 0,
            'detailed_peaks': all_peaks
        }
        
    def _adaptive_peak_detection(self, y_signal: np.ndarray, wavelength_array: np.ndarray,
                               expected_peaks: int = 3, flat_top_threshold: float = 0.90) -> Dict[str, Any]:
        """
        Adaptive peak detection - Automatically select best detection method based on signal characteristics
        """
        # Analyze signal characteristics
        signal_characteristics = self._analyze_signal_characteristics(y_signal, flat_top_threshold=flat_top_threshold)

        # Select detection method based on signal characteristics
        if signal_characteristics['has_flat_top']:
            # Flat top signal detected
            result = self._flat_top_peak_detection(y_signal, wavelength_array, flatness_threshold=flat_top_threshold)

        elif signal_characteristics['peak_count'] > expected_peaks:
            # Multi-peak signal detected
            result = self._multi_peak_detection(y_signal, wavelength_array, max_peaks=expected_peaks)

        else:
            # Use standard detection
            result = self._standard_peak_detection(y_signal, wavelength_array)

            # Subpixel accuracy optimization
            if result['success'] and len(result['peak_wavelengths']) > 0:
                subpixel_result = self._subpixel_peak_detection(
                    y_signal, wavelength_array, result['peak_indices']
                )
                # Merge results, use subpixel accuracy
                result.update(subpixel_result)
                result['original_method'] = 'standard'
                result['method'] = 'standard_enhanced'

        # Add signal characteristics info
        result['signal_characteristics'] = signal_characteristics

        return result
        
    def _create_empty_result(self, method: str) -> Dict[str, Any]:
        """Create empty result dictionary"""
        return {
            'peak_wavelengths': np.array([]),
            'peak_indices': np.array([], dtype=int),
            'peak_heights': np.array([]),
            'confidences': np.array([]),
            'peak_count': 0,
            'method': method,
            'success': False,
            'signal_characteristics': {}
        }
        
    def _analyze_signal_characteristics(self, y_signal: np.ndarray, flat_top_threshold: float = 0.90) -> Dict[str, Any]:
        """Analyze signal characteristics"""
        signal_max = np.max(y_signal)
        signal_mean = np.mean(y_signal)
        
        # Check for flat top
        high_threshold = signal_max * flat_top_threshold
        flat_top_ratio = np.sum(y_signal >= high_threshold) / len(y_signal)
        has_flat_top = flat_top_ratio > 0.01  # More than 1% of area is flat top
        
        # Detect number of peaks
        peak_count_estimate = len(find_peaks(y_signal, height=signal_max*0.1)[0])
        
        # Calculate SNR
        noise_level = np.std(y_signal[y_signal < signal_max*0.2])
        snr = signal_max / noise_level if noise_level > 0 else float('inf')
        
        # Optimize symmetry calculation (vectorized operation)
        signal_center = len(y_signal) // 2
        left_half = y_signal[:signal_center]
        right_half = y_signal[signal_center:]
        
        # Calculate similarity between left and right halves
        if len(left_half) == len(right_half):
            symmetry = 1.0 - np.mean(np.abs(left_half - right_half[::-1])) / signal_max
        else:
            symmetry = 0.5
            
        return {
            'has_flat_top': has_flat_top,
            'peak_count': peak_count_estimate,
            'snr': snr,
            'flat_top_ratio': flat_top_ratio,
            'signal_max': signal_max,
            'signal_mean': signal_mean,
            'noise_level': noise_level,
            'symmetry': symmetry,
            'signal_range': signal_max - signal_mean
        }
        
    def _calculate_snr(self, signal: np.ndarray, peak_idx: int, window_size: int = 20) -> float:
        """Calculate SNR (optimized version)"""
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(signal), peak_idx + window_size + 1)
        
        peak_height = signal[peak_idx]
        
        # Optimize noise level calculation (vectorized operation)
        window_signal = signal[start_idx:end_idx]
        noise_mask = window_signal < peak_height * 0.5
        
        if np.any(noise_mask):
            noise_level = np.std(window_signal[noise_mask])
        else:
            noise_level = np.std(window_signal)
        
        return peak_height / noise_level if noise_level > 0 else float('inf')
        
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        success_rate = (self.detection_stats['successful_detections'] / 
                       self.detection_stats['total_detections'] 
                       if self.detection_stats['total_detections'] > 0 else 0.0)
                       
        return {
            'success_rate': success_rate,
            'total_detections': self.detection_stats['total_detections'],
            'successful_detections': self.detection_stats['successful_detections'],
            'average_confidence': self.detection_stats['average_confidence']
        }
        
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_confidence': 0.0
        }


def test_peak_detection() -> None:
    """Test peak detection function"""
    print("=== Test Peak Detection System ===")
    
    # Create detection system
    detector = PeakDetectionSystem()
    
    # Create test signal
    wavelength_range = np.linspace(1520, 1570, 1000)
    
    # Test 1: Single Gaussian peak
    print("\nTest 1: Single Gaussian peak")
    y1 = np.exp(-(wavelength_range - 1540)**2 / (2 * 1.0**2))
    result1 = detector.detect_peaks(y1, wavelength_range, method='adaptive')
    print(f"Detection result: {result1['peak_count']} peaks")
    if result1['success']:
        print(f"Peak positions: {result1['peak_wavelengths']}")
        print(f"Confidences: {result1['confidences']}")
        print(f"Method used: {result1['method']}")
    
    # Test 2: Flat top signal
    print("\nTest 2: Flat top signal")
    y2 = np.ones_like(wavelength_range)
    y2[(wavelength_range >= 1539) & (wavelength_range <= 1541)] = 1.0
    y2 = y2 + np.random.normal(0, 0.05, len(wavelength_range))
    result2 = detector.detect_peaks(y2, wavelength_range, method='adaptive')
    print(f"Detection result: {result2['peak_count']} peaks")
    if result2['success']:
        print(f"Peak positions: {result2['peak_wavelengths']}")
        print(f"Method used: {result2['method']}")
    
    # Test 3: Multi-peak signal
    print("\nTest 3: Multi-peak signal")
    y3 = (np.exp(-(wavelength_range - 1530)**2 / (2 * 1.0**2)) +
          np.exp(-(wavelength_range - 1540)**2 / (2 * 1.2**2)) +
          np.exp(-(wavelength_range - 1550)**2 / (2 * 0.8**2)))
    y3 = y3 + np.random.normal(0, 0.02, len(wavelength_range))
    result3 = detector.detect_peaks(y3, wavelength_range, method='adaptive')
    print(f"Detection result: {result3['peak_count']} peaks")
    if result3['success']:
        print(f"Peak positions: {result3['peak_wavelengths']}")
        print(f"Confidences: {result3['confidences']}")
        print(f"Method used: {result3['method']}")
        if 'detailed_peaks' in result3:
            for peak in result3['detailed_peaks']:
                print(f"  Peak details: λ={peak['wavelength']:.2f}nm, "
                      f"Height={peak['height']:.3f}, SNR={peak['snr']:.2f}")
    
    # Output statistics
    stats = detector.get_detection_stats()
    print(f"\nDetection statistics:")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")


if __name__ == "__main__":
    test_peak_detection()