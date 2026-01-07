"""
Independent Signal Tracker
Stage3 Core Module: Independent tracking unit for each FBG sensor, supporting asynchronous switching

Core Functions:
1. Single signal independent tracking
2. Local signal extraction and atom matching
3. Tracking quality evaluation and atom set update
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

from direction_prediction import DirectionPredictionModel
from peak_detection import PeakDetectionSystem
from atom_set_manager import AtomSet


class SignalStatus(Enum):
    """Signal status enumeration"""
    NORMAL = "normal"
    SWITCHING = "switching"
    LOST = "lost"
    RECOVERING = "recovering"


@dataclass
class TrackingResult:
    """Tracking result data class"""
    signal_id: str
    wavelength_offset: float
    quality_score: float
    confidence: float
    peak_wavelength: float
    peak_amplitude: float
    timestamp: float
    status: str


class IndependentSignalTracker:
    """
    Independent Signal Tracker
    
    One tracker instance per FBG sensor, implementing independent tracking and switching
    """
    
    def __init__(self, signal_id: str, wavelength_range: Tuple[float, float],
                 phi_global: np.ndarray, wavelength_array: np.ndarray,
                 direction_model: DirectionPredictionModel = None):
        """
        Initialize independent signal tracker
        
        Args:
            signal_id: Signal identifier
            wavelength_range: Wavelength range (min, max)
            phi_global: Global dictionary matrix
            wavelength_array: Wavelength array
            direction_model: Direction prediction model
        """
        self.signal_id = signal_id
        self.wavelength_range = wavelength_range
        self.phi_global = phi_global
        self.wavelength_array = wavelength_array
        
        # Direction prediction model
        self.direction_model = direction_model or DirectionPredictionModel()
        
        # Peak detection system
        self.peak_detector = PeakDetectionSystem()
        
        # Active atom set
        self.active_atom_set: Optional[AtomSet] = None
        
        # Candidate atom sets
        self.candidate_atom_sets: List[AtomSet] = []
        
        # Tracking status
        self.status = SignalStatus.NORMAL
        self.health_status = "good"
        
        # Tracking history
        self.offset_history = []
        self.quality_history = []
        self.confidence_history = []
        
        # Performance parameters
        self.quality_threshold = 0.7
        self.confidence_threshold = 0.6
        self.max_candidate_sets = 3
        
        # Statistics
        self.tracking_stats = {
            'total_frames': 0,
            'successful_tracks': 0,
            'switching_events': 0,
            'average_quality': 0.0,
            'average_confidence': 0.0
        }
        
    def initialize_tracking(self, initial_frame: np.ndarray, 
                          expected_peak_count: int = 1) -> bool:
        """
        Initialize tracking
        
        Args:
            initial_frame: Initial frame signal
            expected_peak_count: Expected peak count
            
        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Extract local signal
            local_signal = self._extract_local_signal(initial_frame)
            if local_signal is None or len(local_signal) == 0:
                return False
                
            # Peak detection
            peak_result = self.peak_detector.detect_peaks(local_signal, self.wavelength_array)
            if not peak_result or not peak_result.get('success', False) or peak_result.get('peak_count', 0) == 0:
                return False
                
            # Get main peak wavelength
            peak_wavelengths = peak_result.get('peak_wavelengths', [])
            if len(peak_wavelengths) == 0:
                return False
            main_peak_wavelength = peak_wavelengths[0]
                
            # Create initial atom set
            from atom_set_manager import AtomSetStatus
            import time
            
            # Find atom index closest to main peak
            atom_idx = np.argmin(np.abs(self.wavelength_array - main_peak_wavelength))
            
            self.active_atom_set = AtomSet(
                id=f"{self.signal_id}_initial",
                atom_indices=np.array([atom_idx]),
                reference_wavelengths=np.array([main_peak_wavelength]),
                reference_offsets=np.array([main_peak_wavelength - np.mean(self.wavelength_range)]),
                creation_time=time.time(),
                quality_score=0.8,
                status=AtomSetStatus.ACTIVE
            )
            
            # Update direction prediction model
            import time
            self.direction_model.update_history(main_peak_wavelength, time.time())
            
            # Record initial result
            self._record_tracking_result(main_peak_wavelength, 0.8, 0.9, peak_result)
            
            self.status = SignalStatus.NORMAL
            self.health_status = "good"
            
            return True
            
        except Exception as e:
            print(f"Warning: Tracker initialization failed {self.signal_id}: {e}")
            self.status = SignalStatus.LOST
            self.health_status = "poor"
            return False
            
    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> Optional[TrackingResult]:
        """
        Process new frame
        
        Args:
            frame: Input frame
            frame_index: Frame index
            
        Returns:
            TrackingResult: Tracking result, None if failed
        """
        start_time = time.time()
        try:
            self.tracking_stats['total_frames'] += 1
            
            # Extract local signal
            local_signal = self._extract_local_signal(frame)
            if local_signal is None:
                return None
                
            # Peak detection
            peak_result = self.peak_detector.detect_peaks(local_signal, self.wavelength_array)
            if not peak_result or not peak_result.get('success', False) or peak_result.get('peak_count', 0) == 0:
                return None
                
            # Calculate offset
            peak_wavelengths = peak_result.get('peak_wavelengths', [])
            if len(peak_wavelengths) == 0:
                return None
            main_peak_wavelength = peak_wavelengths[0]  # Use the wavelength of the first peak
            offset = self._calculate_wavelength_offset(main_peak_wavelength)
            
            # Evaluate quality
            quality = self._evaluate_tracking_quality(peak_result, local_signal)
            confidence = self._calculate_confidence(quality, offset)
            
            # Check if atom set switch is needed (increase threshold to reduce switching)
            if quality < self.quality_threshold:
                print(f"Signal {self.signal_id}: Quality {quality:.3f} < threshold {self.quality_threshold}, attempting switch")
                success = self._attempt_atom_switch(local_signal, peak_result)
                if not success:
                    self.status = SignalStatus.LOST
                    return None
            else:
                # Quality is good enough, skip atom switching
                pass
                    
            # Update tracking history
            self._record_tracking_result(main_peak_wavelength, quality, confidence, peak_result)
            
            # Update direction prediction
            self.direction_model.update_prediction(offset, confidence)
            
            # Create tracking result
            result = TrackingResult(
                signal_id=self.signal_id,
                wavelength_offset=offset,
                quality_score=quality,
                confidence=confidence,
                peak_wavelength=main_peak_wavelength,
                peak_amplitude=peak_result.get('peak_heights', [0.0])[0] if peak_result.get('peak_heights') else 0.0,
                timestamp=time.time(),
                status=self.status.value
            )
            
            self.tracking_stats['successful_tracks'] += 1
            self.status = SignalStatus.NORMAL
            
            # Performance monitoring
            processing_time = time.time() - start_time
            if processing_time > 5.0:  # Warn for frames taking longer than 5 seconds
                print(f"WARNING: Signal {self.signal_id} frame {frame_index} took {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"Warning: Frame processing failed {self.signal_id}: {e}")
            self.status = SignalStatus.LOST
            return None
            
    def _extract_local_signal(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract local signal"""
        try:
            min_idx = np.argmin(np.abs(self.wavelength_array - self.wavelength_range[0]))
            max_idx = np.argmin(np.abs(self.wavelength_array - self.wavelength_range[1]))
            
            if min_idx >= max_idx or max_idx >= len(frame):
                return None
                
            return frame[min_idx:max_idx]
            
        except Exception:
            return None
            
    def _calculate_wavelength_offset(self, peak_wavelength: float) -> float:
        """Calculate wavelength offset"""
        return peak_wavelength - np.mean(self.wavelength_range)
        
    def _evaluate_tracking_quality(self, peak_result, local_signal: np.ndarray) -> float:
        """Evaluate tracking quality"""
        try:
            # Simplified quality evaluation: based on peak amplitude and SNR
            if hasattr(peak_result, 'amplitude'):
                amplitude_score = min(1.0, peak_result.amplitude / 0.5)
            else:
                amplitude_score = 0.5
                
            # SNR evaluation
            signal_power = np.mean(local_signal ** 2)
            noise_power = np.var(local_signal - np.mean(local_signal))
            snr_score = signal_power / (signal_power + noise_power) if noise_power > 0 else 0.5
            
            return (amplitude_score + snr_score) / 2.0
            
        except Exception:
            return 0.5
            
    def _calculate_confidence(self, quality: float, offset: float) -> float:
        """Calculate confidence"""
        try:
            # Confidence calculation based on quality and offset stability
            quality_factor = quality
            
            # Offset stability factor
            if len(self.offset_history) > 0:
                offset_stability = 1.0 / (1.0 + abs(offset - np.mean(self.offset_history[-5:])))
            else:
                offset_stability = 0.8
                
            return (quality_factor * 0.7 + offset_stability * 0.3)
            
        except Exception:
            return 0.5
            
    def _attempt_atom_switch(self, local_signal: np.ndarray, peak_result) -> bool:
        """Attempt atom set switch"""
        try:
            # Get main peak wavelength
            peak_wavelengths = peak_result.get('peak_wavelengths', [])
            if len(peak_wavelengths) == 0:
                return False
            main_peak_wavelength = peak_wavelengths[0]
            
            # Create new atom set
            from atom_set_manager import AtomSetStatus
            import time
            
            # Find atom index closest to main peak
            atom_idx = np.argmin(np.abs(self.wavelength_array - main_peak_wavelength))
            
            new_atom_set = AtomSet(
                id=f"{self.signal_id}_{int(time.time())}",
                atom_indices=np.array([atom_idx]),
                reference_wavelengths=np.array([main_peak_wavelength]),
                reference_offsets=np.array([main_peak_wavelength - np.mean(self.wavelength_range)]),
                creation_time=time.time(),
                quality_score=0.0,
                status=AtomSetStatus.CANDIDATE
            )
            
            # Evaluate quality of new atom set
            quality = self._evaluate_tracking_quality(peak_result, local_signal)
            new_atom_set.quality_score = quality
            
            if quality > self.quality_threshold * 0.8:  # Slightly looser threshold
                self.active_atom_set = new_atom_set
                self.status = SignalStatus.SWITCHING
                self.tracking_stats['switching_events'] += 1
                return True
                
            return False
            
        except Exception:
            return False
            
    def _record_tracking_result(self, wavelength: float, quality: float, 
                              confidence: float, peak_result) -> None:
        """Record tracking result"""
        offset = self._calculate_wavelength_offset(wavelength)
        
        self.offset_history.append(offset)
        self.quality_history.append(quality)
        self.confidence_history.append(confidence)
        
        # Maintain history length
        max_history = 100
        if len(self.offset_history) > max_history:
            self.offset_history = self.offset_history[-max_history:]
            self.quality_history = self.quality_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]
            
        # Update statistics
        if self.quality_history:
            self.tracking_stats['average_quality'] = np.mean(self.quality_history)
            self.tracking_stats['average_confidence'] = np.mean(self.confidence_history)
            
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        stats = self.tracking_stats.copy()
        stats.update({
            'signal_id': self.signal_id,
            'status': self.status.value,
            'health_status': self.health_status,
            'current_offset': self.offset_history[-1] if self.offset_history else 0.0,
            'history_length': len(self.offset_history),
            'success_rate': (self.tracking_stats['successful_tracks'] / 
                           max(1, self.tracking_stats['total_frames']))
        })
        return stats
        
    def reset(self) -> None:
        """Reset tracker"""
        self.active_atom_set = None
        self.candidate_atom_sets = []
        self.status = SignalStatus.NORMAL
        self.health_status = "good"
        self.offset_history = []
        self.quality_history = []
        self.confidence_history = []
        self.tracking_stats = {
            'total_frames': 0,
            'successful_tracks': 0,
            'switching_events': 0,
            'average_quality': 0.0,
            'average_confidence': 0.0
        }
        
    def __del__(self):
        """Destructor"""
        try:
            self.reset()
        except Exception:
            pass