"""
Signal Overlap Detection & Separation
Stage3 Core Module: Solves signal overlap problems based on direction prediction, enabling accurate separation of multiple signals

Core Features:
1. Signal overlap detection
2. Direction prediction guided separation
3. Complex overlap scenario handling
4. Separation quality evaluation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.signal import find_peaks, correlate
from scipy.optimize import minimize, curve_fit
from scipy.stats import gaussian_kde
import time

warnings.filterwarnings('ignore')


class OverlapScenario(Enum):
    """Overlap Scenario Enum"""
    NO_OVERLAP = "no_overlap"
    TEMPERATURE_DRIFT = "temperature_drift"  # Same direction drift
    STRAIN_INDUCED = "strain_induced"      # Opposite direction drift
    MIXED_EFFECT = "mixed_effect"           # Mixed effect
    SIGNAL_CROSSING = "signal_crossing"     # Signal crossing


@dataclass
class OverlapDetection:
    """Overlap Detection Result"""
    scenario: OverlapScenario
    overlapping_pairs: List[Tuple[str, str]]
    overlap_severity: float
    confidence: float
    predicted_separations: Dict[Tuple[str, str], float]


@dataclass
class SeparationResult:
    """Separation Result"""
    separated_signals: Dict[str, np.ndarray]
    separation_weights: Dict[Tuple[str, str], float]
    quality_scores: Dict[str, float]
    overall_quality: float
    method_used: str
    processing_time: float


class SignalSeparationEngine:
    """
    Signal Separation Engine
    
    Solves signal overlap problems based on direction prediction
    """
    
    def __init__(self, min_signal_distance: float = 0.5):
        """
        Initialize signal separation engine
        
        Args:
            min_signal_distance: Minimum signal distance (nm)
        """
        self.min_signal_distance = min_signal_distance
        
        # Separation method configuration
        self.separation_methods = {
            'direction_weighted': self._direction_weighted_separation,
            'temporal_separation': self._temporal_separation,
            'frequency_domain': self._frequency_domain_separation,
            'iterative_separation': self._iterative_separation,
            'hybrid_approach': self._hybrid_separation
        }
        
        # Performance statistics
        self.separation_stats = {
            'total_separations': 0,
            'successful_separations': 0,
            'average_quality': 0.0,
            'scenario_distribution': {scenario: 0 for scenario in OverlapScenario},
            'method_usage': {}
        }
        
    def detect_signal_overlapping(self, signal_predictions: Dict[str, Dict],
                                 current_signal: np.ndarray = None) -> OverlapDetection:
        """
        Detect signal overlap
        
        Args:
            signal_predictions: Prediction info for each signal
            current_signal: Current signal (optional)
            
        Returns:
            Overlap detection result
        """
        overlapping_pairs = []
        overlap_severities = []
        
        signal_ids = list(signal_predictions.keys())
        
        for i, signal_i in enumerate(signal_ids):
            for j, signal_j in enumerate(signal_ids[i+1:], i+1):
                # Calculate predicted distance
                pred_i = signal_predictions[signal_i].get('predicted_position', 0.0)
                pred_j = signal_predictions[signal_j].get('predicted_position', 0.0)
                
                predicted_distance = abs(pred_i - pred_j)
                
                # Check for overlap
                if predicted_distance < self.min_signal_distance:
                    overlapping_pairs.append((signal_i, signal_j))
                    severity = 1.0 - (predicted_distance / self.min_signal_distance)
                    overlap_severities.append(severity)
                    
        # Determine overlap scenario
        scenario = self._classify_overlap_scenario(signal_predictions, overlapping_pairs)
        
        # Calculate overall severity
        overall_severity = np.mean(overlap_severities) if overlap_severities else 0.0
        
        # Calculate confidence
        confidence = self._calculate_overlap_confidence(
            signal_predictions, overlapping_pairs, scenario
        )
        
        # Predict separation distance
        predicted_separations = {}
        for signal_i, signal_j in overlapping_pairs:
            predicted_separations[(signal_i, signal_j)] = self._predict_separation_distance(
                signal_predictions[signal_i], signal_predictions[signal_j]
            )
            
        return OverlapDetection(
            scenario=scenario,
            overlapping_pairs=overlapping_pairs,
            overlap_severity=overall_severity,
            confidence=confidence,
            predicted_separations=predicted_separations
        )
        
    def _classify_overlap_scenario(self, signal_predictions: Dict[str, Dict],
                                 overlapping_pairs: List[Tuple[str, str]]) -> OverlapScenario:
        """Classify overlap scenario"""
        if not overlapping_pairs:
            return OverlapScenario.NO_OVERLAP
            
        # Analyze direction of change
        directions = {}
        for signal_id, prediction in signal_predictions.items():
            velocity = prediction.get('predicted_velocity', 0.0)
            directions[signal_id] = velocity
            
        # Check for same direction drift
        velocities = list(directions.values())
        if all(v > 0 for v in velocities) or all(v < 0 for v in velocities):
            # Check if velocities are similar
            velocity_std = np.std(velocities)
            if velocity_std < 0.1:  # Similar velocities
                return OverlapScenario.TEMPERATURE_DRIFT
            else:
                return OverlapScenario.MIXED_EFFECT
                
        # Check for signal crossing
        for signal_i, signal_j in overlapping_pairs:
            if directions[signal_i] * directions[signal_j] < 0:  # Opposite directions
                return OverlapScenario.SIGNAL_CROSSING
                
        # Default to strain induced opposite drift
        return OverlapScenario.STRAIN_INDUCED
        
    def _calculate_overlap_confidence(self, signal_predictions: Dict[str, Dict],
                                    overlapping_pairs: List[Tuple[str, str]],
                                    scenario: OverlapScenario) -> float:
        """Calculate overlap detection confidence"""
        if not overlapping_pairs:
            return 0.0
            
        # Based on prediction confidence
        prediction_confidences = []
        for signal_id, prediction in signal_predictions.items():
            conf = prediction.get('confidence', 0.0)
            prediction_confidences.append(conf)
            
        avg_prediction_confidence = np.mean(prediction_confidences) if prediction_confidences else 0.0
        
        # Scenario-based confidence adjustment
        scenario_confidence_factors = {
            OverlapScenario.TEMPERATURE_DRIFT: 0.9,
            OverlapScenario.STRAIN_INDUCED: 0.8,
            OverlapScenario.MIXED_EFFECT: 0.6,
            OverlapScenario.SIGNAL_CROSSING: 0.7,
            OverlapScenario.NO_OVERLAP: 1.0
        }
        
        scenario_factor = scenario_confidence_factors.get(scenario, 0.5)
        
        # Overlap count based confidence adjustment
        overlap_factor = min(1.0, len(overlapping_pairs) / 3.0)
        
        return avg_prediction_confidence * scenario_factor * overlap_factor
        
    def _predict_separation_distance(self, prediction_i: Dict, prediction_j: Dict) -> float:
        """Predict separation distance"""
        # Predict based on historical direction difference
        direction_i = prediction_i.get('predicted_velocity', 0.0)
        direction_j = prediction_j.get('predicted_velocity', 0.0)
        
        # Predict future separation distance
        time_horizon = 1.0  # Predict 1 time unit ahead
        predicted_separation = abs((direction_i - direction_j) * time_horizon)
        
        # Ensure minimum separation distance
        return max(predicted_separation, 0.1)
        
    def direction_guided_signal_separation(self, current_signal: np.ndarray,
                                         signal_predictions: Dict[str, Dict],
                                         wavelength_array: np.ndarray) -> SeparationResult:
        """
        Direction prediction guided signal separation main entry
        
        Args:
            current_signal: Current mixed signal
            signal_predictions: Prediction info for each signal
            wavelength_array: Wavelength array
            
        Returns:
            Separation result
        """
        start_time = time.time()
        
        # Detect overlap
        overlap_detection = self.detect_signal_overlapping(signal_predictions, current_signal)
        
        if overlap_detection.scenario == OverlapScenario.NO_OVERLAP:
            # No overlap, return directly
            return SeparationResult(
                separated_signals={},
                separation_weights={},
                quality_scores={},
                overall_quality=1.0,
                method_used="no_separation_needed",
                processing_time=time.time() - start_time
            )
            
        # Select separation method
        separation_method = self._select_separation_method(overlap_detection)
        
        # Perform separation
        if separation_method in self.separation_methods:
            result = self.separation_methods[separation_method](
                current_signal, signal_predictions, wavelength_array, overlap_detection
            )
        else:
            # Default to direction weighted separation
            result = self._direction_weighted_separation(
                current_signal, signal_predictions, wavelength_array, overlap_detection
            )
            
        result.processing_time = time.time() - start_time
        
        # Update statistics
        self._update_separation_stats(overlap_detection, result)
        
        return result
        
    def _select_separation_method(self, overlap_detection: OverlapDetection) -> str:
        """Select best separation method"""
        scenario = overlap_detection.scenario
        severity = overlap_detection.overlap_severity
        confidence = overlap_detection.confidence
        
        if scenario == OverlapScenario.TEMPERATURE_DRIFT:
            if severity < 0.5:
                return 'temporal_separation'
            else:
                return 'direction_weighted'
                
        elif scenario == OverlapScenario.STRAIN_INDUCED:
            if confidence > 0.8:
                return 'direction_weighted'
            else:
                return 'iterative_separation'
                
        elif scenario == OverlapScenario.SIGNAL_CROSSING:
            return 'frequency_domain'
            
        elif scenario == OverlapScenario.MIXED_EFFECT:
            return 'hybrid_approach'
            
        else:
            return 'direction_weighted'
            
    def _direction_weighted_separation(self, current_signal: np.ndarray,
                                     signal_predictions: Dict[str, Dict],
                                     wavelength_array: np.ndarray,
                                     overlap_detection: OverlapDetection) -> SeparationResult:
        """Direction weighted separation algorithm"""
        separated_signals = {}
        separation_weights = {}
        quality_scores = {}
        
        # Calculate separation weights for each overlapping pair
        for signal_i, signal_j in overlap_detection.overlapping_pairs:
            # Get historical direction
            direction_i = self._get_historical_direction(signal_predictions[signal_i])
            direction_j = self._get_historical_direction(signal_predictions[signal_j])
            
            # Calculate direction difference
            direction_difference = abs(direction_i - direction_j)
            
            # Calculate separation weights
            if direction_difference > 0.1:  # Significant direction difference
                # Normalize direction as weight
                total_direction = abs(direction_i) + abs(direction_j) + 1e-6
                weight_i = abs(direction_i) / total_direction
                weight_j = abs(direction_j) / total_direction
                
                # Ensure weights sum to 1
                weight_sum = weight_i + weight_j
                weight_i /= weight_sum
                weight_j /= weight_sum
                
            else:
                # Similar direction, equal distribution
                weight_i = weight_j = 0.5
                
            separation_weights[(signal_i, signal_j)] = (weight_i, weight_j)
            
        # Apply weights to separate signals
        all_signals = set()
        for signal_i, signal_j in overlap_detection.overlapping_pairs:
            all_signals.add(signal_i)
            all_signals.add(signal_j)
            
        # Generate separated signal for each signal
        for signal_id in all_signals:
            separated_signal = np.zeros_like(current_signal)
            total_weight = 0.0
            
            for (signal_i, signal_j), (weight_i, weight_j) in separation_weights.items():
                if signal_id == signal_i:
                    # Generate template for this signal
                    template = self._generate_signal_template(
                        signal_predictions[signal_i], wavelength_array
                    )
                    separated_signal += weight_i * template
                    total_weight += weight_i
                elif signal_id == signal_j:
                    template = self._generate_signal_template(
                        signal_predictions[signal_j], wavelength_array
                    )
                    separated_signal += weight_j * template
                    total_weight += weight_j
                    
            if total_weight > 0:
                separated_signal /= total_weight
                
            # Calculate separation quality
            quality = self._calculate_separation_quality(
                separated_signal, current_signal, signal_predictions[signal_id]
            )
            
            separated_signals[signal_id] = separated_signal
            quality_scores[signal_id] = quality
            
        # Calculate overall quality
        overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
        
        return SeparationResult(
            separated_signals=separated_signals,
            separation_weights=separation_weights,
            quality_scores=quality_scores,
            overall_quality=overall_quality,
            method_used="direction_weighted",
            processing_time=0.0  # Will be set in main function
        )
        
    def _temporal_separation(self, current_signal: np.ndarray,
                           signal_predictions: Dict[str, Dict],
                           wavelength_array: np.ndarray,
                           overlap_detection: OverlapDetection) -> SeparationResult:
        """Temporal separation algorithm"""
        # Separation based on temporal continuity
        separated_signals = {}
        quality_scores = {}
        
        for signal_id, prediction in signal_predictions.items():
            # Generate signal based on historical trends
            historical_signal = self._generate_historical_signal(prediction, wavelength_array)
            separated_signals[signal_id] = historical_signal
            
            # Calculate quality
            quality = self._calculate_separation_quality(
                historical_signal, current_signal, prediction
            )
            quality_scores[signal_id] = quality
            
        overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
        
        return SeparationResult(
            separated_signals=separated_signals,
            separation_weights={},
            quality_scores=quality_scores,
            overall_quality=overall_quality,
            method_used="temporal_separation",
            processing_time=0.0
        )
        
    def _frequency_domain_separation(self, current_signal: np.ndarray,
                                   signal_predictions: Dict[str, Dict],
                                   wavelength_array: np.ndarray,
                                   overlap_detection: OverlapDetection) -> SeparationResult:
        """Frequency domain separation algorithm"""
        # Simplified implementation: Frequency domain feature separation based on signal shape
        separated_signals = {}
        quality_scores = {}
        
        # Calculate frequency domain representation of current signal
        current_fft = np.fft.fft(current_signal)
        
        for signal_id, prediction in signal_predictions.items():
            # Generate frequency domain template for signal
            template = self._generate_signal_template(prediction, wavelength_array)
            template_fft = np.fft.fft(template)
            
            # Frequency domain filtering
            ifft_filter = template_fft / (np.abs(template_fft) + 1e-6)
            separated_fft = current_fft * np.abs(ifft_filter)
            
            # Inverse transform back to time domain
            separated_signal = np.real(np.fft.ifft(separated_fft))
            separated_signals[signal_id] = separated_signal
            
            # Calculate quality
            quality = self._calculate_separation_quality(
                separated_signal, current_signal, prediction
            )
            quality_scores[signal_id] = quality
            
        overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
        
        return SeparationResult(
            separated_signals=separated_signals,
            separation_weights={},
            quality_scores=quality_scores,
            overall_quality=overall_quality,
            method_used="frequency_domain",
            processing_time=0.0
        )
        
    def _iterative_separation(self, current_signal: np.ndarray,
                            signal_predictions: Dict[str, Dict],
                            wavelength_array: np.ndarray,
                            overlap_detection: OverlapDetection) -> SeparationResult:
        """Iterative separation algorithm"""
        separated_signals = {}
        quality_scores = {}
        
        # Iteratively separate each signal
        residual_signal = current_signal.copy()
        
        for signal_id, prediction in signal_predictions.items():
            # Generate template for current signal
            template = self._generate_signal_template(prediction, wavelength_array)
            
            # Calculate matching coefficient
            match_coeff = np.dot(residual_signal, template) / (np.dot(template, template) + 1e-6)
            
            # Extract signal
            extracted_signal = match_coeff * template
            separated_signals[signal_id] = extracted_signal
            
            # Update residual
            residual_signal -= extracted_signal
            
            # Calculate quality
            quality = self._calculate_separation_quality(
                extracted_signal, current_signal, prediction
            )
            quality_scores[signal_id] = quality
            
        overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
        
        return SeparationResult(
            separated_signals=separated_signals,
            separation_weights={},
            quality_scores=quality_scores,
            overall_quality=overall_quality,
            method_used="iterative_separation",
            processing_time=0.0
        )
        
    def _hybrid_separation(self, current_signal: np.ndarray,
                         signal_predictions: Dict[str, Dict],
                         wavelength_array: np.ndarray,
                         overlap_detection: OverlapDetection) -> SeparationResult:
        """Hybrid separation algorithm"""
        # Hybrid algorithm combining multiple methods
        
        # First try direction weighted separation
        direction_result = self._direction_weighted_separation(
            current_signal, signal_predictions, wavelength_array, overlap_detection
        )
        
        # If quality is insufficient, try iterative separation
        if direction_result.overall_quality < 0.6:
            iterative_result = self._iterative_separation(
                current_signal, signal_predictions, wavelength_array, overlap_detection
            )
            
            # Select better result
            if iterative_result.overall_quality > direction_result.overall_quality:
                return iterative_result
                
        return direction_result
        
    def _get_historical_direction(self, prediction: Dict) -> float:
        """Get historical direction"""
        return prediction.get('predicted_velocity', 0.0)
        
    def _generate_signal_template(self, prediction: Dict, wavelength_array: np.ndarray) -> np.ndarray:
        """Generate signal template"""
        # Simplified implementation: Generate Gaussian signal
        center_wavelength = prediction.get('predicted_position', np.mean(wavelength_array))
        width = 1.0  # Fixed width
        amplitude = 1.0  # Fixed amplitude
        
        template = amplitude * np.exp(-(wavelength_array - center_wavelength)**2 / (2 * width**2))
        
        return template
        
    def _generate_historical_signal(self, prediction: Dict, wavelength_array: np.ndarray) -> np.ndarray:
        """Generate signal based on history"""
        # Generate signal based on historical positions
        historical_positions = prediction.get('position_history', [])
        
        if historical_positions:
            # Use recent position
            recent_position = historical_positions[-1]
        else:
            recent_position = prediction.get('predicted_position', np.mean(wavelength_array))
            
        return self._generate_signal_template(
            {'predicted_position': recent_position}, wavelength_array
        )
        
    def _calculate_separation_quality(self, separated_signal: np.ndarray,
                                   original_signal: np.ndarray,
                                   prediction: Dict) -> float:
        """Calculate separation quality"""
        # Calculate correlation with original signal
        min_len = min(len(separated_signal), len(original_signal))
        sep_norm = separated_signal[:min_len]
        orig_norm = original_signal[:min_len]
        
        # Normalize
        sep_norm = (sep_norm - np.mean(sep_norm)) / (np.std(sep_norm) + 1e-6)
        orig_norm = (orig_norm - np.mean(orig_norm)) / (np.std(orig_norm) + 1e-6)
        
        correlation = np.corrcoef(sep_norm, orig_norm)[0, 1]
        
        if np.isnan(correlation):
            correlation = 0.0
            
        # Consider signal energy
        energy_ratio = np.sum(sep_norm**2) / (np.sum(orig_norm**2) + 1e-6)
        energy_factor = min(1.0, energy_ratio)
        
        # Comprehensive quality score
        quality = abs(correlation) * 0.8 + energy_factor * 0.2
        
        return max(0.0, min(1.0, quality))
        
    def _update_separation_stats(self, overlap_detection: OverlapDetection,
                               result: SeparationResult) -> None:
        """Update separation statistics"""
        self.separation_stats['total_separations'] += 1
        
        if result.overall_quality > 0.5:
            self.separation_stats['successful_separations'] += 1
            
        # Update average quality
        current_avg = self.separation_stats['average_quality']
        total_count = self.separation_stats['total_separations']
        self.separation_stats['average_quality'] = (
            (current_avg * (total_count - 1) + result.overall_quality) / total_count
        )
        
        # Update scenario distribution
        self.separation_stats['scenario_distribution'][overlap_detection.scenario] += 1
        
        # Update method usage statistics
        method = result.method_used
        if method not in self.separation_stats['method_usage']:
            self.separation_stats['method_usage'][method] = 0
        self.separation_stats['method_usage'][method] += 1
        
    def get_separation_report(self) -> Dict:
        """Get separation report"""
        total_separations = self.separation_stats['total_separations']
        
        if total_separations == 0:
            return {
                'total_separations': 0,
                'success_rate': 0.0,
                'average_quality': 0.0,
                'most_common_scenario': None,
                'most_used_method': None
            }
            
        success_rate = self.separation_stats['successful_separations'] / total_separations
        
        # Most common scenario
        most_common_scenario = max(
            self.separation_stats['scenario_distribution'].items(),
            key=lambda x: x[1]
        )[0].value
        
        # Most used method
        if self.separation_stats['method_usage']:
            most_used_method = max(
                self.separation_stats['method_usage'].items(),
                key=lambda x: x[1]
            )[0]
        else:
            most_used_method = None
            
        return {
            'total_separations': total_separations,
            'success_rate': success_rate,
            'average_quality': self.separation_stats['average_quality'],
            'most_common_scenario': most_common_scenario,
            'most_used_method': most_used_method,
            'scenario_distribution': {
                scenario.value: count for scenario, count in 
                self.separation_stats['scenario_distribution'].items()
            },
            'method_usage': self.separation_stats['method_usage']
        }
        
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.separation_stats = {
            'total_separations': 0,
            'successful_separations': 0,
            'average_quality': 0.0,
            'scenario_distribution': {scenario: 0 for scenario in OverlapScenario},
            'method_usage': {}
        }


def test_signal_separation():
    """Test signal separation functionality"""
    print("=== Test Signal Separation Engine ===")
    
    # Create separation engine
    separation_engine = SignalSeparationEngine()
    
    # Create test data
    wavelength_array = np.linspace(1520, 1570, 1000)
    
    # Create simulated signal predictions
    signal_predictions = {
        'FBG1': {
            'predicted_position': 1540.0,
            'predicted_velocity': 0.1,
            'confidence': 0.8,
            'position_history': [1539.8, 1539.9, 1540.0]
        },
        'FBG2': {
            'predicted_position': 1540.3,
            'predicted_velocity': 0.12,
            'confidence': 0.7,
            'position_history': [1540.0, 1540.1, 1540.3]
        },
        'FBG3': {
            'predicted_position': 1545.0,
            'predicted_velocity': -0.05,
            'confidence': 0.9,
            'position_history': [1545.2, 1545.1, 1545.0]
        }
    }
    
    # Create mixed signal
    current_signal = np.zeros_like(wavelength_array)
    for prediction in signal_predictions.values():
        template = np.exp(-(wavelength_array - prediction['predicted_position'])**2 / (2 * 1.0**2))
        current_signal += template
        
    # Add noise
    current_signal += np.random.normal(0, 0.05, len(wavelength_array))
    
    print("Testing overlap detection...")
    overlap_detection = separation_engine.detect_signal_overlapping(signal_predictions, current_signal)
    print(f"Overlap scenario: {overlap_detection.scenario.value}")
    print(f"Overlapping pairs: {overlap_detection.overlapping_pairs}")
    print(f"Overlap severity: {overlap_detection.overlap_severity:.3f}")
    print(f"Detection confidence: {overlap_detection.confidence:.3f}")
    
    print("\nTesting signal separation...")
    separation_result = separation_engine.direction_guided_signal_separation(
        current_signal, signal_predictions, wavelength_array
    )
    
    print(f"Separation method: {separation_result.method_used}")
    print(f"Overall quality: {separation_result.overall_quality:.3f}")
    print(f"Processing time: {separation_result.processing_time:.4f}s")
    print(f"Separated signals count: {len(separation_result.separated_signals)}")
    
    if separation_result.separated_signals:
        print("Signal quality:")
        for signal_id, quality in separation_result.quality_scores.items():
            print(f"  {signal_id}: {quality:.3f}")
            
    if separation_result.separation_weights:
        print("Separation weights:")
        for (signal_i, signal_j), weights in separation_result.separation_weights.items():
            print(f"  {signal_i}-{signal_j}: ({weights[0]:.3f}, {weights[1]:.3f})")
    
    # Output separation report
    print(f"\nSeparation Report:")
    report = separation_engine.get_separation_report()
    print(f"Total separations: {report['total_separations']}")
    print(f"Success rate: {report['success_rate']:.1%}")
    print(f"Average quality: {report['average_quality']:.3f}")
    print(f"Most common scenario: {report['most_common_scenario']}")
    print(f"Most used method: {report['most_used_method']}")


if __name__ == "__main__":
    test_signal_separation()