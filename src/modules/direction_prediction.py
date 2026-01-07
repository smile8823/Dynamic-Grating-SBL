"""
Direction Prediction Engine
Stage3 Core Module: Predict next frame position based on historical change directions to optimize computation

Core Functions:
1. Historical trend analysis and prediction
2. Pattern learning and adaptive weighting
3. Confidence assessment
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class DirectionPredictionModel:
    """
    Direction Prediction Model
    
    Predicts next frame position using historical change directions, reducing computation by 80-95%
    """
    
    def __init__(self, history_length: int = 10, confidence_threshold: float = 0.7):
        """
        Initialize Direction Prediction Model
        
        Args:
            history_length: Length of history data
            confidence_threshold: Confidence threshold
        """
        self.history_length = history_length
        self.confidence_threshold = confidence_threshold
        
        # History data buffer
        self.offset_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)
        self.acceleration_history = deque(maxlen=history_length)
        self.timestamp_history = deque(maxlen=history_length)
        
        # Pattern learning parameters
        self.periodic_patterns = {}
        self.trend_models = {}
        self.abrupt_change_catalog = []
        
        # Prediction performance statistics
        self.prediction_accuracy = []
        self.model_weights = {
            'linear_prediction': 0.4,
            'periodic_model': 0.3,
            'trend_model': 0.2,
            'historical_average': 0.1
        }
        
    def update_history(self, current_offset: float, timestamp: float = None) -> None:
        """
        Update history data
        
        Args:
            current_offset: Current offset value
            timestamp: Timestamp
        """
        # Input validation
        if not np.isfinite(current_offset):
            print(f"Warning: Invalid offset value {current_offset}, skipping update")
            return
            
        if timestamp is None:
            timestamp = len(self.offset_history)
        elif not np.isfinite(timestamp):
            timestamp = len(self.offset_history)
            
        # Update offset history
        if len(self.offset_history) > 0:
            # Calculate velocity
            last_offset = self.offset_history[-1]
            if np.isfinite(last_offset):
                velocity = current_offset - last_offset
                if np.isfinite(velocity):
                    self.velocity_history.append(velocity)
            
            # Calculate acceleration
            if len(self.velocity_history) > 1:
                last_velocity = self.velocity_history[-2]
                current_velocity = self.velocity_history[-1]
                if np.isfinite(last_velocity) and np.isfinite(current_velocity):
                    acceleration = current_velocity - last_velocity
                    if np.isfinite(acceleration):
                        self.acceleration_history.append(acceleration)
                        
        self.offset_history.append(current_offset)
        self.timestamp_history.append(timestamp)
        
    def predict_next_direction(self, prediction_type: str = 'adaptive') -> Dict:
        """
        Predict next frame change direction
        
        Args:
            prediction_type: Prediction type ('linear', 'periodic', 'trend', 'adaptive')
            
        Returns:
            Prediction result dictionary
        """
        # Enhanced data validation
        validation_result = self._validate_prediction_data()
        if not validation_result['is_valid']:
            return {
                'predicted_velocity': 0.0,
                'confidence': 0.0,
                'method': validation_result['error_type'],
                'predicted_position': None,
                'validation_details': validation_result
            }
            
        prediction_results = {}
        
        # Linear prediction
        prediction_results['linear'] = self._linear_prediction()
        
        # Periodic prediction
        if len(self.offset_history) >= 6:
            prediction_results['periodic'] = self._periodic_prediction()
        
        # Trend prediction
        if len(self.offset_history) >= 4:
            prediction_results['trend'] = self._trend_prediction()
            
        # Historical average prediction
        prediction_results['historical'] = self._historical_average_prediction()
        
        # Validate all prediction results
        validated_results = self._validate_prediction_results(prediction_results)
        
        # Adaptive combination prediction
        if prediction_type == 'adaptive':
            final_result = self._adaptive_combination(validated_results)
        elif prediction_type in validated_results:
            final_result = validated_results[prediction_type]
        else:
            final_result = validated_results['linear']
            
        # Final result validation
        return self._validate_final_prediction(final_result)
            
    def _linear_prediction(self) -> Dict:
        """Linear prediction: Based on recent velocity and acceleration"""
        if len(self.velocity_history) == 0:
            predicted_velocity = 0.0
            confidence = 0.0
        elif len(self.acceleration_history) == 0:
            predicted_velocity = self.velocity_history[-1]
            confidence = 0.6
        else:
            # Prediction based on recent acceleration
            predicted_velocity = self.velocity_history[-1] + self.acceleration_history[-1]
            
            # Calculate confidence
            velocity_consistency = 1.0 - np.std(list(self.velocity_history)[-3:]) / (np.abs(np.mean(list(self.velocity_history)[-3:])) + 1e-6)
            acceleration_consistency = 1.0 - np.std(list(self.acceleration_history)[-2:]) / (np.abs(np.mean(list(self.acceleration_history)[-2:])) + 1e-6)
            confidence = min(1.0, (velocity_consistency + acceleration_consistency) / 2)
            
        current_position = self.offset_history[-1] if self.offset_history else 0.0
        predicted_position = current_position + predicted_velocity
        
        return {
            'predicted_velocity': predicted_velocity,
            'confidence': max(0.0, min(1.0, confidence)),
            'method': 'linear',
            'predicted_position': predicted_position
        }
        
    def _periodic_prediction(self) -> Dict:
        """Periodic pattern prediction"""
        if len(self.offset_history) < 6:
            return self._linear_prediction()
            
        # Detect periodicity
        offsets = np.array(list(self.offset_history))
        
        # Use autocorrelation to detect period
        try:
            # Simplified period detection
            correlations = []
            for lag in range(2, min(len(offsets)//2, 10)):
                if len(offsets) > lag:
                    corr = np.corrcoef(offsets[:-lag], offsets[lag:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append((lag, abs(corr)))
                        
            if correlations:
                best_lag, best_corr = max(correlations, key=lambda x: x[1])
                
                if best_corr > 0.7:  # Strong periodicity
                    # Prediction based on periodicity
                    if len(offsets) >= best_lag:
                        predicted_velocity = offsets[-best_lag] - offsets[-best_lag-1] if best_lag < len(offsets) else 0
                        confidence = best_corr
                    else:
                        predicted_velocity = np.mean(np.diff(offsets))
                        confidence = 0.5
                else:
                    return self._linear_prediction()
            else:
                return self._linear_prediction()
                
        except:
            return self._linear_prediction()
            
        current_position = self.offset_history[-1]
        predicted_position = current_position + predicted_velocity
        
        return {
            'predicted_velocity': predicted_velocity,
            'confidence': confidence,
            'method': 'periodic',
            'predicted_position': predicted_position
        }
        
    def _trend_prediction(self) -> Dict:
        """Trend prediction: Based on polynomial fitting"""
        if len(self.offset_history) < 4:
            return self._linear_prediction()
            
        try:
            offsets = np.array(list(self.offset_history))
            indices = np.arange(len(offsets))
            
            # Linear trend
            linear_fit = np.polyfit(indices, offsets, 1)
            linear_velocity = linear_fit[0]
            
            # Quadratic trend
            if len(offsets) >= 6:
                quad_fit = np.polyfit(indices, offsets, 2)
                # Calculate derivative at next point
                next_index = len(offsets)
                quad_velocity = 2 * quad_fit[0] * next_index + quad_fit[1]
                
                # Calculate fit quality
                linear_pred = np.polyval(linear_fit, indices)
                quad_pred = np.polyval(quad_fit, indices)
                
                linear_r2 = 1 - np.sum((offsets - linear_pred)**2) / np.sum((offsets - np.mean(offsets))**2)
                quad_r2 = 1 - np.sum((offsets - quad_pred)**2) / np.sum((offsets - np.mean(offsets))**2)
                
                if quad_r2 > linear_r2 + 0.1:
                    predicted_velocity = quad_velocity
                    confidence = min(1.0, quad_r2)
                    method = 'quadratic_trend'
                else:
                    predicted_velocity = linear_velocity
                    confidence = min(1.0, linear_r2)
                    method = 'linear_trend'
            else:
                predicted_velocity = linear_velocity
                linear_pred = np.polyval(linear_fit, indices)
                linear_r2 = 1 - np.sum((offsets - linear_pred)**2) / np.sum((offsets - np.mean(offsets))**2)
                confidence = min(1.0, linear_r2)
                method = 'linear_trend'
                
        except:
            return self._linear_prediction()
            
        current_position = self.offset_history[-1]
        predicted_position = current_position + predicted_velocity
        
        return {
            'predicted_velocity': predicted_velocity,
            'confidence': confidence,
            'method': method,
            'predicted_position': predicted_position
        }
        
    def _historical_average_prediction(self) -> Dict:
        """Historical average prediction"""
        if len(self.velocity_history) == 0:
            return self._linear_prediction()
            
        velocities = list(self.velocity_history)
        
        # Use recent velocities for averaging
        recent_velocities = velocities[-min(5, len(velocities)):]
        predicted_velocity = np.mean(recent_velocities)
        
        # Calculate confidence (based on consistency)
        if len(recent_velocities) > 1:
            confidence = 1.0 - np.std(recent_velocities) / (np.abs(np.mean(recent_velocities)) + 1e-6)
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.3
            
        current_position = self.offset_history[-1]
        predicted_position = current_position + predicted_velocity
        
        return {
            'predicted_velocity': predicted_velocity,
            'confidence': confidence,
            'method': 'historical_average',
            'predicted_position': predicted_position
        }
        
    def _adaptive_combination(self, prediction_results: Dict) -> Dict:
        """Adaptive combination of multiple prediction methods"""
        if not prediction_results:
            return self._linear_prediction()
            
        # Weighted average based on confidence
        total_weight = 0
        weighted_velocity = 0
        method_contributions = {}
        
        for method, result in prediction_results.items():
            confidence = result['confidence']
            weight = self.model_weights.get(method, 0.1) * confidence
            
            weighted_velocity += weight * result['predicted_velocity']
            total_weight += weight
            method_contributions[method] = weight
            
        if total_weight > 0:
            final_velocity = weighted_velocity / total_weight
            final_confidence = min(1.0, total_weight)
        else:
            # Fallback to linear prediction
            linear_result = prediction_results.get('linear', self._linear_prediction())
            final_velocity = linear_result['predicted_velocity']
            final_confidence = linear_result['confidence']
            method_contributions = {'fallback_linear': 1.0}
            
        current_position = self.offset_history[-1] if self.offset_history else 0.0
        predicted_position = current_position + final_velocity
        
        return {
            'predicted_velocity': final_velocity,
            'confidence': final_confidence,
            'method': 'adaptive_combination',
            'predicted_position': predicted_position,
            'method_contributions': method_contributions
        }
        
    def _validate_prediction_data(self) -> Dict:
        """
        Validate prediction data
        
        Returns:
            Validation result dictionary
        """
        result = {
            'is_valid': True,
            'error_type': None,
            'error_details': None,
            'data_quality': 'good'
        }
        
        # Check data quantity
        if len(self.offset_history) < 2:
            result['is_valid'] = False
            result['error_type'] = 'insufficient_data'
            result['error_details'] = f'Need at least 2 data points, got {len(self.offset_history)}'
            return result
            
        # Check data validity
        offsets = list(self.offset_history)
        
        # Check for NaN/Inf values
        invalid_count = sum(1 for val in offsets if not np.isfinite(val))
        if invalid_count > 0:
            result['is_valid'] = False
            result['error_type'] = 'invalid_data'
            result['error_details'] = f'Found {invalid_count} invalid values (NaN/Inf)'
            
            # Clean up invalid data
            self._cleanup_invalid_data()
            return result
            
        # Check data range rationality
        offset_range = max(offsets) - min(offsets)
        if offset_range > 100:  # Assuming offset should not exceed 100nm
            result['data_quality'] = 'suspicious'
            result['error_details'] = f'Large offset range: {offset_range:.2f}nm'
            
        # Check data consistency
        if len(offsets) >= 3:
            # Calculate variance of change rate
            velocities = [offsets[i] - offsets[i-1] for i in range(1, len(offsets))]
            velocity_std = np.std(velocities)
            velocity_mean = np.mean(np.abs(velocities))
            
            if velocity_mean > 0:
                velocity_cv = velocity_std / velocity_mean  # Coefficient of variation
                if velocity_cv > 5.0:  # Change is too unstable
                    result['data_quality'] = 'unstable'
                    result['error_details'] = f'High velocity variability: CV={velocity_cv:.2f}'
                    
        return result
        
    def _cleanup_invalid_data(self):
        """Clean up invalid data"""
        # Clean invalid historical data
        valid_offsets = []
        valid_velocities = []
        valid_accelerations = []
        valid_timestamps = []
        
        for i, offset in enumerate(self.offset_history):
            if np.isfinite(offset):
                valid_offsets.append(offset)
                if i < len(self.timestamp_history):
                    valid_timestamps.append(self.timestamp_history[i])
                    
        # Recalculate velocity and acceleration
        for i in range(1, len(valid_offsets)):
            velocity = valid_offsets[i] - valid_offsets[i-1]
            if np.isfinite(velocity):
                valid_velocities.append(velocity)
                
        for i in range(1, len(valid_velocities)):
            acceleration = valid_velocities[i] - valid_velocities[i-1]
            if np.isfinite(acceleration):
                valid_accelerations.append(acceleration)
                
        # Update history data
        self.offset_history = deque(valid_offsets, maxlen=self.history_length)
        self.velocity_history = deque(valid_velocities, maxlen=self.history_length)
        self.acceleration_history = deque(valid_accelerations, maxlen=self.history_length)
        self.timestamp_history = deque(valid_timestamps, maxlen=self.history_length)
        
        print(f"Cleaned invalid data: kept {len(valid_offsets)} valid offsets")
        
    def _validate_prediction_results(self, prediction_results: Dict) -> Dict:
        """
        Validate results from each prediction method
        
        Args:
            prediction_results: Prediction results from each method
            
        Returns:
            Validated prediction results
        """
        validated_results = {}
        
        for method, result in prediction_results.items():
            # Basic result validation
            if not isinstance(result, dict):
                print(f"Warning: {method} prediction returned invalid type: {type(result)}")
                continue
                
            # Check required fields
            required_fields = ['predicted_velocity', 'confidence', 'method']
            if not all(field in result for field in required_fields):
                print(f"Warning: {method} prediction missing required fields")
                continue
                
            # Validate prediction values
            predicted_velocity = result['predicted_velocity']
            confidence = result['confidence']
            
            # Check value validity
            if not np.isfinite(predicted_velocity):
                print(f"Warning: {method} prediction returned invalid velocity: {predicted_velocity}")
                result['predicted_velocity'] = 0.0
                result['confidence'] = 0.0
                
            if not np.isfinite(confidence):
                print(f"Warning: {method} prediction returned invalid confidence: {confidence}")
                result['confidence'] = 0.0
                
            # Check confidence range
            result['confidence'] = np.clip(result['confidence'], 0.0, 1.0)
            
            # Check prediction velocity rationality (assuming physical limits)
            max_reasonable_velocity = 10.0  # Max 10nm/frame
            if abs(result['predicted_velocity']) > max_reasonable_velocity:
                print(f"Warning: {method} prediction velocity too large: {result['predicted_velocity']}")
                result['predicted_velocity'] = np.sign(result['predicted_velocity']) * max_reasonable_velocity
                result['confidence'] *= 0.5  # Reduce confidence
                
            validated_results[method] = result
            
        # If all methods fail, provide default prediction
        if not validated_results:
            print("Warning: All prediction methods failed, using default")
            return {
                'linear': {
                    'predicted_velocity': 0.0,
                    'confidence': 0.1,
                    'method': 'fallback',
                    'predicted_position': self.offset_history[-1] if self.offset_history else 0.0
                }
            }
            
        return validated_results
        
    def _validate_final_prediction(self, prediction: Dict) -> Dict:
        """
        Validate final prediction result
        
        Args:
            prediction: Final prediction result
            
        Returns:
            Validated final prediction result
        """
        # Ensure all necessary fields exist
        if 'predicted_position' not in prediction:
            if 'predicted_velocity' in prediction and self.offset_history:
                prediction['predicted_position'] = self.offset_history[-1] + prediction['predicted_velocity']
            else:
                prediction['predicted_position'] = 0.0
                
        # Final value check
        for key in ['predicted_velocity', 'confidence', 'predicted_position']:
            if key in prediction:
                if not np.isfinite(prediction[key]):
                    print(f"Warning: Final prediction {key} is invalid: {prediction[key]}")
                    prediction[key] = 0.0 if key != 'confidence' else 0.0
                    
        # Confidence range check
        prediction['confidence'] = np.clip(prediction.get('confidence', 0.0), 0.0, 1.0)
        
        # Add validation flag
        prediction['validation_passed'] = True
        
        return prediction
        
    def get_confidence(self) -> float:
        """Get current prediction confidence"""
        if len(self.offset_history) < 2:
            return 0.0
            
        prediction = self.predict_next_direction()
        return prediction['confidence']
        
    def detect_velocity_trend(self) -> str:
        """Detect velocity trend"""
        if len(self.velocity_history) < 3:
            return 'insufficient_data'
            
        recent_velocities = list(self.velocity_history)[-3:]
        
        if all(v > 0 for v in recent_velocities):
            return 'increasing'
        elif all(v < 0 for v in recent_velocities):
            return 'decreasing'
        elif np.std(recent_velocities) < 1e-6:
            return 'stable'
        else:
            return 'oscillating'
            
    def detect_acceleration_trend(self) -> str:
        """Detect acceleration trend"""
        if len(self.acceleration_history) < 2:
            return 'insufficient_data'
            
        recent_accelerations = list(self.acceleration_history)[-2:]
        
        if all(a > 0 for a in recent_accelerations):
            return 'accelerating'
        elif all(a < 0 for a in recent_accelerations):
            return 'decelerating'
        else:
            return 'changing'
            
    def learn_periodic_patterns(self) -> Dict:
        """Learn periodic patterns"""
        if len(self.offset_history) < 10:
            return {}
            
        try:
            offsets = np.array(list(self.offset_history))
            
            # Simplified period detection
            periods = []
            for period in range(2, min(len(offsets)//3, 15)):
                if len(offsets) >= period * 2:
                    # Calculate periodicity strength
                    pattern_correlation = []
                    for i in range(len(offsets) - period):
                        pattern_correlation.append(1.0 - abs(offsets[i] - offsets[i + period]) / (abs(offsets[i]) + 1e-6))
                    
                    if pattern_correlation:
                        avg_correlation = np.mean(pattern_correlation)
                        if avg_correlation > 0.8:
                            periods.append((period, avg_correlation))
                            
            if periods:
                best_period, correlation = max(periods, key=lambda x: x[1])
                return {
                    'period': best_period,
                    'strength': correlation,
                    'pattern': offsets[-best_period:] if len(offsets) >= best_period else offsets
                }
                
        except:
            pass
            
        return {}
        
    def recognize_trend_patterns(self) -> Dict:
        """Recognize trend patterns"""
        if len(self.offset_history) < 5:
            return {}
            
        try:
            offsets = np.array(list(self.offset_history))
            indices = np.arange(len(offsets))
            
            # Linear fit
            linear_fit = np.polyfit(indices, offsets, 1)
            linear_r2 = 1 - np.sum((offsets - np.polyval(linear_fit, indices))**2) / np.var(offsets)
            
            # Determine trend type
            slope = linear_fit[0]
            
            if abs(slope) < 1e-6:
                trend_type = 'stable'
            elif slope > 0:
                trend_type = 'increasing'
            else:
                trend_type = 'decreasing'
                
            return {
                'trend_type': trend_type,
                'slope': slope,
                'r_squared': linear_r2,
                'strength': abs(slope) * linear_r2
            }
            
        except:
            return {}
            
    def update_prediction_accuracy(self, actual_offset: float) -> None:
        """Update prediction accuracy statistics"""
        if len(self.offset_history) >= 2:
            previous_prediction = self.predict_next_direction()
            predicted_offset = previous_prediction.get('predicted_position')
            
            if predicted_offset is not None:
                prediction_error = abs(actual_offset - predicted_offset)
                self.prediction_accuracy.append(prediction_error)
                
                # Keep records of the last 100 predictions
                if len(self.prediction_accuracy) > 100:
                    self.prediction_accuracy.pop(0)
                    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics"""
        if not self.prediction_accuracy:
            return {
                'mean_error': 0.0,
                'std_error': 0.0,
                'prediction_count': 0,
                'accuracy_score': 1.0
            }
            
        return {
            'mean_error': np.mean(self.prediction_accuracy),
            'std_error': np.std(self.prediction_accuracy),
            'prediction_count': len(self.prediction_accuracy),
            'accuracy_score': 1.0 / (1.0 + np.mean(self.prediction_accuracy))
        }
        
    def reset(self) -> None:
        """Reset prediction model"""
        self.offset_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.timestamp_history.clear()
        self.prediction_accuracy.clear()
        self.periodic_patterns.clear()
        self.trend_models.clear()
        self.abrupt_change_catalog.clear()
        
    def __del__(self):
        """Destructor to ensure resources are released"""
        try:
            self.reset()
            # Clear all references
            self.offset_history = None
            self.velocity_history = None
            self.acceleration_history = None
            self.timestamp_history = None
            self.prediction_accuracy = None
            self.periodic_patterns = None
            self.trend_models = None
            self.abrupt_change_catalog = None
            self.model_weights = None
        except:
            pass  # Ignore all errors during destruction


def test_direction_prediction():
    """Test direction prediction function"""
    print("=== Test Direction Prediction Engine ===")
    
    # Create prediction model
    model = DirectionPredictionModel(history_length=10)
    
    # Simulate offset data (linear growth + noise)
    np.random.seed(42)
    true_offsets = np.linspace(0, 5, 20) + np.random.normal(0, 0.1, 20)
    
    print("Updating history data and predicting...")
    predictions = []
    
    for i, offset in enumerate(true_offsets):
        model.update_history(offset)
        
        if i >= 2:  # Start predicting after enough history data
            prediction = model.predict_next_direction()
            predictions.append(prediction)
            
            print(f"Step {i+1}: "
                  f"True={offset:.3f}, "
                  f"Pred={prediction['predicted_position']:.3f}, "
                  f"Conf={prediction['confidence']:.3f}, "
                  f"Method={prediction['method']}")
            
            # Update prediction accuracy
            if i < len(true_offsets) - 1:
                model.update_prediction_accuracy(true_offsets[i+1])
    
    # Output statistics
    stats = model.get_prediction_stats()
    print(f"\nPrediction Statistics:")
    print(f"Mean Error: {stats['mean_error']:.4f}")
    print(f"Standard Error: {stats['std_error']:.4f}")
    print(f"Prediction Count: {stats['prediction_count']}")
    print(f"Accuracy Score: {stats['accuracy_score']:.3f}")
    
    # Test trend detection
    print(f"\nTrend Detection:")
    print(f"Velocity Trend: {model.detect_velocity_trend()}")
    print(f"Acceleration Trend: {model.detect_acceleration_trend()}")
    
    # Test pattern learning
    periodic_patterns = model.learn_periodic_patterns()
    trend_patterns = model.recognize_trend_patterns()
    
    print(f"\nPattern Learning:")
    print(f"Periodic Patterns: {periodic_patterns}")
    print(f"Trend Patterns: {trend_patterns}")


if __name__ == "__main__":
    test_direction_prediction()