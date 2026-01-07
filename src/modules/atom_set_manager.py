"""
Atom Set Management System
Stage3 Core Module: Dynamic Atom Set Update and Seamless Offset Handover

Core Features:
1. Candidate Atom Set Generation
2. Seamless Handover Mechanism
3. Global Reference System Management
4. Exception Handling and Recovery
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import warnings

warnings.filterwarnings('ignore')


class AtomSetIDGenerator:
    """Atom Set Unique ID Generator"""
    
    def __init__(self):
        self.used_ids: Set[str] = set()
        self.counter = 0
        
    def generate_unique_id(self, signal_id: str, atom_set_type: str = "default") -> str:
        """
        Generate unique atom set ID
        
        Args:
            signal_id: Signal identifier
            atom_set_type: Atom set type
            
        Returns:
            Unique ID string
        """
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            # Generate candidate ID: signal_id_type_timestamp_counter
            timestamp = int(time.time() * 1000000)  # Microsecond precision
            candidate_id = f"{signal_id}_{atom_set_type}_{timestamp}_{self.counter}"
            
            if candidate_id not in self.used_ids:
                self.used_ids.add(candidate_id)
                self.counter += 1
                return candidate_id
                
            # If ID conflict, increment counter and retry
            self.counter += 1
            
        # If still conflict in rare cases, use UUID to ensure uniqueness
        import uuid
        fallback_id = f"{signal_id}_{atom_set_type}_{uuid.uuid4().hex[:8]}"
        self.used_ids.add(fallback_id)
        return fallback_id
        
    def is_id_used(self, atom_set_id: str) -> bool:
        """Check if ID is already used"""
        return atom_set_id in self.used_ids
        
    def release_id(self, atom_set_id: str) -> None:
        """Release ID (when atom set is destroyed)"""
        self.used_ids.discard(atom_set_id)
        
    def get_used_count(self) -> int:
        """Get count of used IDs"""
        return len(self.used_ids)
        
    def reset(self) -> None:
        """Reset generator"""
        self.used_ids.clear()
        self.counter = 0


class AtomSetStatus(Enum):
    """Atom Set Status Enum"""
    ACTIVE = "active"
    CANDIDATE = "candidate"
    EXPIRED = "expired"
    FAILED = "failed"


class HandoverPhase(Enum):
    """Handover Phase Enum"""
    IDLE = "idle"
    PARALLEL_OPERATION = "parallel_operation"
    COORDINATE_MAPPING = "coordinate_mapping"
    SWITCHING_EXECUTION = "switching_execution"
    CONFIRMATION_LOCKING = "confirmation_locking"


@dataclass
class AtomSet:
    """Enhanced Atom Set Data Class"""
    id: str
    atom_indices: np.ndarray
    reference_wavelengths: np.ndarray
    reference_offsets: np.ndarray
    creation_time: float
    quality_score: float
    usage_count: int = 0
    last_update_time: float = field(default_factory=time.time)
    status: AtomSetStatus = AtomSetStatus.CANDIDATE
    global_reference_offset: float = 0.0
    performance_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if self.last_update_time == 0.0:
            self.last_update_time = self.creation_time


@dataclass
class HandoverResult:
    """Handover Result Data Class"""
    success: bool
    old_atom_set_id: str
    new_atom_set_id: str
    coordinate_mapping: Dict[str, float]
    switching_time: float
    quality_improvement: float
    confidence: float


class AtomSetManager:
    """
    Atom Set Manager
    
    Responsible for dynamic atom set updates and seamless handovers
    """
    
    def __init__(self, phi_global: np.ndarray, wavelength_array: np.ndarray,
                 max_candidate_sets: int = 5, quality_threshold: float = 0.7):
        """
        Initialize Atom Set Manager
        
        Args:
            phi_global: Global dictionary matrix
            wavelength_array: Wavelength array
            max_candidate_sets: Maximum number of candidate sets
            quality_threshold: Quality threshold
        """
        self.phi_global = phi_global
        self.wavelength_array = wavelength_array
        self.max_candidate_sets = max_candidate_sets
        self.quality_threshold = quality_threshold
        
        # Atom set storage
        self.active_atom_sets: Dict[str, AtomSet] = {}  # signal_id -> AtomSet
        self.candidate_atom_sets: Dict[str, List[AtomSet]] = {}  # signal_id -> [AtomSet]
        self.expired_atom_sets: List[AtomSet] = []
        
        # Global reference system management
        self.global_reference_offsets: Dict[str, float] = {}  # signal_id -> global_offset
        
        # Handover management
        self.handover_phase: Dict[str, HandoverPhase] = {}
        self.handover_results: List[HandoverResult] = []
        
        # ID generator
        self.id_generator = AtomSetIDGenerator()
        
        # Performance statistics
        self.management_stats = {
            'total_generations': 0,
            'total_switches': 0,
            'successful_switches': 0,
            'average_quality_improvement': 0.0,
            'average_switching_time': 0.0,
            'atom_set_lifetimes': []
        }
        
    def generate_candidate_atom_sets(self, signal_id: str, current_frame: np.ndarray,
                                   current_performance: float) -> List[AtomSet]:
        """
        Generate candidate atom sets
        
        Args:
            signal_id: Signal identifier
            current_frame: Current frame signal
            current_performance: Current performance score
            
        Returns:
            List of candidate atom sets
        """
        candidates = []
        
        # Strategy 1: Local update
        local_candidates = self._generate_local_update_candidates(
            signal_id, current_frame, current_performance
        )
        candidates.extend(local_candidates)
        
        # Strategy 2: Neighborhood expansion
        neighborhood_candidates = self._generate_neighborhood_expansion_candidates(
            signal_id, current_frame
        )
        candidates.extend(neighborhood_candidates)
        
        # Strategy 3: Global rematch (if performance is poor)
        if current_performance < 0.5:
            global_candidates = self._generate_global_rematch_candidates(
                signal_id, current_frame
            )
            candidates.extend(global_candidates)
            
        # Evaluate and filter candidates
        evaluated_candidates = []
        for candidate in candidates:
            quality = self._evaluate_atom_set_quality(candidate, current_frame)
            candidate.quality_score = quality
            if quality > self.quality_threshold:
                evaluated_candidates.append(candidate)
                
        # Sort by quality
        evaluated_candidates.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Limit number of candidates (reduce candidates to improve performance)
        max_candidates = min(self.max_candidate_sets, 3)  # Max 3 candidates
        final_candidates = evaluated_candidates[:max_candidates]
        
        print(f"Generated {len(final_candidates)} candidates (from {len(evaluated_candidates)} evaluated)")
        
        # Update candidate storage
        if signal_id not in self.candidate_atom_sets:
            self.candidate_atom_sets[signal_id] = []
            
        self.candidate_atom_sets[signal_id] = final_candidates
        
        # Update statistics
        self.management_stats['total_generations'] += len(final_candidates)
        
        return final_candidates
        
    def _generate_local_update_candidates(self, signal_id: str, current_frame: np.ndarray,
                                        current_performance: float) -> List[AtomSet]:
        """Local update candidates"""
        candidates = []
        
        # Get current active set
        if signal_id not in self.active_atom_sets:
            return candidates
            
        active_set = self.active_atom_sets[signal_id]
        
        # Strategy 1: Keep partial effective atoms, replace failed atoms
        if current_performance < 0.7:
            # Analyze atom performance
            atom_performances = self._analyze_atom_performances(active_set, current_frame)
            
            # Keep good performing atoms
            good_atoms = []
            poor_atoms = []
            
            for i, (atom_idx, performance) in enumerate(zip(active_set.atom_indices, atom_performances)):
                if performance > 0.5:
                    good_atoms.append(atom_idx)
                else:
                    poor_atoms.append(i)
                    
            if len(good_atoms) >= 1 and len(poor_atoms) > 0:
                # Find replacement for failed atoms
                for poor_atom_idx in poor_atoms:
                    replacement_candidates = self._find_replacement_atoms(
                        signal_id, current_frame, good_atoms, poor_atom_idx
                    )
                    
                    for replacement_atom in replacement_candidates[:2]:  # Max 2 replacements
                        new_atom_indices = list(active_set.atom_indices)
                        new_atom_indices[poor_atom_idx] = replacement_atom
                        
                        candidate = AtomSet(
                            id=self.id_generator.generate_unique_id(signal_id, "local_update"),
                            atom_indices=np.array(new_atom_indices),
                            reference_wavelengths=active_set.reference_wavelengths.copy(),
                            reference_offsets=active_set.reference_offsets.copy(),
                            creation_time=time.time(),
                            quality_score=0.0  # Will be set in subsequent evaluation
                        )
                        candidates.append(candidate)
                        
        return candidates
        
    def _generate_neighborhood_expansion_candidates(self, signal_id: str,
                                                 current_frame: np.ndarray) -> List[AtomSet]:
        """Neighborhood expansion candidates"""
        candidates = []
        
        if signal_id not in self.active_atom_sets:
            return candidates
            
        active_set = self.active_atom_sets[signal_id]
        
        # Find similar neighboring atoms for each atom
        for atom_idx in active_set.atom_indices:
            # Find similar neighboring atoms
            similar_atoms = self._find_similar_atoms(atom_idx, current_frame)
            
            # Generate new set
            for similar_atom in similar_atoms[:3]:  # Max 3 similar atoms
                new_atom_indices = list(active_set.atom_indices)
                
                # Replace one atom with similar atom
                replace_idx = np.random.randint(len(new_atom_indices))
                new_atom_indices[replace_idx] = similar_atom
                
                candidate = AtomSet(
                    id=self.id_generator.generate_unique_id(signal_id, "neighborhood"),
                    atom_indices=np.array(new_atom_indices),
                    reference_wavelengths=active_set.reference_wavelengths.copy(),
                    reference_offsets=active_set.reference_offsets.copy(),
                    creation_time=time.time(),
                    quality_score=0.0
                )
                candidates.append(candidate)
                
        return candidates
        
    def _generate_global_rematch_candidates(self, signal_id: str,
                                          current_frame: np.ndarray) -> List[AtomSet]:
        """Global rematch candidates"""
        candidates = []
        
        # Use peak detection to find current main features
        try:
            from peak_detection import PeakDetectionSystem
            detector = PeakDetectionSystem()
            
            # Extract local signal
            if signal_id in self.active_atom_sets:
                active_set = self.active_atom_sets[signal_id]
                center_wavelength = np.mean(active_set.reference_wavelengths)
                signal_range = (center_wavelength - 5, center_wavelength + 5)
            else:
                center_wavelength = np.mean(self.wavelength_array)
                signal_range = (center_wavelength - 10, center_wavelength + 10)
                
            # Find index corresponding to wavelength range
            min_idx = np.argmin(np.abs(self.wavelength_array - signal_range[0]))
            max_idx = np.argmin(np.abs(self.wavelength_array - signal_range[1]))
            
            local_signal = current_frame[min_idx:max_idx]
            local_wavelengths = self.wavelength_array[min_idx:max_idx]
            
            # Peak detection
            peak_result = detector.detect_peaks(local_signal, local_wavelengths)
            
            if peak_result['success'] and peak_result['peak_count'] > 0:
                # Generate atom set for each peak
                for peak_wavelength in peak_result['peak_wavelengths'][:2]:  # Max 2 peaks
                    atom_set = self._create_atom_set_from_peak(
                        signal_id, peak_wavelength, current_frame
                    )
                    if atom_set:
                        candidates.append(atom_set)
                        
        except Exception as e:
            print(f"Error in global rematch candidate generation: {e}")
            
        return candidates
        
    def _find_replacement_atoms(self, signal_id: str, current_frame: np.ndarray,
                              good_atom_indices: List[int], poor_atom_position: int) -> List[int]:
        """Find replacement atoms"""
        # Find similar atoms based on good atoms
        if not good_atom_indices:
            # Randomly select some atoms
            return np.random.choice(self.phi_global.shape[1], 5, replace=False)
            
        # Calculate features of good atoms
        good_atom_features = self.phi_global[:, good_atom_indices]
        mean_feature = np.mean(good_atom_features, axis=1)
        
        # Optimize similarity calculation (vectorized operation)
        valid_atom_indices = [idx for idx in range(self.phi_global.shape[1]) 
                             if idx not in good_atom_indices]
        
        if not valid_atom_indices:
            return list(good_atom_indices)[:10]  # If no other atoms, return partial good atoms
        
        # Batch calculate similarity
        valid_features = self.phi_global[:, valid_atom_indices]
        similarities = np.array([np.corrcoef(mean_feature, valid_features[:, i])[0, 1] 
                               for i in range(valid_features.shape[1])])
        
        # Handle NaN values and sort
        valid_similarities = similarities[~np.isnan(similarities)]
        valid_indices = np.array(valid_atom_indices)[~np.isnan(similarities)]
        
        if len(valid_similarities) > 0:
            # Use argpartition to improve sorting efficiency
            top_indices = np.argpartition(valid_similarities, -min(10, len(valid_similarities)))[-10:]
            sorted_top_indices = top_indices[np.argsort(valid_similarities[top_indices])[::-1]]
            return [valid_indices[i] for i in sorted_top_indices]
        else:
            return list(good_atom_indices)[:10]
        
    def _find_similar_atoms(self, atom_idx: int, current_frame: np.ndarray) -> List[int]:
        """Find similar atoms"""
        atom_feature = self.phi_global[:, atom_idx]
        
        # Optimize similar atom search (vectorized operation)
        other_indices = [idx for idx in range(self.phi_global.shape[1]) if idx != atom_idx]
        
        if not other_indices:
            return []
        
        # Batch calculate similarity
        other_features = self.phi_global[:, other_indices]
        similarities = np.array([np.corrcoef(atom_feature, other_features[:, i])[0, 1] 
                               for i in range(other_features.shape[1])])
        
        # Handle NaN values and sort
        valid_similarities = similarities[~np.isnan(similarities)]
        valid_indices = np.array(other_indices)[~np.isnan(similarities)]
        
        if len(valid_similarities) > 0:
            # Use argpartition to improve sorting efficiency
            top_indices = np.argpartition(valid_similarities, -min(5, len(valid_similarities)))[-5:]
            sorted_top_indices = top_indices[np.argsort(valid_similarities[top_indices])[::-1]]
            return [valid_indices[i] for i in sorted_top_indices]
        else:
            return []
        
    def _create_atom_set_from_peak(self, signal_id: str, peak_wavelength: float,
                                 current_frame: np.ndarray) -> Optional[AtomSet]:
        """Create atom set from peak"""
        try:
            # Find dictionary atom corresponding to peak
            peak_idx = np.argmin(np.abs(self.wavelength_array - peak_wavelength))
            
            # Select atoms around peak
            atom_window = 10
            min_atom_idx = max(0, peak_idx - atom_window)
            max_atom_idx = min(self.phi_global.shape[1], peak_idx + atom_window)
            
            candidate_atoms = np.arange(min_atom_idx, max_atom_idx)
            
            # Optimize match calculation (vectorized operation)
            candidate_features = self.phi_global[:, candidate_atoms]
            min_len = min(len(current_frame), candidate_features.shape[0])
            
            # Batch calculate correlation
            correlations = np.array([
                np.corrcoef(current_frame[:min_len], candidate_features[:min_len, i])[0, 1]
                for i in range(candidate_features.shape[1])
            ])
            
            # Handle NaN values and sort
            valid_correlations = np.abs(correlations[~np.isnan(correlations)])
            valid_atom_indices = np.array(candidate_atoms)[~np.isnan(correlations)]
            
            if len(valid_correlations) > 0:
                # Use argpartition to improve sorting efficiency
                top_indices = np.argpartition(valid_correlations, -min(3, len(valid_correlations)))[-3:]
                sorted_top_indices = top_indices[np.argsort(valid_correlations[top_indices])[::-1]]
                best_atoms = [valid_atom_indices[i] for i in sorted_top_indices]
            else:
                # If no valid correlation, randomly select 3 atoms
                best_atoms = candidate_atoms[:3] if len(candidate_atoms) >= 3 else candidate_atoms.tolist()
            
            if best_atoms:
                return AtomSet(
                    id=self.id_generator.generate_unique_id(signal_id, "global"),
                    atom_indices=np.array(best_atoms),
                    reference_wavelengths=np.array([peak_wavelength]),
                    reference_offsets=np.array([0.0]),
                    creation_time=time.time(),
                    quality_score=0.0
                )
                
        except Exception as e:
            print(f"Error creating atom set from peak: {e}")
            
        return None
        
    def _analyze_atom_performances(self, atom_set: AtomSet, current_frame: np.ndarray) -> List[float]:
        """Analyze atom performance"""
        performances = []
        
        for atom_idx in atom_set.atom_indices:
            atom_signal = self.phi_global[:, atom_idx]
            
            # Calculate match with current signal
            min_len = min(len(current_frame), len(atom_signal))
            if min_len > 0:
                correlation = np.corrcoef(
                    current_frame[:min_len], atom_signal[:min_len]
                )[0, 1]
                
                if not np.isnan(correlation):
                    performances.append(abs(correlation))
                else:
                    performances.append(0.0)
            else:
                performances.append(0.0)
                
        return performances
        
    def _evaluate_atom_set_quality(self, atom_set: AtomSet, current_frame: np.ndarray) -> float:
        """Evaluate atom set quality"""
        try:
            # Reconstruct signal
            reconstructed_signal = np.zeros_like(current_frame)
            for atom_idx in atom_set.atom_indices:
                atom_signal = self.phi_global[:, atom_idx]
                
                # Ensure length consistency
                min_len = min(len(current_frame), len(atom_signal))
                if min_len > 0:
                    # Simplified weighted combination
                    weight = 1.0 / len(atom_set.atom_indices)
                    reconstructed_signal[:min_len] += weight * atom_signal[:min_len]
                    
            # Calculate reconstruction quality
            min_len = min(len(current_frame), len(reconstructed_signal))
            if min_len > 0:
                current_norm = current_frame[:min_len]
                recon_norm = reconstructed_signal[:min_len]
                
                # Normalize
                current_norm = (current_norm - np.mean(current_norm)) / (np.std(current_norm) + 1e-6)
                recon_norm = (recon_norm - np.mean(recon_norm)) / (np.std(recon_norm) + 1e-6)
                
                # Calculate correlation
                correlation = np.corrcoef(current_norm, recon_norm)[0, 1]
                
                if not np.isnan(correlation):
                    return max(0.0, correlation)
                    
        except Exception as e:
            print(f"Error evaluating atom set quality: {e}")
            
        return 0.0
        
    def execute_seamless_handover(self, signal_id: str, new_atom_set: AtomSet) -> HandoverResult:
        """
        Execute seamless handover
        
        Args:
            signal_id: Signal identifier
            new_atom_set: New atom set
            
        Returns:
            Handover result
        """
        start_time = time.time()
        
        old_atom_set_id = None
        if signal_id in self.active_atom_sets:
            old_atom_set_id = self.active_atom_sets[signal_id].id
            
        # Set handover phase
        self.handover_phase[signal_id] = HandoverPhase.PARALLEL_OPERATION
        
        try:
            # Phase 1: Parallel operation period
            quality_improvement = self._parallel_operation_phase(signal_id, new_atom_set)
            
            # Phase 2: Coordinate mapping
            coordinate_mapping = self._coordinate_mapping_phase(signal_id, new_atom_set)
            
            # Phase 3: Handover execution
            switch_success = self._switching_execution_phase(signal_id, new_atom_set, coordinate_mapping)
            
            # Phase 4: Confirmation and locking
            final_success = self._confirmation_locking_phase(signal_id, new_atom_set)
            
            switching_time = time.time() - start_time
            
            if final_success:
                # Handover successful
                result = HandoverResult(
                    success=True,
                    old_atom_set_id=old_atom_set_id,
                    new_atom_set_id=new_atom_set.id,
                    coordinate_mapping=coordinate_mapping,
                    switching_time=switching_time,
                    quality_improvement=quality_improvement,
                    confidence=0.9
                )
                
                # Update statistics
                self.management_stats['total_switches'] += 1
                self.management_stats['successful_switches'] += 1
                
            else:
                # Handover failed
                result = HandoverResult(
                    success=False,
                    old_atom_set_id=old_atom_set_id,
                    new_atom_set_id=new_atom_set.id,
                    coordinate_mapping={},
                    switching_time=switching_time,
                    quality_improvement=0.0,
                    confidence=0.1
                )
                
                self.management_stats['total_switches'] += 1
                
            self.handover_results.append(result)
            self.handover_phase[signal_id] = HandoverPhase.IDLE
            
            return result
            
        except Exception as e:
            print(f"Error during handover: {e}")
            self.handover_phase[signal_id] = HandoverPhase.IDLE
            
            return HandoverResult(
                success=False,
                old_atom_set_id=old_atom_set_id,
                new_atom_set_id=new_atom_set.id,
                coordinate_mapping={},
                switching_time=time.time() - start_time,
                quality_improvement=0.0,
                confidence=0.0
            )
            
    def _parallel_operation_phase(self, signal_id: str, new_atom_set: AtomSet) -> float:
        """Parallel operation phase"""
        # Simplified implementation: Evaluate performance difference between new and old sets
        if signal_id in self.active_atom_sets:
            old_quality = self.active_atom_sets[signal_id].quality_score
            quality_improvement = new_atom_set.quality_score - old_quality
        else:
            quality_improvement = new_atom_set.quality_score
            
        return quality_improvement
        
    def _coordinate_mapping_phase(self, signal_id: str, new_atom_set: AtomSet) -> Dict[str, float]:
        """Coordinate mapping phase"""
        coordinate_mapping = {}
        
        if signal_id in self.active_atom_sets:
            old_atom_set = self.active_atom_sets[signal_id]
            
            # Calculate system offset between sets
            if len(old_atom_set.reference_wavelengths) > 0 and len(new_atom_set.reference_wavelengths) > 0:
                old_ref = np.mean(old_atom_set.reference_wavelengths)
                new_ref = np.mean(new_atom_set.reference_wavelengths)
                
                # System offset
                system_offset = new_ref - old_ref
                
                # Set global reference offset
                if signal_id in self.global_reference_offsets:
                    new_global_offset = self.global_reference_offsets[signal_id] + system_offset
                else:
                    new_global_offset = system_offset
                    
                coordinate_mapping['system_offset'] = system_offset
                coordinate_mapping['global_offset'] = new_global_offset
                coordinate_mapping['old_reference'] = old_ref
                coordinate_mapping['new_reference'] = new_ref
                
                # Update global reference system
                self.global_reference_offsets[signal_id] = new_global_offset
                
        return coordinate_mapping
        
    def _switching_execution_phase(self, signal_id: str, new_atom_set: AtomSet,
                                 coordinate_mapping: Dict[str, float]) -> bool:
        """Switching execution phase"""
        try:
            # Apply offset correction
            if 'global_offset' in coordinate_mapping:
                new_atom_set.global_reference_offset = coordinate_mapping['global_offset']
                
            # Update reference offset
            if len(new_atom_set.reference_offsets) > 0:
                new_atom_set.reference_offsets += coordinate_mapping.get('system_offset', 0.0)
                
            return True
            
        except Exception as e:
            print(f"Error in switching execution: {e}")
            return False
            
    def _confirmation_locking_phase(self, signal_id: str, new_atom_set: AtomSet) -> bool:
        """Confirmation and locking phase"""
        try:
            # Update active atom set
            if signal_id in self.active_atom_sets:
                old_atom_set = self.active_atom_sets[signal_id]
                old_atom_set.status = AtomSetStatus.EXPIRED
                self.expired_atom_sets.append(old_atom_set)
                
                # Release ID of old atom set
                self.id_generator.release_id(old_atom_set.id)
                
            # Set new set as active
            new_atom_set.status = AtomSetStatus.ACTIVE
            new_atom_set.usage_count += 1
            new_atom_set.last_update_time = time.time()
            
            self.active_atom_sets[signal_id] = new_atom_set
            
            # Cleanup candidate list
            if signal_id in self.candidate_atom_sets:
                self.candidate_atom_sets[signal_id] = [
                    candidate for candidate in self.candidate_atom_sets[signal_id]
                    if candidate.id != new_atom_set.id
                ]
                
            return True
            
        except Exception as e:
            print(f"Error in confirmation locking: {e}")
            return False
            
    def get_atom_set_info(self, signal_id: str = None) -> Dict:
        """Get atom set information"""
        if signal_id is None:
            # Return information for all signals
            all_info = {}
            for sid in self.active_atom_sets.keys():
                all_info[sid] = self._get_single_atom_set_info(sid)
            return all_info
        else:
            return self._get_single_atom_set_info(signal_id)
            
    def _get_single_atom_set_info(self, signal_id: str) -> Dict:
        """Get atom set information for single signal"""
        info = {
            'signal_id': signal_id,
            'active_atom_set': None,
            'candidate_count': 0,
            'global_reference_offset': self.global_reference_offsets.get(signal_id, 0.0),
            'handover_phase': self.handover_phase.get(signal_id, HandoverPhase.IDLE).value
        }
        
        if signal_id in self.active_atom_sets:
            atom_set = self.active_atom_sets[signal_id]
            info['active_atom_set'] = {
                'id': atom_set.id,
                'atom_count': len(atom_set.atom_indices),
                'quality_score': atom_set.quality_score,
                'usage_count': atom_set.usage_count,
                'creation_time': atom_set.creation_time,
                'last_update': atom_set.last_update_time,
                'status': atom_set.status.value,
                'global_reference_offset': atom_set.global_reference_offset
            }
            
        if signal_id in self.candidate_atom_sets:
            info['candidate_count'] = len(self.candidate_atom_sets[signal_id])
            
        return info
        
    def cleanup_expired_atom_sets(self, max_age_seconds: float = 3600) -> int:
        """
        Cleanup expired atom sets and release their IDs
        
        Args:
            max_age_seconds: Maximum retention time (seconds)
            
        Returns:
            Number of cleaned atom sets
        """
        current_time = time.time()
        cleanup_count = 0
        
        # Cleanup expired atom sets
        remaining_expired = []
        for expired_set in self.expired_atom_sets:
            age = current_time - expired_set.creation_time
            if age > max_age_seconds:
                # Release ID
                self.id_generator.release_id(expired_set.id)
                cleanup_count += 1
            else:
                remaining_expired.append(expired_set)
                
        self.expired_atom_sets = remaining_expired
        
        # Cleanup expired candidate sets
        for signal_id, candidates in self.candidate_atom_sets.items():
            remaining_candidates = []
            for candidate in candidates:
                age = current_time - candidate.creation_time
                if age <= max_age_seconds / 2:  # Candidate sets have shorter retention time
                    remaining_candidates.append(candidate)
                else:
                    self.id_generator.release_id(candidate.id)
                    cleanup_count += 1
            self.candidate_atom_sets[signal_id] = remaining_candidates
            
        return cleanup_count
        
    def get_management_report(self) -> Dict:
        """Get management report"""
        total_switches = self.management_stats['total_switches']
        
        if total_switches > 0:
            success_rate = self.management_stats['successful_switches'] / total_switches
        else:
            success_rate = 0.0
            
        return {
            'total_atom_sets': len(self.active_atom_sets),
            'total_candidates': sum(len(candidates) for candidates in self.candidate_atom_sets.values()),
            'total_expired': len(self.expired_atom_sets),
            'total_switches': total_switches,
            'success_rate': success_rate,
            'average_quality_improvement': self.management_stats['average_quality_improvement'],
            'average_switching_time': self.management_stats['average_switching_time'],
            'global_reference_offsets': self.global_reference_offsets.copy(),
            'recent_handovers': self.handover_results[-5:] if self.handover_results else [],
            'id_management': {
                'used_ids_count': self.id_generator.get_used_count(),
                'id_generator_active': True
            }
        }
        
    def cleanup_all_resources(self) -> int:
        """Cleanup all resources"""
        try:
            cleanup_count = 0
            
            # Cleanup all active atom sets
            for signal_id in list(self.active_atom_sets.keys()):
                atom_set = self.active_atom_sets[signal_id]
                atom_set.status = AtomSetStatus.EXPIRED
                self.expired_atom_sets.append(atom_set)
                # Release ID
                self.id_generator.release_id(atom_set.id)
                cleanup_count += 1
                
            self.active_atom_sets.clear()
            
            # Cleanup candidate sets
            for signal_id, candidates in self.candidate_atom_sets.items():
                for candidate in candidates:
                    self.id_generator.release_id(candidate.id)
                    cleanup_count += 1
                    
            self.candidate_atom_sets.clear()
            
            # Cleanup expired sets
            self.expired_atom_sets.clear()
            
            # Cleanup other resources
            self.global_reference_offsets.clear()
            self.handover_phase.clear()
            self.handover_results.clear()
            
            # Reset ID generator
            self.id_generator.reset()
            
            # Reset statistics
            self.management_stats = {
                'total_generations': 0,
                'total_switches': 0,
                'successful_switches': 0,
                'average_quality_improvement': 0.0,
                'average_switching_time': 0.0,
                'atom_set_lifetimes': []
            }
            
            return cleanup_count
            
        except Exception as e:
            print(f"Error during resource cleanup: {e}")
            return 0
            
    def __del__(self):
        """Destructor, ensure resources are released"""
        try:
            self.cleanup_all_resources()
            # Clear all references
            self.phi_global = None
            self.wavelength_array = None
            self.active_atom_sets = None
            self.candidate_atom_sets = None
            self.expired_atom_sets = None
            self.global_reference_offsets = None
            self.handover_phase = None
            self.handover_results = None
            self.id_generator = None
            self.management_stats = None
        except:
            pass  # Ignore all errors during destruction


def test_atom_set_manager():
    """Test Atom Set Management System"""
    print("=== Testing Atom Set Management System ===")
    
    # Create mock data
    wavelength_array = np.linspace(1520, 1570, 1000)
    phi_global = np.random.randn(1000, 100)
    
    # Create manager
    manager = AtomSetManager(phi_global, wavelength_array)
    
    # Create test signal
    test_signal = np.random.randn(1000)
    
    print("Generating candidate atom sets...")
    candidates = manager.generate_candidate_atom_sets("FBG1", test_signal, 0.6)
    print(f"Generated {len(candidates)} candidate sets")
    
    if candidates:
        best_candidate = candidates[0]
        print(f"Best candidate quality: {best_candidate.quality_score:.3f}")
        
        print("\nExecuting seamless handover...")
        handover_result = manager.execute_seamless_handover("FBG1", best_candidate)
        
        print(f"Handover successful: {handover_result.success}")
        print(f"Switching time: {handover_result.switching_time:.4f}s")
        print(f"Quality improvement: {handover_result.quality_improvement:.3f}")
        print(f"Confidence: {handover_result.confidence:.3f}")
        
        if handover_result.coordinate_mapping:
            print(f"Coordinate mapping: {handover_result.coordinate_mapping}")
    
    # Get atom set information
    print("\nAtom Set Information:")
    info = manager.get_atom_set_info("FBG1")
    print(f"Active Set ID: {info['active_atom_set']['id'] if info['active_atom_set'] else 'None'}")
    print(f"Candidate count: {info['candidate_count']}")
    print(f"Global reference offset: {info['global_reference_offset']:.3f}")
    
    # Get management report
    print("\nManagement Report:")
    report = manager.get_management_report()
    print(f"Total atom sets: {report['total_atom_sets']}")
    print(f"Total candidates: {report['total_candidates']}")
    print(f"Total switches: {report['total_switches']}")
    print(f"Success rate: {report['success_rate']:.1%}")


if __name__ == "__main__":
    test_atom_set_manager()
