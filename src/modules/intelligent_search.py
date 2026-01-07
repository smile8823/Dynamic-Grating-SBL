"""
Intelligent Search Controller
Stage3 Core Module: Intelligent search and atom set optimization

Core Functions:
1. Intelligent atom set search
2. Search strategy optimization
3. Performance evaluation and feedback
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class SearchResult:
    """Search Result"""
    atom_indices: List[int]
    quality_score: float
    computation_time: float
    algorithm_used: str


class IntelligentSearchController:
    """
    Intelligent Search Controller
    
    Provides multiple search strategies for atom set optimization
    """
    
    def __init__(self, phi_matrix: np.ndarray, max_atoms: int = 10):
        """
        Initialize Intelligent Search Controller
        
        Args:
            phi_matrix: Dictionary matrix
            max_atoms: Maximum number of atoms
        """
        self.phi_matrix = phi_matrix
        self.signal_dimension, self.dict_size = phi_matrix.shape
        self.max_atoms = max_atoms
        
        # Search parameters
        self.search_algorithms = ['greedy', 'genetic', 'local_search']
        self.max_iterations = 100
        self.quality_threshold = 0.5
        
        # Statistics
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'average_time': 0.0,
            'best_quality': 0.0
        }
    
    def search_optimal_atom_set(self, signal: np.ndarray, 
                               current_atoms: Optional[List[int]] = None,
                               algorithm: str = 'greedy') -> SearchResult:
        """
        Search for optimal atom set
        
        Args:
            signal: Input signal
            current_atoms: Current atom indices
            algorithm: Search algorithm
            
        Returns:
            SearchResult: Search result
        """
        start_time = time.time()
        
        try:
            if algorithm == 'greedy':
                result = self._greedy_search(signal, current_atoms)
            elif algorithm == 'local_search':
                result = self._local_search(signal, current_atoms)
            elif algorithm == 'genetic':
                result = self._genetic_search(signal, current_atoms)
            else:
                # Default to greedy search
                result = self._greedy_search(signal, current_atoms)
            
            result.computation_time = time.time() - start_time
            result.algorithm_used = algorithm
            
            # Update statistics
            self.search_stats['total_searches'] += 1
            if result.quality_score > self.quality_threshold:
                self.search_stats['successful_searches'] += 1
            
            self.search_stats['average_time'] = (
                (self.search_stats['average_time'] * (self.search_stats['total_searches'] - 1) + 
                 result.computation_time) / self.search_stats['total_searches']
            )
            
            if result.quality_score > self.search_stats['best_quality']:
                self.search_stats['best_quality'] = result.quality_score
            
            return result
            
        except Exception as e:
            print(f"Search failed: {e}")
            # Return default result
            return SearchResult(
                atom_indices=current_atoms or list(range(min(self.max_atoms, self.dict_size))),
                quality_score=0.1,
                computation_time=time.time() - start_time,
                algorithm_used=algorithm
            )
    
    def _greedy_search(self, signal: np.ndarray, 
                      current_atoms: Optional[List[int]] = None) -> SearchResult:
        """Greedy search algorithm"""
        try:
            # Calculate correlation
            correlations = np.abs(np.dot(self.phi_matrix.T, signal))
            
            # Select atoms with highest correlation
            if current_atoms is None:
                selected_atoms = []
            else:
                selected_atoms = current_atoms.copy()
            
            # Add missing atoms
            remaining_slots = self.max_atoms - len(selected_atoms)
            if remaining_slots > 0:
                # Exclude selected atoms
                correlations[np.array(selected_atoms)] = 0
                
                # Select atoms with highest correlation
                new_atoms = np.argsort(correlations)[-remaining_slots:]
                selected_atoms.extend(new_atoms.tolist())
            
            # Calculate quality score
            quality_score = self._evaluate_atom_set_quality(signal, selected_atoms)
            
            return SearchResult(
                atom_indices=selected_atoms,
                quality_score=quality_score,
                computation_time=0.0,
                algorithm_used='greedy'
            )
            
        except Exception:
            return SearchResult(
                atom_indices=current_atoms or [0],
                quality_score=0.1,
                computation_time=0.0,
                algorithm_used='greedy'
            )
    
    def _local_search(self, signal: np.ndarray, 
                     current_atoms: Optional[List[int]] = None) -> SearchResult:
        """Local search algorithm"""
        # Simplified implementation: search near current atoms
        if current_atoms is None or len(current_atoms) == 0:
            return self._greedy_search(signal, current_atoms)
        
        best_atoms = current_atoms.copy()
        best_quality = self._evaluate_atom_set_quality(signal, best_atoms)
        
        # Try to replace each atom
        for i, atom_idx in enumerate(best_atoms):
            # Try to replace with adjacent atoms
            for delta in [-2, -1, 1, 2]:
                new_atom = atom_idx + delta
                if 0 <= new_atom < self.dict_size and new_atom not in best_atoms:
                    test_atoms = best_atoms.copy()
                    test_atoms[i] = new_atom
                    
                    test_quality = self._evaluate_atom_set_quality(signal, test_atoms)
                    if test_quality > best_quality:
                        best_atoms = test_atoms
                        best_quality = test_quality
        
        return SearchResult(
            atom_indices=best_atoms,
            quality_score=best_quality,
            computation_time=0.0,
            algorithm_used='local_search'
        )
    
    def _genetic_search(self, signal: np.ndarray, 
                       current_atoms: Optional[List[int]] = None) -> SearchResult:
        """Genetic search algorithm (simplified version)"""
        # Simplified implementation: random sampling + selection
        population_size = 10
        generations = 5
        
        # Initialize population
        if current_atoms is None:
            population = [
                np.random.choice(self.dict_size, self.max_atoms, replace=False).tolist()
                for _ in range(population_size)
            ]
        else:
            # Generate mutations based on current atoms
            population = [current_atoms.copy()]
            for _ in range(population_size - 1):
                individual = current_atoms.copy()
                # Randomly replace 1-2 atoms
                num_mutations = np.random.randint(1, 3)
                for _ in range(num_mutations):
                    idx = np.random.randint(0, len(individual))
                    new_atom = np.random.randint(0, self.dict_size)
                    while new_atom in individual:
                        new_atom = np.random.randint(0, self.dict_size)
                    individual[idx] = new_atom
                population.append(individual)
        
        # Evolution
        best_individual = None
        best_quality = 0.0
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                quality = self._evaluate_atom_set_quality(signal, individual)
                fitness_scores.append(quality)
                
                if quality > best_quality:
                    best_quality = quality
                    best_individual = individual.copy()
            
            # Selection and mutation (simplified version)
            # Select the best individuals to keep
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices[:5]]
            
            # Generate new individuals
            while len(population) < population_size:
                parent = population[np.random.randint(0, len(population))]
                offspring = parent.copy()
                
                # Mutation
                if np.random.random() < 0.3:  # 30% mutation rate
                    idx = np.random.randint(0, len(offspring))
                    offspring[idx] = np.random.randint(0, self.dict_size)
                
                population.append(offspring)
        
        return SearchResult(
            atom_indices=best_individual or current_atoms or [0],
            quality_score=best_quality,
            computation_time=0.0,
            algorithm_used='genetic'
        )
    
    def _evaluate_atom_set_quality(self, signal: np.ndarray, 
                                 atom_indices: List[int]) -> float:
        """Evaluate atom set quality"""
        try:
            if len(atom_indices) == 0:
                return 0.0
            
            # Construct sub-dictionary
            sub_phi = self.phi_matrix[:, atom_indices]
            
            # Calculate sparse representation (least squares)
            if sub_phi.shape[1] == 1:
                coeffs = np.dot(sub_phi.T, signal) / np.dot(sub_phi.T, sub_phi)
            else:
                coeffs = np.linalg.lstsq(sub_phi, signal, rcond=None)[0]
            
            # Reconstruct signal
            reconstructed = sub_phi @ coeffs
            
            # Calculate reconstruction error
            error = np.mean((signal - reconstructed) ** 2)
            signal_power = np.mean(signal ** 2)
            
            # Calculate SNR
            if signal_power > 0:
                snr = signal_power / (error + 1e-10)
                quality = snr / (1 + snr)  # Normalize to [0,1]
            else:
                quality = 0.0
            
            return min(1.0, quality)
            
        except Exception:
            return 0.1
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return self.search_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset statistics"""
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'average_time': 0.0,
            'best_quality': 0.0
        }


if __name__ == "__main__":
    # Simple test
    phi = np.random.randn(100, 50)
    signal = np.random.randn(100)
    
    controller = IntelligentSearchController(phi, max_atoms=5)
    result = controller.search_optimal_atom_set(signal)
    
    print(f"Search Result:")
    print(f"  Atom Indices: {result.atom_indices}")
    print(f"  Quality Score: {result.quality_score:.3f}")
    print(f"  Computation Time: {result.computation_time:.3f}s")
    print(f"  Algorithm: {result.algorithm_used}")