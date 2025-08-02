"""
Pucker-specific experiment management.

Handles q-fold parameter optimization and pucker-specific analysis
as demonstrated in the original notebook experiments.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..config.experiment_config import ExperimentConfig, PUCKER_Q_FOLD_RANGES
from ..data.models import Suite
from ..preprocessing.pucker_analysis import determine_pucker_data
from ..clustering.hierarchical_clustering import pre_clustering
from ..clustering.cluster_refinement import refine_clusters_with_pns
from ..clustering.pns_clustering import PNSClusterer
from ..utils.io_utils import create_output_directory


class PuckerExperimentManager:
    """Manages pucker-specific experiments and q-fold optimization."""
    
    def __init__(self, 
                 output_dir: str = "q_fold_experiments",
                 min_cluster_size: int = 3):
        """
        Initialize pucker experiment manager.
        
        Args:
            output_dir: Directory for experiment outputs
            min_cluster_size: Minimum cluster size
        """
        self.output_dir = Path(output_dir)
        self.min_cluster_size = min_cluster_size
        self.results = {}
    
    def run_q_fold_experiments(self, 
                              suites: List[Suite],
                              scaled_coords: np.ndarray,
                              pucker_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run q-fold parameter experiments for all pucker types.
        
        This reproduces the q-fold experiments from the notebook.
        
        Args:
            suites: List of Suite objects
            scaled_coords: Scaled coordinate data
            pucker_types: Pucker types to analyze
            
        Returns:
            Dictionary containing q-fold experiment results
        """
        if pucker_types is None:
            pucker_types = list(PUCKER_Q_FOLD_RANGES.keys())
        
        self.output_dir.mkdir(exist_ok=True)
        
        print("Running q-fold experiments...")
        
        for pucker_type in pucker_types:
            print(f"\nAnalyzing {pucker_type}...")
            
            # Create pucker-specific output directory
            pucker_dir = self.output_dir / pucker_type
            pucker_dir.mkdir(exist_ok=True)
            
            # Get pucker-specific data
            pucker_indices, _ = determine_pucker_data(suites, pucker_type)
            if not pucker_indices:
                print(f"  No suites found for {pucker_type}")
                continue
            
            scaled_coords_subset = scaled_coords[pucker_indices]
            q_fold_values = PUCKER_Q_FOLD_RANGES[pucker_type]
            
            # Run experiments for each q-fold value
            pucker_results = {}
            
            for qf in q_fold_values:
                result = self._run_single_q_fold_experiment(
                    scaled_coords_subset, qf, pucker_type, pucker_dir
                )
                pucker_results[qf] = result
                
                cluster_sizes = [len(c) for c in result['clusters']]
                print(f"  q_fold={qf:.2f} → clusters: {cluster_sizes}")
            
            self.results[pucker_type] = pucker_results
            
            # Save results for this pucker type
            self._save_pucker_results(pucker_type, pucker_results)
        
        return self.results
    
    def _run_single_q_fold_experiment(self, 
                                    data: np.ndarray,
                                    q_fold: float,
                                    pucker_type: str,
                                    output_dir: Path) -> Dict[str, Any]:
        """Run clustering experiment for a single q-fold value."""
        from scipy.cluster.hierarchy import single as single_linkage
        
        # Run hierarchical clustering
        clusters, outliers, distance_threshold = pre_clustering(
            input_data=data,
            m=self.min_cluster_size,
            percentage=0.0,
            string_folder=str(output_dir),
            method=single_linkage,
            q_fold=q_fold,
            distance="low_res_suite_shape"
        )
        
        # Sort clusters by size
        sorted_clusters = sorted(clusters, key=len, reverse=True)
        
        return {
            'q_fold': q_fold,
            'clusters': sorted_clusters,
            'outliers': outliers,
            'distance_threshold': distance_threshold,
            'n_clusters': len(sorted_clusters),
            'cluster_sizes': [len(c) for c in sorted_clusters],
            'total_clustered': sum(len(c) for c in sorted_clusters)
        }
    
    def _save_pucker_results(self, pucker_type: str, results: Dict[float, Any]) -> None:
        """Save results for a pucker type."""
        output_file = self.output_dir / f"{pucker_type}_q_fold_results.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
    
    def find_optimal_q_fold(self, 
                           pucker_type: str,
                           criterion: str = 'max_clusters') -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal q-fold value for a pucker type.
        
        Args:
            pucker_type: Pucker type to analyze
            criterion: Optimization criterion ('max_clusters', 'balance', etc.)
            
        Returns:
            Tuple of (optimal_q_fold, result_info)
        """
        if pucker_type not in self.results:
            raise ValueError(f"No results found for {pucker_type}")
        
        pucker_results = self.results[pucker_type]
        
        if criterion == 'max_clusters':
            # Find q-fold that gives maximum number of clusters
            best_qf = max(pucker_results.keys(), 
                         key=lambda qf: pucker_results[qf]['n_clusters'])
        
        elif criterion == 'balance':
            # Find q-fold that balances number of clusters and cluster sizes
            def balance_score(qf):
                result = pucker_results[qf]
                n_clusters = result['n_clusters']
                if n_clusters == 0:
                    return 0
                
                cluster_sizes = result['cluster_sizes']
                size_std = np.std(cluster_sizes) if len(cluster_sizes) > 1 else 0
                # Prefer more clusters with more balanced sizes
                return n_clusters * (1 / (1 + size_std))
            
            best_qf = max(pucker_results.keys(), key=balance_score)
        
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return best_qf, pucker_results[best_qf]
    
    def generate_q_fold_summary(self) -> Dict[str, Any]:
        """Generate summary of q-fold experiments."""
        summary = {
            'pucker_summaries': {},
            'optimal_q_folds': {},
            'overall_statistics': {}
        }
        
        total_experiments = 0
        
        for pucker_type, results in self.results.items():
            # Find optimal q-fold
            optimal_qf, optimal_result = self.find_optimal_q_fold(pucker_type)
            
            summary['pucker_summaries'][pucker_type] = {
                'n_q_fold_values_tested': len(results),
                'optimal_q_fold': optimal_qf,
                'optimal_n_clusters': optimal_result['n_clusters'],
                'optimal_cluster_sizes': optimal_result['cluster_sizes']
            }
            
            summary['optimal_q_folds'][pucker_type] = optimal_qf
            total_experiments += len(results)
        
        summary['overall_statistics'] = {
            'total_experiments': total_experiments,
            'pucker_types_analyzed': len(self.results)
        }
        
        return summary


def run_pucker_analysis(suites: List[Suite],
                       scaled_coords: np.ndarray,
                       pucker_types: Optional[List[str]] = None,
                       output_dir: str = "pucker_analysis",
                       find_optimal_q_fold: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive pucker analysis including q-fold optimization.
    
    Args:
        suites: List of Suite objects
        scaled_coords: Scaled coordinate data
        pucker_types: Pucker types to analyze
        output_dir: Output directory
        find_optimal_q_fold: Whether to run q-fold optimization
        
    Returns:
        Dictionary containing analysis results
    """
    manager = PuckerExperimentManager(output_dir=output_dir)
    
    results = {
        'pucker_statistics': {},
        'clustering_results': {}
    }
    
    # Get basic pucker statistics
    for pucker_type in (pucker_types or list(PUCKER_Q_FOLD_RANGES.keys())):
        indices, pucker_suites = determine_pucker_data(suites, pucker_type)
        results['pucker_statistics'][pucker_type] = {
            'n_suites': len(pucker_suites),
            'percentage': len(pucker_suites) / len(suites) * 100 if suites else 0
        }
    
    # Run q-fold experiments if requested
    if find_optimal_q_fold:
        q_fold_results = manager.run_q_fold_experiments(
            suites, scaled_coords, pucker_types
        )
        results['q_fold_experiments'] = q_fold_results
        results['q_fold_summary'] = manager.generate_q_fold_summary()
    
    return results