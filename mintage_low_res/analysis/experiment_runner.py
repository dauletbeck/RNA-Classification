"""
Main experiment runner for low-resolution RNA analysis.

Orchestrates the complete experimental pipeline from PDB parsing
through clustering and result generation.
"""

import os
import time
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from ..data.models import Suite
from ..data.cache import CacheManager
from ..config.experiment_config import ExperimentConfig, PUCKER_TYPES
from ..parsing.pdb_parser import parse_pdb_files
from ..preprocessing.coordinate_scaling import scale_coordinates
from ..preprocessing.pucker_analysis import determine_pucker_data
from ..clustering.hierarchical_clustering import pre_clustering
from ..clustering.cluster_refinement import refine_clusters_with_pns
from ..clustering.pns_clustering import PNSClusterer
from ..utils.io_utils import create_output_directory


class ExperimentRunner:
    """Orchestrates low-resolution RNA analysis experiments."""
    
    def __init__(self, 
                 config: Optional[ExperimentConfig] = None,
                 cache_dir: str = "cache",
                 output_dir: str = "results"):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            cache_dir: Directory for caching intermediate results
            output_dir: Directory for output files
        """
        self.config = config or ExperimentConfig()
        self.cache_manager = CacheManager(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.timings = {}
    
    def run_complete_pipeline(self, 
                            pdb_directory: str,
                            pucker_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete low-resolution analysis pipeline.
        
        Args:
            pdb_directory: Directory containing PDB files
            pucker_types: List of pucker types to analyze (default: all)
            
        Returns:
            Dictionary containing all experiment results
        """
        start_time = time.time()
        
        pucker_types = pucker_types or self.config.pucker_types
        
        print("=== Low-Resolution RNA Analysis Pipeline ===")
        print(f"PDB Directory: {pdb_directory}")
        print(f"Pucker Types: {pucker_types}")
        print(f"Min Cluster Size: {self.config.min_cluster_size}")
        print(f"PNS Scale: {self.config.pns_scale}")
        
        # Step 1: Parse PDB files
        print("\n1. Parsing PDB files...")
        suites = self._parse_pdb_files(pdb_directory)
        print(f"   Parsed {len(suites)} suites")
        
        # Step 2: Scale coordinates
        print("\n2. Scaling coordinates...")
        scaled_coords, lambda_d, lambda_alpha = self._scale_coordinates(suites)
        print(f"   Scaling factors: λ_d={lambda_d:.3f}, λ_α={lambda_alpha:.3f}")
        
        # Step 3: Analyze each pucker type
        print("\n3. Analyzing pucker types...")
        pucker_results = {}
        
        for pucker_type in pucker_types:
            print(f"\n   Analyzing {pucker_type}...")
            result = self._analyze_pucker_type(
                suites, scaled_coords, pucker_type
            )
            pucker_results[pucker_type] = result
            print(f"   {pucker_type}: {len(result['mode_clusters'])} final clusters")
        
        # Compile final results
        total_time = time.time() - start_time
        
        self.results = {
            'suites': suites,
            'scaled_coordinates': scaled_coords,
            'scaling_factors': {'lambda_d': lambda_d, 'lambda_alpha': lambda_alpha},
            'pucker_results': pucker_results,
            'config': self.config,
            'timings': self.timings,
            'total_time': total_time,
            'summary': self._generate_summary(pucker_results)
        }
        
        print(f"\n=== Pipeline Complete ({total_time:.1f}s) ===")
        return self.results
    
    def _parse_pdb_files(self, pdb_directory: str) -> List[Suite]:
        """Parse PDB files with caching."""
        cache_key = f"suites_{Path(pdb_directory).name}"
        print(f"suites_{Path(pdb_directory)}")
        
        def parse_func():
            return parse_pdb_files(pdb_directory, cache_dir=self.cache_manager.cache_dir)
        
        start_time = time.time()
        suites = self.cache_manager.load(cache_key)
        if suites is None:
            suites = parse_func()
            self.cache_manager.save(cache_key, suites)
        
        self.timings['parsing'] = time.time() - start_time
        return suites
    
    def _scale_coordinates(self, suites: List[Suite]) -> Tuple[Any, float, float]:
        """Scale coordinates with caching."""
        cache_key = "scaled_coordinates"
        
        def scale_func():
            return scale_coordinates(
                suites,
                scale_distance_variance=True,
                scale_alpha_variance=False,
                preserve_distance_mean=True,
                preserve_alpha_mean=True
            )
        
        start_time = time.time()
        cached_result = self.cache_manager.load(cache_key)
        if cached_result is None:
            result = scale_func()
            self.cache_manager.save(cache_key, result)
        else:
            result = cached_result
        
        self.timings['scaling'] = time.time() - start_time
        return result
    
    def _analyze_pucker_type(self, 
                           suites: List[Suite], 
                           scaled_coords: Any, 
                           pucker_type: str) -> Dict[str, Any]:
        """Analyze a specific pucker type."""
        start_time = time.time()
        
        # Get pucker-specific data
        pucker_indices, pucker_suites = determine_pucker_data(suites, pucker_type)
        
        if not pucker_indices:
            return {
                'pucker_indices': [],
                'scaled_coords_subset': None,
                'hierarchical_clusters': [],
                'mode_clusters': [],
                'metadata': {'error': 'No suites found for this pucker type'}
            }
        
        scaled_coords_subset = scaled_coords[pucker_indices]
        
        # Run hierarchical clustering with optimal q_fold
        optimal_q_fold = self.config.get_optimal_q_fold(pucker_type)
        
        from scipy.cluster.hierarchy import single as single_linkage
        
        clusters, outliers, distance_threshold = pre_clustering(
            input_data=scaled_coords_subset,
            m=self.config.min_cluster_size,
            percentage=0.0,
            string_folder=str(self.output_dir / "clustering"),
            method=single_linkage,
            q_fold=optimal_q_fold,
            distance="low_res_suite_shape"
        )
        
        # Prepare data for PNS clustering
        pns_clusterer = PNSClusterer(scale=self.config.pns_scale)
        angle_matrix = pns_clusterer.prepare_angle_matrix(scaled_coords_subset)
        
        # Refine clusters with PNS
        mode_clusters, refinement_metadata = refine_clusters_with_pns(
            scale=self.config.pns_scale,
            data=angle_matrix,
            cluster_list=clusters,
            outlier_list=outliers,
            min_cluster_size=self.config.min_cluster_size
        )
        
        analysis_time = time.time() - start_time
        self.timings[f'analysis_{pucker_type}'] = analysis_time
        
        return {
            'pucker_indices': pucker_indices,
            'scaled_coords_subset': scaled_coords_subset,
            'angle_matrix': angle_matrix,
            'hierarchical_clusters': clusters,
            'outliers': outliers,
            'mode_clusters': mode_clusters,
            'distance_threshold': distance_threshold,
            'optimal_q_fold': optimal_q_fold,
            'metadata': {
                'n_suites': len(pucker_suites),
                'n_hierarchical_clusters': len(clusters),
                'n_mode_clusters': len(mode_clusters),
                'analysis_time': analysis_time,
                **refinement_metadata
            }
        }
    
    def _generate_summary(self, pucker_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'pucker_statistics': {},
            'cluster_statistics': {},
            'timing_summary': dict(self.timings)
        }
        
        for pucker_type, result in pucker_results.items():
            if 'metadata' in result:
                meta = result['metadata']
                summary['pucker_statistics'][pucker_type] = {
                    'n_suites': meta.get('n_suites', 0),
                    'n_final_clusters': meta.get('n_mode_clusters', 0),
                    'analysis_time': meta.get('analysis_time', 0)
                }
        
        total_clusters = sum(
            stats.get('n_final_clusters', 0) 
            for stats in summary['pucker_statistics'].values()
        )
        
        summary['cluster_statistics'] = {
            'total_final_clusters': total_clusters,
            'average_clusters_per_pucker': total_clusters / len(pucker_results) if pucker_results else 0
        }
        
        return summary
    
    def save_results(self, filename: str = "experiment_results.pkl") -> None:
        """Save experiment results to file."""
        output_path = self.output_dir / filename
        self.cache_manager.save(str(output_path.stem), self.results)
        print(f"Results saved to {output_path}")


def run_low_res_experiments(pdb_directory: str,
                          pucker_types: Optional[List[str]] = None,
                          config: Optional[ExperimentConfig] = None,
                          cache_dir: str = "cache",
                          output_dir: str = "results") -> Dict[str, Any]:
    """
    Run low-resolution RNA analysis experiments.
    
    This is the main entry point function that matches the interface
    expected by the clean notebook.
    
    Args:
        pdb_directory: Directory containing PDB files
        pucker_types: List of pucker types to analyze
        config: Experiment configuration
        cache_dir: Cache directory
        output_dir: Output directory
        
    Returns:
        Dictionary containing experiment results
    """
    runner = ExperimentRunner(
        config=config,
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    return runner.run_complete_pipeline(pdb_directory, pucker_types)