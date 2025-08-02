"""
Sugar pucker analysis for RNA structures.

Provides functionality to classify RNA suites based on their sugar pucker
conformations (C2'-endo vs C3'-endo).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Sequence
from ..data.models import Suite
from ..config.experiment_config import PUCKER_TYPES


class PuckerAnalyzer:
    """Analyzer for RNA sugar pucker conformations."""
    
    def __init__(self):
        """Initialize pucker analyzer with classification thresholds."""
        # Nu angle thresholds for C2'-endo classification
        self.c2_endo_min = 300.0
        self.c2_endo_max = 350.0
    
    def classify_suite_pucker(self, suite: Suite) -> Optional[str]:
        """
        Classify a suite's pucker type based on nu angles.
        
        Args:
            suite: Suite object with nu angle data
            
        Returns:
            Pucker type string ('c2c2', 'c2c3', 'c3c2', 'c3c3') or None
        """
        # Check if nu angles are available
        if (suite._nu_1[0] is None or suite._nu_2[0] is None):
            return None
        
        nu1 = suite._nu_1[0]
        nu2 = suite._nu_2[0]
        
        # Classify each sugar based on nu angle
        sugar1_c2 = self.c2_endo_min < nu1 < self.c2_endo_max
        sugar2_c2 = self.c2_endo_min < nu2 < self.c2_endo_max
        
        # Determine pucker combination
        if sugar1_c2 and sugar2_c2:
            return 'c2c2'
        elif sugar1_c2 and not sugar2_c2:
            return 'c2c3'
        elif not sugar1_c2 and sugar2_c2:
            return 'c3c2'
        else:  # not sugar1_c2 and not sugar2_c2
            return 'c3c3'
    
    def get_pucker_distances(self, suite: Suite) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate distances from canonical pucker conformations.
        
        Args:
            suite: Suite object
            
        Returns:
            Tuple of (distance_1, distance_2) from canonical conformations
        """
        if suite._nu_1[0] is None or suite._nu_2[0] is None:
            return None, None
        
        nu1 = suite._nu_1[0]
        nu2 = suite._nu_2[0]
        
        # Calculate distances from canonical C2'-endo (325°) and C3'-endo (162°)
        c2_canonical = 325.0
        c3_canonical = 162.0
        
        # Distance from C2'-endo for each sugar
        dist1_c2 = abs(nu1 - c2_canonical)
        dist2_c2 = abs(nu2 - c2_canonical)
        
        # Distance from C3'-endo for each sugar  
        dist1_c3 = abs(nu1 - c3_canonical)
        dist2_c3 = abs(nu2 - c3_canonical)
        
        # Choose minimum distance for each sugar
        distance_1 = min(dist1_c2, dist1_c3)
        distance_2 = min(dist2_c2, dist2_c3)
        
        return distance_1, distance_2


def determine_pucker_data(suites: Sequence[Suite], pucker_name: str) -> Tuple[List[int], List[Suite]]:
    """
    Determine suites belonging to a specific sugar pucker type.
    
    This is the main function that matches the original interface from
    the notebook experiments.
    
    Args:
        suites: Sequence of Suite objects
        pucker_name: Pucker type ('c2c2', 'c3c3', 'c2c3', 'c3c2', or 'all')
        
    Returns:
        Tuple of (indices, filtered_suites) for the specified pucker type
    """
    analyzer = PuckerAnalyzer()
    
    if pucker_name == 'all':
        indices = list(range(len(suites)))
        return indices, list(suites)
    
    if pucker_name not in PUCKER_TYPES:
        print(f"Warning: Unknown pucker name '{pucker_name}'. Valid types: {PUCKER_TYPES}")
        return [], []
    
    # Filter suites by pucker type
    pucker_data = []
    
    for i, suite in enumerate(suites):
        if suite._nu_1[0] is None or suite._nu_2[0] is None:
            continue
        
        suite_pucker = analyzer.classify_suite_pucker(suite)
        if suite_pucker == pucker_name:
            pucker_data.append([i, suite])
    
    if len(pucker_data) == 0:
        return [], []
    
    pucker_array = np.array(pucker_data, dtype=object)
    indices = pucker_array[:, 0].astype(int).tolist()
    filtered_suites = pucker_array[:, 1].tolist()
    
    return indices, filtered_suites


def sort_data_into_cluster(suite_data: np.ndarray, 
                          cluster_list: List[List[int]], 
                          min_cluster_length: int) -> Tuple[np.ndarray, List[int]]:
    """
    Sort suite data into clusters, filtering by minimum cluster size.
    
    Args:
        suite_data: Array of suite coordinate data
        cluster_list: List of clusters (each cluster is list of indices)
        min_cluster_length: Minimum cluster size to include
        
    Returns:
        Tuple of (sorted_data, cluster_lengths)
    """
    data_sorted_by_cluster = np.array([])
    cluster_len_list = []
    
    for cluster in cluster_list:
        if len(cluster) <= min_cluster_length:
            continue
        
        cluster_data = suite_data[cluster]
        
        if data_sorted_by_cluster.size == 0:
            data_sorted_by_cluster = cluster_data
        else:
            data_sorted_by_cluster = np.vstack([data_sorted_by_cluster, cluster_data])
        
        cluster_len_list.append(len(cluster))
    
    return data_sorted_by_cluster, cluster_len_list


def get_pucker_statistics(suites: Sequence[Suite]) -> Dict[str, int]:
    """
    Get statistics on pucker type distribution.
    
    Args:
        suites: Sequence of Suite objects
        
    Returns:
        Dictionary with counts for each pucker type
    """
    analyzer = PuckerAnalyzer()
    stats = {pucker: 0 for pucker in PUCKER_TYPES}
    stats['unknown'] = 0
    
    for suite in suites:
        pucker_type = analyzer.classify_suite_pucker(suite)
        if pucker_type:
            stats[pucker_type] += 1
        else:
            stats['unknown'] += 1
    
    return stats