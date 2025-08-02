"""
Hierarchical clustering for low-resolution RNA data.

Implements the pre-clustering algorithm used in the experimental pipeline,
with proper distance metrics and outlier detection.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform

from ..utils.io_utils import create_output_directory


class HierarchicalClusterer:
    """Hierarchical clustering with outlier removal."""
    
    def __init__(self, distance_metric: str = "low_res_suite_shape"):
        """
        Initialize hierarchical clusterer.
        
        Args:
            distance_metric: Distance metric to use
        """
        self.distance_metric = distance_metric
    
    def cluster(self, 
                input_data: np.ndarray,
                min_cluster_size: int,
                outlier_percentage: float,
                q_fold: float,
                linkage_method: Callable,
                output_dir: Optional[str] = None) -> Tuple[List[List[int]], List[int], float]:
        """
        Perform hierarchical clustering with outlier removal.
        
        Args:
            input_data: Data array to cluster
            min_cluster_size: Minimum cluster size
            outlier_percentage: Maximum outlier percentage
            q_fold: Q-fold parameter for distance threshold
            linkage_method: Scipy linkage method (average, single, etc.)
            output_dir: Optional output directory for plots
            
        Returns:
            Tuple of (clusters, outliers, distance_threshold)
        """
        # Prepare data
        if self.distance_metric != 'torus':
            try:
                input_data = input_data.reshape((input_data.shape[0], -1))
            except IndexError:
                print("no shape_space")
                pass
        
        n = input_data.shape[0]
        dimension_number = input_data.shape[1]
        
        # Calculate initial distance matrix and clustering
        distance_matrix = self._calculate_distance_matrix(input_data)
        linkage_result = linkage_method(distance_matrix)
        
        # Find outlier threshold
        d_max = self._find_outlier_threshold(
            linkage_result, outlier_percentage, n, min_cluster_size
        )
        
        print(f"Distance threshold: {d_max}")
        
        # Iterative clustering with outlier removal
        cluster_points = input_data.copy()
        outlier_list = []
        cluster_list = []
        counter = 0
        current_n = n
        
        while current_n > 0:
            # Recalculate distances for remaining points
            points_reshaped = cluster_points.reshape(current_n, dimension_number)
            distance_points = self._calculate_distance_matrix(points_reshaped)
            cluster_tree = linkage_method(distance_points)
            
            # Form clusters at threshold
            f_cluster = fcluster(cluster_tree, d_max, criterion='distance')
            
            # Identify outlier clusters (too small)
            outlier_cluster_indices = [
                i for i in range(1, max(f_cluster) + 1) 
                if sum(f_cluster == i) < min_cluster_size
            ]
            
            # Extract outliers
            outlier_points = [
                cluster_points[i] for i in range(current_n) 
                if f_cluster[i] in outlier_cluster_indices
            ]
            
            # Keep non-outlier points
            valid_indices = [
                i for i in range(current_n) 
                if f_cluster[i] not in outlier_cluster_indices
            ]
            
            cluster_points = cluster_points[valid_indices]
            current_n = cluster_points.shape[0]
            
            print(f"Iteration {counter}: {current_n} points remaining")
            
            if current_n > 0:
                outlier_list.extend(outlier_points)
                print(f"Total outliers: {len(outlier_list)}")
            
            # Check for infinite loop
            counter += 1
            if counter > 100:
                print("Warning: Maximum iterations reached")
                break
        
        # Final clustering on remaining points
        if current_n > 0:
            final_distance = self._calculate_distance_matrix(cluster_points)
            final_linkage = linkage_method(final_distance)
            final_clusters = fcluster(final_linkage, d_max * q_fold, criterion='distance')
            
            # Convert to cluster lists
            cluster_list = self._extract_cluster_lists(final_clusters, cluster_points, input_data)
        
        return cluster_list, outlier_list, d_max
    
    def _calculate_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate distance matrix based on the specified metric."""
        if self.distance_metric == "low_res_suite_shape":
            # Use Euclidean distance for low-resolution coordinates
            distances = pdist(data, metric='euclidean')
            return squareform(distances)
        else:
            # Default to Euclidean
            distances = pdist(data, metric='euclidean')
            return squareform(distances)
    
    def _find_outlier_threshold(self, linkage_result: np.ndarray, 
                              percentage: float, n: int, m: int) -> float:
        """Find the distance threshold for outlier detection."""
        # Simple threshold based on linkage distances
        distances = linkage_result[:, 2]
        sorted_distances = np.sort(distances)
        
        # Use percentile of distances as threshold
        if percentage > 0:
            threshold_idx = int((1 - percentage) * len(sorted_distances))
            d_max = sorted_distances[threshold_idx]
        else:
            # Use median distance
            d_max = np.median(sorted_distances)
        
        return d_max
    
    def _extract_cluster_lists(self, cluster_labels: np.ndarray, 
                             cluster_points: np.ndarray, 
                             original_data: np.ndarray) -> List[List[int]]:
        """Extract cluster lists as indices into original data."""
        # Create mapping from points back to original indices
        index_map = {}
        for i, point in enumerate(original_data):
            key = tuple(point.flatten())
            index_map[key] = i
        
        # Group by cluster labels
        unique_labels = np.unique(cluster_labels)
        cluster_lists = []
        
        for label in unique_labels:
            cluster_indices = []
            label_mask = cluster_labels == label
            
            for point in cluster_points[label_mask]:
                key = tuple(point.flatten())
                if key in index_map:
                    cluster_indices.append(index_map[key])
            
            if cluster_indices:
                cluster_lists.append(cluster_indices)
        
        return cluster_lists


def pre_clustering(input_data: np.ndarray,
                  m: int,
                  percentage: float,
                  string_folder: str,
                  method: Callable,
                  q_fold: float,
                  distance: str = 'torus') -> Tuple[List[List[int]], List[int], float]:
    """
    Pre-clustering function that matches the original interface.
    
    Args:
        input_data: Data array to cluster
        m: Minimum cluster size
        percentage: Outlier percentage
        string_folder: Output folder (for compatibility)
        method: Scipy linkage method
        q_fold: Q-fold parameter
        distance: Distance metric
        
    Returns:
        Tuple of (clusters, outliers, distance_threshold)
    """
    clusterer = HierarchicalClusterer(distance_metric=distance)
    
    return clusterer.cluster(
        input_data=input_data,
        min_cluster_size=m,
        outlier_percentage=percentage,
        q_fold=q_fold,
        linkage_method=method,
        output_dir=string_folder
    )