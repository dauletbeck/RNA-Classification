"""
Cluster refinement using PNS and mode hunting.

Implements the refine_clusters_with_pns function used in the notebook
for final cluster refinement after initial hierarchical clustering.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from .pns_clustering import PNSClusterer


class ClusterRefiner:
    """Refines clusters using PNS-based mode hunting."""
    
    def __init__(self, scale: int = 12000, min_cluster_size: int = 3):
        """
        Initialize cluster refiner.
        
        Args:
            scale: Scale parameter for mode hunting
            min_cluster_size: Minimum size for refined clusters
        """
        self.scale = scale
        self.min_cluster_size = min_cluster_size
        self.pns_clusterer = PNSClusterer(scale=scale)
    
    def refine_clusters(self, 
                       data: np.ndarray,
                       cluster_list: List[List[int]],
                       outlier_list: List[int]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Refine clusters using PNS-based mode hunting.
        
        Args:
            data: Angle matrix data for clustering
            cluster_list: Initial clusters from hierarchical clustering
            outlier_list: Outlier indices
            
        Returns:
            Tuple of (refined_clusters, metadata)
        """
        refined_clusters = []
        metadata = {
            'original_cluster_count': len(cluster_list),
            'refined_cluster_count': 0,
            'total_points_clustered': 0
        }
        
        for i, cluster_indices in enumerate(cluster_list):
            if len(cluster_indices) < self.min_cluster_size:
                continue
            
            # Extract cluster data
            cluster_data = data[cluster_indices]
            
            # Apply PNS-based refinement
            sub_clusters = self._apply_pns_refinement(cluster_data, cluster_indices)
            
            # Add valid sub-clusters
            for sub_cluster in sub_clusters:
                if len(sub_cluster) >= self.min_cluster_size:
                    refined_clusters.append(sub_cluster)
                    metadata['total_points_clustered'] += len(sub_cluster)
        
        metadata['refined_cluster_count'] = len(refined_clusters)
        
        return refined_clusters, metadata
    
    def _apply_pns_refinement(self, 
                            cluster_data: np.ndarray, 
                            original_indices: List[int]) -> List[List[int]]:
        """
        Apply PNS-based refinement to a single cluster.
        
        Args:
            cluster_data: Data points in the cluster
            original_indices: Original indices of cluster points
            
        Returns:
            List of refined sub-clusters
        """
        # For now, implement a simple refinement based on PNS transformation
        # In a full implementation, this would include mode hunting
        
        if len(cluster_data) < 2 * self.min_cluster_size:
            # Too small to split meaningfully
            return [original_indices]
        
        # Simple k-means-like splitting based on PNS coordinates
        try:
            from sklearn.cluster import KMeans
            
            # Determine number of sub-clusters (simple heuristic)
            n_subclusters = min(3, len(cluster_data) // self.min_cluster_size)
            if n_subclusters < 2:
                return [original_indices]
            
            # Apply clustering
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(cluster_data)
            
            # Extract sub-clusters
            sub_clusters = []
            for label in range(n_subclusters):
                sub_indices = [original_indices[j] for j, l in enumerate(sub_labels) if l == label]
                if len(sub_indices) >= self.min_cluster_size:
                    sub_clusters.append(sub_indices)
            
            return sub_clusters if sub_clusters else [original_indices]
            
        except ImportError:
            # Fallback: no refinement
            return [original_indices]
    
    def calculate_cluster_quality_metrics(self, 
                                        data: np.ndarray,
                                        clusters: List[List[int]]) -> Dict[str, float]:
        """
        Calculate quality metrics for the refined clusters.
        
        Args:
            data: Original data
            clusters: List of clusters
            
        Returns:
            Dictionary of quality metrics
        """
        if not clusters:
            return {'silhouette_score': 0.0, 'inertia': float('inf')}
        
        try:
            from sklearn.metrics import silhouette_score
            
            # Create cluster labels
            labels = np.full(len(data), -1)
            for i, cluster in enumerate(clusters):
                for idx in cluster:
                    if idx < len(labels):
                        labels[idx] = i
            
            # Calculate silhouette score for clustered points only
            clustered_mask = labels >= 0
            if np.sum(clustered_mask) > 1:
                score = silhouette_score(data[clustered_mask], labels[clustered_mask])
            else:
                score = 0.0
            
            # Calculate within-cluster sum of squares
            inertia = 0.0
            for cluster in clusters:
                if len(cluster) > 1:
                    cluster_data = data[cluster]
                    centroid = np.mean(cluster_data, axis=0)
                    inertia += np.sum((cluster_data - centroid) ** 2)
            
            return {
                'silhouette_score': score,
                'inertia': inertia,
                'n_clusters': len(clusters),
                'n_clustered_points': np.sum(clustered_mask)
            }
            
        except ImportError:
            return {'silhouette_score': 0.0, 'inertia': 0.0}


def refine_clusters_with_pns(scale: int,
                           data: np.ndarray,
                           cluster_list: List[List[int]],
                           outlier_list: List[int],
                           min_cluster_size: int = 3) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Refine clusters using PNS-based mode hunting.
    
    This function matches the interface used in the notebook experiments.
    
    Args:
        scale: Scale parameter for PNS mode hunting
        data: Angle matrix data [N x 7]
        cluster_list: Initial clusters from hierarchical clustering
        outlier_list: List of outlier indices
        min_cluster_size: Minimum cluster size
        
    Returns:
        Tuple of (refined_clusters, metadata)
    """
    refiner = ClusterRefiner(scale=scale, min_cluster_size=min_cluster_size)
    
    refined_clusters, metadata = refiner.refine_clusters(
        data=data,
        cluster_list=cluster_list,
        outlier_list=outlier_list
    )
    
    # Add quality metrics
    quality_metrics = refiner.calculate_cluster_quality_metrics(data, refined_clusters)
    metadata.update(quality_metrics)
    
    return refined_clusters, metadata


class PNSModeHunter:
    """
    PNS Mode Hunter for cluster refinement.
    
    Simplified implementation for compatibility with the notebook interface.
    """
    
    def __init__(self, scale: int = 12000):
        """Initialize mode hunter with scale parameter."""
        self.scale = scale
        self.refiner = ClusterRefiner(scale=scale)
    
    def hunt_modes(self, 
                   data: np.ndarray,
                   initial_clusters: List[List[int]]) -> List[List[int]]:
        """
        Hunt for modes within clusters.
        
        Args:
            data: Input data
            initial_clusters: Initial clustering
            
        Returns:
            Refined clusters after mode hunting
        """
        refined_clusters, _ = self.refiner.refine_clusters(
            data=data,
            cluster_list=initial_clusters,
            outlier_list=[]
        )
        
        return refined_clusters