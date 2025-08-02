"""
Principal Nested Spheres (PNS) clustering for RNA data.

Implements PNS-based clustering and analysis for low-resolution RNA coordinates,
including the necessary geometric transformations.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class PNS:
    """
    Principal Nested Spheres implementation.
    
    Simplified version focused on the functionality needed for the
    low-resolution pipeline.
    """
    
    def __init__(self, mode: str = 'great', verbose: bool = False):
        """
        Initialize PNS with specified mode.
        
        Args:
            mode: PNS mode ('great' for great circles)
            verbose: Whether to print debug information
        """
        self.mode = mode
        self.verbose = verbose
        self.dists_ = None
        self.center_ = None
        self.fitted_ = False
    
    def fit(self, X: np.ndarray) -> 'PNS':
        """
        Fit PNS to the data.
        
        Args:
            X: Data points on sphere (N x 3)
            
        Returns:
            Self for chaining
        """
        if X.shape[1] != 3:
            raise ValueError("PNS expects 3D points on sphere")
        
        # Calculate spherical mean as center
        self.center_ = self._spherical_mean(X)
        
        # Project to tangent space and get principal components
        tangent_coords = self._project_to_tangent_space(X, self.center_)
        
        # PCA on tangent space
        U, s, Vt = np.linalg.svd(tangent_coords, full_matrices=False)
        
        # Project back to angles
        if self.mode == 'great':
            # For great circle mode, use first two principal components
            projected = U[:, :2] @ np.diag(s[:2])
            self.dists_ = self._tangent_to_angles(projected)
        else:
            # Default behavior
            self.dists_ = self._tangent_to_angles(tangent_coords[:, :2])
        
        self.fitted_ = True
        return self
    
    def _spherical_mean(self, X: np.ndarray) -> np.ndarray:
        """Calculate mean direction on sphere."""
        mean_vec = np.mean(X, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-10:
            return np.array([0., 0., 1.])  # North pole default
        return mean_vec / norm
    
    def _project_to_tangent_space(self, X: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Project sphere points to tangent space at center."""
        # Logarithmic map (inverse exponential map)
        dots = np.dot(X, center)
        dots = np.clip(dots, -1.0, 1.0)
        
        angles = np.arccos(np.abs(dots))
        
        # Project to tangent space
        tangent_vecs = np.zeros_like(X)
        nonzero_mask = angles > 1e-10
        
        if np.any(nonzero_mask):
            projected = X[nonzero_mask] - dots[nonzero_mask, None] * center
            proj_norms = np.linalg.norm(projected, axis=1)
            
            valid_mask = proj_norms > 1e-10
            if np.any(valid_mask):
                normalized = projected[valid_mask] / proj_norms[valid_mask, None]
                full_valid = np.zeros(len(nonzero_mask), dtype=bool)
                full_valid[nonzero_mask] = valid_mask
                tangent_vecs[full_valid] = normalized * angles[nonzero_mask][valid_mask, None]
        
        return tangent_vecs
    
    def _tangent_to_angles(self, tangent_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert tangent space coordinates to angles."""
        # Simple conversion to theta, phi angles
        x, y = tangent_coords[:, 0], tangent_coords[:, 1]
        
        theta = np.degrees(np.arctan2(y, x))
        phi = np.degrees(np.sqrt(x**2 + y**2))
        
        return theta, phi


class PNSClusterer:
    """PNS-based clustering for RNA structural data."""
    
    def __init__(self, scale: int = 12000):
        """
        Initialize PNS clusterer.
        
        Args:
            scale: Scale parameter for mode hunting
        """
        self.scale = scale
    
    def fit_pns_to_spherical_data(self, theta_deg: np.ndarray, phi_deg: np.ndarray) -> PNS:
        """
        Fit PNS to spherical coordinate data.
        
        Args:
            theta_deg: Polar angles in degrees
            phi_deg: Azimuthal angles in degrees
            
        Returns:
            Fitted PNS object
        """
        from ..preprocessing.spherical_transforms import spherical_to_vec
        
        # Convert to 3D unit vectors
        vectors = spherical_to_vec(theta_deg, phi_deg)
        
        # Fit PNS
        pns = PNS(mode='great', verbose=True)
        pns.fit(vectors)
        
        return pns
    
    def transform_distance_coordinates(self, d2: np.ndarray, d3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform distance coordinates using exponential map to sphere.
        
        Args:
            d2, d3: Distance coordinates
            
        Returns:
            Tuple of (theta_d, phi_d) after PNS transformation
        """
        from ..preprocessing.spherical_transforms import exponential_map
        
        # Combine and center distances
        V = np.column_stack([d2, d3])
        V -= np.mean(V, axis=0)
        
        # Map to sphere
        S2_d = exponential_map(V, p=np.array([0, 0, 1]))
        
        # Fit PNS
        pns = PNS(mode='great', verbose=True)
        pns.fit(S2_d)
        
        theta_d, phi_d = pns.dists_
        return theta_d, phi_d
    
    def prepare_angle_matrix(self, scaled_coords: np.ndarray) -> np.ndarray:
        """
        Prepare angle matrix for PNS-based clustering.
        
        This follows the transformation pipeline from the notebook.
        
        Args:
            scaled_coords: Scaled coordinate array [d2, d3, alpha, theta1, phi1, theta2, phi2]
            
        Returns:
            Transformed angle matrix ready for clustering
        """
        d2_s, d3_s, alpha_s, theta1, phi1, theta2, phi2 = scaled_coords.T
        
        # Transform spherical coordinates through PNS
        pns_S2_1 = self.fit_pns_to_spherical_data(theta1, phi1)
        theta1_new, phi1_new = pns_S2_1.dists_
        
        pns_S2_2 = self.fit_pns_to_spherical_data(theta2, phi2)
        theta2_new, phi2_new = pns_S2_2.dists_
        
        # Transform distance coordinates
        theta_d, phi_d = self.transform_distance_coordinates(d2_s, d3_s)
        
        # Combine into angle matrix (add 180 for plotting as in notebook)
        angle_matrix = np.column_stack([
            theta_d + 180,
            phi_d + 180,
            alpha_s,
            theta1_new + 180,
            phi1_new + 180,
            theta2_new + 180,
            phi2_new + 180
        ])
        
        return angle_matrix


def unfold_points(points: np.ndarray, spheres: List[np.ndarray]) -> np.ndarray:
    """
    Unfold points through nested spheres.
    
    Simplified implementation for compatibility.
    """
    # Simple identity transformation for now
    return points


def as_matrix(data: Any) -> np.ndarray:
    """Convert data to matrix format."""
    if isinstance(data, (list, tuple)):
        return np.array(data)
    return np.asarray(data)