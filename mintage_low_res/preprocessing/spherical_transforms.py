"""
Spherical coordinate transformations and geometry.

Contains functions for working with spherical coordinates and transformations
between different coordinate systems, as used in the notebook experiments.
"""

import numpy as np
from typing import Tuple


def spherical_to_vec(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """
    Convert spherical angles to 3D unit vectors.
    
    Args:
        theta_deg: Polar angles in degrees
        phi_deg: Azimuthal angles in degrees
        
    Returns:
        Array of unit vectors on the sphere
    """
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])


def arc_distance(a: np.ndarray, b: float) -> np.ndarray:
    """
    Calculate shortest signed arc distance between angles.
    
    Args:
        a: Array of angles in radians
        b: Reference angle in radians
        
    Returns:
        Array of signed arc distances
    """
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def exponential_map(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Map points from tangent space to sphere using exponential map.
    
    This function maps a point cloud in the tangent space at point p
    to the sphere using the exponential map.
    
    Args:
        V: Point cloud N x R^m in tangent space
        p: Point of tangency R^m (should be unit vector)
        
    Returns:
        Points mapped to the sphere
    """
    N, M = V.shape
    
    # Center the data
    V_mean = V.mean(axis=0)
    V_centered = V - V_mean
    
    # Pad with zeros to make it 3D if needed
    if M < 3:
        V_padded = np.column_stack([V_centered, np.zeros(N)])
    else:
        V_padded = V_centered
    
    # Calculate norms
    V_norm = np.linalg.norm(V_padded, axis=1)[:, None]
    
    # Avoid division by zero
    nonzero_mask = (V_norm.flatten() > 1e-10)
    result = np.zeros((N, 3))
    
    if np.any(nonzero_mask):
        V_normalized = V_padded[nonzero_mask] / V_norm[nonzero_mask]
        result[nonzero_mask] = (
            np.cos(V_norm[nonzero_mask]) * p + 
            np.sin(V_norm[nonzero_mask]) * V_normalized
        )
    
    # Points with zero norm map to the tangency point
    result[~nonzero_mask] = p
    
    return result


def spherical_mean(vectors: np.ndarray) -> np.ndarray:
    """
    Calculate the mean of unit vectors on the sphere.
    
    Args:
        vectors: Array of unit vectors (N x 3)
        
    Returns:
        Mean direction as unit vector
    """
    mean_vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean_vector)
    
    if norm < 1e-10:
        # If vectors cancel out, return arbitrary direction
        return np.array([1.0, 0.0, 0.0])
    
    return mean_vector / norm


def vec_to_spherical(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D unit vectors to spherical coordinates.
    
    Args:
        vectors: Array of unit vectors (N x 3)
        
    Returns:
        Tuple of (theta_deg, phi_deg) in degrees
    """
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    
    # Avoid numerical issues
    z_clipped = np.clip(z, -1.0, 1.0)
    
    theta_rad = np.arccos(z_clipped)
    phi_rad = np.arctan2(y, x)
    
    theta_deg = np.degrees(theta_rad)
    phi_deg = np.degrees(phi_rad)
    
    return theta_deg, phi_deg


class SphericalProjector:
    """Helper class for working with spherical projections."""
    
    def __init__(self, reference_point: np.ndarray = None):
        """
        Initialize projector with reference point.
        
        Args:
            reference_point: Reference point on sphere (default: north pole)
        """
        if reference_point is None:
            self.reference_point = np.array([0., 0., 1.])
        else:
            self.reference_point = reference_point / np.linalg.norm(reference_point)
    
    def project_to_tangent_space(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project sphere points to tangent space at reference point.
        
        Args:
            vectors: Points on sphere (N x 3)
            
        Returns:
            Points in tangent space
        """
        # Logarithmic map (inverse of exponential map)
        dots = np.dot(vectors, self.reference_point)
        dots = np.clip(dots, -1.0, 1.0)
        
        angles = np.arccos(np.abs(dots))
        
        # Handle the case where vector equals reference point
        nonzero_mask = angles > 1e-10
        tangent_vectors = np.zeros_like(vectors)
        
        if np.any(nonzero_mask):
            # Project onto tangent space
            projected = vectors[nonzero_mask] - dots[nonzero_mask, None] * self.reference_point
            projected_norms = np.linalg.norm(projected, axis=1)
            
            valid_mask = projected_norms > 1e-10
            if np.any(valid_mask):
                normalized = projected[valid_mask] / projected_norms[valid_mask, None]
                tangent_vectors[nonzero_mask][valid_mask] = (
                    normalized * angles[nonzero_mask][valid_mask, None]
                )
        
        return tangent_vectors
    
    def project_from_tangent_space(self, tangent_vectors: np.ndarray) -> np.ndarray:
        """
        Project tangent space points back to sphere.
        
        Args:
            tangent_vectors: Points in tangent space (N x 3)
            
        Returns:
            Points on sphere
        """
        norms = np.linalg.norm(tangent_vectors, axis=1)
        
        # Handle zero vectors
        zero_mask = norms < 1e-10
        sphere_points = np.zeros_like(tangent_vectors)
        sphere_points[zero_mask] = self.reference_point
        
        # Non-zero vectors
        if np.any(~zero_mask):
            normalized = tangent_vectors[~zero_mask] / norms[~zero_mask, None]
            sphere_points[~zero_mask] = (
                np.cos(norms[~zero_mask, None]) * self.reference_point +
                np.sin(norms[~zero_mask, None]) * normalized
            )
        
        return sphere_points