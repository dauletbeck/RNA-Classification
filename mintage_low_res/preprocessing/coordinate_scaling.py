"""
Coordinate scaling for low-resolution RNA data.

Refactored from mintage/utils/scale_low_res_coordinates.py with improved
structure and clear separation of concerns.
"""

import numpy as np
from typing import Sequence, Tuple, List
from ..utils.geometry import arc_distance, spherical_to_vec
from ..data.models import Suite


def circular_mean(angles_rad: np.ndarray) -> Tuple[float, float]:
    """
    Calculate circular mean and variance for angles in radians.
    
    Args:
        angles_rad: Array of angles in radians
        
    Returns:
        Tuple of (mean_angle, variance)
    """
    # Convert to complex numbers on unit circle
    z = np.exp(1j * angles_rad)
    mean_z = np.mean(z)
    
    # Mean angle and variance
    mean_angle = np.angle(mean_z)
    variance = 1 - np.abs(mean_z)
    
    return mean_angle, variance


class CoordinateScaler:
    """Handles scaling of low-resolution coordinates."""
    
    def __init__(self,
                 scale_distance_variance: bool = True,
                 scale_alpha_variance: bool = True,
                 preserve_distance_mean: bool = True,
                 preserve_alpha_mean: bool = True):
        """
        Initialize coordinate scaler.
        
        Args:
            scale_distance_variance: Whether to scale distance coordinates
            scale_alpha_variance: Whether to scale alpha angle
            preserve_distance_mean: Whether to preserve distance means
            preserve_alpha_mean: Whether to preserve alpha mean
        """
        self.scale_distance_variance = scale_distance_variance
        self.scale_alpha_variance = scale_alpha_variance
        self.preserve_distance_mean = preserve_distance_mean
        self.preserve_alpha_mean = preserve_alpha_mean
        
    def scale_coordinates(self, suites: Sequence[Suite]) -> Tuple[np.ndarray, float, float]:
        """
        Scale low-resolution coordinates of Suite objects.
        
        Args:
            suites: Sequence of Suite objects
            
        Returns:
            Tuple of (scaled_coordinates, lambda_d, lambda_alpha)
        """
        # Extract coordinates from suites
        if not suites:
            raise ValueError("No Suite objects provided for coordinate scaling")
            
        coords = np.array([s.low_resolution_coordinates() for s in suites])
        
        # Handle empty coordinates
        if coords.size == 0:
            raise ValueError("No valid coordinates found in Suite objects")
            
        # Ensure coords is 2D
        if coords.ndim == 1:
            if len(coords) < 7:
                raise ValueError(f"Invalid coordinate dimensions: expected 7 values, got {len(coords)}")
            coords = coords.reshape(1, -1)
            
        d2, d3, alpha_deg = coords[:, 0], coords[:, 1], coords[:, 2]
        theta1, phi1 = coords[:, 3], coords[:, 4]
        theta2, phi2 = coords[:, 5], coords[:, 6]
        
        # Calculate Fréchet variances
        var_d, mean_dist = self._calculate_distance_variance(d2, d3)
        var_a, alpha_mu = self._calculate_alpha_variance(alpha_deg)
        var_b1, var_b2 = self._calculate_spherical_variances(theta1, phi1, theta2, phi2)
        
        # Calculate target variances
        v_d_target = (var_b1 + var_b2) / 3
        v_a_target = (var_b1 + var_b2) / 6
        
        # Calculate scale factors
        lambda_d = np.sqrt(v_d_target / var_d) if self.scale_distance_variance else 1.0
        lambda_alpha = np.sqrt(v_a_target / var_a) if self.scale_alpha_variance else 1.0
        
        # Apply scaling
        d2_s, d3_s = self._scale_distances(d2, d3, mean_dist, lambda_d)
        alpha_s = self._scale_alpha(alpha_deg, alpha_mu, lambda_alpha)
        
        # Package scaled coordinates
        scaled_coords = np.column_stack([
            d2_s, d3_s, alpha_s, theta1, phi1, theta2, phi2
        ])
        
        return scaled_coords, lambda_d, lambda_alpha
    
    def _calculate_distance_variance(self, d2: np.ndarray, d3: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate variance for distance coordinates."""
        dist_mat = np.column_stack([d2, d3])
        mean_dist = dist_mat.mean(axis=0)
        var_d = np.mean(np.sum((dist_mat - mean_dist) ** 2, axis=1))
        return var_d, mean_dist
    
    def _calculate_alpha_variance(self, alpha_deg: np.ndarray) -> Tuple[float, float]:
        """Calculate variance for alpha angle."""
        alpha_rad = np.radians(alpha_deg) % (2 * np.pi)
        alpha_mu_mod, var_a = circular_mean(alpha_rad)
        alpha_mu = (alpha_mu_mod + np.pi) % (2 * np.pi) - np.pi
        return var_a, alpha_mu
    
    def _calculate_spherical_variances(self, theta1: np.ndarray, phi1: np.ndarray, 
                                     theta2: np.ndarray, phi2: np.ndarray) -> Tuple[float, float]:
        """Calculate variances for spherical coordinates."""
        vec1 = spherical_to_vec(theta1, phi1)
        vec2 = spherical_to_vec(theta2, phi2)
        
        var_b1 = np.mean(np.sum((vec1 - vec1.mean(axis=0)) ** 2, axis=1))
        var_b2 = np.mean(np.sum((vec2 - vec2.mean(axis=0)) ** 2, axis=1))
        
        return var_b1, var_b2
    
    def _scale_distances(self, d2: np.ndarray, d3: np.ndarray, 
                        mean_dist: np.ndarray, lambda_d: float) -> Tuple[np.ndarray, np.ndarray]:
        """Scale distance coordinates."""
        dist_mat = np.column_stack([d2, d3])
        
        if self.preserve_distance_mean:
            new_dist = mean_dist + lambda_d * (dist_mat - mean_dist)
        else:
            new_dist = lambda_d * dist_mat
            
        return new_dist[:, 0], new_dist[:, 1]
    
    def _scale_alpha(self, alpha_deg: np.ndarray, alpha_mu: float, lambda_alpha: float) -> np.ndarray:
        """Scale alpha angle."""
        if self.preserve_alpha_mean:
            alpha_rad = np.radians(alpha_deg) % (2 * np.pi)
            a_shifted = alpha_mu + lambda_alpha * arc_distance(alpha_rad, alpha_mu)
            a_shifted = (a_shifted + np.pi) % (2 * np.pi) - np.pi
            return np.degrees(a_shifted)
        else:
            return lambda_alpha * alpha_deg


def scale_coordinates(suites: Sequence[Suite],
                     scale_distance_variance: bool = True,
                     scale_alpha_variance: bool = True,
                     preserve_distance_mean: bool = True,
                     preserve_alpha_mean: bool = True,
                     store_attr: str = "scaled_low_res_coords") -> Tuple[np.ndarray, float, float]:
    """
    Scale low-resolution coordinates of Suite objects.
    
    This is the main function that matches the original interface.
    
    Args:
        suites: Sequence of Suite objects
        scale_distance_variance: Whether to scale distance coordinates
        scale_alpha_variance: Whether to scale alpha angle
        preserve_distance_mean: Whether to preserve distance means
        preserve_alpha_mean: Whether to preserve alpha mean
        store_attr: Attribute name to store scaled coordinates in Suite objects
        
    Returns:
        Tuple of (scaled_coordinates, lambda_d, lambda_alpha)
    """
    scaler = CoordinateScaler(
        scale_distance_variance=scale_distance_variance,
        scale_alpha_variance=scale_alpha_variance,
        preserve_distance_mean=preserve_distance_mean,
        preserve_alpha_mean=preserve_alpha_mean
    )
    
    scaled_coords, lambda_d, lambda_alpha = scaler.scale_coordinates(suites)
    
    # Store scaled coordinates in Suite objects
    for suite, row in zip(suites, scaled_coords):
        setattr(suite, store_attr, row.tolist())
    
    return scaled_coords, lambda_d, lambda_alpha