"""
Geometric utility functions for RNA structure analysis.

Contains dihedral angle calculations, rotations, and spherical transformations
used in the low-resolution pipeline.
"""

import numpy as np
from math import atan2, pi, sqrt, cos, sin
from typing import List, Union


def dihedral(point_list: List[np.ndarray], verbose: bool = False, 
             rna_distances: bool = True, long: bool = False) -> float:
    """
    Calculate dihedral angle from four 3D points.
    
    Args:
        point_list: List of 4 points in R^3
        verbose: Print debug information
        rna_distances: Check for reasonable RNA distances
        long: Use larger distance tolerance
        
    Returns:
        Dihedral angle in degrees, or None if invalid
    """
    if len(point_list) != 4:
        return None
        
    # Calculate bond vectors
    b = [_diff(point_list[i], point_list[i + 1]) for i in range(3)]
    
    # Check distances for RNA structures
    for bi in b:
        distance_sq = _dot(bi, bi)
        if distance_sq > (20 if long else 3) and rna_distances:
            if verbose:
                print(f'Atoms too far apart: {distance_sq}')
            return None
    
    # Calculate cross products
    c = [_cross(b[i], b[i + 1]) for i in range(2)]
    
    # Calculate dihedral angle
    angle = atan2(_dot(_cross(c[0], c[1]), _normalize(b[1])), _dot(c[0], c[1]))
    
    return (360 - angle * (180 / pi)) % 360


def arc_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Calculate shortest signed arc distance between angles."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def spherical_to_vec(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Convert spherical angles to 3D unit vectors."""
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])


def exponential_map(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Map points from tangent space to sphere using exponential map.
    
    Args:
        V: Point cloud N x R^m
        p: Point of tangency R^m
        
    Returns:
        Points mapped to sphere
    """
    N, M = V.shape
    V_mean = V.mean(axis=0)
    V_centered = V - V_mean
    V_padded = np.column_stack([V_centered, np.zeros(N)])
    V_norm = np.linalg.norm(V_padded, axis=1)[:, None]
    
    return np.cos(V_norm) * p + np.sin(V_norm) * (V_padded / V_norm)


def rotation(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate rotation matrix that rotates v1 to v2.
    
    Args:
        v1: Source vector
        v2: Target vector
        
    Returns:
        3x3 rotation matrix
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # If vectors are already aligned
    if np.allclose(v1_norm, v2_norm):
        return np.eye(3)
    
    # If vectors are opposite
    if np.allclose(v1_norm, -v2_norm):
        # Find perpendicular vector
        if abs(v1_norm[0]) < 0.9:
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([0, 1, 0])
        perp = perp - np.dot(perp, v1_norm) * v1_norm
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    
    # Rodrigues' rotation formula
    v = np.cross(v1_norm, v2_norm)
    s = np.linalg.norm(v)
    c = np.dot(v1_norm, v2_norm)
    
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    
    return np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))


# Helper functions for dihedral calculation
def _diff(p: List[float], q: List[float]) -> List[float]:
    """Calculate p - q for 3D points."""
    return [p[i] - q[i] for i in range(3)]


def _cross(p: List[float], q: List[float]) -> List[float]:
    """Calculate cross product of two 3D vectors."""
    return [
        p[1] * q[2] - p[2] * q[1],
        p[2] * q[0] - p[0] * q[2],
        p[0] * q[1] - p[1] * q[0]
    ]


def _dot(p: List[float], q: List[float]) -> float:
    """Calculate dot product of two 3D vectors."""
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


def _normalize(v: List[float]) -> List[float]:
    """Normalize a 3D vector."""
    norm = sqrt(_dot(v, v))
    return [x / norm for x in v]