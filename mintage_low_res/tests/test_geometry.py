"""Tests for geometry utility functions."""

import numpy as np
import pytest
from ..utils.geometry import (
    dihedral, arc_distance, spherical_to_vec, 
    exponential_map, rotation
)


class TestGeometry:
    """Test geometric utility functions."""
    
    def test_spherical_to_vec(self):
        """Test spherical to vector conversion."""
        # Test simple cases
        theta_deg = np.array([0, 90, 180])
        phi_deg = np.array([0, 0, 0])
        
        vecs = spherical_to_vec(theta_deg, phi_deg)
        
        assert vecs.shape == (3, 3)
        
        # Test north pole (theta=0)
        np.testing.assert_allclose(vecs[0], [0, 0, 1], atol=1e-10)
        
        # Test equator (theta=90)
        np.testing.assert_allclose(vecs[1], [1, 0, 0], atol=1e-10)
        
        # Test south pole (theta=180)
        np.testing.assert_allclose(vecs[2], [0, 0, -1], atol=1e-10)
    
    def test_arc_distance(self):
        """Test arc distance calculation."""
        # Test zero distance
        angles = np.array([0, np.pi, 2*np.pi])
        ref_angle = 0
        
        distances = arc_distance(angles, ref_angle)
        
        expected = np.array([0, np.pi, 0])
        np.testing.assert_allclose(distances, expected, atol=1e-10)
    
    def test_dihedral_simple(self):
        """Test dihedral angle calculation with simple geometry."""
        # Create four points in a known configuration
        points = [
            [0, 0, 0],  # Origin
            [1, 0, 0],  # Along X
            [1, 1, 0],  # In XY plane
            [1, 1, 1]   # Out of plane
        ]
        
        angle = dihedral(points, rna_distances=False)
        
        # Should return a valid angle
        assert angle is not None
        assert 0 <= angle <= 360
    
    def test_dihedral_invalid_distance(self):
        """Test dihedral with points too far apart."""
        # Create points that are too far apart for RNA
        points = [
            [0, 0, 0],
            [10, 0, 0],  # Too far
            [10, 10, 0],
            [10, 10, 10]
        ]
        
        angle = dihedral(points, rna_distances=True)
        
        # Should return None for invalid RNA distances
        assert angle is None
    
    def test_exponential_map(self):
        """Test exponential map to sphere."""
        # Test with simple 2D data
        V = np.array([[0, 0], [1, 0], [0, 1]])
        p = np.array([0, 0, 1])  # North pole
        
        mapped = exponential_map(V, p)
        
        assert mapped.shape == (3, 3)
        
        # Check that points are on unit sphere
        norms = np.linalg.norm(mapped, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
    
    def test_rotation_identity(self):
        """Test rotation between identical vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        
        R = rotation(v1, v2)
        
        # Should be identity matrix
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
    
    def test_rotation_orthogonal(self):
        """Test rotation between orthogonal vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        
        R = rotation(v1, v2)
        
        # Check that rotation is valid
        assert np.abs(np.linalg.det(R) - 1) < 1e-10  # Proper rotation
        
        # Check that it rotates v1 to v2
        rotated = R @ v1
        np.testing.assert_allclose(rotated, v2, atol=1e-10)