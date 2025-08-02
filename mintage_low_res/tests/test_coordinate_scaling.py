"""Tests for coordinate scaling functionality."""

import numpy as np
import pytest
from unittest.mock import Mock

from ..preprocessing.coordinate_scaling import (
    CoordinateScaler,
    scale_coordinates,
    circular_mean
)
from ..data.models import Suite


class TestCircularMean:
    """Test circular mean calculation."""
    
    def test_circular_mean_simple(self):
        """Test circular mean with simple angles."""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        
        mean_angle, variance = circular_mean(angles)
        
        # Mean should be close to zero (or 2π)
        assert abs(mean_angle) < 1e-10 or abs(mean_angle - 2*np.pi) < 1e-10
        
        # Variance should be between 0 and 1
        assert 0 <= variance <= 1
    
    def test_circular_mean_concentrated(self):
        """Test circular mean with concentrated angles."""
        angles = np.array([0.1, 0.2, 0.0, -0.1, -0.2])
        
        mean_angle, variance = circular_mean(angles)
        
        # Mean should be close to zero
        assert abs(mean_angle) < 0.5
        
        # Variance should be small (concentrated data)
        assert variance < 0.5


class TestCoordinateScaler:
    """Test coordinate scaling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scaler = CoordinateScaler()
    
    def create_mock_suite(self, coords):
        """Create a mock suite with specified coordinates."""
        suite = Mock(spec=Suite)
        suite.low_resolution_coordinates.return_value = coords
        return suite
    
    def create_test_suites(self, n_suites=10):
        """Create test suites with random coordinates."""
        np.random.seed(42)  # For reproducibility
        suites = []
        
        for i in range(n_suites):
            # Create coordinates: [d2, d3, alpha, theta1, phi1, theta2, phi2]
            coords = [
                np.random.uniform(3, 6),    # d2
                np.random.uniform(4, 7),    # d3
                np.random.uniform(60, 140), # alpha
                np.random.uniform(0, 180),  # theta1
                np.random.uniform(-180, 180), # phi1
                np.random.uniform(0, 180),  # theta2
                np.random.uniform(-180, 180)  # phi2
            ]
            suites.append(self.create_mock_suite(coords))
        
        return suites
    
    def test_scale_coordinates_basic(self):
        """Test basic coordinate scaling."""
        suites = self.create_test_suites(20)
        
        scaled_coords, lambda_d, lambda_alpha = self.scaler.scale_coordinates(suites)
        
        # Check output shape
        assert scaled_coords.shape == (20, 7)
        
        # Check scaling factors are positive
        assert lambda_d > 0
        assert lambda_alpha > 0
        
        # Check that coordinates are finite
        assert np.all(np.isfinite(scaled_coords))
    
    def test_scale_coordinates_no_distance_scaling(self):
        """Test scaling with distance variance scaling disabled."""
        scaler = CoordinateScaler(scale_distance_variance=False)
        suites = self.create_test_suites(10)
        
        scaled_coords, lambda_d, lambda_alpha = scaler.scale_coordinates(suites)
        
        # Lambda_d should be 1.0 when distance scaling is disabled
        assert lambda_d == 1.0
        assert lambda_alpha > 0  # Alpha scaling still enabled
    
    def test_scale_coordinates_no_alpha_scaling(self):
        """Test scaling with alpha variance scaling disabled."""
        scaler = CoordinateScaler(scale_alpha_variance=False)
        suites = self.create_test_suites(10)
        
        scaled_coords, lambda_d, lambda_alpha = scaler.scale_coordinates(suites)
        
        # Lambda_alpha should be 1.0 when alpha scaling is disabled
        assert lambda_d > 0  # Distance scaling still enabled
        assert lambda_alpha == 1.0
    
    def test_scale_coordinates_preserve_means(self):
        """Test that means are preserved when requested."""
        scaler = CoordinateScaler(
            preserve_distance_mean=True,
            preserve_alpha_mean=True
        )
        suites = self.create_test_suites(50)
        
        # Get original coordinates
        original_coords = np.array([s.low_resolution_coordinates() for s in suites])
        
        scaled_coords, _, _ = scaler.scale_coordinates(suites)
        
        # Distance means should be approximately preserved (d2, d3)
        orig_d_mean = np.mean(original_coords[:, :2], axis=0)
        scaled_d_mean = np.mean(scaled_coords[:, :2], axis=0)
        np.testing.assert_allclose(orig_d_mean, scaled_d_mean, rtol=1e-10)
        
        # Alpha mean should be approximately preserved
        # (This is more complex due to circular statistics)
        # We just check that the scaled alpha is reasonable
        assert np.all(scaled_coords[:, 2] >= -180)
        assert np.all(scaled_coords[:, 2] <= 360)


class TestScaleCoordinatesFunction:
    """Test the main scale_coordinates function."""
    
    def create_mock_suite(self, coords):
        """Create a mock suite with specified coordinates."""
        suite = Mock(spec=Suite)
        suite.low_resolution_coordinates.return_value = coords
        return suite
    
    def test_scale_coordinates_function(self):
        """Test the main scale_coordinates function."""
        # Create test suites
        suites = []
        for i in range(10):
            coords = [4.0, 5.0, 90.0, 45.0, 0.0, 135.0, 90.0]
            suites.append(self.create_mock_suite(coords))
        
        # Scale coordinates
        scaled_coords, lambda_d, lambda_alpha = scale_coordinates(
            suites,
            scale_distance_variance=True,
            scale_alpha_variance=True,
            preserve_distance_mean=True,
            preserve_alpha_mean=True,
            store_attr="test_scaled_coords"
        )
        
        # Check outputs
        assert scaled_coords.shape == (10, 7)
        assert lambda_d > 0
        assert lambda_alpha > 0
        
        # Check that attribute was stored in suites
        for suite in suites:
            assert hasattr(suite, 'test_scaled_coords')
    
    def test_scale_coordinates_empty_suites(self):
        """Test scaling with empty suite list."""
        suites = []
        
        # This should raise an error or handle gracefully
        with pytest.raises((IndexError, ValueError)):
            scale_coordinates(suites)
    
    def test_scale_coordinates_single_suite(self):
        """Test scaling with single suite."""
        coords = [4.0, 5.0, 90.0, 45.0, 0.0, 135.0, 90.0]
        suite = self.create_mock_suite(coords)
        suites = [suite]
        
        scaled_coords, lambda_d, lambda_alpha = scale_coordinates(suites)
        
        # Should handle single suite gracefully
        assert scaled_coords.shape == (1, 7)
        assert lambda_d > 0
        assert lambda_alpha > 0