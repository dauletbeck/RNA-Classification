"""Tests for pucker analysis functionality."""

import numpy as np
import pytest
from unittest.mock import Mock

from ..preprocessing.pucker_analysis import (
    PuckerAnalyzer, 
    determine_pucker_data,
    sort_data_into_cluster,
    get_pucker_statistics
)
from ..data.models import Suite


class TestPuckerAnalyzer:
    """Test pucker analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PuckerAnalyzer()
    
    def create_mock_suite(self, nu1_angle, nu2_angle):
        """Create a mock suite with specified nu angles."""
        suite = Mock(spec=Suite)
        suite._nu_1 = [nu1_angle]
        suite._nu_2 = [nu2_angle]
        return suite
    
    def test_classify_c2c2(self):
        """Test classification of C2'-C2' pucker."""
        suite = self.create_mock_suite(325.0, 330.0)  # Both C2'-endo
        
        pucker_type = self.analyzer.classify_suite_pucker(suite)
        
        assert pucker_type == 'c2c2'
    
    def test_classify_c3c3(self):
        """Test classification of C3'-C3' pucker."""
        suite = self.create_mock_suite(160.0, 170.0)  # Both C3'-endo
        
        pucker_type = self.analyzer.classify_suite_pucker(suite)
        
        assert pucker_type == 'c3c3'
    
    def test_classify_c2c3(self):
        """Test classification of C2'-C3' pucker."""
        suite = self.create_mock_suite(325.0, 160.0)  # C2'-endo, C3'-endo
        
        pucker_type = self.analyzer.classify_suite_pucker(suite)
        
        assert pucker_type == 'c2c3'
    
    def test_classify_c3c2(self):
        """Test classification of C3'-C2' pucker."""
        suite = self.create_mock_suite(160.0, 325.0)  # C3'-endo, C2'-endo
        
        pucker_type = self.analyzer.classify_suite_pucker(suite)
        
        assert pucker_type == 'c3c2'
    
    def test_classify_invalid_angles(self):
        """Test classification with missing angles."""
        suite = self.create_mock_suite(None, 325.0)
        
        pucker_type = self.analyzer.classify_suite_pucker(suite)
        
        assert pucker_type is None
    
    def test_get_pucker_distances(self):
        """Test pucker distance calculation."""
        suite = self.create_mock_suite(325.0, 160.0)
        
        dist1, dist2 = self.analyzer.get_pucker_distances(suite)
        
        assert dist1 is not None
        assert dist2 is not None
        assert dist1 >= 0
        assert dist2 >= 0


class TestDeterminePuckerData:
    """Test the determine_pucker_data function."""
    
    def create_test_suites(self):
        """Create test suites with known pucker types."""
        suites = [
            self.create_mock_suite(325.0, 330.0),  # c2c2
            self.create_mock_suite(160.0, 170.0),  # c3c3
            self.create_mock_suite(325.0, 160.0),  # c2c3
            self.create_mock_suite(160.0, 325.0),  # c3c2
            self.create_mock_suite(None, 325.0),   # invalid
        ]
        return suites
    
    def create_mock_suite(self, nu1_angle, nu2_angle):
        """Create a mock suite with specified nu angles."""
        suite = Mock(spec=Suite)
        suite._nu_1 = [nu1_angle]
        suite._nu_2 = [nu2_angle]
        return suite
    
    def test_determine_c2c2_data(self):
        """Test filtering for c2c2 pucker type."""
        suites = self.create_test_suites()
        
        indices, filtered_suites = determine_pucker_data(suites, 'c2c2')
        
        assert len(indices) == 1
        assert indices[0] == 0  # First suite is c2c2
        assert len(filtered_suites) == 1
    
    def test_determine_c3c3_data(self):
        """Test filtering for c3c3 pucker type."""
        suites = self.create_test_suites()
        
        indices, filtered_suites = determine_pucker_data(suites, 'c3c3')
        
        assert len(indices) == 1
        assert indices[0] == 1  # Second suite is c3c3
    
    def test_determine_all_data(self):
        """Test getting all suites."""
        suites = self.create_test_suites()
        
        indices, filtered_suites = determine_pucker_data(suites, 'all')
        
        assert len(indices) == len(suites)
        assert len(filtered_suites) == len(suites)
    
    def test_invalid_pucker_name(self):
        """Test with invalid pucker name."""
        suites = self.create_test_suites()
        
        indices, filtered_suites = determine_pucker_data(suites, 'invalid')
        
        assert len(indices) == 0
        assert len(filtered_suites) == 0


class TestSortDataIntoCluster:
    """Test cluster data sorting functionality."""
    
    def test_sort_data_basic(self):
        """Test basic cluster sorting."""
        # Create test data
        data = np.random.rand(10, 3)
        clusters = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
        min_size = 2
        
        sorted_data, cluster_lengths = sort_data_into_cluster(
            data, clusters, min_size
        )
        
        assert sorted_data.shape[0] == 10  # All points included
        assert len(cluster_lengths) == 3   # All clusters above min_size
        assert cluster_lengths == [3, 4, 3]
    
    def test_sort_data_filter_small_clusters(self):
        """Test filtering of small clusters."""
        data = np.random.rand(8, 3)
        clusters = [[0, 1], [2, 3, 4], [5], [6, 7]]  # One cluster too small
        min_size = 2
        
        sorted_data, cluster_lengths = sort_data_into_cluster(
            data, clusters, min_size
        )
        
        assert sorted_data.shape[0] == 5   # Only large clusters
        assert len(cluster_lengths) == 2   # Two clusters above min_size
        assert cluster_lengths == [2, 3]
    
    def test_sort_data_empty_clusters(self):
        """Test with empty cluster list.""" 
        data = np.random.rand(5, 3)
        clusters = []
        min_size = 1
        
        sorted_data, cluster_lengths = sort_data_into_cluster(
            data, clusters, min_size
        )
        
        assert sorted_data.size == 0
        assert len(cluster_lengths) == 0


class TestGetPuckerStatistics:
    """Test pucker statistics functionality."""
    
    def create_mock_suite(self, nu1_angle, nu2_angle):
        """Create a mock suite with specified nu angles."""
        suite = Mock(spec=Suite)
        suite._nu_1 = [nu1_angle]
        suite._nu_2 = [nu2_angle]
        return suite
    
    def test_get_statistics(self):
        """Test getting pucker statistics."""
        suites = [
            self.create_mock_suite(325.0, 330.0),  # c2c2
            self.create_mock_suite(325.0, 330.0),  # c2c2
            self.create_mock_suite(160.0, 170.0),  # c3c3
            self.create_mock_suite(None, 325.0),   # unknown
        ]
        
        stats = get_pucker_statistics(suites)
        
        assert stats['c2c2'] == 2
        assert stats['c3c3'] == 1
        assert stats['c2c3'] == 0
        assert stats['c3c2'] == 0
        assert stats['unknown'] == 1