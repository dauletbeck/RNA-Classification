"""Pytest configuration and fixtures for mintage_low_res tests."""

import pytest
import numpy as np
from unittest.mock import Mock

from ..data.models import Suite


@pytest.fixture
def mock_suite():
    """Create a mock Suite object for testing."""
    suite = Mock(spec=Suite)
    
    # Set default attributes
    suite._nu_1 = [325.0]  # C2'-endo
    suite._nu_2 = [330.0]  # C2'-endo
    suite._name = "test_suite"
    suite._filename = "test"
    suite.complete_suite = True
    
    # Default low-resolution coordinates
    suite.low_resolution_coordinates.return_value = [
        4.5,   # d2
        5.2,   # d3
        110.0, # alpha
        75.0,  # theta1
        45.0,  # phi1
        105.0, # theta2
        -30.0  # phi2
    ]
    
    return suite


@pytest.fixture
def test_coordinates():
    """Create test coordinate data."""
    np.random.seed(42)
    n_points = 20
    
    coords = np.column_stack([
        np.random.uniform(3, 6, n_points),      # d2
        np.random.uniform(4, 7, n_points),      # d3
        np.random.uniform(60, 140, n_points),   # alpha
        np.random.uniform(0, 180, n_points),    # theta1
        np.random.uniform(-180, 180, n_points), # phi1
        np.random.uniform(0, 180, n_points),    # theta2
        np.random.uniform(-180, 180, n_points)  # phi2
    ])
    
    return coords


@pytest.fixture
def test_clusters():
    """Create test cluster data."""
    clusters = [
        [0, 1, 2, 3],      # Cluster 1: 4 points
        [4, 5, 6, 7, 8],   # Cluster 2: 5 points
        [9, 10, 11],       # Cluster 3: 3 points
        [12, 13],          # Cluster 4: 2 points (small)
    ]
    
    outliers = [14, 15, 16]
    
    return clusters, outliers


@pytest.fixture
def sample_pdb_content():
    """Create sample PDB file content for testing."""
    pdb_lines = [
        "ATOM      1  P     A A   1      10.000  20.000  30.000  1.00 10.00           P",
        "ATOM      2  O5*   A A   1      11.000  21.000  31.000  1.00 10.00           O",
        "ATOM      3  C5*   A A   1      12.000  22.000  32.000  1.00 10.00           C",
        "ATOM      4  C4*   A A   1      13.000  23.000  33.000  1.00 10.00           C",
        "ATOM      5  O4*   A A   1      14.000  24.000  34.000  1.00 10.00           O",
        "ATOM      6  C3*   A A   1      15.000  25.000  35.000  1.00 10.00           C",
        "ATOM      7  O3*   A A   1      16.000  26.000  36.000  1.00 10.00           O",
        "ATOM      8  C2*   A A   1      17.000  27.000  37.000  1.00 10.00           C",
        "ATOM      9  O2*   A A   1      18.000  28.000  38.000  1.00 10.00           O",
        "ATOM     10  C1*   A A   1      19.000  29.000  39.000  1.00 10.00           C",
        "ATOM     11  N9    A A   1      20.000  30.000  40.000  1.00 10.00           N",
        "ATOM     12  C4    A A   1      21.000  31.000  41.000  1.00 10.00           C",
        # Second residue
        "ATOM     20  P     U A   2      30.000  40.000  50.000  1.00 10.00           P",
        "ATOM     21  O5*   U A   2      31.000  41.000  51.000  1.00 10.00           O",
        "ATOM     22  C5*   U A   2      32.000  42.000  52.000  1.00 10.00           C",
        "ATOM     23  C4*   U A   2      33.000  43.000  53.000  1.00 10.00           C",
        "ATOM     24  O4*   U A   2      34.000  44.000  54.000  1.00 10.00           O",
        "ATOM     25  C3*   U A   2      35.000  45.000  55.000  1.00 10.00           C",
        "ATOM     26  O3*   U A   2      36.000  46.000  56.000  1.00 10.00           O",
        "ATOM     27  C2*   U A   2      37.000  47.000  57.000  1.00 10.00           C",
        "ATOM     28  O2*   U A   2      38.000  48.000  58.000  1.00 10.00           O",
        "ATOM     29  C1*   U A   2      39.000  49.000  59.000  1.00 10.00           C",
        "ATOM     30  N1    U A   2      40.000  50.000  60.000  1.00 10.00           N",
        "ATOM     31  C2    U A   2      41.000  51.000  61.000  1.00 10.00           C",
    ]
    
    return '\n'.join(pdb_lines)


@pytest.fixture
def temp_pdb_file(tmp_path, sample_pdb_content):
    """Create a temporary PDB file for testing."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(sample_pdb_content)
    return str(pdb_file)