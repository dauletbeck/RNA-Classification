"""Integration tests for the complete pipeline."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from ..analysis.experiment_runner import run_low_res_experiments
from ..config.experiment_config import ExperimentConfig


class TestIntegration:
    """Integration tests for the complete analysis pipeline."""
    
    def create_test_pdb_files(self, temp_dir, n_files=2):
        """Create test PDB files for integration testing."""
        pdb_template = """ATOM      1  P     A A   1      10.000  20.000  30.000  1.00 10.00           P
ATOM      2  O5*   A A   1      11.000  21.000  31.000  1.00 10.00           O
ATOM      3  C5*   A A   1      12.000  22.000  32.000  1.00 10.00           C
ATOM      4  C4*   A A   1      13.000  23.000  33.000  1.00 10.00           C
ATOM      5  O4*   A A   1      14.000  24.000  34.000  1.00 10.00           O
ATOM      6  C3*   A A   1      15.000  25.000  35.000  1.00 10.00           C
ATOM      7  O3*   A A   1      16.000  26.000  36.000  1.00 10.00           O
ATOM      8  C2*   A A   1      17.000  27.000  37.000  1.00 10.00           C
ATOM      9  O2*   A A   1      18.000  28.000  38.000  1.00 10.00           O
ATOM     10  C1*   A A   1      19.000  29.000  39.000  1.00 10.00           C
ATOM     11  N9    A A   1      20.000  30.000  40.000  1.00 10.00           N
ATOM     12  C4    A A   1      21.000  31.000  41.000  1.00 10.00           C
ATOM     20  P     U A   2      30.000  40.000  50.000  1.00 10.00           P
ATOM     21  O5*   U A   2      31.000  41.000  51.000  1.00 10.00           O
ATOM     22  C5*   U A   2      32.000  42.000  52.000  1.00 10.00           C
ATOM     23  C4*   U A   2      33.000  43.000  53.000  1.00 10.00           C
ATOM     24  O4*   U A   2      34.000  44.000  54.000  1.00 10.00           O
ATOM     25  C3*   U A   2      35.000  45.000  55.000  1.00 10.00           C
ATOM     26  O3*   U A   2      36.000  46.000  56.000  1.00 10.00           O
ATOM     27  C2*   U A   2      37.000  47.000  57.000  1.00 10.00           C
ATOM     28  O2*   U A   2      38.000  48.000  58.000  1.00 10.00           O
ATOM     29  C1*   U A   2      39.000  49.000  59.000  1.00 10.00           C
ATOM     30  N1    U A   2      40.000  50.000  60.000  1.00 10.00           N
ATOM     31  C2    U A   2      41.000  51.000  61.000  1.00 10.00           C
ATOM     40  P     G A   3      50.000  60.000  70.000  1.00 10.00           P
ATOM     41  O5*   G A   3      51.000  61.000  71.000  1.00 10.00           O
ATOM     42  C5*   G A   3      52.000  62.000  72.000  1.00 10.00           C
ATOM     43  C4*   G A   3      53.000  63.000  73.000  1.00 10.00           C
ATOM     44  O4*   G A   3      54.000  64.000  74.000  1.00 10.00           O
ATOM     45  C3*   G A   3      55.000  65.000  75.000  1.00 10.00           C
ATOM     46  O3*   G A   3      56.000  66.000  76.000  1.00 10.00           O
ATOM     47  C2*   G A   3      57.000  67.000  77.000  1.00 10.00           C
ATOM     48  O2*   G A   3      58.000  68.000  78.000  1.00 10.00           O
ATOM     49  C1*   G A   3      59.000  69.000  79.000  1.00 10.00           C
ATOM     50  N9    G A   3      60.000  70.000  80.000  1.00 10.00           N
ATOM     51  C4    G A   3      61.000  71.000  81.000  1.00 10.00           C
"""
        
        pdb_files = []
        for i in range(n_files):
            pdb_file = Path(temp_dir) / f"test_{i:03d}.pdb"
            pdb_file.write_text(pdb_template)
            pdb_files.append(str(pdb_file))
        
        return pdb_files
    
    @pytest.mark.integration
    def test_complete_pipeline_small_dataset(self):
        """Test the complete pipeline with a small synthetic dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test PDB files
            pdb_files = self.create_test_pdb_files(temp_dir, n_files=3)
            
            # Create minimal configuration
            config = ExperimentConfig(
                min_cluster_size=1,  # Small clusters for test data
                pns_scale=1000,      # Smaller scale for speed
                pucker_types=['c3c3'] # Only one pucker type for speed
            )
            
            # Create temporary cache and output directories
            cache_dir = Path(temp_dir) / "cache"
            output_dir = Path(temp_dir) / "output"
            
            try:
                # Run the complete pipeline
                results = run_low_res_experiments(\n",
                    pdb_directory=temp_dir,
                    pucker_types=['c3c3'],  # Simplified for testing
                    config=config,
                    cache_dir=str(cache_dir),
                    output_dir=str(output_dir)
                )
                
                # Validate results structure
                assert 'suites' in results
                assert 'scaled_coordinates' in results
                assert 'scaling_factors' in results
                assert 'pucker_results' in results
                assert 'summary' in results
                
                # Check that we got some suites
                assert len(results['suites']) > 0
                
                # Check scaling factors
                scaling_factors = results['scaling_factors']
                assert 'lambda_d' in scaling_factors
                assert 'lambda_alpha' in scaling_factors
                assert scaling_factors['lambda_d'] > 0
                assert scaling_factors['lambda_alpha'] > 0
                
                # Check pucker results
                assert 'c3c3' in results['pucker_results']
                c3c3_results = results['pucker_results']['c3c3']
                assert 'metadata' in c3c3_results
                assert 'mode_clusters' in c3c3_results
                
                # Check summary
                summary = results['summary']
                assert 'pucker_statistics' in summary
                assert 'timing_summary' in summary
                
                print(f"Integration test successful!")
                print(f"Parsed {len(results['suites'])} suites")
                print(f"Total time: {results['total_time']:.2f}s")
                
            except Exception as e:
                pytest.fail(f"Integration test failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_caching(self):
        """Test that caching works correctly.""" 
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            pdb_files = self.create_test_pdb_files(temp_dir, n_files=2)
            
            config = ExperimentConfig(
                min_cluster_size=1,
                pns_scale=1000,
                pucker_types=['c3c3']
            )
            
            cache_dir = Path(temp_dir) / "cache"
            output_dir = Path(temp_dir) / "output"
            
            # Run pipeline first time
            results1 = run_low_res_experiments(
                pdb_directory=temp_dir,
                pucker_types=['c3c3'],
                config=config,
                cache_dir=str(cache_dir),
                output_dir=str(output_dir)
            )
            
            first_time = results1['total_time']
            
            # Run pipeline second time (should use cache)
            results2 = run_low_res_experiments(
                pdb_directory=temp_dir,
                pucker_types=['c3c3'],
                config=config,
                cache_dir=str(cache_dir),
                output_dir=str(output_dir)
            )
            
            second_time = results2['total_time']
            
            # Second run should be faster due to caching
            # (Though this might not always be true in a test environment)
            assert len(results1['suites']) == len(results2['suites'])
            
            # Results should be essentially the same
            np.testing.assert_allclose(
                results1['scaled_coordinates'],
                results2['scaled_coordinates'],
                rtol=1e-10
            )
            
            print(f"First run: {first_time:.2f}s, Second run: {second_time:.2f}s")
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty directory (no PDB files)
            config = ExperimentConfig()
            
            # This should handle the empty directory gracefully
            try:
                results = run_low_res_experiments(
                    pdb_directory=temp_dir,
                    pucker_types=['c3c3'],
                    config=config,
                    cache_dir=str(Path(temp_dir) / "cache"),
                    output_dir=str(Path(temp_dir) / "output")
                )
                
                # Should get empty results, not crash
                assert 'suites' in results
                assert len(results['suites']) == 0
                
            except Exception as e:
                # If it does raise an exception, it should be informative
                assert "No PDB files found" in str(e) or "No suites" in str(e)


if __name__ == "__main__":
    # Run a simple integration test
    test = TestIntegration()
    test.test_complete_pipeline_small_dataset()