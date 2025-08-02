#!/usr/bin/env python3
"""
Test script to verify the robustness improvements in PNS.
"""

import numpy as np
import sys
import os

# Add the mintage directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mintage'))

from pnds.PNDS_PNS import PNS

def test_robustness():
    """Test the robustness improvements in PNS."""
    
    # Test 1: Normal case
    print("Test 1: Normal case")
    np.random.seed(42)
    n, d = 100, 5
    data = np.random.randn(n, d)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    pns = PNS(verbose=True)
    result = pns.fit(data)
    print(f"Normal case: spheres={len(pns.spheres_) if pns.spheres_ else 0}")
    
    # Test 2: Degenerate case (fewer points than dimensions)
    print("\nTest 2: Degenerate case")
    np.random.seed(42)
    n, d = 3, 5  # Fewer points than dimensions
    data = np.random.randn(n, d)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    pns = PNS(verbose=True)
    result = pns.fit(data)
    print(f"Degenerate case: spheres={len(pns.spheres_) if pns.spheres_ else 0}")
    
    # Test 3: Edge case with very small data
    print("\nTest 3: Very small dataset")
    np.random.seed(42)
    n, d = 2, 3
    data = np.random.randn(n, d)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    pns = PNS(verbose=True)
    result = pns.fit(data)
    print(f"Small dataset: spheres={len(pns.spheres_) if pns.spheres_ else 0}")
    
    # Test 4: Test different modes
    print("\nTest 4: Testing different modes")
    np.random.seed(42)
    n, d = 50, 4
    data = np.random.randn(n, d)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    for mode in ['great', 'small', 'adaptive']:
        pns = PNS(verbose=False, mode=mode)
        result = pns.fit(data)
        print(f"Mode '{mode}': spheres={len(pns.spheres_) if pns.spheres_ else 0}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_robustness() 