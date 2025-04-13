#!/usr/bin/env python3
"""
mint_age.py

Implementation of the full MINT-AGE pipeline combining:
  1) AGE (Algorithm C.1) from 'age.py'
  2) MINT (Algorithm C.3) from 'mint.py'

Usage:
  python mint_age.py
or in another script:
  from mint_age import mint_age_pipeline
  final_clusters, outliers = mint_age_pipeline(X, dmax=..., kappa=..., q=...)
"""

import numpy as np

# 1) Import the AGE step
from age import age_clustering

# 2) Import the MINT step
from mint import mint_step

# (Optional) If your MINT step uses real torus PCA from pns.py, you can import it here:
# from pns import pns_loop, pns_nested_mean
# (But in this minimal example, we rely on the stub placeholders inside mint.py.)

def mint_age_pipeline(X, dmax=3.0, kappa=20, q=0.15):
    """
    Run the entire MINT-AGE procedure on data X.

    :param X: np.array of shape (n, d). Data is on a torus or in R^d (with appropriate distance).
    :param dmax: float. Maximal outlier distance for the AGE step.
    :param kappa: int. Minimal cluster size for the AGE step.
    :param q: float. Relative branching distance in [0, 1] for the AGE adaptive cut.

    :return: (final_clusters, outliers)
      final_clusters : list of final cluster index sets (each set is a list/array of indices).
      outliers       : list of outlier indices from the AGE step.
    """
    # 1) AGE pre-clustering
    preclusters, outliers = age_clustering(X, dmax, kappa, q)

    # 2) MINT step refinement
    #    (splits large clusters if the 1D representation is "good enough" and multi-modal)
    final_clusters = mint_step(X, preclusters)

    return final_clusters, outliers

def main():
    """
    Demonstration of MINT-AGE with synthetic 2D data.
    """
    np.random.seed(42)
    # Synthetic data:
    # cluster 1 near [0,0], cluster 2 near [5,5], plus some outliers
    c1 = np.random.normal(loc=[0, 0], scale=0.5, size=(30, 2))
    c2 = np.random.normal(loc=[5, 5], scale=0.5, size=(20, 2))
    o  = np.array([[10,10], [9.5,9.8], [10.1,10.2]])
    X  = np.vstack([c1, c2, o])  # shape ~ (53, 2)

    # Run MINT-AGE pipeline
    final_clusters, outliers = mint_age_pipeline(X, dmax=3.0, kappa=5, q=0.15)

    print("=== MINT-AGE Pipeline Results ===")
    print("Number of final clusters:", len(final_clusters))
    for i, fc in enumerate(final_clusters, start=1):
        print(f" Cluster {i}, size={len(fc)}: {sorted(fc)}")
    print("Outliers:", outliers)

if __name__ == "__main__":
    main()
