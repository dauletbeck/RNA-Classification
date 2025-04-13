#!/usr/bin/env python3
"""
mint.py

Implementation of the "MINT" step (Algorithm C.3) from Mardia et al. (2022).

Given:
  - Pre-clusters from AGE (Algorithm C.1).
  - Data X on a torus (or manifold).
  - Minimal cluster size kappa, etc.

We repeatedly:
  - Take one cluster from the list R,
  - Try torus PCA with 4 sets of flags (GC/MC + SI/SO),
  - If a "good" 1D representation is found (4*SSR <= Frechet Var),
    we do circular mode hunting in 1D:
      - If multiple modes => split cluster
      - Else => try next PCA flags
    if none of the 4 flags yields a multi-mode cluster => finalize that cluster.

References:
  Mardia et al. (2022), Eltzner et al. (2018) for torus PCA, 
  plus circular mode hunting (Supplement C.3).
"""

import numpy as np

##############################################################################
# 1) Placeholders / stubs for geometry and statistics
##############################################################################

def torus_pca_1D(X, flags):
    """
    Stub for the actual torus PCA. In practice, you might:
      - use `pns_loop` from pns.py,
      - or do "Torus PCA" from Eltzner et al. (2018),
      - then reduce to 1D, returning (X_1D, mu).
    
    :param X: (n, d) array of data on the torus
    :param flags: tuple/string specifying (GC/MC, SI/SO)
    :return: (X_1D, mu)
       X_1D: 1D array of length n giving each point's coordinate on the 1D submanifold
       mu:   the "nested mean" or center in the original torus geometry
    """
    # ----------------------------------------------------
    # Placeholder approach: we simply return the first coordinate as "1D"
    # and use the naive Euclidean mean as "mu".
    # Replace this with a genuine torus PCA.
    # ----------------------------------------------------
    X_1D = X[:, 0]  # just pick the 0-th dimension
    mu = np.mean(X, axis=0)  # naive "mean" in R^d
    return X_1D, mu

def torus_distance_squared(x, y):
    """
    Stub for distance^2 on the torus T^d. 
    Real version might do wraparound or geodesic distance:
      sum_j min(|x_j-y_j|, 2π - |x_j-y_j|)^2 
    """
    return np.sum((x - y)**2)

def sum_of_squared_distances(X, center):
    """
    Sum of squared distances of each row in X to 'center', using torus distance.
    """
    return np.sum([torus_distance_squared(x, center) for x in X])

def sum_of_squared_residuals(X, X_1D, reconstruct_func=None):
    """
    For each 1D coordinate in X_1D, get the "reconstructed" point in T^d,
    measure distance^2 to actual X[i].
    
    Reconstructing from 1D -> T^d typically requires the geometry from torus PCA.
    """
    if reconstruct_func is None:
        # Dummy: interpret the 1D coordinate X_1D as just the 0-th dimension
        # and keep the other dims from X. Minimizing residual along dimension 0 only.
        rss = 0.0
        for i, val_1d in enumerate(X_1D):
            x_orig = X[i]
            x_hat = x_orig.copy()
            x_hat[0] = val_1d
            rss += torus_distance_squared(x_orig, x_hat)
        return rss
    else:
        # If you have a real reconstruction: X_1D -> T^d
        rss = 0.0
        for i, val_1d in enumerate(X_1D):
            x_hat = reconstruct_func(val_1d)
            rss += torus_distance_squared(X[i], x_hat)
        return rss

def circular_mode_hunting(X_1D):
    """
    Stub function to detect multiple modes in a circle. 
    In the real code, you'd convert each 1D value into an angle in [0, 2π)
    and do a kernel-based or Silverman test for multiple modes.
    
    Return:
      - list of index subsets (each subset is a new subcluster),
      - or None if unimodal.
    """
    # For demonstration, let's define a trivial "bimodal if stdev > 1.0"
    import numpy as np
    angles = np.mod(X_1D, 2*np.pi)  # interpret as angles
    if np.std(angles) > 1.0:
        # Bimodal: split at angle=π
        idx1 = np.where(angles < np.pi)[0]
        idx2 = np.where(angles >= np.pi)[0]
        return [idx1, idx2]
    else:
        return None

##############################################################################
# 2) Implementation of the MINT step (Algorithm C.3)
##############################################################################

def mint_step(
    X,
    preclusters,
    # Possibly other parameters controlling "goodness" thresholds, etc.
    # e.g., you might pass in a significance level for mode hunting
):
    """
    Implements the MINT-step from Mardia et al. (2022), Algorithm C.3.
    
    Inputs:
      X           : (n, d) data array (on the torus).
      preclusters : list of arrays (or lists) of point indices, from AGE step.
    
    Returns:
      F           : final list of clusters (each a list of indices).
    """

    # The text calls R the "remaining cluster list" and F the "final cluster list"
    R = [np.array(c) for c in preclusters]  # transform each cluster to np.array
    F = []  # final clusters

    # We define the "flag sets" as described: (GC,SI), (GC,SO), (MC,SI), (MC,SO)
    flag_sets = [
        ("GC", "SI"),  # m=1
        ("GC", "SO"),  # m=2
        ("MC", "SI"),  # m=3
        ("MC", "SO")   # m=4
    ]

    while len(R) > 0:
        # Step 1: take a cluster C from R
        C_indices = R.pop(0)       # remove from front
        C_coords = X[C_indices]    # the actual points in that cluster
        m = 1
        cluster_finalized = False

        # Repeated attempts with different flags
        while m <= 5 and (not cluster_finalized):
            if m == 5:
                # If we reach m=5 => finalize the cluster "as is"
                F.append(C_indices)
                cluster_finalized = True
                break

            # Step 2: Perform torus PCA with the (m)-th flags
            flags = flag_sets[m-1]   # e.g. ("GC", "SI")
            X_1D, mu = torus_pca_1D(C_coords, flags)

            # Step 3: Evaluate "4 * SSR <= sum of squared distances from X to mu"
            # Frechet variance approx = sum_i d^2(X[i], mu). We check 4*SSR <= ...
            ss_to_mu = sum_of_squared_distances(C_coords, mu)
            ss_resid = sum_of_squared_residuals(C_coords, X_1D)

            if 4.0 * ss_resid <= ss_to_mu:
                # "Good enough" 1D representation => do circular mode hunting
                subclusters = circular_mode_hunting(X_1D)
                if subclusters is not None and len(subclusters) > 1:
                    # => multiple subclusters => add them to R for further analysis
                    for sc in subclusters:
                        R.append(C_indices[sc])  # re-insert subclusters for next iteration
                    cluster_finalized = True
                else:
                    # => unimodal => go to next flags
                    m += 1
            else:
                # => Condition not satisfied => try next flags
                m += 1

    # Return final clusters
    # The outliers from the AGE step remain outliers; we do not re-insert them here.
    return F

##############################################################################
# Optional testing
##############################################################################

if __name__ == "__main__":
    # Synthetic example
    np.random.seed(42)
    # Suppose we had some preclusters from age_clustering:
    # We'll mock them up.
    cluster1_inds = list(range(0, 20))
    cluster2_inds = list(range(20, 35))
    cluster3_inds = list(range(35, 40))  # maybe a smaller cluster

    preclusters_mock = [cluster1_inds, cluster2_inds, cluster3_inds]

    # Make synthetic data X
    # cluster1 near [0,0], cluster2 near [5,5], cluster3 near [10,10]
    c1 = np.random.normal(loc=[0,0], scale=0.5, size=(20,2))
    c2 = np.random.normal(loc=[5,5], scale=0.5, size=(15,2))
    c3 = np.random.normal(loc=[10,10], scale=0.5, size=(5,2))
    X = np.vstack([c1, c2, c3])  # shape (40,2)

    F = mint_step(X, preclusters_mock)
    print("=== MINT Step Results ===")
    for i, fc in enumerate(F, start=1):
        print(f"Final cluster {i}, size={len(fc)}: indices={sorted(fc)}")
