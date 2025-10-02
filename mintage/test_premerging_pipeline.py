#!/usr/bin/env python3
"""
Test script for the new pre-merging pipeline.
Tests the workflow: preclusters → merge → PNS mode hunting
"""

import pickle
import numpy as np
import os
from clustering.cluster_merging import cluster_merging
from pnds.PNS_mode_hunter import refine_clusters_with_pns
from pnds.PNDS_PNS import PNS

def spherical_to_vec(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.column_stack([np.sin(t) * np.cos(p),
                            np.sin(t) * np.sin(p),
                            np.cos(t)])

def exponential_map(V, p):
    N, M = V.shape[0], V.shape[1]
    V_mean = V.mean(axis=0)
    V -= V_mean
    V = np.column_stack([V, np.zeros(N)])
    V_norm = np.linalg.norm(V, axis=1)[:, None]
    return np.cos(V_norm) * p + np.sin(V_norm) * (V / V_norm)

def test_single_pucker(test_filename='precluster_qf_0.08_c2c2_precluster.pkl'):
    """Test the new pipeline on a single pucker type"""

    result_dir = 'preclustering_results/minimal_q_fold_no_outlier'
    filepath = os.path.join(result_dir, test_filename)

    if not os.path.exists(filepath):
        print(f"Test file not found: {filepath}")
        return False

    print(f"Testing new pipeline with: {test_filename}")

    # Load precluster data
    with open(filepath, 'rb') as f:
        result = pickle.load(f)

    name = result['name']
    clusters = result['clusters']
    outliers = result['outliers']
    scaled_coords_by_pucker = result['scaled_coords']

    print(f"[{name}] Original preclusters: {len(clusters)} clusters")

    try:
        # Step 1: Process coordinates
        d2_s, d3_s, alpha_s, theta1, phi1, theta2, phi2 = scaled_coords_by_pucker.T

        S2_1 = spherical_to_vec(theta1, phi1)
        pns_S2_1 = PNS(mode='great', verbose=False).fit(S2_1)
        theta1, phi1 = pns_S2_1.dists_

        S2_2 = spherical_to_vec(theta2, phi2)
        pns_S2_2 = PNS(mode='great', verbose=False).fit(S2_2)
        theta2, phi2 = pns_S2_2.dists_

        V = np.column_stack([d2_s, d3_s])
        V -= np.mean(V, axis=0)

        S2_d = exponential_map(V, p=np.array([0,0,1]))
        pns_S2_d = PNS(mode='great', verbose=False).fit(S2_d)
        theta_d, phi_d = pns_S2_d.dists_

        angle_matrix = np.column_stack([
            theta_d + 180,
            phi_d + 180,
            alpha_s,
            theta1 + 180,
            phi1 + 180,
            theta2 + 180,
            phi2 + 180
        ])

        # Step 2: Apply cluster merging to preclusters
        print(f"[{name}] Applying cluster merging to preclusters...")
        os.makedirs(f'./out/cluster_merging_preclusters/{name}/', exist_ok=True)
        merged_preclusters = cluster_merging(
            cluster_index_lists=clusters,
            dihedral_angles=angle_matrix,
            folder=f'./out/cluster_merging_preclusters/{name}/',
            circular=True,
            plot=False
        )
        print(f"[{name}] After merging: {len(merged_preclusters)} clusters")

        # Step 3: Run PNS mode hunting on merged preclusters
        print(f"[{name}] Running PNS mode hunting on merged preclusters...")
        mode_clusters, _ = refine_clusters_with_pns(
            scale=12000,
            data=angle_matrix,
            cluster_list=merged_preclusters,
            outlier_list=outliers,
            min_cluster_size=3,
            enable_cluster_merging=False,
            merging_plot=False
        )

        print(f"[{name}] Final result: {len(mode_clusters)} mode clusters")

        # Print cluster sizes for comparison
        original_sizes = [len(c) for c in clusters]
        merged_sizes = [len(c) for c in merged_preclusters]
        final_sizes = [len(c) for c in mode_clusters]

        print(f"[{name}] Original cluster sizes: {sorted(original_sizes, reverse=True)}")
        print(f"[{name}] Merged cluster sizes: {sorted(merged_sizes, reverse=True)}")
        print(f"[{name}] Final cluster sizes: {sorted(final_sizes, reverse=True)}")

        return True

    except Exception as e:
        print(f"[{name}] Error in pipeline: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Pre-Merging Pipeline ===")
    success = test_single_pucker()
    print(f"Pipeline test: {'SUCCESS' if success else 'FAILED'}")