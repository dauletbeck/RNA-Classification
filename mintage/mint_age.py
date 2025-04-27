#!/usr/bin/env python3

"""
mint_age_pipeline_modular.py

A modular version of the MINT-AGE style pipeline:
1) Parse data from PDB files
2) Procrustes alignment
3) Pre-clustering (average linkage, outlier detection)
4) Post-clustering (torus PCA + mode hunting)
5) (Optional) plotting
"""

import os
import pickle
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import average, fcluster

from shape_analysis import procrustes_analysis

# pnds/PNDS_RNA_clustering.py
from pnds.PNDS_RNA_clustering import new_multi_slink

# utils/help_plot_functions.py
from utils.help_plot_functions import plot_clustering

# utils/data_functions.py
from utils.data_functions import rotate_y_optimal_to_x, rotation_matrix_x_axis

# parsing/parse_functions.py
from parsing.parse_functions import parse_pdb_files

from utils.pucker_data_functions import determine_pucker_data, procrustes_for_each_pucker
from clustering.cluster_improving import cluster_merging
import shape_analysis
from pnds import PNDS_RNA_clustering


def parse_data(input_pdb_dir):
    """
    Step 0: Parse data from a local directory containing PDB files.
    Uses parse_pdb_files(...) from your code.
    Returns a list of 'suite' objects (or similar) extracted from PDBs.
    """
    print("[MINT-AGE] Parsing data from:", input_pdb_dir)
    suites = parse_pdb_files(input_pdb_dir, input_pdb_folder=input_pdb_dir)
    print(f"Parsed {len(suites)} suite objects.")
    return suites


def procrustes_step(suites, output_folder, recalculate=False):
    """
    Step 1: Perform Procrustes alignment on the suite objects.
    We can load pre-aligned data from a pickle if recalculate=False
    or just do it fresh if recalculate=True or no file found.
    """
    print("[MINT-AGE] Step 1: Procrustes alignment...")

    procrustes_file = os.path.join(output_folder, "procrustes_suites.pickle")

    if recalculate:
        # Re-run procrustes alignment
        suites = procrustes_analysis(suites, overwrite=True)
        with open(procrustes_file, "wb") as f:
            pickle.dump(suites, f)
    else:
        # Attempt to load from file
        if os.path.isfile(procrustes_file):
            with open(procrustes_file, "rb") as f:
                suites = pickle.load(f)
            print("[MINT-AGE] Loaded pre-aligned suites from pickle.")
        else:
            # If no file, run alignment anyway
            suites = procrustes_analysis(suites, overwrite=True)
            with open(procrustes_file, "wb") as f:
                pickle.dump(suites, f)

    return suites


def AGE(
    suites,
    method=average,
    outlier_percentage=0.15,
    min_cluster_size=20,
):
    """
    Step 2: Pre-clustering using average linkage + simple outlier detection.

    We:
     - Extract dihedral angles (or any other representation) from each suite
     - Compute distances (e.g. Euclidean for demonstration)
     - Link them (method=average by default)
     - Determine a threshold to flag ~ 'outlier_percentage' fraction as outliers
     - Cut tree at that threshold
     - Return the set of cluster indices for further refinement.
    """
    print("[MINT-AGE] Step 2: Average-linkage pre-clustering + outlier detection...")

    # Filter out suites lacking ._dihedral_angles
    dihedral_data = np.array([s._dihedral_angles for s in suites if s._dihedral_angles is not None])
    suite_indices = [i for i, s in enumerate(suites) if s._dihedral_angles is not None]

    # Distances (Euclidean placeholder; replace with torus if needed)
    dist_vec = pdist(dihedral_data, metric="euclidean")

    # Linkage
    linkage_matrix = method(dist_vec)

    # Determine threshold
    threshold = find_outlier_threshold_simple(
        linkage_matrix,
        percentage=outlier_percentage,
        data_count=dihedral_data.shape[0],
        min_cluster_size=min_cluster_size
    )

    # Cluster labels
    cluster_labels = fcluster(linkage_matrix, threshold, criterion="distance")

    # Mark outliers as those in clusters < min_cluster_size
    cluster_counts = Counter(cluster_labels)
    outlier_clusters = [c for c, count in cluster_counts.items() if count < min_cluster_size]
    outlier_indices = [i for i, c in enumerate(cluster_labels) if c in outlier_clusters]

    # Build final cluster list (list of lists)
    cluster_list = []
    for c, count in cluster_counts.items():
        if count >= min_cluster_size:
            idxs = [suite_indices[i] for i, lab in enumerate(cluster_labels) if lab == c]
            cluster_list.append(idxs)

    # Label suites accordingly
    for c_idx, clust in enumerate(cluster_list):
        for sidx in clust:
            if not hasattr(suites[sidx], "clustering"):
                suites[sidx].clustering = {}
            suites[sidx].clustering["precluster"] = c_idx
    for sidx in outlier_indices:
        if not hasattr(suites[suite_indices[sidx]], "clustering"):
            suites[suite_indices[sidx]].clustering = {}
        suites[suite_indices[sidx]].clustering["precluster"] = "outlier"

    return cluster_list


def MINT(suites, cluster_list):
    """
    Step 3: Post-clustering (Torus PCA + mode hunting).
    Here we call 'new_multi_slink' from PNDS_RNA_clustering on each cluster.
    Returns a list of "refined" cluster index sets.
    """
    print("[MINT-AGE] Step 3: Torus PCA + mode hunting on each pre-cluster...")

    refined_clusters = []
    for clust_indices in cluster_list:
        # Gather dihedral angles for this cluster
        cluster_data = [suites[i]._dihedral_angles for i in clust_indices]
        if len(cluster_data) == 0:
            continue
        cluster_data = np.array(cluster_data)

        # new_multi_slink can return subclusters + noise
        # scale=12000 is domain-specific, adapt as needed
        subclusters, noise = new_multi_slink(
            scale=12000,
            data=cluster_data,
            cluster_list=[list(range(len(cluster_data)))],
            outlier_list=[]
        )
        # Map subclusters back to original suite indices
        for sc in subclusters:
            refined_clusters.append([clust_indices[idx] for idx in sc])

    # Label final clusters
    final_label = 0
    for clust in refined_clusters:
        for sidx in clust:
            if not hasattr(suites[sidx], "clustering"):
                suites[sidx].clustering = {}
            suites[sidx].clustering["mint_age_cluster"] = final_label
        final_label += 1

    return refined_clusters


def find_outlier_threshold_simple(linkage_matrix, percentage, data_count, min_cluster_size):
    """
    A simplified approach:
    - Sort distances in ascending order
    - For each distance 'd', cut the tree and see how many points
      belong to clusters of size < min_cluster_size.
    - If that fraction is <= percentage, we pick 'd'.
    """
    if percentage <= 0:
        return linkage_matrix[-1, 2] + 1.0

    distances = linkage_matrix[:, 2]
    for d in np.sort(distances):
        labs = fcluster(linkage_matrix, d, criterion='distance')
        counts = Counter(labs)
        small_cluster_pts = sum(cnt for cnt in counts.values() if cnt < min_cluster_size)
        outlier_frac = small_cluster_pts / data_count
        if outlier_frac <= percentage:
            return d
    return linkage_matrix[-1, 2] + 1.0


def filter_suites(suites, pucker_type=None):
    """
    Filter suites to match the logic in precluster_pruned.py.
    Optionally filter by pucker type.
    """
    filtered = [
        s for s in suites
        if getattr(s, 'procrustes_five_chain_vector', None) is not None
        and getattr(s, '_dihedral_angles', None) is not None
        and getattr(s, 'atom_types', None) == 'atm'
    ]
    if pucker_type and pucker_type != 'all':
        # Use determine_pucker_data to further filter by pucker type
        _, filtered = determine_pucker_data(filtered, pucker_type)
    return filtered


def run_mint_age_pipeline(
    input_pdb_dir,
    output_folder="./out/mint_age_pipeline",
    recalculate=False,
    min_cluster_size=20,
    outlier_percentage=0.15,
    method=average,
    plot=True,
    pucker_types=None,
    q_fold=0.15
):
    """
    Main assembler function for the entire MINT-AGE pipeline.
    Now loops over pucker types and applies filtering and per-pucker Procrustes.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if pucker_types is None:
        pucker_types = ['c3c3', 'c3c2', 'c2c3', 'c2c2', 'all']

    all_results = {}

    # Step 0: parse data
    suites = parse_data(input_pdb_dir)

    for pucker in pucker_types:
        print(f"\n[MINT-AGE] Processing pucker type: {pucker}")

        # Step 0.5: Filter suites for this pucker type
        filtered_suites = filter_suites(suites, pucker_type=pucker)
        if not filtered_suites:
            print(f"[MINT-AGE] No suites found for pucker type {pucker}")
            continue

        # Step 1: Procrustes (per-pucker if not 'all')
        if pucker != 'all':
            procrustes_data = np.array([s.procrustes_five_chain_vector for s in filtered_suites])
            procrustes_data_backbone = np.array([s.procrustes_complete_suite_vector for s in filtered_suites])
            procrustes_data, procrustes_data_backbone = procrustes_for_each_pucker(
                filtered_suites, procrustes_data, procrustes_data_backbone, pucker
            )
            # Optionally update suite objects with new procrustes data if needed
        else:
            filtered_suites = procrustes_step(filtered_suites, output_folder, recalculate=recalculate)

        # Prepare dihedral angles array
        dihedral_angles_suites = np.array([s._dihedral_angles for s in filtered_suites])

        # --- PRE-CLUSTERING (as in precluster_pruned.py) ---
        cluster_list, cluster_outlier_list, name_precluster = shape_analysis.pre_clustering(
            input_data=dihedral_angles_suites,
            m=min_cluster_size,
            percentage=outlier_percentage,
            string_folder=output_folder,
            method=method,
            q_fold=q_fold,
            distance="torus"
        )

        # --- POST-CLUSTERING (as in precluster_pruned.py) ---
        cluster_list_mode, noise1 = PNDS_RNA_clustering.new_multi_slink(
            scale=12000,
            data=dihedral_angles_suites,
            cluster_list=cluster_list,
            outlier_list=cluster_outlier_list,
            min_cluster_size=min_cluster_size
        )

        # --- CLUSTER MERGING (optional, as in precluster_pruned.py) ---
        try:
            merged_clusters = cluster_merging(cluster_list_mode, dihedral_angles_suites, plot=False)
        except Exception as e:
            print(f"[MINT-AGE] cluster_merging failed: {e}")
            merged_clusters = cluster_list_mode

        # --- PLOTTING (optional) ---
        if plot:
            final_plot_dir = os.path.join(output_folder, f"final_plots_{pucker}")
            print(f"[MINT-AGE] Plotting final refined clusters for {pucker}...")
            plot_clustering(
                suites=filtered_suites,
                cluster_list=merged_clusters,
                name=final_plot_dir,
                outlier_list=[]
            )

        # --- SAVE FINAL RESULTS ---
        final_pickle = os.path.join(output_folder, f"mint_age_final_suites_{pucker}.pickle")
        with open(final_pickle, "wb") as f:
            pickle.dump(filtered_suites, f)

        print(f"[MINT-AGE] Pipeline complete for {pucker}. Saved final suites to {final_pickle}")
        print(f"[MINT-AGE] Total final clusters: {len(merged_clusters)}")
        all_results[pucker] = filtered_suites

    return all_results


if __name__ == "__main__":
    pdb_dir = "/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/data/rna2020_pruned_pdbs/"

    # Loop over all pucker types as in precluster_pruned.py
    final_suites = run_mint_age_pipeline(
        input_pdb_dir=pdb_dir,
        output_folder="/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/results/mint_age_pipeline",
        recalculate=True,
        min_cluster_size=3,
        outlier_percentage=0.02,
        method=average,
        plot=True,
        pucker_types=['c3c3', 'c3c2', 'c2c3', 'c2c2', 'all'],
        q_fold=0.05  # or loop over q_fold values as needed
    )

    print("[MINT-AGE] Done.")
