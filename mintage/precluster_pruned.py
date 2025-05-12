import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from scipy.cluster.hierarchy import average as average_linkage

from utils import write_files
from utils.plot_clusters import plot_all_cluster_combinations, plot_low_res
from utils.pucker_data_functions import (
    get_suites_from_pdb,
    determine_pucker_data,
    procrustes_for_each_pucker,
    sort_data_into_cluster,
    create_csv,
)
import shape_analysis
from utils import plot_functions
from pnds import PNDS_RNA_clustering
from clustering.cluster_improving import cluster_merging

# Constants
OUTPUT_DIR = Path("./out/newdata_without_hets_11_23")
PICKLE_DIR = Path("./out/saved_suite_lists")


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def cluster_pruned_rna(
    name: str,
    min_cluster_size: int = 20,
    max_outlier_dist_percent: float = 0.15,
    q_fold: float = 0.15,
    do_clustering: bool = True,
) -> None:
    """
    Perform hierarchical clustering on RNA pucker data and generate plots.

    Args:
        name: Identifier for the pucker subset (e.g., 'all', 'c3c3').
        min_cluster_size: Minimum number of points in a valid cluster.
        max_outlier_dist_percent: Threshold for outlier distance.
        q_fold: Quantile threshold for clustering.
        do_clustering: Whether to compute clusters or load existing ones.
    """
    # Prepare output directories
    folder_plots = OUTPUT_DIR
    ensure_dir(folder_plots)

    # Load and filter suites
    suites = get_suites_from_pdb()
    filtered = [s for s in suites
                if s.procrustes_five_chain_vector is not None
                and s.dihedral_angles is not None
                and s.atom_types == 'atm']
    print(f"semi-complete suites: {len(filtered)}")

    # Compute pucker-specific data
    _, filtered = determine_pucker_data(filtered, name)
    print(f"{name} suites: {len(filtered)}")

    dihedral_angles = np.array([s.dihedral_angles for s in filtered])
    procrustes_full = np.array([s.procrustes_five_chain_vector for s in filtered])
    procrustes_backbone = np.array([s.procrustes_complete_suite_vector for s in filtered])

    if procrustes_full.size == 0:
        print(f"No data for pucker '{name}'")
        return

    if name != 'all':
        procrustes_full, procrustes_backbone = procrustes_for_each_pucker(
            filtered, procrustes_full, procrustes_backbone, name
        )

    # Determine pickle paths
    pickle_base = PICKLE_DIR / f"cluster_indices_{name}_qfold{q_fold}"
    pickle_mode = PICKLE_DIR / f"cluster_indices_mode_{name}_qfold{q_fold}"
    ensure_dir(PICKLE_DIR)

    # Decide whether to run clustering
    if not do_clustering or not (pickle_base.with_suffix('.pickle').exists() and pickle_mode.with_suffix('.pickle').exists()):
        do_clustering = True

    if do_clustering:
        # Step 1: Pre-clustering
        clusters, outliers, _ = shape_analysis.pre_clustering(
            input_data=dihedral_angles,
            m=min_cluster_size,
            percentage=max_outlier_dist_percent,
            string_folder=str(folder_plots),
            method=average_linkage,
            q_fold=q_fold,
            distance="torus",
        )

        # Sort and plot pre-clusters
        sorted_clusters = clusters
        cluster_sizes = [len(c) for c in sorted_clusters]
        data_by_cluster, _ = sort_data_into_cluster(dihedral_angles, sorted_clusters, min_cluster_size)
        proc_by_cluster, _ = sort_data_into_cluster(procrustes_full, sorted_clusters, min_cluster_size)
        back_by_cluster, _ = sort_data_into_cluster(procrustes_backbone, sorted_clusters, min_cluster_size)

        # Plot dihedral clusters
        qfold_dir = OUTPUT_DIR / name / str(q_fold)
        ensure_dir(qfold_dir)
        plot_functions.my_scatter_plots(
            data_by_cluster,
            filename=str(qfold_dir / f"{name}_outlier{max_outlier_dist_percent}_qfold{q_fold}"),
            set_title=f"dihedral angles suites {name}",
            number_of_elements=cluster_sizes,
            legend=True,
            s=30,
            legend_with_clustersize=True,
        )

        # Step 2: Mode hunting & PCA clustering
        mode_clusters, _ = PNDS_RNA_clustering.new_multi_slink(
            scale=12000,
            data=dihedral_angles,
            cluster_list=clusters,
            outlier_list=outliers,
            min_cluster_size=min_cluster_size,
        )
        mode_sizes = [len(c) for c in mode_clusters]
        stacked = np.vstack([dihedral_angles[c] for c in mode_clusters])
        plot_functions.my_scatter_plots(
            stacked,
            filename=str(qfold_dir / f"{name}_mode_outlier{max_outlier_dist_percent}_qfold{q_fold}"),
            set_title=f"dihedral angles suites {name}",
            number_of_elements=mode_sizes,
            legend=True,
            s=45,
            legend_with_clustersize=True,
        )

        # Save cluster indices
        write_files.write_data_to_pickle(mode_clusters, str(pickle_mode))
        write_files.write_data_to_pickle(clusters, str(pickle_base))

    else:
        mode_clusters = write_files.read_data_from_pickle(str(pickle_mode))
        clusters = write_files.read_data_from_pickle(str(pickle_base))

    # Step 3: Cluster merging
    merged = cluster_merging(mode_clusters, dihedral_angles, plot=False)
    merged_sizes = [len(c) for c in merged]
    merged_data = np.vstack([dihedral_angles[c] for c in merged])
    plot_functions.my_scatter_plots(
        merged_data,
        filename=str(qfold_dir / f"{name}_mode_merged_outlier{max_outlier_dist_percent}_qfold{q_fold}"),
        set_title=f"dihedral angles suites {name}",
        number_of_elements=merged_sizes,
        legend=True,
        s=45,
        legend_with_clustersize=True,
    )

    # Optional extra plots
    # plot_all_cluster_combinations(dihedral_angles, clusters, folder_plots, name, q_fold, max_outlier_dist_percent, mode=False)
    # plot_low_res(clusters, procrustes_full, procrustes_backbone, name, f"_mode_outlier{max_outlier_dist_percent}_qfold{q_fold}", folder_plots=qfold_dir)
    # create_csv(filtered, merged, name)


if __name__ == "__main__":
    names = ['c3c3', 'c3c2', 'c2c3', 'c2c2', 'all']
    min_size = 3
    configurations = [
        ('c2c2', 0.02, 0.05),
        ('c2c3', 0.02, 0.07),
        ('c3c2', 0.02, 0.05),
        ('c3c3', 0.02, 0.09),
    ]
    for name, outlier, qf in configurations:
        cluster_pruned_rna(
            name,
            min_cluster_size=min_size,
            max_outlier_dist_percent=outlier,
            q_fold=qf,
            do_clustering=True,
        )
