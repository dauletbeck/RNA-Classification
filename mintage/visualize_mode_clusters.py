#!/usr/bin/env python3
"""
Script to visualize post-clustering results from PNS mode hunting.
Loads mode_clusters_results.pkl and creates scatter plots for each pucker type.
"""

import pickle
import numpy as np
from utils.plot_functions import my_scatter_plots
from utils.scale_low_res_coordinates import scale_low_res_coords
from utils.pucker_data_functions import determine_pucker_data
from parsing.parse_functions import parse_pdb_files

def load_and_visualize_clusters():
    """Load clustering results and create visualizations"""
    
    # Load the post-clustering results
    with open('/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/mintage/postclustering_results/minimal_q_fold_no_outlier_premerged_postcluster.pkl', 'rb') as f:
        mode_clusters_results = pickle.load(f)
    
    # Also load the original data for visualization
    input_pdb_dir = "/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/data/rna2020_pruned_pdbs/"
    suites = parse_pdb_files(input_pdb_dir, input_pdb_folder=input_pdb_dir)
    
    print(f"Loaded clustering results for {len(mode_clusters_results)} pucker types")
    
    # Process each pucker type
    for result in mode_clusters_results:
        pucker_name = result['name']
        mode_clusters = result['mode_clusters']
        
        print(f"\n=== {pucker_name.upper()} ===")
        print(f"Number of clusters: {len(mode_clusters)}")
        
        # Get the indices for this pucker type
        indices, _ = determine_pucker_data(suites, pucker_name)
        pucker_suites = [suites[i] for i in indices]
        
        # Scale coordinates
        scaled_coords, lambda_d, lambda_alpha = scale_low_res_coords(pucker_suites)
        
        # Create angle matrix (same as in low_res_script.py)
        angle_matrix = create_angle_matrix_from_scaled(scaled_coords)
        
        # Create cluster assignments for visualization
        cluster_assignments = create_cluster_assignments(mode_clusters, len(angle_matrix))
        
        # Sort clusters by size (descending order) for legend, but reverse for plotting order
        cluster_info = [(i, len(cluster), cluster) for i, cluster in enumerate(mode_clusters)]
        cluster_info.sort(key=lambda x: x[1], reverse=True)  # Sort by size descending
        
        # For legend (largest to smallest)
        legend_indices = [info[0] for info in cluster_info]
        legend_sizes = [info[1] for info in cluster_info]
        
        # For plotting (smallest to largest so biggest appears on top)
        plot_cluster_info = list(reversed(cluster_info))
        plot_clusters = [info[2] for info in plot_cluster_info]
        plot_sizes = [info[1] for info in plot_cluster_info]
        # reversing for the legend
        plot_sizes = plot_sizes[::-1]
        
        # Reorder angle_matrix data to match plotting order (smallest to largest)
        sorted_angle_matrix = []
        for cluster in plot_clusters:
            for idx in cluster:
                if idx < len(angle_matrix):
                    sorted_angle_matrix.append(angle_matrix[idx])
        sorted_angle_matrix = np.array(sorted_angle_matrix) if sorted_angle_matrix else angle_matrix
        
        if len(mode_clusters) > 0:
            # Create scatter plot
            filename = f'mode_clusters_{pucker_name}_visualization_min_q_merged_updated_merging'
            
            my_scatter_plots(
                input_data=sorted_angle_matrix,
                filename=filename,
                number_of_elements=plot_sizes,
                legend_with_clustersize=True,
                legend_titles=[f"Cluster {legend_indices[i]+1}" for i in range(len(legend_indices))],
                suite_titles=[r'$\theta_d$', r'$\phi_d$', r'$\alpha_s$', 
                             r'$\theta_1$', r'$\phi_1$', r'$\theta_2$', r'$\phi_2$'],
                axis_min=0,
                axis_max=360,
                markerscale=5,
                s=10,
                fontsize=30
            )
            
            print(f"Created visualization: {filename}.png")
            
            # Print cluster statistics (now sorted by size)
            print("Cluster sizes (sorted largest to smallest):", legend_sizes)
            total_points = sum(legend_sizes)
            print(f"Total points: {total_points}")
            for i, (orig_idx, size) in enumerate(zip(legend_indices, legend_sizes)):
                pct = (size / total_points) * 100 if total_points > 0 else 0
                print(f"  Cluster {orig_idx+1} (now position {i+1}): {size} points ({pct:.1f}%)")

def create_angle_matrix_from_scaled(scaled_coords):
    """Create the same 7D angle matrix as in low_res_script.py from scaled coordinates"""
    from pnds.PNDS_PNS import PNS
    
    # Extract from scaled coords
    d2_s, d3_s, alpha_s, theta1, phi1, theta2, phi2 = scaled_coords.T
    
    def spherical_to_vec(theta_deg, phi_deg):
        t, p = np.radians(theta_deg), np.radians(phi_deg)
        return np.column_stack([np.sin(t) * np.cos(p),
                                np.sin(t) * np.sin(p),
                                np.cos(t)])
    
    def exponential_map(V, p):
        """
        V - point cloud N x R^m
        p - point of tangency R^m
        """
        N, M = V.shape[0], V.shape[1]
        V_mean = V.mean(axis=0)
        # check if points are centered at 0, center them
        V -= V_mean
        V = np.column_stack([V, np.zeros(N)])
        V_norm = np.linalg.norm(V, axis=1)[:, None]
        
        return np.cos(V_norm) * p + np.sin(V_norm) * (V / V_norm)
    
    # Apply PNS to S2 data
    S2_1 = spherical_to_vec(theta1, phi1)
    pns_S2_1 = PNS(mode='great', verbose=False).fit(S2_1)
    theta1_pns, phi1_pns = pns_S2_1.dists_

    S2_2 = spherical_to_vec(theta2, phi2)
    pns_S2_2 = PNS(mode='great', verbose=False).fit(S2_2)
    theta2_pns, phi2_pns = pns_S2_2.dists_

    # Process distance data
    V = np.column_stack([d2_s, d3_s])
    V -= np.mean(V, axis=0)

    S2_d = exponential_map(V, p=np.array([0,0,1]))
    pns_S2_d = PNS(mode='great', verbose=False).fit(S2_d)
    theta_d, phi_d = pns_S2_d.dists_
    
    # Create final angle matrix
    angle_matrix = np.column_stack([
        theta_d + 180,
        phi_d + 180, 
        alpha_s,
        theta1_pns + 180,
        phi1_pns + 180,
        theta2_pns + 180,
        phi2_pns + 180
    ])
    
    return angle_matrix

def create_cluster_assignments(mode_clusters, total_points):
    """Create array indicating cluster membership for each point"""
    assignments = np.full(total_points, -1)  # -1 for outliers
    
    for cluster_id, cluster_indices in enumerate(mode_clusters):
        for idx in cluster_indices:
            if idx < total_points:
                assignments[idx] = cluster_id
                
    return assignments

if __name__ == "__main__":
    load_and_visualize_clusters()