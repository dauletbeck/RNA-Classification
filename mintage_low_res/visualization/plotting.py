"""
Core plotting functions for cluster visualization.

Provides scatter plots and cluster visualization functionality
matching the interface used in the original experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from ..config.constants import COLORS_SCATTER, MARKERS, LOW_RES_LABELS, LOW_RES_RANGES
from ..utils.io_utils import ensure_directory_exists


def create_scatter_plots(data_by_cluster: np.ndarray,
                        filename: str,
                        set_title: str,
                        suite_titles: List[str] = None,
                        list_ranges: List[List[float]] = None,
                        number_of_elements: List[int] = None,
                        legend: bool = True,
                        s: int = 20,
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create scatter plots for clustered data.
    
    This function matches the interface of the original scatter_plots function
    used in the notebook experiments.
    
    Args:
        data_by_cluster: Data organized by clusters
        filename: Output filename (without extension)
        set_title: Plot title
        suite_titles: Labels for each dimension
        list_ranges: Ranges for each dimension
        number_of_elements: Number of elements in each cluster
        legend: Whether to show legend
        s: Point size
        figsize: Figure size
    """
    if suite_titles is None:
        suite_titles = LOW_RES_LABELS
    
    if list_ranges is None:
        list_ranges = LOW_RES_RANGES
    
    n_dims = min(len(suite_titles), data_by_cluster.shape[1])
    
    # Create subplots
    n_cols = 3 if n_dims >= 6 else 2
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(set_title, fontsize=16)
    
    # Plot each pair of dimensions
    plot_idx = 0
    cluster_start = 0
    
    for i in range(0, n_dims - 1, 2):
        if plot_idx >= n_rows * n_cols:
            break
        
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        # Plot each cluster
        for cluster_idx, n_elements in enumerate(number_of_elements or [len(data_by_cluster)]):
            if cluster_start + n_elements > len(data_by_cluster):
                break
            
            cluster_data = data_by_cluster[cluster_start:cluster_start + n_elements]
            
            if i + 1 < n_dims and cluster_data.shape[1] > i + 1:
                ax.scatter(cluster_data[:, i], cluster_data[:, i + 1],
                          c=COLORS_SCATTER[cluster_idx % len(COLORS_SCATTER)],
                          marker=MARKERS[cluster_idx % len(MARKERS)],
                          s=s,
                          alpha=0.7,
                          label=f'Cluster {cluster_idx + 1} (n={n_elements})')
            
            cluster_start += n_elements
        
        # Set labels and ranges
        if i < len(suite_titles):
            ax.set_xlabel(suite_titles[i])
        if i + 1 < len(suite_titles):
            ax.set_ylabel(suite_titles[i + 1])
        
        if i < len(list_ranges):
            ax.set_xlim(list_ranges[i])
        if i + 1 < len(list_ranges):
            ax.set_ylim(list_ranges[i + 1])
        
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
        cluster_start = 0  # Reset for next subplot
    
    # Remove empty subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    # Add legend
    if legend and number_of_elements:
        fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    ensure_directory_exists(filename)
    plt.savefig(f"{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_results(clusters: List[List[int]],
                        data: np.ndarray,
                        output_dir: str,
                        pucker_type: str,
                        title_prefix: str = "Clusters") -> None:
    """
    Plot cluster results for a specific pucker type.
    
    Args:
        clusters: List of clusters (each cluster is list of indices)
        data: Full dataset
        output_dir: Output directory
        pucker_type: Pucker type being analyzed
        title_prefix: Prefix for plot title
    """
    if not clusters:
        print(f"No clusters to plot for {pucker_type}")
        return
    
    # Prepare data by cluster
    cluster_sizes = [len(cluster) for cluster in clusters]
    
    # Create concatenated data array
    data_by_cluster = []
    for cluster in clusters:
        if len(cluster) > 0:
            cluster_data = data[cluster]
            data_by_cluster.append(cluster_data)
    
    if not data_by_cluster:
        return
    
    data_by_cluster = np.vstack(data_by_cluster)
    
    # Create plots
    output_path = Path(output_dir) / f"{pucker_type}_clusters"
    
    create_scatter_plots(
        data_by_cluster=data_by_cluster,
        filename=str(output_path),
        set_title=f"{title_prefix} - {pucker_type.upper()}",
        number_of_elements=cluster_sizes,
        legend=True
    )
    
    print(f"Cluster plots saved: {output_path}.png")


def plot_coordinate_distributions(data: np.ndarray,
                                 output_path: str,
                                 title: str = "Coordinate Distributions") -> None:
    """
    Plot distributions of low-resolution coordinates.
    
    Args:
        data: Coordinate data [N x 7]
        output_path: Output file path
        title: Plot title
    """
    n_coords = min(7, data.shape[1])
    labels = LOW_RES_LABELS[:n_coords]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_coords):
        ax = axes[i]
        ax.hist(data[:, i], bins=50, alpha=0.7, density=True)
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Density')
        ax.set_title(f'{labels[i]} Distribution')
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    if n_coords < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_size_distribution(cluster_sizes: List[int],
                                  output_path: str,
                                  title: str = "Cluster Size Distribution") -> None:
    """
    Plot distribution of cluster sizes.
    
    Args:
        cluster_sizes: List of cluster sizes
        output_path: Output file path
        title: Plot title
    """
    if not cluster_sizes:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(cluster_sizes, bins=min(20, len(set(cluster_sizes))), alpha=0.7)
    ax1.set_xlabel('Cluster Size')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Cluster Size Histogram')
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of individual clusters
    cluster_indices = range(1, len(cluster_sizes) + 1)
    bars = ax2.bar(cluster_indices, cluster_sizes, alpha=0.7)
    ax2.set_xlabel('Cluster Index')
    ax2.set_ylabel('Cluster Size')
    ax2.set_title('Individual Cluster Sizes')
    ax2.grid(True, alpha=0.3)
    
    # Color bars by size
    max_size = max(cluster_sizes)
    for bar, size in zip(bars, cluster_sizes):
        bar.set_color(plt.cm.viridis(size / max_size))
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_plot(results_dict: Dict[str, Any],
                          output_path: str,
                          metric: str = 'n_clusters') -> None:
    """
    Create comparison plots across different pucker types or parameters.
    
    Args:
        results_dict: Dictionary of results to compare
        output_path: Output file path
        metric: Metric to compare
    """
    names = list(results_dict.keys())
    values = []
    
    for name, result in results_dict.items():
        if isinstance(result, dict) and metric in result:
            values.append(result[metric])
        else:
            values.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(names, values, alpha=0.7)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()