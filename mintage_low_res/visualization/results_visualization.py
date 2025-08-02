"""
Results visualization for experiment summaries and analysis.

Provides high-level plotting functions for experiment results,
statistics, and comparative analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..utils.io_utils import ensure_directory_exists
from .plotting import create_comparison_plot


def plot_experiment_summary(results: Dict[str, Any],
                           output_dir: str = "plots") -> None:
    """
    Create comprehensive visualization of experiment results.
    
    Args:
        results: Complete experiment results dictionary
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if 'summary' not in results:
        print("No summary found in results")
        return
    
    summary = results['summary']
    
    # Plot pucker statistics
    if 'pucker_statistics' in summary:
        plot_pucker_statistics(
            summary['pucker_statistics'],
            str(output_path / "pucker_statistics.png")
        )
    
    # Plot timing information
    if 'timing_summary' in summary:
        plot_timing_summary(
            summary['timing_summary'],
            str(output_path / "timing_summary.png")
        )
    
    # Plot cluster statistics
    if 'cluster_statistics' in summary:
        plot_cluster_statistics(
            summary['cluster_statistics'],
            str(output_path / "cluster_statistics.png")
        )
    
    # Plot individual pucker results
    if 'pucker_results' in results:
        plot_pucker_cluster_results(
            results['pucker_results'],
            str(output_path / "pucker_cluster_results.png")
        )
    
    print(f"Summary plots saved to {output_path}")


def plot_pucker_statistics(pucker_stats: Dict[str, Dict[str, Any]],
                          output_path: str) -> None:
    """
    Plot statistics for different pucker types.
    
    Args:
        pucker_stats: Dictionary of pucker statistics
        output_path: Output file path
    """
    if not pucker_stats:
        return
    
    pucker_types = list(pucker_stats.keys())
    n_suites = [stats.get('n_suites', 0) for stats in pucker_stats.values()]
    n_clusters = [stats.get('n_final_clusters', 0) for stats in pucker_stats.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Number of suites per pucker type
    bars1 = ax1.bar(pucker_types, n_suites, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Number of Suites')
    ax1.set_title('Suites per Pucker Type')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, n_suites):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}', ha='center', va='bottom')
    
    # Number of final clusters per pucker type
    bars2 = ax2.bar(pucker_types, n_clusters, alpha=0.7, color='lightcoral')
    ax2.set_ylabel('Number of Final Clusters')
    ax2.set_title('Final Clusters per Pucker Type')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, n_clusters):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_timing_summary(timing_data: Dict[str, float],
                       output_path: str) -> None:
    """
    Plot timing information for different analysis steps.
    
    Args:
        timing_data: Dictionary of timing information
        output_path: Output file path
    """
    if not timing_data:
        return
    
    # Separate general timing from pucker-specific timing
    general_steps = {}
    pucker_steps = {}
    
    for key, value in timing_data.items():
        if key.startswith('analysis_'):
            pucker_type = key.replace('analysis_', '')
            pucker_steps[pucker_type] = value
        else:
            general_steps[key] = value
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # General timing
    if general_steps:
        steps = list(general_steps.keys())
        times = list(general_steps.values())
        
        bars1 = axes[0].bar(steps, times, alpha=0.7, color='lightgreen')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('General Processing Times')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.2f}s', ha='center', va='bottom')
    
    # Pucker-specific timing
    if pucker_steps:
        puckers = list(pucker_steps.keys())
        times = list(pucker_steps.values())
        
        bars2 = axes[1].bar(puckers, times, alpha=0.7, color='orange')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_title('Pucker Analysis Times')
        
        # Add value labels
        for bar, time in zip(bars2, times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_statistics(cluster_stats: Dict[str, Any],
                           output_path: str) -> None:
    """
    Plot overall cluster statistics.
    
    Args:
        cluster_stats: Dictionary of cluster statistics
        output_path: Output file path
    """
    if not cluster_stats:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a simple summary plot
    metrics = []
    values = []
    
    if 'total_final_clusters' in cluster_stats:
        metrics.append('Total Final\\nClusters')
        values.append(cluster_stats['total_final_clusters'])
    
    if 'average_clusters_per_pucker' in cluster_stats:
        metrics.append('Average Clusters\\nper Pucker')
        values.append(cluster_stats['average_clusters_per_pucker'])
    
    if metrics:
        bars = ax.bar(metrics, values, alpha=0.7, color='mediumpurple')
        ax.set_ylabel('Count')
        ax.set_title('Overall Cluster Statistics')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}' if isinstance(value, float) else f'{value}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pucker_cluster_results(pucker_results: Dict[str, Dict[str, Any]],
                               output_path: str) -> None:
    """
    Plot detailed results for each pucker type.
    
    Args:
        pucker_results: Dictionary of pucker-specific results
        output_path: Output file path
    """
    if not pucker_results:
        return
    
    pucker_types = list(pucker_results.keys())
    n_types = len(pucker_types)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = [
        ('n_suites', 'Number of Suites', 'lightblue'),
        ('n_hierarchical_clusters', 'Hierarchical Clusters', 'lightcoral'),
        ('n_mode_clusters', 'Final Mode Clusters', 'lightgreen'),
        ('analysis_time', 'Analysis Time (s)', 'orange')
    ]
    
    for i, (metric, title, color) in enumerate(metrics):
        if i >= len(axes):
            break
        
        ax = axes[i]
        values = []
        
        for pucker_type in pucker_types:
            result = pucker_results[pucker_type]
            if 'metadata' in result and metric in result['metadata']:
                values.append(result['metadata'][metric])
            else:
                values.append(0)
        
        bars = ax.bar(pucker_types, values, alpha=0.7, color=color)
        ax.set_ylabel(title.split('(')[0].strip())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            label = f'{value:.2f}' if isinstance(value, float) else f'{value}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom')
    
    plt.tight_layout()
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_q_fold_optimization_results(q_fold_results: Dict[str, Dict[float, Any]],
                                    output_path: str) -> None:
    """
    Plot q-fold optimization results.
    
    Args:
        q_fold_results: Dictionary of q-fold experiment results
        output_path: Output file path
    """
    if not q_fold_results:
        return
    
    pucker_types = list(q_fold_results.keys())
    n_types = len(pucker_types)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if n_types == 1:
        axes = [axes.flatten()[0]]
    else:
        axes = axes.flatten()[:n_types]
    
    for i, pucker_type in enumerate(pucker_types):
        if i >= len(axes):
            break
        
        ax = axes[i]
        results = q_fold_results[pucker_type]
        
        q_folds = sorted(results.keys())
        n_clusters = [results[qf]['n_clusters'] for qf in q_folds]
        
        ax.plot(q_folds, n_clusters, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Q-fold Parameter')
        ax.set_ylabel('Number of Clusters')
        ax.set_title(f'{pucker_type.upper()} Q-fold Optimization')
        ax.grid(True, alpha=0.3)
        
        # Highlight optimal point
        if n_clusters:
            max_idx = np.argmax(n_clusters)
            ax.plot(q_folds[max_idx], n_clusters[max_idx], 
                   'ro', markersize=10, alpha=0.7, 
                   label=f'Optimal: {q_folds[max_idx]:.2f}')
            ax.legend()
    
    # Remove unused subplots
    for i in range(n_types, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()