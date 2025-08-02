"""
Experiment configuration for pucker analysis and clustering.

Contains the configurations used in low_res_pipeline.ipynb.
"""

import numpy as np
from typing import Dict, List, Tuple

# Pucker types for RNA sugar conformations
PUCKER_TYPES = ['c2c2', 'c2c3', 'c3c2', 'c3c3']

# Default clustering parameters
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_PNS_SCALE = 12000

# Q-fold value ranges for different pucker types
# From the notebook experiments
PUCKER_Q_FOLD_RANGES = {
    'c2c2': np.round(np.arange(0.00, 0.101, 0.01), 3),
    'c2c3': np.round(np.arange(0.00, 0.101, 0.01), 3), 
    'c3c2': np.round(np.arange(0.00, 0.101, 0.01), 3),
    'c3c3': np.round(np.arange(0.00, 0.101, 0.01), 3),
}

# Optimal q_fold values found from experiments
OPTIMAL_Q_FOLD_VALUES = {
    'c2c2': 0.31,
    'c3c2': 0.37,
    'c3c3': 0.35,
    'c2c3': 0.05,  # Placeholder - add your optimal value
}

class ExperimentConfig:
    """Configuration class for low-resolution experiments."""
    
    def __init__(self, 
                 min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                 pns_scale: int = DEFAULT_PNS_SCALE,
                 pucker_types: List[str] = None):
        self.min_cluster_size = min_cluster_size
        self.pns_scale = pns_scale
        self.pucker_types = pucker_types or PUCKER_TYPES.copy()
    
    def get_q_fold_range(self, pucker_type: str) -> np.ndarray:
        """Get the q_fold range for a specific pucker type."""
        return PUCKER_Q_FOLD_RANGES.get(pucker_type, np.arange(0.0, 0.1, 0.01))
    
    def get_optimal_q_fold(self, pucker_type: str) -> float:
        """Get the optimal q_fold value for a pucker type."""
        return OPTIMAL_Q_FOLD_VALUES.get(pucker_type, 0.05)