"""
MINTAGE Low-Resolution RNA Analysis Pipeline

This package contains a clean, refactored version of the low-resolution RNA
structure analysis pipeline used in the low_res_pipeline.ipynb experiments.

Main Components:
- parsing: PDB file parsing and Suite object creation
- preprocessing: Coordinate scaling and pucker analysis
- clustering: Hierarchical and PNS-based clustering
- analysis: Experiment orchestration and pucker-specific analysis
- visualization: Plotting and results visualization
"""

__version__ = "1.0.0"
__author__ = "RNA Classification Research Team"

from .analysis.experiment_runner import run_low_res_experiments
from .parsing.pdb_parser import parse_pdb_files
from .preprocessing.coordinate_scaling import scale_coordinates
from .preprocessing.pucker_analysis import determine_pucker_data

__all__ = [
    "run_low_res_experiments",
    "parse_pdb_files", 
    "scale_coordinates",
    "determine_pucker_data"
]