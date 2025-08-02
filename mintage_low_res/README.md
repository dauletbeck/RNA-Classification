# MINTAGE Low-Resolution RNA Analysis

A clean, refactored implementation of the low-resolution RNA structure analysis pipeline, organized into a well-structured Python package.

## Overview

This package provides tools for analyzing RNA structure at low resolution, focusing on sugar pucker conformations and clustering analysis. It's a complete refactoring of the original `low_res_pipeline.ipynb` notebook with improved organization, modularity, and maintainability.

## Features

- **Clean Architecture**: Well-organized modules with clear separation of concerns
- **Type Safety**: Full type hints throughout the codebase
- **Caching**: Automatic caching of expensive computations
- **Configuration**: Easy-to-use configuration system
- **Visualization**: Comprehensive plotting and analysis tools
- **Testing**: Full test suite with unit and integration tests

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RNA-Classification/mintage_low_res

# Install dependencies (you may want to use a virtual environment)
pip install numpy scipy matplotlib scikit-learn pandas jupyter pytest

# Optional: Install in development mode
pip install -e .
```

## Quick Start

### Method 1: Complete Pipeline (Recommended)

```python
from mintage_low_res import run_low_res_experiments
from mintage_low_res.config import ExperimentConfig

# Configure the analysis
config = ExperimentConfig(
    min_cluster_size=3,
    pns_scale=12000,
    pucker_types=['c2c2', 'c2c3', 'c3c2', 'c3c3']
)

# Run the complete analysis pipeline
results = run_low_res_experiments(
    pdb_directory="/path/to/pdb/files/",
    config=config,
    cache_dir="cache",
    output_dir="results"
)

# Results contain everything: parsed suites, clusters, statistics, plots
print(f"Analyzed {len(results['suites'])} RNA suites")
print(f"Found {len(results['pucker_results'])} pucker types")
```

### Method 2: Step-by-Step Analysis

```python
from mintage_low_res.parsing import parse_pdb_files
from mintage_low_res.preprocessing import scale_coordinates, determine_pucker_data
from mintage_low_res.clustering import pre_clustering, refine_clusters_with_pns
from mintage_low_res.visualization import create_scatter_plots

# Step 1: Parse PDB files
suites = parse_pdb_files("/path/to/pdb/files/")

# Step 2: Scale coordinates
scaled_coords, lambda_d, lambda_alpha = scale_coordinates(suites)

# Step 3: Analyze specific pucker type
pucker_indices, pucker_suites = determine_pucker_data(suites, 'c3c3')
pucker_coords = scaled_coords[pucker_indices]

# Step 4: Cluster analysis
from scipy.cluster.hierarchy import single as single_linkage

clusters, outliers, _ = pre_clustering(
    input_data=pucker_coords,
    m=3,  # min cluster size
    percentage=0.0,
    string_folder="output",
    method=single_linkage,
    q_fold=0.35,
    distance="low_res_suite_shape"
)

# Step 5: Visualize results
create_scatter_plots(
    data_by_cluster=pucker_coords,
    filename="c3c3_clusters",
    set_title="C3'-C3' Pucker Clusters"
)
```

## Project Structure

```
mintage_low_res/
├── config/                 # Configuration and constants
│   ├── constants.py        # RNA atom definitions, plotting settings
│   └── experiment_config.py # Experiment parameters and settings
├── data/                   # Data models and caching
│   ├── models.py          # Suite, AtomData, Residue classes
│   └── cache.py           # Pickle cache management
├── parsing/                # PDB parsing and Suite creation
│   ├── pdb_parser.py      # Main parsing interface
│   ├── atom_extractor.py  # Atomic coordinate extraction
│   └── suite_builder.py   # Suite object construction
├── preprocessing/          # Data preprocessing
│   ├── coordinate_scaling.py     # Coordinate scaling and normalization
│   ├── pucker_analysis.py       # Sugar pucker classification
│   └── spherical_transforms.py  # Spherical coordinate transformations
├── clustering/             # Clustering algorithms
│   ├── hierarchical_clustering.py # Pre-clustering with outlier detection
│   ├── pns_clustering.py         # Principal Nested Spheres clustering
│   └── cluster_refinement.py     # PNS-based cluster refinement
├── analysis/               # High-level analysis orchestration
│   ├── experiment_runner.py     # Main experiment pipeline
│   └── pucker_experiments.py    # Pucker-specific analysis
├── visualization/          # Plotting and visualization
│   ├── plotting.py              # Core plotting functions
│   └── results_visualization.py # Results and summary plots
├── utils/                  # Utility functions
│   ├── geometry.py        # Geometric calculations
│   └── io_utils.py        # File I/O utilities
├── notebooks/              # Clean example notebooks
│   └── low_res_pipeline_clean.ipynb
└── tests/                  # Test suite
    ├── test_*.py          # Unit tests
    ├── test_integration.py # Integration tests
    └── conftest.py        # Test fixtures
```

## Key Concepts

### RNA Suite
A Suite represents a pair of consecutive RNA nucleotides with:
- **Atomic coordinates**: Backbone, ring, and hydrogen atoms
- **Low-resolution coordinates**: 7D representation [d₂, d₃, α, θ₁, φ₁, θ₂, φ₂]
- **Sugar pucker data**: ν₁, ν₂ angles for pucker classification
- **Chain representations**: 5-chain, 6-chain, 7-chain coordinate sets

### Pucker Types
- **c2c2**: Both sugars in C2'-endo conformation
- **c2c3**: First sugar C2'-endo, second sugar C3'-endo  
- **c3c2**: First sugar C3'-endo, second sugar C2'-endo
- **c3c3**: Both sugars in C3'-endo conformation

### Analysis Pipeline
1. **Parse PDB files** → Extract atomic coordinates into Suite objects
2. **Scale coordinates** → Normalize variances and preserve means
3. **Classify puckers** → Group suites by sugar pucker conformations
4. **Hierarchical clustering** → Initial clustering with outlier removal
5. **PNS transformation** → Map to spherical coordinates 
6. **Cluster refinement** → Mode hunting with Principal Nested Spheres
7. **Visualization** → Generate plots and analysis summaries

## Running Tests

```bash
# Run unit tests
python tests/run_tests.py

# Run integration tests (slower)
python tests/run_tests.py integration

# Or use pytest directly
pytest tests/ -v
```

## Configuration

Customize analysis through the `ExperimentConfig` class:

```python
from mintage_low_res.config import ExperimentConfig

config = ExperimentConfig(
    min_cluster_size=3,           # Minimum points per cluster
    pns_scale=12000,             # PNS mode hunting scale
    pucker_types=['c2c2', 'c3c3'] # Pucker types to analyze
)

# Access optimal parameters found from experiments
optimal_q_fold = config.get_optimal_q_fold('c3c3')  # Returns 0.35
```

## Output Files

The pipeline generates:
- **Cache files**: Parsed suites, scaled coordinates (`.pkl`)
- **Cluster results**: Hierarchical and refined clusters
- **Plots**: Scatter plots, statistics, timing summaries (`.png`)
- **Analysis summaries**: Comprehensive experiment results

## Comparison with Original Notebook

| Aspect | Original Notebook | Refactored Package |
|--------|------------------|-------------------|
| **Organization** | Single 1000+ line notebook | Modular package with 20+ files |
| **Reusability** | Copy-paste code blocks | Import and use functions |
| **Testing** | Manual verification | Automated test suite |
| **Configuration** | Hard-coded parameters | Configurable settings |
| **Caching** | Manual pickle management | Automatic caching system |
| **Documentation** | Inline comments | Full API documentation |
| **Error Handling** | Basic try-catch | Comprehensive error handling |
| **Type Safety** | No type hints | Full type annotations |

## Contributing

1. Follow the existing code organization and style
2. Add tests for new functionality
3. Update documentation for API changes
4. Use type hints throughout
5. Maintain backward compatibility with the original notebook interface

## License

This project is part of the RNA-Classification research repository. Please refer to the main repository for licensing information.