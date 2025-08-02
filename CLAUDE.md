# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an RNA classification research project with two main components:
- **MINTAGE**: Low-resolution RNA structure analysis pipeline
- **RNAprecis**: High-resolution RNA structure learning system

The project focuses on classifying RNA sugar pucker conformations (c2c2, c2c3, c3c2, c3c3) using geometric analysis and machine learning techniques.

## Repository Structure

- `mintage/`: Core MINTAGE pipeline for low-resolution RNA analysis
  - `utils/Suite_class.py`: Central Suite class representing RNA nucleotide pairs
  - `pnds/`: Principal Nested Data Structure analysis modules
  - `clustering/`: Gaussian mixture models and mode hunting algorithms
  - `geometry/`: Hypersphere and geometric utility functions
  - `parsing/`: PDB file parsing and data extraction
- `RNAprecis/`: High-resolution structure learning components
- `data/`: Contains PDB files and cluster comparison data
  - `rna2020_pruned_pdbs/`: Primary dataset of RNA structures

## Core Architecture

### Suite Class (`mintage/utils/Suite_class.py`)
The fundamental data structure representing RNA nucleotide pairs with:
- Microscopic level: backbone atoms, hydrogen atoms, oxygen atoms, ring atoms
- Mesoscopic level: sugar ring coordinates, five/six/seven-chain representations
- Dihedral angles: backbone conformations and chi angles

### Processing Pipeline
1. **Parse PDB files** → Extract atomic coordinates into Suite objects
2. **Scale coordinates** → Normalize distance and angle variances
3. **Pre-clustering** → Hierarchical clustering with outlier detection
4. **PNS Analysis** → Principal Nested Spheres for geometric analysis
5. **Mode hunting** → Gaussian mixture model clustering refinement

### Key Algorithms
- **PNS (Principal Nested Spheres)**: Geometric analysis on spherical data structures
- **Mode hunting**: Gaussian mixture model-based cluster refinement
- **Procrustes analysis**: Shape alignment for structural comparison

## Development Commands

### Running Analysis
```bash
# Navigate to mintage directory for main pipeline
cd mintage

# Run low-resolution pipeline (Jupyter notebook)
jupyter notebook low_res_pipeline.ipynb

# Run modular MINT-AGE pipeline
python mint_age.py

# Run high-resolution learning
cd ../RNAprecis
python main_learn_high_res_structure.py
```

### Python Environment
- Python 3.11+ required
- Primary dependencies: numpy, scipy, matplotlib, pandas
- No formal requirements.txt found - dependencies imported directly in code

### Testing
- Test files located in `mintage/tests/`
- No automated test runner configured
- Manual testing through example scripts in `mintage/examples/`

## Key Files for Development

- `mintage/low_res_pipeline.ipynb`: Main analysis workflow notebook
- `mintage/utils/Suite_class.py`: Core data structure (line 6 contains Suite class definition)
- `mintage/pnds/PNDS_PNS.py`: **Principal Nested Spheres implementation** - Recently refactored for improved readability
  - `PNS` class (line 164): Main PNS fitting algorithm with statistical model selection
  - `fit()` method (line 196): Primary entry point for PNS fitting process
  - `_fit_sphere_at_dimension()` (line 279): Core sphere fitting logic for each dimension
  - `_perform_sphere_fitting()` (line 591): Routes to appropriate fitting method based on mode
- `mintage/pnds/PNS_mode_hunter.py`: Mode hunting implementation (line 48 contains PNS import)
- `mintage/shape_analysis.py`: Procrustes and pre-clustering functions
- `RNAprecis/main_learn_high_res_structure.py`: High-resolution learning entry point (line 38 contains learn_algorithm function)

## Data Processing Notes

- Input: PDB files from `data/rna2020_pruned_pdbs/`
- Intermediate: Pickled cluster results and scaled coordinates
- Output: Cluster assignments and analysis plots
- Pucker types: c2c2, c2c3, c3c2, c3c3 classifications

## Recent Code Improvements

### Principal Nested Spheres (PNS) Refactoring (2025)

The PNS implementation in `mintage/pnds/PNDS_PNS.py` was significantly refactored to improve code readability, maintainability, and developer experience:

#### Key Improvements Made:
- **Enhanced Method Names**: Renamed cryptic method names to descriptive, self-documenting names
- **Better Variable Naming**: Replaced abbreviated variables with clear, meaningful names
- **Improved Code Organization**: Logical separation of concerns with clear method hierarchy
- **Comprehensive Documentation**: Added detailed docstrings with parameters and return types
- **Enhanced Error Handling**: Better error messages and verbose output for debugging

#### Major Method Renamings:
- `_get_sphere()` → `_fit_sphere_at_dimension()` - Core sphere fitting for each dimension
- `_choose_mode()` → `_choose_fitting_mode()` - Select fitting strategy based on dimension
- `_nested_mean()` → `_compute_circular_mean_2d()` - Circular statistics for 2D points
- `_get_functions()` → `_create_sphere_objective_functions()` - Generate optimization functions
- `_small_circle_fit()` → `_fit_small_sphere_multistart()` - Multi-start small sphere fitting
- `_fit_function()` → `_fit_objective_function()` - Robust least squares optimization
- `_new_seed()` → `_generate_well_separated_seed()` - Generate well-separated initial points

#### Improved Variable Names:
- `X` → `data_matrix` - Input data for PNS fitting
- `list_spheres` → `fitted_spheres`/`previous_spheres` - Clearer sphere collections
- `N, d` → `n_samples, current_dim` - Explicit dimension terminology
- `f, f2, g, g2` → `spherical_distance_objective`, `great_sphere_objective`, etc.

#### New Code Structure:
- **Main Workflow**: `fit()` → `_fit_sphere_at_dimension()` → `_perform_sphere_fitting()`
- **Specialized Fitting**: `_fit_great_sphere()`, `_fit_torus_sphere()`, `_fit_small_sphere_with_test()`
- **Utility Methods**: `_handle_insufficient_data_case()`, `_reset_fitting_results()`

The refactored code maintains full backward compatibility while being significantly more readable and easier to maintain.

## PNS Implementation Details

### Overview
Principal Nested Spheres (PNS) is a dimensionality reduction technique that fits a sequence of nested spheres to high-dimensional data on the unit sphere. The implementation supports three fitting modes:

- **Great Sphere Mode**: Fits spheres passing through the origin (radius = 0)
- **Torus Mode**: Optimizes sphere parameters using torus-based distance metrics
- **Scale Mode**: Uses statistical model selection to choose between great and small spheres

### Core Workflow
```
fit(data_matrix) 
├── _fit_sphere_at_dimension()
│   ├── _choose_fitting_mode() 
│   └── _perform_sphere_fitting()
│       ├── _fit_great_sphere() [for great mode]
│       ├── _fit_torus_sphere() [for torus mode]  
│       └── _fit_small_sphere_with_test() [for scale mode]
└── _compute_circular_mean_2d() [final 2D case]
```

### Key Methods Documentation

#### Primary Interface
- **`fit(data_matrix)`**: Main entry point. Fits nested spheres iteratively until reaching 2D
- **`_choose_fitting_mode(dimension)`**: Selects fitting strategy based on current dimension and configuration

#### Sphere Fitting Methods
- **`_fit_sphere_at_dimension()`**: Handles sphere fitting at current dimension with error recovery
- **`_perform_sphere_fitting()`**: Routes to appropriate fitting method based on mode
- **`_fit_great_sphere()`**: Fits great sphere using spherical distance minimization
- **`_fit_torus_sphere()`**: Two-stage fitting: spherical distance then torus distance optimization
- **`_fit_small_sphere_with_test()`**: Fits small sphere with statistical test for great vs small preference

#### Optimization Infrastructure
- **`_create_sphere_objective_functions()`**: Creates optimization functions for different sphere types
- **`_fit_objective_function()`**: Robust least squares optimization with automatic restarts
- **`_fit_small_sphere_multistart()`**: Multi-start optimization to avoid local minima
- **`_generate_well_separated_seed()`**: Generates initial seeds with >45° separation

#### Utility Methods
- **`_compute_circular_mean_2d()`**: Computes circular mean and residuals for final 2D points
- **`_handle_insufficient_data_case()`**: Handles edge cases with insufficient data
- **`_reset_fitting_results()`**: Resets results when errors occur

### Usage Example
```python
from pnds.PNDS_PNS import PNS
import numpy as np

# Prepare sphere data
data = np.random.randn(100, 5)
data /= np.linalg.norm(data, axis=1, keepdims=True)

# Fit PNS with torus mode until dimension 3
pns = PNS(great_until_dim=3, max_repetitions=10, verbose=True, mode='torus')
pns.fit(data)

# Access results
spheres = pns.spheres_      # Fitted sphere objects
points = pns.points_        # Projected points at each level  
distances = pns.dists_      # Distances from spheres in degrees
```

### Configuration Parameters
- **`great_until_dim`**: Use great sphere mode until this dimension (default: 2)
- **`max_repetitions`**: Maximum optimization attempts for robustness (default: 10)
- **`verbose`**: Enable detailed progress output (default: False)
- **`mode`**: Force specific fitting mode: 'great', 'torus', or None for automatic (default: None)
- **`half`**: Use half-space for torus distance calculations (default: False)

## Working with the Codebase

- Most analysis is done through Jupyter notebooks rather than scripts
- Heavy use of pickle files for intermediate data storage
- Plotting functions integrated throughout - check `utils/plot_functions.py`
- When working with Suite objects, access geometric data through private attributes (`_backbone_atoms`, `_dihedral_angles`, etc.)