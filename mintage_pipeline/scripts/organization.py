#!/usr/bin/env python3
import os
import shutil

# The name of the top-level directory where you want everything to be organized
TOP_LEVEL_DIR = "rna_classification"

# A mapping of each file/folder to the subdirectory (under TOP_LEVEL_DIR) where it should go.
# An empty string "" means "place directly under TOP_LEVEL_DIR."
DESTINATION_MAP = {
    # Top-level text/license/IDE files
    "GPL3.txt": "",  # will be moved and optionally renamed to LICENSE below
    "RNA_Project_Suites.iml": "",
    
    # Main workflow
    "main_work_with_suites.py": "main",

    # Parsing
    "parse_functions.py": "parsing",
    "parse_pdb_and_create_suites.py": "parsing",
    "read_base_pairs.py": "parsing",
    "read_clash_files.py": "parsing",
    "read_erraser_output.py": "parsing",

    # PNDS
    "PNDS_PNS.py": "pnds",
    "PNDS_PNS_alt.py": "pnds",
    "PNDS_RNA_clustering.py": "pnds",
    "PNDS_RNA_clustering_alt.py": "pnds",
    "PNDS_geometry.py": "pnds",
    "PNDS_geometry_alt.py": "pnds",
    "PNDS_io.py": "pnds",
    "PNDS_io_als.py": "pnds",
    "PNDS_plot.py": "pnds",
    "PNDS_plot_alt.py": "pnds",
    "PNDS_tree.py": "pnds",
    "PNDS_tree_alt.py": "pnds",

    # Clustering
    "cluster_improving.py": "clustering",
    "cluster_merging.py": "clustering",
    "gaussian_mixture_model_1d.py": "clustering",
    "gaussian_modehunting.py": "clustering",
    "gaussian_modehunting_v2.py": "clustering",

    # Multiscale analysis
    "Multiscale_modes.py": "multiscale_analysis",
    "multiscale_modes_linear.py": "multiscale_analysis",
    "corona_different_models.py": "multiscale_analysis",

    # Shape analysis
    "shape_analysis.py": "shape_analysis",

    # Utilities
    "Suite_class.py": "utils",
    "constants.py": "utils",
    "data_functions.py": "utils",
    "help_plot_functions.py": "utils",
    "plot_functions.py": "utils",
    "auxiliary_plot_functions.py": "utils",

    # Examples
    "PNDS_clusters_example.py": "examples",
    "PNDS_clusters_example_alt.py": "examples",

    # Data folders/files
    "rna2020_pruned_pdbs": "data",
    "cluster_data": "data",
    "trimmed_test_suites": "data",
    "suite_outliers_with_answers.csv": "data",
    "somefile.kin": "data",

    # Results/output folders
    "annotated_testing_results": "results",
    "cluster_comparison": "results",
    "out": "results",
    "test_cluster_merging_output": "results",
    "test_cluster_separation_output": "results",
    "test_output": "results",
    "validation": "results",

    # Figures/images
    "final_projection.png": "figures",
    "original_data.png": "figures",
    "unfolded_data.png": "figures",

    # Tests
    "test.py": "tests",
    "test.ipynb": "tests",

    # Misc
    "organization.py": "scripts",  # If you want your re-org script in a scripts folder
}

def main():
    # Create the top-level directory if it doesn't exist
    if TOP_LEVEL_DIR and not os.path.exists(TOP_LEVEL_DIR):
        os.makedirs(TOP_LEVEL_DIR)

    for item, subfolder in DESTINATION_MAP.items():
        src_path = os.path.join(".", item)
        if not os.path.exists(src_path):
            print(f"Warning: '{item}' not found. Skipping.")
            continue

        # Determine the destination directory (within TOP_LEVEL_DIR)
        if subfolder:
            dest_dir = os.path.join(TOP_LEVEL_DIR, subfolder)
        else:
            dest_dir = os.path.join(TOP_LEVEL_DIR)

        # Create the destination folder if needed
        os.makedirs(dest_dir, exist_ok=True)

        # Destination path for the file/folder
        dest_path = os.path.join(dest_dir, item)

        print(f"Moving '{item}' -> '{dest_path}'")
        shutil.move(src_path, dest_path)

    # Optionally rename GPL3.txt to LICENSE if you want a more standard name
    gpl_path = os.path.join(TOP_LEVEL_DIR, "GPL3.txt")
    license_path = os.path.join(TOP_LEVEL_DIR, "LICENSE")
    if os.path.exists(gpl_path):
        print(f"Renaming 'GPL3.txt' -> 'LICENSE'")
        os.rename(gpl_path, license_path)

    # Clean up or ignore cache folders if present
    pycache_dir = os.path.join(".", "__pycache__")
    if os.path.exists(pycache_dir):
        print(f"Removing '__pycache__' directory.")
        shutil.rmtree(pycache_dir)

    print("\nReorganization complete!")


if __name__ == "__main__":
    main()
