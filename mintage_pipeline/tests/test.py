import numpy as np
from PNDS_PNS_alt import pns_loop
from PNDS_geometry_alt import RESHify_1D, unRESHify_1D
from PNDS_io import export_csv
from plot_functions import scatter_plots
from matplotlib import pyplot as plt

def scatter_plots(input_data, filename, axis_min=None, axis_max=None, set_title=None, number_of_elements=None):
    if input_data.ndim == 1:
        # If input is 1D, reshape it to 2D
        input_data = input_data.reshape(1, -1)
    
    n = input_data.shape[1]
    if n < 2:
        print(f"Warning: Cannot create scatter plot for data with {n} dimensions.")
        return

    if number_of_elements is None:
        number_of_elements = input_data.shape[0]

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(set_title, fontsize=16)

def test_principal_nested_spheres():
    # Generate sample data
    np.random.seed(42)
    data = np.random.rand(100, 7) * 360  # 100 points, 7 dimensions

    # Prepare data for PNS
    sphere_points, means, half = RESHify_1D(data, False)

    # Run PNS
    spheres, projected_points, distances = pns_loop(sphere_points, 10, 10, degenerate=False, verbose=True, mode='torus', half=half)

    # Print some results
    print("Original data shape:", data.shape)
    print("Projected points shapes:")
    for i, points in enumerate(projected_points):
        print(f"  Level {i}: {points.shape}")

    try:
        # Unfold the results
        unfolded_data = unRESHify_1D(projected_points[-2], means, half)
        print("Unfolded data shape:", unfolded_data.shape)

        # Plot the results
        scatter_plots(data, filename='original_data', axis_min=0, axis_max=360, 
                    set_title='Original Data', number_of_elements=None)
        scatter_plots(unfolded_data, filename='unfolded_data', axis_min=0, axis_max=360, 
                    set_title='Unfolded Data', number_of_elements=None)
    except IndexError as e:
        print(f"Error during unfolding: {str(e)}")
        print("This error might be due to PNS reducing the dimensionality more than expected.")

    # Plot the final projection
    final_projection = projected_points[-1]
    if final_projection.ndim == 1:
        final_projection = final_projection.reshape(1, -1)
    scatter_plots(final_projection, filename='final_projection', 
                set_title='Final Projection', number_of_elements=None)

    # Export results
    try:
        export_csv({'original_data': data, 'projected_data': unfolded_data}, 'pns_results.csv')
        print("Results exported to pns_results.csv")
    except NameError:
        print("Unfolding failed, exporting only original data")
        export_csv({'original_data': data}, 'pns_results.csv')
if __name__ == "__main__":
    test_principal_nested_spheres()