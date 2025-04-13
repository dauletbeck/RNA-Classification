"""
This python script is the starting point of this project:
Step 1: Use the methods in parse_pdb_and_create_suites.py to create a list of objects of the Suite-type (see
Suite_class.py).
Step 2: Use the methods in read_clash_files.py to extend the suites with information about atom-clashes.
Step 3: Use the methods in shape_analysis to extend the suites with information about the shape. I.e. use the procrustes
        algorithm to mean align the shapes and cluster the data.
Step 4: Parse the ERRASER files. And make statistics with the ERRASER data.
Step 5: Apply CLEAN (CLassification basEd on mutliscAle eNhancement) .
Step 6: Apply CLEAN to two examples of the Corona data set.
"""
import numpy as np

from clash_correction import multiscale_correction
from clean_mintage_code.plot_functions import build_fancy_chain_plot
from shape_analysis import shape_six_chain, shape_five_chain
from corona_different_models import working_with_different_models
from parse_functions import parse_pdb_files, parse_clash_files, parse_erraser_files, shape_analysis_suites

# Starting point:
if __name__ == '__main__':
    string_folder = './out/saved_suite_lists/'
    # dataset 05:
    # pdb_folder = None
    pdb_folder = './rna2020_pruned_pdbs/'
    # Step 1:
    suites = parse_pdb_files(input_string_folder=string_folder, input_pdb_folder=pdb_folder)
    # Step 2:
    suites = parse_clash_files(input_suites=suites, input_string_folder=string_folder)
    # Step 2b: Not in the code at the moment:
    # suites = parse_base_pairs(input_suites=suites, input_string_folder=string_folder)
    # Step 3:
    # dataset 05:
    # suites = shape_analysis_suites(input_suites=suites, input_string_folder=string_folder)
    five_chain_complete_suites = [suite for suite in suites if suite._five_chain[0] is not None
                                  and suite.dihedral_angles is not None and
                                  suite._five_chain[0][0] is not None and suite._five_chain[1][0] is not None
                                  and suite._five_chain[2][0] is not None and
                                  suite._five_chain[3][0] is not None and suite._five_chain[4][0] is not None]
    complete_five_chains = np.array([suite._five_chain for suite in five_chain_complete_suites])
    print(len(complete_five_chains))
    length_vector = []
    for vector in complete_five_chains:
        length = np.linalg.norm(vector)
        length_vector.append(length)
    print("min----------")
    index = length_vector.index(min(length_vector))
    print(five_chain_complete_suites[index]._name)
    print(five_chain_complete_suites[index]._filename)
    print(min(length_vector))
    print("max----------")
    index = length_vector.index(max(length_vector))
    print(five_chain_complete_suites[index]._name)
    print(five_chain_complete_suites[index]._filename)
    print(max(length_vector))
    print(np.mean(np.array(length_vector)))

    suites = shape_analysis_suites(input_suites=suites, input_string_folder=string_folder, outlier_percentage=0,
                                   min_cluster_size=1, overwrite=False, rerotate=True)
    shape_five_chain(input_suites=suites, input_string_folder='./out/saved_suite_lists/')
    shape_six_chain(input_suites=suites, input_string_folder='./out/saved_suite_lists/')

    # Step 4:
    suites = parse_erraser_files(input_suites=suites, input_string_folder=string_folder)
    # Step 5:
    suites = multiscale_correction(input_suites=suites)
    # Step 6:
    working_with_different_models(input_suites=suites)

    print('test')
