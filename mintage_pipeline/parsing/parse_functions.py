"""
This functions are help functions for the starting point 'main_work_with_suites.py'. They return a list of suites and
are wrapper functions for all main steps described in 'main_work_with_suites.py'.
"""
import os
import pickle

import parse_pdb_and_create_suites
import read_base_pairs
import read_clash_files
import read_erraser_output
import shape_analysis


def parse_pdb_files(input_string_folder, input_pdb_folder=None):
    """
    This function creates a list with suite objects from all pdb files stored in the folder 'input_pdb_folder'.
    :param input_string_folder:  A string to store/load the results.
    :param input_pdb_folder:  A string. The path where the pdb files are stored.
    :return: A list with suite objects.
    """
    string_file = input_string_folder + 'suites_input.pickle'
    # Check if the results are stored in the folder input_string_folder. If True: load the results.
    if os.path.isfile(string_file):
        with open(string_file, 'rb') as f:
            suites_pdb = pickle.load(f)
    # Else: Use the functions from 'parse_pdb_and_create_suites.py' to create the list of suites.
    else:
        if input_pdb_folder is None:
            suites_pdb = parse_pdb_and_create_suites.get_all_pdb_files(folder='./pdb_data')
        else:
            suites_pdb = parse_pdb_and_create_suites.get_all_pdb_files(folder=input_pdb_folder)
        if not os.path.exists(input_string_folder):
            os.makedirs(input_string_folder)
        with open(string_file, 'wb') as f:
            pickle.dump(suites_pdb, f)
    return suites_pdb


def parse_clash_files(input_suites, input_string_folder, folder_validation_files=None, model_number=False):
    """
    This function needs a list of suites (input_suites) and adds the information about the clashes to the suite objects
    which are stored in the clash-files in the folder 'folder_validation_files'.
    :param input_suites: A list with suite objects.
    :param input_string_folder: A string to store/load the results.
    :param folder_validation_files:  A string. The path where the clash-files are stored.
    :param model_number: Boolean: True if we have more than one model stored in the pdb files.
    :return: The modified suite list.
    """
    string_file = input_string_folder + 'suites_clash.pickle'
    # Check if the results are stored in the folder input_string_folder. If True: load the results.
    if os.path.isfile(string_file):
        with open(string_file, 'rb') as f:
            suites_clash = pickle.load(f)
    # Else: Use the functions from 'read_clash_files.py' to create the list of suites.
    else:
        if folder_validation_files is None:
            suites_clash = read_clash_files.get_clashes(folder='./validation', suites=input_suites)
        else:
            suites_clash = read_clash_files.get_clashes(folder=folder_validation_files, suites=input_suites,
                                                        model_number=model_number)
        suites_clash = read_clash_files.work_with_clashes(suites_clash)
        if not os.path.exists(input_string_folder):
            os.makedirs(input_string_folder)
        with open(string_file, 'wb') as f:
            pickle.dump(suites_clash, f)
    return suites_clash


def parse_erraser_files(input_suites, input_string_folder):
    """
    This function gets a folder with the corrected ERRASER PDB files and adds the ERRASER corrected coordinates and some
    other information to the list of suite objects.
    :param input_suites: A list with suite objects.
    :param input_string_folder: A string to store/load the results.
    :return: The modified suite list.
    """
    string_file = input_string_folder + 'suites_erraser.pickle'
    # Check if the results are stored in the folder input_string_folder. If True: load the results.
    if os.path.isfile(string_file):
        with open(string_file, 'rb') as f:
            suites_pdb = pickle.load(f)
    # Else: Use the functions from 'read_erraser_output.py' to create the list of suites.
    else:
        suites_pdb = read_erraser_output.import_erraser_pdb(folder='./erraser_pdb_data', input_suites=input_suites)
        if not os.path.exists(input_string_folder):
            os.makedirs(input_string_folder)
        with open(string_file, 'wb') as f:
            pickle.dump(suites_pdb, f)
    return suites_pdb


def parse_clash_files_clash_score(input_suites, input_string_folder, folder_validation_files=None):
    """
    Not used at the moment.
    :param input_suites:
    :param input_string_folder:
    :param folder_validation_files:
    :return:
    """
    string_file = input_string_folder + 'suites_clash.pickle'
    if os.path.isfile(string_file):
        with open(string_file, 'rb') as f:
            suites_clash = pickle.load(f)
    else:
        if folder_validation_files is None:
            suites_clash = read_clash_files.get_clashes_clash_score(folder='./validation', suites=input_suites)
        else:
            suites_clash = read_clash_files.get_clashes_clash_score(folder=folder_validation_files, suites=input_suites)
        suites_clash = read_clash_files.work_with_clashes(suites_clash)
        if not os.path.exists(input_string_folder):
            os.makedirs(input_string_folder)
        with open(string_file, 'wb') as f:
            pickle.dump(suites_clash, f)
    return suites_clash


def parse_base_pairs(input_suites, input_string_folder):
    """
    Not used at the moment.
    :param input_suites: A list with suite objects.
    :param input_string_folder:  A string to store/load the results.
    :return: The modified suite list.
    """
    string_file = input_string_folder + 'suites_base_pairs.pickle'
    if os.path.isfile(string_file):
        with open(string_file, 'rb') as f:
            suites_base_pairs = pickle.load(f)
    else:
        suites_base_pairs = read_base_pairs.get_base_pairs(folder='./base_pairs', suites=input_suites)
        if not os.path.exists(input_string_folder):
            os.makedirs(input_string_folder)
        with open(string_file, 'wb') as f:
            pickle.dump(suites_base_pairs, f)
    return suites_base_pairs


def shape_analysis_suites(input_suites, input_string_folder, outlier_percentage=0.15, min_cluster_size=20,
                          overwrite=False, rerotate=False, old_data=False):
    """
    This function get a list of suite objects and adds all shape information to the list of suites.
    Used Algorithms:
    -Procrustes Algorithm
    -Clustering of the suites (Step 1: Pre clustering, Step 2: Torus PCA, Step 3: Mode Hunting).
    :param input_suites: A list with suite objects
    :param input_string_folder: A string to store/load the results.
    :param outlier_percentage: default 0.15
    :param min_cluster_size: default 20
    :param overwrite: overwrites the save file if it exists
    :param rerotate: in procrustes analysis: rotates data 'as in paper'
    :return: The modified suite list.
    """
    string_file = input_string_folder + 'suites_shape.pickle'
    # Check if the results are stored in the folder input_string_folder. If True: load the results.
    if os.path.isfile(string_file) and not overwrite:
        with open(string_file, 'rb') as f:
            suites_cluster = pickle.load(f)

    # Else: Use the functions from 'shape_analysis.py' to create the list of suites.
    else:
        # Determine the Procrustes information:
        suites_procrustes = shape_analysis.procrustes_analysis(input_suites, overwrite=overwrite, rerotate=rerotate,
                                                               old_data=old_data)
        # Cluster the suites:
        suites_cluster = shape_analysis.branch_cutting_with_correction(input_suites=suites_procrustes,
                                                                       m=min_cluster_size,
                                                                       percentage=outlier_percentage,
                                                                       clustering='suite', q_fold=0.15,
                                                                       clean=True)
        # Average Clustering of the mesoscopic shapes:
        shape_analysis.average_clustering(input_suites=suites_procrustes, m=5, percentage=0.5, clean=True)
        if not os.path.exists(input_string_folder):
            os.makedirs(input_string_folder)
        with open(string_file, 'wb') as f:
            pickle.dump(suites_cluster, f)
    return suites_cluster
