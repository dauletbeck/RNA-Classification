from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pickle
import os
import math
import pandas as pd
# import seaborn as sn
import csv
import requests
import re
from utils import write_files
from collections import Counter

from utils.data_functions import calculate_angle_3_points, rotation, rotation_matrix_x_axis, rotation_matrix_z_axis, mean_on_sphere_init
from parsing.parse_functions import parse_pdb_files
import shape_analysis
from shape_analysis import procrustes_on_suite_class

def determine_pucker_data(cluster_suites, pucker_name):
    """
    Determines the suites from cluster_suites belonging to sugar pucker pucker_name.
    :param cluster_suites: The complete list of training suites objects.
    :param pucker_name: 'c2c2', 'c3c3', 'c2c3' or 'c3c2'
    return: pucker indices, pucker suites
    """
    if (pucker_name == 'c_2_c_2_suites' or pucker_name == 'c2c2'):
        pucker_index_and_suites = [[i, suite] for i, suite in enumerate(cluster_suites) if
                                   300 < suite._nu_1[0] < 350 and 300 < suite._nu_2[0] < 350]
    elif (pucker_name == 'c_3_c_3_suites' or pucker_name == 'c3c3'):
        pucker_index_and_suites = [[i, suite] for i, suite in enumerate(cluster_suites) if
                                   not (300 < suite._nu_1[0] < 350) and not (300 < suite._nu_2[0] < 350)]
    elif (pucker_name == 'c_3_c_2_suites' or pucker_name == 'c3c2'):
        pucker_index_and_suites = [[i, suite] for i, suite in enumerate(cluster_suites) if
                                   not (300 < suite._nu_1[0] < 350) and 300 < suite._nu_2[0] < 350]
    elif (pucker_name == 'c_2_c_3_suites' or pucker_name == 'c2c3'):
        pucker_index_and_suites = [[i, suite] for i, suite in enumerate(cluster_suites) if
                                   300 < suite._nu_1[0] < 350 and not (300 < suite._nu_2[0] < 350)]
    elif pucker_name == 'all':
        pucker_index_and_suites = cluster_suites
    else:
        print("PUCKER-NAME not found!")
        pucker_index_and_suites = [[None, None]]
    if len(pucker_index_and_suites) < 1:
        pucker_index_and_suites = [[None, None]]

    pucker_index_and_suites = np.array(pucker_index_and_suites)

    return pucker_index_and_suites[:, 0], pucker_index_and_suites[:, 1]

def procrustes_for_each_pucker(input_suites, pucker_name):
    """
    Filters suites by pucker type and returns the procrustes data arrays for the filtered suites.
    Returns: filtered_suites, procrustes_data, procrustes_data_backbone
    """
    # Filter for suites with required attributes
    cluster_suites = [suite for suite in input_suites if getattr(suite, 'procrustes_five_chain_vector', None) is not None
                      and getattr(suite, 'dihedral_angles', None) is not None]
    cluster_suites = [suite for suite in cluster_suites if getattr(suite, 'atom_types', None) == 'atm']

    # Further filter by pucker type
    _, cluster_suites = determine_pucker_data(cluster_suites, pucker_name)
    print(f'{pucker_name} suites: {len(cluster_suites)}')

    procrustes_data = np.array([suite.procrustes_five_chain_vector for suite in cluster_suites])
    procrustes_data_backbone = np.array([suite.procrustes_complete_suite_vector for suite in cluster_suites])
    return cluster_suites, procrustes_data, procrustes_data_backbone