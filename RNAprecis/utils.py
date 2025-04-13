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
import write_files
from collections import Counter

from clean_mintage_code.data_functions import calculate_angle_3_points, rotation, rotation_matrix_x_axis, rotation_matrix_z_axis, mean_on_sphere_init
from clean_mintage_code.parse_functions import parse_pdb_files
from clean_mintage_code import shape_analysis
from clean_mintage_code.shape_analysis import procrustes_on_suite_class

# from plotting import plotting_perpendicular_distance_sorting_test_suites_to_pucker_pair_group, distance_angle_sphere_test_data, calculate_xy_data_test, plot_high_res, plot_confusion_matrix_clustering_comparison, low_res_training_plots, calculate_xy_data_training
# from plotting import *


# Util functions
def remove_duplicates_from_suite_list(test_suites):
    single_names = []
    test_suites_new = []
    for i in range(len(test_suites)):
        if test_suites[i].name not in single_names:
            test_suites_new.append(test_suites[i])
            single_names.append(test_suites[i].name)
    test_suites = test_suites_new
    return test_suites

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def min_distance_to_x_y(angle, matrix):
    rotated_vector = rotation_matrix_x_axis(angle)@matrix[1]
    return rotated_vector[2]**2

def min_y_distance(angle, matrix):
    return ((rotation_matrix_z_axis(angle)@matrix[3])[1]-(rotation_matrix_z_axis(angle)@matrix[1])[1])**2

def min_angle_distance(angle, matrix):
    return (calculate_angle_3_points([rotation_matrix_z_axis(angle)@matrix[3], np.array([0, 0, 0]), np.array([1, 0, 0])]) - calculate_angle_3_points([rotation_matrix_z_axis(angle)@matrix[1], np.array([-1, 0, 0]), np.array([1, 0, 0])]))**2


def cart_to_spherical(point):
    x = point[0]
    y = point[1]
    z = point[2]
    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x)
    theta = math.acos(z / r)
    return [r, theta, phi]

def cart_to_spherical_x_z_switch(point):
    z = point[0]
    y = point[1]
    x = point[2]
    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x)
    theta = math.acos(z / r)
    return [r, theta, phi]


def spherical_to_cart_x_z_switch(theta, phi):
    z = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    x = np.cos(theta)
    return np.array([x, y, z])


def shift_and_rotate_data(data, index_1, index_2, index_3):
    """
    In the first step, the data matrix is shifted so that the landmarks with index_1 are equal to the origin.
    In the second step, the data is rotated so that the landmarks with the index_2 are on the x-axis.
    In the third step, the data is rotated around the x-axis so that the landmarks with index_3 are in the x-y plane and
    have a positive y-entry
    :param data:
    :param index_1:
    :param index_2:
    :param index_3:
    :return: The transformed data matrix
    """

    # First step:
    new_data = [data[i] - data[i, index_1, :] for i in range(len(data))]

    # Second step
    rotation_matrices = [rotation(new_data[i][index_2] / np.linalg.norm(new_data[i][index_2]), np.array([1, 0, 0])) for
                         i in range(len(new_data))]
    new_data = [(rotation_matrices[i] @ new_data[i].T).T for i in range(len(new_data))]

    # Third step:

    # 3.1: Rotate the data so that the landmarks with index_3 are in the x-y plane:
    def distance_plane(angle, x):
        return np.linalg.norm((rotation_matrix_x_axis(angle) @ x)[2])
    angles = []
    for i in range(len(new_data)):
        print(i, len(new_data))
        index = np.argmin([minimize(fun=distance_plane,
                                    x0=[ang],
                                    method="Powell",
                                    args=(new_data[i][index_3])).fun for ang in np.linspace(0, 2 * np.pi, 3)])
        angles.append(minimize(fun=distance_plane,
                               x0=[np.linspace(0, 2 * np.pi, 3)[index]],
                               method="Powell",
                               args=(new_data[i][index_3])))
    angles = [angle.x[0] for angle in angles]
    new_data_ = [(rotation_matrix_x_axis(angles[i]) @ new_data[i].T).T for i in range(len(angles))]
    # 3.1: Rotate the data so that the landmarks with index_3 have a positive y-entry
    for i in range(len(new_data_)):
        if new_data_[i][index_3, 1] < 0:
            new_data_[i] = (rotation_matrix_x_axis(np.pi) @ new_data_[i].T).T
    return new_data_
    
def load_outlier_file_and_add_possible_answer_to_suite(complete_test_suites_2):
    """
    Load the outlier file (suite_outliers_with_answers.csv) and save the possible answers in the suite objects.
    :param complete_test_suites_2:
    :return:
    """
    csv_as_list = []
    with open('suite_outliers_with_answers.csv', newline='') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_file:
            csv_as_list.append(row)
    answers = [[csv_as_list[i][0].replace(' ', '')] for i in range(len(csv_as_list)) if
               len(csv_as_list[i]) == 3 and i > 0]
    answers = [answer if ':' not in answer[0] else answer[0].split(':') for answer in answers]
    outlier_residues = [[csv_as_list[i][1].replace(' ', '')] for i in range(len(csv_as_list)) if
                        len(csv_as_list[i]) == 3 and i > 0]
    outlier_residues = [outlier if ':' not in outlier[0] else outlier[0].split(':') for outlier in outlier_residues]
    for i in range(len(outlier_residues)):
        for j in range(len(outlier_residues[i])):
            pdb_name = outlier_residues[i][j][:4]
            chain_name = outlier_residues[i][j][5]
            first_residue_number = int(outlier_residues[i][j][7:-2])
            for suite in complete_test_suites_2:
                if (
                        pdb_name == suite._filename and chain_name == suite._name_chain and first_residue_number == suite._number_second_residue):
                    suite.answer = answers[i]



def help_function_to_create_table_with_all_pdb_files_for_the_latex_file(complete_test_suites_with_answer):
    """
    This function creates the table with the pdb names, the residue number of the test suites with the corresponding
    resolution from the appendix.
    :param complete_test_suites_with_answer: The list of suite objects.
    :return:
    """
    nr_tables = 4
    nr_elements = 6
    name_set_list = list(
        set([complete_test_suites_with_answer[i]._filename for i in range(len(complete_test_suites_with_answer))]))
    name_set_list.sort()
    url_list = ['https://www.rcsb.org/structure/' + name_set_list[i] + '/' for i in range(len(name_set_list))]
    res_list = []
    for i in range(len(url_list)):
        print(i, len(url_list))
        response = requests.get(url_list[i])
        index = str(response.content).find('Resolution')
        print(str(response.content)[index:index + 50])
        resolution = re.search(r"[-+]?\d*\.\d+|\d+", str(response.content)[index:]).group()
        print(name_set_list[i], resolution)
        res_list.append(resolution)
    nr_rows = int(np.round(len(set([complete_test_suites_with_answer[i]._filename for i in
                                    range(len(complete_test_suites_with_answer))])) / nr_tables))
    index_list = [i * nr_rows for i in range(nr_tables)] + [int(len(name_set_list))]
    for table_index in range(nr_tables):
        print('\\begin{minipage}[b]{0.47\\textwidth}')
        print('\\begin{tabular}{c|c|c}')
        print('PDB &   suite name & Res \\\\')
        print('\hline')
        for i in np.arange(index_list[table_index], index_list[table_index + 1]):
            pdb_name = name_set_list[i]
            index_list_ = [complete_test_suites_with_answer[j]._name_chain + str(
                complete_test_suites_with_answer[j]._number_second_residue) for j in
                           range(len(complete_test_suites_with_answer)) if
                           complete_test_suites_with_answer[j]._filename == pdb_name]
            string_name = ''
            if len(index_list_) < nr_elements:
                for j in range(len(index_list_)):
                    if j == 0:
                        string_name = string_name + index_list_[j]
                    else:
                        string_name = string_name + ', ' + index_list_[j]
                print(name_set_list[i] + ' & ' + string_name + ' &  ' + res_list[i] + ' \\\\')
            else:
                for j in range(len(index_list_)):
                    if j == 0:
                        string_name = ''
                        string_name = string_name + index_list_[j]
                    else:
                        if j % nr_elements == 0:
                            if j < nr_elements + 1:
                                print(name_set_list[i] + ' & ' + string_name + ' &  ' + res_list[i] + ' \\\\')
                            else:
                                print('"' + ' & ' + string_name + ' & "\\\\')
                            string_name = ''
                        if j % nr_elements == 0:
                            string_name = string_name + index_list_[j]
                        else:
                            string_name = string_name + ', ' + index_list_[j]
                print('"' + ' & ' + string_name + ' & " \\\\')

        print('\end{tabular}')
        print('\end{minipage}')
        print('\hspace*{0.02\\textwidth}')

def calculate_xy_data_test(angles_low_resolution, folder, procrustes_data):
    """
    This function calculates the 5 point representation (P is the origin, the two C1' atoms in the xy-plane
    and with the same positive y-value).

    :param angles_low_resolution: The angles between the consecutive landmarks in the data matrix procrustes_data.
    :param folder: The plots are saved here.
    :param procrustes_data: Data matrix calculated by Procrustes algorithm. It is important that the third landmark is
    the center point.
    :return: x_y_data_2:  The 5 point representation (P is the origin, the two C1' atoms in the xy-plane
      and with the same positive y-value).  It holds : x_y_data_2.shape = (nr_test_suites, 5, 3).
    """

    # Step 1: rotate the individual configuration matrices so that the fourth landmark is on the x-axis:
    rotation_matrices = [
        rotation(np.array(procrustes_data[i, 3, :]) / np.linalg.norm(procrustes_data[i, 3, :]), np.array([1, 0, 0])) for
        i in range(len(procrustes_data))]
    x_axis_data = np.array([(rotation_matrices[i] @ procrustes_data[i].T).T for i in range(len(procrustes_data))])

    # Step 2: rotate the individual configuration matrices around the x-axis so that the second landmark is on x-y-plane
    angle_list = [minimize(min_distance_to_x_y, 0, args=(x_axis_data[i]), method='BFGS').x[0] for i in
                  range(len(x_axis_data))]
    x_y_data = np.array([(rotation_matrix_x_axis(angle_list[i]) @ x_axis_data[i].T).T for i in range(len(angle_list))])
    # If the y-value of the second landmark is negative, then rotate again by the angle pi around the x-axis to obtain
    # a positive value
    for i in range(len(x_y_data)):
        if x_y_data[i][1, 1] < 0:
            x_y_data[i] = (rotation_matrix_x_axis(np.pi) @ x_y_data[i].T).T

    # Step 3: Rotate around the z-axis so that the y-values of landmark 2 and landmark 4 are the same:
    rotation_matrices_2 = [rotation(x_y_data[i][3] / np.linalg.norm(x_y_data[i][3]), np.array(
        [np.cos((90 - angles_low_resolution[1][i] / 2) * np.pi / 180),
         np.sin((90 - angles_low_resolution[1][i] / 2) * np.pi / 180), 0])) for i in range(len(x_y_data))]

    x_y_data_2 = np.array([(rotation_matrices_2[i] @ x_y_data[i].T).T for i in range(len(angle_list))])
    angle_list_2 = [minimize(min_y_distance, 0, args=(x_y_data_2[i]), method='BFGS').x[0] for i in
                    range(len(x_y_data_2))]
    x_y_data_2 = np.array(
        [(rotation_matrix_z_axis(angle_list_2[i]) @ x_y_data_2[i].T).T for i in range(len(angle_list))])

    # A test plot:
    plt.scatter(x_y_data_2[:, 1, 0], x_y_data_2[:, 1, 1], c='black')
    plt.scatter(x_y_data_2[:, 2, 0], x_y_data_2[:, 2, 1], c='green')
    plt.scatter(x_y_data_2[:, 3, 0], x_y_data_2[:, 3, 1], c='blue')
    plt.savefig(folder + 'scatter'  + '_new_par_phi_theta_test')
    plt.close()

    return x_y_data_2


def distance_angle_sphere_test_data_version_two(complete_test_suites_with_answer, folder, string_file_low_res_test,
                                    string_file_test_suites, string_file_xy_data):
    """
    This function returns a list containing for each parameter (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list
    containing the corresponding parameter for each element from test_suites and the 5 point representation
    (P is the origin, the two C1' atoms in the xy-plane and with the same positive y-value)
    :param complete_test_suites_with_answer: A list with suite objects.
    :param folder: The plots are saved here.
    :param string_file_low_res_test: String: The location where test_suites is stored or read from.
    :param string_file_test_suites: String: The location where low_res_test is stored or read from.
    :param string_file_xy_data: String: The location where x_y_data_2_test is stored or read from.
    :return:
      -- low_res_test: a list of lists: A list containing for each parameter
      (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list containing the corresponding parameter for each
      element from test_suites.
      -- x_y_data_2:  The 5 point representation (P is the origin, the two C1' atoms in the xy-plane
      and with the same positive y-value).
    """
    # Use the Procrustes algorithm, with the following modification. The third (python index=2) landmark (the P atom)
    # is made the origin.
    string = './out/procrustes/five_chain_complete_size_shape_test' + '.pickle'
    string_plot = './out/procrustes/five_chain_test'
    complete_five_chains = np.array([suite._five_chain for suite in complete_test_suites_with_answer])
    procrustes_data = procrustes_on_suite_class(complete_five_chains, string, string_plot,
                                                origin_index=2)[0]

    # Compute the distances between consecutive landmarks: d_1, d_2, d_3 and d_4:
    distances_low_resolution = [
        [np.linalg.norm(procrustes_data[i][j] - procrustes_data[i][j + 1]) for i in range(len(procrustes_data))] for j
        in range(4)]
    # d_1 and d_4 are almost constant: take only d_2 and d_3
    distances_center = [distances_low_resolution[1], distances_low_resolution[2]]

    # Calculate the angles for every 3 consecutive atoms:
    angles_low_resolution = [
        [calculate_angle_3_points([procrustes_data[i][j], procrustes_data[i][j + 1], procrustes_data[i][j + 2]]) for i
         in range(len(procrustes_data))] for j in range(3)]

    # The function calculates the 5 point representation (P is the origin, the two C1' atoms in the xy-plane and with
    # the same positive y-value).
    x_y_data_2 = calculate_xy_data_test(angles_low_resolution, folder, procrustes_data)
    # Determine phi and theta based on the transformed data matrix:
    phi_1_theta_1 = list(
        np.array([cart_to_spherical_x_z_switch(x_y_data_2[i][0] -x_y_data_2[i][1])[1:] for i in range(len(x_y_data_2))]).T)
    phi_2_theta_2 = list(
        np.array([cart_to_spherical_x_z_switch(x_y_data_2[i][4] - x_y_data_2[i][3])[1:] for i in range(len(x_y_data_2))]).T)
    low_res_test = distances_center + [angles_low_resolution[1]] + phi_1_theta_1 + phi_2_theta_2
    with open(string_file_test_suites, 'wb') as f:
        pickle.dump(complete_test_suites_with_answer, f)
    with open(string_file_low_res_test, 'wb') as f:
        pickle.dump(low_res_test, f)
    with open(string_file_xy_data, 'wb') as f:
        pickle.dump(x_y_data_2, f)
    return low_res_test, x_y_data_2

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

def get_procrustes_data_name(name_, input_suites):
    """
    This function gets all training suites and returns the subset of suites that belong to 'name_' by calling the
    determine_pucker_data function. In a second step, the Procrustes algorithm is applied to the sub-data set.
    :param name_: 'c2c2', 'c3c3', 'c2c3' or 'c3c2'
    :param input_suites: The complete list of training suites objects.
    :return: cluster_suites: The sub list of suite objects belonging to name_.
    procrustes_data: numpy array from the low detail representation.
    procrustes_data_backbone: numpy array from the high detail representation.
    """
    cluster_suites = [suite for suite in input_suites if suite.procrustes_five_chain_vector is not None
                      and suite.dihedral_angles is not None]
    # type atm to get only pdb ATM not heteroatoms
    cluster_suites = [suite for suite in cluster_suites if suite.atom_types == 'atm']

    print(f'semi-complete suites: {len(cluster_suites)}')
    _, cluster_suites = determine_pucker_data(cluster_suites, name_)
    print(f'{name_} suites: {len(cluster_suites)}')
    # rotate data again for each pucker
    procrustes_data = np.array([suite.procrustes_five_chain_vector for suite in cluster_suites])
    procrustes_data_backbone = np.array([suite.procrustes_complete_suite_vector for suite in cluster_suites])
    return cluster_suites, procrustes_data, procrustes_data_backbone

def read_in_cluster_comparison_files(cluster_list_high_res_franziska, cluster_suites, name_):
    """
    This function reads in two different files (csv/xlsx format) for each name_, which together have a suite conformer
    for each suite. The reason for this is that not all suites were labeled in the first iteration and therefore some
    that were missing in the first iteration were stored in the subfolder './cluster_comparison/these_were_missing_classes/'.
    :param cluster_list_high_res_franziska: A list of lists with indices from the MINT-AGE clustering.
    :param cluster_suites: The list of suite objects.
    :param name_: A string from [c3c3, c2c3, c3c2, c2c2].
    :return:
    """

    # Importing the first cluster comparison file:
    cluster_comparison = pd.read_excel('./cluster_comparison/' + name_ + '.xlsx')
    cluster_names_richardson = [[] for k in range(len(cluster_list_high_res_franziska))]
    in_richardson_list = []
    richardson_cluster_list = {}
    for i in range(len(cluster_comparison)):
        print(i, len(cluster_comparison))
        try:
            pdb_name = cluster_comparison['pdbid'][i]
            chain = cluster_comparison['chain'][i]
            res_number = int(cluster_comparison['resseq'][i])
            # cluster_number = int(cluster_comparison['cluster'][i])
            name_richardson_cluster = cluster_comparison['suitename'][i]
            for j in range(len(cluster_suites)):
                if cluster_suites[j]._filename == pdb_name and cluster_suites[j]._name_chain == chain and \
                        cluster_suites[j]._number_second_residue == res_number:
                    if j not in in_richardson_list:
                        in_richardson_list.append(j)
                        if name_richardson_cluster in richardson_cluster_list.keys():
                            richardson_cluster_list[name_richardson_cluster].append(j)
                        else:
                            richardson_cluster_list[name_richardson_cluster] = [j]
                        j_index_in_cluster_list = [j in cluster_list_high_res_franziska[k] for k in
                                                   range(len(cluster_list_high_res_franziska))]
                        if True in j_index_in_cluster_list:
                            index_j_cluster = np.argmax(j_index_in_cluster_list)
                            cluster_names_richardson[index_j_cluster].append(name_richardson_cluster)
        except:
            pass

    # Importing the second cluster comparison file:
    cluster_comparison_2_as_list = []
    with open('./cluster_comparison/these_were_missing_classes/' + name_ + '.csv', newline='') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in csv_file:
            cluster_comparison_2_as_list.append(row)
    for i in range(len(cluster_comparison_2_as_list)):
        print(i, len(cluster_comparison_2_as_list))
        try:
            pdb_name = cluster_comparison_2_as_list[i][0]
            chain = cluster_comparison_2_as_list[i][1]
            res_number = int(cluster_comparison_2_as_list[i][2])
            # cluster_number = int(cluster_comparison['cluster'][i])
            name_richardson_cluster = cluster_comparison_2_as_list[i][4]
            for j in range(len(cluster_suites)):
                if cluster_suites[j]._filename == pdb_name and cluster_suites[j]._name_chain == chain and \
                        cluster_suites[j]._number_second_residue == res_number:
                    if j not in in_richardson_list:
                        in_richardson_list.append(j)
                        if name_richardson_cluster in richardson_cluster_list.keys():
                            richardson_cluster_list[name_richardson_cluster].append(j)
                        else:
                            richardson_cluster_list[name_richardson_cluster] = [j]
                        j_index_in_cluster_list = [j in cluster_list_high_res_franziska[k] for k in
                                                   range(len(cluster_list_high_res_franziska))]
                        if True in j_index_in_cluster_list:
                            index_j_cluster = np.argmax(j_index_in_cluster_list)
                            cluster_names_richardson[index_j_cluster].append(name_richardson_cluster)

        except:
            pass


    # Saving the cluster results as csv:
    names_ = []
    cluster_number_list = []
    suitename = []
    for key_ in list(richardson_cluster_list.keys()):
        for index_ in richardson_cluster_list[key_]:
            index_in_cluster_list = [index_ in cluster_list_high_res_franziska[k] for k in
                                     range(len(cluster_list_high_res_franziska))]
            if True in j_index_in_cluster_list:
                cluster_number = np.argmax(index_in_cluster_list)+1
            else:
                cluster_number = 'outlier'
            names_.append(cluster_suites[index_]._name)
            cluster_number_list.append(cluster_number)
            suitename.append(key_)
    create_excel_from_lists([names_, cluster_number_list, suitename],
                            ['pdb_file_chain_name_and_residue_numbers', 'MINT-AGE', 'suitename'],
                            './cluster_comparison/' + name_ + 'cluster_comp.xlsx')

    return cluster_names_richardson, richardson_cluster_list

def create_excel_from_lists(lists, headers, output_file):
    # Check if all lists have the same length
    list_lengths = [len(lst) for lst in lists]
    if len(set(list_lengths)) != 1:
        raise ValueError("Lists must have the same length.")

    # Create a DataFrame from the lists and headers
    data = {header: lst for header, lst in zip(headers, lists)}
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file with headers
    df.to_excel(output_file, index=False)
    print(f"Excel file created successfully: {output_file}")



def create_table_clustering_results(cluster_list_high_res_franziska, cluster_names_richardson_numbers, cluster_suites):
    """
    A function that automatically creates a table that is in latex format.
    :param cluster_list_high_res_franziska:
    :param cluster_names_richardson_numbers:
    :param cluster_suites:
    :return:
    """

    all_cluster_elements = [element for sublist in cluster_list_high_res_franziska for element in sublist]
    outlier = [i for i in range(len(cluster_suites)) if i not in all_cluster_elements]
    print('\\begin{minipage}[b]{0.47\\textwidth}')
    print('\\begin{tabular}{c|c|c}')
    print('MINT-AGE &  nr elements  &  Richardson \\\\')
    print('\hline')
    for i in range(len(cluster_list_high_res_franziska)):
        list_names_and_numbers = [
            cluster_names_richardson_numbers[i][j][0] + ' (' + str(cluster_names_richardson_numbers[i][j][1]) + ')' for
            j in range(len(cluster_names_richardson_numbers[i]))]
        print(str(i + 1) + ' & ' + str(len(cluster_list_high_res_franziska[i])) + ' & ' + str(list_names_and_numbers)[
                                                                                          1:-1].replace("'",
                                                                                                        "") + '\\\\')
    print('outlier & ' + str(len(outlier)) + ' & ' + '\\\\')
    print('\end{tabular}')
    print('\end{minipage}')
    print('\hspace*{0.02\\textwidth}')

def distance_angle_sphere_training_data_version_two(cluster_suites, folder_plots, name_):
    """
    This function returns a list containing for each parameter (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list
    containing the corresponding parameter for each element from test_suites and the 5 point representation
    (P is the origin, the two C1' atoms in the xy-plane and with the same positive y-value)
    :param cluster_suites: A list of suite objects
    :param folder_plots: the plots are saved here
    :param name_: A string from [c3c3, c2c3, c3c2, c2c2].
    :return:
      -- low_res: a list of lists: A list containing for each parameter
      (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list containing the corresponding parameter for each
      element from test_suites.
      -- x_y_data_2:  The 5 point representation (P is the origin, the two C1' atoms in the xy-plane
      and with the same positive y-value).
    """
    # Use the Procrustes algorithm, with the following modification. The third (python index=2) landmark (the P atom)
    # is made the origin.
    string = './out/procrustes/five_chain_complete_size_shape_training' + name_ + '.pickle'
    string_plot = './out/procrustes/five_chain_training' + name_
    complete_five_chains = np.array([suite._five_chain for suite in cluster_suites])
    procrustes_data = procrustes_on_suite_class(complete_five_chains, string, string_plot,
                                                origin_index=2)[0]

    # Compute the distances between consecutive landmarks: d_1, d_2, d_3 and d_4:
    distances_low_resolution = [
        [np.linalg.norm(procrustes_data[i][j] - procrustes_data[i][j + 1]) for i in range(len(procrustes_data))] for j
        in range(4)]
    # d_1 and d_4 are almost constant: take only d_2 and d_3
    distances_center = [distances_low_resolution[1], distances_low_resolution[2]]

    # Calculate the angles for every 3 consecutive atoms:
    angles_low_resolution = [
        [calculate_angle_3_points([procrustes_data[i][j], procrustes_data[i][j + 1], procrustes_data[i][j + 2]]) for i
         in range(len(procrustes_data))] for j in range(3)]
    # The function calculates the 5 point representation (P is the origin, the two C1' atoms in the xy-plane and with
    # the same positive y-value).
    angle_list, x_y_data_2 = calculate_xy_data_training(angles_low_resolution, folder_plots, name_, procrustes_data)
    phi_1_theta_1 = list(
        np.array([cart_to_spherical_x_z_switch(x_y_data_2[i][0] -x_y_data_2[i][1])[1:] for i in range(len(angle_list))]).T)
    phi_2_theta_2 = list(
        np.array([cart_to_spherical_x_z_switch(x_y_data_2[i][4] - x_y_data_2[i][3])[1:] for i in range(len(angle_list))]).T)
    low_res = distances_center + [angles_low_resolution[1]] + phi_1_theta_1 + phi_2_theta_2
    return low_res, x_y_data_2

def calculate_xy_data_training(angles_low_resolution, folder_plots, name_, procrustes_data):
    """
    This function calculates the 5 point representation (P is the origin, the two C1' atoms in the xy-plane
    and with the same positive y-value).

    :param angles_low_resolution: The angles between the consecutive landmarks in the data matrix procrustes_data.
    :param folder_plots: The plots are saved here.
    :param procrustes_data: Data matrix calculated by Procrustes algorithm. It is important that the third landmark is
    the center point.
    :return: x_y_data_2:  The 5 point representation (P is the origin, the two C1' atoms in the xy-plane
      and with the same positive y-value).  It holds : x_y_data_2.shape = (nr_training_suites, 5, 3).
    """

    # Step 1: rotate the individual configuration matrices so that the fourth landmark is on the x-axis:
    rotation_matrices = [
        rotation(np.array(procrustes_data[i, 3, :]) / np.linalg.norm(procrustes_data[i, 3, :]), np.array([1, 0, 0])) for
        i in range(len(procrustes_data))]
    x_axis_data = np.array([(rotation_matrices[i] @ procrustes_data[i].T).T for i in range(len(procrustes_data))])

    # Step 2: rotate the individual configuration matrices around the x-axis so that the second landmark is on x-y-plane
    angle_list = [minimize(min_distance_to_x_y, 0, args=(x_axis_data[i]), method='BFGS').x[0] for i in
                  range(len(x_axis_data))]
    x_y_data = np.array([(rotation_matrix_x_axis(angle_list[i]) @ x_axis_data[i].T).T for i in range(len(angle_list))])
    # If the y-value of the second landmark is negative, then rotate again by the angle pi around the x-axis to obtain
    # a positive value
    for i in range(len(x_y_data)):
        if x_y_data[i][1, 1] < 0:
            x_y_data[i] = (rotation_matrix_x_axis(np.pi) @ x_y_data[i].T).T
    # Step 3: Rotate around the z-axis so that the y-values of landmark 2 and landmark 4 are the same:
    rotation_matrices_2 = [rotation(x_y_data[i][3] / np.linalg.norm(x_y_data[i][3]), np.array(
        [np.cos((90 - angles_low_resolution[1][i] / 2) * np.pi / 180),
         np.sin((90 - angles_low_resolution[1][i] / 2) * np.pi / 180), 0])) for i in range(len(x_y_data))]
    x_y_data_2 = np.array([(rotation_matrices_2[i] @ x_y_data[i].T).T for i in range(len(angle_list))])
    angle_list_2 = [minimize(min_y_distance, 0, args=(x_y_data_2[i]), method='BFGS').x[0] for i in
                    range(len(x_y_data_2))]
    x_y_data_2 = np.array(
        [(rotation_matrix_z_axis(angle_list_2[i]) @ x_y_data_2[i].T).T for i in range(len(angle_list))])

    # A test plot:
    plt.scatter(x_y_data_2[:, 1, 0], x_y_data_2[:, 1, 1], c='black')
    plt.scatter(x_y_data_2[:, 2, 0], x_y_data_2[:, 2, 1], c='green')
    plt.scatter(x_y_data_2[:, 3, 0], x_y_data_2[:, 3, 1], c='blue')
    plt.savefig(folder_plots + 'scatter' + name_ + '_new_par_phi_theta_test')
    plt.close()
    return angle_list, x_y_data_2

def create_result_table_for_all_test_elements(index_list, result_list):
    """
    This function creates a result table (in latex format) for all test items listed in the appendix of the paper. For
    all three categories (which are plotted in blue, red and orange) individual tables (which are individually created
    slightly differently).
    :param index_list:
    :param result_list: A list of dictionaries. Each dictionary belongs to a test element
    :return:
    """
    dist_list = [np.min(result_list[i]['distance_next_frechet_cluster']) for i in index_list]
    argsort = np.argsort(dist_list)
    index_list_new = np.array(index_list)[argsort]
    print('\\begin{minipage}[b]{0.47\\textwidth}')
    print('\\begin{tabular}{c|c|c|c}')
    # print('PDB &  chain  &  res & answer & $p_1$ & $p_2$ & Cluster  \\\\')
    print('name & dkbc & RNAprecis & d \\\\')
    print('\hline')
    counter = 0
    for i in index_list_new:
        true_false_ = [len(set(result_list[i]['next_frechet_mean'][k]) & set(result_list[i]['answer'])) > 0 for k in
                       range(len(result_list[i]['next_frechet_mean']))]
        answer_rna_p = str(result_list[i]['next_frechet_mean'][0])[1:-1]
        answer_rna_p = answer_rna_p.replace("'", "")
        answer_rna_p = answer_rna_p.replace('&', '\&')
        answer_rna_p = answer_rna_p.replace('#', '\#')
        cluster = [
            '\textbf{' + str(result_list[i]['next_frechet_mean_nr'][k] + 1) + '} (' + answer_rna_p + ')' if true_false_[
                k] else
            str(result_list[i]['next_frechet_mean_nr'][k] + 1) + ' (' + answer_rna_p + ')' for k in
            range(len(result_list[i]['next_frechet_mean_nr']))]
        cluster = str(cluster)[1:-1].replace("'", "")
        cluster = cluster.replace("\\\\", "\\")
        answer = str(result_list[i]['answer'])[1:-1]
        answer = answer.replace("'", "")
        answer = answer.replace('&', '\&')
        answer = answer.replace('#', '\#')
        print(result_list[i]['suite']._filename + '\_' +
              result_list[i]['suite']._name_chain + '\_' +
              str(result_list[i]['suite']._number_second_residue) + ' & ' +
              answer + ' & ' +
              # str(np.round(result_list[i]['suite'].pucker_distance_1, 2)) + ' & ' +
              # str(np.round(result_list[i]['suite'].pucker_distance_2, 2)) + ' & ' +
              cluster + ' & ' +
              str(np.round(result_list[i]['distance_next_frechet_cluster'][0], 2)) + '\\\\')
        if counter % 80 == 0 and counter > 0:
            print('\end{tabular}')
            print('\end{minipage}')
            print('\hspace*{0.02\\textwidth}')
            print('\\begin{minipage}[b]{0.47\\textwidth}')
            print('\\begin{tabular}{c|c|c|c}')
            # print('PDB &  chain  &  res & answer & $p_1$ & $p_2$ & Cluster  \\\\')
            print('name & dkbc & RNAprecis & d \\\\')
            print('\hline')
        counter = counter + 1
    print('\end{tabular}')
    print('\end{minipage}')
    print('\hspace*{0.02\\textwidth}')
    # red elements:
    list_ = [j for j in range(len(result_list)) if not result_list[j]['possible']]
    dist_list = [np.min(result_list[i]['distance_next_frechet_cluster']) for i in list_]
    argsort = np.argsort(dist_list)
    index_list_new = np.array(list_)[argsort]
    print('######################################################################################')
    print('######################################################################################')
    print('\\begin{minipage}[b]{0.47\\textwidth}')
    print('\\begin{tabular}{c|c|c|c|c}')
    print('name & dkbc & $p_1$ & $p_2$ & RNAprecis & d \\\\')
    # print('name & answer & Cluster  \\\\')
    print('\hline')
    counter = 0
    for i in index_list_new:
        true_false_ = [len(set(result_list[i]['next_frechet_mean'][k]) & set(result_list[i]['answer'])) > 0 for k in
                       range(len(result_list[i]['next_frechet_mean']))]
        answer_rna_p = str(result_list[i]['next_frechet_mean'][0])[1:-1]
        answer_rna_p = answer_rna_p.replace("'", "")
        answer_rna_p = answer_rna_p.replace('&', '\&')
        answer_rna_p = answer_rna_p.replace('#', '\#')
        cluster = [
            '\textbf{' + str(result_list[i]['next_frechet_mean_nr'][k] + 1) + '} (' + answer_rna_p + ')' if true_false_[
                k] else
            str(result_list[i]['next_frechet_mean_nr'][k] + 1) + ' (' + answer_rna_p + ')' for k in
            range(len(result_list[i]['next_frechet_mean_nr']))]
        cluster = str(cluster)[1:-1].replace("'", "")
        cluster = cluster.replace("\\\\", "\\")
        answer = str(result_list[i]['answer'])[1:-1]
        answer = answer.replace("'", "")
        answer = answer.replace('&', '\&')
        print(result_list[i]['suite']._filename + '\_' +
              result_list[i]['suite']._name_chain + '\_' +
              str(result_list[i]['suite']._number_second_residue) + ' & ' +
              answer + ' & ' +
              str(np.round(result_list[i]['suite'].pucker_distance_1, 2)) + ' & ' +
              str(np.round(result_list[i]['suite'].pucker_distance_2, 2)) + ' & ' +
              cluster + ' & ' +
              str(np.round(result_list[i]['distance_next_frechet_cluster'][0], 2)) + '\\\\')
        if counter % 80 == 0 and counter > 0:
            print('\end{tabular}')
            print('\end{minipage}')
            print('\hspace*{0.02\\textwidth}')
            print('\\begin{minipage}[b]{0.47\\textwidth}')
            print('\\begin{tabular}{c|c|c|c|c}')
            print('name & dkbc & $p_1$ & $p_2$ & RNAprecis  \\\\')
            print('\hline')
        counter = counter + 1
    print('\end{tabular}')
    print('\end{minipage}')
    print('\hspace*{0.02\\textwidth}')
    # orange elements:
    list_ = [j for j in range(len(result_list)) if j not in index_list and result_list[j]['possible']]
    dist_list = [np.min(result_list[i]['distance_next_frechet_cluster']) for i in list_]
    argsort = np.argsort(dist_list)
    index_list_new = np.array(list_)[argsort]
    print('######################################################################################')
    print('######################################################################################')
    print('\\begin{minipage}[b]{0.47\\textwidth}')
    print('\\begin{tabular}{c|c|c|c|c|c}')
    print('name & dkbc & RNAprecis & $d_1$ & first match & $d_2$ \\\\')
    # print('name & answer & Cluster  \\\\')
    print('\hline')
    counter = 0
    for i in index_list_new:
        true_false_ = [len(set(result_list[i]['next_frechet_mean_all'][k]) & set(result_list[i]['answer'])) > 0 for k in
                       range(len(result_list[i]['next_frechet_mean_all']))]

        answer_rna_p = str(result_list[i]['next_frechet_mean'][0])[1:-1]
        answer_rna_p = answer_rna_p.replace("'", "")
        answer_rna_p = answer_rna_p.replace('&', '\&')
        answer_rna_p = answer_rna_p.replace('#', '\#')
        cluster = [
            '\textbf{' + str(result_list[i]['next_frechet_mean_nr'][k] + 1) + '} (' + answer_rna_p + ')' if true_false_[
                k] else
            str(result_list[i]['next_frechet_mean_nr'][k] + 1) + ' (' + answer_rna_p + ')' for k in
            range(len(result_list[i]['next_frechet_mean_nr']))]
        cluster = str(cluster)[1:-1].replace("'", "")
        cluster = cluster.replace("\\\\", "\\")
        if True not in true_false_:
            first_match = '-'
            first_match_distance = '-'
        else:
            k = min([i for i in range(len(true_false_)) if true_false_[i]])
            answer_rna_p_ = str(result_list[i]['next_frechet_mean_all'][k])[1:-1]
            answer_rna_p_ = answer_rna_p_.replace("'", "")
            answer_rna_p_ = answer_rna_p_.replace('&', '\&')
            answer_rna_p_ = answer_rna_p_.replace('#', '\#')
            # k = min([i for i in range(len(true_false_)) if true_false_[i]])
            first_match = str(result_list[i]['next_frechet_mean_nr_all'][k] + 1) + ' (' + answer_rna_p_ + ')'
            first_match = first_match.replace("\\\\", "\\")
            first_match_distance = str(np.round(result_list[i]['distance_next_frechet_cluster_all'][k], 2))
        # cluster = ['\textbf{' + str(result_list[i]['next_frechet_mean_nr_all'][k]+1) + '} (' + str(np.round(result_list[i]['distance_next_frechet_cluster_all'][k], 2)) + ')' if true_false_[k] else
        #           str(result_list[i]['next_frechet_mean_nr_all'][k]+1) + ' (' + str(np.round(result_list[i]['distance_next_frechet_cluster_all'][k], 2)) + ')' for k in range(len(result_list[i]['next_frechet_mean_nr_all']))]

        answer = str(result_list[i]['answer'])[1:-1]
        answer = answer.replace("'", "")
        answer = answer.replace('&', '\&')
        print(result_list[i]['suite']._filename + '\_' +
              result_list[i]['suite']._name_chain + '\_' +
              str(result_list[i]['suite']._number_second_residue) + ' & ' +
              answer + ' & ' +
              # str(np.round(result_list[i]['suite'].pucker_distance_1, 2)) + ' & ' +
              # str(np.round(result_list[i]['suite'].pucker_distance_2, 2)) + ' & ' +
              cluster + ' & ' +
              str(np.round(result_list[i]['distance_next_frechet_cluster'][0], 2)) + ' & ' +
              first_match + ' & ' +
              first_match_distance + '\\\\')
        if counter % 80 == 0 and counter > 0:
            print('\end{tabular}')
            print('\end{minipage}')
            print('\hspace*{0.02\\textwidth}')
            print('\\begin{minipage}[b]{0.47\\textwidth}')
            print('\\begin{tabular}{c|c|c|c|c|c}')
            print('name & dkbc & RNAprecis & $d_1$ & first match & $d_2$ \\\\')
            print('\hline')
        counter = counter + 1
    print('\end{tabular}')
    print('\end{minipage}')
    print('\hspace*{0.02\\textwidth}')

def create_xticks_and_yticks_for_confusion_matrix(cluster_names_richardson_numbers, names_classes, names_richardson):
    names_learning_ = []
    for i in range(len(names_classes)):
        str_ = ''
        for j in range(len(names_classes[i])):
            richardson_i_j_list_names = [cluster_names_richardson_numbers[names_classes[i][j]][k][0] for k in
                                         range(len(cluster_names_richardson_numbers[names_classes[i][j]]))]
            if '!!' in richardson_i_j_list_names:
                richardson_i_j_list_names.remove('!!')
            richardson_i_j_str = str(richardson_i_j_list_names)[1:-1].replace("'", "")
            if j == 0:
                str_ = str_ + 'Cl ' + str(names_classes[i][j] + 1) + '(' + richardson_i_j_str + ')'
            elif j == 3:
                str_ = str_ + ',\n Cl ' + str(names_classes[i][j] + 1) + '(' + richardson_i_j_str + ')'
            else:
                str_ = str_ + ', Cl ' + str(names_classes[i][j] + 1) + '(' + richardson_i_j_str + ')'
        names_learning_.append(str_)
    names_learning = names_learning_
    names_richardson = [names_richardson[i][1:-1] for i in range(len(names_richardson))]
    names_richardson = [names_richardson[i].replace("'", "") for i in range(len(names_richardson))]
    names_richardson = [names_richardson[i].replace(" ", "") for i in range(len(names_richardson))]
    return names_learning, names_richardson

def create_confusion_matrix_learning(all_classes, frechet, index_list, result_list):
    names_richardson = list(set([str(result_list[j]['answer']) for j in range(len(result_list))]))
    number_el_richardson = np.array(
        [-sum([str(result_list[j]['answer']) == names_richardson[i] for j in range(len(result_list))]) for i in
         range(len(names_richardson))])
    names_richardson = list(np.array(names_richardson)[np.argsort(number_el_richardson)])
    names_learning = [list(set(flatten_list(result_list[j]['next_frechet_mean']))) for j in range(len(result_list))]
    classes_learning = [list(set(result_list[j]['next_frechet_mean_nr'])) for j in range(len(result_list))]
    for i in range(len(names_learning)):
        names_learning[i].sort()
        classes_learning[i].sort()
        if '!!' in names_learning[i]:
            names_learning[i].remove('!!')
        names_learning[i] = str(names_learning[i])
    seen = set()
    seen_classes = set()
    unique_names = [names_learning[i] for i in range(len(names_learning)) if
                    not (names_learning[i] in seen or seen.add(names_learning[i]))]
    unique_classes = [classes_learning[i] for i in range(len(names_learning)) if
                      not (names_learning[i] in seen_classes or seen_classes.add(names_learning[i]))]
    names_learning = unique_names
    number_el_learning = []
    for i in range(len(names_learning)):
        run_list = []
        for j in range(len(result_list)):
            dummy_name = list(set(flatten_list(result_list[j]['next_frechet_mean'])))
            dummy_name.sort()
            if '!!' in dummy_name:
                dummy_name.remove('!!')
            dummy_name = str(dummy_name)
            if dummy_name == names_learning[i]:
                run_list.append(j)
        number_el_learning.append(-len(run_list))
    names_learning = list(np.array(names_learning)[np.argsort(number_el_learning)])
    names_classes = [unique_classes[nr] for nr in np.argsort(number_el_learning)]
    list_of_lists_of_indices_richardson = [[] for i in range(len(names_richardson))]
    list_of_lists_of_indices_learning = [[] for i in range(len(names_learning))]
    for i in range(len(result_list)):
        for j in range(len(names_richardson)):
            if str(result_list[i]['answer']) == names_richardson[j]:
                list_of_lists_of_indices_richardson[j].append(i)
        for j in range(len(names_learning)):
            if not frechet:
                if not all_classes:
                    if str(result_list[i]['cluster_name']) == names_learning[j]:
                        list_of_lists_of_indices_learning[j].append(i)
                else:
                    if str(result_list[i]['cluster_names']) == names_learning[j]:
                        list_of_lists_of_indices_learning[j].append(i)
            else:
                dummy_name = list(set(flatten_list(result_list[i]['next_frechet_mean'])))
                dummy_name.sort()
                if '!!' in dummy_name:
                    dummy_name.remove('!!')
                dummy_name = str(dummy_name)
                if str(dummy_name) == names_learning[j]:
                    list_of_lists_of_indices_learning[j].append(i)
    confusion_matrix = np.zeros((len(list_of_lists_of_indices_richardson), len(list_of_lists_of_indices_learning)))
    color_matrix = [[] for i in range(confusion_matrix.shape[0])]
    for i in range(confusion_matrix.shape[0]):
        for j in range(0, confusion_matrix.shape[1]):
            cluster_i = list_of_lists_of_indices_richardson[i]
            cluster_j = list_of_lists_of_indices_learning[j]
            confusion_matrix[i, j] = len([a for a in cluster_i if a in cluster_j])
            if confusion_matrix[i, j] == 0:
                color_matrix[i].append('white')
            else:
                # take the first element:
                first_element = [a for a in cluster_i if a in cluster_j][0]
                if True in [richardson_cluster_names_sorted[counter] in names_richardson[i] for counter in
                            range(len(richardson_cluster_names_sorted))]:
                    if first_element in index_list:
                        color_matrix[i].append('C0')
                    else:
                        color_matrix[i].append('C1')
                else:
                    color_matrix[i].append('red')
                # print('test')
            # confusion_matrix[j, i] = len([b for b in cluster_j if b in cluster_i])
    return color_matrix, confusion_matrix, names_classes, names_richardson
