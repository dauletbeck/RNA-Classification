from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pickle
import os
import math
import pandas as pd
import seaborn as sn
from clean_mintage_code import shape_analysis
from clean_mintage_code.parse_functions import parse_pdb_files
from code_king import low_detail_array
import write_files

from clean_mintage_code.plot_functions import scatter_plots
from clean_mintage_code.data_functions import calculate_angle_3_points, rotation, rotation_matrix_x_axis, rotation_matrix_z_axis, mean_on_sphere_init
from clean_mintage_code.shape_analysis import procrustes_on_suite_class
from clean_mintage_code.plot_functions import build_fancy_chain_plot
from clean_mintage_code.constants import COLORS_SCATTER, COLORS, MARKERS

# from utils import *
from utils import create_table_clustering_results, distance_angle_sphere_test_data_version_two, distance_angle_sphere_training_data_version_two, flatten_list, get_procrustes_data_name, help_function_to_create_table_with_all_pdb_files_for_the_latex_file, load_outlier_file_and_add_possible_answer_to_suite, min_distance_to_x_y, min_y_distance, cart_to_spherical, read_in_cluster_comparison_files, remove_duplicates_from_suite_list, shift_and_rotate_data, min_angle_distance, create_result_table_for_all_test_elements


# Plotting functions

def plot_single_test_element(cluster_list_high_res_franziska, folder,
                             frechet_answers_ordered, i, index_names, test_suite, x_y_data_2_test, xy_data_2, cluster_suites):
    if not os.path.exists(folder):
        os.makedirs(folder)

    COLORS = ['black', 'blue', 'grey', 'orange', 'pink', 'yellow']
    list_of_lists = [cluster_list_high_res_franziska[k] for k in index_names]
    nr_elements = [len(list_of_lists[k]) for k in range(len(list_of_lists))]  # + [1] + [1]
    colors = flatten_list([[COLORS[i]] * nr_elements[i] for i in range(len(nr_elements))]) + ['red']
    np_array = np.vstack([xy_data_2[flatten_list(list_of_lists)],
                          x_y_data_2_test[i].reshape(1, 5, 3)])
    # if all_plots:
    build_fancy_chain_plot(np_array,
                            folder + test_suite._name + 'to_' + str(frechet_answers_ordered).replace(" ", ""),
                            colors=colors, lw_vec=[0.5] * (len(colors) - 1) + [4], without_legend=True)
    input_suites = [cluster_suites[index_] for index_ in flatten_list(list_of_lists)] + [test_suite]
    low_detail_array(low_detail_data=np_array, input_suites=input_suites,
                     filename=folder + test_suite._name + 'to_' + str(frechet_answers_ordered).replace(" ", ""),
                     plot_low_res=True, color_list=colors, group_list=None,
                     lw_list=[0.5] * (len(colors) - 1) + [3])
    print('test')


def plotting_perpendicular_distance_sorting_test_suites_to_pucker_pair_group(complete_test_suites_with_answer, folder):
    """
    In part 1, the data for the first sugar ring and in the part 2 for the second sugar ring are initially presented as
    follows:
    - N1/N9 origin
    - C1' positive x-axis
    - P x-y plane (positive y value)
    The data is then plotted in the representation and the information to which Pucker pair group the element belongs is
    added to the suite object.
    :param complete_test_suites_with_answer: A list of suite objects.
    :param folder: The plots are saved here
    :return:
    """

    # Part 1: For the first sugar ring:

    # data: N1/N9 - C1'- P - C1' - N1/N9 - P (from next ring). Note that we also need the P from the next ring to
    # determine the perpendicular distance from the second sugar ring
    six_chain_data = np.array([suite._six_chain for suite in complete_test_suites_with_answer])
    # Index N1/N9 (first ring):
    index_1 = 0
    # Index C1' (first ring):
    index_2 = 1
    # Index P (first ring):
    index_3 = 2
    new_data_first = shift_and_rotate_data(six_chain_data, index_1=index_1, index_2=index_2, index_3=index_3)
    index_list_c_3 = [i for i in range(len(new_data_first)) if np.array(new_data_first)[i, index_3, 1] > 2.9]
    index_list_c_2 = [i for i in range(len(new_data_first)) if np.array(new_data_first)[i, index_3, 1] <= 2.9]
    fontsize = 17
    plt.scatter(np.array(new_data_first)[:, index_1, 0], np.array(new_data_first)[:, index_1, 1], c='black',
                label='N1/N9')
    plt.scatter(np.array(new_data_first)[:, index_2, 0], np.array(new_data_first)[:, index_2, 1], c='blue', label="C1'")
    plt.scatter(np.array(new_data_first)[index_list_c_2, index_3, 0],
                np.array(new_data_first)[index_list_c_2, index_3, 1], c='green', label="P (C2' endo)")
    plt.scatter(np.array(new_data_first)[index_list_c_3, index_3, 0],
                np.array(new_data_first)[index_list_c_3, index_3, 1], c='orange', label="P (C3' endo)")
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.legend()
    plt.savefig(folder + 'first_pucker_both')
    plt.close()

    # For the first ring the information to which Pucker pair group the element belongs is  added to the suite object.
    for i in range(len(complete_test_suites_with_answer)):
        dist_1 = np.array(new_data_first)[i, index_3, 1]
        complete_test_suites_with_answer[i].pucker_distance_1 = dist_1
        if dist_1 > 2.9:
            complete_test_suites_with_answer[i].pucker = 'c3'
        else:
            complete_test_suites_with_answer[i].pucker = 'c2'


    # Part 2: For the second sugar ring:

    # Index N1/N9 (second ring):
    index_1 = 4
    # Index C1' (second ring):
    index_2 = 3
    # Index P (second ring):
    index_3 = 5
    new_data_second = shift_and_rotate_data(six_chain_data, index_1=index_1, index_2=index_2, index_3=index_3)
    index_list_c_3 = [i for i in range(len(new_data_second)) if np.array(new_data_second)[i, index_3, 1] > 2.9]
    index_list_c_2 = [i for i in range(len(new_data_second)) if np.array(new_data_second)[i, index_3, 1] <= 2.9]
    plt.scatter(np.array(new_data_second)[:, index_1, 0], np.array(new_data_second)[:, index_1, 1], c='black',
                label='N1/N9')
    plt.scatter(np.array(new_data_second)[:, index_2, 0], np.array(new_data_second)[:, index_2, 1], c='blue',
                label="C1'")
    plt.scatter(np.array(new_data_second)[index_list_c_2, index_3, 0],
                np.array(new_data_second)[index_list_c_2, index_3, 1], c='green', label="P (C2' endo)")
    plt.scatter(np.array(new_data_second)[index_list_c_3, index_3, 0],
                np.array(new_data_second)[index_list_c_3, index_3, 1], c='orange', label="P (C3' endo)")
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.legend()
    plt.savefig(folder + 'second_pucker_both')
    plt.close()
    # For the second ring the information to which Pucker pair group the element belongs is  added to the suite object.
    for i in range(len(complete_test_suites_with_answer)):
        dist_2 = np.array(new_data_second)[i, index_3, 1]
        complete_test_suites_with_answer[i].pucker_distance_2 = dist_2
        if dist_2 > 2.9:
            complete_test_suites_with_answer[i].pucker = complete_test_suites_with_answer[i].pucker + 'c3'
        else:
            complete_test_suites_with_answer[i].pucker = complete_test_suites_with_answer[i].pucker + 'c2'

def get_test_suites(low_res_string, input_string_folder, recalculate):
    """
    The function loads the test suites (which are stored in the folder './trimmed_test_suites/'). In addition,
    the file suite_outliers_with_answers.csv is loaded and all suites that are not in the
    suite_outliers_with_answers.csv are removed. For each element, the two perpendicular distances are calculated and
    the possible answers stored in the file suite_outliers_with_answers.csv are added to the suite object.
    If the function has already been executed and recalculate==False, then the files from the last call are simply
    loaded.
    :param low_res_string: String that determines how the low detail representation is used.
    Currently, 'distance_angle_sphere_version_two' and ''distance_angle_sphere_version' are implemented. However,
    only the representation 'distance_angle_sphere_version_two' is used for the paper.
    :param input_string_folder: The folder in which the suite objects are saved and loaded.
    :param recalculate: In [True, False].
    :return:
     -- test_suites: a list of suite objects for all test suites
     -- low_res_test: a list of lists: A list containing for each parameter
        (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list containing the corresponding parameter for each
        element from test_suites
     -- x_y_data_2_test: The 5 point representation ( P is the origin, the two C1' atoms in the xy-plane and with the
        same positive y-value).  It holds: x_y_data_2_test.shape = (nr_test_suites, 5, 3).
    """
    # The location where test_suites, low_res_test and x_y_data_2_test are stored or read from:
    string_file_test_suites = input_string_folder + 'suites_cluster_test' + low_res_string + 'test_suites.pickle'
    string_file_low_res_test = input_string_folder + 'suites_cluster_test' + low_res_string + 'low_res_test.pickle'
    string_file_xy_data = input_string_folder + 'suites_cluster_test' + low_res_string + 'xy_data.pickle'
    # Check whether the function has already been called and whether the files have been saved:
    if os.path.isfile(string_file_test_suites) and not recalculate:
        with open(string_file_test_suites, 'rb') as f:
            complete_test_suites_with_answer = pickle.load(f)
        with open(string_file_low_res_test, 'rb') as f:
            low_res_test = pickle.load(f)
        with open(string_file_xy_data, 'rb') as f:
            x_y_data_2 = pickle.load(f)

    # If the function has not yet been called, perform all calculations of the function as explained in the description.
    else:
        # The plots are saved here
        folder = './out/test_data_pucker_plots/' + low_res_string + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # test suites are stored in the following folder:
        pdb_folder_test = './trimmed_test_suites/'
        string_folder_test = './out/saved_suite_lists_test/'
        # create the test objects:
        test_suites = parse_pdb_files(input_string_folder=string_folder_test, input_pdb_folder=pdb_folder_test)
        # remove objects which are not complete:
        test_suites = [suite for suite in test_suites if suite.dihedral_angles is not None and None not in suite._six_chain]
        # Some suites can occur in different pdb files. Remove the duplicates:
        test_suites = remove_duplicates_from_suite_list(test_suites)

        # Only work with suites that have the atom type 'atm'
        complete_test_suites = [suite for suite in test_suites if suite.dihedral_angles is not None and None not in suite._six_chain]
        complete_test_suites_2 = [suite for suite in complete_test_suites if suite.atom_types == 'atm']

        # Load the outlier file (suite_outliers_with_answers.csv) and save the possible answers in the suite objects.
        load_outlier_file_and_add_possible_answer_to_suite(complete_test_suites_2)
        complete_test_suites_with_answer = [suite for suite in complete_test_suites_2 if suite.answer is not None]
        plotting_perpendicular_distance_sorting_test_suites_to_pucker_pair_group(complete_test_suites_with_answer, folder)

        if low_res_string == 'distance_angle_sphere_version_two':
            low_res_test, x_y_data_2 = distance_angle_sphere_test_data_version_two(complete_test_suites_with_answer, folder,
                                                           string_file_low_res_test, string_file_test_suites, string_file_xy_data)
        # if low_res_string == 'distance_angle_sphere':
        #     low_res_test, x_y_data_2 = distance_angle_sphere_test_data(complete_test_suites_with_answer, folder,
        #                                                    string_file_low_res_test, string_file_test_suites, string_file_xy_data)
    create_table = False
    if create_table:
        # If one would like to automatically create the tables with the pdb names, the residue number of the test suites
        # with the corresponding resolution from the attachment, then one can call this function
        help_function_to_create_table_with_all_pdb_files_for_the_latex_file(complete_test_suites_with_answer)
    return complete_test_suites_with_answer, low_res_test, x_y_data_2

def cluster_trainings_suites(training_suites, name_, input_string_folder, recalculate,
                             low_res_string ='distance_angle_sphere_version_two'):
    """
    This function performs the following if it has not yet been executed (i.e. there are no precalculations), or
    recalculate==True performs the following:
    - First, the suite objects (from the training_suites) that belong to the sugar folding name_ are filtered.
    - Secondly, either the clustering is executed again or the cluster list that was determined with CLEAN-MINTAGE is
      loaded.
    - Finally, the cluster comparison files created by the Richardson group are loaded and used to create the plots
      (plots of the individual clusters and confusion matrices) and also the tables listing the number of elements in
       each cluster.
    :param training_suites: A list with all suite objects (from c3c3, c2c3, c3c2 and c2c2).
    :param name_: A string from [c3c3, c2c3, c3c2, c2c2]. Depending on the string, clustering is performed for the
                  corresponding sub-data set.
    :param input_string_folder: The folder in which the suite objects and all other objects returned by this function
                                are saved and loaded.
    :param recalculate: Either True or False, depending on whether you want to perform all calculations in this function
                        once again.
    :param low_res_string: String that determines how the low detail representation is used. Currently,
                           'distance_angle_sphere_version_two' and ''distance_angle_sphere_version' are implemented.
                           However, only the representation 'distance_angle_sphere_version_two' is used for the paper.
    :return:
    """
    # The location from where the objects returned by the function are stored or read from:
    string_file_cluster_suites = input_string_folder + 'suites_cluster_training' + name_ + low_res_string +'cluster_suites.pickle'
    string_file_cluster_list = input_string_folder + 'suites_cluster_training' + name_ + low_res_string +'cluster_list.pickle'
    string_file_list_elements = input_string_folder + 'suites_cluster_training' + name_ + low_res_string +'list_elements.pickle'
    string_file_cluster_names_richardson = input_string_folder + 'suites_cluster_training' + name_ + low_res_string +'cluster_names_richardson.pickle'
    string_file_cluster_names_richardson_all = input_string_folder + 'suites_cluster_training' + name_ + low_res_string +'cluster_names_richardson_all.pickle'
    string_file_low_res = input_string_folder + 'suites_cluster_training' + name_ + low_res_string + 'low_res.pickle'
    string_names_sorted = input_string_folder + 'suites_cluster_training' + name_ + low_res_string + 'names_sorted.pickle'
    string_xy_data = input_string_folder + 'suites_cluster_training' + name_ + low_res_string + 'xy_data.pickle'
    # Check if the results are stored in the folder input_string_folder. If True: load the results.
    if os.path.isfile(string_file_cluster_suites) and not recalculate:
        with open(string_file_cluster_suites, 'rb') as f:
            cluster_suites = pickle.load(f)
        with open(string_file_cluster_list, 'rb') as f:
            cluster_list_high_res_franziska = pickle.load(f)
        with open(string_file_list_elements, 'rb') as f:
            list_elements_franziska = pickle.load(f)
        with open(string_file_cluster_names_richardson, 'rb') as f:
            cluster_names_richardson_dominant = pickle.load(f)
        with open(string_file_cluster_names_richardson_all, 'rb') as f:
            cluster_names_richardson_numbers = pickle.load(f)
        with open(string_file_low_res, 'rb') as f:
            low_res = pickle.load(f)
        with open(string_names_sorted, 'rb') as f:
            richardson_cluster_names_sorted = pickle.load(f)
        with open(string_xy_data, 'rb') as f:
            x_y_data_2 = pickle.load(f)
    else:
        folder_plots = './out/cluster_trainings_suites/' + low_res_string + '/'
        if not os.path.exists(folder_plots):
            os.makedirs(folder_plots)

        # Perform the Procrustes algorithm (for plotting reasons):
        training_suites = shape_analysis.procrustes_analysis(training_suites)
        # Filter out only the data that belongs to the data set (from [c3c3, c2c3, c3c2, c2c2]):
        cluster_suites, procrustes_data, procrustes_data_backbone = get_procrustes_data_name(name_, input_suites=training_suites)

        recalculate_clustering = False
        if not recalculate_clustering:
            # Load cluster results master thesis Franziska: these are the tuning parameters for the q_fold value:
            if name_ is 'c3c3':
                q_fold = 0.09
            if name_ is 'c3c2':
                q_fold = 0.05
            if name_ is 'c2c3':
                q_fold = 0.07
            if name_ is 'c2c2':
                q_fold = 0.05
            # Load the cluster results already calculated:
            cluster_list_high_res_franziska = write_files.read_data_from_pickle("./out/saved_suite_lists_franziska/cluster_indices_merged_" + name_ + "_qfold" + str(q_fold))
            cluster_list_high_res_franziska = [list(set(cluster_list_high_res_franziska[i])) for i in range(len(cluster_list_high_res_franziska))]

        else:
            ###### Todo: insert code clustering
            pass

        # Determine the dihedral angle representation and determine the number of elements in each cluster for the
        # plot functions:
        dihedral_angles_suites = np.array([suite.dihedral_angles for suite in cluster_suites])
        list_elements_franziska = [len(cluster_list_high_res_franziska[i]) for i in range(len(cluster_list_high_res_franziska))]
        # Read in the cluster comparison files. For each element from our training data set there is a suite conformer,
        # which was determined with suitename and saved in a csv.
        cluster_names_richardson, richardson_cluster_list = read_in_cluster_comparison_files(cluster_list_high_res_franziska, cluster_suites, name_)
        # For each cluster, determine the dominant suite conformer and also a list of all suite conformers that occur in
        # the cluster:
        cluster_names_richardson_dominant = [Counter(cluster_names_richardson[i]).most_common(1)[0][0] for i in range(len(cluster_names_richardson))]
        cluster_names_richardson_numbers = [Counter(cluster_names_richardson[i]).most_common(len(Counter(cluster_names_richardson[i]))) for i in range(len(cluster_names_richardson))]
        # A plot function that generates the scatterplots for the cluster results.
        plot_high_res(dihedral_angles_suites, list_elements_franziska, name_, cluster_list_high_res_franziska, folder_plots, cluster_names_richardson_dominant, cluster_names_richardson_numbers)
        # A function to compare the MINT-AGE cluster results with the suitenames labeling:
        richardson_cluster_names_sorted = plot_confusion_matrix_clustering_comparison(cluster_list_high_res_franziska,
                                                                                      cluster_suites, folder_plots,
                                                                                      name_, richardson_cluster_list)

        # Currently only 'distance_angle_sphere_version_two' is maintained:
        if low_res_string == 'distance_angle_sphere_version_two':
            low_res, x_y_data_2 = distance_angle_sphere_training_data_version_two(cluster_suites, folder_plots, name_)

        # if low_res_string == 'distance_angle_sphere':
        #     low_res, x_y_data_2 = distance_angle_sphere_training_data(cluster_suites, folder_plots, name_)

        # Create the low-detail training plots:
        low_res_training_plots(cluster_list_high_res_franziska, cluster_names_richardson_numbers, folder_plots,
                               list_elements_franziska, low_res, low_res_string, name_)

        with open(string_file_cluster_suites, 'wb') as f:
            pickle.dump(cluster_suites, f)
        with open(string_file_cluster_list, 'wb') as f:
            pickle.dump(cluster_list_high_res_franziska, f)
        with open(string_file_list_elements, 'wb') as f:
            pickle.dump(list_elements_franziska, f)
        with open(string_file_cluster_names_richardson, 'wb') as f:
            pickle.dump(cluster_names_richardson_dominant, f)
        with open(string_file_cluster_names_richardson_all, 'wb') as f:
            pickle.dump(cluster_names_richardson_numbers, f)
        with open(string_file_low_res, 'wb') as f:
            pickle.dump(low_res, f)
        with open(string_names_sorted, 'wb') as f:
            pickle.dump(richardson_cluster_names_sorted, f)
        with open(string_xy_data, 'wb') as f:
            pickle.dump(x_y_data_2, f)
    create_table = True
    if create_table:
        # A function that automatically creates a table that is in latex format:
        create_table_clustering_results(cluster_list_high_res_franziska, cluster_names_richardson_numbers,
                                        cluster_suites)
    return cluster_suites, cluster_list_high_res_franziska, list_elements_franziska, cluster_names_richardson_dominant, low_res, richardson_cluster_names_sorted, cluster_names_richardson_numbers, x_y_data_2


def training_data_pucker_plots(training_suites):
    """
    Pucker plots for the training suites. A total of 3 plots are generated: The following is plotted for C2' and C3'
    endo: The dihedral angle 'n_2', the transformed data in which the perpendicular distance can be read (N1/N9 origin,
    C1' x-axis with positive x value and P in the x-y plane with positive y value) and an exemplary ribose ring for both
    foldings.
    """
    # The plots are saved here
    folder = './out/trainings_data_pucker_plots/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    string_ = 'nu_2_first_ring'
    nu = np.array([training_suites[i]._nu_1[0] for i in range(len(training_suites)) if
                   None not in training_suites[i]._five_chain])

    # Index N1/N9:
    index_1 = 0
    # Index C1':
    index_2 = 1
    # Index P:
    index_3 = 2
    plot_pucker(folder, index_1, index_2, index_3, nu, string_, training_suites)


def plot_pucker(folder, index_1, index_2, index_3, nu, string_, training_suites):
    """
    Pucker plots for the training suites. A total of 3 plots are generated: The following is plotted for C2' and C3'
    endo: The dihedral angle 'n_2', the transformed data in which the perpendicular distance can be read (N1/N9 origin,
    C1' x-axis with positive x value and P in the x-y plane with positive y value) and an exemplary ribose ring for both
    foldings.
    :param folder: The plots are saved here.
    :param index_1: Index of N1/N9.
    :param index_2: Index of C1'.
    :param index_3: Index of P.
    :param nu: dihedral angle list.
    :param string_: A string that is passed for the plot.
    :param training_suites: The list of training suites.
    :return:
    """
    fontsize = 17
    data = np.array([training_suites[i]._five_chain for i in range(len(training_suites)) if
                     None not in training_suites[i]._five_chain])
    index_set_1 = [i for i in range(len(nu)) if nu[i] > 100]
    index_set_2 = [i for i in range(len(nu)) if nu[i] < 100]


    ## Plot 1: Plot 1. Histogram plot.
    plt.hist([(nu[i] + 180) % 360 - 180 for i in range(len(nu)) if i in index_set_1], 50, color='green')
    plt.hist([(nu[i] + 180) % 360 - 180 for i in range(len(nu)) if i in index_set_2], 50, color='orange')
    plt.xlabel(r'$\nu_2$', fontsize=fontsize)
    plt.ylabel('frequency', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(folder + string_ + 'trafo', bbox_inches='tight')
    plt.close()

    # Plot 2: Two exemplary rings are plotted here.
    plot_ring_C3_and_C2(folder, index_set_1, index_set_2, training_suites)

    # Transform data: N1/N9 origin, C1' x-axis with positive x value and P in the x-y plane with positive y value:
    new_data_ = shift_and_rotate_data(data, index_1, index_2, index_3)
    # Plot 3: Plot of the data in the transformed representation.
    plt.scatter(np.array(new_data_)[index_set_1][:, index_1, 0], np.array(new_data_)[index_set_1][:, index_1, 1],
                c='black', label='N1/N9')
    plt.scatter(np.array(new_data_)[index_set_1][:, index_2, 0], np.array(new_data_)[index_set_1][:, index_2, 1],
                c='blue', label="C1'")
    plt.scatter(np.array(new_data_)[index_set_1][:, index_3, 0], np.array(new_data_)[index_set_1][:, index_3, 1],
                c='green', label="P (C3' endo)")
    plt.scatter(np.array(new_data_)[index_set_2][:, index_1, 0], np.array(new_data_)[index_set_2][:, index_1, 1],
                c='black')
    plt.scatter(np.array(new_data_)[index_set_2][:, index_2, 0], np.array(new_data_)[index_set_2][:, index_2, 1],
                c='blue')
    plt.scatter(np.array(new_data_)[index_set_2][:, index_3, 0], np.array(new_data_)[index_set_2][:, index_3, 1],
                c='orange', label="P (C2' endo)")
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(folder + string_ + 'both', bbox_inches='tight')
    plt.close()

def plot_ring_C3_and_C2(folder, index_set_1, index_set_2, training_suites):
    suites_complete = [training_suites[i] for i in range(len(training_suites)) if
                       None not in training_suites[i]._five_chain]
    suite_1 = suites_complete[index_set_1[0]]
    ring_atoms_1 = suite_1.ring_atoms[:4]
    N_atom_1 = suite_1._five_chain[0]
    backbone_up_to_p_1 = suite_1.backbone_atoms[1:5]
    suite_2 = suites_complete[index_set_2[0]]
    ring_atoms_2 = suite_2.ring_atoms[:4]
    N_atom_2 = suite_2._five_chain[0]
    backbone_up_to_p_2 = suite_2.backbone_atoms[1:5]
    data_two_pucker_plots = np.array([np.vstack([N_atom_1, ring_atoms_1[1], ring_atoms_1[0], backbone_up_to_p_1[:2],
                                                 ring_atoms_1[2], ring_atoms_1[1], ring_atoms_1[0],
                                                 backbone_up_to_p_1]),
                                      np.vstack([N_atom_2, ring_atoms_2[1], ring_atoms_2[0], backbone_up_to_p_2[:2],
                                                 ring_atoms_2[2], ring_atoms_2[1], ring_atoms_2[0],
                                                 backbone_up_to_p_2])])
    data_new = shift_and_rotate_data(data_two_pucker_plots, 1, 2, 3)
    data_new = np.array(data_new)[:, 1:7, :]
    data_new = np.array(data_new)
    atom_colors = ['grey', 'blue', 'grey', 'grey', 'grey', 'grey']
    build_fancy_chain_plot(data_new, colors=['green', 'orange'], filename=folder + 'example_plots', lw_vec=[3, 2],
                           alpha_line_vec=[1, 1], without_legend=True, plot_atoms=True, atom_size_vector=[20, 20],
                           atom_alpha=1,
                           atom_color_matrix=np.array([atom_colors, atom_colors]))
    

def plot_high_res(dihedral_angles_suites, list_elements_franziska, name_, cluster_list_high_res_franziska, folder, names_cluster, cluster_names_richardson_numbers):
    """
    A plot function that is used in cluster_trainings_suites
    """
    dihedral_angles_suites_sorted = np.vstack([dihedral_angles_suites[cluster_list_high_res_franziska[i]] for i in range(len(cluster_list_high_res_franziska))])
    legend_names = ['Cl ' + str(i+1) + ' (' + (str([cluster_names_richardson_numbers[i][j][0] for j in range(len(cluster_names_richardson_numbers[i])) if cluster_names_richardson_numbers[i][j][0] != '!!'] ).replace("'", "")[1:-1]).replace(" ", "") + ')'  for i in range(len(cluster_list_high_res_franziska))]
    if name_ == 'c3c3':
        legend_names[0] = legend_names[0][:15] + '\n        ' + legend_names[0][15:]
    scatter_plots(dihedral_angles_suites_sorted,
                      filename=folder + 'scatter' + name_ + '_suites_with_richardson_classes',
                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                                    r'$\gamma$', r'$\delta_{2}$'], s=100, fontsize=30, fontsize_axis=15, fontsize_legend=28,
                      number_of_elements=list_elements_franziska, legend_names=legend_names)
    suite_titles = [r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                    r'$\gamma$', r'$\delta_{2}$']
    fontsize = 13
    s=20
    legend_names = ['Cl ' + str(i+1) for i in range(len(cluster_list_high_res_franziska))]
    if name_ == 'c3c3':
        i=4
        j=3
        for number_element in range(len(list_elements_franziska)):
            if number_element == 0:
                plt.scatter(dihedral_angles_suites_sorted[:list_elements_franziska[number_element], j],
                            dihedral_angles_suites_sorted[:list_elements_franziska[number_element], i],
                            c=COLORS_SCATTER[number_element], s=s, marker=MARKERS[number_element], label=legend_names[number_element])
            else:
                plt.scatter(dihedral_angles_suites_sorted[sum(list_elements_franziska[:number_element]):
                                                          sum(list_elements_franziska[:number_element + 1]), j],
                            dihedral_angles_suites_sorted[sum(list_elements_franziska[:number_element]):
                                                          sum(list_elements_franziska[:number_element + 1]), i],
                            c=COLORS_SCATTER[number_element], s=s, marker=MARKERS[number_element], label=legend_names[number_element])
        plt.xlabel(suite_titles[j], fontsize=fontsize)
        plt.ylabel(suite_titles[i], fontsize=fontsize)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.xticks([0, 90, 180, 270, 360], ['0', '90', '180', '270', '360'], fontsize=fontsize)
        plt.yticks([0, 90, 180, 270, 360], ['0', '90', '180', '270', '360'], fontsize=fontsize)
        plt.legend(fontsize=fontsize, ncol=4, columnspacing=0.6, markerscale=2,handletextpad=0.1)
        plt.title('HDC3C3', fontsize=fontsize)
        plt.savefig(folder + name_ +'high_res_i' + str(i) + '_j_' + str(j), bbox_inches='tight')
        plt.close()

def low_res_training_plots(cluster_list_high_res_franziska, cluster_names_richardson_numbers, folder_plots,
                           list_elements_franziska, low_res, low_res_string, name_):
    """
    This function is used to create the low-detail training plots.
    """
    low_res_array = np.array([np.array([low_res[j][k] for j in range(len(low_res))]) for k in range(len(low_res[0]))])
    low_res_cluster_data = np.vstack(
        [low_res_array[cluster_list_high_res_franziska[i]] for i in range(len(cluster_list_high_res_franziska))])
    titles_new_par = [r"$d_2$", r"$d_3$", r"$\alpha$", r"$\theta_1$", r"$\phi_1$", r"$\theta_2$", r"$\phi_2$"]
    legend_names = ['Cl ' + str(i + 1) + ' (' + (
    str([cluster_names_richardson_numbers[i][j][0] for j in range(len(cluster_names_richardson_numbers[i])) if
         cluster_names_richardson_numbers[i][j][0] != '!!']).replace("'", "")[1:-1]).replace(" ", "") + ')' for i in
                    range(len(cluster_list_high_res_franziska))]
    low_res_cluster_data_plot = np.array(low_res_cluster_data).copy()
    low_res_cluster_data_plot[:, 3:] = low_res_cluster_data_plot[:, 3:] * 180 / np.pi
    list_ranges = [[3.1, 5.4],
                   [4, 6.3],
                   [0, 180], [0, 180], [-180, 180], [0, 180], [-180, 180]]
    scatter_plots(low_res_cluster_data_plot,
                  filename=folder_plots + 'scatter' + name_ + low_res_string,
                  suite_titles=titles_new_par, s=100, fontsize=30,
                  number_of_elements=list_elements_franziska, list_ranges=list_ranges,
                  fontsize_axis=15, fontsize_legend=20, legend_names=legend_names)
    if name_ == 'c3c3':
        fontsize = 13
        plt.hist(low_res_array[cluster_list_high_res_franziska[0]][:, 1], color=COLORS[0], label=legend_names[0])
        plt.hist(low_res_array[cluster_list_high_res_franziska[1]][:, 1], color=COLORS[1], label=legend_names[1])
        plt.legend(fontsize=fontsize)
        plt.title(legend_names[0] + ' vs. ' + legend_names[1], fontsize=fontsize)
        plt.xlabel(r'$d_3$ in $\AA$', fontsize=fontsize)
        plt.xlim([4, 6.3])
        plt.ylabel('number of elements')
        plt.savefig(folder_plots + '1a_vs_1c')
        plt.close()
    fontsize = 13
    s = 20
    legend_names = ['Cl ' + str(i + 1) for i in range(len(cluster_list_high_res_franziska))]
    if name_ == 'c3c3':
        i = 6
        j = 4
        for number_element in range(len(list_elements_franziska)):
            if number_element == 0:
                plt.scatter(low_res_cluster_data_plot[:list_elements_franziska[number_element], j],
                            low_res_cluster_data_plot[:list_elements_franziska[number_element], i],
                            c=COLORS_SCATTER[number_element], s=s, marker=MARKERS[number_element],
                            label=legend_names[number_element])
            else:
                plt.scatter(low_res_cluster_data_plot[sum(list_elements_franziska[:number_element]):
                                                      sum(list_elements_franziska[:number_element + 1]), j],
                            low_res_cluster_data_plot[sum(list_elements_franziska[:number_element]):
                                                      sum(list_elements_franziska[:number_element + 1]), i],
                            c=COLORS_SCATTER[number_element], s=s, marker=MARKERS[number_element],
                            label=legend_names[number_element])
        plt.xlabel(titles_new_par[j], fontsize=fontsize)
        plt.ylabel(titles_new_par[i], fontsize=fontsize)
        plt.xticks([-180, -90, 0, 90, 180], ['-180', '-90', '0', '90', '180'], fontsize=fontsize)
        plt.yticks([-180, -90, 0, 90, 180], ['-180', '-90', '0', '90', '180'], fontsize=fontsize)
        plt.title('LDC3C3', fontsize=fontsize)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(folder_plots + name_ + 'low_res_i' + str(i) + '_j_' + str(j), bbox_inches='tight')
        plt.close()


def plot_confusion_matrix_clustering_comparison(cluster_list_high_res_franziska, cluster_suites, folder_plots, name_,
                                                richardson_cluster_list):
    """
    A function to compare the MINT-AGE cluster results with the suitenames labeling
    :param cluster_list_high_res_franziska: A list of lists with indices from the MINT-AGE clustering.
    :param cluster_suites: The list of suite objects.
    :param folder_plots: The plots are saved here.
    :param name_: A string from [c3c3, c2c3, c3c2, c2c2].
    :param richardson_cluster_list: list of lists with the suite conforms in the lists.
    :return:
    """
    richardson_cluster_names = list(richardson_cluster_list.keys())
    arg_sort_list = np.argsort(
        [-len(richardson_cluster_list[richardson_cluster_names[i]]) for i in range(len(richardson_cluster_list))])
    richardson_cluster_names_sorted = list(np.array(richardson_cluster_names)[arg_sort_list])
    if '!!' in richardson_cluster_names_sorted:
        richardson_cluster_names_sorted.remove('!!')
        richardson_cluster_names_sorted = richardson_cluster_names_sorted + ['!!']
    richardson_cluster_sorted = [richardson_cluster_list[richardson_cluster_names_sorted[i]] for i in
                                 range(len(richardson_cluster_names_sorted))]
    all_cluster_elements = [element for sublist in cluster_list_high_res_franziska for element in sublist]
    outlier = [i for i in range(len(cluster_suites)) if i not in all_cluster_elements]
    cluster_list_with_outliers = cluster_list_high_res_franziska + [outlier]
    confusion_matrix = np.zeros((len(cluster_list_with_outliers), len(richardson_cluster_sorted)))
    fontsize = 20
    for i in range(confusion_matrix.shape[0]):
        for j in range(0, confusion_matrix.shape[1]):
            cluster_i = cluster_list_with_outliers[i]
            cluster_j = richardson_cluster_sorted[j]
            confusion_matrix[i, j] = len([a for a in cluster_i if a in cluster_j])
    df_cm = pd.DataFrame(confusion_matrix,
                         index=[str(i + 1) for i in range(len(cluster_list_with_outliers) - 1)] + ['R'],
                         columns=richardson_cluster_names_sorted)
    plt.figure(figsize=(20, 20))
    ax = sn.heatmap(df_cm, annot=True, fmt=".0f", robust=True, linewidths=1, cmap="YlGnBu", linecolor='black',
                    annot_kws={"size": fontsize}, cbar=True, square=False)
    ax.tick_params(labelsize=fontsize)
    cbar_ = ax.collections[0].colorbar
    cbar_.ax.tick_params(labelsize=fontsize)
    plt.xlabel('suitename', fontsize=fontsize)
    plt.ylabel('MINT-AGE', fontsize=fontsize)
    if name_ == 'c2c3':
        plt.title("HDC2C3", fontsize=fontsize)
    if name_ == 'c3c3':
        plt.title("HDC3C3", fontsize=fontsize)
    if name_ == 'c2c2':
        plt.title("HDC2C2", fontsize=fontsize)
    if name_ == 'c3c2':
        plt.title("HDC3C2", fontsize=fontsize)
    plt.savefig(folder_plots + name_ + 'confusion_matrix.png', bbox_inches='tight')
    plt.close()
    return richardson_cluster_names_sorted




def plot_results_learning(folder, result_list, string_, all_classes=False, frechet=False,
                          cluster_names_richardson_numbers=None, lambda_=None, sigma=None,
                          low_res_array=None,
                          cluster_list_high_res_franziska=None, name_):
    """

    :param folder: The results are plotted here.
    :param result_list: A list of dictionaries. Each dictionary belongs to a test element
    :param string_:
    :param all_classes:
    :param frechet:
    :param cluster_names_richardson_numbers:
    :param lambda_:
    :param sigma:
    :param low_res_array:
    :param cluster_list_high_res_franziska:
    :return:
    """

    # The results are plotted here:
    folder = folder + '/lambda_' + str(lambda_) + '_sigma_' + str(sigma) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # indices of the test elements that are plotted in blue
    index_list = [i for i in range(len(result_list)) if
                  len(set(flatten_list(result_list[i]['next_frechet_mean'])) & set(result_list[i]['answer']))>0]

    # specify how many elements are blue, red or orange:
    true_false_list = [j for j in range(len(result_list)) if result_list[j]['possible']]
    orange_list = [i for i in range(len(result_list)) if i in true_false_list and i not in index_list]
    print('all', len(result_list))
    print('blue', len(index_list), np.round(len(index_list)/len(result_list),2))
    print('red',  len(result_list)-len(true_false_list), np.round((len(result_list)-len(true_false_list))/len(result_list), 2))
    print('orange', len(orange_list), np.round(len(orange_list)/len(result_list), 2))

    # Plot 1: plot whether the ratio of the blue, red and orange groups changes depending on the mahalanobis distance
    fontsize = 17
    distances_nearest = [result_list[i]['distance_next_frechet_cluster'][0] for i in range(len(result_list))]
    distances_nearest_groups = [[0, 3], [3, 6], [6, 9],  [9, np.max(distances_nearest)]]
    true_false_list = [j for j in range(len(result_list)) if result_list[j]['possible']]
    groups = [[[], [], []] for i in range(len(distances_nearest_groups))]
    for i in range(len(result_list)):
        for j in range(len(distances_nearest_groups)):
            if distances_nearest[i] >= distances_nearest_groups[j][0] and distances_nearest[i] <= distances_nearest_groups[j][1]:
                if i in index_list:
                    groups[j][0].append(i)
                else:
                    groups[j][1].append(i)
                    if i not in true_false_list:
                        groups[j][2].append(i)
    x_values_1 = [i*4 for i in range(len(distances_nearest_groups))]
    x_values_2 = [i*4+1 for i in range(len(distances_nearest_groups))]
    x_values_tiks = [i*4+0.5 for i in range(len(distances_nearest_groups))]
    y_values_1 = [len(groups[i][0]) for i in range(len(distances_nearest_groups))]
    y_values_2 = [len(groups[i][1]) for i in range(len(distances_nearest_groups))]
    y_values_3 = [len(groups[i][2]) for i in range(len(distances_nearest_groups))]
    plt.bar(x_values_1, y_values_1, label='correct answer')
    plt.bar(x_values_2, y_values_2, label='false answer')
    plt.bar(x_values_2, y_values_3, label='false answer', color='red')
    plt.xticks(x_values_tiks, [str(distances_nearest_groups[i][0]) + '-' + str(np.round(distances_nearest_groups[i][1], 1)) for i in range(len(distances_nearest_groups))], fontsize=fontsize)
    plt.ylabel('number of test suites', fontsize=fontsize)
    plt.xlabel('distance', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(folder + name_ + string_ + 'distance_frechet', bbox_inches='tight')
    plt.close()

    # Plot 2: plot whether the ratio of the blue, red and orange groups changes depending on the resolution of the
    # experiment
    resolutions = [float(result_list[i]['resolution']) for i in range(len(result_list))]
    resolutions_groups = [[np.min(resolutions), 2], [2, 2.5], [2.5, 3],  [3, np.max(resolutions)]]
    groups = [[[], [], []] for i in range(len(resolutions_groups))]
    true_false_list = [j for j in range(len(result_list)) if result_list[j]['possible']]
    for i in range(len(result_list)):
        for j in range(len(resolutions_groups)):
            if resolutions[i] >= resolutions_groups[j][0] and resolutions[i] < resolutions_groups[j][1]:
                if i in index_list:
                    groups[j][0].append(i)
                else:
                    groups[j][1].append(i)
                    if i not in true_false_list:
                        groups[j][2].append(i)

    x_values_1 = [i*4 for i in range(len(resolutions_groups))]
    x_values_2 = [i*4+1 for i in range(len(resolutions_groups))]
    x_values_tiks = [i*4+0.5 for i in range(len(resolutions_groups))]
    y_values_1 = [len(groups[i][0]) for i in range(len(resolutions_groups))]
    y_values_2 = [len(groups[i][1]) for i in range(len(resolutions_groups))]
    y_values_3 = [len(groups[i][2]) for i in range(len(resolutions_groups))]
    plt.bar(x_values_1, y_values_1, label='correct answer')
    plt.bar(x_values_2, y_values_2, label='false answer')
    plt.bar(x_values_2, y_values_3, color='red')
    plt.xticks(x_values_tiks, [str(resolutions_groups[i][0]) + '-' + str(resolutions_groups[i][1]) for i in range(len(resolutions_groups))], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('number of test suites', fontsize=fontsize)
    plt.xlabel(r'resolution in $\AA$', fontsize=fontsize)
    plt.savefig(folder + name_ + string_ + 'resolutions', bbox_inches='tight')
    plt.close()

    # Plot 3: plot whether the ratio of the blue, red and orange groups changes depending on prev or next !!
    key_list = ['prev_is_bangbang', 'next_is_bangbang']
    key_list_tiks = ['all', 'prev !!', 'next !!', 'prev and next !!', 'prev or next !!', 'prev and next not !!']
    true_false_list_all = [True for j in range(len(result_list))]
    true_false_list_0 = [result_list[j][key_list[0]] == 'True' for j in range(len(result_list))]
    true_false_list_1 = [result_list[j][key_list[1]] == 'True' for j in range(len(result_list))]
    true_false_list_2 = [bool(true_false_list_0[i]*true_false_list_1[i]) for i in range(len(true_false_list_0))]
    true_false_list_3 = [bool(true_false_list_0[i]+true_false_list_1[i]) for i in range(len(true_false_list_0))]
    true_false_list_4 = [True if (not true_false_list_0[i] and not true_false_list_1[i]) else False for i in range(len(true_false_list_0))]
    list_of_lists = [true_false_list_all, true_false_list_0, true_false_list_1, true_false_list_2, true_false_list_3, true_false_list_4]
    groups = [[[], [], []] for i in range(len(key_list_tiks))]
    for i in range(len(result_list)):
        for j in range(len(key_list_tiks)):
            if list_of_lists[j][i]:
                if i in index_list:
                    groups[j][0].append(i)
                else:
                    groups[j][1].append(i)
                    if i not in true_false_list:
                        groups[j][2].append(i)
    fontsize = 12
    fig = plt.figure(figsize=(12,4))
    x_values_1 = [i*8 for i in range(len(list_of_lists))]
    x_values_2 = [i*8+1 for i in range(len(list_of_lists))]
    x_values_tiks = [i*8+0.5 for i in range(len(list_of_lists))]
    y_values_1 = [len(groups[i][0]) for i in range(len(list_of_lists))]
    y_values_2 = [len(groups[i][1]) for i in range(len(list_of_lists))]
    y_values_3 = [len(groups[i][2]) for i in range(len(list_of_lists))]
    plt.bar(x_values_1, y_values_1, label='correct answer')
    plt.bar(x_values_2, y_values_2, label='false answer')
    plt.bar(x_values_2, y_values_3, label='false answer', color ='red')
    plt.xticks(x_values_tiks, [key_list_tiks[i] for i in range(len(list_of_lists))], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('number of test suites', fontsize=fontsize)
    plt.xlabel('group', fontsize=fontsize)
    plt.savefig(folder + name_ + string_ + 'group', bbox_inches='tight')
    plt.close()

    # Plot 4: Plot the total number in the blue, red and orange categories:
    fontsize = 17
    key_list_tiks = ['all', 'answer in list']
    true_false_list_0 = [not result_list[j]['possible'] for j in range(len(result_list))]
    list_of_lists = [true_false_list_all, true_false_list_0]
    groups = [[[], []] for i in range(len(key_list_tiks))]
    for i in range(len(result_list)):
        for j in range(2):
            if list_of_lists[j][i]:
                if i in index_list:
                    groups[j][0].append(i)
                else:
                    groups[j][1].append(i)
    x_values_1 = [0]
    x_values_2 = [1]
    y_values_1 = [len(groups[0][0])]
    y_values_2 = [len(groups[0][1])]
    y_values_3 = [len(groups[1][1]) for i in range(len(list_of_lists))]
    plt.bar(x_values_1, y_values_1, label='correct answer')
    plt.bar(x_values_2, y_values_2, label='false answer')
    plt.bar(x_values_2, y_values_3, label='not possible', color='red')
    plt.xticks([],[])
    plt.yticks(fontsize=fontsize)
    plt.ylabel('number of test suites', fontsize=fontsize)
    plt.xlabel('all elements', fontsize=fontsize)
    plt.savefig(folder + name_ + string_ + 'possible', bbox_inches='tight')
    plt.close()

    # Plot 5: create confusion_matrix from the paper:
    color_matrix, confusion_matrix, names_classes, names_richardson = create_confusion_matrix_learning(all_classes,
                                                                                                       frechet,
                                                                                                       index_list,
                                                                                             result_list)

    filename = 'confusion_matrix' + name_
    names_learning, names_richardson = create_xticks_and_yticks_for_confusion_matrix(cluster_names_richardson_numbers,
                                                                                     names_classes, names_richardson)
    if name_ == 'c2c2':
        fontsize=50
    if name_ == 'c3c3':
        fontsize=90
    if name_ == 'c3c2':
        fontsize=70
    if name_ == 'c2c3':
        fontsize=80
    plot_confusion_matrix(color_matrix, confusion_matrix, filename, folder, fontsize, names_learning, names_richardson)

    #create the correction plots showing the assignment to the MINT-AGE clusters for the blue, red and orange categories.
    create_correction_plots_c3c3(cluster_list_high_res_franziska, cluster_names_richardson_numbers, folder, index_list,
                                 low_res_array, orange_list, result_list, true_false_list)

    create_table = True
    if create_table:
        # This function creates a result table for all test items listed in the appendix of the paper
        create_result_table_for_all_test_elements(index_list, result_list)

def create_correction_plots_c3c3(cluster_list_high_res_franziska, cluster_names_richardson_numbers, folder, index_list,
                                 low_res_array, orange_list, result_list, true_false_list, list_elements_franziska):
    """
    create the correction plots showing the assignment to the MINT-AGE clusters for the blue, red and orange categories.
    """
    low_res_cluster_data = np.vstack(
        [low_res_array[cluster_list_high_res_franziska[i]] for i in range(len(cluster_list_high_res_franziska))])
    titles_new_par = [r"$d_2$", r"$d_3$", r"$\alpha$", r"$\theta_1$", r"$\phi_1$", r"$\theta_2$", r"$\phi_2$"]
    low_res_cluster_data_plot = np.array(low_res_cluster_data).copy()
    low_res_cluster_data_plot[:, 3:] = low_res_cluster_data_plot[:, 3:] * 180 / np.pi

    legend_names = ['Cl ' + str(i + 1) + ' (' + (
    str([cluster_names_richardson_numbers[i][j][0] for j in range(len(cluster_names_richardson_numbers[i])) if
         cluster_names_richardson_numbers[i][j][0] != '!!']).replace("'", "")[1:-1]).replace(" ", "") + ')' for i in
                    range(len(cluster_list_high_res_franziska))]
    plot_names = ['blue', 'orange', 'red']
    for plot_name in plot_names:
        i = 6
        j = 4
        fontsize = 13
        s = 20
        s_test = 50
        for number_element in range(len(list_elements_franziska)):
            if number_element == 0:
                plt.scatter(low_res_cluster_data_plot[:list_elements_franziska[number_element], j],
                            low_res_cluster_data_plot[:list_elements_franziska[number_element], i],
                            c=COLORS_SCATTER[number_element], s=s, marker=MARKERS[number_element],
                            label=legend_names[number_element])
            else:
                plt.scatter(low_res_cluster_data_plot[sum(list_elements_franziska[:number_element]):
                                                      sum(list_elements_franziska[:number_element + 1]), j],
                            low_res_cluster_data_plot[sum(list_elements_franziska[:number_element]):
                                                      sum(list_elements_franziska[:number_element + 1]), i],
                            c=COLORS_SCATTER[number_element], s=s, marker=MARKERS[number_element],
                            label=legend_names[number_element])
        lw = 1
        alpha = 1
        if plot_name == 'blue':
            plot_index_list = index_list
            color_ = 'C0'
        if plot_name == 'orange':
            plot_index_list = orange_list
            color_ = 'C1'
        if plot_name == 'red':
            plot_index_list = [k for k in range(len(result_list)) if k not in true_false_list]
            color_ = 'red'

        for run_index in plot_index_list:
            test_x = result_list[run_index]['low_res_test'][j] * 180 / np.pi
            test_y = result_list[run_index]['low_res_test'][i] * 180 / np.pi
            mu_x = result_list[run_index]['mu'][j] * 180 / np.pi
            mu_y = result_list[run_index]['mu'][i] * 180 / np.pi
            if np.abs(mu_x - test_x) < 180 and np.abs(mu_y - test_y) < 180:
                plt.plot([mu_x, test_x], [mu_y, test_y],
                         color=COLORS_SCATTER[result_list[run_index]['next_frechet_mean_nr'][0]], alpha=alpha,
                         linewidth=lw)
            elif np.abs(mu_x - test_x) > 180 and np.abs(mu_y - test_y) < 180:
                list_x = [mu_x, test_x]
                list_y = [mu_y, test_y]
                argmin_x = np.argmin(list_x)
                argmax_x = np.argmax(list_x)
                left_element_y = list_y[argmin_x]
                right_element_y = list_y[argmax_x]
                left_element_x = list_x[argmin_x]
                right_element_x = list_x[argmax_x]
                c = -(right_element_y - left_element_y) / (360 - np.abs(mu_x - test_x))
                plt.plot([-180, left_element_x], [right_element_y + c * (180 - right_element_x), left_element_y],
                         color=COLORS_SCATTER[result_list[run_index]['next_frechet_mean_nr'][0]], alpha=alpha,
                         linewidth=lw)
                plt.plot([180, right_element_x], [right_element_y + c * (180 - right_element_x), right_element_y],
                         color=COLORS_SCATTER[result_list[run_index]['next_frechet_mean_nr'][0]], alpha=alpha,
                         linewidth=lw)
            elif np.abs(mu_x - test_x) < 180 and np.abs(mu_y - test_y) > 180:
                list_x = [mu_x, test_x]
                list_y = [mu_y, test_y]
                # list_y = [input_data2[i, y], input_data1[i, y]]
                argmin_y = np.argmin(list_y)
                argmax_y = np.argmax(list_y)
                lower_element_y = list_y[argmin_y]
                upper_element_y = list_y[argmax_y]
                lower_element_x = list_x[argmin_y]
                upper_element_x = list_x[argmax_y]
                c = -(lower_element_x - upper_element_x) / (360 - np.abs(lower_element_y - upper_element_y))
                plt.plot([upper_element_x - c * (180 - upper_element_y), lower_element_x], [-180, lower_element_y],
                         color=COLORS_SCATTER[result_list[run_index]['next_frechet_mean_nr'][0]], alpha=alpha,
                         linewidth=lw)
                plt.plot([upper_element_x - c * (180 - upper_element_y), upper_element_x], [180, upper_element_y],
                         color=COLORS_SCATTER[result_list[run_index]['next_frechet_mean_nr'][0]], alpha=alpha,
                         linewidth=lw)
        for run_index in plot_index_list:
            test_x = result_list[run_index]['low_res_test'][j] * 180 / np.pi
            test_y = result_list[run_index]['low_res_test'][i] * 180 / np.pi
            plt.scatter([test_x],
                        [test_y], color=color_,
                        marker="X", s=s_test)
        plt.xlabel(titles_new_par[j], fontsize=fontsize)
        plt.ylabel(titles_new_par[i], fontsize=fontsize)
        plt.xticks([-180, -90, 0, 90, 180], ['-180', '-90', '0', '90', '180'], fontsize=fontsize)
        plt.yticks([-180, -90, 0, 90, 180], ['-180', '-90', '0', '90', '180'], fontsize=fontsize)
        plt.title('LDTC3C3', fontsize=fontsize)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        plt.savefig(folder + name_ + plot_name + 'low_res_i' + str(i) + '_j_' + str(j), bbox_inches='tight')
        plt.close()


def plot_confusion_matrix(color_matrix, confusion_matrix, filename, folder, fontsize, names_learning, names_richardson):
    step_size_x = 4
    step_size_y = 4
    x_values = [i * step_size_x for i in range(confusion_matrix.shape[0] + 1)]
    x_values_tiks = [i * step_size_x + step_size_x / 2 for i in range(confusion_matrix.shape[0])]
    y_values = [i * step_size_y for i in range(confusion_matrix.shape[1] + 1)]
    y_values_tiks = [i * step_size_y + step_size_y / 2 for i in range(confusion_matrix.shape[1])]
    fig, ax = plt.subplots(figsize=(3 * len(x_values_tiks), 4 * len(y_values_tiks)))
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            rect = plt.Rectangle((x_values[i], y_values[j]), step_size_x, step_size_y, facecolor=color_matrix[i][j],
                                 edgecolor='black')
            ax.add_patch(rect)
            plt.text(x_values_tiks[i], y_values_tiks[j], str(int(confusion_matrix[i, j])), fontsize=fontsize,
                     ha='center', va='center')
    for i in range(confusion_matrix.shape[0] + 1):
        for j in range(confusion_matrix.shape[1] + 1):
            plt.plot([x_values[i], x_values[i]], [np.min(y_values), np.max(y_values)], color='black', lw=3)
            plt.plot([np.min(y_values), np.max(x_values)], [y_values[j], y_values[j]], color='black', lw=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks(x_values_tiks, names_richardson, fontsize=fontsize, rotation=90)
    plt.yticks(y_values_tiks, names_learning, fontsize=fontsize)
    plt.ylabel('RNAprecis', fontsize=fontsize)
    plt.xlabel('possible answers', fontsize=fontsize)
    if name_ == 'c2c3':
        plt.title("LDTC2C3", fontsize=fontsize)
    if name_ == 'c3c3':
        plt.title("LDTC3C3", fontsize=fontsize)
    if name_ == 'c2c2':
        plt.title("LDTC2C2", fontsize=fontsize)
    if name_ == 'c3c2':
        plt.title("LDTC3C2", fontsize=fontsize)
    plt.savefig(folder + filename + '.png', bbox_inches='tight')
    plt.close()

    arr = plt.imread(folder + filename + '.png')
    min_1 = np.min([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    max_1 = np.max([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    min_2 = np.min([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    max_2 = np.max([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    arr_new = arr[min_1:max_1 + 1, min_2:max_2 + 1, :]
    # transparent_region_1 = [i for i in range(arr_new.shape[0]) if np.all(arr_new[i, :, :] == 1)]
    transparent_region_2 = [i for i in range(arr_new.shape[1]) if np.all(arr_new[:, i, :] == 1)]
    list_2 = []
    for i in transparent_region_2:
        added = False
        for sub_list in list_2:
            if (i - 1) in sub_list:
                sub_list.append(i)
                added = True
        if not added:
            list_2.append([i])
    remove_list = []
    counter = 0
    for list in list_2:
        if counter > 0:
            if len(list) == max([len(list_) for list_ in list_2]):
                for element in list:
                    remove_list.append(element)
        counter = counter + 1
    not_remove_list = [i for i in range(arr_new.shape[1]) if i not in remove_list]
    arr_new = arr_new[:, not_remove_list, :]
    transparent_region_1 = [i for i in range(arr_new.shape[0]) if np.all(arr_new[i, :, :] == 1)]
    list_1 = []
    for i in transparent_region_1:
        added = False
        for sub_list in list_1:
            if (i - 1) in sub_list:
                sub_list.append(i)
                added = True
        if not added:
            list_1.append([i])
    remove_list = []
    counter = 0
    for list in list_1:
        if len(list) == max([len(list_) for list_ in list_1]) or len(list) == np.sort([len(list_) for list_ in list_1])[-2]:
            if len(list) > 0:
                for element in list:
                    remove_list.append(element)
        counter = counter + 1
    not_remove_list = [i for i in range(arr_new.shape[0]) if i not in remove_list]
    arr_new = arr_new[not_remove_list, :, :]
    # "ValueError: ndarray is not C-contiguous" else
    arr_new = arr_new.copy(order='C')
    plt.imsave(folder + filename + '.png', arr_new)
    plt.close()

def plot_all_high_and_low_detail_shapes(training_suites, string_folder, folder):
    """
    This function creates 3 different plots:
     -- For the training data, the low detail and high detail plots are created for all pucker-pair groups.
     -- For the test data (these are read in separately in the function), the low detail plots are created for all
        pucker-pair groups.
    :param training_suites: list of suite objects.
    :param string_folder: The string_folder specifies where the suite objects are stored.
    :param folder: A string: the plots are saved here.
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Pucker pairs:
    names = ['c2c2', 'c3c3', 'c2c3', 'c3c2']
    xy_data_list = []
    low_res_array = []
    cluster_suite_list = []
    # 'distance_angle_sphere_version_two': means that the low detail shapes are calculated as described in the
    # paper: P is the origin, the two C1' atoms in the xy-plane and with the same positive y-value
    low_res_string = 'distance_angle_sphere_version_two'
    # calculate the low detail representation for all pucker-pair groups and save different representations of them in
    # the three lists xy_data_list (landmarks), low_res_array (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_2) and
    # cluster_suite_list where the suite objects are stored.
    for i in range(len(names)):
        cluster_suites, _, _, _, low_res, _, _, xy_data_2 = \
            cluster_trainings_suites(
            training_suites, name_=names[i], low_res_string=low_res_string, input_string_folder=string_folder,
            recalculate=True)
        xy_data_list.append(xy_data_2)
        low_res_array.append(np.array(low_res))
        cluster_suite_list.append(cluster_suites)

    nr_elements = [len(xy_data_2) for xy_data_2 in xy_data_list]
    low_res_array = np.hstack(low_res_array).transpose()

    # Plot 1: Low-detail presentation of the training suites.
    titles_new_par = [r"$d_2$", r"$d_3$", r"$\alpha$", r"$\theta_1$", r"$\phi_1$", r"$\theta_2$", r"$\phi_2$"]
    low_res_cluster_data_plot = np.array(low_res_array).copy()
    low_res_cluster_data_plot[:, 3:] = low_res_cluster_data_plot[:, 3:] *180/np.pi
    list_ranges = [[3.1, 5.4],
                   [4, 6.3],
                   [0, 180], [0, 180], [-180, 180], [0,180], [-180, 180]]
    legend_names = ['LDC3C3', 'LDC2C3', 'LDC3C2', 'LDC2C2']
    scatter_plots(low_res_cluster_data_plot,
                  filename=folder + 'scatter' + low_res_string,
                  suite_titles=titles_new_par, s=100, fontsize=30,
                  number_of_elements=nr_elements, list_ranges=list_ranges,
                  fontsize_axis=15, fontsize_legend=30, legend_names=legend_names)

    # Plot 2: High-detail presentation of the training suites.
    dihedral_angles_suites_sorted = [suite.dihedral_angles for cluster_suites in cluster_suite_list for suite in cluster_suites]
    legend_names = ['HDC3C3', 'HDC2C3', 'HDC3C2', 'HDC2C2']
    scatter_plots(np.array(dihedral_angles_suites_sorted),
                  filename=folder + 'scatter' + '_suites',
                  suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                                r'$\gamma$', r'$\delta_{2}$'], s=100, fontsize=30, fontsize_axis=15, fontsize_legend=30,
                  number_of_elements=nr_elements, legend_names=legend_names)



    # Plot 3: Low-detail presentation of the test suites.
    # First load all test suites:
    test_suites, low_res_test, x_y_data_2_test = get_test_suites(low_res_string=low_res_string, input_string_folder=string_folder, recalculate=False)
    # Dividing the test suites into 4 different pucker pair group data sets:
    c3_c3_index = [i for i in range(len(test_suites)) if test_suites[i].pucker=='c3c3']
    c2_c3_index = [i for i in range(len(test_suites)) if test_suites[i].pucker=='c2c3']
    c3_c2_index = [i for i in range(len(test_suites)) if test_suites[i].pucker=='c3c2']
    c2_c2_index = [i for i in range(len(test_suites)) if test_suites[i].pucker=='c2c2']
    low_res_array_sorted = np.hstack([np.array(low_res_test)[:, c3_c3_index], np.array(low_res_test)[:, c2_c3_index], np.array(low_res_test)[:, c3_c2_index], np.array(low_res_test)[:, c2_c2_index]]).transpose()
    low_res_cluster_data_plot_test = np.array(low_res_array_sorted).copy()
    low_res_cluster_data_plot_test[:, 3:] = low_res_cluster_data_plot_test[:, 3:] *180/np.pi
    legend_names = ['LDTC3C3', 'LDTC2C3', 'LDTC3C2', 'LDTC2C2']
    scatter_plots(low_res_cluster_data_plot_test,
                  filename=folder + 'scatter_test' + low_res_string,
                  suite_titles=titles_new_par, s=100, fontsize=30,
                  number_of_elements=[len(c3_c3_index), len(c2_c3_index), len(c3_c2_index), len(c2_c2_index)], list_ranges=list_ranges,
                  fontsize_axis=15, fontsize_legend=30, legend_names=legend_names)
    print('test')