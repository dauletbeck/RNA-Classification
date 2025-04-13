import os
import sys
import re
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import write_files
import seaborn as sn
import math
import csv
import pickle
from clean_mintage_code.constants import COLORS, MARKERS
from collections import Counter
sys.path.append('./clean_mintage_code/')
from code_king import low_detail_array

from utils import *
from plotting import *

from clean_mintage_code import shape_analysis
from clean_mintage_code.constants import COLORS_SCATTER
from clean_mintage_code.parse_functions import parse_pdb_files
from clean_mintage_code.plot_functions import build_fancy_chain_plot
from clean_mintage_code.shape_analysis import procrustes_on_suite_class

from clean_mintage_code.plot_functions import scatter_plots
from clean_mintage_code.data_functions import calculate_angle_3_points, rotation, rotation_matrix_x_axis, rotation_matrix_z_axis, mean_on_sphere_init
cwd = os.getcwd()
new_wd = os.path.join(cwd, "clean_mintage_code")
os.chdir(new_wd)
sys.path.append(cwd)




def learn_algorithm(cluster_suites, cluster_list_high_res_franziska, list_elements_franziska,
                    cluster_names_richardson_dominant, low_res, test_suites, low_res_test, richardson_names_pucker,
                    cluster_names_richardson_numbers, xy_data_2, x_y_data_2_test, name_):
    """
    This function consists of three large blocks:
    - In the first step, the center point and the covariance matrices for the data projected into the tangential space
      are calculated for each cluster.
    - In a second step, the cluster that is closest to the distance from step 1 is determined for each test object.
    - In the third step, the analysis plots are created.
    :param cluster_suites: The list of training suite objects (not used at the moment)
    :param cluster_list_high_res_franziska: A list of lists with indices from the MINT-AGE clustering.
    :param list_elements_franziska: A list giving the size of all clusters (not used at the moment)
    :param cluster_names_richardson_dominant: A list of strings indicating the dominant suite conformer for each
                                              MINT-AGE cluster (not used at the moment)
    :param low_res: a list of lists for the training suites: A list containing for each parameter
                    (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list containing the corresponding parameter for
                    each element from test_suites.
    :param test_suites: The list of test suite objects
    :param low_res_test: a list of lists for the test suites: A list containing for each parameter
                        (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list containing the corresponding parameter for
                        each element from test_suites.
    :param richardson_names_pucker: A list of all suite conformers in the subset
    :param cluster_names_richardson_numbers:
    :param xy_data_2: The 5 point representation for the training data (P is the origin, the two C1' atoms in the xy-plane
                      and with the same positive y-value).
    :param x_y_data_2_test: The 5 point representation for the test data (P is the origin, the two C1' atoms in the xy-plane
                            and with the same positive y-value).
    :param name_: A string from [c3c3, c2c3, c3c2, c2c2].
    :return:
    """
    # plots are saved here:
    folder = './out/test_data_low_best/' + low_res_string + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    cluster_names_richardson_classes_all = [[cluster_[0] for cluster_ in cluster_names_richardson_numbers[i] if cluster_[0] is not '!!'] for i in range(len(cluster_names_richardson_numbers))]
    low_res_array = np.array([np.array([low_res[j][k] for j in range(len(low_res))]) for k in range(len(low_res[0]))])
    test_pucker_index_list = [i for i in range(len(test_suites)) if test_suites[i].pucker == name_]

    # Step 1:
    transform = True
    lambda_, training_mean_shapes, R_1_list, R_2_list, mu_list, S_R_list = get_mahalanobis_distance_for_each_cluster(cluster_list_high_res_franziska, low_res, transform)

    # Step 2:
    # Go through all test objects and calculate the mahalanobis distance and add the next elements
    # A list of dictionaries. Each dictionary belongs to a test element
    result_list = []
    for i in test_pucker_index_list:
        test_suite = test_suites[i]
        answer = test_suite.answer
        is_cluster_names_in_answer = [richardson_cluster_names_sorted[counter] in answer for counter in range(len(richardson_cluster_names_sorted))]
        low_res_test_suite = np.array([low_res_test[j][i] for j in range(len(low_res_test))])

        distance_list_frechet = []
        for k in range(len(training_mean_shapes)):
            def distance_function(x):
                if transform:
                    s_1 = R_1_list[k] @ x[-6:-3]
                    s_1_tilde = s_1[1:]/np.linalg.norm(s_1[1:])*np.arccos(s_1[0])
                    s_2 = R_2_list[k] @ x[-3:]
                    s_2_tilde = s_2[1:]/np.linalg.norm(s_2[1:])*np.arccos(s_2[0])
                    x_sub = np.hstack([x[:3], s_1_tilde, s_2_tilde])
                else:
                    x_sub = x
                return np.sqrt((mu_list[k]-x_sub) @ np.linalg.inv(S_R_list[k]) @ (mu_list[k]-x_sub))

            s_1_test = spherical_to_cart_x_z_switch(low_res_test_suite[3], low_res_test_suite[4])
            s_2_test = spherical_to_cart_x_z_switch(low_res_test_suite[5], low_res_test_suite[6])
            test_element = np.array([low_res_test_suite[0], low_res_test_suite[1], low_res_test_suite[2],
                                     s_1_test[0], s_1_test[1], s_1_test[2], s_2_test[0], s_2_test[1], s_2_test[2]])
            distance_list_frechet.append(distance_function(test_element))
        only_min = True
        if only_min:
            sigma = 'only_min'
            index_names = [i for i in range(len(distance_list_frechet)) if distance_list_frechet[i]<=np.min(distance_list_frechet)]
            arg_sort = np.argsort(np.array(distance_list_frechet))
            frechet_answers_ordered_all = [cluster_names_richardson_classes_all[i] for i in arg_sort]
            frechet_answers_ordered_nr_all = [i for i in arg_sort]
            distances_all = [distance_list_frechet[i] for i in arg_sort]
        else:
            sigma = 3
            index_names = [i for i in range(len(distance_list_frechet)) if distance_list_frechet[i]<=sigma]
        arg_sort = np.argsort(np.array(distance_list_frechet))
        frechet_answers_ordered = [cluster_names_richardson_classes_all[i] for i in arg_sort if i in index_names]
        frechet_answers_ordered_nr = [i for i in arg_sort if i in index_names]
        distances = [distance_list_frechet[i] for i in arg_sort if i in index_names]

        if len(frechet_answers_ordered) == 0:
            distances.append(np.min(distance_list_frechet))
            frechet_answers_ordered.append(cluster_names_richardson_classes_all[np.argmin(distance_list_frechet)])
            frechet_answers_ordered_nr.append(np.argmin(distance_list_frechet))
        all_plots = True
        if all_plots:
            plot_single_test_element(cluster_list_high_res_franziska, folder + '/test_elements/',
                                     frechet_answers_ordered, i, index_names, test_suite, x_y_data_2_test, xy_data_2, cluster_suites)
        result_dict = {'suite': test_suite, 'answer': answer,
                       'possible': True in is_cluster_names_in_answer,
                       'distance_next_frechet_cluster': distances,
                       'next_frechet_mean': frechet_answers_ordered,
                       'next_frechet_mean_nr': frechet_answers_ordered_nr,
                       'distance_next_frechet_cluster_all': distances_all,
                       'next_frechet_mean_all': frechet_answers_ordered_all,
                       'next_frechet_mean_nr_all': frechet_answers_ordered_nr_all,
                       'low_res_test' : low_res_test_suite,
                       'mu': training_mean_shapes[frechet_answers_ordered_nr[0]]
                       }
        result_list.append(result_dict)

    # Step 3:
    # Loading information for the plot function ('at_start', 'at_end', 'prev_is_bangbang', 'next_is_bangbang',
    # 'resolution', 'deposit_year')
    load_information_for_plot_function(result_list)
    # Plots and results:
    plot_results_learning(folder, result_list, 'all', all_classes=True, frechet=True,
                          cluster_names_richardson_numbers=cluster_names_richardson_numbers,
                          lambda_=lambda_, sigma=sigma, low_res_array=low_res_array,
                          cluster_list_high_res_franziska=cluster_list_high_res_franziska, name_=name_)


def get_mahalanobis_distance_for_each_cluster(cluster_list_high_res_franziska, low_res, transform):
    d_2 = low_res[0]
    d_3 = low_res[1]
    alpha_2 = low_res[2]
    theta_1 = low_res[3]
    phi_1 = low_res[4]
    theta_2 = low_res[5]
    phi_2 = low_res[6]
    spherical_1_list = [
        np.array([spherical_to_cart_x_z_switch(theta_1[i], phi_1[i]) for i in cluster_list_high_res_franziska[j]]) for j
        in range(len(cluster_list_high_res_franziska))]
    mean_1_list = [mean_on_sphere_init(spherical_1_list[i])[0] for i in range(len(spherical_1_list))]
    spherical_2_list = [
        np.array([spherical_to_cart_x_z_switch(theta_2[i], phi_2[i]) for i in cluster_list_high_res_franziska[j]]) for j
        in range(len(cluster_list_high_res_franziska))]
    mean_2_list = [mean_on_sphere_init(spherical_2_list[i])[0] for i in range(len(spherical_2_list))]
    training_mean_shapes = [np.array([np.mean(np.array(d_2)[cluster_list_high_res_franziska[i]]),
                                      np.mean(np.array(d_3)[cluster_list_high_res_franziska[i]]),
                                      np.mean(np.array(alpha_2)[cluster_list_high_res_franziska[i]]),
                                      cart_to_spherical_x_z_switch(mean_1_list[i])[1],
                                      cart_to_spherical_x_z_switch(mean_1_list[i])[2],
                                      cart_to_spherical_x_z_switch(mean_2_list[i])[1],
                                      cart_to_spherical_x_z_switch(mean_2_list[i])[2]]) for i in
                            range(len(cluster_list_high_res_franziska))]

    S_R_list = []
    mu_list = []
    R_1_list = []
    R_2_list = []
    for k in range(len(training_mean_shapes)):
        mu = np.array([np.mean(np.array(d_2)[cluster_list_high_res_franziska[k]]),
                       np.mean(np.array(d_3)[cluster_list_high_res_franziska[k]]),
                       np.mean(np.array(alpha_2)[cluster_list_high_res_franziska[k]]),
                       mean_1_list[k][0], mean_1_list[k][1], mean_1_list[k][2],
                       mean_2_list[k][0], mean_2_list[k][1], mean_2_list[k][2]])
        elements = [np.array([np.array(d_2)[cluster_list_high_res_franziska[k][element_counter]],
                              np.array(d_3)[cluster_list_high_res_franziska[k][element_counter]],
                              np.array(alpha_2)[cluster_list_high_res_franziska[k][element_counter]],
                              spherical_1_list[k][element_counter][0], spherical_1_list[k][element_counter][1],
                              spherical_1_list[k][element_counter][2],
                              spherical_2_list[k][element_counter][0], spherical_2_list[k][element_counter][1],
                              spherical_2_list[k][element_counter][2]]) for element_counter in
                    range(len(cluster_list_high_res_franziska[k]))]


        if transform:
            elements_old = elements.copy()
            R_1 = rotation(mean_1_list[k], np.array([1, 0, 0]))
            R_2 = rotation(mean_2_list[k], np.array([1, 0, 0]))
            R_1_list.append(R_1)
            R_2_list.append(R_2)
            spherical_1_list_transformed = [(R_1 @ spherical_1_list[k][element_counter])[1:] / np.linalg.norm(
                (R_1 @ spherical_1_list[k][element_counter])[1:]) * np.arccos(
                (R_1 @ spherical_1_list[k][element_counter])[0]) for element_counter in
                                            range(len(cluster_list_high_res_franziska[k]))]
            spherical_2_list_transformed = [(R_2 @ spherical_2_list[k][element_counter])[1:] / np.linalg.norm(
                (R_2 @ spherical_2_list[k][element_counter])[1:]) * np.arccos(
                (R_2 @ spherical_2_list[k][element_counter])[0]) for element_counter in
                                            range(len(cluster_list_high_res_franziska[k]))]

            elements = [np.array([np.array(d_2)[cluster_list_high_res_franziska[k][element_counter]],
                                  np.array(d_3)[cluster_list_high_res_franziska[k][element_counter]],
                                  np.array(alpha_2)[cluster_list_high_res_franziska[k][element_counter]],
                                  spherical_1_list_transformed[element_counter][0],
                                  spherical_1_list_transformed[element_counter][1],
                                  spherical_2_list_transformed[element_counter][0],
                                  spherical_2_list_transformed[element_counter][1]])
                        for element_counter in range(len(cluster_list_high_res_franziska[k]))]
            mu = np.array([np.mean(np.array(d_2)[cluster_list_high_res_franziska[k]]),
                           np.mean(np.array(d_3)[cluster_list_high_res_franziska[k]]),
                           np.mean(np.array(alpha_2)[cluster_list_high_res_franziska[k]]),
                           0, 0, 0, 0])
        elements_array = np.array(elements)

        S_R = (1 / (len(elements) - 1)) * np.dot((elements_array - mu).T, (elements_array - mu))
        if k == 0:
            if os.path.isfile('./out/SR_target.csv'):
                S_R_target = np.loadtxt('./out/SR_target.csv', delimiter=',')
            else:
                if name_ == 'c3c3':
                    S_R_target = S_R.copy()
                    np.savetxt('./out/SR_target.csv', S_R, delimiter=',')
                else:
                    print('Warning no S_R_target loaded. Work with first cluster from class')
                    S_R_target = S_R.copy()
        lambda_ = 0.5
        if elements_array.shape[1] < 20:
            S_R = lambda_ * S_R + (1 - lambda_) * S_R_target

        mu_list.append(mu)
        S_R_list.append(S_R)

        def distance_function(x):
            if transform:
                s_1 = R_1 @ x[-6:-3]
                s_1_tilde = s_1[1:] / np.linalg.norm(s_1[1:]) * np.arccos(s_1[0])
                s_2 = R_2 @ x[-3:]
                s_2_tilde = s_2[1:] / np.linalg.norm(s_2[1:]) * np.arccos(s_2[0])
                x_sub = np.hstack([x[:3], s_1_tilde, s_2_tilde])
            else:
                x_sub = x
            return (1 / np.sqrt(np.linalg.matrix_rank(S_R) - 1)) * np.sqrt(
                (mu - x_sub) @ np.linalg.inv(S_R) @ (mu - x_sub))
    return lambda_, training_mean_shapes, R_1_list, R_2_list, mu_list, S_R_list


def load_information_for_plot_function(result_list):
    """
    Loading information for the plot function ('at_start', 'at_end', 'prev_is_bangbang', 'next_is_bangbang',
    'resolution', 'deposit_year'). The information is added to result_list.
    :param result_list: A list of dictionaries. Each dictionary belongs to a test element
    """
    with open('./annotated_testing_results/' + name_ + '_annote.csv', newline='') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=';', quotechar='|')
        counter = 0
        for row in csv_file:
            if counter == 0:
                row_0 = row
                for i in range(len(row_0)):
                    row_0[i] = row_0[i].replace(' ', '')
            else:
                if counter == 1:
                    for i in range(len(row)):
                        csv_as_list = {row_0[i]: [row[i]] for i in range(len(row_0))}
                else:
                    for i in range(len(row)):
                        csv_as_list[row_0[i]].append(row[i])
            counter = counter + 1
    for i in range(len(result_list)):
        for j in range(len(csv_as_list[row_0[0]])):
            if result_list[i]['suite']._filename == csv_as_list['pdbid'][j] and result_list[i]['suite']._name_chain == \
                    csv_as_list['chain'][j] and result_list[i]['suite']._number_second_residue == int(
                    csv_as_list['resseq'][j]):
                for k in range(3, len(row_0)):
                    result_list[i][row_0[k]] = csv_as_list[row_0[k]][j]


def table_training_suites(training_suites):
    nr_tables = 3
    nr_elements = 6
    name_set_list = list(set([training_suites[i]._filename for i in range(len(training_suites))]))
    name_set_list.sort()
    url_list = ['https://www.rcsb.org/structure/'+ name_set_list[i] + '/' for i in range(len(name_set_list))]
    res_list = []
    for i in range(len(url_list)):
        print(i, len(url_list))
        response = requests.get(url_list[i])
        index = str(response.content).find('Resolution')
        print(str(response.content)[index:index+50])
        resolution = re.search(r"[-+]?\d*\.\d+|\d+", str(response.content)[index:]).group()
        print(resolution)
        res_list.append(resolution)

    nr_rows = int(np.round(len(set([training_suites[i]._filename for i in range(len(training_suites))])) / nr_tables))
    index_list = [i*nr_rows for i in range(nr_tables)] + [int(len(name_set_list))]
    for table_index in range(nr_tables):
        print('\\begin{minipage}[b]{0.3\\textwidth}')
        print('\\begin{tabular}{c|c}')
        print('PDB & Res \\\\')
        print('\hline')
        for i in np.arange(index_list[table_index], index_list[table_index+1]):
            pdb_name = name_set_list[i]
            #index_list_ = [training_suites[j]._name_chain + str(training_suites[j]._number_second_residue) for j in range(len(training_suites)) if training_suites[j]._filename == pdb_name]
            #string_name = ''
            print(name_set_list[i] + ' &  ' + res_list[i] + ' \\\\')
        print('\end{tabular}')
        print('\end{minipage}')
        print('\hspace*{0.02\\textwidth}')
    print('test')


if __name__ == '__main__':
    # The string_folder specifies where the suite objects are stored. If parse_pdb_files has already been executed,
    # then only the list of suite objects is loaded.
    string_folder = './out/saved_suite_lists/'
    # The folder in which the pdb files from the training data set are saved:
    pdb_folder = './rna2020_pruned_pdbs/'
    # Import the pdb files from the folder 'pdb_folder' and create the training_suites objects.
    training_suites = parse_pdb_files(input_string_folder=string_folder, input_pdb_folder=pdb_folder)
    # Plot functions for the paper:
    all_plots = True
    if all_plots:
        # Pucker plots for the training suites. A total of 3 plots are generated: The following is plotted for C2' and
        # C3' endo:
        # -- The dihedral angle 'nu_2',
        # -- the transformed data in which the perpendicular distance can be read (N1/N9 origin, C1' x-axis with
        #    positive x value and P in the x-y plane with positive y value) and an exemplary
        # -- the ribose ring for both foldings.
        training_data_pucker_plots(training_suites)
        # This function creates 3 different plots:
        # -- For the training data, the low detail and high detail plots are created for all pucker-pair groups.
        # -- For the test data (these are read in separately in the function), the low detail plots are created for all
        #    pucker-pair groups.
        plot_all_high_and_low_detail_shapes(training_suites, string_folder, './out/low_res_training/')

    # 'distance_angle_sphere_version_two': means that the low detail shapes are calculated as described in the
    # paper: P is the origin, the two C1' atoms in the xy-plane and with the same positive y-value
    low_res_string = 'distance_angle_sphere_version_two'
    # The function loads the test suites (which are stored in the folder './trimmed_test_suites/') and returns the
    # following 3 objects:
    # -- test_suites: a list of suite objects for all test suites
    # -- low_res_test: a list of lists: A list containing for each parameter
    #    (d_2, d_3, alpha, theta_1, phi_1, theta_2, phi_1) a list containing the corresponding parameter for each
    #    element from test_suites
    # -- x_y_data_2_test: The 5 point representation ( P is the origin, the two C1' atoms in the xy-plane and with the
    # same positive y-value).  It holds : x_y_data_2_test.shape = (nr_test_suites, 5, 3).
    test_suites, low_res_test, x_y_data_2_test = get_test_suites(low_res_string=low_res_string,
                                                                 input_string_folder=string_folder, recalculate=False)
    # Perform the two steps (first clustering with CLEAN-MINTAGE then learning with RNA-precis) for the four data sets
    # individually. Note that you have to start with c3c3 when calculating for the first time, as the covariance matrix
    # of the first cluster from c3c3 (mainly 1a) is required for the other datasets in the learning step.
    for name_ in ['c3c3', 'c3c2', 'c2c3', 'c2c2']:
        # Executing the MINT-AGE clustering for the corresponding subdata set. Plots are also created and compared with
        # suitename labeling. The low detail representations for the suites are also calculated.
        # Note that this function is only called if it has not yet been called or recalucuate = True.
        cluster_suites, cluster_list_high_res_franziska, list_elements_franziska, cluster_names_richardson_dominant, \
            low_res, richardson_cluster_names_sorted, cluster_names_richardson_numbers, xy_data_2 = \
            cluster_trainings_suites(training_suites, name_=name_, low_res_string=low_res_string,
                                     input_string_folder=string_folder, recalculate=False)
        # The RNAprecis step:
        learn_algorithm(cluster_suites, cluster_list_high_res_franziska, list_elements_franziska,
                        cluster_names_richardson_dominant, low_res, test_suites, low_res_test, richardson_cluster_names_sorted,
                        cluster_names_richardson_numbers, xy_data_2, x_y_data_2_test, name_)


    table_training_suites(training_suites)