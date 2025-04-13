import json
import os

import numpy as np
import re

from matplotlib import pyplot as plot

import plot_functions
from constants import COLORS, COLORS_SCATTER
from plot_functions import build_fancy_chain_plot, scatter_plots, hist_own_plot, scatter_plots_two
from collections import Counter
from PNDS_PNS import torus_mean_and_var


def plot_clustering(suites, cluster_list, name, outlier_list, plot_combinations=False, dihedral_angles_suites=None):
    """
    A plotting function which plots the suites and the mesoscopic shapes of all clusters in the 'cluster_list'.
    :param suites: A list of suite objects.
    :param cluster_list: A list of lists with integers (each integer corresponds to a suite in the list of suites).
    :param name: The string of the dictionary.
    :param outlier_list: A list of integers. (each integer corresponds to a suite in the list of suites).
    :return:
    """
    if not os.path.exists(name):
        os.makedirs(name)
    procrustes_data_backbone = np.array([suite.procrustes_complete_suite_vector for suite in suites])
    procrustes_data = np.array([suite.procrustes_complete_mesoscopic_vector for suite in suites])

    for i in range(len(cluster_list)):
        build_fancy_chain_plot(procrustes_data[cluster_list[i]], filename=name + 'cluster_nr' + str(i) + '_mesoscopic')
        build_fancy_chain_plot(procrustes_data_backbone[cluster_list[i]],
                               filename=name + 'cluster_nr' + str(i) + '_suite')
    if len(outlier_list) > 0:
        build_fancy_chain_plot(procrustes_data[outlier_list], filename=name + 'outlier_mesoscopic')
        build_fancy_chain_plot(procrustes_data_backbone[outlier_list], filename=name + 'outlier_suite')
    if plot_combinations:
        for i in range(len(cluster_list)):
            for j in range(i, len(cluster_list)):
                build_fancy_chain_plot(procrustes_data_backbone[list(cluster_list[i]) + list(cluster_list[j])],
                                       filename=name + 'cluster_nr' + str(i) + 'and' + str(j) + '_suite',
                                       colors=[COLORS[0]] * len(cluster_list[i]) + [COLORS[1]] * len(cluster_list[j]),
                                       specific_legend_colors=[COLORS[0], COLORS[1]],
                                       specific_legend_strings=["Cluster " + str(i), "Cluster " + str(j)],
                                       create_label=False)
                scatter_plots(dihedral_angles_suites[list(cluster_list[i]) + list(cluster_list[j])],
                              filename=name + 'cluster_nr' + str(i) + 'and' + str(j) + '_suite_scatter',
                              number_of_elements=[len(cluster_list[i]), len(cluster_list[j])])

    # This function plots if desired some manually selected plots (time intensive). Usually not used.
    specific_plots(cluster_list, name, procrustes_data_backbone)


def specific_plots(cluster_list, name, procrustes_data_backbone):
    # this function creates some specific plots. Usually not used.
    specific_plot = False
    if specific_plot:
        i = 0
        j = 1
        k = 3
        l = 9
        m = 16
        build_fancy_chain_plot(procrustes_data_backbone[
                                   list(cluster_list[i]) + list(cluster_list[j]) + list(cluster_list[k]) + list(
                                       cluster_list[l]) + list(cluster_list[m])],
                               filename=name + 'cluster_nr' + str(i) + 'and' + str(j) + 'and' + str(k) + 'and' + str(
                                   l) + 'and' + str(m) + '_suite',
                               colors=[COLORS_SCATTER[i]] * len(cluster_list[i]) + [COLORS_SCATTER[j]] * len(
                                   cluster_list[j]) + [
                                          COLORS_SCATTER[k]] * len(cluster_list[k]) + [COLORS_SCATTER[l]] * len(
                                   cluster_list[l]) + [
                                          COLORS_SCATTER[m]] * len(cluster_list[m]),
                               specific_legend_colors=[COLORS_SCATTER[i], COLORS_SCATTER[j], COLORS_SCATTER[k],
                                                       COLORS_SCATTER[l], COLORS_SCATTER[m]],
                               specific_legend_strings=["Cluster " + str(i + 1), "Cluster " + str(j + 1),
                                                        "Cluster " + str(k + 1), "Cluster " + str(l + 1),
                                                        "Cluster " + str(m + 1)],
                               create_label=False,
                               alpha_line_vec=[0.1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[k]) + [1] * len(cluster_list[l]) + [1] * len(cluster_list[m]),
                               plot_atoms=True,
                               atom_alpha_vector=[0.1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[k]) + [1] * len(cluster_list[l]) + [1] * len(cluster_list[m]),
                               atom_color_vector=[COLORS_SCATTER[i]] * len(cluster_list[i]) + [COLORS_SCATTER[j]] * len(
                                   cluster_list[j]) + [
                                                     COLORS_SCATTER[k]] * len(cluster_list[k]) + [
                                                     COLORS_SCATTER[l]] * len(cluster_list[l]) + [
                                                     COLORS_SCATTER[m]] * len(cluster_list[m]), atom_size=0.1,
                               without_legend=True)

        i = 0
        j = 1
        k = 4
        l = 6
        m = 7
        build_fancy_chain_plot(procrustes_data_backbone[
                                   list(cluster_list[i]) + list(cluster_list[j]) + list(cluster_list[k]) + list(
                                       cluster_list[l]) + list(cluster_list[m])],
                               filename=name + 'cluster_nr' + str(i) + 'and' + str(j) + 'and' + str(k) + 'and' + str(
                                   l) + 'and' + str(m) + '_suite',
                               colors=[COLORS[0]] * len(cluster_list[i]) + [COLORS[1]] * len(cluster_list[j]) + [
                                   COLORS[2]] * len(cluster_list[k]) + [COLORS[3]] * len(cluster_list[l]) + [
                                          COLORS[4]] * len(cluster_list[m]),
                               specific_legend_colors=[COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4]],
                               specific_legend_strings=["Cluster " + str(i + 1), "Cluster " + str(j + 1),
                                                        "Cluster " + str(k + 1), "Cluster " + str(l + 1),
                                                        "Cluster " + str(m + 1)],
                               create_label=False,
                               alpha_line_vec=[0.1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[k]) + [1] * len(cluster_list[l]) + [1] * len(cluster_list[m]),
                               plot_atoms=True,
                               atom_alpha_vector=[0.1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[k]) + [1] * len(cluster_list[l]) + [1] * len(cluster_list[m]),
                               atom_color_vector=[COLORS[0]] * len(cluster_list[i]) + [COLORS[1]] * len(
                                   cluster_list[j]) + [
                                                     COLORS[2]] * len(cluster_list[k]) + [COLORS[3]] * len(
                                   cluster_list[l]) + [
                                                     COLORS[4]] * len(cluster_list[m]), atom_size=0.1,
                               without_legend=True)
        build_fancy_chain_plot(procrustes_data_backbone[cluster_list[1]],
                               filename=name + 'cluster_nr' + str(1) + '_suite_',
                               plot_atoms=True, specific_legend_colors=[COLORS[0]],
                               specific_legend_strings=["Pre cluster " + str(2)],
                               create_label=False)
        build_fancy_chain_plot(procrustes_data_backbone[cluster_list[0]],
                               filename=name + 'cluster_nr' + str(0) + '_suite_',
                               plot_atoms=True, specific_legend_colors=[COLORS[0]],
                               specific_legend_strings=["Pre cluster " + str(1)],
                               create_label=False)
        # plot: Introduction
        i = 8
        j = 11
        build_fancy_chain_plot(procrustes_data_backbone[list(cluster_list[i]) + list(cluster_list[j])],
                               filename=name + 'cluster_nr' + str(i) + 'and' + str(j) + '_suite',
                               colors=[COLORS_SCATTER[i]] * len(cluster_list[i]) + [COLORS_SCATTER[j]] * len(
                                   cluster_list[j]),
                               create_label=False,
                               alpha_line_vec=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]), lw=0.2,
                               plot_atoms=True,
                               atom_alpha_vector=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]),
                               atom_color_vector=[COLORS_SCATTER[i]] * len(cluster_list[i]) + [COLORS_SCATTER[j]] * len(
                                   cluster_list[j]), atom_size=0.2, without_legend=True)
        i = 1
        j = 4
        k = 5
        l = 6
        build_fancy_chain_plot(procrustes_data_backbone[
                                   list(cluster_list[i]) + list(cluster_list[j]) + list(cluster_list[k]) + list(
                                       cluster_list[l])],
                               filename=name + 'cluster_nr' + str(i) + 'and' + str(j) + 'and' + str(k) + 'and' + str(
                                   l) + '_suite',
                               colors=[COLORS[1]] * len(cluster_list[i]) + [COLORS[2]] * len(cluster_list[j]) + [
                                   'dimgrey'] * len(cluster_list[k]) + [COLORS[3]] * len(cluster_list[l]),
                               create_label=False,
                               alpha_line_vec=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[k]) + [1] * len(cluster_list[l]),
                               plot_atoms=True,
                               atom_alpha_vector=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[k]) + [1] * len(cluster_list[l]),
                               atom_color_vector=[COLORS[1]] * len(cluster_list[i]) + [COLORS[2]] * len(
                                   cluster_list[j]) + [
                                                     'dimgrey'] * len(cluster_list[k]) + [COLORS[3]] * len(
                                   cluster_list[l]), atom_size=0.1, without_legend=True)
        j = 5
        k = 6
        l = 10
        build_fancy_chain_plot(
            procrustes_data_backbone[list(cluster_list[j]) + list(cluster_list[k]) + list(cluster_list[l])],
            filename=name + 'cluster_nr' + str(j) + 'and' + str(k) + 'and' + str(l) + 'and' + '_suite',
            colors=[COLORS[1]] * len(cluster_list[j]) + [COLORS[2]] * len(cluster_list[k]) + [COLORS[3]] * len(
                cluster_list[l]),
            specific_legend_colors=[COLORS[1], COLORS[2], COLORS[3]],
            specific_legend_strings=["Cluster " + str(j), "Cluster " + str(k), "Cluster " + str(l)],
            create_label=False,
            alpha_line_vec=[1] * len(cluster_list[j]) + [1] * len(cluster_list[k]) + [1] * len(cluster_list[l]),
            plot_atoms=True,
            atom_alpha_vector=[1] * len(cluster_list[j]) + [1] * len(cluster_list[k]) + [1] * len(cluster_list[l]),
            atom_color_vector=[COLORS[1]] * len(cluster_list[j]) + [COLORS[2]] * len(cluster_list[k]) + [
                COLORS[3]] * len(
                cluster_list[l]), atom_size=0.1)


def shift_plot_function_cluster_and_repaired(atom_number, best_cluster_list, clash_suite, cluster_number_index,
                                             mean_cluster_list_second, new_shape_list_second, procrustes_data,
                                             second_atom_index, shift_atom_list, shift_atom_list_second,
                                             sorted_cluster_list, third_string, procrustes_data_backbone,
                                             complete_suites, number_neighbors, backbone_validation_list_2,
                                             cluster=True):
    second_shift_atoms = [shift_atom_list_second[i] for i in range(len(shift_atom_list_second)) if
                          shift_atom_list[atom_number][0] in shift_atom_list_second[i]]

    if cluster:
        specific_legend_strings = ['Cluster mean',
                                   'Shifted Sugar rings: ' + str(second_shift_atoms[second_atom_index][0] + 1) + ' and '
                                   + str(second_shift_atoms[second_atom_index][1] + 1), 'The clash shape']
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'mesoscopic_'
    else:
        specific_legend_strings = ['Neighbour mean',
                                   'Shifted Sugar rings: ' + str(second_shift_atoms[second_atom_index][0] + 1) + ' and '
                                   + str(second_shift_atoms[second_atom_index][1] + 1), 'The clash shape']
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'mesoscopic'
    build_fancy_chain_plot(
        np.vstack((mean_cluster_list_second[atom_number][cluster_number_index][second_atom_index].reshape(1, 6, 3),
                   new_shape_list_second[atom_number][cluster_number_index][second_atom_index].reshape(1, 6, 3),
                   procrustes_data[clash_suite].reshape(1, 6, 3))),
        colors=[COLORS[0], COLORS[2], COLORS[1]],
        create_label=False,
        lw=1,
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[0], COLORS[2], COLORS[1]],
        filename=filename)

    cluster_index_list = sorted_cluster_list[best_cluster_list[atom_number][cluster_number_index]]
    if cluster:
        specific_legend_strings = ['Cluster with ' + str(len(cluster_index_list)) + ' elements',
                                   'Shifted sugar rings: ' + str(second_shift_atoms[second_atom_index][0] + 1) + ' and '
                                   + str(second_shift_atoms[second_atom_index][1] + 1),
                                   'The clash shape']
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'mesoscopic_cluster_nr_' + str(
            best_cluster_list[atom_number][cluster_number_index])
    else:
        specific_legend_strings = ['The closest ' + str(len(cluster_index_list)) + ' elements',
                                   'Shifted sugar rings: ' + str(second_shift_atoms[second_atom_index][0] + 1) + ' and '
                                   + str(second_shift_atoms[second_atom_index][1] + 1),
                                   'The clash shape']
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'mesoscopic_all_neighbours'
    build_fancy_chain_plot(np.vstack((procrustes_data[cluster_index_list],
                                      new_shape_list_second[atom_number][cluster_number_index][
                                          second_atom_index].reshape(1, 6, 3),
                                      procrustes_data[clash_suite].reshape(1, 6, 3))),
                           colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[2]] + [COLORS[1]],
                           lw_vec=[0.1] * len(cluster_index_list) + [2] + [2],
                           create_label=False,
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0], COLORS[2], COLORS[1]],
                           filename=filename)
    if len(complete_suites[clash_suite].erraser) > 0 and complete_suites[clash_suite].erraser[
        'rotated_mesoscopic_sugar_rings'] is not None:
        erraser_mesoscopic = complete_suites[clash_suite].erraser['rotated_mesoscopic_sugar_rings']
        build_fancy_chain_plot(np.vstack((procrustes_data[cluster_index_list],
                                          new_shape_list_second[atom_number][cluster_number_index][
                                              second_atom_index].reshape(1, 6, 3),
                                          procrustes_data[clash_suite].reshape(1, 6, 3),
                                          erraser_mesoscopic.reshape(1, 6, 3))),
                               colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[2]] + [COLORS[1]] + [COLORS[3]],
                               lw_vec=[0.1] * len(cluster_index_list) + [2] + [2] + [2],
                               create_label=False,
                               specific_legend_strings=specific_legend_strings + ['Erraser Mesoscopic'],
                               specific_legend_colors=[COLORS[0], COLORS[2], COLORS[1], COLORS[3]],
                               filename=filename + 'erraser')

    if cluster:
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'suite_cluster_nr_' + \
                   str(best_cluster_list[atom_number][cluster_number_index])
    else:
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'suite_all_neighbours'

    build_fancy_chain_plot(np.vstack((procrustes_data_backbone[cluster_index_list],
                                      procrustes_data_backbone[clash_suite].reshape(1, 10, 3))),
                           colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[1]],
                           lw_vec=[0.1] * len(cluster_index_list) + [2],
                           create_label=False,
                           specific_legend_strings=['The corresponding ' + str(len(cluster_index_list)) + ' suites',
                                                    'The clash suite'],
                           specific_legend_colors=[COLORS[0], COLORS[1]],
                           filename=filename)

    if len(complete_suites[clash_suite].erraser) > 0 and complete_suites[clash_suite].erraser[
        'rotated_backbone'] is not None:
        erraser_suite = complete_suites[clash_suite].erraser['rotated_backbone']
        build_fancy_chain_plot(np.vstack((procrustes_data_backbone[cluster_index_list],
                                          procrustes_data_backbone[clash_suite].reshape(1, 10, 3),
                                          erraser_suite.reshape(1, 10, 3))),
                               colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[1]] + [COLORS[3]],
                               lw_vec=[0.1] * len(cluster_index_list) + [2] + [2],
                               create_label=False,
                               specific_legend_strings=['The corresponding ' + str(len(cluster_index_list)) + ' suites',
                                                        'The clash suite',
                                                        'Erraser suite'],
                               specific_legend_colors=[COLORS[0], COLORS[1], COLORS[3]],
                               filename=filename + 'erraser')

    micro_cluster_list = [complete_suites[i].clustering['suite_True'] for i in cluster_index_list]
    micro_elements = []
    colors = []
    names = []
    important_cluster_list = []
    colors_legend = []
    counter = 4
    for micro_cluster in Counter(micro_cluster_list).most_common(100):
        if micro_cluster[1] / len(cluster_index_list) > 0.25 and micro_cluster[0] is not None:
            important_cluster_list = important_cluster_list + [micro_cluster[0]]
            micro_elements = micro_elements + [cluster_index_list[i] for i in range(len(cluster_index_list)) if
                                               micro_cluster_list[i] == micro_cluster[0]]
            colors = colors + [COLORS[counter]] * micro_cluster[1]
            colors_legend = colors_legend + [COLORS[counter]]
            names = names + ['Corresponding suite cluster with ' + str(micro_cluster[1]) + ' elements']
            counter = counter + 1

    if cluster:
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'suite_cluster_nr_' + \
                   str(best_cluster_list[atom_number][cluster_number_index]) + '_suite_cluster'
    else:
        filename = third_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + 'suite_all_neighbours_cluster'
    build_fancy_chain_plot(np.vstack((procrustes_data_backbone[micro_elements],
                                      procrustes_data_backbone[clash_suite].reshape(1, 10, 3))),
                           colors=colors + [COLORS[1]],
                           lw_vec=[0.1] * len(colors) + [2],
                           create_label=False,
                           specific_legend_strings=names + ['The clash suite'],
                           specific_legend_colors=colors_legend + [COLORS[1]],
                           filename=filename)
    if len(complete_suites[clash_suite].erraser) > 0 and complete_suites[clash_suite].erraser[
        'rotated_backbone'] is not None:
        erraser_suite = complete_suites[clash_suite].erraser['rotated_backbone']
        build_fancy_chain_plot(np.vstack((procrustes_data_backbone[micro_elements],
                                          procrustes_data_backbone[clash_suite].reshape(1, 10, 3),
                                          erraser_suite.reshape(1, 10, 3))),
                               colors=colors + [COLORS[1]] + [COLORS[3]],
                               lw_vec=[0.1] * len(colors) + [2] + [2],
                               create_label=False,
                               specific_legend_strings=names + ['The clash suite'] + ['Erraser suite'],
                               specific_legend_colors=colors_legend + [COLORS[1]] + [COLORS[3]],
                               filename=filename + 'erraser')

    for key in complete_suites[clash_suite].base_pairs.keys():
        print(clash_suite, complete_suites[clash_suite].base_pairs)
        base_pair_list = [i for i in range(len(complete_suites)) if
                          complete_suites[clash_suite].base_pairs[key][0] == complete_suites[i]._name]
        if len(base_pair_list) == 1:
            base_pair_suite = complete_suites[base_pair_list[0]]
            neighbors_base_pair_dummy = np.array(
                [np.linalg.norm(procrustes_data[base_pair_list[0]] - procrustes_data[element]) for element in
                 range(procrustes_data.shape[0])]).argsort()[1:]

            neighbors_base_pair = [neighbors_base_pair_dummy[i] for i in range(len(neighbors_base_pair_dummy)) if
                                   neighbors_base_pair_dummy[i] not in backbone_validation_list_2][0:number_neighbors]

            specific_legend_strings = ['The closest ' + str(len(cluster_index_list)) + ' elements',
                                       'Base pair with the sugar ring ' + str(key)]
            filename = third_string + 'best/' + 'clash_suite_is_' + str(
                clash_suite) + 'mesoscopic_base_pair_key_is_' + str(key)

            build_fancy_chain_plot(np.vstack((procrustes_data[neighbors_base_pair],
                                              base_pair_suite.procrustes_complete_mesoscopic_vector.reshape(
                                                  (1, 6, 3)))),
                                   colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[2]],
                                   lw_vec=[0.1] * len(cluster_index_list) + [2],
                                   create_label=False,
                                   specific_legend_strings=specific_legend_strings,
                                   specific_legend_colors=[COLORS[0], COLORS[2]],
                                   filename=filename)

    base_pair_string = third_string + 'best/all_base_pairs_reverse/'
    if not os.path.exists(base_pair_string):
        os.makedirs(base_pair_string)
    base_pairs = complete_suites[clash_suite].base_pairs
    if len(complete_suites[clash_suite].base_pairs) == 4 and (int(re.findall(r'\d+', base_pairs['5'][0])[-1]) + 1 ==
                                                              int(re.findall(r'\d+', base_pairs['4'][0])[-1]) ==
                                                              int(re.findall(r'\d+', base_pairs['3'][0])[-1]) - 1 ==
                                                              int(re.findall(r'\d+', base_pairs['2'][0])[-1]) - 2):
        atom_color_matrix_pair = [[COLORS[0]] * len(cluster_index_list) + [COLORS[1]]] * 6
        atom_color_matrix_pair = np.array(atom_color_matrix_pair).T
        # atom_color_matrix_pair = atom_color_matrix.copy()

        atom_color_matrix_pair[atom_color_matrix_pair.shape[0] - 1, 1] = 'grey'
        atom_color_matrix_pair[atom_color_matrix_pair.shape[0] - 1, 2] = 'blue'
        atom_color_matrix_pair[atom_color_matrix_pair.shape[0] - 1, 3] = 'plum'
        atom_color_matrix_pair[atom_color_matrix_pair.shape[0] - 1, 4] = 'teal'

        atom_color_matrix = [[COLORS[0]] * len(cluster_index_list) + [COLORS[1]] + [COLORS[1]]] * 6
        atom_color_matrix = np.array(atom_color_matrix).T
        atom_color_matrix[atom_color_matrix.shape[0] - 1, 1] = 'teal'
        atom_color_matrix[atom_color_matrix.shape[0] - 1, 2] = 'plum'
        atom_color_matrix[atom_color_matrix.shape[0] - 1, 3] = 'blue'
        atom_color_matrix[atom_color_matrix.shape[0] - 1, 4] = 'grey'
        atom_color_matrix[atom_color_matrix.shape[0] - 2, 1] = 'teal'
        atom_color_matrix[atom_color_matrix.shape[0] - 2, 2] = 'plum'
        atom_color_matrix[atom_color_matrix.shape[0] - 2, 3] = 'blue'
        atom_color_matrix[atom_color_matrix.shape[0] - 2, 4] = 'grey'
        base_pair_list = [i for i in range(len(complete_suites)) if
                          complete_suites[clash_suite].base_pairs['4'][0] == complete_suites[i]._name]
        if len(base_pair_list) == 1:
            base_pair_suite = complete_suites[base_pair_list[0]]
            neighbors_base_pair_dummy = np.array(
                [np.linalg.norm(procrustes_data[base_pair_list[0]] - procrustes_data[element]) for element in
                 range(procrustes_data.shape[0])]).argsort()[1:]

            neighbors_base_pair = [neighbors_base_pair_dummy[i] for i in range(len(neighbors_base_pair_dummy)) if
                                   neighbors_base_pair_dummy[i] not in backbone_validation_list_2][0:number_neighbors]

            specific_legend_strings = ['The closest ' + str(len(cluster_index_list)) + ' elements',
                                       'A mesoscopic shape that has base bonds with the clash shape']
            filename = base_pair_string + 'clash_suite_is_' + str(clash_suite) + 'mesoscopic_base_pair'

            build_fancy_chain_plot(np.vstack((procrustes_data[neighbors_base_pair],
                                              base_pair_suite.procrustes_complete_mesoscopic_vector.reshape(
                                                  (1, 6, 3)))),
                                   colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[1]],
                                   lw_vec=[0.1] * len(cluster_index_list) + [2],
                                   create_label=False,
                                   specific_legend_strings=specific_legend_strings,
                                   specific_legend_colors=[COLORS[0], COLORS[1]],
                                   filename=filename, atom_color_matrix=atom_color_matrix_pair, plot_atoms=True,
                                   atom_size_vector=[0.1] * len(cluster_index_list) + [10],
                                   atom_alpha_vector=[0.1] * len(cluster_index_list) + [1])

            specific_legend_strings = ['The closest ' + str(len(cluster_index_list)) + ' elements',
                                       'Shifted Sugar rings: ' + str(
                                           second_shift_atoms[second_atom_index][0] + 1) + ' and '
                                       + str(second_shift_atoms[second_atom_index][1] + 1), 'The clash shape']
            filename = base_pair_string + 'clash_suite_is_' + str(clash_suite)
            build_fancy_chain_plot(np.vstack((procrustes_data[cluster_index_list],
                                              new_shape_list_second[atom_number][cluster_number_index][
                                                  second_atom_index].reshape(1, 6, 3),
                                              procrustes_data[clash_suite].reshape(1, 6, 3))),
                                   colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[2]] + [COLORS[1]],
                                   lw_vec=[0.1] * len(cluster_index_list) + [2] + [2],
                                   create_label=False,
                                   specific_legend_strings=specific_legend_strings,
                                   specific_legend_colors=[COLORS[0], COLORS[2], COLORS[1]],
                                   filename=filename, atom_color_matrix=atom_color_matrix, plot_atoms=True,
                                   atom_size_vector=[0.1] * len(cluster_index_list) + [10] + [10],
                                   atom_alpha_vector=[0.1] * len(cluster_index_list) + [1] + [1])

            filename = base_pair_string + 'clash_suite_is_' + str(clash_suite) + 'suite_base_pair'
            build_fancy_chain_plot(np.vstack((procrustes_data_backbone[neighbors_base_pair],
                                              base_pair_suite.procrustes_complete_suite_vector.reshape((1, 10, 3)))),
                                   colors=[COLORS[0]] * len(cluster_index_list) + [COLORS[1]],
                                   lw_vec=[0.1] * len(cluster_index_list) + [2],
                                   create_label=False,
                                   specific_legend_strings=[
                                       'The corresponding ' + str(len(cluster_index_list)) + ' suites',
                                       'The outlier suite'],
                                   specific_legend_colors=[COLORS[0], COLORS[1]],
                                   filename=filename)


def help_plot_function_cluster_mean(clash_suite, first_string, mean_cluster_list, new_shape_list, procrustes_data,
                                    shift_atom_list, which_atom_is_best, which_cluster_is_best_index, cluster=True):
    if cluster:
        build_fancy_chain_plot(
            np.vstack((mean_cluster_list[which_atom_is_best][which_cluster_is_best_index].reshape(1, 6, 3),
                       new_shape_list[which_atom_is_best][which_cluster_is_best_index].reshape(1, 6, 3),
                       procrustes_data[clash_suite].reshape(1, 6, 3))),
            colors=[COLORS[0], COLORS[2], COLORS[1]],
            create_label=False,
            specific_legend_strings=['Cluster mean',
                                     'The shifted shape',
                                     'The clash shape'],
            lw=1,
            specific_legend_colors=[COLORS[0], COLORS[2], COLORS[1]],
            filename=first_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + '_shift_atom_' +
                     str(shift_atom_list[which_atom_is_best]) + '_after_shift')
    else:
        build_fancy_chain_plot(
            np.vstack((mean_cluster_list[which_atom_is_best][which_cluster_is_best_index].reshape(1, 6, 3),
                       new_shape_list[which_atom_is_best][which_cluster_is_best_index].reshape(1, 6, 3),
                       procrustes_data[clash_suite].reshape(1, 6, 3))),
            colors=[COLORS[0], COLORS[2], COLORS[1]],
            create_label=False,
            specific_legend_strings=['Neighbour mean',
                                     'The shifted shape',
                                     'The mesoscopic clash-shape '],
            lw=1,
            specific_legend_colors=[COLORS[0], COLORS[2], COLORS[1]],
            filename=first_string + 'best/' + 'clash_suite_is_' + str(clash_suite) + '_shift_atom_' +
                     str(shift_atom_list[which_atom_is_best]) + '_after_shift_neighbour')


def corona_plot_A33_A34(best_string, clash_and_repair_shapes, clash_suite_list, cluster_numbers, cluster_numbers_unique,
                        colors_meso, colors_validation, complete_suites_new, mean_shape_list, suite_backbone_list,
                        suite_correction_list, suite_index):
    color_model = [COLORS[1] if clash_suite_list[i] else COLORS[0] for i in range(len(cluster_numbers))]
    lw_model = [1 if clash_suite_list[i] else 0.3 for i in range(len(cluster_numbers))]
    specific_legend_strings = [
                                  'Suites $\mathfrak{c}_1,\dots, \mathfrak{c}_4, \mathfrak{c}_6,\dots, \mathfrak{c}_{10}$'] + [
                                  'Suite $\mathfrak{c}_5$'] + [
                                  'Torus mean from cluster  ' + str(cluster_numbers_unique[0] + 1)]
    build_fancy_chain_plot(np.vstack((np.array(suite_backbone_list), np.array(mean_shape_list))),
                           colors=color_model + colors_validation[:len(cluster_numbers_unique)],
                           create_label=False,
                           lw_vec=lw_model + [1] * len(cluster_numbers_unique),
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + [COLORS[1]] + colors_validation[
                                                                              :len(cluster_numbers_unique)],
                           filename=best_string + 'clash_suite' + complete_suites_new[suite_index].name + 'suite_1',
                           without_legend=True)
    color_model = [COLORS[1] if clash_suite_list[i] else COLORS[0] for i in range(len(cluster_numbers))]
    lw_model = [1 if clash_suite_list[i] else 0.3 for i in range(len(cluster_numbers))]
    specific_legend_strings = [
                                  'Suites $\mathfrak{c}_1,\dots, \mathfrak{c}_4, \mathfrak{c}_6,\dots, \mathfrak{c}_{10}$'] + [
                                  'Suite $\mathfrak{c}_5$'] + [
                                  'Torus mean from cluster  ' + str(cluster_numbers_unique[0] + 1)]
    build_fancy_chain_plot(np.vstack((np.array(suite_backbone_list), np.array(suite_correction_list))),
                           colors=color_model + 10 * colors_validation[:1],
                           create_label=False,
                           lw_vec=lw_model + [0.3] * 10,
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + [COLORS[1]] + colors_validation[
                                                                              :len(cluster_numbers_unique)],
                           filename=best_string + 'clash_suite' + complete_suites_new[suite_index].name + 'suite_1_new',
                           without_legend=True)
    specific_legend_strings = [
                                  r'Mesoscopics $m_{\mathfrak{c}_1},\dots, m_{\mathfrak{c}_{4}}, m_{\mathfrak{c}_{6}},\dots, m_{\mathfrak{c}_{10}}$'] + [
                                  '$m_{\mathfrak{c}_5}$'] + [
                                  r'Corrected mesoscopics $\mu_{\tau_{\mathfrak{c}_1}},\dots, \mu_{\tau_{\mathfrak{c}_{10}}}$']
    build_fancy_chain_plot(clash_and_repair_shapes,
                           colors=color_model + colors_meso,
                           create_label=False,
                           lw_vec=lw_model + [1] * len(cluster_numbers),
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + [COLORS[1]] + colors_validation[
                                                                              :len(cluster_numbers_unique)],
                           filename=best_string + 'clash_suite' + complete_suites_new[
                               suite_index].name + 'meso_cluster_all__', without_legend=True)
    specific_legend_strings = [
                                  r'Mesoscopics $m_{\mathfrak{c}_1},\dots, m_{\mathfrak{c}_{4}}, m_{\mathfrak{c}_{6}},\dots, m_{\mathfrak{c}_{10}}$'] + [
                                  '$m_{\mathfrak{c}_5}$'] + [
                                  r'Corrected mesoscopics $\mu_{\tau_{\mathfrak{c}_1}},\dots, \mu_{\tau_{\mathfrak{c}_{10}}}$']
    build_fancy_chain_plot(clash_and_repair_shapes,
                           colors=color_model + colors_meso,
                           create_label=False,
                           lw_vec=lw_model + [1] * len(cluster_numbers),
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + [COLORS[1]] + colors_validation[
                                                                              :len(cluster_numbers_unique)],
                           filename=best_string + 'clash_suite' + complete_suites_new[
                               suite_index].name + 'meso_cluster_all_new', without_legend=True)


def create_corona_plots(best_string, clash_and_repair_shapes, clash_suite_list, cluster_numbers, cluster_numbers_unique,
                        colors_meso, colors_validation, complete_suites_new, mean_shape_list, number_of_mean,
                        suite_backbone_list, suite_index):
    color_model = [COLORS[1] if clash_suite_list[i] else COLORS[0] for i in range(len(cluster_numbers))]
    lw_model = [1 if clash_suite_list[i] else 0.3 for i in range(len(cluster_numbers))]
    specific_legend_strings = ['The suites of the different models without clash'] + [
        'The suites of the different models with a clash'] + [
                                  'Cluster number ' + str(cluster_numbers_unique[i] + 1) + '; ' + str(
                                      number_of_mean[i]) + ' times' for i in range(len(number_of_mean))]
    build_fancy_chain_plot(np.vstack((np.array(suite_backbone_list), np.array(mean_shape_list))),
                           colors=color_model + colors_validation[:len(cluster_numbers_unique)],
                           create_label=False,
                           lw_vec=lw_model + [1] * len(cluster_numbers_unique),
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + [COLORS[1]] + colors_validation[
                                                                              :len(cluster_numbers_unique)],
                           filename=best_string + 'clash_suite' + complete_suites_new[suite_index].name + 'suite',
                           without_legend=True)
    specific_legend_strings = ['The mesoscopics of the different models without clash'] + [
        'The mesocopics of the different models with a clash'] + [
                                  'Cluster number ' + str(cluster_numbers_unique[i] + 1) + '; ' + str(
                                      number_of_mean[i]) + ' times' for i in range(len(number_of_mean))]
    build_fancy_chain_plot(clash_and_repair_shapes,
                           colors=color_model + colors_meso,
                           create_label=False,
                           lw_vec=lw_model + [1] * len(cluster_numbers),
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + [COLORS[1]] + colors_validation[
                                                                              :len(cluster_numbers_unique)],
                           filename=best_string + 'clash_suite' + complete_suites_new[
                               suite_index].name + 'meso_cluster', without_legend=True)
    if False not in clash_suite_list:
        specific_legend_strings = [r'Mesoscopics $m_{\mathfrak{c}_1},\dots, m_{\mathfrak{c}_{10}}$'] + [
            r'Corrected mesoscopics $\mu_{\tau_{\mathfrak{c}_1}},\dots, \mu_{\tau_{\mathfrak{c}_{10}}}$']
        build_fancy_chain_plot(clash_and_repair_shapes,
                               colors=color_model + colors_meso,
                               create_label=False,
                               lw_vec=lw_model + [1] * len(cluster_numbers),
                               specific_legend_strings=specific_legend_strings,
                               specific_legend_colors=[COLORS[1]] + colors_validation[:len(cluster_numbers_unique)],
                               filename=best_string + 'clash_suite' + complete_suites_new[
                                   suite_index].name + 'meso_cluster_all', without_legend=True)

        specific_legend_strings = ['Suites $\mathfrak{c}_1,\dots, \mathfrak{c}_{10}$'] + [
            'Torus mean from cluster  ' + str(cluster_numbers_unique[0] + 1)]
        build_fancy_chain_plot(np.vstack((np.array(suite_backbone_list), np.array(mean_shape_list))),
                               colors=[COLORS[1]] * len(cluster_numbers) + colors_validation[
                                                                           :len(cluster_numbers_unique)],
                               create_label=False,
                               lw_vec=[0.3] * len(cluster_numbers) + [1] * len(cluster_numbers_unique),
                               specific_legend_strings=specific_legend_strings,
                               specific_legend_colors=[COLORS[1]] + colors_validation[:len(cluster_numbers_unique)],
                               filename=best_string + 'all_models' + complete_suites_new[
                                   suite_index].name + 'suite_all', without_legend=True)


def single_corona_plots(best_string, complete_suites_new, counter, mesoscopic_clash_shape_copy, neighbors_cluster,
                        procrustes_data_backbone_benchmark, procrustes_data_small_rotated, suite_backbone, suite_index,
                        y_bar):
    specific_legend_strings = [
        r'Mesoscopics from $C_{i_{\mathfrak{c}_{' + str(counter) + '}}} \cap U_{\mathfrak{c}_1}$',
        r'Mesoscopic clash shape $m_{\mathfrak{c}_{' + str(counter) + '}}$',
        r'Corrected mesoscopic $m_{\tau_{\mathfrak{c}_{' + str(counter) + '}}}$']
    build_fancy_chain_plot(
        np.vstack((procrustes_data_small_rotated[neighbors_cluster],
                   mesoscopic_clash_shape_copy.reshape(1, 6, 3),
                   y_bar.reshape(1, 6, 3))),
        colors=[COLORS[0]] * len(neighbors_cluster) + [COLORS[1]] + [COLORS[2]],
        create_label=False,
        lw_vec=[0.1] * len(neighbors_cluster) + [2] + [2],
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[0], COLORS[1], COLORS[2]],
        filename=best_string + complete_suites_new[suite_index].name + 'meso_model_' + str(
            complete_suites_new[suite_index].model_number), without_legend=True)
    specific_legend_strings = [r'Suites in $C_{i_{\mathfrak{c}_{' + str(counter) + '}}} \cap U_{\mathfrak{c}_1}$',
                               r'Clash suite $\mathfrak{c}_{' + str(counter) + '}$',
                               r'Suite cluster mean $\tau_{\mathfrak{c}_{' + str(counter) + '}}$']
    build_fancy_chain_plot(
        np.vstack((procrustes_data_backbone_benchmark[neighbors_cluster],
                   suite_backbone.reshape(1, 10, 3),
                   np.mean(procrustes_data_backbone_benchmark[neighbors_cluster], axis=0).reshape(1, 10, 3))),
        colors=[COLORS[0]] * len(neighbors_cluster) + [COLORS[1]] + [COLORS[2]],
        create_label=False,
        lw_vec=[0.1] * len(neighbors_cluster) + [1] + [1],
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[0], COLORS[1], COLORS[2]],
        filename=best_string + complete_suites_new[suite_index].name + 'start_suite' + str(
            complete_suites_new[suite_index].model_number), without_legend=True)


def plots_beginning_correction(input_suites):
    resolution = np.loadtxt('Resolution', dtype=np.str)
    resolutions = [np.float(resolution[i, 1]) for i in range(resolution.shape[0])]
    hist_own_plot(resolutions, x_label=r'Resolution in $\mathring{A}$', y_label='Number of pdb files',
                  filename='histogram_resolution_all_pdb_files', bins=10, density=False, y_ticks=[0, 5, 10, 15, 20])
    complete_suites = [suite for suite in input_suites if suite.complete_suite]
    clash_suites = [suite for suite in complete_suites if len(suite.bb_bb_one_suite) > 0]
    clash_procrustes_backbone = [suite.procrustes_complete_suite_vector for suite in clash_suites]
    clash_procrustes_mesoscopic = [suite.procrustes_complete_mesoscopic_vector for suite in clash_suites]
    build_fancy_chain_plot(np.array(clash_procrustes_backbone), plot_backbone_atoms=True,
                           specific_legend_strings=[r'Clash suites'], specific_legend_colors=['black'],
                           filename='clash_suites', create_label=False, atom_size=0.1, alpha_atoms=0.6)
    build_fancy_chain_plot(np.array(clash_procrustes_mesoscopic), plot_atoms=True,
                           specific_legend_strings=[r'Clash mesoscopic shapes'], specific_legend_colors=['black'],
                           filename='clash_mesos', create_label=False, atom_size=0.5, alpha_atoms=1, lw_=0.1)
    for suite in complete_suites:
        for i in range(resolution.shape[0]):
            if suite._filename == resolution[i, 0]:
                suite.resolution = np.float(resolution[i, 1])
    clashscore_list = np.loadtxt('clashscore_list', dtype=np.str)
    raw_clashscore_list = clashscore_list[1:, 1]
    erraser_clashscore_list = clashscore_list[1:, 2]
    plot.scatter([np.float(i) for i in raw_clashscore_list], [np.float(i) for i in erraser_clashscore_list],
                 color='darkgreen')
    plot.plot(np.arange(0, 70), color='grey', lw=0.5)
    plot.xlabel('raw clashscore', fontdict={'size': 15})
    plot.ylabel('ERRASER clashscore', fontdict={'size': 15})
    plot.savefig('./out/clashscore')
    plot.close()
    clashscore_list_original = np.loadtxt('clashscore_list_erraser_paper', dtype=np.str)
    raw_clashscore_list_original = clashscore_list_original[1:, 1]
    erraser_clashscore_list_original = clashscore_list_original[1:, 2]
    plot.scatter([np.float(i) for i in raw_clashscore_list], [np.float(i) for i in erraser_clashscore_list],
                 label='our pdb files', c='blue')
    plot.scatter([np.float(i) for i in raw_clashscore_list_original],
                 [np.float(i) for i in erraser_clashscore_list_original],
                 label='Correcting pervasive errors in rna crystallography through enumerative structure prediction',
                 c='orange')
    plot.plot(np.arange(0, 70))
    plot.xlabel('raw clashscore')
    plot.ylabel('erraser clashscore')
    plot.savefig('./out/clashscore_with_erraser')
    plot.close()
    return complete_suites


def plot_correction_results(all_distances, all_distances_resolution, all_nr_in_cluster, all_single_distances,
                            complete_suites, erraser_corrections, folder):
    build_fancy_chain_plot(np.array(erraser_corrections), plot_backbone_atoms=True,
                           specific_legend_strings=[r'Clash suites'], specific_legend_colors=['black'],
                           filename='clash_suites_our_algorithm', create_label=False, atom_size=0.1, alpha_atoms=0.7,
                           without_legend=True)

    hist_own_plot(all_distances, x_label='Angstrom', y_label='Number of clash suites', filename=folder + 'hist',
                  bins=20, density=False, y_ticks=None)

    hist_own_plot(all_distances_resolution, x_label=r'$\widetilde{d}_{\mathfrak{c}}$', y_label='Number of clash suites',
                  filename=folder + 'hist_dist_resolution',
                  bins=20, density=False, y_ticks=None)

    hist_own_plot(all_nr_in_cluster, x_label=r'Elements in $U_{\mathfrak{c}} \cap C_{j_{\mathfrak{c}}}$',
                  y_label='Number of clash suites', filename=folder + 'hist_number',
                  bins=20, density=False, y_ticks=[0, 5, 10, 15])
    dihedral_angle_start = [suite.dihedral_angles for suite in complete_suites if
                            len(suite.bb_bb_one_suite) > 0 and len(suite.erraser) > 0]
    dihedral_angle_start_all = [suite.dihedral_angles for suite in complete_suites if len(suite.bb_bb_one_suite) > 0]
    dihedral_angle_erraser = [suite.erraser['dihedral_angles'] for suite in complete_suites if
                              len(suite.bb_bb_one_suite) > 0 and len(suite.erraser) > 0]
    dihedral_angle_clean = [suite.clean['torus mean'] for suite in complete_suites if
                            len(suite.bb_bb_one_suite) > 0 and len(suite.erraser) > 0]
    dihedral_angle_clean_second = [suite.clean['torus mean second'] for suite in complete_suites if
                                   len(suite.bb_bb_one_suite) > 0 and len(suite.erraser) > 0]
    dihedral_angle_clean_all = [suite.clean['torus mean'] for suite in complete_suites if
                                len(suite.bb_bb_one_suite) > 0]

    dihedral_angle_clean_all_second = [suite.clean['torus mean second'] for suite in complete_suites if
                                       len(suite.bb_bb_one_suite) > 0]
    plot.scatter(all_nr_in_cluster, all_distances_resolution, color='darkgreen')
    plot.ylabel(r'$\widetilde{d}_{\mathfrak{c}}$', fontdict={'size': 15})
    plot.xlabel(r'Elements in $U_{\mathfrak{c}} \cap C_{j_{\mathfrak{c}}}$', fontdict={'size': 15})
    plot.tight_layout()
    plot.savefig(folder + 'scatter')
    plot.close()
    for j in range(6):
        plot.hist([all_single_distances[i][j] for i in range(len(all_single_distances))], bins=20)
        plot.xlabel('Angstrom')
        plot.savefig(folder + 'hist_' + str(j))
        plot.close()
    for j in range(6):
        plot.hist([np.sort(all_single_distances[i])[j] for i in range(len(all_single_distances))], bins=20)
        plot.xlabel('Angstrom')
        plot.savefig(folder + 'hist_sort' + str(j))
        plot.close()
    scatter_plots_two(np.array(dihedral_angle_start), np.array(dihedral_angle_erraser), np.array(dihedral_angle_clean),
                      filename=folder + 'scatter_angles',
                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$',
                                    r'$\delta_{2}$'], plot_line=True, s=80, without_axis=False, all_titles=True,
                      legend_=True)
    scatter_plots_two(np.array(dihedral_angle_start), np.array(dihedral_angle_erraser), np.array(dihedral_angle_clean),
                      filename=folder + 'scatter_angles_all',
                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$',
                                    r'$\delta_{2}$'], plot_line=True, s=80, all_titles=True, without_axis=False,
                      legend_=True)

    scatter_plots_two(np.array(dihedral_angle_start), input_data3=np.array(dihedral_angle_clean),
                      input_data4=dihedral_angle_clean_second,
                      filename=folder + 'CLEAN_both',
                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$',
                                    r'$\delta_{2}$'], plot_line=True, s=80, without_axis=False, legend_=True)

    scatter_plots_two(np.array(dihedral_angle_start_all), input_data3=np.array(dihedral_angle_clean_all),
                      input_data4=dihedral_angle_clean_all_second,
                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$',
                                    r'$\delta_{2}$'], plot_line=True, s=40, all_titles=False, without_axis=False,
                      legend_=True,
                      filename=folder + 'CLEAN_both_all')
    scatter_plots_two(np.array(dihedral_angle_start), np.array(dihedral_angle_erraser),
                      filename=folder + 'scatter_angles_erraser',
                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$',
                                    r'$\delta_{2}$'], plot_line=True, s=80, without_axis=False, legend_=True)
    scatter_plots_two(np.array(dihedral_angle_start), np.array(dihedral_angle_erraser),
                      filename=folder + 'scatter_angles_all_erraser',
                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$',
                                    r'$\delta_{2}$'], plot_line=True, s=80, all_titles=True, without_axis=False)
    titles = [r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta_{2}$']
    for i in range(len(titles)):
        for j in range(i):
            scatter_plots_two(np.array(dihedral_angle_start)[:, [j, i]], np.array(dihedral_angle_erraser)[:, [j, i]],
                              filename=folder + 'scatter_angles_erraser' + titles[j] + titles[i],
                              suite_titles=[titles[j], titles[i]],
                              plot_line=True, s=80, all_titles=True, without_axis=False)
            scatter_plots_two(np.array(dihedral_angle_start)[:, [j, i]],
                              input_data3=np.array(dihedral_angle_clean)[:, [j, i]],
                              filename=folder + 'scatter_angles_clean' + titles[j] + titles[i],
                              suite_titles=[titles[j], titles[i]], plot_line=True, s=80, without_axis=False,
                              all_titles=True)
            input_data_4 = [[dihedral_angle_clean_second[m][j]] + [dihedral_angle_clean_second[m][i]] if
                            dihedral_angle_clean_second[m] is not None else None for m in
                            range(len(dihedral_angle_clean_second))]
            scatter_plots_two(np.array(dihedral_angle_start)[:, [j, i]],
                              input_data3=np.array(dihedral_angle_clean)[:, [j, i]],
                              input_data4=input_data_4,
                              filename=folder + 'scatter_angles_both_clean' + titles[j] + titles[i],
                              suite_titles=[titles[j], titles[i]], plot_line=True, s=80, without_axis=False,
                              all_titles=True)


def plot_4_ring_correction(best_string, clash_suite, complete_suites, meso_shift_shape, neighbors, neighbors_cluster,
                           neighbour_mean, number_neighbors, procrustes_data, procrustes_data_backbone,
                           corrected_mesoscopic_coordinates, mesoscopic_clash_shape_rotated,
                           procrustes_data_small_rotated):
    specific_legend_strings = [r'Mesoscopics from $C_{j_{\mathfrak{c}}} \cap U_{\mathfrak{c}}$',
                               r'Mesoscopic clash shape',
                               r'Mesoscopic cluster mean $\mu_{\mathfrak{c}}$']

    build_fancy_chain_plot(
        np.vstack((procrustes_data_small_rotated[neighbors],
                   procrustes_data_small_rotated[clash_suite].reshape(1, 6, 3),
                   neighbour_mean.reshape(1, 6, 3))),
        colors=[COLORS[0]] * len(neighbors) + [COLORS[1]] + [COLORS[2]],
        create_label=False,
        lw_vec=[0.1] * len(neighbors) + [1.5, 1.5],
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[0], COLORS[1], COLORS[2]],
        filename=best_string + complete_suites[clash_suite].name + 'start_meso', without_legend=True)

    specific_legend_strings = [r'Suites in $C_{i_{\mathfrak{c}}} \cap U_{\mathfrak{c}}$',
                               r'Suites in $U_{\mathfrak{c}} ~ \backslash ~ C_{i_{\mathfrak{c}}}$',
                               r'Clash suite $\mathfrak{c}$',
                               r'Suite cluster mean $\tau_{\mathfrak{c}}$']

    build_fancy_chain_plot(
        np.vstack((procrustes_data_backbone[neighbors],
                   procrustes_data_backbone[clash_suite].reshape(1, 10, 3),
                   np.mean(procrustes_data_backbone[neighbors], axis=0).reshape(1, 10, 3))),
        colors=[COLORS[0]] * len(neighbors) + [COLORS[1]] + [COLORS[2]],
        create_label=False,
        lw_vec=[0.1] * len(neighbors) + [1] + [1],
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[0], COLORS[4], COLORS[1], COLORS[2]],
        filename=best_string + complete_suites[clash_suite].name + 'start_suite', without_legend=True)

    specific_legend_strings = [r'The suites from $\mathfrak{H}^{(\mathfrak{c})}$ that are not in the main cluster',
                               r'The main suite cluster in $\mathfrak{H}^{(\mathfrak{c})}$ with ' + str(
                                   len(neighbors_cluster)) + ' elements',
                               'The clash suite']
    build_fancy_chain_plot(
        np.vstack((procrustes_data_backbone[neighbors_cluster],
                   procrustes_data_backbone[[i for i in neighbors if i not in neighbors_cluster]],
                   procrustes_data_backbone[clash_suite].reshape(1, 10, 3))),
        colors=[COLORS[0]] * len(neighbors_cluster) + [COLORS[4]] * (number_neighbors - len(neighbors_cluster)) + [
            COLORS[1]],
        create_label=False,
        lw_vec=[0.3] * number_neighbors + [2],
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[0], COLORS[4], COLORS[1]],
        filename=best_string + complete_suites[clash_suite].name + 'suite_neighbors', without_legend=True)

    build_fancy_chain_plot(
        np.vstack((procrustes_data[[i for i in neighbors if i not in neighbors_cluster]],
                   procrustes_data[neighbors_cluster],
                   procrustes_data[clash_suite].reshape(1, 6, 3))),
        colors=[COLORS[4]] * (number_neighbors - len(neighbors_cluster)) + [COLORS[0]] * len(neighbors_cluster) + [
            COLORS[1]],
        create_label=False,
        lw_vec=[0.4] * number_neighbors + [2],
        alpha_line_vec=[1] * (number_neighbors - len(neighbors_cluster)) + [0.7] * len(neighbors_cluster) + [1],
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[0], COLORS[4], COLORS[1]],
        filename=best_string + complete_suites[clash_suite].name + 'mesoscopic_neighbors', without_legend=True)

    if len(complete_suites[clash_suite].erraser) > 0 and complete_suites[clash_suite].erraser[
        'rotated_backbone'] is not None:
        erraser_suite = complete_suites[clash_suite].erraser['rotated_backbone']
        build_fancy_chain_plot(np.vstack((procrustes_data_backbone[neighbors_cluster],
                                          procrustes_data_backbone[clash_suite].reshape(1, 10, 3),
                                          erraser_suite.reshape(1, 10, 3))),
                               colors=[COLORS[0]] * len(neighbors_cluster) + [COLORS[1]] + [COLORS[3]],
                               lw_vec=[0.1] * len(neighbors_cluster) + [2] + [2],
                               create_label=False,
                               specific_legend_strings=[r'Suites in $U_{\mathfrak{c}} \cap C_{i_{\mathfrak{c}}}$'] + [
                                   r'Clash suite $\mathfrak{c}$'] + ['ERRASER suite'],
                               specific_legend_colors=[COLORS[0]] + [COLORS[1]] + [COLORS[3]],
                               filename=best_string + complete_suites[clash_suite].name + 'erraser',
                               without_legend=True)

    specific_legend_strings = ['Corrected mesoscopic strand',
                               'Mesoscopic clash strand']
    build_fancy_chain_plot(
        np.vstack((corrected_mesoscopic_coordinates.reshape(1, 6, 3),
                   complete_suites[clash_suite].mesoscopic_sugar_rings.reshape(1, 6, 3))),
        colors=[COLORS[2], COLORS[1]],
        create_label=False,
        lw=2,
        specific_legend_strings=specific_legend_strings,
        specific_legend_colors=[COLORS[2], COLORS[1]],
        filename=best_string + complete_suites[clash_suite].name + 'original_meso', not_scale=True, without_legend=True)

    atom_color_matrix = [[COLORS[4]] * (number_neighbors - len(neighbors_cluster)) + [COLORS[0]] * len(
        neighbors_cluster) + [COLORS[1]]] * 6
    atom_color_matrix = np.array(atom_color_matrix).T
    atom_color_matrix[atom_color_matrix.shape[0] - 1, 1:5] = 'teal'
    # atom_color_matrix[atom_color_matrix.shape[0]-1, 1:5]
    specific_legend_strings = [r'Mesoscopic shapes of $U_{\mathfrak{c}}$',
                               r'Mesoscopic clash shape $m_\mathfrak{c}$']
    build_fancy_chain_plot(np.vstack((procrustes_data[[i for i in neighbors if i not in neighbors_cluster]],
                                      procrustes_data[neighbors_cluster],
                                      procrustes_data_small_rotated[clash_suite].reshape(1, 6, 3))),
                           colors=[COLORS[4]] * (number_neighbors - len(neighbors_cluster)) + [COLORS[0]] * len(
                               neighbors_cluster) + [COLORS[1]],
                           lw_vec=[0.4] * number_neighbors + [2],
                           create_label=False,
                           alpha_line_vec=[1] * (number_neighbors - len(neighbors_cluster)) + [0.7] * len(
                               neighbors_cluster) + [1],
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0], COLORS[1]],
                           filename=best_string + complete_suites[clash_suite].name + '_plot_middle',
                           atom_color_matrix=atom_color_matrix, plot_atoms=True,
                           atom_size_vector=[0.1] * len(neighbors) + [20],
                           atom_alpha_vector=[0.1] * len(neighbors) + [1], without_legend=True)

    specific_legend_strings = [r'Suites $U_{\mathfrak{c}}$',
                               r'Clash suite $\mathfrak{c}$']
    build_fancy_chain_plot(np.vstack((procrustes_data_backbone[neighbors],
                                      procrustes_data_backbone[clash_suite].reshape(1, 10, 3))),
                           colors=[COLORS[0]] * len(neighbors) + [COLORS[1]],
                           lw_vec=[0.1] * len(neighbors) + [2],
                           create_label=False,
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0], COLORS[1]],
                           filename=best_string + complete_suites[clash_suite].name + '_suite_all', without_legend=True)
    print(best_string)


def plot_all_clash_suites_before_and_after_ERRASER_correction(erraser_clash_backbone, start_clash_backbone, string):
    plot_functions.build_fancy_chain_plot(erraser_clash_backbone, filename=string + 'all_erraser_clash_suites',
                                          plot_backbone_atoms=True, create_label=False,
                                          specific_legend_strings=['Corresponding suites in $\mathfrak{E}$'],
                                          specific_legend_colors=['black'], alpha_backbone_atoms=0.7,
                                          without_legend=True)
    plot_functions.build_fancy_chain_plot(start_clash_backbone, filename=string + 'all_start_clash_suites',
                                          plot_backbone_atoms=True, create_label=False,
                                          specific_legend_strings=['Clash suites in $\mathfrak{R}$'],
                                          specific_legend_colors=['black'], alpha_backbone_atoms=0.7,
                                          without_legend=True)


def plot_all_single_clash_suites_before_and_after_correction_by_ERRASER(erraser_clash_backbone,
                                                                        erraser_clash_backbone_original,
                                                                        erraser_clash_suites, start_clash_backbone,
                                                                        start_clash_backbone_original, string,
                                                                        string_single):
    if not os.path.exists(string_single):
        os.makedirs(string_single)
    for i in range(len(erraser_clash_suites)):
        if 'erraser' in erraser_clash_suites[i].clashscore.keys():
            f_name = string_single + erraser_clash_suites[i].name + 'clash_after_correction'

        else:
            f_name = string_single + erraser_clash_suites[i].name + 'clash_free'
        plot_functions.build_fancy_chain_plot(np.vstack((erraser_clash_backbone[i].reshape(1, 10, 3),
                                                         start_clash_backbone[i].reshape(1, 10, 3))),
                                              filename=f_name, colors=[COLORS[3], COLORS[0]],
                                              lw_vec=[2, 2],
                                              specific_legend_strings=['Erraser correction', 'Clash suite'],
                                              specific_legend_colors=[COLORS[3], COLORS[0]], create_label=False,
                                              plot_backbone_atoms=True, alpha_backbone_atoms=1, atom_size=4)

        # plot_functions.build_fancy_chain_plot(np.vstack((erraser_clash_backbone_original[i].reshape(1, 10, 3),
        #                                                  start_clash_backbone_original[i].reshape(1, 10, 3))),
        #                                       filename=f_name + 'original', colors=[COLORS[3], COLORS[0]],
        #                                       lw_vec=[2, 2],
        #                                       specific_legend_strings=['Erraser correction', 'Clash suite'],
        #                                       specific_legend_colors=[COLORS[3], COLORS[0]], create_label=False,
        #                                       plot_backbone_atoms=True, alpha_backbone_atoms=1, atom_size=4)

        with open(f_name + '.txt', 'w') as file:
            file.write(json.dumps(erraser_clash_suites[i].clashscore))
    string_real = string + 'single_real/'
    if not os.path.exists(string_real):
        os.makedirs(string_real)
    # for i in range(len(erraser_clash_suites)):
    #     plot_functions.build_fancy_chain_plot(np.vstack((erraser_clash_suites[i]._backbone_atoms.reshape(1, 10, 3),
    #                                                      erraser_clash_suites[i].erraser['backbone_atoms'].reshape(1,
    #                                                                                                                10,
    #                                                                                                                3))),
    #                                           filename=string_real + str(i), colors=[COLORS[3], COLORS[0]],
    #                                           lw_vec=[1, 1],
    #                                           specific_legend_strings=['Erraser correction', 'Clash suite'],
    #                                           specific_legend_colors=[COLORS[3], COLORS[0]], create_label=False)


def plot_ERRASER_clustering_results(both_data_sets, cluster_list_erraser, erraser_clash_in_cluster,
                                    erraser_clash_outlier, erraser_data, name, start_data):
    for i in range(len(cluster_list_erraser)):
        if len(erraser_clash_in_cluster[i]) > 0:
            plot_functions.build_fancy_chain_plot(np.vstack((both_data_sets[cluster_list_erraser[i]],
                                                             both_data_sets[erraser_clash_in_cluster[i]],
                                                             start_data[erraser_clash_in_cluster[i]])),
                                                  colors=[COLORS[0]] * len(cluster_list_erraser[i]) + [COLORS[3]] * len(
                                                      erraser_clash_in_cluster[i]) + [COLORS[1]] * len(
                                                      erraser_clash_in_cluster[i]),

                                                  filename=name + 'in_cluster_nr' + str(i + 1) + '_erraser',
                                                  lw_vec=[0.1] * len(cluster_list_erraser[i]) + [1] * len(
                                                      erraser_clash_in_cluster[i]) + [1] * len(
                                                      erraser_clash_in_cluster[i]),
                                                  specific_legend_strings=['Cluster number ' + str(i), 'ERRASER suites',
                                                                           'Clash suites'],
                                                  specific_legend_colors=[COLORS[0], COLORS[3], COLORS[1]],
                                                  create_label=False)
    plot_functions.build_fancy_chain_plot(erraser_data[erraser_clash_outlier],
                                          filename=name + 'clash_suites_outlier_erraser')


def plot_all_classes(cluster_list_backbone, dihedral_angles, backbone_validation_list_2, folder):
    clusters_dihedral_angles = np.array(dihedral_angles)[cluster_list_backbone[0]]

    for i in range(1, len(cluster_list_backbone)):
        clusters_dihedral_angles = np.vstack(
            (clusters_dihedral_angles, np.array(dihedral_angles)[cluster_list_backbone[i]]))

    torus_mean_list = [torus_mean_and_var(np.array(dihedral_angles)[cluster_list_backbone[i]])[0] for i in
                       range(len(cluster_list_backbone))]

    original_richardson_list = [
        [81, -148, -71, -65, 174, 54, 81],
        [84, -142, -68, -68, -138, 58, 86],
        [86, -115, -92, -56, 138, 62, 79],
        [82, -169, -95, -64, -178, 51, 82],
        [83, -143, -138, -57, 161, 49, 82],
        [85, -144, 173, -71, 164, 46, 85],
        [83, -150, 121, -71, 157, 49, 81],
        [81, -141, -69, 167, 160, 51, 85],
        [84, -121, -103, 70, 170, 53, 85],
        [85, -116, -156, 66, -179, 55, 86],
        [80, -158, 63, 68, 143, 50, 83],
        [81, -159, -79, -111, 83, 168, 86],
        [80, -163, -69, 153, -166, 179, 84],
        [81, -157, -66, 172, 139, 176, 84],
        [87, -136, 80, 67, 109, 176, 84],
        [84, -145, -71, -60, 177, 58, 145],
        [83, -140, -71, -63, -138, 54, 144],
        [85, -134, 168, -67, 178, 49, 148],
        [83, -154, -82, -164, 162, 51, 145],
        [83, -154, 53, 164, 148, 50, 148],
        [84, -123, -140, 68, -160, 54, 146],
        [81, -161, -71, 180, -165, 178, 147],
        [82, -155, 69, 63, 115, 176, 146],
        [84, -143, -73, -63, -135, -66, 151],
        [85, -127, -112, 63, -178, -64, 150],
        [145, -100, -71, -72, -167, 53, 84],
        [146, -100, 170, -62, 170, 51, 84],
        [149, -137, 139, -75, 158, 48, 84],
        [148, -168, -146, -71, 151, 42, 85],
        [148, -103, 165, -155, 165, 49, 83],
        [145, -97, 80, -156, -170, 58, 85],
        [149, -89, -119, 62, 176, 54, 87],
        [150, -110, -172, 80, -162, 61, 89],
        [147, -119, 89, 59, 161, 52, 83],
        [148, -99, -70, -64, 177, 176, 87],
        [144, -133, -156, 74, -143, -166, 81],
        [149, -85, 100, 81, -112, -178, 83],
        [150, -92, 85, 64, -169, 177, 86],
        [142, -116, 66, 72, 122, -178, 84],
        [146, -101, -69, -68, -150, 54, 148],
        [145, -115, 163, -66, 172, 46, 146],
        [148, -112, 112, -85, 165, 57, 146],
        [150, -100, -146, 72, -152, 57, 148],
        [146, -102, 90, 68, 173, 56, 148],
        [150, -112, 170, -82, 84, 176, 148],
        [147, -104, -64, -73, -165, -66, 150],
    ]
    number_elements_richardson = [4637, 15, 14, 33, 36, 25, 19, 78, 16, 20, 14, 42, 275, 20, 12, 168, 52, 14, 12, 42,
                                  27, 7, 6, 13,
                                  16, 126, 12, 29, 16, 18, 16, 24, 9, 18, 17, 9, 6, 18, 9, 40, 27, 14, 13, 39, 8, 12]
    original_richardson_list_0_360 = []
    for i in range(len(original_richardson_list)):
        torus_mean = original_richardson_list[i].copy()
        for j in range(len(torus_mean)):
            if torus_mean[j] < 0:
                torus_mean[j] = 360 + torus_mean[j]
        original_richardson_list_0_360.append(np.array(torus_mean))

    torus_mean_list_richardson = []
    for i in range(len(torus_mean_list)):
        torus_mean = torus_mean_list[i].copy()
        for j in range(len(torus_mean)):
            if torus_mean[j] > 180:
                torus_mean[j] = -(360 - torus_mean[j])
        torus_mean_list_richardson.append(torus_mean)

    plot_functions.scatter_plots(np.vstack((np.array(dihedral_angles)[cluster_list_backbone[5]],
                                            original_richardson_list_0_360[4],
                                            original_richardson_list_0_360[5],
                                            original_richardson_list_0_360[6])),
                                 filename=folder + 'compare_richardson_cluster_nr_4_5_6_mintage_cluster_5',
                                 number_of_elements=[len(cluster_list_backbone[5]), 1, 1, 0, 1],
                                 alpha_first=0.7, s=500,
                                 suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                                               r'$\gamma$', r'$\delta_{2}$'], legend=False, dummy_legend=True,
                                 axis_min=None)

    argmin_list = []
    argmin_list2 = []
    for i in range(len(torus_mean_list)):
        torus_mean = torus_mean_list[i]
        diffs = []
        for j in range(len(original_richardson_list_0_360)):
            diff = 0
            for k in range(len(original_richardson_list_0_360[j])):
                diff_one_dim = original_richardson_list_0_360[j][k] - torus_mean[k]
                diff_one_dim = np.min([(360 - diff_one_dim) ** 2, diff_one_dim ** 2])
                diff = diff + diff_one_dim
            diffs.append(np.sqrt(diff))
        argmin_ = np.argmin(diffs)
        argmin_list.append(argmin_)
        min_ = diffs[argmin_]
        diffs[argmin_] = np.inf
        argmin_2 = np.argmin(diffs)
        argmin_list2.append(argmin_2)
        min_2 = diffs[argmin_2]
        print(
            "cluster " + str(i + 1) + "\n" + str(torus_mean_list[i]) + "\n" + str(torus_mean_list_richardson[i]) + "\n")
        print("to cluster " + str(argmin_) + "\n" + str(original_richardson_list_0_360[argmin_]) + "\n" + str(
            original_richardson_list[argmin_]) + "\n",
              "argmin=" + str(argmin_) + ", Min=" + str(min_) + "\n" + "argmin=" + str(argmin_2) + ", Min=" + str(
                  min_2) + "\n")
        # plot_functions.scatter_plots(np.vstack((np.array(dihedral_angles)[cluster_list_backbone[i]], original_richardson_list_0_360[argmin_],original_richardson_list_0_360[argmin_2])),
        #                              filename=folder + 'compare_richardson_nr_' + str(i+1) + "_richardson" + str(argmin_)+ "_richardson" + str(argmin_2),
        #                              number_of_elements=[len(cluster_list_backbone[i]), 1, 1],
        #                              alpha_first=1, s=20,
        #                              suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
        #                                            r'$\gamma$', r'$\delta_{2}$'], legend=False)

    print([number_elements_richardson[number] for number in range(len(number_elements_richardson)) if
           number not in argmin_list])

    argmin_list = []
    for i in range(len(original_richardson_list_0_360)):
        torus_mean = original_richardson_list_0_360[i]
        diffs = []
        for j in range(len(torus_mean_list)):
            diff = 0
            for k in range(len(torus_mean)):
                diff_one_dim = torus_mean_list[j][k] - torus_mean[k]
                diff_one_dim = np.min([(360 - diff_one_dim) ** 2, diff_one_dim ** 2])
                diff = diff + diff_one_dim
            diffs.append(np.sqrt(diff))
        argmin_ = np.argmin(diffs)
        argmin_list.append(argmin_)
        min_ = diffs[argmin_]
        plot_functions.scatter_plots(
            np.vstack((np.array(dihedral_angles)[cluster_list_backbone[argmin_]], original_richardson_list_0_360[i])),
            filename=folder + 'compare_richardson_cluster_nr_' + str(i) + "_mintage_cluster" + str(argmin_),
            number_of_elements=[len(cluster_list_backbone[argmin_]), 1],
            alpha_first=1, s=20,
            suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                          r'$\gamma$', r'$\delta_{2}$'], legend=False)

    number_elements = [len(cluster) for cluster in cluster_list_backbone]
    plot_functions.scatter_plots(clusters_dihedral_angles, filename=folder + 'all_classes_1',
                                 number_of_elements=number_elements,
                                 alpha_first=1, s=20,
                                 suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                                               r'$\gamma$', r'$\delta_{2}$'])
    dihedral_angles_clean = np.array(dihedral_angles)[
        [i for i in range(len(dihedral_angles)) if i not in backbone_validation_list_2]]
    plot_functions.scatter_plots(dihedral_angles_clean, filename=folder + 'all_data_1', alpha_first=0.5, s=20,
                                 suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                                               r'$\gamma$', r'$\delta_{2}$'])
    titles = [r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta_{2}$']
    for i in range(len(titles)):
        for j in range(i):
            plot_functions.scatter_plots(clusters_dihedral_angles[:, [j, i]],
                                         filename=folder + 'all_classes' + titles[j] + titles[i],
                                         number_of_elements=number_elements, alpha_first=0.5, s=20,
                                         suite_titles=[titles[j], titles[i]], all_titles=True, fontsize=20,
                                         legend=False)
            plot_functions.scatter_plots(dihedral_angles_clean[:, [j, i]],
                                         filename=folder + 'all_data' + titles[j] + titles[i],
                                         alpha_first=0.5, s=20,
                                         suite_titles=[titles[j], titles[i]], all_titles=True, fontsize=20,
                                         legend=False)
