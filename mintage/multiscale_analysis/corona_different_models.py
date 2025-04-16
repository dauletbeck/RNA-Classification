import os

import numpy as np

import clash_correction
from clash_correction import create_cluster_list
from utils.constants import COLORS
from utils.data_functions import rotate_y_optimal_to_x, procrustes_algorithm_short
from utils.help_plot_functions import corona_plot_A33_A34, create_corona_plots, single_corona_plots
from parsing.parse_functions import parse_pdb_files, parse_clash_files
from utils.plot_functions import build_fancy_chain_plot


def working_with_different_models(input_suites):

    string_folder = './out/SARS-CoV-2/rna_different_models/saved_suite_list/'
    input_pdb_folder = './SARS-CoV-2/rna_different_models/reduce'
    input_validation_folder = './SARS-CoV-2/rna_different_models/validation_reports'
    new_suites = parse_pdb_files(string_folder, input_pdb_folder=input_pdb_folder)
    suite_names = [suite.name for suite in new_suites]
    model_number = [sum([suite_names[i] == suite_names[k] for i in range(k)]) + 1 for k in range(len(suite_names))]
    for i in range(len(new_suites)):
        new_suites[i].model_number = model_number[i]

    new_suites = parse_clash_files(input_suites=new_suites, input_string_folder=string_folder,
                                   folder_validation_files=input_validation_folder, model_number=True)

    procrustes_data_backbone_new, procrustes_data_new, procrustes_data_backbone_old, procrustes_data_old = mean_align_to_other_dataset(input_suites, new_suites)

    complete_suites_old = [suite for suite in input_suites if suite.complete_suite]
    complete_suites_new = [suite for suite in new_suites if suite.complete_suite]
    string_ = 'suite_True'
    best_string = './out/Covid_correction/'
    if not os.path.exists(best_string):
        os.makedirs(best_string)
    cluster_list_backbone = create_cluster_list(complete_suites_old, string_)

    backbone_validation_list_2 = [i for i in range(len(complete_suites_old)) if len(complete_suites_old[i].bb_bb_one_suite) > 0 or len(complete_suites_old[i].bb_bb_neighbour_clashes) > 0]
    procrustes_data_backbone_benchmark = np.array([suite.procrustes_complete_suite_vector for suite in input_suites if suite.complete_suite])
    number_neighbors = 50
    suite_names_unique = [suite_names[k] for k in [i for i in range(len(model_number)) if model_number[i]==1]]
    #for name in suite_names_unique:
    for name in ['6xrz_A28_A29', '6xrz_A33_A34']:
        suite_with_same_name = [i for i in range(len(complete_suites_new)) if complete_suites_new[i].name == name]
        mesoscopic_clash_shape_list = []
        mesoscopic_repair_shape_list = []
        cluster_numbers = []
        suite_backbone_list = []
        counter = 1
        number_in_cluster_list = []
        suite_correction_list = []
        for suite_index in suite_with_same_name:

            data = procrustes_data_old.copy()
            x = procrustes_data_new[suite_index].copy()
            # Step 1: rotate all data optimal to the clash mesoscopic:
            for i in range(len(data)):
                data[i] = rotate_y_optimal_to_x(x=x, y=data[i])
            # Step 2: Find the most similar mesoscopic elements:
            neighbors_dummy = np.array([np.linalg.norm(procrustes_data_new[suite_index] - data[element]) for element in
                                        range(data.shape[0])]).argsort()[1:]
            neighbors = [neighbors_dummy[i] for i in range(len(neighbors_dummy)) if
                         neighbors_dummy[i] not in backbone_validation_list_2][0:number_neighbors]

            nr_in_cluster = [len(set(cluster) & set(neighbors)) for cluster in cluster_list_backbone]
            number_in_cluster_list.append(np.max(nr_in_cluster))
            # Step 3: Find the main cluster.
            cluster_nr = np.argmax(nr_in_cluster)
            cluster_numbers.append(cluster_nr)
            neighbors_cluster = list(set(cluster_list_backbone[cluster_nr]) & set(neighbors))
            procrustes_data_small_rotated = procrustes_data_old.copy()
            procrustes_data_small_rotated[neighbors_cluster] = procrustes_algorithm_short(procrustes_data_old[neighbors_cluster])[0]
            neighbour_mean = np.mean(procrustes_data_small_rotated[neighbors_cluster], axis=0)
            neighbour_mean = neighbour_mean - np.mean(neighbour_mean, axis=0)
            # Step 4: Rotate the clash shape optimal to the Procrustes mean
            mesoscopic_clash_shape_copy = procrustes_data_new[suite_index].copy()
            x = mesoscopic_clash_shape_copy
            y = neighbour_mean
            # Step 5: Calculate the corrected mesoscopic shape.
            distance_to_z, x, y, y_bar = clash_correction.inner_orthogonal_projection(backbone_mean=np.mean(procrustes_data_backbone_old[neighbors_cluster], axis=0), x=x, z=y)
            suite_backbone = procrustes_data_backbone_new[suite_index]
            suite_backbone_list.append(suite_backbone)
            suite_correction_list.append(np.mean(procrustes_data_backbone_old[neighbors_cluster], axis=0))

            # Step 6: Plot the results for every single model.
            single_corona_plots(best_string, complete_suites_new, counter, mesoscopic_clash_shape_copy,
                                neighbors_cluster, procrustes_data_backbone_benchmark,
                                procrustes_data_small_rotated, suite_backbone, suite_index, y_bar)
            counter = counter + 1
            mesoscopic_clash_shape_list.append(mesoscopic_clash_shape_copy)
            mesoscopic_repair_shape_list.append(y_bar)
        if len(mesoscopic_repair_shape_list) > 0:
            all_corona_plots_of_the_different_models(best_string, cluster_list_backbone, cluster_numbers,
                                                     complete_suites_new, mesoscopic_clash_shape_list,
                                                     mesoscopic_repair_shape_list, name, procrustes_data_backbone_old,
                                                     suite_backbone_list, suite_correction_list, suite_index,
                                                     suite_with_same_name)


def all_corona_plots_of_the_different_models(best_string, cluster_list_backbone, cluster_numbers, complete_suites_new,
                                             mesoscopic_clash_shape_list, mesoscopic_repair_shape_list, name,
                                             procrustes_data_backbone_old, suite_backbone_list, suite_correction_list,
                                             suite_index, suite_with_same_name):
    clash_and_repair_shapes = \
    procrustes_algorithm_short(np.array(mesoscopic_clash_shape_list + mesoscopic_repair_shape_list))[0]
    build_fancy_chain_plot(clash_and_repair_shapes,
                           colors=[COLORS[0]] * len(mesoscopic_clash_shape_list) + [COLORS[5]] * len(
                               mesoscopic_repair_shape_list),
                           create_label=False,
                           lw_vec=[0.5] * (len(mesoscopic_clash_shape_list) + len(mesoscopic_repair_shape_list)),
                           specific_legend_strings=['The different mesocopics of the different models',
                                                    'The corresponding correction mesocopics'],
                           specific_legend_colors=[COLORS[0], COLORS[5]],
                           filename=best_string + 'all_models' + complete_suites_new[suite_index].name + 'meso', without_legend=True)
    cluster_numbers_unique = list(set(cluster_numbers))
    mean_shape_list = []
    number_of_mean = []
    for cluster in cluster_numbers_unique:
        mean_shape = np.mean(procrustes_data_backbone_old[cluster_list_backbone[cluster]], axis=0)
        mean_shape_list.append(mean_shape)
        number_of_cluster = np.sum([cluster_numbers[i] == cluster for i in range(len(cluster_numbers))])
        number_of_mean.append(number_of_cluster)
    colors_validation = ['darkgreen', 'royalblue', 'grey', 'pink'] + COLORS
    specific_legend_strings = ['The suites of the different models'] + [
        'Cluster number ' + str(cluster_numbers_unique[i] + 1) + '; ' + str(number_of_mean[i]) + ' times' for i in
        range(len(number_of_mean))]
    build_fancy_chain_plot(np.vstack((np.array(suite_backbone_list), np.array(mean_shape_list))),
                           colors=[COLORS[0]] * len(cluster_numbers) + colors_validation[:len(cluster_numbers_unique)],
                           create_label=False,
                           lw_vec=[0.3] * len(cluster_numbers) + [1] * len(cluster_numbers_unique),
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + colors_validation[:len(cluster_numbers_unique)],
                           filename=best_string + 'all_models' + complete_suites_new[suite_index].name + 'suite', without_legend=True)
    colors_meso = []
    for cluster in cluster_numbers:
        index_ = cluster_numbers_unique.index(cluster)
        colors_meso.append(colors_validation[index_])
    specific_legend_strings = ['The mesoscopics of the different models'] + [
        'Cluster number ' + str(cluster_numbers_unique[i] + 1) + '; ' + str(number_of_mean[i]) + ' times' for i in
        range(len(number_of_mean))]
    build_fancy_chain_plot(clash_and_repair_shapes,
                           colors=[COLORS[0]] * len(cluster_numbers) + colors_meso,
                           create_label=False,
                           lw_vec=[0.3] * len(cluster_numbers) + [1] * len(cluster_numbers),
                           specific_legend_strings=specific_legend_strings,
                           specific_legend_colors=[COLORS[0]] + colors_validation[:len(cluster_numbers_unique)],
                           filename=best_string + 'all_models' + complete_suites_new[suite_index].name + 'meso_cluster', without_legend=True)
    clash_suite_list = [len(complete_suites_new[suite_index].bb_bb_one_suite) > 0 for suite_index in
                        suite_with_same_name]
    if True in clash_suite_list:
        if name == '6xrz_A33_A34':
            corona_plot_A33_A34(best_string, clash_and_repair_shapes, clash_suite_list, cluster_numbers,
                                cluster_numbers_unique, colors_meso, colors_validation, complete_suites_new,
                                mean_shape_list, suite_backbone_list, suite_correction_list, suite_index)
        create_corona_plots(best_string, clash_and_repair_shapes, clash_suite_list, cluster_numbers,
                            cluster_numbers_unique, colors_meso, colors_validation, complete_suites_new,
                            mean_shape_list, number_of_mean, suite_backbone_list, suite_index)


def mean_align_to_other_dataset(input_suites, new_suites):
    procrustes_data_old = np.array(
        [suite.procrustes_complete_mesoscopic_vector for suite in input_suites if suite.complete_suite])
    procrustes_mean = np.mean(procrustes_data_old, axis=0)
    procrustes_data_new = np.array([suite.mesoscopic_sugar_rings for suite in new_suites if suite.complete_suite])
    for i in range(len(procrustes_data_new)):
        procrustes_data_new[i] = procrustes_data_new[i] - np.mean(procrustes_data_new[i], axis=0)

        x = procrustes_mean
        y = procrustes_data_new[i].copy()
        procrustes_data_new[i] = rotate_y_optimal_to_x(x, y)
        #alpha = leastsq(data_functions.rotation_function, x0=np.array([0.0, 0.0, 0.0]),
        #                args=(procrustes_data_new[i], procrustes_mean))
        #x_rot = np.dot(procrustes_data_new[i], np.transpose(data_functions.rotation_matrix_x_axis(alpha[0][0])))
        #x_y_rot = np.dot(x_rot, np.transpose(data_functions.rotation_matrix_y_axis(alpha[0][1])))
        #procrustes_data_new[i] = np.dot(x_y_rot, np.transpose(data_functions.rotation_matrix_z_axis(alpha[0][2])))

    procrustes_data_backbone_old = np.array(
        [suite.procrustes_complete_suite_vector for suite in input_suites if suite.complete_suite])
    procrustes_backbone_mean = np.mean(procrustes_data_backbone_old, axis=0)
    procrustes_data_backbone_new = np.array([suite.backbone_atoms for suite in new_suites if suite.complete_suite])
    for i in range(len(procrustes_data_new)):
        procrustes_data_backbone_new[i] = procrustes_data_backbone_new[i] - np.mean(procrustes_data_backbone_new[i],
                                                                                    axis=0)
        x = procrustes_backbone_mean
        y = procrustes_data_backbone_new[i].copy()
        procrustes_data_backbone_new[i] = rotate_y_optimal_to_x(x, y)
        # alpha = leastsq(data_functions.rotation_function, x0=np.array([0.0, 0.0, 0.0]),
        #                 args=(procrustes_data_backbone_new[i], procrustes_backbone_mean))
        # x_rot = np.dot(procrustes_data_backbone_new[i],
        #                np.transpose(data_functions.rotation_matrix_x_axis(alpha[0][0])))
        # x_y_rot = np.dot(x_rot, np.transpose(data_functions.rotation_matrix_y_axis(alpha[0][1])))
        # procrustes_data_backbone_new[i] = np.dot(x_y_rot,
        #                                          np.transpose(data_functions.rotation_matrix_z_axis(alpha[0][2])))
    return procrustes_data_backbone_new, procrustes_data_new, procrustes_data_backbone_old, procrustes_data_old