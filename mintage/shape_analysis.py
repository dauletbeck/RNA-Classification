# Standard library imports
import collections
import csv
import os
import pickle

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib import pyplot as plot
import numpy as np
from scipy.cluster.hierarchy import average, fcluster, single, ward, centroid, weighted
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stat

# Local/custom imports
from utils.constants import COLORS, COLORS_SCATTER, mean_shapes_all, MARKERS
from utils.data_functions import (
    procrustes_algorithm_short,
    mean_on_sphere,
    rotate_y_optimal_to_x,
    rotation,
    rotation_matrix_x_axis
)
from utils.help_plot_functions import plot_clustering
import multiscale_analysis.multiscale_modes_linear
import multiscale_analysis.Multiscale_modes
from utils.plot_functions import build_fancy_chain_plot, hist_own_plot, scatter_plots
import pnds.PNDS_geometry
import pnds.PNDS_PNS
from pnds.PNDS_PNS import compare_likelihoods, torus_mean_and_var
from pnds import PNDS_RNA_clustering


def procrustes_on_suite_class(np_array, string, string_plot, shape=False, mean_shape=None, origin_index=None,
                              overwrite=False):
    """
    This method is a help function. It checks if the Procrustes algorithm has already been calculated and in this case
    loads the corresponding file. Otherwise the procrustes algorithm is called.
    :param np_array: The array of pre shapes.
    :param string: The name of the fild where the result of the Procrustes algorithm is stored
    :param string_plot: The name of the folder where the plots of the Procrustes algorithm is stored.
    :return:
    """

    print('procrustes_on_suite_class', string_plot)
    if os.path.isfile(string) and not overwrite:
        with open(string, 'rb') as f:
            procrustes_complete_suites = pickle.load(f)
    else:
        procrustes_data, shift_array, scale_array, rotation_matrices = procrustes_algorithm_short(
            np_array, string_plot, shape=shape, origin_index=origin_index, mean_shape=mean_shape)
        procrustes_complete_suites = [procrustes_data, shift_array, scale_array, rotation_matrices]

        with open(string, 'wb') as f:
            pickle.dump(procrustes_complete_suites, f)

    return procrustes_complete_suites


def procrustes_analysis(suites, rerotate=False, overwrite=False, old_data=False):
    """
    A function that assigns procrustes attributes to suite objects.
    :param suites: A list of suites.
    :return:
    """
    complete_suites = [suite for suite in suites if suite.complete_suite]
    five_chain_complete_suites = [suite for suite in suites if suite._five_chain[0] is not None
                                  and suite.dihedral_angles is not None and
                                  suite._five_chain[0][0] is not None and suite._five_chain[1][0] is not None
                                  and suite._five_chain[2][0] is not None and
                                  suite._five_chain[3][0] is not None and suite._five_chain[4][0] is not None]

    complete_five_chains = np.array([suite._five_chain for suite in five_chain_complete_suites])

    if old_data:
        complete_five_chains = np.array([suite._five_chain for suite in complete_suites])
        five_chain_complete_suites = complete_suites

    mean_shapes = np.array([None, None, None, None, None, None])
    if rerotate:
        for i in range(0, 6):
            mean_shapes[i] = np.array(mean_shapes_all[i])

    string = './out/procrustes/five_chain_complete_size_shape.pickle'
    string_plot = './out/procrustes/five_chain'
    procrustes_complete_five_chain = procrustes_on_suite_class(complete_five_chains, string, string_plot,
                                               origin_index=2, overwrite=overwrite, mean_shape=mean_shapes[0])
    for i in range(len(five_chain_complete_suites)):
        five_chain_complete_suites[i].procrustes_five_chain_vector = procrustes_complete_five_chain[0][i]
        five_chain_complete_suites[i].procrustes_five_chain_shift = procrustes_complete_five_chain[1][i]
        five_chain_complete_suites[i].procrustes_five_chain_scale = procrustes_complete_five_chain[2][i]
        five_chain_complete_suites[i].procrustes_five_chain_rotation = procrustes_complete_five_chain[3][i]

    complete_six_chains = np.array([suite._six_chain for suite in complete_suites])
    string = './out/procrustes/six_chain_complete_size_shape.pickle'
    string_plot = './out/procrustes/six_chain'
    procrustes_complete_six_chain = procrustes_on_suite_class(complete_six_chains, string, string_plot,
                                                              overwrite=overwrite, mean_shape=mean_shapes[1])
    for i in range(len(complete_suites)):
        complete_suites[i].procrustes_six_chain_vector = procrustes_complete_six_chain[0][i]
        complete_suites[i].procrustes_six_chain_shift = procrustes_complete_six_chain[1][i]
        complete_suites[i].procrustes_six_chain_scale = procrustes_complete_six_chain[2][i]
        complete_suites[i].procrustes_six_chain_rotation = procrustes_complete_six_chain[3][i]

    complete_suites_bb_atoms = np.array([suite.backbone_atoms for suite in five_chain_complete_suites])
    string = './out/procrustes/suites_complete_size_shape.pickle'
    string_plot = './out/procrustes/suites'
    print(string_plot)
    procrustes_complete_suites = procrustes_on_suite_class(complete_suites_bb_atoms, string, string_plot,
                                                           overwrite=overwrite, mean_shape=mean_shapes[2])

    for i in range(len(five_chain_complete_suites)):
        five_chain_complete_suites[i].procrustes_complete_suite_vector = procrustes_complete_suites[0][i]
        five_chain_complete_suites[i].procrustes_complete_suite_shift = procrustes_complete_suites[1][i]
        five_chain_complete_suites[i].procrustes_complete_suite_scale = procrustes_complete_suites[2][i]
        five_chain_complete_suites[i].procrustes_complete_suite_rotation = procrustes_complete_suites[3][i]

    complete_mesoscopic_rings = np.array([suite.mesoscopic_sugar_rings for suite in complete_suites])
    string = './out/procrustes/mesoscopic_complete_size_shape.pickle'
    string_plot = './out/procrustes/mesoscopic'
    procrustes_complete_mesoscopic = procrustes_on_suite_class(complete_mesoscopic_rings, string, string_plot,
                                                               overwrite=overwrite, mean_shape=mean_shapes[3])
    for i in range(len(complete_suites)):
        complete_suites[i].procrustes_complete_mesoscopic_vector = procrustes_complete_mesoscopic[0][i]
        complete_suites[i].procrustes_complete_mesoscopic_shift = procrustes_complete_mesoscopic[1][i]
        complete_suites[i].procrustes_complete_mesoscopic_scale = procrustes_complete_mesoscopic[2][i]
        complete_suites[i].procrustes_complete_mesoscopic_rotation = procrustes_complete_mesoscopic[3][i]

    # procrustes shape
    string = './out/procrustes/suites_complete_shape.pickle'
    string_plot = './out/procrustes/suites_shape'
    procrustes_complete_suites_shape = procrustes_on_suite_class(complete_suites_bb_atoms, string, string_plot,
                                                        shape=True, overwrite=overwrite, mean_shape=mean_shapes[4])
    for i in range(len(complete_suites)):
        complete_suites[i].procrustes_complete_suite_shape_space_vector = procrustes_complete_suites_shape[0][i]
        complete_suites[i].procrustes_complete_suite_shape_space_shift = procrustes_complete_suites_shape[1][i]
        complete_suites[i].procrustes_complete_suite_shape_space_scale = procrustes_complete_suites_shape[2][i]
        complete_suites[i].procrustes_complete_suite_shape_space_rotation = procrustes_complete_suites_shape[3][i]

    string = './out/procrustes/mesoscopic_complete_shape.pickle'
    string_plot = './out/procrustes/mesoscopic_shape'
    procrustes_complete_mesoscopic_shape = procrustes_on_suite_class(complete_mesoscopic_rings, string, string_plot,
                                                            shape=True, overwrite=overwrite, mean_shape=mean_shapes[5])
    for i in range(len(complete_suites)):
        complete_suites[i].procrustes_complete_mesoscopic_shape_space_vector = procrustes_complete_mesoscopic_shape[0][i]
        complete_suites[i].procrustes_complete_mesoscopic_shape_space_shift = procrustes_complete_mesoscopic_shape[1][i]
        complete_suites[i].procrustes_complete_mesoscopic_shape_space_scale = procrustes_complete_mesoscopic_shape[2][i]
        complete_suites[i].procrustes_complete_mesoscopic_shape_space_rotation = procrustes_complete_mesoscopic_shape[3][i]

    return suites


def average_clustering(input_suites, m, percentage, clean, plot=True):
    """
    A function that clusters the mesoscopics using the classical average clustering.
    :param input_suites:
    :param m:
    :param percentage:
    :param clean:
    :param plot:
    :return:
    """
    method = average
    string = './out/average_mesoscopic_clustering/' + 'size_and_shape_m' + str(m) + 'percentage_' + str(
        percentage) + '/'
    if not os.path.exists(string):
        os.makedirs(string)
    cluster_suites = [suite for suite in input_suites if suite.complete_suite]
    if clean:
        cluster_suites = [suite for suite in cluster_suites if len(suite.bb_bb_neighbour_clashes) == 0
                          and len(suite.bb_bb_one_suite) == 0]
    procrustes_data = np.array([suite.procrustes_complete_mesoscopic_vector for suite in cluster_suites])
    procrustes_data_backbone = np.array([suite.procrustes_complete_suite_vector for suite in cluster_suites])
    distance_data = pdist(procrustes_data.reshape(procrustes_data.shape[0], procrustes_data.shape[1] * 3))
    cluster_data = method(distance_data)
    threshold = find_outlier_threshold(cluster=cluster_data, percentage=percentage,
                                       input_data_shape_0=procrustes_data.shape[0], m=1)
    f_cluster = fcluster(cluster_data, threshold, criterion='distance')
    outlier_cluster_index = [i for i in range(1, max(f_cluster) + 1) if sum(f_cluster == i) <= 1]
    biggest_cluster_information = np.array(collections.Counter(iter(f_cluster)).most_common(np.max(f_cluster)))
    outlier_list = [j for j in range(procrustes_data.shape[0]) if f_cluster[j] in outlier_cluster_index]
    cluster_list = []
    biggest_cluster_information = np.array(collections.Counter(iter(f_cluster)).most_common(np.max(f_cluster)))
    for i in range(biggest_cluster_information.shape[0]):
        if biggest_cluster_information[i, 1] > m:  # and biggest_cluster_information[i, 1] <= m:
            index_list = [k for k in range(procrustes_data.shape[0]) if
                          f_cluster[k] == biggest_cluster_information[i, 0]]
            cluster_list = cluster_list + [index_list]

    cluster_numbers = [suite.clustering['suite_True'] for suite in cluster_suites]
    cluster_number_list = [[cluster_numbers[cluster_list[i][j]] for j in range(len(cluster_list[i]))]
                           for i in range(len(cluster_list))]
    cluster_cut = [
        collections.Counter(cluster_number_list[i]).most_common(np.max(1))[0][1] / len(cluster_number_list[i])
        * 100 for i in range(len(cluster_number_list))]

    hist_own_plot(cluster_cut, x_label='percentages of single MINT-AGE classes',
                  y_label='number of simple mesoscopic clusters', filename=string + 'hist_percentages',
                  bins=None, density=False)
    dihedral_angles = [suite.dihedral_angles * np.pi / 180 - np.pi for suite in cluster_suites]
    var_different_suite_cluster = [np.sqrt(torus_mean_and_var(np.array(dihedral_angles)[cluster_list[i]])[1])
                                   for i in range(len(cluster_list))]

    hist_own_plot(var_different_suite_cluster, x_label='torus standard deviation',
                  y_label='number of simple mesoscopic clusters', filename=string + 'hist',
                  bins=None, density=False)
    print('test')
    plot_all = False
    if plot_all:
        i = 0
        j = 29
        # k=43
        l = 54
        m = 91
        colors = [COLORS[0]] * len(cluster_list[i]) + [COLORS[2]] * len(cluster_list[j]) + ['magenta'] * len(
            cluster_list[l]) + [COLORS[1]] * len(cluster_list[m])
        build_fancy_chain_plot(procrustes_data_backbone[
                                   list(cluster_list[i]) + list(cluster_list[j]) + list(cluster_list[l]) + list(
                                       cluster_list[m])],
                               filename=string + 'cluster_nr' + str(i) + 'and' + str(j) + 'and' + str(l) + 'and' + str(
                                   m) + '_suite',
                               colors=colors,
                               create_label=False,
                               alpha_line_vec=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[l]) + [1] * len(cluster_list[m]),
                               plot_atoms=True,
                               atom_alpha_vector=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[l]) + [1] * len(cluster_list[m]),
                               atom_color_vector=colors, atom_size=0.5, lw=0.4)

        build_fancy_chain_plot(procrustes_data[
                                   list(cluster_list[i]) + list(cluster_list[j]) + list(cluster_list[l]) + list(
                                       cluster_list[m])],
                               filename=string + 'cluster_nr' + str(i) + 'and' + str(j) + 'and' + str(l) + 'and' + str(
                                   m) + '_meso',
                               colors=colors,
                               create_label=False,
                               alpha_line_vec=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[l]) + [1] * len(cluster_list[m]),
                               plot_atoms=True,
                               atom_alpha_vector=[1] * len(cluster_list[i]) + [1] * len(cluster_list[j]) + [1] * len(
                                   cluster_list[l]) + [1] * len(cluster_list[m]),
                               atom_color_vector=colors, atom_size=0.5, lw=0.4)

    plot_clustering(cluster_suites, name=string, cluster_list=cluster_list, outlier_list=outlier_list)


def branch_cutting_with_correction(input_suites, m, percentage, clustering, clean, q_fold, plot=True):
    """
    First Step: Pre clustering.
    Second: Using Mode Hunting and Torus PCA to post cluster the data
    :param input_suites: A list with suite objects.
    :param m: An integer. The minimum cluster size.
    :param percentage: A float. The maximal outlier distance is described by the percentage of elements in a branch with
                       less than m elements.
    :param clustering: A string in ['suite', 'mesoscopic', 'combination_mesoscopic_suite'].
    :param clean: Boolean. If True: Clustering of the training data. If False: Clustering of the admissible data.
    :param q_fold: A float. The percentage value of the q_fold described in the paper.
    :param plot: If True: Plot all figures.
    """
    # Step 1: Pre-Clustering
    method = average
    if clean:
        string = './out/clustering/training_suites/m_is_' + str(m) + 'percentage_is_' + str(
            percentage) + 'q_fold_is' + str(q_fold) + '/'
    else:
        string = './out/clustering/admissible_suites/m_is_' + str(m) + 'percentage_is_' + str(
            percentage) + 'q_fold_is' + str(q_fold) + '/'
    cluster_suites = [suite for suite in input_suites if suite.complete_suite]
    if clean:
        cluster_suites = [suite for suite in cluster_suites if len(suite.bb_bb_neighbour_clashes) == 0
                          and len(suite.bb_bb_one_suite) == 0]
    # clean_suites = [suite for suite in complete_suites if len(suite.clash_list) == 0] #  suites with no clash
    if clustering == 'suite':
        procrustes_data = np.array([suite.procrustes_complete_suite_vector for suite in cluster_suites])
    if clustering == 'mesoscopic':
        procrustes_data = np.array([suite.procrustes_complete_mesoscopic_vector for suite in cluster_suites])
    if clustering == 'combination_mesoscopic_suite':
        procrustes_data = np.array([np.vstack((suite.procrustes_complete_mesoscopic_vector,
                                               suite.procrustes_complete_suite_vector)) for suite in cluster_suites])

    if clean:
        dihedral_angles_suites = np.array([suite._dihedral_angles for suite in cluster_suites if
                                           len(suite.bb_bb_one_suite) == 0 and len(suite.bb_bb_neighbour_clashes) == 0])
    else:
        dihedral_angles_suites = np.array([suite._dihedral_angles for suite in cluster_suites])

    cluster_list, outlier_list, name = pre_clustering(input_data=dihedral_angles_suites, m=m,
                                                      percentage=percentage,
                                                      string_folder=string, method=method,
                                                      q_fold=q_fold)

    if plot:
        plot_clustering(cluster_suites, name=name, cluster_list=cluster_list, outlier_list=outlier_list)

    # Step 2: sing Mode Hunting and Torus PCA to post cluster the data.
    cluster_list, noise = PNDS_RNA_clustering.new_multi_slink(scale=12000, data=dihedral_angles_suites,
                                                              cluster_list=cluster_list, outlier_list=outlier_list)
    outlier_list = [i for i in range(procrustes_data.shape[0]) if
                    i not in [cluster_element for cluster in cluster_list for cluster_element in cluster]]
    if plot:
        plot_clustering(cluster_suites, name=name + 'Torus_PCA/', cluster_list=cluster_list,
                        outlier_list=outlier_list)  # , plot_combinations=True, dihedral_angles_suites=dihedral_angles_suites)
        # plot_clustering(cluster_suites, name=name + 'Torus_PCA_noise/', cluster_list=noise, outlier_list=outlier_list)

    # cluster_list, outlier_list, name = cluster_cutting(procrustes_data=procrustes_data, cluster_list=cluster_list, m=m,
    # name=name)

    # if plot:
    #     plot_clustering(cluster_suites, name=name, cluster_list=cluster_list, outlier_list=outlier_list)

    for suite_number in range(len(cluster_suites)):
        cluster_suites[suite_number].clustering[clustering + '_' + str(clean)] = None
        for cluster_number in range(len(cluster_list)):
            if suite_number in cluster_list[cluster_number]:
                cluster_suites[suite_number].clustering[clustering + '_' + str(clean)] = cluster_number
    return input_suites


def data_analysis_low_res(input_data, dihedral_data, m, percentage, string_folder, method, q_fold, distance='torus', cluster_list=None):
    if distance != 'torus':
        try:
            input_data_copy = input_data.copy()
            input_data = input_data.reshape((input_data.shape[0], input_data.shape[1] * input_data.shape[2]))
        except IndexError:
            print("no shape_space")
            pass

    without_zero = np.delete(input_data_copy, 2, axis=1)
    without_zero_flat = without_zero.reshape((without_zero.shape[0], without_zero.shape[1] * without_zero.shape[2]))
    cov = np.cov(without_zero_flat.transpose())
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    G = eigenvectors
    transformed_data = G.T@without_zero_flat.transpose()
    transformed_data_lower = transformed_data[:9]


    index_map = {str(input_data[i]): i for i in range(input_data.shape[0])}
    # sp_number = np.sqrt(50*input_data.shape[0])
    # print(sp_number)
    n = input_data.shape[0]
    dimension_number = input_data.shape[1]
    cluster_points = input_data.copy()
    reshape_data = input_data

    distance_data = distance_matrix(reshape_data, distance)



    distance_matrix_ = squareform(distance_data)
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity='precomputed')
    projektion = mds.fit_transform(distance_matrix_)



    #plt.scatter(projektion[:, 0], projektion[:, 1])
    #plt.savefig(string_folder + 'MDS')
    #plt.close()
    s=10
    number_of_elements = [len(cluster_list[i]) for i in range(len(cluster_list))]
    for number_element in range(len(number_of_elements)):
        plt.scatter(projektion[cluster_list[number_element], 0],
                    projektion[cluster_list[number_element], 1],
                    c=COLORS_SCATTER[number_element], s=s, marker=MARKERS[number_element])
    #scatter_plots(projektion.transpose(), number_of_elements=list_elements, filename= string_folder + 'cluster_MDS')
    plt.savefig(string_folder + 'MDS_clustering')
    plt.close()
    min_value = np.min(projektion[:, 0])
    max_value = np.max(projektion[:, 0])
    normalized_values = (projektion[:, 0] - min_value) / (max_value - min_value)
    color_index_vector = np.array([int(normalized_values[i]*255) for i in range(len(normalized_values))])
    color_range = plt.cm.coolwarm(np.linspace(0, 1, 256))
    colors = [color_range[value] for value in color_index_vector]
    marker_list_all = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P']
    min_value_1 = np.min(projektion[:, 1])
    max_value_1= np.max(projektion[:, 1])
    normalized_values = (projektion[:, 1] - min_value_1) / (max_value_1 - min_value_1)
    marker_index_vector = np.array([int(normalized_values[i]*(len(marker_list_all)-1)) for i in range(len(normalized_values))])
    marker_list = [marker_list_all[index_] for index_ in marker_index_vector]
    for i in range(len(normalized_values)):
        plt.scatter(projektion[i, 0], projektion[i, 1], color=colors[i], marker=marker_list[i])
    plt.savefig(string_folder + 'MDS')
    plt.close()

    scatter_plots(dihedral_data,
                  filename=string_folder + 'MDS_suites',
                  suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$',
                                r'$\gamma$', r'$\delta_{2}$'], s=30, fontsize=30, color_and_marker_list=[colors, marker_list])



    cluster_tree = single(distance_data)



    cluster_tree = method(distance_data)
    biggest_cluster_point = cluster_tree.shape[0] - 1
    first_value = int(np.min(cluster_tree[biggest_cluster_point, :2])) - cluster_points.shape[0]
    second_value = int(np.max(cluster_tree[biggest_cluster_point, :2])) - cluster_points.shape[0]


    mean_point = np.mean(input_data, axis=0)
    distances = [np.linalg.norm(mean_point-input_data[i]) for i in range(len(input_data))]
    plot.hist(distances, bins=30)
    plot.savefig(string_folder + 'distances')
    plot.close()


    diff_distances = [cluster_tree[i+1, 2]-cluster_tree[i, 2] for i in range(len(cluster_tree)-1)]
    plot.plot(diff_distances)
    plot.savefig(string_folder+'diff_distances')
    plt.close()
    print()
    print('test')
    return [], [], []


# AGE step
def pre_clustering(input_data, m, percentage, string_folder, method, q_fold, distance='torus'):
    """
    The pre clustering described in the paper.
    :param distance: default: 'torus', procrustes: 'sphere' (shape), 'euclidean' (size_shape, shape)
    :param input_data: A np.array with the shape
    :param m: The minimal cluster size.
    :param percentage: A float value. It describes the maximal outlier distance.
    :param string_folder: The name of the folder where the plots are stored.
    :param method: A fuction (average, single, ward, ... ) from scipy.cluster.hierarchy.
    :param q_fold: A float value (introduced in the paper).
    :return:
    """
    if distance != 'torus':
        try:
            input_data = input_data.reshape((input_data.shape[0], input_data.shape[1] * input_data.shape[2]))
        except IndexError:
            print("no shape_space")
            pass

    index_map = {str(input_data[i]): i for i in range(input_data.shape[0])}
    # sp_number = np.sqrt(50*input_data.shape[0])
    # print(sp_number)
    n = input_data.shape[0]
    dimension_number = input_data.shape[1]
    cluster_points = input_data.copy()
    reshape_data = input_data

    distance_data = distance_matrix(reshape_data, distance)

    cluster_data = method(distance_data)
    d_max = find_outlier_threshold(cluster=cluster_data, percentage=percentage, input_data_shape_0=n, m=m)

    # Step 1:
    outlier_list = []
    cluster_list = []
    counter = 0
    while n > 0:
        # Step 2:
        points_reshape = cluster_points.reshape(n, dimension_number)
        distance_points = distance_matrix(points_reshape, distance)  # again to keep dmax
        cluster_tree = method(distance_points)
        # Step 3:
        f_cluster = fcluster(cluster_tree, d_max, criterion='distance')
        outlier_cluster_nr_list = [i for i in range(1, max(f_cluster) + 1) if sum(f_cluster == i) < m]
        outlier_sub_list = [cluster_points[i] for i in range(n) if f_cluster[i] in outlier_cluster_nr_list]
        not_outlier_sub_list_number = [i for i in range(n) if not f_cluster[i] in outlier_cluster_nr_list]
        cluster_points = cluster_points[not_outlier_sub_list_number]
        n = cluster_points.shape[0]
        print(n)
        if n > 0:
            outlier_list = outlier_list + outlier_sub_list
            # Step 4:
            s_p = np.sqrt(n + m ** 2)
            print(s_p)
            points_reshape = cluster_points.reshape(n, dimension_number)
            distance_points = distance_matrix(points_reshape, distance)
            cluster_tree = method(distance_points)
            # Step 5:
            sub_cluster_list, sub_list = branch_cutting(cluster_tree, cluster_points, s_p, q_fold)
            cluster_list = cluster_list + [cluster_points[sub_cluster_list]]
            cluster_points = cluster_points[sub_list]
            n = cluster_points.shape[0]
            # if not os.path.exists(string_folder):  # not needed
            #    os.makedirs(string_folder)
            counter = counter + 1

    cluster_index = [[index_map[str(cluster_list[j][i])] for i in range(cluster_list[j].shape[0])] for j in
                     range(len(cluster_list))]
    outlier_index = [index_map[str(outlier_list[i])] for i in range(len(outlier_list))]
    print(d_max)
    return cluster_index, outlier_index, string_folder

# Copy from PNDS_PNS
def distance_matrix(data, distance='torus'):
    """This function calculates a distance vector that can be used by scipy.cluster.hierarchy's agglomerative clustering
     algorithms. Depending on the string 'distance', a distance matrix is calculated.
    :param data: The data matrix (number of points) x (number of dimensions).
    :param distance: A string from ['torus', 'sphere'].
    :return: A distance vector with the length(number of shapes)*(number of shapes-1)/2.
    """
    if distance == 'torus':
        sum_dihedral_differences = np.zeros(int(data.shape[0] * (data.shape[0] - 1) / 2))
        for i in range(data.shape[1]):
            diff_one_dim = pdist(data[:, i].reshape((data.shape[0], 1)))
            # diff_one_dim = np.min((2*np.pi - diff_one_dim, diff_one_dim), axis=0) ** 2
            diff_one_dim = np.min((360 - diff_one_dim, diff_one_dim), axis=0) ** 2
            sum_dihedral_differences = sum_dihedral_differences + diff_one_dim
        return np.sqrt(sum_dihedral_differences)
    if distance == 'sphere':
        return np.arccos(1 - pdist(data, 'cosine'))
    if distance == 'euclidean':
        return pdist(data, 'euclidean')
    if distance == 'low_res_suite_shape':
        return pdist(data, d_low_res_suite_shape)

def d_low_res_suite_shape(x, y):
    """
    A distance function for the low resolution suite shape.
    Notation is according to the paper.
    :param x: A vector with the first shape.
    :param y: A vector with the second shape.
    :return: The distance between the two shapes.
    """
    d_2_x, d_3_x, alpha_x, theta_1_x, phi_1_x, theta_2_x, phi_2_x = x
    d_2_y, d_3_y, alpha_y, theta_1_y, phi_1_y, theta_2_y, phi_2_y = y

    # angles to radians
    alpha_x = np.deg2rad(alpha_x)
    alpha_y = np.deg2rad(alpha_y)

    theta_1_x = np.deg2rad(theta_1_x)
    theta_1_y = np.deg2rad(theta_1_y)
    phi_1_x = np.deg2rad(phi_1_x)
    phi_1_y = np.deg2rad(phi_1_y)
    theta_2_x = np.deg2rad(theta_2_x)
    theta_2_y = np.deg2rad(theta_2_y)
    phi_2_x = np.deg2rad(phi_2_x)
    phi_2_y = np.deg2rad(phi_2_y)
    
    # d_d2, d_d3, d_alpha are euclidean distances
    d_d2 = d_2_x - d_2_y
    d_d3 = d_3_x - d_3_y
    d_alpha = alpha_x - alpha_y
    # d_theta_1, d_phi_1, d_theta_2, d_phi_2 are spherical distances. Expansion is based on computing the dot product of vectors (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)).
    # sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)
    d_s1 = np.arccos(np.sin(theta_1_x) * np.sin(theta_1_y) * np.cos(phi_1_x - phi_1_y) + np.cos(theta_1_x) * np.cos(theta_1_y))
    d_s2 = np.arccos(np.sin(theta_2_x) * np.sin(theta_2_y) * np.cos(phi_2_x - phi_2_y) + np.cos(theta_2_x) * np.cos(theta_2_y))
    return np.sqrt(d_d2 ** 2 + d_d3 ** 2 + d_alpha ** 2 + d_s1 ** 2 + d_s2 ** 2)



def circular_mean(data):
    """
    This function gets da data vector with values between 0 and 2*pi and returns the mean and the variance corresponding
    to the torus metric. It uses the function variances.
    :param data: A one-dimensional vector with values between 0 and 2pi.
    :return: A list with [mean(data), var(data)].
    """
    n = len(data)
    mean0 = np.mean(data)
    var0 = np.var(data)
    sorted_points = np.sort(data)
    candidates = variances(mean0, var0, n, sorted_points)
    candidates[:, 0] = (candidates[:, 0] + (2 * np.pi)) % (2 * np.pi)
    # Use the unbiased estimator:
    candidates[:, 1] = candidates[:, 1] * n / (n - 1)
    return candidates[np.argmin(candidates[:, 1])]


def variances(mean0, var0, n, points):
    """
    This function is a help function for the function circular_mean.
    :param mean0: The mean of the data (not regarding the torus metric see circular_mean).
    :param var0: The var of the data (not regarding the torus metric see circular_mean).
    :param n: The length of the data (see circular_mean).
    :param points: The sorted data (see circular_mean).
    :return: A list of candidates [[mean_1, var_1],...,[mean_n, var_n]]
    """
    means = (mean0 + np.linspace(0, 2 * np.pi, n, endpoint=False)) % (2 * np.pi)
    means[means >= np.pi] -= 2 * np.pi
    parts = [(sum(points) / n) if means[0] < 0 else 0]
    parts += [((sum(points[:i]) / i) if means[i] >= 0 else (sum(points[i:]) / (n - i)))
              for i in range(1, len(means))]
    # Formula (6) from Hotz, Huckemann:
    means = [[means[i],
              var0 + (0 if i == 0 else
                      ((4 * np.pi * i / n) * (np.pi + parts[i] - mean0) -
                       (2 * np.pi * i / n) ** 2) if means[i] >= 0 else
                      ((4 * np.pi * (n - i) / n) * (np.pi - parts[i] + mean0) -
                       (2 * np.pi * (n - i) / n) ** 2))]
             for i in range(len(means))]
    return np.array(means)


def find_outlier_threshold(cluster, percentage, input_data_shape_0, counter_step_size=30, percentage_threshold=0.04,
                           m=1):
    """
    This function returns for a given percentage value and given linkage matrix the corresponding threshold value such
    that fcluster(cluster, threshold, criterion='distance') has percentage outliers.
    :param cluster: A linkage matrix.
    :param percentage: A float value between 0 and 1.
    :param input_data_shape_0: The number of elements in the data set.
    :param counter_step_size: The size of the steps in the while loop.
    :param percentage_threshold: A value between 0 and 1.
    :return:
    """
    percentage_outliers = 1
    counter = 0
    if len(cluster) < 1000:
        percentage_threshold *= 1000 / len(cluster)
    if percentage <= 0:
        return cluster[-1, 2] + 1
    while percentage_outliers > percentage:
        if len(cluster) < counter:
            return cluster[-1, 2] + 1
        threshold = cluster[counter, 2]
        f_cluster = fcluster(cluster, threshold, criterion='distance')
        cluster_information = collections.Counter(iter(f_cluster))
        outlier_number = np.array([cluster_information[i] for i in cluster_information if cluster_information[i] <= m])
        percentage_outliers = sum(outlier_number) / input_data_shape_0
        if np.abs(percentage_outliers - percentage) > percentage_threshold:
            counter = counter + counter_step_size
        else:
            counter = counter + 1
    return threshold


def branch_cutting(cluster_tree, cluster_points, s_p, q_fold):
    """
    The branch cutting step from the pre clustering as described in the paper.
    :param cluster_tree:
    :param cluster_points:
    :param s_p:
    :param q_fold:
    :return:
    """
    help_list = []
    help_list_tuple = []
    true_value = True
    biggest_cluster_point = cluster_tree.shape[0] - 1
    while true_value:
        first_value = int(np.min(cluster_tree[biggest_cluster_point, :2])) - cluster_points.shape[0]
        second_value = int(np.max(cluster_tree[biggest_cluster_point, :2])) - cluster_points.shape[0]
        # print(first_value)
        if first_value > -1:
            first_cluster_size = int(cluster_tree[first_value, 3])
            second_cluster_size = int(cluster_tree[second_value, 3])
            if first_cluster_size < second_cluster_size:
                smaller_cluster = first_value
                bigger_cluster = second_value
            else:
                smaller_cluster = second_value
                bigger_cluster = first_value
            help_list_tuple = help_list_tuple + [[smaller_cluster, bigger_cluster]]
            if int(cluster_tree[smaller_cluster, 3]) > s_p and (
                    (1 - cluster_tree[bigger_cluster, 2] / cluster_tree[biggest_cluster_point, 2] > q_fold) and (
                    1 - cluster_tree[smaller_cluster, 2] / cluster_tree[biggest_cluster_point, 2] > q_fold)):
                help_list = help_list + [smaller_cluster]
            biggest_cluster_point = bigger_cluster
        else:
            if second_value < 0:
                true_value = False
            else:
                biggest_cluster_point = second_value
    if len(help_list) < 1:
        return [i for i in range(cluster_points.shape[0])], []
    last_smaller_cluster = help_list[len(help_list) - 1]
    for i in range(len(help_list_tuple)):
        if help_list_tuple[i][0] == last_smaller_cluster:
            help_list = help_list + [help_list_tuple[i][1]]

    biggest_cluster_in_l = help_list[np.argmax([int(cluster_tree[int_][3]) for int_ in help_list])]

    one_element = return_one_element(cluster_tree, biggest_cluster_in_l)
    f_cluster = fcluster(cluster_tree, cluster_tree[biggest_cluster_in_l, 2], criterion='distance')
    return [i for i in range(cluster_points.shape[0]) if f_cluster[i] == f_cluster[one_element]], \
           [i for i in range(cluster_points.shape[0]) if not f_cluster[i] == f_cluster[one_element]]


def return_one_element(cluster_tree, biggest_cluster_in_l):
    """
    A help function for the branch cutting algorithm.
    :param cluster_tree:
    :param biggest_cluster_in_l:
    :return:
    """
    run_value = biggest_cluster_in_l
    # first_value = np.int(np.min(cluster_tree[run_value, :2]))-cluster_tree.shape[0]  - 1

    # second_value = np.int(np.max(cluster_tree[biggest_cluster_in_l, :2]))-cluster_tree.shape[0] - 1
    true_value = True
    while true_value:
        first_value = int(np.min(cluster_tree[run_value, :2])) - cluster_tree.shape[0] - 1
        if first_value < 0:
            return first_value + cluster_tree.shape[0] + 1
        run_value = first_value


def pns_and_mode_hunting_on_cluster_list(cluster_list, project_data_to_sphere, string, data, euclidean_pca=True):
    run_cluster_list = cluster_list.copy()
    final_cluster_list = []
    counter = 1
    while len(run_cluster_list) > 0:
        print(counter)
        new_clusters = pns_and_mode_hunting(data[run_cluster_list[0]], string=string, cluster=run_cluster_list[0],
                                            run_cluster_list=run_cluster_list, final_cluster_list=final_cluster_list,
                                            project_data_to_sphere=project_data_to_sphere, counter=counter)
        print('len run cluster list ', len(run_cluster_list))
        print('len final cluster list', len(final_cluster_list))
        counter = counter + 1
    return final_cluster_list


def pca_and_mode_hunting_on_cluster_list(cluster_list, project_data_to_sphere, string, data, euclidean_pca=True):
    run_cluster_list = cluster_list.copy()
    final_cluster_list = []
    counter = 1
    while len(run_cluster_list) > 0:
        print(counter)
        new_clusters = pca_and_mode_hunting(data[run_cluster_list[0]], string=string, cluster=run_cluster_list[0],
                                            run_cluster_list=run_cluster_list, final_cluster_list=final_cluster_list,
                                            project_data_to_sphere=False, counter=counter)
        print('len run cluster list ', len(run_cluster_list))
        print('len final cluster list', len(final_cluster_list))
        counter = counter + 1
    return final_cluster_list


def loss_projection_on_sphere(p, data):
    y = np.linalg.norm(data - p[:-1], axis=1)
    return np.sum((y - p[-1]) ** 2)


def project_data_on_sphere(data, no_test=True, string=None, dummy_list=None):
    mean_data_start = np.mean(data, axis=0)
    start_radius = np.mean(np.abs(mean_data_start - data))
    res = minimize(fun=loss_projection_on_sphere,
                   x0=np.hstack([mean_data_start, start_radius]),
                   method='L-BFGS-B', options={'ftol': 1e-5},
                   args=(data))

    n = data.shape[0]
    d = data.shape[1]

    squared_residuals_fit_sphere = res.fun
    chi_sphere = n * np.log(squared_residuals_fit_sphere)

    pca_ew, pca_ev = np.linalg.eig(np.cov(data.T))
    squared_residuals_fit_plane = np.min(pca_ew) * n
    chi_plane = n * np.log(squared_residuals_fit_plane)

    chi2 = 1 - stat.chi2.cdf(chi_plane - chi_sphere, 1)

    chi_test = chi2 > 0.05
    print("radius", res.x[-1])
    radii = np.linalg.norm(data - res.x[:-1], axis=1)
    likelihood = compare_likelihoods(radii, d, False, euclidean=True)
    print("d", d, "likelihood", likelihood, "chi2", chi2)

    if (not chi_test and not likelihood) or d == 2 or no_test:
        data_shifted = data - res.x[:-1]
        data_on_sphere = data_shifted / np.linalg.norm(data_shifted, axis=1)[:, np.newaxis]
        
        # Calculate the residuals (distances from the original points to the fitted sphere)
        residuals = np.linalg.norm(data - data_on_sphere, axis=1)
        
        if d == 2 and string is not None:
            plot_two_dimensional_projection(data, data_on_sphere, data_shifted, res, string, dummy_list=dummy_list)
        
        # Return both projected data and residuals
        return data_on_sphere, residuals
    else:
        order = np.argsort(pca_ew)[::-1]
        pca_ev = pca_ev[:, order]
        proj = pca_ev[:, :d - 1]
        data_d_minus_1 = np.dot(proj.T, data.T).T
        
        # Recursive call to project lower-dimensional data
        return project_data_on_sphere(data_d_minus_1)


def plot_two_dimensional_projection(data, data_on_sphere, data_shifted, res, string, dummy_list=None):
    if not os.path.exists(string):
        os.makedirs(string)
    figure, axes = plot.subplots(figsize=(10, 10))
    if dummy_list is None:
        plot.scatter(data[:, 0], data[:, 1], label='data points', c="black", s=25)
    else:
        COLORS_PLOT = ['black', 'darkgreen', 'orange', 'pink', 'yellow', 'gold', 'navy', 'magenta', 'darkviolet',
                       'tomato', 'peru']
        marker_list = ["s", "o", "*"]
        for i in range(len(dummy_list)):
            plot.scatter(data[sum(dummy_list[:i]):sum(dummy_list[:i + 1]), 0],
                         data[sum(dummy_list[:i]):sum(dummy_list[:i + 1]), 1], c=COLORS_PLOT[i], s=50,
                         marker=marker_list[i])
    # plot.scatter([res.x[0]], [res.x[1]], label='mean sphere', c="darkblue", s=50)
    a, b = np.polyfit(data[:, 0], data[:, 1], 1)
    x = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 100)
    plot.plot(x, a * x + b, c="firebrick", label="Main Euclidean principal component", linewidth=3)
    plot.plot([], [], c="steelblue", label="Main principal nested circle", linewidth=3)
    draw_circle = plot.Circle(res.x[:-1], res.x[-1], label='sphere', fill=False, linewidth=3, color="steelblue")
    plot.gcf().gca().add_artist(draw_circle)
    axes.axis('equal')
    plot.xlim([np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1])
    plot.ylim([np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1])
    plot.legend(prop={'size': 15})
    plot.savefig(string + 'data.png')
    plot.close()

    if dummy_list is not None:
        figure, axes = plot.subplots(figsize=(10, 10))
        a, b = np.polyfit(data[:, 0], data[:, 1], 1)
        x = np.linspace(np.min(data[:, 0]) - 2, np.max(data[:, 0]) + 2, 100)
        plot.plot(x, a * x + b, c="firebrick", label="Main Euclidean principal component", linewidth=3)
        plot.plot([], [], c="steelblue", label="Main principal nested circle", linewidth=3)
        draw_circle = plot.Circle(res.x[:-1], res.x[-1], label='sphere', fill=False, linewidth=3, color="steelblue",
                                  alpha=1)
        plot.gcf().gca().add_artist(draw_circle)
        COLORS_PLOT = ['black', 'darkgreen', 'orange', 'pink', 'yellow', 'gold', 'navy', 'magenta', 'darkviolet',
                       'tomato', 'peru']
        size_list = [100, 200, 100]
        for i in range(len(dummy_list)):
            plot.scatter(data_on_sphere[sum(dummy_list[:i]):sum(dummy_list[:i + 1]), 0] * res.x[-1] + res.x[0],
                         data_on_sphere[sum(dummy_list[:i]):sum(dummy_list[:i + 1]), 1] * res.x[-1] + res.x[1],
                         c=COLORS_PLOT[i], s=size_list[i], marker=marker_list[i])

        data_x_shifted = data[:, 0]
        data_y_shifted = data[:, 1] - b
        b_line = np.array([1, a])
        angle = np.arccos((data_x_shifted + a * data_y_shifted) / (
                (1 + a ** 2) ** (1 / 2) * (data_x_shifted ** 2 + data_y_shifted ** 2) ** (1 / 2)))
        a_1 = (data_x_shifted ** 2 + data_y_shifted ** 2) ** (1 / 2) * np.cos(angle)
        projected_data_x = a_1
        projected_data_y = a * a_1 + b
        alpha_list = [1, 1, 1]
        for i in range(len(dummy_list)):
            plot.scatter(projected_data_x[sum(dummy_list[:i]):sum(dummy_list[:i + 1])],
                         projected_data_y[sum(dummy_list[:i]):sum(dummy_list[:i + 1])],
                         c=COLORS_PLOT[i], s=size_list[i], alpha=alpha_list[i], marker=marker_list[i])

        axes.axis('equal')
        plot.xlim([np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1])
        plot.ylim([np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1])
        plot.legend(prop={'size': 15})
        plot.savefig(string + 'data_transformation.png')
        plot.close()

    plot.scatter(data_shifted[:, 0], data_shifted[:, 1], label='data points shifted')
    plot.legend()
    plot.savefig(string + 'data_shifted.png')
    plot.close()
    figure, axes = plot.subplots()
    plot.scatter(data_on_sphere[:, 0], data_on_sphere[:, 1], label='data points on sphere')
    draw_circle = plot.Circle((0, 0), 1, label='sphere', fill=False)
    plot.gcf().gca().add_artist(draw_circle)
    plot.legend()
    plot.xlim((-1.3, 1.3))
    plot.ylim((-1.3, 1.3))
    plot.savefig(string + 'data_on_sphere.png')
    plot.close()


def pns_and_mode_hunting(data, string, cluster, run_cluster_list, final_cluster_list, project_data_to_sphere, counter):
    new_clusters = []
    # radii = np.arccos(np.abs(np.einsum('ij,j->i', points, tmp[:-1])))
    # #tmp2 = tmp.copy()
    # tmp2 = fit((g2 if mode is 'torus' else f2), d, tmp[:-1], verbose, False)
    # log_res_small = np.log(np.sum(f(tmp[:-1])**2))
    # log_res_great = np.log(np.sum(f2(tmp2)**2))
    # chi2 = 1 - stat.chi2.cdf(log_res_great - log_res_small, 1)
    if project_data_to_sphere:
        sphere_data = project_data_on_sphere(data, no_test=False)
    else:
        sphere_data = data
    spheres, projected_points, distances = PNDS_RNA_clustering.pns_loop(sphere_data, 10000, mode='scale')
    print('len_data', len(data))

    # Mode Hunting:
    if sphere_data.shape[1] == 2:
        print('test')
    center_point = PNDS_PNS.unfold_points(PNDS_PNS.as_matrix(projected_points[-1]), spheres[:-1])
    mode_hunting = True
    if sphere_data.shape[1] > 2:
        unfolded_1d = PNDS_PNS.unfold_points(projected_points[-2], spheres[:-1])
        relative_residual_variance = (np.mean(PNDS_geometry.sphere_distances(unfolded_1d, sphere_data) ** 2) / np.mean(
            PNDS_geometry.sphere_distances(center_point, sphere_data) ** 2))

        print("Counter", counter, "relative_residual_variance", relative_residual_variance)

        if relative_residual_variance > 0.25:
            print("Too much variance in higher dimension")
            mode_hunting = False

    control_plot = False
    if control_plot:
        build_fancy_chain_plot(data.reshape((data.shape[0], 6, 3)),
                               filename=string + str(counter) + '_run_cluster_rel_var_' + str(
                                   np.round(relative_residual_variance, 3)))
    this_split = True
    #### Change
    quantile = Multiscale_modes.get_quantile(len(sphere_data), 0.05)
    if mode_hunting:
        mode_list, mins = Multiscale_modes.get_modes(distances[-1], 360., quantile)
        if len(mins) > 1:
            this_split = False
            print(len(mins), 'modes expected')
        else:
            mode_hunting = False

    print(len(run_cluster_list))
    if mode_hunting:
        alpha = 0
        while (not this_split) and (alpha < 5):
            alpha += 1
            quantile = Multiscale_modes.get_quantile(len(sphere_data), 0.01 * alpha)
            # print('quantile', quantile)
            dists = distances[-1]
            mode_list, mins = Multiscale_modes.get_modes(dists, 360., quantile)
            if len(mins) > 1:
                print(len(mins), 'modes found at alpha = %.2f' % (0.01 * alpha), flush=True)
                # print(mode_list)
                print(mode_list[0].shape)
                run_cluster_list += [[cluster[cluster_element] for cluster_element in x] for x in mode_list]
                run_cluster_list.remove(cluster)
                return [[cluster[cluster_element] for cluster_element in x] for x in mode_list]
    print(len(run_cluster_list))
    run_cluster_list.remove(cluster)
    final_cluster_list.append(cluster)
    return [cluster]


def pca_and_mode_hunting(data, string, cluster, run_cluster_list, final_cluster_list, project_data_to_sphere, counter):
    if data.shape[0] > 1000:
        print("test")
    if len(data) == 619:
        print("test")
    new_clusters = []

    # sphere_data = data
    pca_ew, pca_ev = np.linalg.eig(np.cov(data.T))
    index = np.argmax(pca_ew.real)
    pc1 = pca_ev[:, index].real
    center = np.mean(data, axis=0)[np.newaxis, :]
    projected_points = np.dot(pc1, (data - center).T)
    mode_hunting = True
    if data.shape[1] > 2:
        relative_residual_variance = 1 - np.max(pca_ew) / np.sum(pca_ew)

        print("Counter", counter, "relative_residual_variance", relative_residual_variance)

        if relative_residual_variance > 1:  # 0.25:
            print("Too much variance in higher dimension")
            mode_hunting = False
    plot.hist(projected_points)
    plot.savefig(str(counter))
    plot.close()
    control_plot = False
    if control_plot:
        build_fancy_chain_plot(data.reshape((data.shape[0], 6, 3)),
                               filename=string + str(counter) + '_run_cluster_rel_var_' + str(
                                   np.round(relative_residual_variance, 3)))
    this_split = True
    quantile = multiscale_modes_linear.get_quantile(len(data), 0.05)
    print(data.shape)
    if mode_hunting:
        mode_list, mins = multiscale_modes_linear.get_modes(projected_points, quantile)
        if len(mins) > 0:
            this_split = False
            print(len(mins) + 1, 'modes expected')
        else:
            mode_hunting = False

    print(len(run_cluster_list))
    if mode_hunting:
        alpha = 0
        while (not this_split) and (alpha < 5):
            alpha += 1
            quantile = multiscale_modes_linear.get_quantile(len(data), 0.01 * alpha)
            # print('quantile', quantile)
            # up, down = multiscale_modes_linear.find_slopes(data, quantile)
            # maxr, minr, extr = multiscale_modes_linear.find_extrema_regions(data, quantile)
            mode_list, mins = multiscale_modes_linear.get_modes(projected_points, quantile)
            if len(mins) > 0:
                print(len(mins) + 1, 'modes found at alpha = %.2f' % (0.01 * alpha), flush=True)
                # print(mode_list)
                print(mode_list[0].shape)
                run_cluster_list += [[cluster[cluster_element] for cluster_element in x] for x in mode_list]
                run_cluster_list.remove(cluster)
                return [[cluster[cluster_element] for cluster_element in x] for x in mode_list]
    print(len(run_cluster_list))
    run_cluster_list.remove(cluster)
    final_cluster_list.append(cluster)
    return [cluster]


def shape_five_chain(input_suites, input_string_folder):
    average_clustering = True
    if average_clustering:
        method = average
        str_ = 'average'
        # percentage = 0.009
        percentage = 0.05
    else:
        method = single
        str_ = 'single'
        percentage = 0.15
    m = 10
    for name_ in ['c_3_c_3_suites']:# ['c_3_c_3_suites', 'c_3_c_2_suites', 'c_2_c_3_suites', 'c_2_c_2_suites']:
        # folder = './out/five_chains/'
        folder = './out/newdata/five_chains/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # cluster_suites = [suite for suite in input_suites if suite.complete_suite]
        cluster_suites = [suite for suite in input_suites if suite.procrustes_five_chain_vector is not None
                          and suite.dihedral_angles is not None]


        if name_ == 'c_2_c_2_suites':
            c_2_c_2_suites = [suite for suite in cluster_suites if
                              suite._nu_1[0] > 300 and suite._nu_1[0] < 350 and suite._nu_2[0] > 300 and suite._nu_2[
                                  0] < 350]
            cluster_suites = c_2_c_2_suites
        if name_ == 'c_3_c_3_suites':
            c_3_c_3_suites = [suite for suite in cluster_suites if
                              not (suite._nu_1[0] > 300 and suite._nu_1[0] < 350) and not (
                                      suite._nu_2[0] > 300 and suite._nu_2[0] < 350)]
            cluster_suites = c_3_c_3_suites
        if name_ == 'c_3_c_2_suites':
            c_3_c_2_suites = [suite for suite in cluster_suites if
                              not (suite._nu_1[0] > 300 and suite._nu_1[0] < 350) and suite._nu_2[0] > 300 and
                              suite._nu_2[0] < 350]
            cluster_suites = c_3_c_2_suites
        if name_ == 'c_2_c_3_suites':
            c_2_c_3_suites = [suite for suite in cluster_suites if
                              suite._nu_1[0] > 300 and suite._nu_1[0] < 350 and not (
                                      suite._nu_2[0] > 300 and suite._nu_2[0] < 350)]
            cluster_suites = c_2_c_3_suites

        folder = folder + str_ + str(percentage) + '/' + name_ + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # cluster_suites = [suite for suite in cluster_suites if
        #                   len(suite.clustering) > 0 and suite.clustering['suite_True'] is not None]
        procrustes_data = np.array([suite.procrustes_five_chain_vector for suite in cluster_suites])
        # procrustes on puckers and then rewrite procrustes data
        string = './out/procrustes/five_chain_complete_size_shape' + name_ + '.pickle'
        string_plot = './out/procrustes/five_chain' + name_
        if len(procrustes_data) == 0:
            print("no procrustes-data for pucker" + name_)
            continue
        procrustes_data_pucker = procrustes_on_suite_class(procrustes_data, string, string_plot, origin_index=2,
                                                           mean_shape=np.array(mean_shapes_all[0]))
        procrustes_data = np.array([procrustes_data_pucker[0][i] for i in range(len(cluster_suites))])

        string = './out/procrustes/suites_complete_size_shape' + name_ + '.pickle'
        string_plot = './out/procrustes/suites' + name_
        procrustes_data_backbone = np.array([suite.procrustes_complete_suite_vector for suite in cluster_suites])
        procrustes_data_backbone_pucker = procrustes_on_suite_class(procrustes_data_backbone, string, string_plot,
                                                                    mean_shape=np.array(mean_shapes_all[2]))
        procrustes_data_backbone = np.array([procrustes_data_backbone_pucker[0][i] for i in range(len(cluster_suites))])

        distance_data = pdist(procrustes_data.reshape(procrustes_data.shape[0], procrustes_data.shape[1] * 3))
        cluster_data = method(distance_data)
        threshold = find_outlier_threshold(cluster=cluster_data, percentage=percentage,
                                           input_data_shape_0=procrustes_data.shape[0], m=1)
        f_cluster = fcluster(cluster_data, threshold, criterion='distance')
        outlier_cluster_index = [i for i in range(1, max(f_cluster) + 1) if sum(f_cluster == i) <= 1]

        biggest_cluster_information = np.array(collections.Counter(iter(f_cluster)).most_common(np.max(f_cluster)))
        outlier_list = [j for j in range(procrustes_data.shape[0]) if f_cluster[j] in outlier_cluster_index]
        cluster_list = []
        biggest_cluster_information = np.array(collections.Counter(iter(f_cluster)).most_common(np.max(f_cluster)))
        plot_all = True
        if plot_all:
            for i in range(biggest_cluster_information.shape[0]):
                if biggest_cluster_information[i, 1] > m:  # and biggest_cluster_information[i, 1] <= m:
                    index_list = [k for k in range(procrustes_data.shape[0]) if
                                  f_cluster[k] == biggest_cluster_information[i, 0]]
                    cluster_list = cluster_list + [index_list]
            for i in range(len(cluster_list)):
                atom_colors = ['darkblue', 'steelblue', 'orange', 'steelblue', 'darkblue']
                atom_color_matrix = len(cluster_list[i]) * [atom_colors]
                build_fancy_chain_plot(procrustes_data_backbone[list(cluster_list[i])],
                                       filename=folder + 'cluster_nr' + str(i) + '_suite') # , without_legend=True

                build_fancy_chain_plot(procrustes_data[list(cluster_list[i])],
                                       filename=folder + 'cluster_nr' + str(i) + '_six_chain', plot_atoms=True,
                                       atom_color_matrix=atom_color_matrix, without_legend=True)

    return 0
    # cluster_suites = [suite for suite in input_suites if suite.complete_suite]
    cluster_suites = [suite for suite in input_suites if suite.procrustes_five_chain_vector is not None
                      and suite.dihedral_angles is not None]
    procrustes_data = np.array([suite.procrustes_five_chain_vector for suite in cluster_suites])
    new_data = [procrustes_data[i] - procrustes_data[i][0] for i in range(len(procrustes_data))]
    rotation_matrices = [rotation(new_data[i][1] / np.linalg.norm(new_data[i][1]), np.array([1, 0, 0])) for i in
                         range(len(procrustes_data))]
    new_data = [(rotation_matrices[i] @ new_data[i].T).T for i in range(len(new_data))]

    def distance_plane(angle, x):
        return np.linalg.norm((rotation_matrix_x_axis(angle) @ x)[2])

    angles = []
    for i in range(len(new_data)):
        index = np.argmin([minimize(fun=distance_plane,
                                    x0=[ang],
                                    method="Powell",
                                    args=(new_data[i][2])).fun for ang in np.linspace(0, 2 * np.pi, 10)])
        angles.append(minimize(fun=distance_plane,
                               x0=[np.linspace(0, 2 * np.pi, 10)[index]],
                               method="Powell",
                               args=(new_data[i][2])))

    # angles = [minimize(fun=distance_plane,
    #          x0=[0],
    #         method="Powell",
    #         args=(new_data[i][2])) for i in range(len(new_data))]
    angles = [angle.x[0] for angle in angles]
    # angles = [angles[i] if ((rotation_matrix_x_axis(angles[i])@new_data[i].T).T)[2,1]>0 else np.pi-angles[i] for i in range(len(angles))]
    new_data_ = [(rotation_matrix_x_axis(angles[i]) @ new_data[i].T).T for i in range(len(angles))]
    for i in range(len(new_data_)):
        if new_data_[i][2, 1] < 0:
            new_data_[i] = (rotation_matrix_x_axis(np.pi) @ new_data_[i].T).T
    # new_data = [(rotation_matrix_x_axis(angles[i])@new_data[i].T).T for i in range(len(angles))]
    # rotation_matrix_x_axis(angles[0])@new_data[0].T
    folder = './out/five_chains/c_2_c_3/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    c_2_index = [i for i in range(len(cluster_suites)) if
                 cluster_suites[i]._nu_1[0] > 300 and cluster_suites[i]._nu_1[0] < 350]
    c_3_index = [i for i in range(len(cluster_suites)) if
                 not (cluster_suites[i]._nu_1[0] > 300 and cluster_suites[i]._nu_1[0] < 350)]
    # build_fancy_chain_plot(np.array(new_data_)[c_2_index + c_3_index][:, :3, :], filename=folder + 'pucker',
    #                       colors=['black']*len(c_2_index) + ['darkred']*len(c_3_index),
    #                       atom_color_matrix=atom_color_matrix, )

    # P atoms:
    plot.scatter(np.array(new_data_)[c_2_index][:, 2, 0], np.array(new_data_)[c_2_index][:, 2, 1], c='black',
                 label=r'$C_2$  pucker')
    plot.scatter(np.array(new_data_)[c_3_index][:, 2, 0], np.array(new_data_)[c_3_index][:, 2, 1], c='darkred',
                 alpha=0.5, label=r'$C_3$  pucker')
    plot.scatter([0], [0], c='darkblue', s=50, label=r'$N_1$ / $N_9$')  # N1 Atom
    plot.scatter(np.array(new_data_)[c_2_index][:, 1, 0], np.array(new_data_)[c_2_index][:, 1, 1], c='steelblue',
                 label=r'$C_1$ atom ($C_2$ pucker)')  # C1 Atom
    plot.scatter(np.array(new_data_)[c_3_index][:, 1, 0], np.array(new_data_)[c_3_index][:, 1, 1], c='mediumvioletred',
                 label=r'$C_1$ atom ($C_3$ pucker)')  # C1 Atom
    plot.xlim(-0.3, 6.5)
    plot.ylim(-0.3, 5.5)
    # plot.legend(loc='center left')
    plot.xlabel('P: x value')
    plot.ylabel('P: y value')
    plot.legend()
    plot.savefig(folder + 'two_d')
    plot.close()

    plot.scatter(np.array(new_data_)[c_2_index][:, 2, 0], np.array(new_data_)[c_2_index][:, 2, 1], c='black',
                 label=r'$C_2$  pucker')
    plot.scatter([0], [0], c='darkblue', s=50, label=r'$N_1$ / $N_9$')  # N1 Atom
    plot.scatter(np.array(new_data_)[c_2_index][:, 1, 0], np.array(new_data_)[c_2_index][:, 1, 1], c='steelblue',
                 label=r'$C_1$')  # C1 Atom
    plot.xlim(-0.3, 6.5)
    plot.ylim(-0.3, 5.5)
    plot.xlabel('P: x value')
    plot.ylabel('P: y value')
    plot.legend()
    plot.savefig(folder + 'two_d_c2')
    plot.close()

    plot.scatter(np.array(new_data_)[c_3_index][:, 2, 0], np.array(new_data_)[c_3_index][:, 2, 1], c='darkred',
                 alpha=0.5, label=r'$C_3$  pucker')
    plot.scatter([0], [0], c='darkblue', s=50, label=r'$N_1$ / $N_9$')  # N1 Atom
    plot.scatter(np.array(new_data_)[c_3_index][:, 1, 0], np.array(new_data_)[c_3_index][:, 1, 1], c='crimson',
                 label=r'$C_1$')  # C1 Atom
    plot.xlim(-0.3, 6.5)
    plot.ylim(-0.3, 5.5)
    plot.xlabel('P: x value')
    plot.ylabel('P: y value')
    plot.legend()
    plot.savefig(folder + 'two_d_c3')
    plot.close()


    """
    folder = './out/five_chains/clash_suites/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    number_neighbors = 50
    clash_suites = [suite for suite in cluster_suites if len(suite.bb_bb_one_suite) > 0]
    backbone_validation_list = [i for i in range(len(cluster_suites)) if len(cluster_suites[i].bb_bb_one_suite) > 0]
    backbone_validation_list_2 = [i for i in range(len(cluster_suites)) if
                                  len(cluster_suites[i].bb_bb_one_suite) > 0 or len(
                                      cluster_suites[i].bb_bb_neighbour_clashes) > 0]
    for k in range(len(backbone_validation_list)):
        clash_suite = backbone_validation_list[k]
        data = procrustes_data.copy()
        x = procrustes_data[clash_suite].copy()
        for i in range(len(data)):
            data[i] = rotate_y_optimal_to_x(x=x, y=data[i])
        neighbors_dummy = np.array([np.linalg.norm(data[clash_suite] - data[element]) for element in
                                    range(data.shape[0])]).argsort()[1:]
        neighbors = [neighbors_dummy[i] for i in range(len(neighbors_dummy)) if
                     neighbors_dummy[i] not in backbone_validation_list_2][0:number_neighbors]
        build_fancy_chain_plot(procrustes_data_backbone[[clash_suite] + neighbors],
                               colors=[COLORS[1]] + [COLORS[0]] * len(neighbors), create_label=False,
                               lw_vec=[2] + [0.3] * number_neighbors,
                               filename=folder + 'clash_suite' + str(k) + '_suite')
        atom_colors = ['darkblue', 'steelblue', 'orange', 'steelblue', 'darkblue', 'orange']
        atom_color_matrix = (len(neighbors) + 1) * [atom_colors]
        build_fancy_chain_plot(procrustes_data[[clash_suite] + neighbors],
                               colors=[COLORS[1]] + [COLORS[0]] * len(neighbors),
                               filename=folder + 'clash_suite' + str(k) + '_six_chain', plot_atoms=True,
                               create_label=False,
                               lw_vec=[2] + [0.3] * number_neighbors,
                               atom_color_matrix=atom_color_matrix)
        print('Test')"""
    print('test')


def shape_six_chain(input_suites, input_string_folder):
    folder = './out/six_chains/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    average_clustering = False
    if average_clustering:
        method = average
        str_ = 'average'
        percentage = 0.05
    else:
        method = average
        str_ = 'single'
        percentage = 0.15
    m = 10
    folder = folder + str_ + str(percentage) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    cluster_suites = [suite for suite in input_suites if suite.complete_suite]
    procrustes_data = np.array([suite.procrustes_six_chain_vector for suite in cluster_suites])
    procrustes_data_backbone = np.array([suite.procrustes_complete_suite_vector for suite in cluster_suites])
    distance_data = pdist(procrustes_data.reshape(procrustes_data.shape[0], procrustes_data.shape[1] * 3))
    cluster_data = method(distance_data)
    threshold = find_outlier_threshold(cluster=cluster_data, percentage=percentage,
                                       input_data_shape_0=procrustes_data.shape[0], m=1)
    f_cluster = fcluster(cluster_data, threshold, criterion='distance')
    outlier_cluster_index = [i for i in range(1, max(f_cluster) + 1) if sum(f_cluster == i) <= 1]
    biggest_cluster_information = np.array(collections.Counter(iter(f_cluster)).most_common(np.max(f_cluster)))
    outlier_list = [j for j in range(procrustes_data.shape[0]) if f_cluster[j] in outlier_cluster_index]
    cluster_list = []
    biggest_cluster_information = np.array(collections.Counter(iter(f_cluster)).most_common(np.max(f_cluster)))
    plot_all = False
    if plot_all:
        for i in range(biggest_cluster_information.shape[0]):
            if biggest_cluster_information[i, 1] > m:  # and biggest_cluster_information[i, 1] <= m:
                index_list = [k for k in range(procrustes_data.shape[0]) if
                              f_cluster[k] == biggest_cluster_information[i, 0]]
                cluster_list = cluster_list + [index_list]
        for i in range(len(cluster_list)):
            atom_colors = ['darkblue', 'steelblue', 'orange', 'steelblue', 'darkblue', 'orange']
            atom_color_matrix = len(cluster_list[i]) * [atom_colors]
            build_fancy_chain_plot(procrustes_data_backbone[list(cluster_list[i])],
                                   filename=folder + 'cluster_nr' + str(i) + '_suite')

            build_fancy_chain_plot(procrustes_data[list(cluster_list[i])],
                                   filename=folder + 'cluster_nr' + str(i) + '_six_chain', plot_atoms=True,
                                   atom_color_matrix=atom_color_matrix)

    folder = './out/six_chains/clash_suites/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    number_neighbors = 50
    clash_suites = [suite for suite in cluster_suites if len(suite.bb_bb_one_suite) > 0]
    backbone_validation_list = [i for i in range(len(cluster_suites)) if len(cluster_suites[i].bb_bb_one_suite) > 0]
    backbone_validation_list_2 = [i for i in range(len(cluster_suites)) if
                                  len(cluster_suites[i].bb_bb_one_suite) > 0 or len(
                                      cluster_suites[i].bb_bb_neighbour_clashes) > 0]
    for k in range(len(backbone_validation_list)):
        clash_suite = backbone_validation_list[k]
        data = procrustes_data.copy()
        x = procrustes_data[clash_suite].copy()
        for i in range(len(data)):
            data[i] = rotate_y_optimal_to_x(x=x, y=data[i])
        neighbors_dummy = np.array([np.linalg.norm(data[clash_suite] - data[element]) for element in
                                    range(data.shape[0])]).argsort()[1:]
        neighbors = [neighbors_dummy[i] for i in range(len(neighbors_dummy)) if
                     neighbors_dummy[i] not in backbone_validation_list_2][0:number_neighbors]
        build_fancy_chain_plot(procrustes_data_backbone[[clash_suite] + neighbors],
                               colors=[COLORS[1]] + [COLORS[0]] * len(neighbors), create_label=False,
                               lw_vec=[2] + [0.3] * number_neighbors,
                               filename=folder + 'clash_suite' + str(k) + '_suite')
        atom_colors = ['darkblue', 'steelblue', 'orange', 'steelblue', 'darkblue', 'orange']
        atom_color_matrix = (len(neighbors) + 1) * [atom_colors]
        build_fancy_chain_plot(procrustes_data[[clash_suite] + neighbors],
                               colors=[COLORS[1]] + [COLORS[0]] * len(neighbors),
                               filename=folder + 'clash_suite' + str(k) + '_six_chain', plot_atoms=True,
                               create_label=False,
                               lw_vec=[2] + [0.3] * number_neighbors,
                               atom_color_matrix=atom_color_matrix)
        print('Test')
        # nr_in_cluster = [len(set(cluster) & set(neighbors)) for cluster in cluster_list_backbone]
        # nr_in_cluster_relative = [nr_in_cluster[i]/len_cluster_list_backbone[i] if nr_in_cluster[i]>number_neighbors/10 else 0 for i in range(len(nr_in_cluster))]
    print('test')
