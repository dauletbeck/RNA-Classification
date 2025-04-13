import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import svm, discriminant_analysis
from numpy import linalg as la
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, fcluster, average
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from collections import Counter

# from clean_mintage_code import plot_functions
# from clean_mintage_code.PNDS_PNS import torus_mean_and_var


# import plot_functions
# from PNDS_PNS import torus_mean_and_var

from mintage_pipeline.utils import plot_functions
from mintage_pipeline.pnds.PNDS_io import find_files, import_csv, import_lists, export_csv
from mintage_pipeline.pnds.PNDS_plot import scatter_plots, var_plot, inv_var_plot, residual_plots

# from clean_mintage_code.PNDS_RNA_clustering import new_multi_slink

# from clean_mintage_code.shape_analysis import distance_matrix
# Due to circular import, here the copy:
# from clean_mintage_code.constants import MARKERS, COLORS_SCATTER
from constants import MARKERS, COLORS_SCATTER

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


def tangent_SVM_PCA(cluster1, cluster2, plot_folder, method_type='LDA'):
    """
    Input: two clusters which appear as they should be separated but are not at the moment
    performs a tangent svm and then pca
    :param cluster1: 7 dim torus data (dihedrals)
    :param cluster2: 7 dim torus data (dihedrals)
    :param plot_folder: folder for plots
    :param method_type: string: 'LDA' or 'SVM'
    """
    # Input: two clusters which should be separated but are not atm
    # cluster1 = dihedral_angles_suites[cluster_list_mode[7]]
    # cluster2 = dihedral_angles_suites[cluster_list_mode[8]]

    # x_j = 7 dim torus data (dihedrals)
    # x_j^orthogonal = x_j - (x_j * w) w
    X = np.vstack([cluster1, cluster2])
    y = np.zeros(len(cluster1))
    y = np.append(y, np.ones(len(cluster2)))

    if method_type == "SVM":
        clf = svm.SVC(kernel="linear")
    else:
        # type == LDA
        clf = discriminant_analysis.LinearDiscriminantAnalysis()

    clf = clf.fit(X, y)

    w = clf.coef_[0]  # trennvector
    w_norm = np.linalg.norm(w)
    w_direction = w / w_norm
    b = clf.intercept_[0]
    # Decision function values are np.dot(X, w) + b (note that the maximal difference is tiny)
    print(np.max(np.dot(X, w) + b - clf.decision_function(X)), 'is zero up to numerical error.')

    # Project data onto the orthogonal complement of the separating direction
    X_ortho = X - np.dot(X, w_direction)[:, np.newaxis] * w_direction[np.newaxis, :]
    print(np.max(np.dot(X_ortho, w)), 'is zero up to numerical error.')

    # Now, perform PCA of the projected data.
    pca = sklearnPCA()  # n_components
    pca_scores = pca.fit_transform(X_ortho)

    X_w = np.dot(X, w_direction) + b / w_norm
    X_pc1 = pca_scores[:len(X), 0]

    # all plots
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(X_w, X_pc1, c=y)
    axs[0, 0].set_title('X_pc1 vs X_w')
    axs[0, 1].hist(X_pc1)
    axs[0, 1].set_title('X_pc1')
    axs[1, 0].hist(X_w)
    axs[1, 0].set_title('X_w')
    plt.savefig(plot_folder + "all_X_w_pc1.png")
    plt.close()

    plt.scatter(X_w, X_pc1, c=y)
    plt.savefig(plot_folder + "scatter_pc1_Xw.png")
    plt.close()

    # das wollen wir:
    plt.hist(X_w)
    plt.savefig(plot_folder + "hist_X_w.png")
    plt.xlabel("X_w")
    plt.close()

    # das nicht
    plt.hist(X_pc1)
    plt.savefig(plot_folder + "hist_pc1.png")
    plt.close()

    # plot the 2 clusters for referenz
    plot_functions.my_scatter_plots(X, filename=plot_folder + "2clusterplot" + method_type,
                                    set_title="dihedral angles suites",
                                    suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                  r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
                                    number_of_elements=[len(cluster1), len(cluster2)], legend=True, s=50,
                                    legend_with_clustersize=True, color_numbers=[2, 8],
                                    legend_titles=[1, 2])


def large_cluster_separation(cluster, cluster_indices, plot_names="", plot=True):
    """
    Separates one cluster if necessary
    :param cluster: dihedral_angles_suites[cluster_list_mode[0]], ONE cluster to divide
    :param cluster_indices: list of indices of param cluster
    :param plot_names: string, additional name for plots
    :param plot: boolean, if plot: plots to folder './out/cluster_separation/'
    """
    plot_folder = './out/cluster_separation/'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # :param min_cluster_size: int, minimal cluster size for mode_hunting
    min_cluster_size = 3
    max_iter = 3

    # cluster1 = dihedral_angles_suites[cluster_list_mode[0]]

    # 1. S_tang = Tangentialraum-Kovarianz am Torus-Frechet-Mittelwert mean_torus.
    # 2. X_centered = X - mean_torus[np.newaxis,: ]
    # distances = np.einsum('nj,jk,nk->n', X_centered, np.linalg.inv(S_tang), X_centered)
    # this means: X_j^T (S_tang)^{-1} X_j
    # 3A. X_core = X[distances<3.75]
    # 3B. mean_core ist Torus-Frechet-mean of X_core and S_core ist Tangentialraum-Kovarianz von X_core.
    # 4. distances_core ... wie oben nur mit S_core aber für alle X_centered!
    # 5. X_main_cluster = X[distances_core < sqrt(chi_square(7).quantile(0.999))]
    # Evtl. auch
    # X_main_cluster = X[distances_core < 7.5]
    # 3B. - 5. iterieren

    def torus_distances(p, q):
        d = np.abs(p - q)
        d[d > 180] -= 360
        return la.norm(d, axis=1)

    def sum_torus_dists(data, x):
        return sum((torus_distances(x, data[:])) ** 2)

    def frechet_mean(data):
        # Zufällige Startposition für die Optimierung?
        initial_guess = data[0]  # np.random.rand(7)
        # Minimierung
        result = minimize(sum_torus_dists, initial_guess, args=(data,))
        frechet_mean = result.x
        return frechet_mean

    X = cluster
    # 1.
    # mean_torus = frechet_mean(X)
    mean_torus, _ = torus_mean_and_var(X, 360)
    mean_torus = mean_torus % 360
    X_centered = (X - mean_torus[np.newaxis, :] + 180) % 360 - 180
    # X_centered = X_centered % 360

    # Tangentialraum-Kovarianz am Torus-Frechet-Mittelwert mean_torus
    # 7x7 matrix
    # S_tang = np.cov(X_centered.T)
    S_tang = np.dot(X_centered.T, X_centered) / (len(X_centered) - 1)

    # only debug:
    eigenvalues, eigenvectors = np.linalg.eig(S_tang)
    # print(f'eigenvalues S_tang: {eigenvalues}, sqrt: {np.sqrt(eigenvalues)}')

    # 2. Berechnung der quadrierten Abstände von jedem Punkt X_j zu mean_torus im Tangentialraum
    # diagonal of X_j^T (S_tang)^{-1} X_j
    S_tang_inverse = np.linalg.inv(S_tang)
    distances = np.einsum('nj,jk,nk->n', X_centered, S_tang_inverse, X_centered)

    # 3A.
    threshold = 7.5 # (3.75 ** 2)#  #   # 6]#
    X_core = X[distances < threshold]
    if plot:
        scatter_plots_ellipses(X, S_tang_inverse, mean_torus,
                               filename=plot_folder + plot_names + "ellipses_X_core_0" + f't{round(threshold, 1)}',
                               number_of_elements=[len(X)], s=15, legend_with_clustersize=True,
                               suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                 r'$\beta$', r'$\gamma$', r'$\delta_{2}$'], scale=threshold)

    for k in range(max_iter):
        X = X_core
        mean_torus2, _ = torus_mean_and_var(X, 360)
        mean_torus2 = mean_torus2 % 360
        X_centered_2 = (X - mean_torus2[np.newaxis, :] + 180) % 360 - 180

        S_tang = np.dot(X_centered_2.T, X_centered_2) / (len(X_centered_2) - 1)
        # eigenvalues, eigenvectors = np.linalg.eig(S_tang)
        # print(f'eigenvalues S_tang: {eigenvalues}, sqrt: {np.sqrt(eigenvalues)}')

        S_tang_inverse2 = np.linalg.inv(S_tang)
        distances2 = np.einsum('nj,jk,nk->n', X_centered_2, S_tang_inverse2, X_centered_2)
        threshold = (3.75 ** 2)

        if plot:
            scatter_plots_ellipses(X_core, S_tang_inverse2, mean_torus2,
                                   filename=plot_folder + plot_names + f"ellipses_X_core_{k+1}"+ f't{round(threshold, 1)}',
                                   number_of_elements=[len(X_core)], s=15, legend_with_clustersize=True,
                                   suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                 r'$\beta$', r'$\gamma$', r'$\delta_{2}$'], scale=threshold
                                   )

        X_core = X[distances2 < threshold]

        if len(X) == len(X_core):
            break

    # fast break
    if len(X_core)/len(cluster) >= 0.99:
        return cluster_indices, []

    if plot:
        plot_functions.my_scatter_plots(X_core, filename=plot_folder + plot_names + "clusterplot_Xcore",
                                        set_title="dihedral angles suites",
                                        suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                      r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
                                        number_of_elements=[len(X_core)], legend=True, s=50,
                                        legend_with_clustersize=True)

    # for i in range(0, 4):
    # 3B. Berechnung des Torus-Frechet-Mittelwerts mean_core für X_core
    mean_core, _ = torus_mean_and_var(X_core, 360)
    mean_core = mean_core % 360

    # Berechnung der Tangentialraum-Kovarianz S_core für X_core
    X_core_centered = (X_core - mean_core[np.newaxis, :] +180) % 360 -180
    # since X core does not contain all datapoints, we need a new var for calculation of the distances:
    X_centered_to_core = (cluster - mean_core[np.newaxis, :] +180) % 360 -180
    S_core = np.dot(X_core_centered.T, X_core_centered) / (len(X_core_centered) - 1)
    S_core_inverse = np.linalg.inv(S_core)

    # 4. Berechnung der quadrierten Abstände von jedem Punkt X_j zu mean_core im Tangentialraum
    distances_core = np.einsum('nj,jk,nk->n', X_centered_to_core, S_core_inverse, X_centered_to_core)

    # 5.
    # threshold = np.sqrt(chi2.ppf(0.999, df=7))
    threshold = 7.5
    X_main_cluster = cluster[distances_core < threshold ** 2]
    X_leftover_clusters = cluster[distances_core >= threshold ** 2]

    if plot:
        scatter_plots_ellipses(cluster, S_core_inverse, mean_core,
                               filename=plot_folder + plot_names + "ellipses_main_cluster" + f't{round(threshold, 1)}',
                               number_of_elements=[len(cluster)], s=15, legend_with_clustersize=True,
                               suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                             r'$\beta$', r'$\gamma$', r'$\delta_{2}$'], scale=threshold**2)

        plot_functions.my_scatter_plots(X_main_cluster, filename=plot_folder + plot_names + "clusterplot_main",
                                        set_title="dihedral angles suites",
                                        suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                      r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
                                        number_of_elements=[len(X_main_cluster)], legend=True, s=50,
                                        legend_with_clustersize=True)

        if len(X_leftover_clusters) > 0:
            plot_functions.my_scatter_plots(X_leftover_clusters,
                                            filename=plot_folder + plot_names + "clusterplot_leftover",
                                            set_title="dihedral angles suites",
                                            suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                          r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
                                            number_of_elements=[len(X_leftover_clusters)], legend=True, s=50,
                                            legend_with_clustersize=True)

    if len(X_leftover_clusters) <= min_cluster_size:
        return cluster_indices, []

    # return X_main_cluster, X_leftover_clusters
    X_main_indices = cluster_indices[distances_core < threshold ** 2]
    X_leftover_indices = cluster_indices[distances_core >= threshold ** 2]
    return X_main_indices, X_leftover_indices

    ## TODO
    cluster_list_left = [range(len(X_leftover_clusters))]
    cluster_list_rest, noise2 = new_multi_slink(scale=12000, data=X_leftover_clusters,
                                                cluster_list=cluster_list_left,
                                                min_cluster_size=min_cluster_size)

    cluster_len_list_left = [len(cluster) for cluster in cluster_list_rest]
    data_to_plot_rest = np.vstack([X_leftover_clusters[cluster] for cluster in cluster_list_rest])
    plot_functions.my_scatter_plots(data_to_plot_rest,
                                    filename=plot_folder + plot_names + "leftover_mode" + "_outlier"
                                             + str(max_outlier_dist_percent) + "_qfold" + str(q_fold),
                                    set_title="dihedral angles suites",
                                    suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                  r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
                                    number_of_elements=cluster_len_list_left, legend=True, s=45,
                                    legend_with_clustersize=True)

    X0_index = cluster_list_mode[0][distances_core < threshold ** 2]
    Xrest_index = cluster_list_mode[0][distances_core >= threshold ** 2]
    X_rest_gesamt_indizes = np.hstack([Xrest_index[cluster] for cluster in cluster_list_rest])

    cluster_len_list = [len(cluster) for cluster in cluster_list_mode]
    cluster_list_mode_copy = cluster_list_mode[:]
    cluster_list_mode_copy[0] = X0_index
    cluster_list_mode_copy.append(X_rest_gesamt_indizes)
    data_to_plot = np.vstack([dihedral_angles_suites[cluster] for cluster in cluster_list_mode_copy])
    cluster_len_list[0] = len(X0_index)
    cluster_len_list_left = [len(cluster) for cluster in cluster_list_rest]

    for i in cluster_len_list_left:
        cluster_len_list.append(i)

    plot_functions.my_scatter_plots(data_to_plot,
                                    filename=plot_folder + plot_names + "mode" + "_outlier"
                                             + str(max_outlier_dist_percent) + "_qfold" + str(q_fold),
                                    set_title="dihedral angles suites",
                                    suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                  r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
                                    number_of_elements=cluster_len_list, legend=True, s=45,
                                    legend_with_clustersize=True)


def cluster_merging(cluster_index_lists, dihedral_angles, plot=True):
    plot_folder = './out/cluster_merging/'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Step 1:
    # d_mean = Abstand der Cluster-Mittelpunkte,
    # d_min = kleinsten Abstand zwischen Punkten aus Cluster 1 vs. Punkten aus Cluster 2,
    # falls d_min > 0.5 d_mean, nächstes Cluster-Paar
    #
    # Alternative, falls hier zu viel ausgesiebt wird:
    # Cluster auf Verbindungslinie der Mittelpunkte projizieren.
    #
    # Step 2: Average linkage clustering,
    # erste Gabelung bei der jeder Ast mindestens 0.75 * Größe des kleineren Clusters an Punkten hat,
    # wieviel % von jedem Cluster enthält jeder Ast?
    # Jeder Cluster muss zu mindestens 75% in einem der Äste enthalten sein,
    # und zwar für die beiden Cluster in verschiedenen Ästen.

    def torus_distances(p, q):
        d = np.abs(p - q)
        d[d > 180] -= 360
        return la.norm(d, axis=1)

    def get_dendrogram(index):
        c_0 = int(linkage_matrix[index, 0])
        c_1 = int(linkage_matrix[index, 1])
        return c_0, c_1

    cluster_list = [dihedral_angles[cluster_index_l] for cluster_index_l in cluster_index_lists]
    cluster_list_return = cluster_index_lists[:]
    merge_list = []
    # Step 1:
    # calculate Torus mean for all Cluster
    cluster_means = [torus_mean_and_var(cluster, 360)[0] for cluster in cluster_list]
    # print(f'cluster means min, max:{min(cluster_means)}, {max(cluster_means)}')
    # each cluster: 7 dim mean -180 to 180 degree

    for i in range(len(cluster_means)):
        for j in range(len(cluster_means)):
            if i >= j:
                continue
            # torus distance between the two cluster means:
            d_mean = np.abs(cluster_means[i] - cluster_means[j])
            d_mean[d_mean > 180] -= 360
            d_mean = la.norm(d_mean)

            # get minimal distance between all points of cluster i and all points of cluster j
            cluster_point_dists = [torus_distances(p1, cluster_list[j]) for p1 in cluster_list[i]]
            d_min = min([min(cluster_point_dists[l]) for l in range(0, len(cluster_point_dists))])
            print(f'{max([max(cluster_point_dists[l]) for l in range(0, len(cluster_point_dists))])}')

            print(f'cluster {i + 1} and {j + 1} dmin: {d_min} <= 0.5 * dmean: {d_mean}? {d_min / d_mean}')
            print(f'cluster means: {cluster_means[i]}, {cluster_means[j]}')
            if d_min <= 0.55 * d_mean:  # 0.5 * d_mean:
                print(f"merging cluster {i + 1} and cluster {j + 1}?")

                # Step 2:
                cluster1 = cluster_list[i]
                cluster2 = cluster_list[j]
                pairdist = distance_matrix(np.vstack([cluster1, cluster2]))
                linkage_matrix = average(pairdist)  # dendrogram
                # linkage_matrix = linkage(pairdist, method='average')  # =Z
                if plot:
                    dn = dendrogram(linkage_matrix)
                    plt.savefig(plot_folder + f"{i + 1}_{j + 1}_dendrogram.png")
                    plt.close()

                # At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n+i.
                # The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]
                # Z[i, 3] represents the number of original observations in the newly formed cluster

                # tree starts at root (all points in one cluster)
                # erste Gabelung bei der jeder Ast mindestens 0.75 * Größe des kleineren Clusters an Punkten hat,
                save_list0 = []
                save_list1 = []
                thresh = 0.75 * min([len(cluster1), len(cluster2)])
                n = len(cluster1) + len(cluster2)
                all75 = False
                for k in range(len(linkage_matrix) - 1, -1, -1):  # count backwards from len(linkage_matrix)-1 to 0
                    # dist = linkage_matrix[k, 2]
                    # clustering = fcluster(linkage_matrix, dist, criterion='distance')
                    # counter_c = Counter(clustering)
                    if linkage_matrix[k, 3] >= thresh * 2:
                        c_index0 = int(linkage_matrix[k, 0] - n)
                        c_index1 = int(linkage_matrix[k, 1] - n)

                        if (linkage_matrix[c_index0, 3] >= thresh) and (linkage_matrix[c_index1, 3] >= thresh):
                            dist = linkage_matrix[c_index1, 2]  # not used
                            all75 = True

                            # Get one part of the cluster tree
                            work_list = [c_index0]
                            while len(work_list) > 0:
                                i1, i2 = get_dendrogram(work_list.pop())
                                if i1 < n:
                                    save_list0.append(i1)
                                else:
                                    work_list.append(i1 - n)
                                if i2 < n:
                                    save_list0.append(i2)
                                else:
                                    work_list.append(i2 - n)
                            work_list = [c_index1]
                            while len(work_list) > 0:
                                i1, i2 = get_dendrogram(work_list.pop())
                                if i1 < n:
                                    save_list1.append(i1)
                                else:
                                    work_list.append(i1 - n)
                                if i2 < n:
                                    save_list1.append(i2)
                                else:
                                    work_list.append(i2 - n)

                            break
                print(f'all 75%? {all75}')
                if all75:
                    # wieviel % von jedem Cluster enthält jeder Ast?
                    # Jeder Cluster muss zu mindestens 75% in einem der Äste enthalten sein,
                    # und zwar für die beiden Cluster in verschiedenen Ästen.
                    c1_index_list_converted = [i for i in range(0, len(cluster1))]
                    c2_index_list_converted = [i for i in range(len(cluster1), len(cluster1) + len(cluster2))]

                    compare_c1_s0 = len([i for i in save_list0 for j in c1_index_list_converted if i == j])
                    compare_c2_s0 = len([i for i in save_list0 for j in c2_index_list_converted if i == j])

                    compare_c1_s1 = len([i for i in save_list1 for j in c1_index_list_converted if i == j])
                    compare_c2_s1 = len([i for i in save_list1 for j in c2_index_list_converted if i == j])

                    remnant_branch = [i for i in range(0, len(cluster1) + len(cluster2)) if
                                      i not in (save_list0 + save_list1)]
                    compare_c1_rem = len([i for i in remnant_branch for j in c1_index_list_converted if i == j])
                    compare_c2_rem = len([i for i in remnant_branch for j in c2_index_list_converted if i == j])

                    divided = False
                    # cluster 1 is in left branch and cluster 2 in right
                    if (compare_c1_s0 >= 0.75 * len(cluster1)) and (compare_c2_s1 >= 0.75 * len(cluster2)):
                        divided = True
                    # cluster 2 is in left branch and cluster 1 in right
                    elif (compare_c1_s1 >= 0.75 * len(cluster1)) and (compare_c2_s0 >= 0.75 * len(cluster2)):
                        divided = True
                    # if we have very sparse cluster: Test if sparse cluster is 75% in tree-remnant
                    elif (compare_c1_rem >= 0.75 * len(cluster1)) and (
                            compare_c2_s0 + compare_c2_s1 >= 0.75 * len(cluster2)):
                        divided = True
                    elif (compare_c2_rem >= 0.75 * len(cluster2)) and (
                            compare_c1_s0 + compare_c1_s1 >= 0.75 * len(cluster1)):
                        divided = True

                    if not divided:
                        print("MERGING")
                        # cluster_list_return = np.delete(cluster_list_return, i)
                        # cluster_list_return = np.delete(cluster_list_return, j - 1)
                        if cluster_list_return[i] is None:
                            for element in merge_list:
                                if i in element:
                                    element.append(i)
                                    break
                        elif cluster_list_return[j] is None:
                            for element in merge_list:
                                if j in element:
                                    element.append(j)
                                    break
                        else:
                            merge_list.append([i, j])

                        cluster_list_return[i] = None
                        cluster_list_return[j] = None
                        # print(f"type: {type(cluster_index_lists[i])}. {type(cluster_index_lists[j])}")
                        # to_append = np.append(cluster_index_lists[i], cluster_index_lists[j])
                        # temp = [e for e in cluster_list_return]
                        # temp.append(to_append)
                        # cluster_list_return = temp  # np.array(temp, dtype=object)
                        # cluster_list_return = np.append(cluster_list_return, np.append(cluster_index_lists[i], cluster_index_lists[j]))
                    else:
                        print("no merging")

                    if plot:
                        plot_functions.my_scatter_plots(np.vstack([cluster1, cluster2]),
                                                        filename=plot_folder + f"{i + 1}_{j + 1}_start_clusterplot_"
                                                                 + f'ratio_{round(d_min / d_mean, 3)}',
                                                        set_title="dihedral angles suites",
                                                        suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$',
                                                                      r'$\alpha$', r'$\beta$', r'$\gamma$',
                                                                      r'$\delta_{2}$'],
                                                        number_of_elements=[len(cluster1), len(cluster2)], legend=True,
                                                        s=50, legend_with_clustersize=True,
                                                        legend_titles=[i + 1, j + 1],
                                                        color_numbers=[2, 8])

                        clustering = fcluster(linkage_matrix, dist, criterion='distance')
                        data = np.vstack([cluster1, cluster2])
                        tmp = []
                        len_list = []
                        for it in range(1, max(clustering) + 1):
                            t = [index for index, cluster_num in enumerate(clustering) if cluster_num == it]
                            tmp.append(np.array(t))
                            len_list.append(len(t))

                        try:
                            tmp = np.array(tmp, dtype=object).astype(int)
                        except:
                            tmp = np.array(tmp, dtype=object)
                        # data_to_plot = np.array([data[tmp[l]] for l in range(len(tmp))], dtype=object)
                        data_to_plot = data[tmp[0]]
                        for l in range(1, len(tmp)):
                            data_to_plot = np.vstack([data_to_plot, data[tmp[l]]])
                        plot_functions.my_scatter_plots(data_to_plot,
                                                        filename=plot_folder + f"{i + 1}_{j + 1}_average_clusterplot_"
                                                                 + f'merging_{not divided}',
                                                        set_title="dihedral angles suites",
                                                        suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$',
                                                                      r'$\alpha$', r'$\beta$', r'$\gamma$',
                                                                      r'$\delta_{2}$'],
                                                        number_of_elements=len_list, legend=True,
                                                        s=50, legend_with_clustersize=True)

                        data_to_plot2 = np.append(save_list0, save_list1)
                        data_to_plot2 = data[data_to_plot2]
                        plot_functions.my_scatter_plots(data_to_plot2,
                                                        filename=plot_folder + f"{i + 1}_{j + 1}_savelists_clusterplot_"
                                                                 + f'merging_{not divided}',
                                                        set_title="dihedral angles suites",
                                                        suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$',
                                                                      r'$\alpha$', r'$\beta$', r'$\gamma$',
                                                                      r'$\delta_{2}$'],
                                                        number_of_elements=[len(save_list0), len(save_list1)],
                                                        legend=True,
                                                        s=50, legend_with_clustersize=True)

                    # clustering = fcluster(linkage_matrix, dist, criterion='distance')
                    # counter_clusters = Counter(clustering[0:len(cluster1)])
                    # # are the points in cluster1 to over 75% in one cluster?
                    # divided = False
                    # for key in counter_clusters.keys():
                    #     if counter_clusters[key] >= 0.75 * len(cluster1):
                    #         divided = True
                    # if not divided:
                    #     print("MERGING")
                    # else:
                    #     divided = False
                    #     counter_clusters = Counter(clustering[len(cluster1):])
                    #     # are the points in cluster2 to over 75% in one cluster?
                    #     for key in counter_clusters.keys():
                    #         if counter_clusters[key] >= 0.75 * len(cluster2):
                    #             divided = True
                    #     if not divided:
                    #         print("MERGING")
                    #     else:
                    #         print("no merging")

    cluster_list_return = [e for e in cluster_list_return if e is not None]
    for element in merge_list:
        to_append = np.array([])
        for index in element:
            to_append = np.append(to_append, cluster_index_lists[index])
        cluster_list_return.append(to_append.astype(int))

    cluster_list_return = sorted(cluster_list_return, key=len, reverse=True)
    return cluster_list_return


def ellipse_function(a, b, c, phi, scale=3.75):
    """
    Debug-plot for large_cluster_separation()
    """
    phi = phi * np.pi / 180.
    # r_phi = a - (a - c) * (np.cos(phi)) ** 2 + 2 * b * np.sin(phi) * np.cos(phi)
    r_phi = a*(np.sin(phi)**2) + c*(np.cos(phi)**2) + 2 * b * np.sin(phi) * np.cos(phi)
    if r_phi <= 0:
        return None
    r_phi = np.sqrt(scale) / np.sqrt(r_phi)
    return r_phi


def ellipse_plot(covariance_matrix, data, means, plot_folder, i, j):
    plt.rcParams.update(plt.rcParamsDefault)
    # i = 0
    # j = 6
    plt.scatter(data[:, i], data[:, j], c="black", s=2)
    a = covariance_matrix[i, i]
    b = covariance_matrix[i, j]
    c = covariance_matrix[j, j]
    plot_array_phi = [x for x in range(0, 360, 1)]
    plot_array_phi_radians = np.radians(plot_array_phi)
    plot_array_r = ([ellipse_function(a, b, c, p) for p in plot_array_phi])
    plot_array_x = [means[i] + r * np.sin(phi)  for phi, r in zip(plot_array_phi_radians, plot_array_r)] # % 360
    plot_array_y = [means[j] + r * np.cos(phi)  for phi, r in zip(plot_array_phi_radians, plot_array_r)] # % 360

    # jumps = np.where(np.abs(plot_array_x[1:]-plot_array_x[:-1]) > 300)[0]
    plt.scatter(plot_array_x, plot_array_y, c="green", s=1, marker='o', alpha=0.5)
    plt.xlim([0, 360])
    plt.ylim([0, 360])
    plt.savefig(plot_folder + f'ellipse_plots_{i}{j}')
    plt.close()

def plot_all_ellipses(covariance_matrix, data, means, plot_folder):
    plt.rcParams.update(plt.rcParamsDefault)
    fig, axs = plt.subplots(7, 7)
    # fig = plt.figure()
    for i in range(0, 7):
        k = 0
        for j in range(6, 0, -1):
            if i > j:
                print(f"{i}, {j}")
                axs[i, j].axis('off')
                continue
                #
            if i == j:
                continue

            # axs = fig.add_subplot(1, 1,(i, k))
            axs[i, k].scatter(data[:, i], data[:, j], c="black", s=2)
            # covariance_matrix = covariance_matrix % 360
            a = covariance_matrix[i, i]
            b = covariance_matrix[i, j]
            c = covariance_matrix[j, j]
            plot_array_phi = [x for x in range(0, 360, 1)]
            plot_array_phi_radians = np.radians(plot_array_phi)
            plot_array_r = ([ellipse_function(a, b, c, p) for p in plot_array_phi])
            plot_array_x = [means[i] + r * np.sin(phi) for phi, r in zip(plot_array_phi_radians, plot_array_r)]
            plot_array_y = [means[j] + r * np.cos(phi) for phi, r in zip(plot_array_phi_radians, plot_array_r)]

            # jumps = np.where(np.abs(plot_array_x[1:]-plot_array_x[:-1]) > 300)[0]
            axs[i, k].scatter(plot_array_x, plot_array_y, c="green", s=1, marker='o', alpha=0.5)
            axs[i, k].set_xlim([0, 360])
            axs[i, k].set_ylim([0, 360])
            k = k+1

    plt.savefig(plot_folder + f'ALL')
    plt.close()

    arr = plt.imread(plot_folder + f'ALL' + '.png')
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
    for list in list_2:
        if len(list) > 75:
            for element in list[74:]:
                remove_list.append(element)
    not_remove_list = [i for i in range(arr_new.shape[1]) if i not in remove_list]
    arr_new = arr_new[:, not_remove_list, :]

    # "ValueError: ndarray is not C-contiguous" else
    arr_new = arr_new.copy(order='C')
    plt.imsave(plot_folder + f'ALL' + '.png', arr_new)


def scatter_plots_ellipses(input_data, covariance_matrix, means, filename=None, axis_min=0, axis_max=360, set_title=None, number_of_elements=None,
                     suite_titles=None, alpha_first=1, s=5, all_titles=False, fontsize=40, legend=True,
                     legend_with_clustersize=False, color_numbers=None, legend_titles=None, markerscale=5, scale=3.75):
    """
    This function gets input data and creates scatter plots.
    :param legend_titles: needs to be list of len number_of_elements
    :param color_numbers: needs to be list of len number_of_elements
    :param input_data: A matrix of dimension (number of data_points) x (number of dimensions)
    :param filename: Should be a string indicating where the plot should be stored and what the plot should be named.
    :param axis_min: The minimum of the range.
    :param axis_max: The maximum of the range.
    :param set_title:
    :param legend: Boolean: If False: without legend.
    :param number_of_elements: If you have more than one group of data.
    """

    fig = plt.figure()
    n = input_data.shape[1]
    size = fig.get_size_inches()

    if color_numbers is None and number_of_elements is not None:
        color_numbers = range(len(number_of_elements))
    else:
        color_numbers = [n - 1 for n in color_numbers]

    # To avoid overlapping of the plots:
    if n > 3:
        fig.set_size_inches((1.2 * size[0] * (n - 1), 1.2 * size[1] * (n - 1)))
        # fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95, wspace=-0.8, hspace=0.4)
    else:
        fig.set_size_inches((1.2 * size[0] * n, 1.2 * size[1] * n))
        # fig.subplots_adjust(left=-0.8, bottom=0.1, right=1.7, top=0.95, wspace=-0.8, hspace=0.4)
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            if suite_titles is None:
                if set_title is None:
                    diag.set_title(r'$x = \alpha_' + str(x + 1) +
                                   r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
                else:
                    diag.set_title(set_title + str(x + 1) + ', ' +
                                   set_title + str(y + 1), fontsize=20)
            else:
                # diag.set_title('x-axis is ' + suite_titles[x] + ', ' + 'y axis is' + suite_titles[y], fontsize=20)
                if y == n - 1 or all_titles:
                    if not all_titles:
                        diag.set_title(suite_titles[x], fontsize=fontsize)
                    else:
                        diag.set_xlabel(suite_titles[x], fontsize=fontsize)
                if x == 0 or all_titles:
                    diag.set_ylabel(suite_titles[y], fontsize=fontsize)

            if axis_min is not None:
                diag.set_aspect('equal')
                diag.set_xlim(axis_min, axis_max)
                diag.set_ylim(axis_min, axis_max)
            else:
                diffs = [np.abs(np.min(input_data[:, z]) - np.max(input_data[:, z])) for z in
                         range(input_data.shape[1])]
                diag.set_xlim((np.min(input_data[:, x]) + np.max(input_data[:, x])) / 2 - np.max(diffs) / 2 - 1,
                              (np.min(input_data[:, x]) + np.max(input_data[:, x])) / 2 + np.max(diffs) / 2 + 1)
                diag.set_ylim((np.min(input_data[:, y]) + np.max(input_data[:, y])) / 2 - np.max(diffs) / 2 - 1,
                              (np.min(input_data[:, y]) + np.max(input_data[:, y])) / 2 + np.max(diffs) / 2 + 1)
                # diag.set_xlim(np.min(input_data[:, x]), np.max(input_data[:, x]))
                # diag.set_ylim(np.min(input_data[:, y]), np.max(input_data[:, y]))
            # only one data set:
            if number_of_elements is None:
                diag.scatter(input_data[:, x], input_data[:, y], marker="D", linewidth=0.1, s=s, c='black')
            else:

                for number_element, number_color in zip(range(len(number_of_elements)), color_numbers):
                    if number_element == 0:
                        diag.scatter(input_data[:number_of_elements[number_element], x],
                                     input_data[:number_of_elements[number_element], y],
                                     c=COLORS_SCATTER[number_color], linewidth=0.1, s=s,
                                     alpha=alpha_first, marker=MARKERS[number_color])
                    else:
                        diag.scatter(input_data[sum(number_of_elements[:number_element]):
                                                sum(number_of_elements[:number_element + 1]), x],
                                     input_data[sum(number_of_elements[:number_element]):
                                                sum(number_of_elements[:number_element + 1]), y],
                                     c=COLORS_SCATTER[number_color], linewidth=0.1, s=s,
                                     marker=MARKERS[number_color])

            a = covariance_matrix[x, x]
            b = covariance_matrix[x, y]
            c = covariance_matrix[y, y]
            plot_array_phi = [x for x in range(0, 360, 1)]
            plot_array_phi_radians = np.radians(plot_array_phi)
            plot_array_r = [ellipse_function(a, b, c, p, scale) for p in plot_array_phi]
            plot_array_r = [r for r in plot_array_r if r is not None]
            plot_array_x = [means[x] + r * np.sin(phi) for phi, r in zip(plot_array_phi_radians, plot_array_r)]
            plot_array_y = [means[y] + r * np.cos(phi) for phi, r in zip(plot_array_phi_radians, plot_array_r)]

            diag.scatter(plot_array_x, plot_array_y, c="green", s=1, marker='o', alpha=0.5)

    if number_of_elements is not None and legend:
        x = n - 3
        y = n - 4

        if legend_titles is None:
            legend_titles = range(1, len(number_of_elements)+1)

        if legend_with_clustersize:
            # temp = [f'size: {l}' for l in number_of_elements]
            # legend_titles = zip(legend_titles, temp)
            legend_titles = [f"{c}, size: {l}" for c, l in zip(legend_titles, number_of_elements)]

        diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
        for i, j in zip(legend_titles, color_numbers):
            if not isinstance(i, str):
                i = str(i)
            if i == 0:
                plt.scatter([], [], c=COLORS_SCATTER[j], s=s, marker=MARKERS[j],
                             label='Class ' + i, alpha=alpha_first)
            else:
                plt.scatter([], [], c=COLORS_SCATTER[j], s=s, marker=MARKERS[j],
                             label='Class ' + i)
        diag.legend(loc='center', markerscale=markerscale, prop={"size": 30})
        diag.set_xticks([])
        diag.set_yticks([])
        diag.axis("off")


    if not (filename is None):
        plt.savefig(filename + '.png', dpi=150)
    else:
        plt.show()
    plt.close()
    arr = plt.imread(filename + '.png')
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
    for list in list_2:
        if len(list) > 75:
            for element in list[74:]:
                remove_list.append(element)
    not_remove_list = [i for i in range(arr_new.shape[1]) if i not in remove_list]
    arr_new = arr_new[:, not_remove_list, :]

    # "ValueError: ndarray is not C-contiguous" else
    arr_new = arr_new.copy(order='C')
    plt.imsave(filename + '.png', arr_new)
    plt.close()


# scatter_plots_ellipses(data, covariance_matrix, means, filename=plot_folder+"ALL", number_of_elements = [len(data)], s=10)


# Set up output folder for test results
test_output_folder = './test_cluster_separation_output/'
if not os.path.exists(test_output_folder):
    os.makedirs(test_output_folder)

# Generate synthetic data for clustering
cluster1 = np.random.rand(10, 7) * 360  # Cluster of 10 points in 7D space
cluster2 = np.random.rand(15, 7) * 360  # Cluster of 15 points in 7D space
cluster3 = np.random.rand(8, 7) * 360   # Cluster of 8 points in 7D space
clusters = [cluster1, cluster2, cluster3]

# Combine clusters into a dihedral_angles list
dihedral_angles = np.vstack([cluster1, cluster2, cluster3])
cluster_indices = [
    np.arange(0, 10),         # Indices for cluster1
    np.arange(10, 25),        # Indices for cluster2
    np.arange(25, 33)         # Indices for cluster3
]

# Test cluster_merging function
def test_cluster_merging():
    
    merged_clusters = cluster_merging(cluster_indices, dihedral_angles, plot=True)
    
    # Display results
    print("Merged Clusters Output:")
    for idx, cluster in enumerate(merged_clusters):
        print(f"Cluster {idx+1}: {cluster}")

test_cluster_merging()

# # Test large_cluster_separation function
# def test_large_cluster_separation():
#     from clean_mintage_code import large_cluster_separation  # Import your function

#     # Test with cluster1 for simplicity
#     cluster_indices, leftover_indices = large_cluster_separation(
#         cluster1, np.arange(len(cluster1)), plot_names="test_separation_", plot=True
#     )
    
#     print("Large Cluster Separation Output:")
#     print("Main Cluster Indices:", cluster_indices)
#     print("Leftover Cluster Indices:", leftover_indices)

# test_large_cluster_separation()


# Check if test output folder has the generated plots for visual confirmation
print("Test complete. Check", test_output_folder, "for plot outputs.")