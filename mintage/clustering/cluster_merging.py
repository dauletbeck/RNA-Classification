# -*- coding: utf-8 -*-
"""
Copyright (c) 2024, Franziska Hoppe

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
"""


import os

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy.cluster.hierarchy import dendrogram, fcluster, average, single
from scipy.spatial.distance import pdist

import plot_functions
from PNDS_PNS import torus_mean_and_var


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
            diff_one_dim = np.min((360 - diff_one_dim, diff_one_dim), axis=0) ** 2
            sum_dihedral_differences = sum_dihedral_differences + diff_one_dim
        return np.sqrt(sum_dihedral_differences)
    if distance == 'sphere':
        return np.arccos(1 - pdist(data, 'cosine'))
    if distance == 'euclidean':
        return pdist(data, 'euclidean')

"""
Step 1:
d_mean = distance of cluster centers,
d_min = lowest distance between points from different clusters,
if d_min > 0.5 d_mean, next pair of clusters

Step 2:
Average linkage clustering, consider all branchings, where each branch contains
at least 75% the number of points of the smaller cluster.
How many % of each cluster per branch?
Of either cluster there must be 75% of the points in separate branches.
"""
def cluster_merging(cluster_index_lists, dihedral_angles, folder, circular=False, plot=True):
    #plot_folder = './out/cluster_merging/'
    plot_folder = folder

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    def torus_distances(p, q):
        d = np.abs(p - q)
        d[d > 180] -= 360
        return la.norm(d, axis=1)

    def cluster_distances(p, q):
        return la.norm(q - p, axis=1)

    def get_dendrogram(index):
        c_0 = int(linkage_matrix[index, 0])
        c_1 = int(linkage_matrix[index, 1])
        return c_0, c_1

    cluster_list = [dihedral_angles[cluster_index_l] for cluster_index_l in cluster_index_lists]
    cluster_list_return = cluster_index_lists[:]
    merge_list = []
    # Step 1:
    # calculate Torus mean for all Cluster
    if circular:
        cluster_means = [torus_mean_and_var(cluster, 360)[0] for cluster in cluster_list]
    else:
        cluster_means = [np.mean(cluster, axis=0) for cluster in cluster_list]
    # each cluster: 7 dim mean -180 to 180 degree

    for i in range(len(cluster_means)):
        for j in range(len(cluster_means)):
            if i >= j:
                continue
            # torus distance between the two cluster means:
            d_mean = np.abs(cluster_means[i] - cluster_means[j])
            if circular:
                d_mean[d_mean > 180] -= 360
            d_mean = la.norm(d_mean)

            # get minimal distance between all points of cluster i and all points of cluster j
            if circular:
                cluster_point_dists = [torus_distances(p1, cluster_list[j]) for p1 in cluster_list[i]]
            else:
                cluster_point_dists = [cluster_distances(p1, cluster_list[j]) for p1 in cluster_list[i]]
            d_min = min([min(cluster_point_dists[l]) for l in range(0, len(cluster_point_dists))])
            print(f'{max([max(cluster_point_dists[l]) for l in range(0, len(cluster_point_dists))])}')

            print(f'cluster {i + 1} and {j + 1} dmin: {d_min} <= 0.5 * dmean: {d_mean}? {d_min / d_mean}')
            print(f'cluster means: {cluster_means[i]}, {cluster_means[j]}')
            if d_min <= 0.55 * d_mean:  # 0.5 * d_mean:
                print(f"merging cluster {i + 1} and cluster {j + 1}?")

                # Step 2:
                cluster1 = cluster_list[i]
                cluster2 = cluster_list[j]
                if circular:
                    pairdist = distance_matrix(np.vstack([cluster1, cluster2]))
                else:
                    pairdist = distance_matrix(np.vstack([cluster1, cluster2]), 'euclidean')
                linkage_matrix = single(pairdist)  # dendrogram
                if plot:
                    dn = dendrogram(linkage_matrix)
                    plt.savefig(plot_folder + f"{i + 1}_{j + 1}_dendrogram.png", dpi=300)
                    plt.close()
                # At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n+i.
                # The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]
                # Z[i, 3] represents the number of original observations in the newly formed cluster
                # tree starts at root (all points in one cluster)
                save_list0 = []
                save_list1 = []

                thresh = 0.75 * min([len(cluster1), len(cluster2)])
                n = len(cluster1) + len(cluster2)
                all75 = False
                for k in range(len(linkage_matrix) - 1, -1, -1):  # count backwards from len(linkage_matrix)-1 to 0
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
                    # How many % of each cluster per branch?
                    # Of either cluster there must be 75% of the points in separate branches.
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
                        merge_info = [-1, -1]
                        if cluster_list_return[i] is None:
                            for a, element in enumerate(merge_list):
                                if ((i in element) and (not j in element)):
                                    merge_info[1] = a
                                    break
                        elif cluster_list_return[j] is None:
                            for a, element in enumerate(merge_list):
                                if ((j in element) and (not i in element)):
                                    merge_info[0] = a
                                    break
                        else:
                            merge_list.append([i, j])

                        if min(merge_info) > -1:
                            merge_info = sorted(merge_info)
                            new_element = sorted(list(set(merge_list.pop(merge_info[1]) + merge_list.pop(merge_info[0]))))
                            merge_info.append(new_element)
                        else:
                            if merge_info[0] > -1:
                                merge_list[merge_info[0]].append(i)
                            elif merge_info[1] > -1:
                                merge_list[merge_info[1]].append(j)

                        cluster_list_return[i] = None
                        cluster_list_return[j] = None
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
    cluster_list_return = [e for e in cluster_list_return if e is not None]
    for element in merge_list:
        to_append = np.array([])
        for index in element:
            to_append = np.append(to_append, cluster_index_lists[index])
        cluster_list_return.append(to_append.astype(int))

    cluster_list_return = sorted(cluster_list_return, key=len, reverse=True)
    return cluster_list_return

def test_cluster_merging():
    # Set up mock data
    # Create some synthetic clusters in a 7-dimensional space
    cluster1 = np.random.rand(10, 7) * 360  # Cluster of 10 points
    cluster2 = np.random.rand(15, 7) * 360  # Cluster of 15 points
    cluster3 = np.random.rand(8, 7) * 360   # Cluster of 8 points

    # Combine clusters into a dihedral_angles list
    dihedral_angles = np.vstack([cluster1, cluster2, cluster3])

    # Define indices representing each cluster in the dihedral_angles array
    cluster_indices = [
        np.arange(0, 10),         # Indices for cluster1
        np.arange(10, 25),        # Indices for cluster2
        np.arange(25, 33)         # Indices for cluster3
    ]

    # Set a test output directory for plot files (optional)
    folder = "./test_cluster_merging_output/"
    
    # Run the cluster_merging function with circular=True and plot=False for simplicity
    merged_clusters = cluster_merging(cluster_indices, dihedral_angles, folder, circular=True, plot=True)
    
    # Print the results
    print("Merged Clusters Output:")
    for idx, cluster in enumerate(merged_clusters):
        print(f"Cluster {idx+1}: {cluster}")

# Run the test
test_cluster_merging()
