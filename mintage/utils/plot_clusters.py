import os

import numpy as np
from matplotlib import pyplot as plt

from utils.plot_functions import build_fancy_chain_plot
from utils.constants import COLORS_SCATTER


def plot_clusters_into_puckers(cluster_suites, cluster_list_mode, folder_plots, max_outlier_dist_percent):
    for name_pucker in ['c_3_c_3_suites', 'c_3_c_2_suites', 'c_2_c_3_suites', 'c_2_c_2_suites']:
        index_list, cluster_suites_p = determine_pucker_data(cluster_suites, name_pucker)

        procrustes_data_each_pucker = np.array([suite.procrustes_five_chain_vector for suite in cluster_suites_p])
        dihedral_angles_suites_p = np.array([suite.dihedral_angles for suite in cluster_suites_p])

        cluster_pucker = []
        for index in index_list:
            for i in range(len(cluster_list_mode)):
                for j in range(len(cluster_list_mode[i])):
                    if cluster_list_mode[i][j] == index:
                        cluster_pucker.append(i + 1)

        if not os.path.exists(folder + str(q_fold) + '/'):
            os.makedirs(folder + str(q_fold) + '/')  # range(min(cluster_pucker), max(cluster_pucker)+2)
        n, bins, edges = plt.hist(cluster_pucker, bins=range(1, 22),
                                  ec="black")
        plt.xticks(bins)
        # plt.xlim(1400)
        plt.savefig(folder + str(q_fold) + '/' + "hist_mode_" + name_pucker + ".png")
        plt.close()

        def listof_clusters(cluster_list):
            # put cluster in lists per index of the datapoint
            # cluster_list = f_cluster = Liste von cluster_indezes at data_index
            cluster_sorted2 = []
            for i in range(1, max(cluster_list) + 1):
                cluster_sorted2.append(
                    [index for c, index in zip(cluster_list, range(0, len(cluster_list))) if c == i])

            cluster_sorted = []
            for i in range(1, max(cluster_list) + 1):
                l = []
                for c, index in zip(cluster_list, range(0, len(cluster_list))):
                    if c == i:
                        l.append(index)
                cluster_sorted.append(l)
            print(f"in listof_clusters: cluster_lists the same? {cluster_sorted2 == cluster_sorted}")
            return cluster_sorted

        clusterindizes_list = listof_clusters(cluster_pucker)
        cluster_data_p, cluster_len_list_p = sort_data_into_cluster(dihedral_angles_suites_p, clusterindizes_list,
                                                                    min_cluster_size)

        # dict_cluster_len = np.array(collections.Counter(iter(cluster_pucker)).most_common(np.max(cluster_pucker)))
        plot_functions.my_scatter_plots(cluster_data_p,
                                        folder_plots + 'all' + "/" + "scatter_plot_" + str(
                                            max_outlier_dist_percent)
                                        + " qfold "
                                        + str(q_fold) + str(name_pucker),
                                        set_title="dihedral angles suites" + 'all',
                                        suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                      r'$\beta$',
                                                      r'$\gamma$', r'$\delta_{2}$'],
                                        number_of_elements=cluster_len_list_p, legend=True, s=30
                                        )


def plot_all_cluster_combinations(dihedral_angles_suites, cluster_list, folder_plots, name_, q_fold,
                                  max_outlier_dist_percent, mode=True):
    if mode:
        folder_plots_combi = folder_plots + name_ + "/" + str(q_fold) + "/" + "mode" + "/"
        filename = folder_plots_combi + name_ + "_mode" + "_outlier" + str(max_outlier_dist_percent) + "_qfold" \
                   + str(q_fold) + "_"
    else:
        folder_plots_combi = folder_plots + name_ + "/" + str(q_fold) + "/" + "pre_cluster" + "/"
        filename = folder_plots_combi + name_ + "_pre_cluster" + "_outlier" + str(max_outlier_dist_percent) + "_qfold" \
                   + str(q_fold) + "_"
    if not os.path.exists(folder_plots_combi):
        os.makedirs(folder_plots_combi)
    # plot of all cluster combis
    for i, cluster1 in enumerate(cluster_list):
        for j, cluster2 in enumerate(cluster_list):
            if len(cluster1) == len(cluster2):
                continue
            if i > j:
                continue
            plot_functions.my_scatter_plots(np.vstack([dihedral_angles_suites[cluster1],
                                                       dihedral_angles_suites[cluster2]]),
                                            filename=filename + str(i + 1) + "_" + str(j + 1),
                                            set_title="dihedral angles suites" + name_,
                                            suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
                                                          r'$\beta$',
                                                          r'$\gamma$', r'$\delta_{2}$'],
                                            number_of_elements=[len(cluster1), len(cluster2)], legend=True, s=45,
                                            color_numbers=[i + 1, j + 1], legend_with_clustersize=True,
                                            legend_titles=[i + 1, j + 1]
                                            )

def plot_cluster_combi():
    pass
    # liste = [2,3,6,7,8, 9]
    # data_to_plot_g = np.vstack([dihedral_angles_suites[cluster] for i, cluster in enumerate(cluster_list_merged) if i in liste])
    # cluster_len_list_g = [len(cluster) for i, cluster in enumerate(cluster_list_merged) if i in liste]
    # plot_functions.my_scatter_plots(data_to_plot_g,
    #                                      filename=folder_plots_qfold + name_ + "_mode_merged" + "_outlier"
    #                                               + str(max_outlier_dist_percent) + "_qfold" + str(q_fold) + "cluster4pb0b",
    #                                     set_title="dihedral angles suites" + name_,
    #                                      suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
    #                                                    r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
    #                                      number_of_elements=cluster_len_list_g, legend=True, s=45,
    #                                      legend_with_clustersize=True, legend_titles=[a+1 for a in liste], color_numbers=[a+1 for a in liste] )


    # path = ".out/c2c2_cluster_suite_comparison_2023oct.csv"
    # if name_ == "c2c2":
    #     path = "./out/c2c2_cluster_suite_comparison_2023oct.csv"
    # elif name_ == "c2c3":
    #     path = "./out/c2c3_cluster_suite_comparison_2023oct.csv"
    # elif name_ == "c3c2":
    #     path = "./out/c3c2_cluster_suite_comparison_2023oct.csv"
    # elif name_ == "c3c3":
    #     path = "./out/c3c3_cluster_suite_comparison_2023oct.csv"
    #
    # R_cluster_ = write_files.get_Richardsons_suites(cluster_suites, path)
    # R_cluster = R_cluster_.values()
    # cluster_len_list_R = [len(cluster) for cluster in R_cluster][5:8]
    # data_to_plot_R = None
    # for j, cluster in enumerate(R_cluster):
    #     if j not in [5,6,7]:
    #         continue
    #     temp = np.array([suite.dihedral_angles for suite in cluster])
    #     if data_to_plot_R is None:
    #         data_to_plot_R = temp
    #         continue
    #     data_to_plot_R = np.vstack([data_to_plot_R, temp])
    # plot_functions.my_scatter_plots(data_to_plot_R,
    #                                 filename=folder_plots_qfold + name_ + "_Richardson2" + "_4pb0b",
    #                                 set_title="dihedral angles suites" + name_,
    #                                 suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
    #                                               r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
    #                                 number_of_elements=cluster_len_list_R, legend=True, s=100,
    #                                 legend_with_clustersize=True, legend_titles=list(R_cluster_.keys())[5:8])


def plot_low_res(cluster_list, procrustes_data, procrustes_data_backbone, name_="", plot_string="",
                 folder_plots="./out/", plot_atoms=False):
    atom_color_matrix_gesamt = []
    line_color_matrix_gesamt = []
    cluster_number = 0
    for i in range(len(cluster_list)):
        line_colors = [COLORS_SCATTER[cluster_number], COLORS_SCATTER[cluster_number], COLORS_SCATTER[cluster_number],
                       COLORS_SCATTER[cluster_number], COLORS_SCATTER[cluster_number]]
        atom_colors = ['darkblue', 'steelblue', 'orange', 'steelblue', 'darkblue']
        atom_color_matrix = len(cluster_list[i]) * [atom_colors]
        line_color_matrix = len(cluster_list[i]) * [line_colors]
        # atom_color_matrix = atom_colors
        atom_color_matrix_gesamt = atom_color_matrix_gesamt + atom_color_matrix
        line_color_matrix_gesamt = line_color_matrix_gesamt + line_color_matrix
        cluster_number = cluster_number + 1

    data_to_plot2 = np.vstack([procrustes_data[cluster] for cluster in cluster_list])
    build_fancy_chain_plot(data_to_plot2, filename=folder_plots + name_ + "_fivechain" + plot_string,
                           plot_atoms=plot_atoms, atom_color_matrix=atom_color_matrix_gesamt, without_legend=True,
                           colors=np.array(line_color_matrix_gesamt)[:, 0])

    data_to_plot3 = np.vstack([procrustes_data_backbone[cluster] for cluster in cluster_list])
    build_fancy_chain_plot(data_to_plot3, filename=folder_plots + name_ + "_sixchain" + plot_string,
                           plot_atoms=plot_atoms, atom_color_matrix=atom_color_matrix_gesamt, without_legend=True,
                           colors=np.array(line_color_matrix_gesamt)[:, 0])

    cluster_len_list = [len(cluster) for cluster in cluster_list]
    # write_files.data_to_cluster(data_to_plot2, cluster_len_list, name=name_ + '_low_res_clustered_suites2.txt')
    # write_files.data_to_cluster(data_to_plot3, cluster_len_list, name=name_ + '_high_res_clustered_suites2.txt')
