import os
import sys

import numpy as np
from scipy.cluster.hierarchy import average

sys.path.append('./clean_mintage_code/')
from utils import write_files
# from R_lab_clustering import create_R_csv
# from plot_clusters import plot_all_cluster_combinations, plot_low_res
from pucker_data_functions import get_suites_from_pdb, determine_pucker_data, procrustes_for_each_pucker, \
    sort_data_into_cluster, create_csv

# from clean_mintage_code import shape_analysis, plot_functions, PNDS_RNA_clustering
import shape_analysis
from utils import plot_functions
from pnds import PNDS_RNA_clustering
from clustering.cluster_improving import cluster_merging

cwd = os.getcwd()
new_wd = os.path.join(cwd, "clean_mintage_code")
os.chdir(new_wd)
sys.path.append(cwd)

suites = get_suites_from_pdb()
input_suites = suites[::]

folder = './out/newdata_without_hets_11_23/'
if not os.path.exists(folder):
    os.makedirs(folder)


# function for Clustering ##################################################################################

def cluster_pruned_rna(name_, min_cluster_size=20, max_outlier_dist_percent=0.15, q_fold=0.15, do_clustering=True):
    folder_plots = folder
    if not os.path.exists(folder_plots):
        os.makedirs(folder_plots)

    method_cluster = average

    # cluster_suites = [suite for suite in input_suites if suite.complete_suite]
    cluster_suites = [suite for suite in input_suites if suite.procrustes_five_chain_vector is not None
                      and suite.dihedral_angles is not None]
    # type atm to get only pdb ATM not heteroatoms
    cluster_suites = [suite for suite in cluster_suites if suite.atom_types == 'atm']
    print(f'semi-complete suites: {len(cluster_suites)}')

    _, cluster_suites = determine_pucker_data(cluster_suites, name_)
    print(f'{name_} suites: {len(cluster_suites)}')
    dihedral_angles_suites = np.array([suite.dihedral_angles for suite in cluster_suites])

    # ---------- Procrustes ---------------------
    # data rotated with Procrustes for WHOLE dataset
    procrustes_data = np.array([suite.procrustes_five_chain_vector for suite in cluster_suites])
    procrustes_data_backbone = np.array([suite.procrustes_complete_suite_vector for suite in cluster_suites])

    # build_fancy_chain_plot(procrustes_data, filename=folder_plots + 'all' + 'test1' + name_,
    #                       plot_atoms=False, without_legend=True)

    if len(procrustes_data) == 0:
        print("no data for pucker" + name_)
        return

    # procrustes on puckers and then rewrite procrustes data
    # for LOW RESOLUTION DATA
    if name_ != 'all':
        # rotate data again for the specific pucker
        procrustes_data, procrustes_data_backbone = procrustes_for_each_pucker(cluster_suites, procrustes_data,
                                                                               procrustes_data_backbone, name_)

    # ----------------- CLUSTERING --------------
    # test if we have cluster data saved
    if not os.path.exists("./out/saved_suite_lists/cluster_indices_mode_" + name_ + "_qfold" + str(q_fold) + ".pickle"):
        do_clustering = True
    if not os.path.exists("./out/saved_suite_lists/cluster_indices_" + name_ + "_qfold" + str(q_fold) + ".pickle"):
        do_clustering = True

    if do_clustering:
        # ----------- Step 1: PRE-CLUSTER -----------------------------
        cluster_list, cluster_outlier_list, name_precluster = shape_analysis.pre_clustering(
            input_data=dihedral_angles_suites, m=min_cluster_size,
            percentage=max_outlier_dist_percent,
            string_folder=folder_plots,
            method=method_cluster,
            q_fold=q_fold, distance="torus")

        cluster_list_sorted = cluster_list
        cluster_len_list = [len(cluster) for cluster in cluster_list_sorted]
        cluster_data, cluster_len_list2 = sort_data_into_cluster(dihedral_angles_suites, cluster_list_sorted,
                                                                 min_cluster_size)
        cluster_data_procrust, _ = sort_data_into_cluster(procrustes_data, cluster_list_sorted,
                                                          min_cluster_size)
        cluster_data_procrust_sixchain, _ = sort_data_into_cluster(procrustes_data_backbone, cluster_list_sorted,
                                                                   min_cluster_size)

        if not os.path.exists(folder_plots + name_ + "/"):
            os.makedirs(folder_plots + name_ + "/")
        folder_plots_qfold = folder_plots + name_ + "/" + str(q_fold) + "/"
        if not os.path.exists(folder_plots_qfold):
            os.makedirs(folder_plots_qfold)
        plot_functions.my_scatter_plots(cluster_data,
                                        folder_plots_qfold + name_ + "_outlier" + str(max_outlier_dist_percent)
                                        + "_qfold" + str(q_fold),
                                        set_title="dihedral angles suites" + name_,
                                        number_of_elements=cluster_len_list, legend=True, s=30,
                                        legend_with_clustersize=True)

        # raw plot of all data points
        # plot_functions.my_scatter_plots(dihedral_angles_suites,
        #                                folder_plots + name_ + "/" + "scatter_plot_raw",
        #                                set_title="dihedral angles suites" + name_,
        #                                suite_titles=[r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$',
        #                                              r'$\beta$', r'$\gamma$', r'$\delta_{2}$'],
        #                                number_of_elements=[len(dihedral_angles_suites)],
        #                                legend=True, s=30)

        # plot low resolution data for clustered high resolution
        plot_str = "_outlier" + str(max_outlier_dist_percent) + "_qfold" + str(q_fold)
        # plot_low_res(cluster_list, procrustes_data, procrustes_data_backbone, name_, plot_str, folder_plots_qfold)

        # ------ Step 2: sing Mode Hunting and Torus PCA to post cluster the data. --------------
        cluster_list_mode, noise1 = PNDS_RNA_clustering.new_multi_slink(scale=12000, data=dihedral_angles_suites,
                                                                        cluster_list=cluster_list,
                                                                        outlier_list=cluster_outlier_list,
                                                                        min_cluster_size=min_cluster_size)

        cluster_len_list = [len(cluster) for cluster in cluster_list_mode]
        data_to_plot = np.vstack([dihedral_angles_suites[cluster] for cluster in cluster_list_mode])
        plot_functions.my_scatter_plots(data_to_plot,
                                        filename=folder_plots_qfold + name_ + "_mode" + "_outlier"
                                                 + str(max_outlier_dist_percent) + "_qfold" + str(q_fold),
                                        set_title="dihedral angles suites" + name_,
                                        number_of_elements=cluster_len_list, legend=True, s=45,
                                        legend_with_clustersize=True)

        # save cluster data
        write_files.write_data_to_pickle(cluster_list_mode,
                                         "./out/saved_suite_lists/cluster_indices_mode_" + name_ + "_qfold" + str(
                                             q_fold))
        write_files.write_data_to_pickle(cluster_list,
                                         "./out/saved_suite_lists/cluster_indices_" + name_ + "_qfold" + str(q_fold))

    else:
        # read cluster  data
        cluster_list_mode = write_files.read_data_from_pickle(
            "./out/saved_suite_lists/cluster_indices_mode_" + name_ + "_qfold" + str(q_fold))
        cluster_list = write_files.read_data_from_pickle(
            "./out/saved_suite_lists/cluster_indices_" + name_ + "_qfold" + str(q_fold))

    # cluster1 = dihedral_angles_suites[cluster_list_mode[0]]
    # large_cluster_separation(cluster1)

    # ---------------------- CLUSTER MERGING -----------------------------
    if not os.path.exists(folder_plots + name_ + "/"):
        os.makedirs(folder_plots + name_ + "/")
    folder_plots_qfold = folder_plots + name_ + "/" + str(q_fold) + "/"
    if not os.path.exists(folder_plots_qfold):
        os.makedirs(folder_plots_qfold)
    cluster_list_merged = cluster_merging(cluster_list_mode, dihedral_angles_suites, plot=False)
    cluster_len_list_merged = [len(cluster) for cluster in cluster_list_merged]
    data_to_plot_merged = np.vstack([dihedral_angles_suites[cluster] for cluster in cluster_list_merged])
    # plot_RichardsonClusters(cluster_suites, name_, folder_plots_qfold)
    # create_R_csv(cluster_suites, cluster_list_merged, name_)
    plot_functions.my_scatter_plots(data_to_plot_merged,
                                    filename=folder_plots_qfold + name_ + "_mode_merged" + "_outlier"
                                             + str(max_outlier_dist_percent) + "_qfold" + str(q_fold),
                                    set_title="dihedral angles suites" + name_,
                                    number_of_elements=cluster_len_list_merged, legend=True, s=45,
                                    legend_with_clustersize=True)

    # FINAL CLUSTERS:
    # cluster_list_merged

    # -------- EXTRA PLOTS -------------
    extra_plots = False

    # if extra_plots:
        # plot_all_cluster_combinations(dihedral_angles_suites, cluster_list, folder_plots, name_, q_fold,
        #                               max_outlier_dist_percent, mode=False)

        # plot_all_cluster_combinations(dihedral_angles_suites, cluster_list_mode, folder_plots, name_, q_fold,
        #                               max_outlier_dist_percent, mode=True)

        # plot_all_cluster_combinations(dihedral_angles_suites, cluster_list_merged, folder_plots, name_ + "_merged",
        #                               q_fold, max_outlier_dist_percent, mode=True)

    # -------- plot low resolution data for clustered high resolution ---------------
    plot_string = "_mode_outlier" + str(max_outlier_dist_percent) + "_qfold" + str(q_fold)
    # plot_low_res(cluster_list_mode, procrustes_data, procrustes_data_backbone, name_, folder_plots= folder_plots_qfold,
    #             plot_string=plot_string)

    # ----------- create csv ---------------------
    # create_csv(cluster_suites, cluster_list_mode, name_)

    # ---------- R Clustering ---------------------
    # plot_RichardsonClusters(cluster_suites, name_, folder_plots_qfold)
    # get_R_Clusters(cluster_suites, name_)
    # write_files.get_Richardsons_suitenames(cluster_suites, cluster_list_merged,
    # "./cluster_data/c3c2_cluster_suite_comparison_2023oct.csv")


# ---------------- MAIN  -------------------------
# names = ['c_3_c_3_suites', 'c_3_c_2_suites', 'c_2_c_3_suites', 'c_2_c_2_suites', 'all']
names = ['c3c3', 'c3c2', 'c2c3', 'c2c2', 'all']

#  -------- All Pucker - not up-to-date ------------
# min_cluster_size = 10
# max_outlier = 0.2
# q_fold = 0.05
# cluster_pruned_rna(names[4], min_cluster_size=min_cluster_size, max_outlier_dist_percent=max_outlier, q_fold=q_fold)
# q_fold = 0.07
# cluster_pruned_rna(names[4], min_cluster_size=min_cluster_size, max_outlier_dist_percent=max_outlier, q_fold=q_fold)
# q_fold = 0.10
# cluster_pruned_rna(names[4], min_cluster_size=min_cluster_size, max_outlier_dist_percent=max_outlier, q_fold=q_fold)

# ---------- Pucker Clustering ------------------
min_cluster_size = 3

# ---------- C2'-C2' Pucker ---------------------
max_outlier = 0.02
q_fold = 0.05
cluster_pruned_rna(names[3], min_cluster_size=min_cluster_size, max_outlier_dist_percent=max_outlier, q_fold=q_fold,
                       do_clustering=True)

# ---------- C2'-C3' Pucker ---------------------
max_outlier = 0.02
q_fold = 0.07
cluster_pruned_rna(names[2], min_cluster_size=min_cluster_size, max_outlier_dist_percent=max_outlier, q_fold=q_fold,
                      do_clustering=True)

# ---------- C3'-C2' Pucker ---------------------
max_outlier = 0.02
q_fold = 0.05
cluster_pruned_rna(names[1], min_cluster_size=min_cluster_size, max_outlier_dist_percent=max_outlier, q_fold=q_fold,
                      do_clustering=True)

# ---------- C3'-C3' Pucker ---------------------
max_outlier = 0.02
# q_fold = 0.078
q_fold = 0.090
cluster_pruned_rna(names[0], min_cluster_size=min_cluster_size, max_outlier_dist_percent=max_outlier,
                      q_fold=q_fold, do_clustering=True)
