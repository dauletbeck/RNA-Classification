import numpy as np

from utils import write_files
from utils.constants import mean_shapes_all
from parsing.parse_functions import parse_pdb_files, parse_clash_files, shape_analysis_suites
from shape_analysis import procrustes_on_suite_class


def get_suites_from_pdb(pdb_type='2020', outlier_percentage=0):
    """
    :param outlier_percentage: default=0, for RNA 2008: 0.15 in shape analysis
    :param pdb_type:  default='2020' Richardson Lab RNA data pruned without suite outlier
    """
    # old_percentage = 0.15
    # new_percentage = 0
    string_folder = './out/saved_suite_lists/'
    if pdb_type == '2020':
        # pdb_folder = './rna2020_pruned_pdbs/'
        pdb_folder = "/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/data/rna2020_pruned_pdbs/"
    else:
        pdb_folder = None  # './pdb_data/'

    # Step 1:
    suites = parse_pdb_files(input_string_folder=string_folder, input_pdb_folder=pdb_folder)
    # Step 2:
    suites = parse_clash_files(input_suites=suites, input_string_folder=string_folder)
    # Step 3:
    suites = shape_analysis_suites(input_suites=suites, input_string_folder=string_folder,
                                   outlier_percentage=outlier_percentage,
                                   min_cluster_size=1, overwrite=False, rerotate=True, old_data=False)
    print(len(suites))
    # shape_five_chain(input_suites=suites, input_string_folder='./out/saved_suite_lists/')

    return suites


def sort_data_into_cluster(suite_data, cluster_list, min_cluster_length):
    data_sorted_by_cluster = np.array([])
    cluster_len_list = []

    for i in range(0, len(cluster_list)):
        if len(cluster_list[i]) <= min_cluster_length:
            # print("too long")
            continue
        if not data_sorted_by_cluster.any():
            data_sorted_by_cluster = suite_data[cluster_list[i]]
            cluster_len_list.append(len(cluster_list[i]))
            continue

        data_sorted_by_cluster = np.vstack([data_sorted_by_cluster, suite_data[cluster_list[i]]])
        cluster_len_list.append(len(cluster_list[i]))

    return data_sorted_by_cluster, cluster_len_list


def determine_pucker_data(cluster_suites, pucker_name):
    """
    Determines the suites from cluster_suites belonging to sugar pucker pucker_name.

    :param cluster_suites: RNA data as list or numpy array (5chains)
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


def procrustes_for_each_pucker(cluster_suites, procrustes_data, procrustes_data_backbone, name_):
    # procrustes on puckers and then rewrite procrustes data
    # for LOW RESOLUTION DATA
    string = './out/procrustes/five_chain_complete_size_shape' + name_ + '.pickle'
    string_plot = './out/procrustes/five_chain' + name_

    # rotate data again for each pucker
    procrustes_data_pucker = procrustes_on_suite_class(procrustes_data, string, string_plot, origin_index=2,
                                                       mean_shape=np.array(mean_shapes_all[0]))
    procrustes_data = np.array([procrustes_data_pucker[0][i] for i in range(len(cluster_suites))])

    # build_fancy_chain_plot(procrustes_data, filename=folder_plots + 'all' + 'test2' + name_,
    #                       plot_atoms=False, without_legend=True)

    # for HIGH RESOLUTION DATA
    string = './out/procrustes/suites_complete_size_shape' + name_ + '.pickle'
    string_plot = './out/procrustes/suites' + name_

    procrustes_data_backbone_pucker = procrustes_on_suite_class(procrustes_data_backbone, string, string_plot,
                                                                mean_shape=np.array(mean_shapes_all[2]))
    procrustes_data_backbone = np.array([procrustes_data_backbone_pucker[0][i] for i in range(len(cluster_suites))])

    # build_fancy_chain_plot(procrustes_data_backbone, filename=folder_plots + 'all' + '_six_chain_' + name_,
    #                       plot_atoms=False, without_legend=True)

    return procrustes_data, procrustes_data_backbone


def create_csv(cluster_suites, cluster_list_mode, name_):
    # not needed:
    cluster_data_for_csv_pdbid = [[cluster_suites[i]._filename for i in cluster_list_mode[cluster_index]] for
                                  cluster_index in range(len(cluster_list_mode))]
    # cluster_data_for_csv[0] = vom 1. Cluster die namen
    cluster_data_for_csv_chain = [[cluster_suites[i]._name_chain for i in cluster_list_mode[cluster_index]] for
                                  cluster_index in range(len(cluster_list_mode))]
    cluster_data_for_csv_residue_num = [
        [cluster_suites[i]._number_second_residue for i in cluster_list_mode[cluster_index]] for
        cluster_index in range(len(cluster_list_mode))]

    # create list for csv:
    cluster_data_for_csv_together = [
        [[cluster_suites[i]._filename, cluster_suites[i]._name_chain, cluster_suites[i]._number_second_residue,
          cluster_index + 1]
         for i in cluster_list_mode[cluster_index]] for cluster_index in range(len(cluster_list_mode))]

    cluster_data_for_csv_ready = cluster_data_for_csv_together[0]
    for c in range(1, len(cluster_list_mode)):
        cluster_data_for_csv_ready = np.vstack([cluster_data_for_csv_ready, cluster_data_for_csv_together[c]])

    # cluster_data_for_csv_together = [np.append(np.append(cluster_data_for_csv_pdbid[i], cluster_data_for_csv_chain[i])]

    write_files.write_csv(cluster_data_for_csv_ready, path="./tmp/" + name_)
