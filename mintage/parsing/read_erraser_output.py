import parsing.parse_pdb_and_create_suites
import utils.data_functions
import shape_analysis
import utils.plot_functions
import utils.constants
from utils.constants import COLORS, COLORS_SCATTER
import pnds.PNDS_RNA_clustering

from scipy.cluster.hierarchy import average
import matplotlib.pyplot as plot
import numpy as np
import os

from utils.help_plot_functions import plot_all_clash_suites_before_and_after_ERRASER_correction, \
    plot_all_single_clash_suites_before_and_after_correction_by_ERRASER, plot_ERRASER_clustering_results


def import_erraser_pdb(folder, input_suites):
    """
    This function adds all ERRASER suites to the data set of suites and makes some data analysis with the ERRASER suites.
    :param: folder: A string with the ERRASER pdb files.
    :param: input_suites: A list with the suite objects.
    """
    files = parse_pdb_and_create_suites.find_files('*.pdb', folder=folder)
    files.sort()
    # create dictionary: The items are the file names of the pdb-data. (rfind give the number of the last occurrence
    data = [x for x in [parse_pdb_and_create_suites.import_pdb_file(f) for f in files]]
    data = [suite for list_ in data for suite in list_]
    mu_mesoscopic = np.mean(np.array([suite.procrustes_complete_mesoscopic_vector for suite in input_suites if suite.complete_suite]), axis=0)
    mu_mesoscopic = mu_mesoscopic/np.linalg.norm(mu_mesoscopic)
    mu_suite = np.mean(np.array([suite.procrustes_complete_suite_vector for suite in input_suites if suite.complete_suite]), axis=0)
    mu_suite = mu_suite/np.linalg.norm(mu_suite)
    names_unique = list(set([erraser_suite.name[:4] for erraser_suite in data]))
    dict_filenames = {names_unique[j]: [i for i in range(len(data)) if data[i].name[:4]==names_unique[j]] for j in range(len(names_unique))}
    # Add some information to the suite objects that have a correction by ERRASER.
    for suite in input_suites:
        if suite._filename in names_unique:
            for erraser_suite in [data[i] for i in dict_filenames[suite._filename]]:
                if erraser_suite.name == suite.name:
                    suite.erraser['backbone_atoms'] = erraser_suite.backbone_atoms
                    suite.erraser['dihedral_angles'] = erraser_suite.dihedral_angles
                    suite.erraser['mesoscopic_sugar_rings'] = erraser_suite.mesoscopic_sugar_rings
                    # rotate the erraser backbone atoms optimal to the mean shape of the not erraser data set.
                    if suite.complete_suite:
                        erraser_backbone_shifted = erraser_suite.backbone_atoms-np.mean(erraser_suite.backbone_atoms, axis=0)
                        erraser_backbone_scaled = erraser_backbone_shifted#/np.linalg.norm(erraser_backbone_shifted)
                        rotated_erraser_backbone = data_functions.rotate_y_optimal_to_x(y=erraser_backbone_scaled, x=mu_suite)  #np.dot(x_y_rot, np.transpose(data_functions.rotation_matrix_z_axis(alpha[0][2])))
                        suite.erraser['rotated_backbone'] = rotated_erraser_backbone
                        # rotate the erraser mesoscopic shapes optimal to the mean shape of the not erraser data set.
                        erraser_mesoscopic_shifted = erraser_suite.mesoscopic_sugar_rings-np.mean(erraser_suite.mesoscopic_sugar_rings, axis=0)
                        erraser_mesoscopic_scaled = erraser_mesoscopic_shifted#/np.linalg.norm(erraser_mesoscopic_shifted)
                        rotated_erraser_mesoscopic = data_functions.rotate_y_optimal_to_x(y=erraser_mesoscopic_scaled, x=mu_mesoscopic) #np.dot(x_y_rot, np.transpose(data_functions.rotation_matrix_z_axis(alpha[0][2])))
                        suite.erraser['rotated_mesoscopic_sugar_rings'] = rotated_erraser_mesoscopic
                    else:
                        suite.erraser['rotated_backbone'] = None
                        suite.erraser['rotated_mesoscopic_sugar_rings'] = None
    # This function creates a lot of plots and clusters the ERRASER suites.
    shape_analysis_erraser(input_suites)
    return input_suites


def shape_analysis_erraser(input_suites):
    """
    This function plots first plots all clash suites and the correction by erraser.
    It also clusters the ERRASER suites as described in the paper.
    :param input_suites: A list with suite objects.
    :return:
    """
    erraser_clash_suites = [suite for suite in input_suites if
                            len(suite.bb_bb_one_suite) > 0 and len(suite.erraser) > 0 and suite.complete_suite]
    erraser_clash_backbone = np.array([suite.erraser['rotated_backbone'] for suite in erraser_clash_suites])
    erraser_clash_backbone_original = np.array([suite.erraser['backbone_atoms'] for suite in erraser_clash_suites])
    start_clash_backbone = np.array([suite.procrustes_complete_suite_vector for suite in erraser_clash_suites])

    start_clash_backbone_original = np.array([suite.backbone_atoms for suite in erraser_clash_suites])
    string ='./out/erraser_analysis/'

    if not os.path.exists(string):
        os.makedirs(string)
    # Plot all clash suites: First the corrected clash suites by ERRASER, then the raw clash suites.
    plot_all_clash_suites_before_and_after_ERRASER_correction(erraser_clash_backbone, start_clash_backbone, string)

    # Add the clash information of the raw dataset and the erraser dataset to suite.clashscore.
    clash_dict = []
    compare_erraser_clashes(clash_dict, erraser_clash_suites, raw=True)
    clash_dict = []
    compare_erraser_clashes(clash_dict, erraser_clash_suites, raw=False)
    
    erraser_clash_suites = [erraser_clash_suites[i] for i in range(len(erraser_clash_suites)) if 'raw' in erraser_clash_suites[i].clashscore.keys()]
    # The number of clashes in the clash suites before and after the ERRASER correction.
    erraser_clashnumber = [len(erraser_clash_suites[i].clashscore['erraser']) if 'erraser' in erraser_clash_suites[i].clashscore.keys() else 0 for i in range(len(erraser_clash_suites))]
    raw_clashnumber = [len(erraser_clash_suites[i].clashscore['raw']) if 'raw' in erraser_clash_suites[i].clashscore.keys() else 0 for i in range(len(erraser_clash_suites))]

    plot.scatter(np.arange(np.max(raw_clashnumber)+1), [sum(raw_clashnumber==i) for i in np.arange(np.max(raw_clashnumber)+1)], label='raw suites', color='black')
    plot.scatter(np.arange(np.max(erraser_clashnumber)+1), [sum(erraser_clashnumber==i) for i in np.arange(np.max(erraser_clashnumber)+1)], label='ERRASER corrected suite', color='darkmagenta', marker='D')
    plot.xticks([0, 1, 2, 3, 4], ['0', '1', '2', '3', '4'])
    plot.xlabel('number of clashes in suite', fontdict={'size': 15})
    plot.ylabel('number of suites', fontdict={'size': 15})
    plot.legend()
    plot.savefig(string+'erraser_hist')
    plot.close()


    string_single = string + 'single_suite_clash_free/'
    plot_all_single_clash_suites_before_and_after_correction_by_ERRASER(erraser_clash_backbone,
                                                                        erraser_clash_backbone_original,
                                                                        erraser_clash_suites, start_clash_backbone,
                                                                        start_clash_backbone_original, string,
                                                                        string_single)

    erraser_suites = [suite for suite in input_suites if len(suite.erraser) > 0 and suite.complete_suite]
    start_data = np.array([suite.procrustes_complete_suite_vector for suite in input_suites if len(suite.erraser) > 0 and suite.complete_suite])
    erraser_data = np.array([suite.erraser['rotated_backbone'] for suite in input_suites if len(suite.erraser) > 0 and suite.complete_suite])
    name = string + 'erraser_clustering/'
    dihedral_angles_suites_start = np.array([suite._dihedral_angles for suite in input_suites if len(suite.erraser) > 0 and suite.complete_suite])
    dihedral_angles_suites_erraser = np.array([suite.erraser['dihedral_angles'] for suite in input_suites if len(suite.erraser) > 0 and suite.complete_suite])
    cluster_list_erraser, outlier_list_erraser, both_data_sets = compare_erraser_clustering(cluster_data=dihedral_angles_suites_erraser,
                                                                            erraser_procrustes=erraser_data,
                                                                            m=20, name=string + 'erraser_clustering/',
                                                                            percentage=0.15, start_procrustes=start_data,
                                                                            string=string + 'erraser_clustering/', suites=input_suites)

    clash_suites_index = [i for i in range(len(erraser_suites)) if len(erraser_suites[i].bb_bb_one_suite) > 0]
    erraser_clash_in_cluster = [[i for i in clash_suites_index if i in cluster_list_erraser[j]] for j in range(len(cluster_list_erraser))]
    erraser_clash_outlier = [i for i in clash_suites_index if i in outlier_list_erraser]
    plot_ERRASER_clustering_results(both_data_sets, cluster_list_erraser, erraser_clash_in_cluster,
                                    erraser_clash_outlier, erraser_data, name, start_data)


def compare_erraser_clashes(clash_dict, erraser_clash_suites, raw):
    """
    This function parses the 
    :param clash_dict: 
    :param erraser_clash_suites: 
    :param raw: 
    :return: 
    """
    erraser_clashscore_string = './phenix_validation_reports/erraser/'
    raw_clashscore_string = './phenix_validation_reports/orginal/'
    if raw:
        files = parse_pdb_and_create_suites.find_files('*.txt', folder=raw_clashscore_string)
    else:
        files = parse_pdb_and_create_suites.find_files('*.txt', folder=erraser_clashscore_string)
    files.sort()
    for i in range(len(files)):
        f = open(files[i], "r")
        file_list = []
        line = f.readline()
        while line:
            line = f.readline()
            file_list.append(line)
        f_name = files[i][files[i].rfind('/') + 1: files[i].rfind('/') + 5]
        for j in range(len(file_list)):
            string_list = file_list[j].split(' ')
            array = [string for string in string_list if string is not '']
            if len(array) == 9:
                chain_1 = array[0]
                number_1 = array[1]
                atom_1 = array[3]
                chain_2 = array[4]
                number_2 = array[5]
                atom_2 = array[7]
                clash_dict.append({'f_name': f_name, 'chain_1': chain_1, 'number_1': number_1, 'atom_1': atom_1,
                                   'chain_2': chain_2, 'number_2': number_2, 'atom_2': atom_2})
    counter = 0
    for clash in clash_dict:
        if clash['atom_1'] in constants.BACKBONE_ATOMS_VALIDATION and clash['atom_2'] in constants.BACKBONE_ATOMS_VALIDATION:
            for suite in erraser_clash_suites:
                if clash['f_name'] == suite._filename and clash['chain_1'] == suite._name_chain and clash['chain_2'] == suite._name_chain \
                        and clash['number_1'] in [str(suite._number_first_residue),
                                                  str(suite._number_second_residue)] and \
                        clash['number_2'] in [str(suite._number_first_residue), str(suite._number_second_residue)]:
                    if raw:
                        if len(suite.clashscore) == 0 or 'raw' not in suite.clashscore.keys():
                            suite.clashscore['raw'] = []
                        suite.clashscore['raw'].append(clash)
                        # print(clash)
                        counter = counter + 1
                    else:
                        if len(suite.clashscore) == 0 or 'erraser' not in suite.clashscore.keys():
                            suite.clashscore['erraser'] = []
                        suite.clashscore['erraser'].append(clash)


def compare_erraser_clustering(cluster_data, erraser_procrustes, m, name, percentage, start_procrustes, string, suites):
    """
    A function that produces the ERRASER clustering from the paper.
    :param cluster_data:
    :param erraser_procrustes:
    :param m: The minimal cluster size.
    :param name: A string. The folder where we store the results.
    :param percentage: A float value. It describes the maximal outlier distance.
    :param start_procrustes:
    :param string:
    :param suites: A list of suites
    :return:
    """
    admissible_not_erraser_suites = [suite for suite in suites if len(suite.erraser)==0 and suite.complete_suite and len(suite.bb_bb_neighbour_clashes)==0]
    dihedral_angles_admissible = np.array([suite._dihedral_angles for suite in admissible_not_erraser_suites])

    cluster_list, outlier_list, _ = shape_analysis.pre_clustering(input_data=np.vstack((cluster_data, dihedral_angles_admissible)),
                                                                  m=m,
                                                                  percentage=percentage,
                                                                  string_folder=string,
                                                                  method=average,
                                                                  q_fold=0.15)

    cluster_list, noise = PNDS_RNA_clustering.new_multi_slink(scale=12000, data=np.vstack((cluster_data, dihedral_angles_admissible)), cluster_list=cluster_list, outlier_list=outlier_list)
    if not os.path.exists(name):
        os.makedirs(name)

    procrustes_admissible = np.array([suite.procrustes_complete_suite_vector for suite in admissible_not_erraser_suites])
    for i in range(len(cluster_list)):
        plot_functions.build_fancy_chain_plot(np.vstack((erraser_procrustes, procrustes_admissible))[cluster_list[i]],
                                              filename=name + 'cluster_nr' + str(i) + '_erraser')
    plot_functions.build_fancy_chain_plot(np.vstack((erraser_procrustes, procrustes_admissible))[outlier_list], filename=name + 'outlier_erraser')


    # Plot some specific clusters:
    i=0
    j=1
    k=2
    #l=6
    m=8
    plot_functions.build_fancy_chain_plot(np.vstack((erraser_procrustes, procrustes_admissible))[list(cluster_list[i]) + list(cluster_list[j]) + list(cluster_list[k]) + list(cluster_list[m])],
                                          filename=name + 'cluster_nr' + str(i) + 'and' + str(j) + 'and' + str(k) + 'and'  + str(m)+ '_suite',
                                          colors=[COLORS_SCATTER[0]]*len(cluster_list[i]) + [COLORS_SCATTER[1]]*len(cluster_list[j]) + [COLORS_SCATTER[3]]*len(cluster_list[k]) + [COLORS_SCATTER[9]]*len(cluster_list[m]),
                                          specific_legend_colors=[COLORS[0], COLORS[1], COLORS[2], COLORS[3]],
                                          specific_legend_strings=["Cluster " + str(i+1), "Cluster " + str(j+1), "Cluster " + str(k+1), "Cluster " + str(m+1)],
                                          create_label=False, alpha_line_vec=[0.1]*len(cluster_list[i]) + [1]*len(cluster_list[j]) + [1]*len(cluster_list[k]) + [1]*len(cluster_list[m]),
                                          plot_atoms=True, atom_alpha_vector=[0.1]*len(cluster_list[i]) + [1]*len(cluster_list[j]) + [1]*len(cluster_list[k]) + [1]*len(cluster_list[m]),
                                          atom_color_vector=[COLORS_SCATTER[0]]*len(cluster_list[i]) + [COLORS_SCATTER[1]]*len(cluster_list[j]) + [COLORS_SCATTER[3]]*len(cluster_list[k]) + [COLORS_SCATTER[9]]*len(cluster_list[m]), atom_size=0.1, without_legend=True)
    return cluster_list, outlier_list, np.vstack((erraser_procrustes, procrustes_admissible))

