import numpy as np
import os
ATOM_COLORS = ['gray', 'blue', 'magenta', 'gold', 'red', 'green']
line_colors = 5*['black', 'gray', 'blue', 'red', 'brown', 'yellow', 'magenta', 'green', 'orange', 'purple', 'pink']


def cluster_to_king(cluster_list, cluster_suites, filename):
    cluster_name_suites = filename + 'suites.kin'
    atom_list = ["C5'", "C4'", "C3'", "O3'", "P'", "O5'", "C5'", "C4'", "C3'", "O3'"]
    atom_colors = 3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]] + [ATOM_COLORS[2]] + [ATOM_COLORS[1]] + \
                                    3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]]
    if os.path.isfile(cluster_name_suites):
        os.remove(cluster_name_suites)
    with open(cluster_name_suites, 'a') as the_file:
        the_file.write('@kinemage\n\n')
        #the_file.write('@group {groupname} collapsible\n\n')
        for i in range(len(cluster_list)):
            the_file.write('@group {cluster ' + str(i+1) + '} collapsible\n\n')
            index_list = cluster_list[i]
            for index in index_list:
                suite = cluster_suites[index]
                procrustes = suite.procrustes_complete_suite_vector
                the_file.write('@subgroup {suite name:' + suite.name + '} dominant\n')
                the_file.write('@vectorlist {lines} \n')
                for atom_index in range(len(atom_list)):
                    atom = atom_list[atom_index]
                    the_file.write("{atom name " + atom + "} " + line_colors[i] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('@balllist {balls} radius=0.1 \n')
                for atom_index in range(len(atom_list)):
                    atom = atom_list[atom_index]
                    the_file.write("{suite name:" + suite.name + " atom name " + atom + "} " + atom_colors[atom_index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('\n')



    atom_list_low = ["N", "C1'", "P'", "C1'", "N"]
    atom_colors_low = [ATOM_COLORS[3], ATOM_COLORS[0], ATOM_COLORS[2], ATOM_COLORS[0], ATOM_COLORS[3]]
    cluster_name_low_res = filename + 'low_res.kin'
    if os.path.isfile(cluster_name_low_res):
        os.remove(cluster_name_low_res)
    with open(cluster_name_low_res, 'a') as the_file:
        the_file.write('@kinemage\n\n')
        for i in range(len(cluster_list)):
            the_file.write('@group {cluster ' + str(i+1) + '} collapsible\n\n')
            index_list = cluster_list[i]
            for index in index_list:
                suite = cluster_suites[index]
                procrustes = suite.procrustes_five_chain_vector
                the_file.write('@subgroup {suite name:' + suite.name + '} dominant\n')
                the_file.write('@vectorlist {lines} \n')
                for atom_index in range(len(atom_list_low)):
                    atom = atom_list_low[atom_index]
                    the_file.write("{atom name " + atom + "} " + line_colors[i] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('@balllist {balls} radius=0.1 \n')
                for atom_index in range(len(atom_list_low)):
                    atom = atom_list_low[atom_index]
                    the_file.write("{suite name:" + suite.name + " atom name " + atom + "} " + atom_colors_low[atom_index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('\n')

    meso_names = ["ribose 1", "ribose 2", "ribose 3", "ribose 4", "ribose 5", "ribose 6"]
    atom_colors_mes = ATOM_COLORS
    cluster_name_low_res = filename + 'meso.kin'
    if os.path.isfile(cluster_name_low_res):
        os.remove(cluster_name_low_res)
    with open(cluster_name_low_res, 'a') as the_file:
        the_file.write('@kinemage\n\n')
        for i in range(len(cluster_list)):
            the_file.write('@group {cluster ' + str(i+1) + '} collapsible\n\n')
            index_list = cluster_list[i]
            for index in index_list:
                if not cluster_suites[index].procrustes_complete_mesoscopic_vector is None:
                    suite = cluster_suites[index]
                    procrustes = suite.procrustes_complete_mesoscopic_vector
                    the_file.write('@subgroup {suite name:' + suite.name + '} dominant\n')
                    the_file.write('@vectorlist {lines} \n')
                    for atom_index in range(len(meso_names)):
                        atom = meso_names[atom_index]
                        the_file.write("{atom name " + atom + "} " + line_colors[i] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                    the_file.write('@balllist {balls} radius=0.1 \n')
                    for atom_index in range(len(meso_names)):
                        atom = meso_names[atom_index]
                        the_file.write("{suite name:" + suite.name + " atom name " + atom + "} " + atom_colors_mes[atom_index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                    the_file.write('\n')


def plot_procrustes_data(suite_list, filename, plot_suites=True, plot_low_res=True, plot_meso=True, color_list=None, group_list=None):
    name_ = filename + '.kin'
    if os.path.isfile(name_):
        os.remove(name_)
    with open(name_, 'a') as the_file:
        the_file.write('@kinemage\n\n')
        if plot_suites:
            atom_list = ["C5'", "C4'", "C3'", "O3'", "P'", "O5'", "C5'", "C4'", "C3'", "O3'"]
            atom_colors = 3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]] + [ATOM_COLORS[2]] + [ATOM_COLORS[1]] + \
                          3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]]

            if group_list is None:
                the_file.write('@group {suites} collapsible\n\n')
            for index in range(len(suite_list)):
                if group_list is not None:
                    if index == 0 or group_list[index] is not group_list[index-1]:
                        the_file.write('@group {' + group_list[index] + '} collapsible\n\n')
                suite = suite_list[index]
                procrustes = suite.procrustes_complete_suite_vector
                the_file.write('@subgroup {suite name:' + suite.name + '} dominant\n')
                the_file.write('@vectorlist {lines} \n')
                for atom_index in range(len(atom_list)):
                    atom = atom_list[atom_index]
                    if color_list is None:
                        the_file.write("{atom name " + atom + "} " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                    else:
                        the_file.write("{atom name " + atom + "} " + color_list[index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('@balllist {balls} radius=0.1 \n')
                for atom_index in range(len(atom_list)):
                    atom = atom_list[atom_index]
                    the_file.write("{suite name:" + suite.name + " atom name " + atom + "} " + atom_colors[atom_index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('\n')
        if plot_low_res:
            atom_list_low = ["N", "C1'", "P'", "C1'", "N"]
            atom_colors_low = [ATOM_COLORS[3], ATOM_COLORS[0], ATOM_COLORS[2], ATOM_COLORS[0], ATOM_COLORS[3]]
            if group_list is None:
                the_file.write('@group {low re} collapsible\n\n')
            for index in range(len(suite_list)):
                if group_list is not None:
                    if index == 0 or group_list[index] is not group_list[index-1]:
                        the_file.write('@group {' + group_list[index] + '} collapsible\n\n')
                suite = suite_list[index]
                procrustes = suite.procrustes_five_chain_vector
                the_file.write('@subgroup {suite name:' + suite.name + '} dominant\n')
                the_file.write('@vectorlist {lines} \n')
                for atom_index in range(len(atom_list_low)):
                    atom = atom_list_low[atom_index]
                    if color_list is None:
                        the_file.write("{atom name " + atom + "} " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                    else:
                        the_file.write("{atom name " + atom + "} " + color_list[index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('@balllist {balls} radius=0.1 \n')
                for atom_index in range(len(atom_list_low)):
                    atom = atom_list_low[atom_index]
                    the_file.write("{suite name:" + suite.name + " atom name " + atom + "} " + atom_colors_low[atom_index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('\n')
        if plot_meso:
            meso_names = ["ribose 1", "ribose 2", "ribose 3", "ribose 4", "ribose 5", "ribose 6"]
            atom_colors_mes = ATOM_COLORS
            if group_list is None:
                the_file.write('@group {mesoscopics} collapsible\n\n')
            for index in range(len(suite_list)):
                if group_list is not None:
                    if index == 0 or group_list[index] is not group_list[index-1]:
                        the_file.write('@group {' + group_list[index] + '} collapsible\n\n')
                if not suite_list[index].procrustes_complete_mesoscopic_vector is None:
                    suite = suite_list[index]
                    procrustes = suite.procrustes_complete_mesoscopic_vector
                    the_file.write('@subgroup {suite name:' + suite.name + '} dominant\n')
                    the_file.write('@vectorlist {lines} \n')
                    for atom_index in range(len(meso_names)):
                        atom = meso_names[atom_index]
                        if color_list is None:
                            the_file.write("{atom name " + atom + "} " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                        else:
                            the_file.write("{atom name " + atom + "} " + color_list[index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                    the_file.write('@balllist {balls} radius=0.1 \n')
                    for atom_index in range(len(meso_names)):
                        atom = meso_names[atom_index]
                        the_file.write("{suite name:" + suite.name + " atom name " + atom + "} " + atom_colors_mes[atom_index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                    the_file.write('\n')

    print('test')


def low_detail_array(low_detail_data, input_suites, filename, plot_low_res=True, color_list=None, group_list=None, lw_list=None):
    name_ = filename + '.kin'
    if os.path.isfile(name_):
        os.remove(name_)
    with open(name_, 'a') as the_file:
        the_file.write('@kinemage\n\n')
        if plot_low_res:
            atom_list_low = ["N", "C1'", "P'", "C1'", "N"]
            atom_colors_low = [ATOM_COLORS[3], ATOM_COLORS[0], ATOM_COLORS[2], ATOM_COLORS[0], ATOM_COLORS[3]]
            if group_list is None:
                the_file.write('@group {low detail} collapsible\n\n')
            for index in range(len(low_detail_data)):
                if group_list is not None:
                    if index == 0 or group_list[index] is not group_list[index-1]:
                        the_file.write('@group {' + group_list[index] + '} collapsible\n\n')
                suite = input_suites[index]
                procrustes = low_detail_data[index]
                the_file.write('@subgroup {suite name:' + suite.name + '} dominant\n')
                the_file.write('@vectorlist {lines}' + ' width=' + str(lw_list[index]) + '\n')
                for atom_index in range(len(atom_list_low)):
                    atom = atom_list_low[atom_index]
                    if color_list is None:
                        the_file.write("{atom name " + atom + "} " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                    else:
                        the_file.write("{atom name " + atom + "} " + color_list[index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('@balllist {balls} radius=0.1 \n')
                for atom_index in range(len(atom_list_low)):
                    atom = atom_list_low[atom_index]
                    the_file.write("{suite name:" + suite.name + " atom name " + atom + "} " + atom_colors_low[atom_index] + " " + str(procrustes[atom_index][0]) + " " + str(procrustes[atom_index][1]) + " " + str(procrustes[atom_index][2]) + "\n")
                the_file.write('\n')


    print('test')