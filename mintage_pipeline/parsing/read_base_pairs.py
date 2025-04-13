import numpy as np
import pandas


def get_base_pairs(folder, suites):
    """
    This opens base_pairs.csv. The file contains all base pairs. This function adds this base pair information to the
    suite list.
    :param folder: The folder where the base_pairs.csv is stored.
    :param suites: A list of suites objects.
    :return: A list of suites
    """
    base_pairs = pandas.read_csv(folder+"/base_pairs.csv").to_numpy()
    for i in range(base_pairs.shape[0]):
        for j in range(len(suites)):
            if base_pairs[i, 0][:4] == suites[j]._filename and base_pairs[i, 0][4] == suites[j]._name_chain and np.int(base_pairs[i, 0][5:9]) == suites[j]._number_first_residue :
                suites[j].base_pairs['3'] = [base_pairs[i, 2][:4] + '_' + base_pairs[i, 2][4] +
                                             str(np.int(base_pairs[i, 2][5:9])) + '_' + base_pairs[i, 2][4] +
                                             str(1 + np.int(base_pairs[i, 2][5:9])),
                                             base_pairs[i, 0][9], base_pairs[i, 2][9]]
            if base_pairs[i, 0][:4] == suites[j]._filename and base_pairs[i, 0][4] == suites[j]._name_chain and np.int(base_pairs[i, 0][5:9]) == suites[j]._number_second_residue :
                suites[j].base_pairs['4'] = [base_pairs[i, 2][:4] + '_' + base_pairs[i, 2][4] +
                                             str(np.int(base_pairs[i, 2][5:9])) + '_' + base_pairs[i, 2][4] +
                                             str(1 + np.int(base_pairs[i, 2][5:9])),
                                             base_pairs[i, 0][9], base_pairs[i, 2][9]]

            if base_pairs[i, 0][:4] == suites[j]._filename and base_pairs[i, 0][4] == suites[j]._name_chain and np.int(base_pairs[i, 0][5:9]) == suites[j]._number_first_residue - (suites[j]._number_second_residue-suites[j]._number_first_residue):
                suites[j].base_pairs['2'] = [base_pairs[i, 2][:4] + '_' + base_pairs[i, 2][4] +
                                             str(np.int(base_pairs[i, 2][5:9])) + '_' + base_pairs[i, 2][4] +
                                             str(1 + np.int(base_pairs[i, 2][5:9])),
                                             base_pairs[i, 0][9], base_pairs[i, 2][9]]
            if base_pairs[i, 0][:4] == suites[j]._filename and base_pairs[i, 0][4] == suites[j]._name_chain and np.int(base_pairs[i, 0][5:9]) == suites[j]._number_second_residue + (suites[j]._number_second_residue-suites[j]._number_first_residue):
                suites[j].base_pairs['5'] = [base_pairs[i, 2][:4] + '_' + base_pairs[i, 2][4] +
                                             str(np.int(base_pairs[i, 2][5:9])) + '_' + base_pairs[i, 2][4] +
                                             str(1 + np.int(base_pairs[i, 2][5:9])),
                                             base_pairs[i, 0][9], base_pairs[i, 2][9]]
    return suites


