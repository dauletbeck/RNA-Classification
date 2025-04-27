import csv
import pickle
import numpy as np
import pandas


def _write_file(text, path):
    with open(path, "w") as file:
        file.write(text)
        print("Done writing data into .txt file")


def _read_file(path):
    with open(path, 'rb') as file:
        text = file.read()
        return text


def _write_suites_to_file(cluster, num, name=None):
    folder = './out/'
    if name is None:
        name = 'clustered_suites.txt'

    # np.savetxt('./out/testi.txt', cluster_data_procrust_sixchain[0].round(3), fmt='%1.3f', delimiter=',')
    with open(folder + name, "a") as file:
        file.write(f"suites cluster {num}:")
        for suite in cluster:
            file.write("\n")
            file.write(np.array2string(suite.round(3), separator=', '))
            file.write("\n")


def data_to_cluster(data, cluster_lens, name=None):
    np.set_printoptions(suppress=True)
    num = 1
    _write_suites_to_file(data[0:cluster_lens[0]], num, name)
    for i in range(len(cluster_lens) - 1):
        num += 1
        _write_suites_to_file(data[sum(cluster_lens[0:i]): sum(cluster_lens[0:i + 1])], num, name)


# low resolution:
# array([[-0.96616421,  5.06313809, -0.67082738],
#        [-2.0577361 ,  4.31163775, -0.04366946],
#        [ 0.        ,  0.        ,  0.        ],
#        [-0.96380312,  1.88667046, -4.789524  ],
#        [ 0.2712553 ,  2.622374  , -4.57072502]])

# high resolution
# cluster_data_procrust_sixchain[0]
# array([[-0.10821482,  2.24721749,  4.09435497],
#        [-0.76492409,  2.44391126,  2.74829943],
#        [-0.14674339,  1.64314785,  1.61343751],
#        [-0.48640348,  0.25975998,  1.64485461],
#        [ 0.62591682, -0.80497605,  1.1615796 ],
#        [ 0.60601409, -0.65148128, -0.42008637],
#        [-0.55054738, -1.02255702, -1.15680162],
#        [-0.38499168, -0.5104429 , -2.57023165],
#        [ 0.76355289, -1.14045415, -3.34904882],
#        [ 0.44634105, -2.46412518, -3.76635767]])

def write_cluster_index_to_txt(cluster_indizes, path=None):
    if path is None:
        path = "./out/indices_test.txt"
    cluster_indizes = np.array(cluster_indizes, dtype=object)
    np.set_printoptions(threshold=100000)
    _write_file(np.array2string(cluster_indizes), path)
    np.set_printoptions(threshold=1000)


def read_cluster_index_from_txt(path=None):
    if path is None:
        path = "./out/indices_test.txt"
    return _read_file(path)


def write_data_to_pickle(data, path=None):
    if path is None:
        path = "./out/indices_test.pickle"
    else:
        path += ".pickle"
    with open(path, "wb") as file:
        pickle.dump(data, file)
        print("Done writing data into pickle file")


def read_data_from_pickle(path=None):
    if path is None:
        path = "./out/indices_test.pickle"
    else:
        path += ".pickle"
    with open(path, 'rb') as file:
        text = pickle.load(file)
        return text


def write_csv(np_array, path):
    path += ".csv"
    df = pandas.DataFrame(np_array)
    df.columns = ["pdbid", "chain", "residue_number", "cluster_id"]
    df.to_csv(path, index=False)  # header=False, index=False
    # with open(path, 'wb', newline='') as file:
    #    csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

def write_csv_Rcluster(np_array, path):
    path += ".csv"
    df = pandas.DataFrame(np_array)
    df.columns = ["pdbid", "chain", "residue_number", "cluster_id", "Rcluster"]
    df.to_csv(path, index=False)  # header=False, index=False

def read_csv(path):
    df = pandas.read_csv(path, header=0)
    return df


def get_Richardsons_suites(suite_array, path, indizes=False):
    # path = ""
    df = read_csv(path)
    cluster_dict = {}
    index_dict = {}

    for i in range(len(df)):
        suite_name = df.iloc[i][4]
        filename = df.iloc[i][0]
        chain = df.iloc[i][1]
        resseq = df.iloc[i][2]

        for nr, suite in enumerate(suite_array):
            if (suite._filename == filename and suite._name_chain == chain
                    and suite._number_second_residue == resseq):
                if suite_name in cluster_dict.keys():
                    cluster_dict[suite_name].append(suite)
                    index_dict[suite_name].append(nr)
                else:
                    cluster_dict[suite_name] = [suite]
                    index_dict[suite_name] = [nr]
                break

    if indizes:
        return index_dict
    else:
        return cluster_dict


def get_Richardsons_suitenames(suite_data, cluster_index_array, path):
    index_list = []
    R_index_dict = get_Richardsons_suites(suite_data, path, indizes=True)
    for cluster in cluster_index_array:
        index_sub_list = []
        for point_index in cluster:
            found = False
            # index_sub_list.append(list(R_index_dict.keys())[list(R_index_dict.values()).index(i)])
            for key_index, ind_list in enumerate(list(R_index_dict.values())):
                if point_index in ind_list:
                    index_sub_list.append(list(R_index_dict.keys())[key_index])
                    found = True
                    break
            if not found:
                index_sub_list.append('no data')

        index_list.append(index_sub_list)
    return index_list
