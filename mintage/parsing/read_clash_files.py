import pickle
from xml.etree import ElementTree as element_tree
import platform

import numpy as np

from parsing.parse_pdb_and_create_suites import find_files
from utils.constants import RNA_BASES_VALIDATION, BACKBONE_ATOMS_VALIDATION


def get_clashes(folder, suites, model_number=False):
    """
    This function reads all files from the folder 'folder' and adds the clash information to the suite objects from the
    list of suites 'suites'.
    :param folder: A string. The path of the stored clash files.
    :param suites: A list with suite objects.
    :param model_number: Boolean: True if we have more than one model stored in the pdb files.
    :return: The modified suite list.
    """
    files = find_files('*.xml', folder=folder)
    for file in files:
        import_clash_file(file, suites, model_number)
    with open('./out/suites.pickle', 'wb') as f:
        pickle.dump(suites, f)
    return suites
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
        if platform.system() == "Windows":
            f_name = files[i][files[i].rfind('\\') + 1: files[i].rfind('\\') + 5]
        else:
            # Linux, Mac
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
                        counter = counter + 1
                    else:
                        if len(suite.clashscore) == 0 or 'erraser' not in suite.clashscore.keys():
                            suite.clashscore['erraser'] = []
                        suite.clashscore['erraser'].append(clash)


def import_clash_file(filename, suites, model_number=False):
    """
    This function adds all clash information from the clash-file 'filename' to the suite objects in the 'suites' list.
    :param filename: A string describing a filename of a clash-file.
    :param suites: A list with suite objects.
    :param model_number: Boolean: True if we have more than one model stored in the pdb files.
    :return: None
    """
    if platform.system() == "Windows":
        indices_of_suites_in_file = [i for i in range(len(suites)) if
                                     filename[filename.rfind('\\') + 1: filename.rfind('\\') + 5] == suites[i]._filename]
    else:
        indices_of_suites_in_file = [i for i in range(len(suites)) if
                                    filename[filename.rfind('/') + 1: filename.rfind('/') + 5] == suites[i]._filename]

    clash_list = []
    tree = element_tree.parse(filename)
    root = tree.getroot()
    for child in root:
        if child.tag == 'ModelledSubgroup':
            # filter to get only the RNA-chains:
            if child.attrib.get('resname') in RNA_BASES_VALIDATION and list(child).__len__() > 0:
                # Check if there are some clashes in the residuum:
                if list(child).__len__() > 0:
                    for number in range(list(child).__len__()):
                        if list(child)[number].tag == 'clash':
                            suite_1, suite_2 = get_the_suites(child.attrib.get('resnum'),
                                                              list(child)[number].attrib.get('atom'))

                            if platform.system() == "Windows":
                                filename_found = filename[filename.rfind('\\') + 1: filename.rfind('\\') + 5]
                            else:
                                filename_found = filename[filename.rfind('/') + 1: filename.rfind('/') + 5]
                            child_clash_info = {'filename': filename_found,
                                                'chain': child.attrib.get('chain'),
                                                'resname': child.attrib.get('resname'),
                                                'icode': child.attrib.get('icode'),
                                                'suite_1': suite_1,
                                                'suite_2': suite_2,
                                                'atom': list(child)[number].attrib.get('atom'),
                                                'cid': list(child)[number].attrib.get('cid'),
                                                'clashmag': list(child)[number].attrib.get('clashmag'),
                                                'dist': list(child)[number].attrib.get('dist'),
                                                'model': child.attrib.get('model')}
                            clash_list.append(child_clash_info)
    # Connection to the second clash atom:
    if len(clash_list) == 0:
        return np.array([])
    clash_id = np.array([clash_list[i]['cid'] for i in range(len(clash_list))]).astype('float')
    clash_tuple_indices = list(
        set(tuple([tuple([i for i, x in enumerate(clash_id) if x == clash_id[j]]) for j in range(len(clash_id))])))

    for clash_tuple in clash_tuple_indices:
        clash_suites = [[clash_list[clash_tuple[clash_list_index]]['suite_1'],
                         clash_list[clash_tuple[clash_list_index]]['suite_2']] for clash_list_index in range(len(clash_tuple))]
        clash_chains = [clash_list[clash_tuple[clash_list_index]]['chain'] for clash_list_index in range(len(clash_tuple))]
        clash_models = [clash_list[clash_tuple[clash_list_index]]['model'] for clash_list_index in range(len(clash_tuple))]
        for suite_index in indices_of_suites_in_file:
            if model_number is False:
                if True in [(suites[suite_index]._number_first_residue in clash_suites[i])
                            and (suites[suite_index]._name_chain in clash_chains[i]) for i in range(len(clash_tuple))]:
                    suite_classification(suites[suite_index], clash_tuple, clash_list)
            else:
                if True in [(suites[suite_index]._number_first_residue in clash_suites[i])
                            and (suites[suite_index]._name_chain in clash_chains[i])
                            and (str(suites[suite_index].model_number) in clash_models[i]) for i in range(len(clash_tuple))]:
                    suite_classification(suites[suite_index], clash_tuple, clash_list)


def get_the_suites(res_number, atom):
    res_number = int(res_number)
    if res_number > 0:
        suite_1 = res_number - 1
        if atom in BACKBONE_ATOMS_VALIDATION[5:7] + BACKBONE_ATOMS_VALIDATION[10:12]:
            suite_2 = res_number - 1
        else:
            suite_2 = res_number
    else:
        if atom in BACKBONE_ATOMS_VALIDATION[5:7] + BACKBONE_ATOMS_VALIDATION[10:12]:
            suite_1 = 0
            suite_2 = 0
        else:
            suite_1 = 0
            suite_2 = 0
    return suite_1, suite_2


def suite_classification(suite, clash_tuple, clash_list):
    """
    This function adds all clash information to a suite object.
    :param suite: A suite object.
    :param clash_tuple: A tuple with two integers (indices of clash_list).
    :param clash_list: A list with clashes. Each element is a dictionary with clash information.
    :return:
    """
    # check if the clash is the first or second one in the table.
    if suite._number_first_residue in [clash_list[clash_tuple[0]]['suite_1'], clash_list[clash_tuple[0]]['suite_2']]:
        tuple_index = 0
        other_index = 1
    else:
        tuple_index = 1
        other_index = 0
    clash_atom = clash_list[clash_tuple[tuple_index]]['atom']
    if len(clash_tuple) == 1:
        other_atom = np.nan
    else:
        other_atom = clash_list[clash_tuple[other_index]]['atom']

    suite_number = [clash_list[clash_tuple[tuple_index]]['suite_1'],
                    clash_list[clash_tuple[tuple_index]]['suite_2']].index(suite._number_first_residue)
    if len(clash_tuple) > 1 and suite._number_first_residue in [clash_list[clash_tuple[other_index]]['suite_1'],
                                                                clash_list[clash_tuple[other_index]]['suite_2']]:
        suite_number_other = [clash_list[clash_tuple[other_index]]['suite_1'],
                              clash_list[clash_tuple[other_index]]['suite_2']].index(suite._number_first_residue)
    else:
        suite_number_other = None

    clash_info = clash_info_function(suite_number_other=suite_number_other, clash_list=clash_list, clash_tuple=clash_tuple,
                                     other_index=other_index, suite=suite)
    my_list = [clash_list[clash_tuple[tuple_index]]['suite_1'], clash_list[clash_tuple[tuple_index]]['suite_2']]
    if len(clash_tuple) > 1:
        other_list = [clash_list[clash_tuple[other_index]]['suite_1'], clash_list[clash_tuple[other_index]]['suite_2']]
    else:
        other_list = None

    suite.clash_list.append({'clash_atom': clash_atom, 'other_atom': other_atom,
                             'suite_number': suite_number,
                             'my_list': my_list,
                             'other_list': other_list,
                             'clash_score': float(clash_list[clash_tuple[0]]['clashmag']),
                             'clash_distance': float(clash_list[clash_tuple[0]]['dist']),
                             'clash_info': clash_info,
                             'suite_number_other': suite_number_other})


def clash_info_function(suite_number_other, clash_list, clash_tuple, other_index, suite):
    """
    This function classifies a clash. It returns 'one_suite_clash' if the clash atoms are in the same suite,
    'non_RNA_clash' if one of the atoms are not an RNA atom, 'local_neighbor' if a maximum of one suite is between the
    two atoms and 'not_local' if more than one suite lies between the two atoms.
    :param suite_number_other: None if not a one_suite_clash. Else 0 or 1.
    :param clash_list: A list with clashes. Each element is a dictionary with clash information.
    :param clash_tuple:  A tuple with two integers (indices of clash_list).
    :param other_index: An integer. The index of the corresponding clash element.
    :param suite: A suite object.
    :return: A string from ['one_suite_clash', 'non_RNA_clash', 'local_neighbor', 'not_local']
    """
    if suite_number_other is not None:
        return 'one_suite_clash'
    if len(clash_tuple) == 1:
        return 'non_RNA_clash'
    int_numbers = [suite._number_first_residue + int_value for int_value in [-2, -1, 1, 2]]
    if len(set(int_numbers) &
           {clash_list[clash_tuple[other_index]]['suite_1'], clash_list[clash_tuple[other_index]]['suite_2']}):
        return 'local_neighbor'
    return 'not_local'


def work_with_clashes(suites):
    """
    This function works with a list of suites. It adds the clash information of a clash to suite.bb_bb_one_suite, if the
    suite has a clash between two backbone atoms in the same suite. It adds the clash information of a clash to
    suite.bb_bb_neighbour_clashes if there is a clash between two backbone atoms in the mesoscopic shape.
    :param suites: A list with suite objects.
    :return: A modified list with suite objects.
    """
    # Filter out the all suites that have a clash between two backbone atoms:
    for suite in suites:
        for clash in suite.clash_list:
            if clash['clash_atom'] in BACKBONE_ATOMS_VALIDATION and clash['other_atom'] in BACKBONE_ATOMS_VALIDATION \
                    and clash['clash_info'] == 'one_suite_clash':
                suite.bb_bb_one_suite.append(clash)

    # Filter out the all suites that have a backbone backbone clash inside the mesoscopic shape.
    for suite_index in range(len(suites)):
        for clash in suites[suite_index].clash_list:
            if clash['clash_atom'] in BACKBONE_ATOMS_VALIDATION and clash['other_atom'] in BACKBONE_ATOMS_VALIDATION:
                suites[suite_index].bb_bb_neighbour_clashes.append(clash)
                for suite_neighbor_index in range(suite_index-5, suite_index+5):
                    if suites[suite_neighbor_index].name in suites[suite_index]._list_of_neighbours:
                        suites[suite_neighbor_index].bb_bb_neighbour_clashes.append(clash)

    return suites


