import platform

import numpy as np
import os
import fnmatch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.constants import RELEVANT_ATOMS, RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS, RELEVANT_RING_ATOMS, \
    RELEVANT_RING_ATOMS_HYDROGEN, RELEVANT_OXYGEN_ATOMS, SUGAR_ATOMS, RELEVANT_ATOMS_ONE_RING, \
    RELEVANT_ATOMS_TWO_RING, ONE_RING_BASES, TWO_RING_BASES, BASES
from utils.data_functions import dihedral
from utils import Suite_class


def import_pdb_file(filename, verbose=False, folder_specified=None):
    """
    This function uses __parse_pdb to get some dictionaries and calculates the important angles and values with the
    function __calc_values_and_angles.
    :param filename: The name/path of the pdb-file.
    :param verbose: If False no documentation.
    :param folder_specified: solves Windows problem with \ and /
    :return: A np.array of shape [nr.residues, nr_angles]
    """
    #try:
    atom_dict, _, residue_types, head_residues, tail_residues, suite_names, atom_types = __parse_pdb(filename, verbose) #
    all_suites = []
    # i = 0,1,2,3,... head = head_residues[0], head_residues[1],...
    for i, head in enumerate(head_residues):
        # the number of residues:
        total_residues = tail_residues[i] - head + 1
        for j in range(1, total_residues):
            this = j + head - 1

            if platform.system() == "Windows" and folder_specified is None:
                suite = __calc_values_and_angles(atom_dict=atom_dict, residue_types=residue_types, this=this, i=i,
                                                 j=j, tail_residues=tail_residues, suite_names=suite_names,
                                                 filename=filename[filename.rfind('\\') + 1:filename.rfind('\\') + 5],
                                                 atom_types=atom_types)
            else:
                # linux, mac
                suite = __calc_values_and_angles(atom_dict=atom_dict, residue_types=residue_types, this=this, i=i,
                                                 j=j, tail_residues=tail_residues, suite_names=suite_names,
                                                 filename=filename[filename.rfind('/') + 1:filename.rfind('/') + 5],
                                                 atom_types=atom_types)
            if suite is not None:
                all_suites.append(suite)
    if len(all_suites) < 1:
        print('WARNING! NO RNA FOUND IN', filename)
    #except Exception as e:
    #    print('An error occured in ' + filename + ":")
    #    print(e)
    #    return []
    return all_suites  # np.array(all_angles)


def __parse_pdb(filename, verbose):
    """
    This function gets a filename and returns dictionaries.
    :param filename: The name/path of the pdb-file.
    :param verbose: If False no documentation.
    :return: atom_dict: A dictionary with all coordinates of each atom type.
             residue_types: a dictionary containing all information about the type of each residuum.
             head_residues: A list of the first numbers of a related molecule.
             tail_residues: A list of the last numbers of a related molecule.
    """
    residues = {}
    chains = {}
    residue_types = {}
    head_residues = []
    tail_residues = []
    atom_dict = {name: {} for name in (RELEVANT_ATOMS + RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS + RELEVANT_RING_ATOMS +
                                       RELEVANT_RING_ATOMS_HYDROGEN + RELEVANT_OXYGEN_ATOMS)}
    suite_names = {}
    residue_dict = {}
    atom_type = ''
    n_atom = 0
    n_residue = 0
    no_chain = True
    atom_types = {name: {} for name in (RELEVANT_ATOMS + RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS + RELEVANT_RING_ATOMS +
                                       RELEVANT_RING_ATOMS_HYDROGEN + RELEVANT_OXYGEN_ATOMS)}

    with open(filename) as datafile:
        for line in datafile:

            # Only the atom positions:
            if line[:4] != 'ATOM' and line[:6] != 'HETATM':
                continue
            type_temp = line[:6]
            # atom_types.append()

            # only atoms with a permissible base:
            if not line[17:20] in BASES:
                if verbose:
                    print('Unknown base type: ' + line[17:20])
            type_test = line[12:16].replace("'", "*")
            residue_test = line[21:27]
            # if not (line[21:27].replace(' ', ''))[-1].isdigit():
            #     print('skipped side-chain', line[21:27])
            #     continue

            # Check for duplicates:
            if (type_test == atom_type and n_atom in residues and
                    residues[n_atom] in residue_test):
                if verbose:
                    print('Skipped duplicate:', line[:27])
                continue
            # The number of the atom:
            n_atom += 1
            # The name of the atom (for example ' O5*'):
            atom_type = type_test
            # Example for residues: {1: 'Q   0'}
            residues[n_atom] = residue_test[:-1]
            # Example for chains: {1: 'Q'}
            chains[n_atom] = residue_test[:1]
            # Check if new residue:
            if (n_atom - 1 not in residues or
                    residues[n_atom] != residues[n_atom - 1]):
                # the number
                n_residue += 1
                # Example: residues[n_atom][1:] = '   0' => int(residues[n_atom][1:]=0:
                residue_dict[int(residues[n_atom][1:])] = n_residue
                # residues_types = {1: '  C'}
                residue_types[n_residue] = line[17:20]

            # Save the atom coordinates for the relevant atoms:
            if atom_type in RELEVANT_ATOMS:
                atom_dict[atom_type][n_residue] = [float(line[30:38]),
                                                   float(line[38:46]),
                                                   float(line[46:54])]
                atom_types[atom_type][n_residue] = str(type_temp)

            if (atom_type in RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS or
                    atom_type in RELEVANT_RING_ATOMS or
                    atom_type in RELEVANT_RING_ATOMS_HYDROGEN or
                    atom_type in RELEVANT_OXYGEN_ATOMS):
                atom_dict[atom_type][n_residue] = [float(line[30:38]),
                                                   float(line[38:46]),
                                                   float(line[46:54])]

            suite_names[n_residue] = line[21:27]
            if no_chain:
                head_residues.append(n_residue)
                no_chain = False
            # Check if the name of the residue changed:
            elif chains[n_atom] != chains[n_atom - 1]:
                head_residues.append(n_residue)
                tail_residues.append(n_residue - 1)
    tail_residues.append(n_residue)
    return atom_dict, residue_dict, residue_types, head_residues, tail_residues, suite_names, atom_types


def help_function_return(atom_dict, a, this):
    try:
        return atom_dict[a][this]
    except:
        return [None, None, None]


def __calc_values_and_angles(atom_dict, residue_types, this, i, j, tail_residues, suite_names, filename, atom_types):
    """
    Calculates the dihedral angles of the suite number 'this'. It also calculates the dihedral chi angle for the
    suite this-2,...,this+3 and for the same suits the centered mean of the sugar.
    :param atom_dict: Dictionary of all atom positions.
    :param residue_types: Dictionary with the base-information.
    :param this: Integer
    :param verbose: If False no documentation.
    :return: A vector with the angles and the positions of the mean of the sugar.
    """

    # RNA is a subset of {1:[x_1, x_2, x_3],...,nr_residues:[x_1, x_2, x_3]} which contains only the rna-numbers.
    rna = atom_dict[' O2*']
    # We delete the suite if the suite is not complete.
    complete = True
    if ((this not in rna) or ((this + 1) not in rna) or
            rna[this] == 0 or rna[this + 1] == 0):
        return None

    if not residue_types[this] in TWO_RING_BASES + ONE_RING_BASES:
        print("Skipping unknown nucleic base:", residue_types[this])
        return None

    atom_types_suite = ([help_function_return(atom_types, a, this) for a in RELEVANT_ATOMS[2:6]] +
                      [help_function_return(atom_types, a, this + 1) for a in RELEVANT_ATOMS[:6]])

    backbone_atoms = ([help_function_return(atom_dict, a, this) for a in RELEVANT_ATOMS[2:6]] +
                      [help_function_return(atom_dict, a, this + 1) for a in RELEVANT_ATOMS[:6]])

    backbone_hydrogen_atoms = ([help_function_return(atom_dict, a, this) for a in
                                RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS] +
                               [help_function_return(atom_dict, a, this + 1) for a in
                                RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS])

    oxygen_atoms = ([help_function_return(atom_dict, a, this + 1) for a in RELEVANT_OXYGEN_ATOMS])

    ring_atoms = ([help_function_return(atom_dict, a, this) for a in RELEVANT_RING_ATOMS] +
                  [help_function_return(atom_dict, a, this + 1) for a in RELEVANT_RING_ATOMS])

    ring_hydrogen_atoms = ([help_function_return(atom_dict, a, this) for a in RELEVANT_RING_ATOMS_HYDROGEN] +
                           [help_function_return(atom_dict, a, this + 1) for a in RELEVANT_RING_ATOMS_HYDROGEN])

    nu_1_atoms = [help_function_return(atom_dict, a, this) for a in SUGAR_ATOMS[:-1]]
    nu_2_atoms = [help_function_return(atom_dict, a, this + 1) for a in SUGAR_ATOMS[:-1]]
    name_residue_1 = suite_names[this]
    name_residue_2 = suite_names[this + 1]

    # five chain data
    # if ((j > 2) and (this < tail_residues[i] - 2)):
    # test if the atoms of the five chain actually belong together or are from a different suite
    num_residue1 = int(name_residue_1[1:].replace(" ", ""))
    num_residue2 = int(name_residue_2[1:].replace(" ", ""))
    if abs(num_residue2-num_residue1) > 1:
        return None
        # cords_five_chain = [None]
    else:
        base_atom_1 = RELEVANT_ATOMS_TWO_RING[2] if residue_types[this] in TWO_RING_BASES else RELEVANT_ATOMS_ONE_RING[2]
        ribose_c_1 = SUGAR_ATOMS[3]
        base_atom_2 = RELEVANT_ATOMS_TWO_RING[2] if residue_types[this + 1] in TWO_RING_BASES else RELEVANT_ATOMS_ONE_RING[2]
        phosphate = RELEVANT_ATOMS[0]
        ATOMS_SIX_CHAIN_1 = [base_atom_1, ribose_c_1]
        ATOMS_SIX_CHAIN_2 = [phosphate, ribose_c_1, base_atom_2]
        cords_six_chain_1 = [help_function_return(atom_dict, a, this) for a in ATOMS_SIX_CHAIN_1]
        cords_six_chain_2 = [help_function_return(atom_dict, a, this + 1) for a in ATOMS_SIX_CHAIN_2]
        cords_five_chain = cords_six_chain_1 + cords_six_chain_2  # + cords_six_chain_3
        hetatoms = 0
        atms = 0
        atm_types = None
        for atm in atom_types_suite:
            if atm == 'HETATM':
                hetatoms += 1
            else:
                atms += 1
        if hetatoms > 0 and atms > 0:
            atm_types = 'mix'
        elif hetatoms > 0:
            atm_types = 'het'
        elif atms > 0:
            atm_types = 'atm'


    #else:
    #    print("parse_pdb else (five chain)")
    #    cords_five_chain = [None]

    # Check if there are adjacent residues
    if ((j > 2) and (this < tail_residues[i] - 2) and this - 2 in rna
            and this - 1 in rna and this + 2 in rna and this + 3 in rna):
        int_list = [-2, -1, 0, 1, 2, 3]
        # Calculate the mean:
        atoms_of_six_sugar = [[atom_dict[a][this + b] for a in SUGAR_ATOMS] for b in int_list]
        # Calculate the (X_1)_mean,...,(X_6)_mean
        mesoscopic_sugar_rings = np.mean(atoms_of_six_sugar, axis=1)
        # base_atom_1 = RELEVANT_ATOMS_TWO_RING[2] if residue_types[this] in TWO_RING_BASES else RELEVANT_ATOMS_ONE_RING[
        #     2]
        # ribose_c_1 = SUGAR_ATOMS[3]
        # base_atom_2 = RELEVANT_ATOMS_TWO_RING[2] if residue_types[this + 1] in TWO_RING_BASES else \
        # RELEVANT_ATOMS_ONE_RING[2]
        phosphate = RELEVANT_ATOMS[0]

        ATOMS_SIX_CHAIN_0 = [phosphate]
        # ATOMS_SIX_CHAIN_1 = [base_atom_1, ribose_c_1]
        # ATOMS_SIX_CHAIN_2 = [phosphate, ribose_c_1, base_atom_2]
        ATOMS_SIX_CHAIN_3 = [phosphate]
        cords_six_chain_0 = [help_function_return(atom_dict, a, this - 1) for a in ATOMS_SIX_CHAIN_0]
        # cords_six_chain_1 = [help_function_return(atom_dict, a, this) for a in ATOMS_SIX_CHAIN_1]
        # cords_six_chain_2 = [help_function_return(atom_dict, a, this + 1) for a in ATOMS_SIX_CHAIN_2]
        cords_six_chain_3 = [help_function_return(atom_dict, a, this + 2) for a in ATOMS_SIX_CHAIN_3]
        # cords_five_chain = cords_six_chain_1 + cords_six_chain_2  # + cords_six_chain_3
        cords_six_chain = cords_six_chain_1 + cords_six_chain_2 + cords_six_chain_3
        cords_seven_chain = cords_six_chain_0 + cords_six_chain_1 + cords_six_chain_2 + cords_six_chain_3
    else:
        mesoscopic_sugar_rings = [None]
        # cords_five_chain = [None]
        cords_six_chain = [None]
        cords_seven_chain = [None]
    try:
        ATOMS_SIX_CHAIN_3 = [phosphate]
        cords_six_chain_3 = [help_function_return(atom_dict, a, this + 2) for a in ATOMS_SIX_CHAIN_3]
        cords_six_chain = cords_six_chain_1 + cords_six_chain_2 + cords_six_chain_3
    except:
        cords_six_chain = [None]
    int_list = [0, 1]
    # Calculate Chi for the two bases: The name of the two bases of the suite:
    relevant_bases = [residue_types[this + b] for b in int_list]
    # Check if there is a missing relevant atom:
    all_atoms_are_available = False not in [(this + int_list[k]) in atom_dict[a] for k in range(2) for a in
                                            map_bases(relevant_bases[k])]
    # Check if the two bases are known:
    # Sometimes one base is not in BASES then return dihedral_angles =[None, None]:
    if set(relevant_bases).issubset(BASES) and all_atoms_are_available:
        dihedral_angles_chi = [dihedral([atom_dict[a][this + int_list[i]] for a in map_bases(relevant_bases[i])], False)
                               for i in range(2)]
    else:
        dihedral_angles_chi = [None for i in range(2)]

    actual_suite = Suite_class.Suite(backbone_atoms=backbone_atoms, backbone_hydrogen_atoms=backbone_hydrogen_atoms,
                                     oxygen_atoms=oxygen_atoms, ring_atoms=ring_atoms,
                                     ring_hydrogen_atoms=ring_hydrogen_atoms,
                                     mesoscopic_sugar_rings=mesoscopic_sugar_rings,
                                     dihedral_angles_chi=dihedral_angles_chi, filename=filename, nu_1=nu_1_atoms,
                                     nu_2=nu_2_atoms, name_residue_1=name_residue_1, name_residue_2=name_residue_2,
                                     five_chain=cords_five_chain, six_chain=cords_six_chain,
                                     seven_chain=cords_seven_chain, atom_types=atm_types)
    return actual_suite


def map_bases(input_base):
    """
    This function the name of a base to a list of names of relevant atoms.
    :param input_base: The name of a base as a string.
    :return: A list of strings.
    """
    if input_base in ONE_RING_BASES:
        return RELEVANT_ATOMS_ONE_RING
    else:
        return RELEVANT_ATOMS_TWO_RING


def find_files(pattern, folder='.'):
    """
    This function find all files in the directory and subdirectories and returns the list of filenames.
    :param pattern: The file-extension to search for.
    :param folder: The root/starting folder.
    :return: A list of all filenames.
    """
    filenames = []
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, pattern):
            filenames.append(os.path.join(root, filename))
    return filenames


def delete_shapes(data):
    """
    This function deletes the coordinates of the shapes if one of the neighbored suits has been skipped.
    :param data: A np.array with the dimension (number of residues) x 29
    :return: A np.array with the dimension (number of residues) x 29
    """
    for i in range(len(data) - 2):
        if not (data[i, data.shape[1] - 2] + 1 == data[i + 1, data.shape[1] - 2]):
            data[i, 9:27] = [np.nan] * 18
            data[i - 1, 9:27] = [np.nan] * 18
            data[i + 1, 9:27] = [np.nan] * 18
            data[i + 2, 9:27] = [np.nan] * 18
    return data


def get_all_pdb_files(folder):
    """
    This function load all pdb files from the path 'folder' and generates a list with suite objects.
    :param folder: A string.
    """
    # files is a list of filenames (of all .pdb files)
    if folder is None:
        files = find_files('*.pdb')
    else:
        files = find_files('*.pdb', folder=folder)
    files.sort()
    # create list: The items are the file names of the pdb-data. (rfind give the number of the last occurrence
    data = [x for x in [import_pdb_file(f, folder_specified=folder) for f in files]]
    data = [suite for list_ in data for suite in list_]
    # Remove suites where the neighbors are site chains.
    for i in range(len(data) - 2):
        if not (int(data[i]._number_first_residue) + 1 == int(data[i + 1]._number_first_residue) or
                int(data[i]._number_first_residue) - 1 == int(data[i + 1]._number_first_residue)) or not \
                data[i]._name[-1].isdigit():
            data[i - 2].mesoscopic_sugar_rings = [None]
            data[i - 2].complete_suite = False
            data[i - 1].mesoscopic_sugar_rings = [None]
            data[i - 1].complete_suite = False
            data[i].mesoscopic_sugar_rings = [None]
            data[i].complete_suite = False
            data[i + 1].mesoscopic_sugar_rings = [None]
            data[i + 1].complete_suite = False
            data[i + 2].mesoscopic_sugar_rings = [None]
            data[i + 2].complete_suite = False

    return data

if __name__=="__main__":
    pdb_dir = "/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/data/rna2020_pruned_pdbs/"
    print(get_all_pdb_files(pdb_dir))