import numpy as np
import re
from utils.data_functions import dihedral
from utils.data_functions import rotation

class Suite:
    def __init__(self, backbone_atoms, backbone_hydrogen_atoms, oxygen_atoms, ring_atoms, ring_hydrogen_atoms,
                 mesoscopic_sugar_rings, dihedral_angles_chi, filename, nu_1, nu_2, name_residue_1, name_residue_2,
                 five_chain, six_chain, seven_chain, atom_types):

        # Microscopic level:
        # All consecutive backbone atoms.
        self.atom_types = atom_types
        self._backbone_atoms = np.array(backbone_atoms)
        # The hydrogen atoms which have an atomic bond with the backbone atoms.
        self._backbone_hydrogen_atoms = np.array(backbone_hydrogen_atoms)
        # The two oxygen atoms which have an atomic bond with the phosphate atom.
        self._oxygen_atoms = np.array(oxygen_atoms)
        # The ring atoms.
        self._ring_atoms = np.array(ring_atoms)

        # The low res chains
        self._five_chain = np.array(five_chain) # N-C-P-C-N
        self._six_chain = np.array(six_chain)
        self._seven_chain = np.array(seven_chain)

        # The hydrogen atoms which have an atomic bond with the ring atoms.
        self._ring_hydrogen_atoms = np.array(ring_hydrogen_atoms)
        # The dihedral angles of all consecutive backbone atoms.
        if None not in self._backbone_atoms:
            self._dihedral_angles = np.array([dihedral(backbone_atoms[i:i + 4], rna_distances=False) for i in range(len(backbone_atoms)-3)])
        else:
            self._dihedral_angles = [None]

        if None not in nu_1:
            self._nu_1 = [dihedral(nu_1, rna_distances=False)]
        else:
            self._nu_1 = [None]
        if None not in nu_2:
            self._nu_2 = [dihedral(nu_2, rna_distances=False)]
        else:
            self._nu_2 = [None]

        # The base angles Chi 1 and Chi 2
        self._dihedral_angles_chi = dihedral_angles_chi
        # The coordinates of the mesoscopic shape:
        self.mesoscopic_sugar_rings = mesoscopic_sugar_rings
        # Check if the suite is complete:
        variables = vars(self).copy()
        try:
            variables.pop("atom_types")
        except Exception:
            pass
        # self.complete_suite = (True not in [None in vars(self)[key] for key in vars(self)])
        self.complete_suite = (True not in [None in variables[key] for key in variables])
        # The name of the suite: The first four letter correspond to the filename. A suite has atoms from two
        # consecutive residues: The rest of the name is composed of the chain name and the number of the residue
        # of the first residue plus the same for the second residue.
        self._name = (filename + '_' + name_residue_1 + '_' + name_residue_2).replace(' ', '')
        # The names of the neighbour suites of the current suite.
        self._list_of_neighbours = determine_neighbors(filename, name_residue_1, name_residue_2)
        self._number_first_residue = int(name_residue_1[1:].replace(' ', '')) # int(re.findall(r'\d+', name_residue_1[1:])[-1])
        self._number_second_residue = int(name_residue_2[1:].replace(' ', '')) #int(re.findall(r'\d+', name_residue_2[1:])[-1])
        print(int(name_residue_1[1:].replace(' ', '')))
        #self._number_first_residue = int(re.findall(r'\d+', name_residue_1)[-1])
        #self._number_second_residue = int(re.findall(r'\d+', name_residue_2)[-1])
        #self._name_chain = name_residue_1[:re.search(r"\d", name_residue_1).start()].replace(' ', '')
        self._name_chain = name_residue_1[:1].replace(' ', '')
        self._filename = filename
        # clash information:
        self.clash_list = []
        self.bb_bb_one_suite = []
        self.bb_bb_neighbour_clashes = []
        # The resolution of the experiment
        self.resolution = None
        # Procrustes information
        # The Procrustes algorithm for all suites with self.complete_suite==True: Suite:
        self.procrustes_complete_suite_shift = None
        self.procrustes_complete_suite_scale = None
        self.procrustes_complete_suite_rotation = None
        self.procrustes_complete_suite_vector = None
        # The Procrustes algorithm for all suites with self.complete_suite==True: Mesoscopic:
        self.procrustes_complete_mesoscopic_shift = None
        self.procrustes_complete_mesoscopic_scale = None
        self.procrustes_complete_mesoscopic_rotation = None
        self.procrustes_complete_mesoscopic_vector = None
        # The procrustes algorithm for shape space: Suite
        self.procrustes_complete_suite_shape_space_shift = None
        self.procrustes_complete_suite_shape_space_scale = None
        self.procrustes_complete_suite_shape_space_rotation = None
        self.procrustes_complete_suite_shape_space_vector = None
        # The procrustes algorithm for shape space: Mesoscopic
        self.procrustes_complete_mesoscopic_shape_space_shift = None
        self.procrustes_complete_mesoscopic_shape_space_scale = None
        self.procrustes_complete_mesoscopic_shape_space_rotation = None
        self.procrustes_complete_mesoscopic_shape_space_vector = None

        # For all suites which have a complete backbone_atom array:
        self.procrustes_suite_shift = None
        self.procrustes_suite_scale = None
        self.procrustes_suite_rotation = None
        self.procrustes_suite_vector = None

        self.clustering = {}
        self.validation_dict = {}
        self.mesoscopic_sub_procrustes = {}

        self.base_pairs = {}
        self.erraser = {}

        # a dictionary to save information which are from the phenix tool 'clashscore'
        self.clashscore = {}
        self.model_number = None
        # CLEAN information:
        self.clean = {}

        self.procrustes_five_chain_shift = None
        self.procrustes_five_chain_scale = None
        self.procrustes_five_chain_rotation = None
        self.procrustes_five_chain_vector = None


        self.procrustes_six_chain_shift = None
        self.procrustes_six_chain_scale = None
        self.procrustes_six_chain_rotation = None
        self.procrustes_six_chain_vector = None

        self.procrustes_seven_chain_shift = None
        self.procrustes_seven_chain_scale = None
        self.procrustes_seven_chain_rotation = None
        self.procrustes_seven_chain_vector = None

        self.low_res_coords = None
        self.low_res_direction1 = None
        self.low_res_direction2 = None

        self.answer = None
        self.pucker = None
        self.pucker_distance_1 = None
        self.pucker_distance_2 = None

    def low_resolution_coordinates (self):
        if (not hasattr(self,'low_res_coords') or (self.low_res_coords == None)):
            deg = 180 / np.pi
            # Center the P atom
            NCPCN = self._five_chain - self._five_chain[2][np.newaxis,:]
            # Get the normal direction to the connecting line between the C atoms
            long = NCPCN[3] - NCPCN[1]
            long = long / np.linalg.norm(long)
            normal = NCPCN[1]
            normal = normal - np.dot(normal, long) * long
            normal = normal / np.linalg.norm(normal)
            # Get the rotation matrix that rotates the normal to the y axis
            rot1 = rotation(normal, np.array([0, 1, 0]))
            NCPCN = np.einsum('ij,nj->ni', rot1, NCPCN)
            # Get the direction of the connecting line between the C atoms
            long = NCPCN[3] - NCPCN[1]
            long = long / np.linalg.norm(long)
            # Get the rotation matrix that rotates the vector to the x axis
            rot2 = np.array([[long[0], 0, long[2]], [0, 1, 0], [-long[2], 0, long[0]]])
            NCPCN = np.einsum('ij,nj->ni', rot2, NCPCN)
            d2_d3 = [np.linalg.norm(NCPCN[1]), np.linalg.norm(NCPCN[3])]
            alpha = np.arccos(np.dot(NCPCN[1], NCPCN[3]) / (d2_d3[0] * d2_d3[1]))*deg
            CNs = [NCPCN[0] - NCPCN[1], NCPCN[4] - NCPCN[3]]
            CNs = [v / np.linalg.norm(v) for v in CNs]
            self.low_res_direction1 = CNs[0]
            self.low_res_direction2 = CNs[1]
            thetas = [np.arccos(v[0])*deg for v in CNs]
            phis = [np.arctan2(v[1], v[2])*deg for v in CNs]
            self.low_res_coords = d2_d3 + [alpha, thetas[0], phis[0], thetas[1], phis[1]]
        return self.low_res_coords.copy()

    def get_NCPs(self):
        tmp = self._five_chain
        NCPCNP = self._six_chain
        NCP1 = NCPCNP[:3] - NCPCNP[1][np.newaxis,:]
        l1 = NCP1[0] - NCP1[1]
        l1 = l1/ np.linalg.norm(l1)
        rot1 = rotation(l1, np.array([-1, 0, 0]))
        NCP1 = np.einsum('ij,nj->ni', rot1, NCP1)
        l2 = NCP1[2].copy()
        l2[0] = 0
        l2 = l2/ np.linalg.norm(l2)
        rot2= rotation(l2, np.array([0, 1, 0]))
        NCP1 = np.einsum('ij,nj->ni', rot2, NCP1)
        NCP2 = NCPCNP[np.array([4,3,5])] - NCPCNP[3][np.newaxis,:]
        l1 = NCP2[0] - NCP2[1]
        l1 = l1/ np.linalg.norm(l1)
        rot1 = rotation(l1, np.array([-1, 0, 0]))
        NCP2 = np.einsum('ij,nj->ni', rot1, NCP2)
        l2 = NCP2[2].copy()
        l2[0] = 0
        l2 = l2/ np.linalg.norm(l2)
        rot2= rotation(l2, np.array([0, 1, 0]))
        NCP2 = np.einsum('ij,nj->ni', rot2, NCP2)
        return np.array([[NCP1[0,0], NCP1[2,0], NCP1[2,1]],
                         [NCP2[0,0], NCP2[2,0], NCP2[2,1]]])

    @property
    def backbone_atoms(self):
        return self._backbone_atoms

    @property
    def backbone_hydrogen_atoms(self):
        return self._backbone_hydrogen_atoms

    @property
    def oxygen_atoms(self):
        return self._oxygen_atoms

    @property
    def ring_atoms(self):
        return self._ring_atoms

    @property
    def ring_hydrogen_atoms(self):
        return self._ring_hydrogen_atoms

    @property
    def dihedral_angles(self):
        return self._dihedral_angles

    @property
    def name(self):
        return self._name

    @property
    def dihedral_angles_chi(self):
        return self._dihedral_angles_chi

    @property
    def list_of_neighbours(self):
        return self._list_of_neighbours


def determine_neighbors(filename, name_residue_1, name_residue_2):
    """
    This function determines the names of the neighbour suites of the current suite.
    :param filename:
    :param name_residue_1:
    :param name_residue_2:
    :return:
    """
    number_first_residue = int(re.findall(r'\d+', name_residue_1[1:])[-1])
    number_second_residue = int(re.findall(r'\d+', name_residue_2[1:])[-1])
    #name_chain = name_residue_1[:re.search(r"\d", name_residue_1).start()].replace(' ', '')
    name_chain = name_residue_1[:1].replace(' ', '')
    # Sometimes the numbers of the residues are in reverse order
    integer_residue = number_second_residue - number_first_residue
    list_of_neighbours = tuple([
        (filename + '_' + name_chain + str(number_first_residue + integer_residue * i) + '_' +
         name_chain + str(number_second_residue + integer_residue * i)).replace(' ', '') for i in [-2, -1, 1, 2]])
    return list_of_neighbours
