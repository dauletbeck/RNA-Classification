"""
Data models for RNA structure analysis.

Cleaned up versions of the original Suite class and supporting structures,
focused only on low-resolution analysis needs.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from ..utils.geometry import dihedral, rotation


@dataclass
class AtomData:
    """Represents a single atom with its properties."""
    name: str
    coordinates: np.ndarray
    atom_type: str  # ATOM or HETATM
    residue_number: int
    chain_id: str


@dataclass
class Residue:
    """Represents a single RNA residue."""
    number: int
    base_type: str  # U, C, G, A
    chain_id: str
    atoms: Dict[str, AtomData]
    
    def get_atom_coords(self, atom_name: str) -> Optional[np.ndarray]:
        """Safely get atom coordinates."""
        atom = self.atoms.get(atom_name)
        return atom.coordinates if atom else None


class Suite:
    """
    Represents a pair of consecutive RNA nucleotides.
    
    Simplified version of the original Suite class, focusing only on
    low-resolution analysis requirements.
    """
    
    def __init__(self, 
                 backbone_atoms: List[np.ndarray],
                 backbone_hydrogen_atoms: List[np.ndarray],
                 oxygen_atoms: List[np.ndarray], 
                 ring_atoms: List[np.ndarray],
                 ring_hydrogen_atoms: List[np.ndarray],
                 mesoscopic_sugar_rings: List[np.ndarray],
                 dihedral_angles_chi: List[float],
                 filename: str,
                 nu_1: List[np.ndarray], 
                 nu_2: List[np.ndarray],
                 name_residue_1: str, 
                 name_residue_2: str,
                 five_chain: List[np.ndarray], 
                 six_chain: List[np.ndarray],
                 seven_chain: List[np.ndarray], 
                 atom_types: str):
        
        # Core atomic data
        self._backbone_atoms = np.array(backbone_atoms)
        self._backbone_hydrogen_atoms = np.array(backbone_hydrogen_atoms)
        self._oxygen_atoms = np.array(oxygen_atoms)
        self._ring_atoms = np.array(ring_atoms)
        self._ring_hydrogen_atoms = np.array(ring_hydrogen_atoms)
        
        # Low-resolution chain representations
        self._five_chain = np.array(five_chain)  # N-C-P-C-N
        self._six_chain = np.array(six_chain)
        self._seven_chain = np.array(seven_chain)
        
        # Sugar and base data
        self.mesoscopic_sugar_rings = mesoscopic_sugar_rings
        self._dihedral_angles_chi = dihedral_angles_chi
        
        # Calculate dihedral angles
        if None not in self._backbone_atoms:
            self._dihedral_angles = np.array([
                dihedral(backbone_atoms[i:i + 4], rna_distances=False) 
                for i in range(len(backbone_atoms) - 3)
            ])
        else:
            self._dihedral_angles = [None]
        
        # Sugar pucker angles
        self._nu_1 = [dihedral(nu_1, rna_distances=False)] if None not in nu_1 else [None]
        self._nu_2 = [dihedral(nu_2, rna_distances=False)] if None not in nu_2 else [None]
        
        # Metadata
        self._filename = filename
        self._name = f"{filename}_{name_residue_1}_{name_residue_2}".replace(' ', '')
        self._name_chain = name_residue_1[:1].replace(' ', '')
        self._number_first_residue = int(name_residue_1[1:].replace(' ', ''))
        self._number_second_residue = int(name_residue_2[1:].replace(' ', ''))
        self.atom_types = atom_types
        
        # Check completeness
        self.complete_suite = self._check_completeness()
        
        # Initialize analysis-specific attributes
        self.low_res_coords = None
        self.low_res_direction1 = None
        self.low_res_direction2 = None
        self.pucker = None
        self.pucker_distance_1 = None
        self.pucker_distance_2 = None
        
    def _check_completeness(self) -> bool:
        """Check if the suite has all required data for analysis."""
        variables = vars(self).copy()
        variables.pop("atom_types", None)
        return not any(None in variables[key] for key in variables if hasattr(variables[key], '__iter__'))
    
    def low_resolution_coordinates(self) -> np.ndarray:
        """
        Calculate low-resolution coordinates: [d2, d3, alpha, theta1, phi1, theta2, phi2]
        
        Returns:
            Array of 7 low-resolution coordinates
        """
        if self.low_res_coords is not None:
            return self.low_res_coords.copy()
            
        deg = 180 / np.pi
        
        # Center the P atom (middle of five-chain N-C-P-C-N)
        NCPCN = self._five_chain - self._five_chain[2][np.newaxis, :]
        
        # Get the normal direction to the connecting line between the C atoms
        long = NCPCN[3] - NCPCN[1]  # C2 - C1 direction
        long = long / np.linalg.norm(long)
        
        normal = NCPCN[1]  # C1 position
        normal = normal - np.dot(normal, long) * long  # Project out long component
        normal = normal / np.linalg.norm(normal)
        
        # Rotate normal to y-axis
        rot1 = rotation(normal, np.array([0, 1, 0]))
        NCPCN = np.einsum('ij,nj->ni', rot1, NCPCN)
        
        # Rotate connecting line to x-axis
        long = NCPCN[3] - NCPCN[1]
        long = long / np.linalg.norm(long)
        rot2 = np.array([[long[0], 0, long[2]], 
                        [0, 1, 0], 
                        [-long[2], 0, long[0]]])
        NCPCN = np.einsum('ij,nj->ni', rot2, NCPCN)
        
        # Calculate distances and angle
        d2_d3 = [np.linalg.norm(NCPCN[1]), np.linalg.norm(NCPCN[3])]
        alpha = np.arccos(np.dot(NCPCN[1], NCPCN[3]) / (d2_d3[0] * d2_d3[1])) * deg
        
        # Calculate base directions
        CNs = [NCPCN[0] - NCPCN[1], NCPCN[4] - NCPCN[3]]  # N-C vectors
        CNs = [v / np.linalg.norm(v) for v in CNs]
        
        self.low_res_direction1 = CNs[0]
        self.low_res_direction2 = CNs[1]
        
        # Calculate spherical angles
        thetas = [np.arccos(v[0]) * deg for v in CNs]
        phis = [np.arctan2(v[1], v[2]) * deg for v in CNs]
        
        self.low_res_coords = d2_d3 + [alpha, thetas[0], phis[0], thetas[1], phis[1]]
        
        return self.low_res_coords.copy()
    
    def get_pucker_distances(self) -> tuple:
        """Get sugar pucker distances for classification."""
        if self.pucker_distance_1 is None or self.pucker_distance_2 is None:
            # This would be calculated based on nu angles
            # Implementation depends on the specific pucker analysis method
            pass
        return self.pucker_distance_1, self.pucker_distance_2
    
    def __str__(self) -> str:
        return f"Suite({self._name}, complete={self.complete_suite})"
    
    def __repr__(self) -> str:
        return self.__str__()