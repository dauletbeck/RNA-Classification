"""
Atom extraction from PDB data.

Handles the extraction of specific atoms needed for Suite creation,
with proper error handling and validation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from ..config.constants import (
    RELEVANT_ATOMS, RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS,
    RELEVANT_RING_ATOMS, RELEVANT_RING_ATOMS_HYDROGEN, 
    RELEVANT_OXYGEN_ATOMS, SUGAR_ATOMS, BASES,
    RELEVANT_ATOMS_ONE_RING, RELEVANT_ATOMS_TWO_RING,
    ONE_RING_BASES, TWO_RING_BASES
)


class AtomExtractor:
    """Extracts specific atom coordinates for Suite creation."""
    
    def __init__(self):
        """Initialize atom extractor."""
        self.relevant_atoms = set(RELEVANT_ATOMS + RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS + 
                                RELEVANT_RING_ATOMS + RELEVANT_RING_ATOMS_HYDROGEN + 
                                RELEVANT_OXYGEN_ATOMS)
    
    def parse_pdb_file(self, filename: str) -> Dict[str, Any]:
        """
        Parse a PDB file and extract atomic data.
        
        Returns:
            Dictionary containing atom_dict, residue_types, etc.
        """
        atom_dict = {name: {} for name in self.relevant_atoms}
        residue_types = {}
        head_residues = []
        tail_residues = []
        suite_names = {}
        atom_types = {name: {} for name in self.relevant_atoms}
        
        residues = {}
        chains = {}
        n_atom = 0
        n_residue = 0
        no_chain = True
        
        with open(filename, 'r') as datafile:
            for line in datafile:
                if line[:4] != 'ATOM' and line[:6] != 'HETATM':
                    continue
                
                type_temp = line[:6]
                
                # Check for valid RNA base
                if line[17:20] not in BASES:
                    continue
                
                atom_name = line[12:16].replace("'", "*")
                residue_info = line[21:27]
                
                # Check for duplicates
                if (atom_name in atom_dict and n_atom in residues and 
                    residues[n_atom] in residue_info):
                    continue
                
                n_atom += 1
                residues[n_atom] = residue_info[:-1]
                chains[n_atom] = residue_info[:1]
                
                # Check if new residue
                if (n_atom - 1 not in residues or 
                    residues[n_atom] != residues[n_atom - 1]):
                    n_residue += 1
                    residue_types[n_residue] = line[17:20]
                
                # Save atom coordinates
                if atom_name in self.relevant_atoms:
                    coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    atom_dict[atom_name][n_residue] = coords
                    atom_types[atom_name][n_residue] = type_temp
                
                suite_names[n_residue] = line[21:27]
                
                # Track chain boundaries
                if no_chain:
                    head_residues.append(n_residue)
                    no_chain = False
                elif chains[n_atom] != chains[n_atom - 1]:
                    head_residues.append(n_residue)
                    tail_residues.append(n_residue - 1)
        
        if n_residue > 0:
            tail_residues.append(n_residue)
        
        return {
            'atom_dict': atom_dict,
            'residue_types': residue_types,
            'head_residues': head_residues,
            'tail_residues': tail_residues,
            'suite_names': suite_names,
            'atom_types': atom_types
        }
    
    def extract_suite_atoms(self, atom_dict: Dict, atom_types: Dict, 
                          residue_types: Dict, this: int) -> Dict[str, Any]:
        """
        Extract all atomic data needed for a Suite at position 'this'.
        
        Args:
            atom_dict: Dictionary of atomic coordinates
            atom_types: Dictionary of atom types (ATOM/HETATM)
            residue_types: Dictionary of residue types
            this: Current residue position
            
        Returns:
            Dictionary with all extracted atomic data
        """
        # Helper function to safely get atom coordinates
        def get_atom(atom_name: str, residue_num: int) -> Optional[List[float]]:
            try:
                return atom_dict[atom_name][residue_num]
            except KeyError:
                return [None, None, None]
        
        # Extract backbone atoms (current and next residue)
        backbone_atoms = (
            [get_atom(a, this) for a in RELEVANT_ATOMS[2:6]] +
            [get_atom(a, this + 1) for a in RELEVANT_ATOMS[:6]]
        )
        
        # Extract hydrogen atoms
        backbone_hydrogen_atoms = (
            [get_atom(a, this) for a in RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS] +
            [get_atom(a, this + 1) for a in RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS]
        )
        
        # Extract oxygen atoms
        oxygen_atoms = [get_atom(a, this + 1) for a in RELEVANT_OXYGEN_ATOMS]
        
        # Extract ring atoms
        ring_atoms = (
            [get_atom(a, this) for a in RELEVANT_RING_ATOMS] +
            [get_atom(a, this + 1) for a in RELEVANT_RING_ATOMS]
        )
        
        # Extract ring hydrogen atoms
        ring_hydrogen_atoms = (
            [get_atom(a, this) for a in RELEVANT_RING_ATOMS_HYDROGEN] +
            [get_atom(a, this + 1) for a in RELEVANT_RING_ATOMS_HYDROGEN]
        )
        
        # Extract sugar atoms for nu angles
        nu_1_atoms = [get_atom(a, this) for a in SUGAR_ATOMS[:-1]]
        nu_2_atoms = [get_atom(a, this + 1) for a in SUGAR_ATOMS[:-1]]
        
        # Extract chi angle atoms
        dihedral_angles_chi = self._extract_chi_angles(
            atom_dict, residue_types, this
        )
        
        # Extract chain representations
        five_chain = self._extract_five_chain(atom_dict, residue_types, this)
        
        # Determine atom types
        atom_types_suite = (
            [atom_types.get(a, {}).get(this, 'ATOM') for a in RELEVANT_ATOMS[2:6]] +
            [atom_types.get(a, {}).get(this + 1, 'ATOM') for a in RELEVANT_ATOMS[:6]]
        )
        
        mixed_types = self._determine_atom_type_mix(atom_types_suite)
        
        return {
            'backbone_atoms': backbone_atoms,
            'backbone_hydrogen_atoms': backbone_hydrogen_atoms,
            'oxygen_atoms': oxygen_atoms,
            'ring_atoms': ring_atoms,
            'ring_hydrogen_atoms': ring_hydrogen_atoms,
            'nu_1_atoms': nu_1_atoms,
            'nu_2_atoms': nu_2_atoms,
            'dihedral_angles_chi': dihedral_angles_chi,
            'five_chain': five_chain,
            'six_chain': [None],  # Simplified for now
            'seven_chain': [None],  # Simplified for now
            'atom_types': mixed_types
        }
    
    def _extract_chi_angles(self, atom_dict: Dict, residue_types: Dict, this: int) -> List[Optional[float]]:
        """Extract chi dihedral angles for base atoms."""
        from ..utils.geometry import dihedral
        
        relevant_bases = [residue_types.get(this), residue_types.get(this + 1)]
        chi_angles = []
        
        for i, base_type in enumerate(relevant_bases):
            if base_type not in BASES:
                chi_angles.append(None)
                continue
            
            # Get relevant atoms for this base type
            base_atoms = (RELEVANT_ATOMS_TWO_RING if base_type in TWO_RING_BASES 
                         else RELEVANT_ATOMS_ONE_RING)
            
            residue_num = this + i
            atom_coords = []
            
            for atom_name in base_atoms:
                coords = atom_dict.get(atom_name, {}).get(residue_num)
                if coords is None:
                    atom_coords = None
                    break
                atom_coords.append(coords)
            
            if atom_coords and len(atom_coords) == 4:
                chi_angle = dihedral(atom_coords, rna_distances=False)
                chi_angles.append(chi_angle)
            else:
                chi_angles.append(None)
        
        return chi_angles
    
    def _extract_five_chain(self, atom_dict: Dict, residue_types: Dict, this: int) -> List[Optional[List[float]]]:
        """Extract five-chain coordinates (N-C-P-C-N)."""
        def get_atom(atom_name: str, residue_num: int) -> Optional[List[float]]:
            return atom_dict.get(atom_name, {}).get(residue_num)
        
        # Get base atoms for each residue
        base_atom_1 = (RELEVANT_ATOMS_TWO_RING[2] if residue_types.get(this) in TWO_RING_BASES 
                      else RELEVANT_ATOMS_ONE_RING[2])
        base_atom_2 = (RELEVANT_ATOMS_TWO_RING[2] if residue_types.get(this + 1) in TWO_RING_BASES 
                      else RELEVANT_ATOMS_ONE_RING[2])
        
        ribose_c = SUGAR_ATOMS[3]  # C1*
        phosphate = RELEVANT_ATOMS[0]  # P
        
        # Build five-chain: N1-C1*-P-C1*-N2
        five_chain = [
            get_atom(base_atom_1, this),        # N1
            get_atom(ribose_c, this),           # C1*
            get_atom(phosphate, this + 1),      # P
            get_atom(ribose_c, this + 1),       # C1*
            get_atom(base_atom_2, this + 1)     # N2
        ]
        
        return five_chain
    
    def _determine_atom_type_mix(self, atom_types_suite: List[str]) -> str:
        """Determine if suite has mixed atom types."""
        hetatm_count = sum(1 for t in atom_types_suite if t == 'HETATM')
        atom_count = sum(1 for t in atom_types_suite if t == 'ATOM')
        
        if hetatm_count > 0 and atom_count > 0:
            return 'mix'
        elif hetatm_count > 0:
            return 'het'
        else:
            return 'atm'