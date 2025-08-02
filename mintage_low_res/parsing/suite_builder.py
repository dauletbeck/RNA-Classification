"""
Suite builder for creating Suite objects from atomic data.

Handles the construction of Suite objects with proper validation
and mesoscopic shape calculations.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..data.models import Suite
from ..config.constants import BASES, TWO_RING_BASES, ONE_RING_BASES, SUGAR_ATOMS
from .atom_extractor import AtomExtractor


class SuiteBuilder:
    """Builds Suite objects from extracted atomic data."""
    
    def __init__(self):
        """Initialize suite builder."""
        self.atom_extractor = AtomExtractor()
    
    def build_suites_from_file(self, filename: str) -> List[Suite]:
        """
        Build all valid Suite objects from a PDB file.
        
        Args:
            filename: Path to PDB file
            
        Returns:
            List of Suite objects
        """
        # Parse PDB file
        parsed_data = self.atom_extractor.parse_pdb_file(filename)
        
        atom_dict = parsed_data['atom_dict']
        residue_types = parsed_data['residue_types']
        head_residues = parsed_data['head_residues']
        tail_residues = parsed_data['tail_residues']
        suite_names = parsed_data['suite_names']
        atom_types = parsed_data['atom_types']
        
        all_suites = []
        file_basename = Path(filename).stem[:4]  # First 4 characters of filename
        
        # Process each chain
        for i, head in enumerate(head_residues):
            tail = tail_residues[i]
            total_residues = tail - head + 1
            
            # Create suites for consecutive residue pairs
            for j in range(1, total_residues):
                this = j + head - 1
                
                # Validate residue types
                if not self._validate_residue_pair(residue_types, this):
                    continue
                
                # Check for consecutive residue numbers
                if not self._check_consecutive_residues(suite_names, this):
                    continue
                
                # Extract atomic data
                suite_data = self.atom_extractor.extract_suite_atoms(
                    atom_dict, atom_types, residue_types, this
                )
                
                # Calculate mesoscopic data if enough context
                mesoscopic_data = self._calculate_mesoscopic_data(
                    atom_dict, residue_types, this, i, head_residues, tail_residues
                )
                
                # Build Suite object
                suite = self._build_suite(
                    suite_data=suite_data,
                    mesoscopic_data=mesoscopic_data,
                    filename=file_basename,
                    suite_names=suite_names,
                    this=this
                )
                
                if suite:
                    all_suites.append(suite)
        
        return all_suites
    
    def _validate_residue_pair(self, residue_types: Dict, this: int) -> bool:
        """Validate that both residues in the pair are valid RNA bases."""
        base1 = residue_types.get(this)
        base2 = residue_types.get(this + 1)
        
        return (base1 in BASES and base2 in BASES)
    
    def _check_consecutive_residues(self, suite_names: Dict, this: int) -> bool:
        """Check if residue numbers are consecutive."""
        try:
            name1 = suite_names.get(this, "")
            name2 = suite_names.get(this + 1, "")
            
            if not name1 or not name2:
                return False
            
            # Extract residue numbers
            num1 = int(name1[1:].strip())
            num2 = int(name2[1:].strip())
            
            return abs(num2 - num1) == 1
        except (ValueError, IndexError):
            return False
    
    def _calculate_mesoscopic_data(self, atom_dict: Dict, residue_types: Dict, 
                                 this: int, chain_idx: int, 
                                 head_residues: List, tail_residues: List) -> Optional[np.ndarray]:
        """
        Calculate mesoscopic sugar ring coordinates.
        
        Requires 6 consecutive residues (this-2 to this+3) to calculate
        the mean sugar coordinates.
        """
        head = head_residues[chain_idx]
        tail = tail_residues[chain_idx]
        
        # Check if we have enough residues for mesoscopic calculation
        if (this - head < 2 or tail - this < 3):
            return None
        
        # Check that all required residues exist
        required_residues = list(range(this - 2, this + 4))
        rna_residues = atom_dict.get(' O2*', {})
        
        for res_num in required_residues:
            if res_num not in rna_residues or not rna_residues[res_num]:
                return None
        
        try:
            # Calculate mean sugar coordinates for 6 consecutive residues
            sugar_coords = []
            for offset in [-2, -1, 0, 1, 2, 3]:
                residue_num = this + offset
                residue_sugar = []
                
                for atom_name in SUGAR_ATOMS:
                    coords = atom_dict.get(atom_name, {}).get(residue_num)
                    if coords is None:
                        return None
                    residue_sugar.append(coords)
                
                sugar_coords.append(residue_sugar)
            
            # Calculate mean coordinates for each residue
            mesoscopic_rings = np.mean(sugar_coords, axis=1)
            return mesoscopic_rings
            
        except Exception:
            return None
    
    def _build_suite(self, suite_data: Dict, mesoscopic_data: Optional[np.ndarray],
                    filename: str, suite_names: Dict, this: int) -> Optional[Suite]:
        """
        Build a Suite object from extracted data.
        
        Args:
            suite_data: Dictionary of extracted atomic data
            mesoscopic_data: Mesoscopic sugar ring coordinates
            filename: PDB filename
            suite_names: Dictionary of residue names
            this: Current residue position
            
        Returns:
            Suite object or None if invalid
        """
        try:
            name_residue_1 = suite_names.get(this, "")
            name_residue_2 = suite_names.get(this + 1, "")
            
            if not name_residue_1 or not name_residue_2:
                return None
            
            # Handle mesoscopic data
            if mesoscopic_data is not None:
                mesoscopic_sugar_rings = mesoscopic_data.tolist()
            else:
                mesoscopic_sugar_rings = [None]
            
            # Create Suite object
            suite = Suite(
                backbone_atoms=suite_data['backbone_atoms'],
                backbone_hydrogen_atoms=suite_data['backbone_hydrogen_atoms'],
                oxygen_atoms=suite_data['oxygen_atoms'],
                ring_atoms=suite_data['ring_atoms'],
                ring_hydrogen_atoms=suite_data['ring_hydrogen_atoms'],
                mesoscopic_sugar_rings=mesoscopic_sugar_rings,
                dihedral_angles_chi=suite_data['dihedral_angles_chi'],
                filename=filename,
                nu_1=suite_data['nu_1_atoms'],
                nu_2=suite_data['nu_2_atoms'],
                name_residue_1=name_residue_1,
                name_residue_2=name_residue_2,
                five_chain=suite_data['five_chain'],
                six_chain=suite_data['six_chain'],
                seven_chain=suite_data['seven_chain'],
                atom_types=suite_data['atom_types']
            )
            
            return suite
            
        except Exception as e:
            print(f"Warning: Failed to build suite at position {this}: {e}")
            return None