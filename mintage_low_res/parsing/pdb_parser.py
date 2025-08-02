"""
Main PDB parsing interface.

Provides the clean public API for parsing PDB files into Suite objects,
with caching and error handling.
"""

import os
from typing import List, Optional
from pathlib import Path

from ..data.models import Suite
from ..data.cache import CacheManager
from ..utils.io_utils import find_pdb_files
from .suite_builder import SuiteBuilder


def parse_pdb_files(input_dir: str, 
                   cache_dir: Optional[str] = None,
                   force_recompute: bool = False) -> List[Suite]:
    """
    Parse all PDB files in a directory and return Suite objects.
    
    Args:
        input_dir: Directory containing PDB files
        cache_dir: Optional cache directory for pickle files
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        List of Suite objects representing RNA nucleotide pairs
    """
    parser = PDBFileParser(cache_dir=cache_dir)
    return parser.parse_directory(input_dir, force_recompute=force_recompute)


class PDBFileParser:
    """Main PDB file parser with caching capabilities."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize parser with optional cache directory."""
        self.cache_manager = CacheManager(cache_dir) if cache_dir else None
        self.suite_builder = SuiteBuilder()
    
    def parse_directory(self, input_dir: str, force_recompute: bool = False) -> List[Suite]:
        """Parse all PDB files in a directory."""
        cache_key = f"suites_{Path(input_dir).name}"
        
        # Try to load from cache
        if not force_recompute and self.cache_manager and self.cache_manager.exists(cache_key):
            cached_suites = self.cache_manager.load(cache_key)
            if cached_suites:
                return cached_suites
        
        # Parse all PDB files
        pdb_files = find_pdb_files(input_dir)
        all_suites = []
        
        for pdb_file in pdb_files:
            try:
                suites = self.parse_file(pdb_file)
                all_suites.extend(suites)
            except Exception as e:
                print(f"Warning: Failed to parse {pdb_file}: {e}")
                continue
        
        # Post-process to handle neighbor relationships
        all_suites = self._post_process_suites(all_suites)
        
        # Cache results
        if self.cache_manager:
            self.cache_manager.save(cache_key, all_suites)
        
        return all_suites
    
    def parse_file(self, filename: str) -> List[Suite]:
        """Parse a single PDB file into Suite objects."""
        return self.suite_builder.build_suites_from_file(filename)
    
    def _post_process_suites(self, suites: List[Suite]) -> List[Suite]:
        """
        Post-process suites to handle neighbor relationships and validation.
        
        This removes suites where neighbors are side chains or have gaps.
        """
        if len(suites) < 3:
            return suites
        
        # Mark invalid suites based on neighbor relationships
        for i in range(len(suites) - 2):
            current = suites[i]
            next_suite = suites[i + 1]
            
            # Check if residue numbers are consecutive
            if not self._are_consecutive_residues(current, next_suite):
                # Mark surrounding suites as incomplete
                for j in range(max(0, i-2), min(len(suites), i+3)):
                    suites[j].mesoscopic_sugar_rings = [None]
                    suites[j].complete_suite = False
        
        return suites
    
    def _are_consecutive_residues(self, suite1: Suite, suite2: Suite) -> bool:
        """Check if two suites have consecutive residue numbers."""
        diff = abs(suite1._number_first_residue - suite2._number_first_residue)
        return diff == 1 and suite1._name_chain == suite2._name_chain