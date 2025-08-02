"""I/O utilities for file operations and directory management."""

import os
import fnmatch
from pathlib import Path
from typing import List


def find_pdb_files(folder: str) -> List[str]:
    """
    Find all PDB files in a directory and subdirectories.
    
    Args:
        folder: Root directory to search
        
    Returns:
        List of PDB file paths
    """
    pdb_files = []
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, '*.pdb'):
            pdb_files.append(os.path.join(root, filename))
    return sorted(pdb_files)


def create_output_directory(output_dir: str) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory path to create
        
    Returns:
        Path object for the created directory
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_basename(filepath: str) -> str:
    """Extract basename without extension from file path."""
    return Path(filepath).stem


def ensure_directory_exists(filepath: str) -> None:
    """Ensure the directory for a file path exists."""
    directory = Path(filepath).parent
    directory.mkdir(parents=True, exist_ok=True)