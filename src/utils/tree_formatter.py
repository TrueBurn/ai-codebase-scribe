"""Tree formatting utilities for project structure visualization."""
import logging
from typing import Dict, List, Optional, Tuple, Any

def format_tree_structure(file_manifest: Dict[str, Any]) -> str:
    """
    Format a file manifest into a visually clear tree structure.
    
    Args:
        file_manifest: Dictionary mapping file paths to file information
        
    Returns:
        A formatted string representing the project structure with clear hierarchy
    """
    logging.debug("Formatting project structure as tree")
    try:
        # Get file paths from manifest
        file_paths = [str(path).replace('\\', '/') for path in file_manifest.keys()]
        
        # Create a nested dictionary to represent the directory structure
        root = {}
        
        for path in file_paths:
            parts = path.split('/')
            current = root
            
            # Build the tree structure
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # Last part (file)
                    if '__files__' not in current:
                        current['__files__'] = []
                    current['__files__'].append(part)
                else:  # Directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        # Format the tree
        lines = []
        _format_tree_node(root, lines, "", "")
        
        return '\n'.join(lines)
    except Exception as e:
        logging.error(f"Error formatting tree structure: {e}")
        return f"Error formatting tree structure: {e}"

def _format_tree_node(node: Dict, lines: List[str], prefix: str, name: str) -> None:
    """
    Recursively format a tree node.
    
    Args:
        node: Dictionary representing a directory or file
        lines: List to append formatted lines to
        prefix: Prefix string for the current line (for drawing the tree)
        name: Name of the current node
    """
    if name:
        # Add the current node to the output
        if name != "__files__":
            lines.append(f"{prefix}{name}")
    
    # Process files in this directory
    if '__files__' in node:
        files = sorted(node['__files__'])
        for i, file in enumerate(files):
            is_last = i == len(files) - 1 and len([k for k in node.keys() if k != '__files__']) == 0
            if is_last:
                lines.append(f"{prefix}└── {file}")
            else:
                lines.append(f"{prefix}├── {file}")
    
    # Process subdirectories
    dirs = sorted([k for k in node.keys() if k != '__files__'])
    for i, dir_name in enumerate(dirs):
        is_last = i == len(dirs) - 1
        
        if is_last:
            # Last item gets a corner piece
            lines.append(f"{prefix}└── {dir_name}/")
            _format_tree_node(node[dir_name], lines, f"{prefix}    ", "")
        else:
            # Non-last items get a T-piece and vertical line
            lines.append(f"{prefix}├── {dir_name}/")
            _format_tree_node(node[dir_name], lines, f"{prefix}│   ", "")

def format_project_structure(file_paths: List[str]) -> str:
    """
    Format a list of file paths into a visually clear tree structure.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        A formatted string representing the project structure with clear hierarchy
    """
    # Create a nested dictionary to represent the directory structure
    root = {}
    
    for path in file_paths:
        path = path.replace('\\', '/')
        parts = path.split('/')
        current = root
        
        # Build the tree structure
        for i, part in enumerate(parts):
            if i == len(parts) - 1:  # Last part (file)
                if '__files__' not in current:
                    current['__files__'] = []
                current['__files__'].append(part)
            else:  # Directory
                if part not in current:
                    current[part] = {}
                current = current[part]
    
    # Format the tree
    lines = []
    _format_tree_node(root, lines, "", "")
    
    return '\n'.join(lines)