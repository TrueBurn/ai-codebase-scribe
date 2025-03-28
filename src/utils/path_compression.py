"""
Path compression utilities for reducing token usage when sending file paths to LLMs.
"""

import re
from typing import Dict, List, Tuple


def compress_paths(file_paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Compress a list of file paths by replacing common prefixes with shorter keys.
    
    Args:
        file_paths: List of file paths to compress
        
    Returns:
        Tuple containing:
            - List of compressed paths
            - Dictionary mapping compression keys to their original values
    """
    # Identify common prefixes
    common_prefixes = _identify_common_prefixes(file_paths)
    
    # Sort prefixes by length (longest first) to ensure proper replacement
    sorted_prefixes = sorted(common_prefixes.items(), key=lambda x: len(x[0]), reverse=True)
    
    # Create compression mapping
    compression_map = {}
    for i, (prefix, _) in enumerate(sorted_prefixes):
        key = f"@{i+1}"
        compression_map[key] = prefix
    
    # Compress paths
    compressed_paths = []
    for path in file_paths:
        compressed = path
        for key, prefix in compression_map.items():
            compressed = compressed.replace(prefix, key)
        compressed_paths.append(compressed)
    
    # Invert the map for decompression
    decompression_map = {v: k for k, v in compression_map.items()}
    
    return compressed_paths, decompression_map


def decompress_paths(compressed_paths: List[str], decompression_map: Dict[str, str]) -> List[str]:
    """
    Decompress paths using the provided decompression map.
    
    Args:
        compressed_paths: List of compressed paths
        decompression_map: Dictionary mapping compression keys to their original values
        
    Returns:
        List of decompressed paths
    """
    decompressed_paths = []
    for path in compressed_paths:
        decompressed = path
        for prefix, key in decompression_map.items():
            decompressed = decompressed.replace(key, prefix)
        decompressed_paths.append(decompressed)
    
    return decompressed_paths


def _identify_common_prefixes(file_paths: List[str]) -> Dict[str, int]:
    """
    Identify common prefixes in a list of file paths.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Dictionary mapping common prefixes to their frequency
    """
    # Common Java/Maven package prefixes
    common_patterns = [
        r'src/main/java/[^/]+/[^/]+/[^/]+/',
        r'src/test/java/[^/]+/[^/]+/[^/]+/',
        r'src/main/resources/',
        r'src/test/resources/'
    ]
    
    prefix_counts = {}
    
    # Find matches for common patterns
    for pattern in common_patterns:
        for path in file_paths:
            matches = re.findall(pattern, path.replace('\\', '/'))
            for match in matches:
                if match in prefix_counts:
                    prefix_counts[match] += 1
                else:
                    prefix_counts[match] = 1
    
    # Only keep prefixes that appear multiple times
    return {k: v for k, v in prefix_counts.items() if v > 1}


def get_compression_explanation(decompression_map: Dict[str, str]) -> str:
    """
    Generate an explanation of the compression scheme for the LLM.
    
    Args:
        decompression_map: Dictionary mapping compression keys to their original values
        
    Returns:
        String explaining the compression scheme
    """
    explanation = "Note: To save tokens, file paths have been compressed using the following scheme:\n\n"
    for key, value in decompression_map.items():
        explanation += f"- {key} = {value}\n"
    explanation += "\nFor example, @1/Controller.java represents src/main/java/com/example/project/Controller.java\n"
    
    return explanation