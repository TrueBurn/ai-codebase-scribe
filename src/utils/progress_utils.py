#!/usr/bin/env python3

"""
Utilities for progress tracking and reporting.
"""

from pathlib import Path
from typing import Any, Optional

from src.utils.progress import ProgressTracker


def create_file_processing_progress_bar(total: int) -> Any:
    """Create a progress bar for file processing.
    
    Args:
        total: Total number of files to process
        
    Returns:
        A progress bar instance
    """
    progress_tracker = ProgressTracker.get_instance()
    return progress_tracker.progress_bar(
        total=total,
        desc="Processing files",
        unit="file",
        ncols=150,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        colour='green'
    )


def create_optimization_progress_bar() -> Any:
    """Create a progress bar for optimization.
    
    Returns:
        A progress bar instance
    """
    progress_tracker = ProgressTracker.get_instance()
    return progress_tracker.progress_bar(
        total=1,
        desc="Optimizing file order",
        unit="step",
        ncols=150,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        colour='blue'
    )


def create_documentation_progress_bar(repo_path: Optional[Path] = None) -> Any:
    """Create a progress bar for documentation generation.
    
    Args:
        repo_path: Optional path to the repository
        
    Returns:
        A progress bar instance
    """
    progress_tracker = ProgressTracker.get_instance(repo_path)
    return progress_tracker.progress_bar(
        total=2,
        desc="Generating documentation",
        unit="file",
        ncols=150,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        colour='yellow'
    )