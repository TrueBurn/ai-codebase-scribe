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
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        colour="green",
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
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        colour="blue",
    )


def create_documentation_progress_bar(repo_path: Optional[Path] = None) -> Any:
    """Create a no-op context manager for clean documentation generation logging.

    Progress bars interfered with clean, informative logs during document generation.
    File processing and repo analysis progress bars are kept for their usefulness.

    Args:
        repo_path: Optional path to the repository

    Returns:
        A no-op context manager (no progress bar for cleaner logs)
    """
    from contextlib import nullcontext

    return nullcontext()


def create_migration_analysis_progress_bar(
    total: int, repo_path: Optional[Path] = None
) -> Any:
    """Create a progress bar for migration analysis.

    Args:
        total: Total number of migration files to analyze
        repo_path: Optional path to the repository

    Returns:
        A progress bar instance
    """
    progress_tracker = ProgressTracker.get_instance(repo_path)
    return progress_tracker.progress_bar(
        total=total,
        desc="Analyzing migrations",
        unit="file",
        ncols=150,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        colour="cyan",
    )
