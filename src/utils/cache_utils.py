#!/usr/bin/env python3

"""
Utilities for cache management and statistics.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.cache import CacheManager
from src.analyzers.codebase import CodebaseAnalyzer
from src.utils.config_class import ScribeConfig


def display_cache_stats(cache_stats: Dict[str, int], total_elapsed: float, cache_enabled: bool) -> None:
    """Display cache statistics.
    
    Args:
        cache_stats: Dictionary containing cache statistics (from_cache, from_llm, skipped)
        total_elapsed: Total elapsed time in seconds
        cache_enabled: Whether caching is enabled
    """
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    total_hrs, total_mins = divmod(total_mins, 60)
    
    if total_hrs > 0:
        total_time_str = f"{total_hrs:02d}:{total_mins:02d}:{total_secs:02d}"
    else:
        total_time_str = f"{total_mins:02d}:{total_secs:02d}"
    
    if cache_enabled:
        total_processed = cache_stats["from_cache"] + cache_stats["from_llm"]
        if total_processed > 0:
            cache_percent = (cache_stats["from_cache"] / total_processed) * 100
            print(f"\nCache statistics: {cache_stats['from_cache']} files from cache ({cache_percent:.1f}%), " +
                  f"{cache_stats['from_llm']} files processed by LLM, {cache_stats['skipped']} binary files skipped")
            print(f"Total processing time: {total_time_str}")


def display_github_cache_stats(config: ScribeConfig, analyzer: CodebaseAnalyzer) -> None:
    """Display GitHub repository cache statistics.
    
    Args:
        config: ScribeConfig instance
        analyzer: CodebaseAnalyzer instance
    """
    if config and hasattr(config, 'github_repo_id') and config.github_repo_id:
        print("\nCache statistics for GitHub repository:")
        print(f"Repository ID: {config.github_repo_id}")
        print(f"Cache directory: {analyzer.cache.get_repo_cache_dir()}")