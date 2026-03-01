#!/usr/bin/env python3

# Standard library imports
import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party imports
import urllib3
from dotenv import load_dotenv

# Set environment variable for libmagic regex memory limit
os.environ["MAGIC_REGEX_MEMORY"] = os.environ.get("MAGIC_REGEX_MEMORY", "10000000")

# Local imports
from src.analyzers.codebase import CodebaseAnalyzer
from src.clients.base_llm import BaseLLMClient
from src.clients.llm_factory import LLMClientFactory
from src.generators.architecture import generate_architecture
from src.generators.contributing import generate_contributing
from src.generators.persistence import generate_persistence, analyze_persistence_layer
from src.analyzers.persistence import has_meaningful_persistence_content
from src.generators.readme import generate_readme
from src.generators.installation import generate_installation
from src.generators.usage import generate_usage
from src.generators.troubleshooting import generate_troubleshooting
from src.utils.readme_refactor import (
    refactor_readme_for_split_docs,
    add_navigation_section_to_readme,
)
from src.models.file_info import FileInfo
from src.utils.badges import generate_badges
from src.utils.cache import CacheManager
from src.utils.cache_utils import display_cache_stats, display_github_cache_stats
from src.utils.config_utils import load_config, update_config_with_args
from src.utils.config_class import ScribeConfig
from src.utils.doc_utils import (
    add_ai_attribution,
    insert_badges_after_title,
    detect_existing_badges,
)
from src.utils.exceptions import (
    ScribeError,
    ConfigurationError,
    RepositoryError,
    FileProcessingError,
    LLMError,
    GitHubError,
)
from src.utils.github_utils import (
    is_valid_github_url,
    clone_github_repository,
    create_git_branch,
    commit_documentation_changes,
    push_branch_to_remote,
    create_pull_request,
    extract_repo_info,
    prepare_github_branch,
    find_existing_pr,
    close_pull_request,
    delete_branch,
)
from src.utils.progress_utils import (
    create_file_processing_progress_bar,
    create_optimization_progress_bar,
    create_documentation_progress_bar,
)
from src.utils.visual_logger import setup_visual_logging, get_visual_logger

# Load environment variables from .env file
load_dotenv()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# Disable bytecode caching
sys.dont_write_bytecode = True


def setup_logging(
    debug: bool = False, log_to_file: bool = True, quiet: bool = False
) -> None:
    """Configure logging with visual enhancements.

    Args:
        debug: If True, sets logging level to DEBUG, otherwise INFO
        log_to_file: If True, logs to both file and console, otherwise just console
        quiet: If True, reduces console output verbosity
    """
    # Set up visual logging which handles both console and file logging
    visual_logger = setup_visual_logging(
        debug=debug,
        log_to_file=log_to_file,
        quiet=quiet,
        enable_rich=True,
    )

    if debug:
        visual_logger.debug("Debug logging enabled with rich formatting")


# Function moved: process_file is now a nested function within process_files


def _sort_files_by_cache_status(
    files: List[Path], analyzer: CodebaseAnalyzer, repo_path: Path
) -> tuple[List[Path], List[Path], Dict[str, Any]]:
    """Sort files by cache status: cached first, uncached last.

    This function efficiently determines which files have valid cache entries
    and which need LLM processing, providing statistics about cache effectiveness.

    Args:
        files: List of all files to process
        analyzer: CodebaseAnalyzer instance with cache access
        repo_path: Repository root path for relative path calculations

    Returns:
        Tuple of (cached_files, uncached_files, sort_stats) where:
        - cached_files: Files with valid cache entries to process first
        - uncached_files: Files needing LLM processing to process second
        - sort_stats: Dictionary with cache effectiveness statistics
    """
    vlogger = get_visual_logger()

    # Filter out files that should be skipped (binary, excluded)
    processable_files = []
    for file_path in files:
        if analyzer.is_binary(file_path):
            continue
        if not analyzer.should_include_file(file_path):
            continue
        processable_files.append(file_path)

    if not processable_files:
        return (
            [],
            [],
            {
                "total_files": 0,
                "cached_files": 0,
                "uncached_files": 0,
                "check_time": 0.0,
                "cache_percent": 0.0,
            },
        )

    # Batch check cache status
    vlogger.info(
        f"Checking cache status for {vlogger.format_number(len(processable_files))} files...",
        emoji="analyze",
    )

    start_time = time.time()
    cached_files, uncached_files = analyzer.cache.check_files_cache_status(
        processable_files
    )
    check_time = time.time() - start_time

    # Calculate statistics
    sort_stats = {
        "total_files": len(processable_files),
        "cached_files": len(cached_files),
        "uncached_files": len(uncached_files),
        "check_time": check_time,
        "cache_percent": (
            (len(cached_files) / len(processable_files) * 100)
            if processable_files
            else 0.0
        ),
    }

    # Log cache status results
    if cached_files:
        vlogger.info(
            f"Found {vlogger.format_number(len(cached_files))} cached files "
            f"({sort_stats['cache_percent']:.1f}%) - processing these first",
            emoji="cache",
        )
    if uncached_files:
        vlogger.info(
            f"Found {vlogger.format_number(len(uncached_files))} uncached files "
            f"({100 - sort_stats['cache_percent']:.1f}%) - will process via LLM",
            emoji="processing",
        )

    return cached_files, uncached_files, sort_stats


async def process_files(
    repo_path: Path,
    llm_client: BaseLLMClient,
    config: ScribeConfig,
    file_list: Optional[List[Path]] = None,
) -> Dict[str, FileInfo]:
    """Process files with cache-prioritized order for optimal UX and performance.

    NEW BEHAVIOR:
    - Cached files processed first (fast - no LLM calls)
    - Uncached files processed second (slow - LLM processing)
    - Separate statistics for each phase
    - Immediate visible progress

    Args:
        repo_path: Path to the repository
        llm_client: LLM client to use for processing
        config: Configuration
        file_list: Optional list of files to process (if None, all files are processed)

    Returns:
        Dictionary mapping file paths to FileInfo objects
    """
    # Initialize analyzer
    analyzer = CodebaseAnalyzer(repo_path, config)

    # Get visual logger instance once for this function
    vlogger = get_visual_logger()

    # Get file manifest
    file_manifest = {}

    # Cache statistics with phase tracking
    cache_stats = {
        "from_cache": 0,
        "from_llm": 0,
        "skipped": 0,
        "time_limit_reached": False,
        # Phase tracking
        "cached_phase_time": 0.0,
        "uncached_phase_time": 0.0,
        "cached_phase_count": 0,
        "uncached_phase_count": 0,
    }

    # Start timing for batch processing
    start_time = time.time()
    time_limit_seconds = (
        config.large_repo.time_limit_minutes * 60
        if config.large_repo.batch_processing
        else None
    )

    # Get list of files to process
    if file_list is None:
        files = analyzer._get_repository_files()
    else:
        files = file_list

    # Determine processing order (existing optimization)
    if config.optimize_order:
        vlogger.info("Optimizing file processing order...", emoji="analyze")
        with create_optimization_progress_bar() as progress:
            files = await determine_processing_order(files, llm_client, progress)

    # Sort by cache status if caching enabled
    cached_files = []
    uncached_files = []

    if config.no_cache or not analyzer.cache.enabled:
        # Cache disabled - process all files normally
        uncached_files = files
        vlogger.info(
            f"Processing {vlogger.format_number(len(files))} files (caching disabled)...",
            emoji="processing",
        )
    else:
        # Split files by cache status
        cached_files, uncached_files, sort_stats = _sort_files_by_cache_status(
            files, analyzer, repo_path
        )

    total_files = len(cached_files) + len(uncached_files)

    # Edge case: No files to process
    if total_files == 0:
        vlogger.warning("No files to process after filtering", emoji="warning")
        display_cache_stats(cache_stats, 0, analyzer.cache.enabled)
        return file_manifest

    # Define nested function for file processing
    async def process_file(file_path: Path, is_cached: bool) -> Optional[FileInfo]:
        """Process a single file with time limit checking.

        Args:
            file_path: Path to the file
            is_cached: Whether this file should have a valid cache entry

        Returns:
            FileInfo object or None if file should be skipped
        """
        # Check time limit before processing each file
        if time_limit_seconds and (time.time() - start_time) >= time_limit_seconds:
            cache_stats["time_limit_reached"] = True
            return None

        # Skip binary files
        if analyzer.is_binary(file_path):
            cache_stats["skipped"] += 1
            return None

        # Skip files that should be excluded
        if not analyzer.should_include_file(file_path):
            cache_stats["skipped"] += 1
            return None

        # Check if file has been processed before
        if is_cached and analyzer.cache.enabled:
            cached_summary = analyzer.cache.get_cached_summary(file_path)
            if cached_summary and not analyzer.cache.is_file_changed(file_path):
                # Use cached summary
                cache_stats["from_cache"] += 1

                # Create FileInfo object from cached summary
                return FileInfo(
                    path=str(file_path.relative_to(repo_path)),
                    language=analyzer.get_file_language(file_path),
                    content=analyzer.read_file(file_path),
                    summary=cached_summary,
                )

        # Process file with LLM
        try:
            # Read file content
            content = analyzer.read_file(file_path)

            # Get file language
            language = analyzer.get_file_language(file_path)

            # Generate summary with LLM
            summary = await llm_client.generate_summary(
                file_path=str(file_path.relative_to(repo_path)),
                content=content,
                file_type=language,
            )

            # Cache summary
            if analyzer.cache.enabled and not config.no_cache:
                analyzer.cache.save_summary(file_path, summary)

            # Update statistics
            cache_stats["from_llm"] += 1

            # Create FileInfo object
            return FileInfo(
                path=str(file_path.relative_to(repo_path)),
                language=language,
                content=content,
                summary=summary,
            )
        except Exception as e:
            # Log error and continue
            logging.error(f"Error processing file {file_path}: {e}")
            return None

    async def limited_process_file(
        semaphore: asyncio.Semaphore, file_path: Path, is_cached: bool
    ) -> Optional[FileInfo]:
        """Wrapper that enforces concurrency limit via semaphore."""
        async with semaphore:
            return await process_file(file_path, is_cached)

    # Process files in two phases with streaming completion
    concurrency = config.get_concurrency()

    with create_file_processing_progress_bar(total_files) as progress:
        # PHASE 1: Process cached files with streaming completion
        if cached_files:
            phase_start = time.time()
            progress.set_description("Processing cached files")

            # Create semaphore and all tasks upfront
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [
                asyncio.create_task(
                    limited_process_file(semaphore, file, is_cached=True)
                )
                for file in cached_files
            ]

            # Process tasks as they complete (streaming)
            for coro in asyncio.as_completed(tasks):
                # Check time limit before awaiting next completion
                if (
                    time_limit_seconds
                    and (time.time() - start_time) >= time_limit_seconds
                ):
                    cache_stats["time_limit_reached"] = True
                    vlogger.warning(
                        f"Time limit reached during cached phase ({vlogger.format_time(str(config.large_repo.time_limit_minutes) + ' minutes')})",
                        emoji="timer",
                    )
                    break

                try:
                    result = await coro
                    if result:
                        file_manifest[result.path] = result
                        cache_stats["cached_phase_count"] += 1
                    progress.update(1)
                except Exception as e:
                    logging.error(f"Error in cached file task: {e}")
                    progress.update(1)

            # Record cached phase time
            cache_stats["cached_phase_time"] = time.time() - phase_start

            # Display cached phase summary
            if cache_stats["cached_phase_count"] > 0:
                cached_time_formatted = f"{cache_stats['cached_phase_time']:.1f}s"
                vlogger.info(
                    f"Cached phase complete: {vlogger.format_number(cache_stats['cached_phase_count'])} files "
                    f"in {vlogger.format_time(cached_time_formatted)}",
                    emoji="cache",
                )

        # PHASE 2: Process uncached files with streaming completion
        if uncached_files and not cache_stats["time_limit_reached"]:
            phase_start = time.time()
            progress.set_description("Processing uncached files")

            # Create semaphore and all tasks upfront
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [
                asyncio.create_task(
                    limited_process_file(semaphore, file, is_cached=False)
                )
                for file in uncached_files
            ]

            # Process tasks as they complete (streaming)
            for coro in asyncio.as_completed(tasks):
                # Check time limit before awaiting next completion
                if (
                    time_limit_seconds
                    and (time.time() - start_time) >= time_limit_seconds
                ):
                    cache_stats["time_limit_reached"] = True
                    vlogger.warning(
                        f"Time limit reached during uncached phase ({vlogger.format_time(str(config.large_repo.time_limit_minutes) + ' minutes')})",
                        emoji="timer",
                    )
                    # Log partial completion
                    vlogger.info(
                        f"Processed {vlogger.format_number(len(file_manifest))} files before time limit.",
                        emoji="processing",
                    )
                    break

                try:
                    result = await coro
                    if result:
                        file_manifest[result.path] = result
                        cache_stats["uncached_phase_count"] += 1
                    progress.update(1)
                except Exception as e:
                    logging.error(f"Error in uncached file task: {e}")
                    progress.update(1)

            # Record uncached phase time
            cache_stats["uncached_phase_time"] = time.time() - phase_start

            # Display uncached phase summary
            if cache_stats["uncached_phase_count"] > 0:
                uncached_time_formatted = f"{cache_stats['uncached_phase_time']:.1f}s"
                vlogger.info(
                    f"Uncached phase complete: {vlogger.format_number(cache_stats['uncached_phase_count'])} files "
                    f"in {vlogger.format_time(uncached_time_formatted)}",
                    emoji="processing",
                )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Display cache statistics
    display_cache_stats(cache_stats, elapsed_time, analyzer.cache.enabled)

    # Add batch processing information to the returned manifest
    if hasattr(cache_stats, "__setitem__"):
        cache_stats["time_limit_reached"] = cache_stats.get("time_limit_reached", False)

    # Return file manifest
    return file_manifest


async def determine_processing_order(
    files: List[Path], llm_client: BaseLLMClient, progress: Optional[Any] = None
) -> List[Path]:
    """Determine optimal processing order for files.

    Args:
        files: List of files to process
        llm_client: LLM client to use for determining order
        progress: Optional progress bar

    Returns:
        Reordered list of files
    """
    # Get file order from LLM
    try:
        # Convert file paths to strings
        file_paths = [str(file) for file in files]

        # Get file order
        ordered_paths = await llm_client.get_file_order(file_paths)

        # Convert back to Path objects
        ordered_files = [Path(path) for path in ordered_paths]

        # Update progress
        if progress:
            progress.update(1)

        # Return ordered files
        return ordered_files
    except Exception as e:
        # Log error and return original order
        logging.error(f"Error determining file order: {e}")
        if progress:
            progress.update(1)
        return files


def check_all_files_processed(
    analyzer: CodebaseAnalyzer, all_files: List[Path]
) -> bool:
    """Check if all files in the repository have been processed and cached.

    Args:
        analyzer: CodebaseAnalyzer instance with cache access
        all_files: List of all files that should be processed

    Returns:
        True if all files have been processed, False otherwise
    """
    if not analyzer.cache.enabled:
        return False

    processed_count = 0
    total_files = 0

    for file_path in all_files:
        # Skip binary files and excluded files (same logic as in process_files)
        if analyzer.is_binary(file_path):
            continue

        rel_path = file_path.relative_to(analyzer.repo_path)
        if not analyzer.should_include_file(rel_path):
            continue

        total_files += 1

        # Check if file has been cached
        cached_summary = analyzer.cache.get_cached_summary(file_path)
        if cached_summary and not analyzer.cache.is_file_changed(file_path):
            processed_count += 1

    # Repository processing status logged via visual logger in main function
    return processed_count == total_files and total_files > 0


async def reuse_batch_pr_cache(
    github_url: str,
    github_token: str,
    repo_path: Path,
    config: ScribeConfig,
    analyzer: "CodebaseAnalyzer",
) -> bool:
    """Check for and reuse cache from an existing batch processing PR.

    If an open batch PR exists, this function:
    1. Fetches the cache files from the PR branch
    2. Closes the PR (since we'll reprocess with the cache)
    3. Deletes the branch
    4. Returns True so processing can continue with the reused cache

    Args:
        github_url: GitHub repository URL
        github_token: GitHub token for authentication
        repo_path: Path to the local repository
        config: Configuration object
        analyzer: CodebaseAnalyzer instance with cache access

    Returns:
        bool: True if cache was reused successfully, False if no PR found or error occurred
    """
    try:
        vlogger = get_visual_logger()
        branch_name = config.large_repo.batch_pr_branch

        # Check if there's an open PR on the batch branch
        vlogger.info(
            f"Checking for existing batch PR on branch: {vlogger.format_filename(branch_name)}",
            emoji="search",
        )
        existing_pr = find_existing_pr(github_url, github_token, branch_name)

        if not existing_pr:
            vlogger.info("No existing batch PR found", emoji="info")
            return False

        vlogger.info(
            f"Found existing batch PR #{existing_pr['number']}: {existing_pr['title']}",
            emoji="github",
        )
        vlogger.info(
            "Reusing cache from batch PR to avoid reprocessing...", emoji="cache"
        )

        # Import git module
        try:
            import git
        except ImportError:
            vlogger.error(
                "GitPython not installed. Cannot reuse batch PR cache.", emoji="error"
            )
            return False

        # Open the repository
        repo = git.Repo(repo_path)
        current_branch = repo.active_branch.name

        # Fetch the batch PR branch from remote
        vlogger.info(
            f"Fetching branch {vlogger.format_filename(branch_name)} from remote...",
            emoji="download",
        )
        try:
            # Fetch the specific branch (important for shallow clones)
            origin = repo.remotes.origin
            # Use refspec to fetch the branch into remote tracking branch
            origin.fetch(f"+refs/heads/{branch_name}:refs/remotes/origin/{branch_name}")

            # Get cache directory path
            cache_dir_name = config.cache.directory
            cache_dir_path = repo_path / cache_dir_name

            # Checkout the batch branch temporarily to get cache files
            vlogger.info(
                f"Temporarily switching to {vlogger.format_filename(branch_name)} to retrieve cache...",
                emoji="branch",
            )
            # Create local tracking branch from remote
            try:
                # Delete local branch if it exists
                try:
                    repo.delete_head(branch_name, force=True)
                except Exception:
                    pass
                # Create new local branch from remote
                repo.git.checkout("-b", branch_name, f"origin/{branch_name}")
            except Exception as e:
                vlogger.error(f"Failed to checkout branch: {e}", emoji="error")
                raise

            # Check if cache directory exists on the PR branch
            if cache_dir_path.exists():
                # Copy cache files to a temporary location
                import tempfile

                temp_cache_dir = Path(tempfile.mkdtemp(prefix="batch_cache_"))
                vlogger.info(
                    "Copying cache files from PR branch to temporary location...",
                    emoji="cache",
                )

                cache_files_copied = 0
                for cache_file in cache_dir_path.glob("*"):
                    if cache_file.is_file():
                        dest_file = temp_cache_dir / cache_file.name
                        shutil.copy2(cache_file, dest_file)
                        cache_files_copied += 1
                        logging.debug(f"Copied cache file: {cache_file.name}")

                vlogger.success(
                    f"Copied {vlogger.format_number(cache_files_copied)} cache file(s)",
                    emoji="complete",
                )

                # Switch back to original branch
                vlogger.info(
                    f"Switching back to {vlogger.format_filename(current_branch)}",
                    emoji="branch",
                )
                repo.git.checkout(current_branch)

                # Ensure cache directory exists in current branch
                os.makedirs(cache_dir_path, exist_ok=True)

                # Copy cache files from temporary location to current branch
                vlogger.info("Applying cache files to current branch...", emoji="cache")
                for cache_file in temp_cache_dir.glob("*"):
                    if cache_file.is_file():
                        dest_file = cache_dir_path / cache_file.name
                        shutil.copy2(cache_file, dest_file)
                        logging.debug(f"Applied cache file: {cache_file.name}")

                # Clean up temporary directory
                shutil.rmtree(temp_cache_dir)

                # Reload cache in analyzer to pick up the reused cache
                vlogger.info(
                    "Reloading cache manager with reused cache files...",
                    emoji="refresh",
                )
                analyzer.cache.load_cache_from_repo()

                vlogger.success(
                    "Cache successfully reused from batch PR!", emoji="complete"
                )
            else:
                vlogger.warning(
                    f"Cache directory not found on PR branch: {cache_dir_path}",
                    emoji="warning",
                )
                # Switch back to original branch even if cache not found
                repo.git.checkout(current_branch)
                # Clean up the local batch branch
                try:
                    repo.delete_head(branch_name, force=True)
                except Exception:
                    pass
                return False

        except Exception as fetch_error:
            vlogger.error(
                f"Error fetching or applying cache from PR branch: {fetch_error}",
                emoji="error",
            )
            logging.error(f"Cache reuse error: {fetch_error}")
            # Ensure we switch back to original branch on error
            try:
                repo.git.checkout(current_branch)
            except Exception:
                pass
            # Clean up the local batch branch if it was created
            try:
                repo.delete_head(branch_name, force=True)
            except Exception:
                pass
            return False

        # Close the PR since we've reused its cache
        vlogger.info(
            f"Closing batch PR #{existing_pr['number']} (cache reused)...",
            emoji="github",
        )
        if close_pull_request(existing_pr["object"]):
            vlogger.success(
                f"Batch PR #{existing_pr['number']} closed successfully",
                emoji="complete",
            )
        else:
            vlogger.warning(
                "Failed to close batch PR (continuing anyway)", emoji="warning"
            )

        # Delete the remote branch to clean up
        vlogger.info(
            f"Deleting remote branch {vlogger.format_filename(branch_name)}...",
            emoji="cleanup",
        )
        if delete_branch(github_url, github_token, branch_name):
            vlogger.success(
                f"Remote branch {vlogger.format_filename(branch_name)} deleted successfully",
                emoji="complete",
            )
        else:
            vlogger.warning(
                "Failed to delete remote branch (continuing anyway)", emoji="warning"
            )

        # Delete the local branch as well
        try:
            repo.delete_head(branch_name, force=True)
            vlogger.success(
                f"Local branch {vlogger.format_filename(branch_name)} deleted successfully",
                emoji="complete",
            )
        except Exception as e:
            logging.debug(f"Could not delete local branch {branch_name}: {e}")

        vlogger.section("Batch PR Cache Reused", style="bright_green")
        vlogger.info(
            "Processing will continue using the cache from the previous batch run",
            emoji="cache",
        )
        vlogger.info(
            "Files already processed will be skipped automatically", emoji="skip"
        )

        return True

    except Exception as e:
        vlogger.error(f"Error reusing batch PR cache: {e}", emoji="error")
        logging.error(f"Batch PR cache reuse failed: {e}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        return False


async def handle_expired_token_during_docs(
    github_url: str,
    github_token: str,
    config: ScribeConfig,
    analyzer: "CodebaseAnalyzer",
    error: Exception,
    doc_type: str,
) -> None:
    """Handle ExpiredTokenException during documentation generation by creating batch PR if applicable.

    Args:
        github_url: GitHub repository URL
        github_token: GitHub token for authentication (should still be valid)
        config: Configuration object
        analyzer: CodebaseAnalyzer instance with cache access
        error: The ExpiredTokenException that occurred
        doc_type: Type of documentation being generated (e.g., 'architecture', 'readme')
    """
    vlogger = get_visual_logger()

    # Check if batch processing with PR creation is enabled
    if not (
        config.large_repo.batch_processing
        and config.large_repo.create_pr_on_batch
        and github_url
        and github_token
    ):
        vlogger.error(
            f"Bedrock token expired during {doc_type} generation. Batch PR creation not enabled.",
            emoji="error",
        )
        vlogger.info(
            "Re-run the documenter to continue with a fresh token", emoji="info"
        )
        return

    vlogger.warning(
        f"Bedrock token expired during {doc_type} documentation generation",
        emoji="timer",
    )
    vlogger.info("Creating batch PR to save cache progress...", emoji="cicd")

    # Calculate statistics for the PR
    all_files = analyzer._get_repository_files()
    processed_count = 0
    total_count = 0

    for file_path in all_files:
        # Skip binary files and excluded files
        if analyzer.is_binary(file_path):
            continue

        rel_path = file_path.relative_to(analyzer.repo_path)
        if not analyzer.should_include_file(rel_path):
            continue

        total_count += 1

        # Check if file has been cached
        cached_summary = analyzer.cache.get_cached_summary(file_path)
        if cached_summary and not analyzer.cache.is_file_changed(file_path):
            processed_count += 1

    # Estimate elapsed time (use time limit as approximation)
    elapsed_time = config.large_repo.time_limit_minutes * 60

    try:
        pr_url = await create_batch_processing_pr(
            github_url=github_url,
            github_token=github_token,
            config=config,
            analyzer=analyzer,
            processed_count=processed_count,
            total_count=total_count,
            elapsed_time=elapsed_time,
        )

        if pr_url:
            vlogger.success(
                f"Batch PR created successfully: {vlogger.format_url(pr_url)}",
                emoji="cicd",
            )
            vlogger.info("Next steps:", emoji="info")
            vlogger.info(
                "  1. Merge the batch PR to save cache progress", emoji="check"
            )
            vlogger.info(
                "  2. Re-run the documenter with a fresh Bedrock token", emoji="sync"
            )
            vlogger.info(
                "  3. Cache will be reused automatically on next run", emoji="cache"
            )
        else:
            vlogger.error("Failed to create batch PR", emoji="error")
    except Exception as pr_error:
        vlogger.error(f"Error creating batch PR: {pr_error}", emoji="error")
        if config.debug:
            vlogger.error(f"Exception details: {traceback.format_exc()}", emoji="debug")


async def create_batch_processing_pr(
    github_url: str,
    github_token: str,
    config: ScribeConfig,
    analyzer: "CodebaseAnalyzer",
    processed_count: int,
    total_count: int,
    elapsed_time: float,
) -> Optional[str]:
    """Create a GitHub PR with cache file for batch processing progress.

    Args:
        github_url: GitHub repository URL
        github_token: GitHub token for authentication
        config: Configuration object
        analyzer: CodebaseAnalyzer instance with cache access
        processed_count: Number of files processed in this batch
        total_count: Total number of files to process
        elapsed_time: Time elapsed in processing (seconds)

    Returns:
        Optional[str]: URL of the created pull request, or None if creation failed
    """
    try:
        # Get visual logger instance once for this function
        vlogger = get_visual_logger()

        # Generate PR title from template
        pr_title = config.large_repo.batch_pr_title_template.format(
            processed=processed_count, total=total_count
        )

        # Generate PR body with processing statistics
        elapsed_minutes = elapsed_time / 60
        pr_body = f"""## Batch Processing Progress Update

This PR contains cache files from a batch processing run that processed **{processed_count} out of {total_count} files** ({processed_count / total_count * 100:.1f}% complete).

### Processing Statistics
- **Files Processed**: {processed_count}/{total_count}
- **Processing Time**: {elapsed_minutes:.1f} minutes
- **Time Limit**: {config.large_repo.time_limit_minutes} minutes
- **Remaining Files**: {total_count - processed_count}

### What's Included
- Updated cache files with file summaries for processed files
- Progress state for incremental processing continuation

### Next Steps
1. **Merge this PR** to save the current batch processing progress
2. **Re-run the documentation generator** to continue processing remaining files
3. **Repeat until all files are processed** and documentation is generated

### Cache Information
The cache files in this PR allow the next run to skip already processed files and continue from where this batch left off, ensuring efficient incremental processing of large repositories.

---
*This PR was automatically created by the batch processing system when the {config.large_repo.time_limit_minutes}-minute time limit was reached.*"""

        # Get repository path from analyzer
        repo_path = analyzer.repo_path

        # Prepare cache for commit (sync cache files to repository if needed)
        vlogger.info("Preparing cache files for batch processing PR...", emoji="cache")
        try:
            if not analyzer.cache.prepare_for_commit():
                vlogger.warning(
                    "Failed to sync cache files to repository", emoji="warning"
                )
                logging.warning("Cache sync failed for batch processing PR")
                return None
            else:
                vlogger.success(
                    "Cache files prepared successfully for PR", emoji="cache"
                )
        except Exception as cache_error:
            vlogger.error(
                f"Error preparing cache for batch processing PR: {cache_error}",
                emoji="error",
            )
            logging.error(f"Cache preparation error for batch PR: {cache_error}")
            return None

        # Create the branch
        branch_name = config.large_repo.batch_pr_branch
        vlogger.info(
            f"Creating branch: {vlogger.format_filename(branch_name)}", emoji="branch"
        )

        # First clean up any existing branch/PR
        await prepare_github_branch(github_url, github_token, branch_name)

        # Create the branch
        create_git_branch(repo_path, branch_name)

        # Commit cache files only (no documentation files)
        vlogger.info("Committing cache files...", emoji="github")
        commit_message = f"Batch processing cache update - {processed_count}/{total_count} files processed"

        # Commit changes (cache files only)
        commit_documentation_changes(
            repo_path, files=[], message=commit_message, cache_manager=analyzer.cache
        )

        # Push branch
        vlogger.info(
            f"Pushing branch {vlogger.format_filename(branch_name)} to remote...",
            emoji="github",
        )
        await push_branch_to_remote(repo_path, branch_name, github_token, github_url)

        # Create PR with batch processing labels
        vlogger.info("Creating batch processing PR...", emoji="github")
        pr_url = await create_pull_request(
            github_url,
            github_token,
            branch_name,
            pr_title,
            pr_body,
            labels=["batch-processing", "cache-update", "automated"],
        )

        if pr_url:
            vlogger.success(
                f"Batch processing PR created successfully: {vlogger.format_url(pr_url)}",
                emoji="complete",
            )
            return pr_url
        else:
            vlogger.error("Failed to create batch processing PR", emoji="error")
            return None

    except Exception as e:
        vlogger.error(f"Error creating batch processing PR: {e}", emoji="error")
        logging.error(f"Batch processing PR creation failed: {e}")
        return None


def add_ai_attribution_to_files(files: List[Path]) -> None:
    """Add AI attribution to files.

    Args:
        files: List of files to add attribution to
    """
    for file in files:
        try:
            # Read file content
            content = file.read_text(encoding="utf-8")

            # Add attribution
            content = add_ai_attribution(content)

            # Write back to file
            file.write_text(content, encoding="utf-8")
        except Exception as e:
            # Log error and continue
            logging.error(f"Error adding attribution to {file}: {e}")


def fix_malformed_headings(content: str) -> str:
    """Fix malformed headings in markdown content.

    This function fixes headings that have extra # characters, like "# # Heading"
    instead of "## Heading".

    Args:
        content: Markdown content to fix

    Returns:
        Fixed markdown content
    """
    # Fix headings with extra # characters (e.g., "# # Heading" -> "## Heading")
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Check for malformed headings like "# # Heading"
        if line.startswith("# #"):
            # Count the number of # characters
            count = 0
            for char in line:
                if char == "#":
                    count += 1
                elif char != " ":
                    break

            # Replace with the correct number of # characters
            fixed_line = "#" * count + line[count:]
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


async def main():
    """Main entry point for the codebase-scribe tool."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for a code repository"
    )

    # Create a mutually exclusive group for repo source
    repo_source = parser.add_mutually_exclusive_group(required=True)
    repo_source.add_argument("--repo", help="Path to the local repository")
    repo_source.add_argument(
        "--github",
        help="GitHub repository URL (e.g., https://github.com/username/repo)",
    )

    # Add authentication options for GitHub
    parser.add_argument(
        "--github-token", help="GitHub Personal Access Token for private repositories"
    )
    parser.add_argument(
        "--keep-clone",
        action="store_true",
        help="Keep cloned repository after processing (default: remove)",
    )

    # Add branch and PR options
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request with generated documentation (GitHub only)",
    )
    parser.add_argument(
        "--branch-name",
        default="docs/auto-generated-readme",
        help="Branch name for PR creation (default: docs/auto-generated-readme)",
    )
    parser.add_argument(
        "--pr-title",
        default="Add AI-generated documentation",
        help="Title for the pull request",
    )
    parser.add_argument(
        "--pr-body",
        default="This PR adds automatically generated documentation files including README.md, ARCHITECTURE.md, CONTRIBUTING.md, PERSISTENCE.md (if persistence layer detected), INSTALLATION.md, USAGE.md, and TROUBLESHOOTING.md.",
        help="Body text for the pull request",
    )

    # Existing arguments
    parser.add_argument("--output", "-o", default="README.md", help="Output file name")
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Configuration file path"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log-file", action="store_true", help="Log debug output to file"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce console output verbosity (only warnings and errors)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode (process only first 5 files)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching of file summaries"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache for this repository before processing",
    )
    parser.add_argument(
        "--optimize-order",
        action="store_true",
        help="Use LLM to determine optimal file processing order",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "bedrock"],
        default=None,
        help="LLM provider to use (overrides config file)",
    )
    parser.add_argument(
        "--no-exit-on-docs-only",
        action="store_true",
        help="Disable early exit when only generated .md files have changed",
    )
    args = parser.parse_args()

    # Set up logging based on debug flag and quiet mode
    setup_logging(debug=args.debug, log_to_file=args.log_file, quiet=args.quiet)

    # Get visual logger instance once for the entire main function
    vlogger = get_visual_logger()

    # Load config and update with command-line args
    config = load_config(args.config)
    config = update_config_with_args(config, args)

    if config.debug:
        logging.debug("Debug mode enabled")

    # Display configuration summary
    vlogger.section("Configuration Summary", style="bright_cyan")
    vlogger.info(
        f"LLM Provider: {vlogger.format_filename(config.llm_provider)}", emoji="config"
    )
    vlogger.info(f"Debug Mode: {'Yes' if config.debug else 'No'}", emoji="config")
    vlogger.info(f"Cache Enabled: {'No' if args.no_cache else 'Yes'}", emoji="cache")

    # Provider-specific configuration details
    if config.llm_provider == "bedrock":
        vlogger.info(
            f"AWS Region: {vlogger.format_filename(config.bedrock.region)}", emoji="aws"
        )
        vlogger.info(
            f"Model: {vlogger.format_filename(config.bedrock.model_id.split('/')[-1])}",
            emoji="aws",
        )
        vlogger.info(
            f"Prompt Caching: {'Yes' if config.bedrock.enable_prompt_caching else 'No'}",
            emoji="prompt_cache",
        )
    elif config.llm_provider == "ollama":
        vlogger.info(
            f"Model: {vlogger.format_filename(config.ollama.model)}", emoji="config"
        )
        vlogger.info(
            f"Base URL: {vlogger.format_url(config.ollama.base_url)}", emoji="config"
        )

    # Batch processing configuration
    if config.large_repo.batch_processing:
        vlogger.info(
            f"Batch Processing: Yes (Time limit: {vlogger.format_time(str(config.large_repo.time_limit_minutes) + ' minutes')})",
            emoji="timer",
        )
    else:
        vlogger.info("Batch Processing: No", emoji="config")

    # Get GitHub token from args or environment
    github_token = args.github_token
    if not github_token and "GITHUB_TOKEN" in os.environ:
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            vlogger.info(
                "Using GitHub token from environment variables", emoji="github"
            )

    # Temp directory for cloned repo
    temp_dir = None

    # Create output directory for potential copying later (only if needed)
    if args.github:
        # Create an output directory in the script's location
        script_dir = Path(__file__).parent.absolute()
        output_dir = script_dir / "generated_docs"
        os.makedirs(output_dir, exist_ok=True)

        # Create a .gitignore file in the output directory if it doesn't exist
        gitignore_path = output_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write("# Ignore all generated documentation\n*\n!.gitignore\n")

        vlogger.info(
            f"Generated files will be saved to: {vlogger.format_path(output_dir)}",
            emoji="file",
        )
    else:
        # Provide default value for output_dir if not using GitHub
        output_dir = Path("generated_docs")

    try:
        # Check if the user wants to create a PR
        create_pr = args.create_pr

        if create_pr and args.repo:
            vlogger.warning(
                "--create-pr can only be used with --github. Ignoring PR creation.",
                emoji="warning",
            )
            create_pr = False

        if create_pr and not github_token:
            vlogger.error(
                "GitHub token is required for PR creation. Use --github-token or set GITHUB_TOKEN environment variable.",
                emoji="error",
            )
            return

        # Check for existing PR before processing
        if create_pr and args.github:
            vlogger.info(
                f"Checking for existing PR on branch: {vlogger.format_filename(args.branch_name)}",
                emoji="search",
            )
            existing_pr = find_existing_pr(args.github, github_token, args.branch_name)

            if existing_pr:
                vlogger.info(
                    f"Found existing PR #{existing_pr['number']}: {existing_pr['title']}",
                    emoji="github",
                )
                vlogger.warning(
                    f"PR already exists at: {vlogger.format_url(existing_pr['url'])}",
                    emoji="warning",
                )
                vlogger.info("Next steps:", emoji="info")
                vlogger.info("  1. Review and merge the existing PR", emoji="check")
                vlogger.info(
                    "  2. Rerun the documenter after the PR is merged", emoji="sync"
                )
                vlogger.info("Exiting to prevent duplicate processing", emoji="skip")
                return
            else:
                vlogger.info(
                    "No existing PR found - proceeding with documentation generation",
                    emoji="check",
                )

        # Process GitHub URL if provided
        if args.github:
            # Validate GitHub URL
            if not is_valid_github_url(args.github):
                vlogger.error(
                    f"Invalid GitHub repository URL: {vlogger.format_url(args.github)}",
                    emoji="error",
                )
                vlogger.info(
                    "Expected format: https://github.com/username/repository",
                    emoji="info",
                )
                return

            # Extract a stable repo identifier for caching
            repo_owner, repo_name = extract_repo_info(args.github)
            repo_id = f"{repo_owner}/{repo_name}"
            vlogger.info(
                f"Using repository ID for caching: {vlogger.format_filename(repo_id)}",
                emoji="repository",
            )

            # Clone the repository
            vlogger.info(
                f"Cloning GitHub repository: {vlogger.format_url(args.github)}",
                emoji="github",
            )
            try:
                temp_dir = await clone_github_repository(args.github, github_token)
                repo_path = Path(temp_dir).absolute()
                vlogger.success(
                    f"Repository cloned successfully to: {vlogger.format_path(repo_path)}",
                    emoji="repository",
                )

                # Tell analyzer to use the stable repo ID for caching
                config.github_repo_id = repo_id
            except Exception as e:
                vlogger.error(f"Error: {e}", emoji="error")
                return
        else:
            # Use provided local repository path
            repo_path = Path(args.repo).absolute()

        # Handle cache clearing first and exit
        if args.clear_cache:
            analyzer = CodebaseAnalyzer(repo_path, config)
            analyzer.cache.clear_repo_cache()
            vlogger.success(
                f"Cleared cache for repository: {vlogger.format_filename(repo_path.name)}",
                emoji="cache",
            )

            # Close the cache connections before clearing all caches
            analyzer.cache.close()

            # Also clear the global cache for this repository
            CacheManager.clear_all_caches(repo_path=repo_path, config=config)

            if temp_dir and not args.keep_clone:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return  # Exit after clearing cache

        # Initialize LLM client using factory
        try:
            # Create and initialize the appropriate LLM client
            llm_client = await LLMClientFactory.create_client(config)
        except Exception as e:
            vlogger.error(f"Failed to initialize LLM client: {e}", emoji="error")
            sys.exit(1)

        # Now start repository analysis
        analyzer = CodebaseAnalyzer(repo_path, config)
        manifest = analyzer.analyze_repository(show_progress=True)

        if not manifest:
            # Empty manifest could mean early exit or no files found
            if temp_dir and not args.keep_clone:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # Set project structure in LLM client
        llm_client.set_project_structure_from_manifest(manifest)

        # Check for and reuse batch PR cache if applicable
        if (
            config.large_repo.batch_processing
            and config.large_repo.create_pr_on_batch
            and args.github
            and github_token
        ):
            vlogger.section("Batch PR Cache Check", style="bright_cyan")
            cache_reused = await reuse_batch_pr_cache(
                github_url=args.github,
                github_token=github_token,
                repo_path=repo_path,
                config=config,
                analyzer=analyzer,
            )

            if cache_reused:
                # Cache was reused successfully - re-analyze repository
                vlogger.info(
                    "Re-analyzing repository with reused cache...", emoji="refresh"
                )
                manifest = analyzer.analyze_repository(show_progress=True)
                llm_client.set_project_structure_from_manifest(manifest)

        # No related repos in open-source version
        related_repo_data = None

        # Process files
        file_manifest = await process_files(repo_path, llm_client, config)

        if not file_manifest:
            vlogger.error("No files were processed successfully!", emoji="error")
            return

        # Check if we're in batch processing mode and if all files have been processed
        all_files_processed = True
        if config.large_repo.batch_processing:
            all_files = analyzer._get_repository_files()
            all_files_processed = check_all_files_processed(analyzer, all_files)

            if not all_files_processed:
                vlogger.section("Batch Processing Status", style="yellow")
                vlogger.warning(
                    "Time-based batch processing active - processed files but not complete",
                    emoji="timer",
                )
                vlogger.info(
                    "Cache updated with processed file summaries", emoji="cache"
                )
                vlogger.info(
                    "Skipping documentation generation until all files are processed",
                    emoji="skip",
                )

                # Create batch processing PR if enabled and running on GitHub repository
                if (
                    config.large_repo.create_pr_on_batch
                    and args.github
                    and github_token
                    and len(file_manifest) > 0
                ):
                    vlogger.info(
                        "Creating batch processing PR with cache files...", emoji="cicd"
                    )

                    # Calculate processing statistics
                    processed_count = len(file_manifest)
                    total_count = len(all_files)
                    elapsed_time = (
                        config.large_repo.time_limit_minutes * 60
                    )

                    try:
                        pr_url = await create_batch_processing_pr(
                            github_url=args.github,
                            github_token=github_token,
                            config=config,
                            analyzer=analyzer,
                            processed_count=processed_count,
                            total_count=total_count,
                            elapsed_time=elapsed_time,
                        )

                        if pr_url:
                            vlogger.success(
                                f"Batch processing PR created: {vlogger.format_url(pr_url)}",
                                emoji="complete",
                            )
                            vlogger.section("Next Steps", style="bright_cyan")
                            vlogger.info(
                                f"1. Review and merge the PR: {vlogger.format_url(pr_url)}",
                                emoji="info",
                            )
                            vlogger.info(
                                "2. Re-run the documentation generator to continue processing",
                                emoji="info",
                            )
                            vlogger.info(
                                f"3. Repeat until all {vlogger.format_number(total_count)} files are processed",
                                emoji="info",
                            )
                        else:
                            vlogger.error(
                                "Failed to create batch processing PR", emoji="error"
                            )
                            vlogger.info(
                                "Cache files have been saved locally", emoji="info"
                            )
                            vlogger.info(
                                "Run the tool again to continue processing remaining files",
                                emoji="info",
                            )
                    except Exception as pr_error:
                        vlogger.error(
                            f"Error creating batch processing PR: {pr_error}",
                            emoji="error",
                        )
                        vlogger.info(
                            "Cache files have been saved locally", emoji="info"
                        )
                        vlogger.info(
                            "Run the tool again to continue processing remaining files",
                            emoji="info",
                        )
                else:
                    vlogger.info(
                        "Run the tool again to continue processing remaining files",
                        emoji="info",
                    )
                    vlogger.info(
                        "Documentation will be generated once all files are processed",
                        emoji="info",
                    )

                return
            else:
                vlogger.success(
                    "All repository files have been processed!", emoji="complete"
                )
                vlogger.info("Proceeding with documentation generation", emoji="start")

        # Generate badges
        badges = generate_badges(file_manifest, repo_path)

        # Analyze persistence layer (before architecture generation)
        persistence_info = None
        if hasattr(config, "persistence") and config.persistence.enabled:
            vlogger.section("Persistence Layer Analysis")
            vlogger.info("Analyzing persistence layer...", emoji="persistence")
            with create_documentation_progress_bar(repo_path) as progress:
                persistence_info = await analyze_persistence_layer(
                    repo_path=repo_path, config=config, llm_client=llm_client
                )

            if persistence_info:
                vlogger.success(
                    "Persistence layer analysis completed", emoji="persistence"
                )
            else:
                vlogger.info("No persistence layer detected", emoji="info")

        # Generate persistence documentation using pre-analyzed data
        persistence_content = None
        if persistence_info and has_meaningful_persistence_content(persistence_info):
            vlogger.section("Persistence Documentation Generation")
            vlogger.info(
                "Generating persistence layer documentation...", emoji="persistence"
            )

            # Track timing
            persist_start_time = time.time()

            with create_documentation_progress_bar(repo_path) as progress:
                persistence_content = await generate_persistence(
                    repo_path=repo_path,
                    file_manifest=file_manifest,
                    llm_client=llm_client,
                    config=config,
                    persistence_info=persistence_info,
                )

            # Calculate metrics
            persist_elapsed = time.time() - persist_start_time

            # Get detailed token metrics from the last operation
            token_metrics = None
            if hasattr(llm_client, "last_operation_metrics"):
                token_metrics = llm_client.last_operation_metrics

            if persistence_content:
                vlogger.document_generation(
                    "persistence", "generated", persist_elapsed, token_metrics
                )
            else:
                vlogger.document_generation("persistence", "failed")
        elif persistence_info:
            vlogger.document_generation("persistence", "skipped")
            vlogger.info("No meaningful database schema found", emoji="info")

        # Generate architecture documentation (with complete context now available)
        vlogger.section("Architecture Documentation Generation")
        vlogger.info("Generating architecture documentation...", emoji="architecture")

        # Track timing
        arch_start_time = time.time()

        try:
            with create_documentation_progress_bar(repo_path) as progress:
                architecture_content = await generate_architecture(
                    repo_path=repo_path,
                    file_manifest=file_manifest,
                    llm_client=llm_client,
                    config=config,
                    persistence_info=persistence_info,
                    related_repo_data=None,
                )

            # Calculate metrics
            arch_elapsed = time.time() - arch_start_time

            # Get detailed token metrics from the last operation
            token_metrics = None
            if hasattr(llm_client, "last_operation_metrics"):
                token_metrics = llm_client.last_operation_metrics

            # Display enhanced success message with detailed metrics
            vlogger.document_generation(
                "architecture", "generated", arch_elapsed, token_metrics
            )

        except Exception as e:
            # Check if it's an ExpiredTokenException
            if "ExpiredTokenException" in str(
                type(e).__name__
            ) or "security token included in the request is expired" in str(e):
                await handle_expired_token_during_docs(
                    github_url=args.github if hasattr(args, "github") else None,
                    github_token=github_token,
                    config=config,
                    analyzer=analyzer,
                    error=e,
                    doc_type="architecture",
                )
                return
            else:
                raise

        # ============================================================================
        # CONCURRENT: INSTALLATION, USAGE, TROUBLESHOOTING, CONTRIBUTING Documentation
        # ============================================================================
        sections_for_refactoring = []

        # Helper function for installation generation with timing
        async def _generate_installation_with_timing():
            if not config.installation.enabled:
                return None, None

            vlogger.info("Generating installation guide...", emoji="package")
            install_start_time = time.time()

            content = await generate_installation(
                repo_path=repo_path,
                file_manifest=file_manifest,
                llm_client=llm_client,
                config=config,
                analyzer=analyzer,
                related_repo_data=None,
                persistence_info=persistence_info,
            )

            elapsed = time.time() - install_start_time
            return content, elapsed

        # Helper function for usage generation with timing
        async def _generate_usage_with_timing():
            if not config.usage.enabled:
                return None, None

            vlogger.info("Generating usage guide...", emoji="rocket")
            usage_start_time = time.time()

            content = await generate_usage(
                repo_path=repo_path,
                file_manifest=file_manifest,
                llm_client=llm_client,
                config=config,
                analyzer=analyzer,
                related_repo_data=None,
                persistence_info=persistence_info,
            )

            elapsed = time.time() - usage_start_time
            return content, elapsed

        # Helper function for troubleshooting generation with timing
        async def _generate_troubleshooting_with_timing():
            if not config.troubleshooting.enabled:
                return None, None

            vlogger.info("Generating troubleshooting guide...", emoji="wrench")
            ts_start_time = time.time()

            content = await generate_troubleshooting(
                repo_path=repo_path,
                file_manifest=file_manifest,
                llm_client=llm_client,
                config=config,
                analyzer=analyzer,
                related_repo_data=None,
                persistence_info=persistence_info,
            )

            elapsed = time.time() - ts_start_time
            return content, elapsed

        # Helper function for contributing generation with timing
        async def _generate_contributing_with_timing():
            if not config.contributing.enabled:
                return None, None

            vlogger.info("Generating contributing guide...", emoji="contributing")
            contrib_start_time = time.time()

            content = await generate_contributing(
                repo_path=repo_path,
                file_manifest=file_manifest,
                llm_client=llm_client,
                config=config,
                analyzer=analyzer,
                related_repo_data=None,
                persistence_info=persistence_info,
            )

            elapsed = time.time() - contrib_start_time
            return content, elapsed

        # Check which docs should be generated
        should_generate_any = any(
            [
                config.installation.enabled,
                config.usage.enabled,
                config.troubleshooting.enabled,
                config.contributing.enabled,
            ]
        )

        installation_content = None
        usage_content = None
        troubleshooting_content = None
        contributing_content = None

        if should_generate_any:
            vlogger.section(
                "Practical Documentation Generation (INSTALLATION, USAGE, TROUBLESHOOTING, CONTRIBUTING)"
            )

            try:
                # Generate all four docs concurrently
                with create_documentation_progress_bar(repo_path) as progress:
                    results = await asyncio.gather(
                        _generate_installation_with_timing(),
                        _generate_usage_with_timing(),
                        _generate_troubleshooting_with_timing(),
                        _generate_contributing_with_timing(),
                        return_exceptions=True,
                    )

                # Process results
                installation_content, install_elapsed = (
                    results[0]
                    if not isinstance(results[0], Exception)
                    else (None, None)
                )
                usage_content, usage_elapsed = (
                    results[1]
                    if not isinstance(results[1], Exception)
                    else (None, None)
                )
                troubleshooting_content, ts_elapsed = (
                    results[2]
                    if not isinstance(results[2], Exception)
                    else (None, None)
                )
                contributing_content, contrib_elapsed = (
                    results[3]
                    if not isinstance(results[3], Exception)
                    else (None, None)
                )

                # Write files and track for refactoring
                if installation_content:
                    installation_path = repo_path / config.installation.output_file
                    installation_path.parent.mkdir(parents=True, exist_ok=True)
                    installation_path.write_text(installation_content, encoding="utf-8")

                    sections_for_refactoring.append(
                        {
                            "section_names": [
                                "Installation",
                                "Setup",
                                "Getting Started",
                                "Prerequisites",
                            ],
                            "doc_path": config.installation.output_file,
                            "doc_title": "INSTALLATION.md",
                        }
                    )

                    token_metrics = (
                        llm_client.last_operation_metrics
                        if hasattr(llm_client, "last_operation_metrics")
                        else None
                    )
                    vlogger.document_generation(
                        "installation", "generated", install_elapsed, token_metrics
                    )

                if usage_content:
                    usage_path = repo_path / config.usage.output_file
                    usage_path.parent.mkdir(parents=True, exist_ok=True)
                    usage_path.write_text(usage_content, encoding="utf-8")

                    sections_for_refactoring.append(
                        {
                            "section_names": [
                                "Usage",
                                "Using",
                                "How to Use",
                                "Basic Usage",
                                "Quick Start",
                            ],
                            "doc_path": config.usage.output_file,
                            "doc_title": "USAGE.md",
                        }
                    )

                    token_metrics = (
                        llm_client.last_operation_metrics
                        if hasattr(llm_client, "last_operation_metrics")
                        else None
                    )
                    vlogger.document_generation(
                        "usage", "generated", usage_elapsed, token_metrics
                    )

                if troubleshooting_content:
                    ts_path = repo_path / config.troubleshooting.output_file
                    ts_path.parent.mkdir(parents=True, exist_ok=True)
                    ts_path.write_text(troubleshooting_content, encoding="utf-8")

                    sections_for_refactoring.append(
                        {
                            "section_names": [
                                "Troubleshooting",
                                "Common Issues",
                                "FAQ",
                                "Debugging",
                                "Problems",
                            ],
                            "doc_path": config.troubleshooting.output_file,
                            "doc_title": "TROUBLESHOOTING.md",
                        }
                    )

                    token_metrics = (
                        llm_client.last_operation_metrics
                        if hasattr(llm_client, "last_operation_metrics")
                        else None
                    )
                    vlogger.document_generation(
                        "troubleshooting", "generated", ts_elapsed, token_metrics
                    )

                if contributing_content:
                    contributing_path = repo_path / config.contributing.output_file
                    contributing_path.parent.mkdir(parents=True, exist_ok=True)
                    contributing_path.write_text(contributing_content, encoding="utf-8")

                    sections_for_refactoring.append(
                        {
                            "section_names": [
                                "Contributing",
                                "Contribution Guidelines",
                                "How to Contribute",
                                "Development",
                                "For Contributors",
                            ],
                            "doc_path": config.contributing.output_file,
                            "doc_title": "CONTRIBUTING.md",
                        }
                    )

                    token_metrics = (
                        llm_client.last_operation_metrics
                        if hasattr(llm_client, "last_operation_metrics")
                        else None
                    )
                    vlogger.document_generation(
                        "contributing", "generated", contrib_elapsed, token_metrics
                    )

            except Exception as e:
                # Check if it's an ExpiredTokenException
                if "ExpiredTokenException" in str(
                    type(e).__name__
                ) or "security token included in the request is expired" in str(e):
                    await handle_expired_token_during_docs(
                        github_url=args.github if hasattr(args, "github") else None,
                        github_token=github_token,
                        config=config,
                        analyzer=analyzer,
                        error=e,
                        doc_type="practical_docs",
                    )
                    return
                else:
                    vlogger.error(f"Error generating practical documentation: {e}")

        # Create file summaries dictionary
        file_summaries = {}
        for path, info in file_manifest.items():
            if hasattr(info, "summary"):
                file_summaries[path] = info.summary
            elif isinstance(info, dict) and "summary" in info:
                file_summaries[path] = info["summary"]

        # Update analyzer's file manifest with processed results
        analyzer.file_manifest = file_manifest

        # Generate README after all practical documentation exists
        vlogger.section("README Generation")
        vlogger.info("Generating README...", emoji="readme")

        # Track timing
        readme_start_time = time.time()

        with create_documentation_progress_bar(repo_path) as progress:
            try:
                readme_content = await generate_readme(
                    repo_path=repo_path,
                    file_manifest=file_manifest,
                    llm_client=llm_client,
                    file_summaries=file_summaries,
                    config=config,
                    analyzer=analyzer,
                    output_dir=str(output_dir),
                    architecture_file_exists=True,
                    persistence_file_exists=bool(persistence_content),
                    related_repo_data=None,
                    persistence_info=persistence_info,
                )
            except Exception as e:
                # Check if it's an ExpiredTokenException first
                if "ExpiredTokenException" in str(
                    type(e).__name__
                ) or "security token included in the request is expired" in str(e):
                    await handle_expired_token_during_docs(
                        github_url=args.github if hasattr(args, "github") else None,
                        github_token=github_token,
                        config=config,
                        analyzer=analyzer,
                        error=e,
                        doc_type="readme",
                    )
                    return

                # If README generation fails, try progressive enhancement
                readme_path = repo_path / "README.md"
                if "timeout" in str(e).lower() and readme_path.exists():
                    vlogger.warning(
                        "README generation timed out. Trying progressive enhancement",
                        emoji="timer",
                    )

                    try:
                        from src.generators.readme import progressive_enhance_readme

                        existing_content = readme_path.read_text(encoding="utf-8")
                        readme_content = await progressive_enhance_readme(
                            existing_content,
                            llm_client,
                            file_manifest,
                            analyzer.derive_project_name(config.debug),
                            True,
                            bool(persistence_content),
                            None,
                        )
                        vlogger.success(
                            "Progressive enhancement completed successfully",
                            emoji="readme",
                        )
                    except Exception as prog_error:
                        vlogger.warning(
                            f"Progressive enhancement also failed: {prog_error}. Using original content",
                            emoji="warning",
                        )
                        readme_content = existing_content
                else:
                    vlogger.warning(
                        f"README generation failed: {e}. Using existing content",
                        emoji="warning",
                    )
                    # Read existing README.md if it exists
                    if readme_path.exists():
                        readme_content = readme_path.read_text(encoding="utf-8")
                    else:
                        # Generate a basic fallback if no existing file
                        readme_content = f"# {analyzer.derive_project_name(config.debug)}\n\nProject documentation will be generated when enhancement is available."

        # Calculate metrics
        readme_elapsed = time.time() - readme_start_time

        # Get detailed token metrics from the last operation
        token_metrics = None
        if hasattr(llm_client, "last_operation_metrics"):
            token_metrics = llm_client.last_operation_metrics

        # Display enhanced success message with detailed metrics
        vlogger.document_generation(
            "readme", "generated", readme_elapsed, token_metrics
        )

        # ============================================================================
        # README Refactoring (if enabled and sections were migrated)
        # ============================================================================
        if config.readme_refactor.enabled and sections_for_refactoring:
            vlogger.section("README Refactoring")
            vlogger.info(
                "Refactoring README to link to split documentation...", emoji="refactor"
            )

            try:
                refactor_readme_for_split_docs(
                    repo_path=repo_path,
                    sections_to_migrate=sections_for_refactoring,
                    config=config,
                )

                if config.readme_refactor.add_navigation_section:
                    add_navigation_section_to_readme(
                        repo_path=repo_path,
                        doc_links=[s["doc_path"] for s in sections_for_refactoring],
                    )

                vlogger.success(
                    f"README refactored successfully ({len(sections_for_refactoring)} sections processed)",
                    emoji="success",
                )

            except Exception as e:
                vlogger.warning(
                    f"README refactoring failed: {e}, leaving as-is", emoji="warning"
                )

        # Add badges to README with deduplication
        if badges:
            # Check if badges already exist to prevent duplication
            has_existing_badges, _ = detect_existing_badges(readme_content)

            if has_existing_badges:
                vlogger.info(
                    "Badges already exist in README, skipping insertion to prevent duplication",
                    emoji="info",
                )
            else:
                # Use the comprehensive badge insertion function
                readme_content = insert_badges_after_title(readme_content, badges)
                vlogger.info(
                    "Successfully added badges to README after title", emoji="badge"
                )

        # Write files
        readme_path = repo_path / args.output
        architecture_path = repo_path / "docs" / "ARCHITECTURE.md"
        contributing_path = (
            repo_path / "CONTRIBUTING.md"
        )

        # Set up persistence documentation path
        persistence_path = None
        if persistence_content:
            persistence_output_file = (
                config.persistence.output_file
                if hasattr(config, "persistence")
                else "docs/PERSISTENCE.md"
            )
            persistence_path = repo_path / persistence_output_file

        # Create docs directory if it doesn't exist
        os.makedirs(repo_path / "docs", exist_ok=True)

        # Fix malformed headings in README content
        readme_content = fix_malformed_headings(readme_content)
        if contributing_content:
            contributing_content = fix_malformed_headings(contributing_content)

        # Write README
        vlogger.section("Writing Documentation Files")
        vlogger.info(
            f"Writing README to {vlogger.format_path(readme_path)}", emoji="readme"
        )
        readme_path.write_text(readme_content, encoding="utf-8")

        # Write architecture documentation
        vlogger.info(
            f"Writing architecture documentation to {vlogger.format_path(architecture_path)}",
            emoji="architecture",
        )
        architecture_path.write_text(architecture_content, encoding="utf-8")

        # Write contributing guide (from concurrent generation or fallback)
        if contributing_content:
            vlogger.info(
                f"Writing contributing guide to {vlogger.format_path(contributing_path)}",
                emoji="contributing",
            )
            contributing_path.write_text(contributing_content, encoding="utf-8")

        # Write persistence documentation if generated
        files_to_attribute = [readme_path, architecture_path]
        if contributing_content:
            files_to_attribute.append(contributing_path)
        if persistence_content and persistence_path:
            vlogger.info(
                f"Writing persistence documentation to {vlogger.format_path(persistence_path)}",
                emoji="persistence",
            )
            # Fix malformed headings in persistence content
            persistence_content = fix_malformed_headings(persistence_content)
            persistence_path.write_text(persistence_content, encoding="utf-8")
            files_to_attribute.append(persistence_path)

        # Note: INSTALLATION, USAGE, TROUBLESHOOTING were already written in concurrent block
        # Add them to attribution list if they exist
        installation_path_check = repo_path / config.installation.output_file
        if installation_path_check.exists():
            vlogger.info(
                "Including INSTALLATION.md for attribution",
                emoji="package",
            )
            files_to_attribute.append(installation_path_check)

        usage_path_check = repo_path / config.usage.output_file
        if usage_path_check.exists():
            vlogger.info(
                "Including USAGE.md for attribution",
                emoji="rocket",
            )
            files_to_attribute.append(usage_path_check)

        troubleshooting_path_check = repo_path / config.troubleshooting.output_file
        if troubleshooting_path_check.exists():
            vlogger.info(
                "Including TROUBLESHOOTING.md for attribution",
                emoji="wrench",
            )
            files_to_attribute.append(troubleshooting_path_check)

        # Add AI attribution to files
        add_ai_attribution_to_files(files_to_attribute)

        # Copy files to output directory if needed
        if output_dir:
            shutil.copy2(readme_path, output_dir / args.output)
            shutil.copy2(architecture_path, output_dir / "ARCHITECTURE.md")
            if contributing_content:
                shutil.copy2(contributing_path, output_dir / "CONTRIBUTING.md")
            if persistence_content and persistence_path:
                shutil.copy2(persistence_path, output_dir / "PERSISTENCE.md")

            # Copy INSTALLATION, USAGE, TROUBLESHOOTING if they exist
            installation_src = repo_path / config.installation.output_file
            if installation_src.exists():
                install_dest = output_dir / installation_src.relative_to(repo_path)
                install_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(installation_src, install_dest)
                vlogger.info(
                    "Copied INSTALLATION.md to output directory",
                    emoji="package",
                )

            usage_src = repo_path / config.usage.output_file
            if usage_src.exists():
                usage_dest = output_dir / usage_src.relative_to(repo_path)
                usage_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(usage_src, usage_dest)
                vlogger.info(
                    "Copied USAGE.md to output directory",
                    emoji="rocket",
                )

            troubleshooting_src = repo_path / config.troubleshooting.output_file
            if troubleshooting_src.exists():
                ts_dest = output_dir / troubleshooting_src.relative_to(repo_path)
                ts_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(troubleshooting_src, ts_dest)
                vlogger.info(
                    "Copied TROUBLESHOOTING.md to output directory",
                    emoji="wrench",
                )

            vlogger.success(
                f"Files copied to {vlogger.format_path(output_dir)}", emoji="complete"
            )

        # Create PR if requested
        if create_pr:
            vlogger.section("GitHub Pull Request Creation")
            vlogger.info("Creating pull request...", emoji="github")
            try:
                # Prepare branch
                github_token = os.environ.get("GITHUB_TOKEN")
                if not github_token:
                    raise GitHubError("GitHub token not found in environment variables")

                # First clean up any existing branch/PR
                await prepare_github_branch(args.github, github_token, args.branch_name)

                # Create the branch
                create_git_branch(repo_path, args.branch_name)

                # Prepare cache for commit (sync cache files to repository if needed)
                vlogger.info("Preparing cache files for commit...", emoji="cache")
                try:
                    if not analyzer.cache.prepare_for_commit():
                        vlogger.warning(
                            "Failed to sync cache files to repository", emoji="warning"
                        )
                        logging.warning("Cache sync failed, but continuing with commit")
                    else:
                        vlogger.success(
                            "Cache files prepared successfully", emoji="cache"
                        )
                except Exception as cache_error:
                    vlogger.warning(
                        f"Error preparing cache for commit: {cache_error}",
                        emoji="warning",
                    )
                    logging.warning(f"Cache preparation error: {cache_error}")

                # Prepare list of files to commit
                files_to_commit = [readme_path, architecture_path]
                if contributing_content:
                    files_to_commit.append(contributing_path)
                if persistence_content and persistence_path:
                    files_to_commit.append(persistence_path)

                # Add INSTALLATION, USAGE, TROUBLESHOOTING if they exist
                installation_path_pr = repo_path / config.installation.output_file
                if installation_path_pr.exists():
                    files_to_commit.append(installation_path_pr)

                usage_path_pr = repo_path / config.usage.output_file
                if usage_path_pr.exists():
                    files_to_commit.append(usage_path_pr)

                troubleshooting_path_pr = repo_path / config.troubleshooting.output_file
                if troubleshooting_path_pr.exists():
                    files_to_commit.append(troubleshooting_path_pr)

                # Commit changes (including any synced cache files)
                commit_documentation_changes(
                    repo_path, files_to_commit, cache_manager=analyzer.cache
                )

                # Push branch
                await push_branch_to_remote(
                    repo_path, args.branch_name, github_token, args.github
                )

                # Create PR with automated and documentation labels
                pr_url = await create_pull_request(
                    args.github,
                    github_token,
                    args.branch_name,
                    args.pr_title,
                    args.pr_body,
                    labels=["automated", "documentation"],
                )

                vlogger.success(
                    f"Pull request created: {vlogger.format_url(pr_url)}",
                    emoji="github",
                )
            except Exception as e:
                vlogger.error(f"Error creating pull request: {e}", emoji="error")

        # Display cache statistics for GitHub repositories
        display_github_cache_stats(config, analyzer)

        # Display prompt cache performance metrics if available
        if hasattr(llm_client, "cache_manager"):
            try:
                metrics = llm_client.cache_manager.get_performance_metrics()
                if isinstance(metrics, dict) and metrics.get("prompt_cache_enabled"):
                    vlogger.prompt_cache_metrics(metrics)
                elif isinstance(metrics, dict):
                    vlogger.info("Prompt caching was disabled for this run", emoji="info")
            except (TypeError, AttributeError):
                pass

        vlogger.success(
            "Documentation generation completed successfully!", emoji="complete"
        )
    finally:
        # Clean up temporary directory if needed
        if temp_dir and not args.keep_clone:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                vlogger.info(
                    f"Removed temporary directory: {vlogger.format_path(temp_dir)}",
                    emoji="file",
                )
            except Exception as e:
                vlogger.warning(
                    f"Error removing temporary directory: {e}", emoji="warning"
                )


if __name__ == "__main__":
    # Get visual logger instance for error handling
    vlogger = get_visual_logger()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        vlogger.warning("Operation cancelled by user", emoji="warning")
        sys.exit(1)
    except Exception as e:
        vlogger.error(f"Error: {e}", emoji="error")
        if os.environ.get("DEBUG") or "--debug" in sys.argv:
            traceback.print_exc()
        sys.exit(1)
