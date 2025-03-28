#!/usr/bin/env python3

# Standard library imports
import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
import warnings
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Third-party imports
import psutil
import urllib3
from dotenv import load_dotenv

# Set environment variable for libmagic regex memory limit
os.environ['MAGIC_REGEX_MEMORY'] = os.environ.get('MAGIC_REGEX_MEMORY', '10000000')

# Local imports
from src.analyzers.codebase import CodebaseAnalyzer
from src.clients.base_llm import BaseLLMClient
from src.clients.llm_factory import LLMClientFactory
from src.generators.architecture import generate_architecture
from src.generators.contributing import generate_contributing
from src.generators.readme import generate_readme
from src.models.file_info import FileInfo
from src.utils.badges import generate_badges
from src.utils.cache import CacheManager
from src.utils.cache_utils import display_cache_stats, display_github_cache_stats
from src.utils.config_utils import load_config, update_config_with_args
from src.utils.config_class import ScribeConfig
from src.utils.doc_utils import add_ai_attribution
from src.utils.exceptions import (
    ScribeError,
    ConfigurationError,
    RepositoryError,
    FileProcessingError,
    LLMError,
    GitHubError
)
from src.utils.github_utils import (
    is_valid_github_url,
    clone_github_repository,
    create_git_branch,
    commit_documentation_changes,
    push_branch_to_remote,
    create_pull_request,
    extract_repo_info,
    prepare_github_branch
)
from src.utils.progress_utils import (
    create_file_processing_progress_bar,
    create_optimization_progress_bar,
    create_documentation_progress_bar
)
from src.utils.progress import ProgressTracker

# Load environment variables from .env file
load_dotenv()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# Disable bytecode caching
sys.dont_write_bytecode = True

def setup_logging(debug: bool = False, log_to_file: bool = True, quiet: bool = False) -> None:
    """Configure logging based on debug flag.
    
    Args:
        debug: If True, sets logging level to DEBUG, otherwise INFO
        log_to_file: If True, logs to both file and console, otherwise just console
        quiet: If True, reduces console output verbosity
    """
    # Set file logging level based on debug flag
    file_log_level = logging.DEBUG if debug else logging.INFO
    
    # Set console logging level based on quiet flag
    if quiet:
        console_log_level = logging.WARNING  # Only warnings and errors to console
    else:
        console_log_level = logging.DEBUG if debug else logging.INFO
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all logs to be processed
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create a timestamped log file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"readme_generator_{timestamp}.log"
        
        # Add file handler with appropriate level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")
    
    # Add console handler with appropriate level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    
    if debug:
        logging.debug("Debug logging enabled")

# Function moved: process_file is now a nested function within process_files

async def process_files(
    repo_path: Path,
    llm_client: BaseLLMClient,
    config: ScribeConfig,
    file_list: Optional[List[Path]] = None
) -> Dict[str, FileInfo]:
    """Process files in the repository.
    
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
    
    # Get file manifest
    file_manifest = {}
    
    # Cache statistics
    cache_stats = {
        "from_cache": 0,
        "from_llm": 0,
        "skipped": 0
    }
    
    # Start timing
    start_time = time.time()
    
    # Get list of files to process
    if file_list is None:
        files = analyzer._get_repository_files()
    else:
        files = file_list
    
    # Determine processing order
    if config.optimize_order:
        print("\nOptimizing file processing order...")
        with create_optimization_progress_bar() as progress:
            files = await determine_processing_order(files, llm_client, progress)
    
    # Process files
    print(f"\nProcessing {len(files)} files...")
    
    # Create progress bar
    with create_file_processing_progress_bar(len(files)) as progress:
        # Define nested function for file processing
        async def process_file(file_path: Path) -> Optional[FileInfo]:
            """Process a single file.
            
            Args:
                file_path: Path to the file
                
            Returns:
                FileInfo object or None if file should be skipped
            """
            # Skip binary files
            if analyzer.is_binary(file_path):
                progress.update(1)
                progress.set_postfix_str(f"Skipping binary file: {file_path.name}")
                cache_stats["skipped"] += 1
                return None
            
            # Skip files that should be excluded
            if not analyzer.should_include_file(file_path):
                progress.update(1)
                progress.set_postfix_str(f"Skipping excluded file: {file_path.name}")
                cache_stats["skipped"] += 1
                return None
            
            # Check if file has been processed before
            if not config.no_cache and analyzer.cache.enabled:
                cached_summary = analyzer.cache.get_cached_summary(file_path)
                if cached_summary and not analyzer.cache.is_file_changed(file_path):
                    # Use cached summary
                    progress.update(1)
                    progress.set_postfix_str(f"Using cached summary: {file_path.name}")
                    cache_stats["from_cache"] += 1
                    
                    # Create FileInfo object from cached summary
                    return FileInfo(
                        path=str(file_path.relative_to(repo_path)),
                        language=analyzer.get_file_language(file_path),
                        content=analyzer.read_file(file_path),
                        summary=cached_summary
                    )
            
            # Process file with LLM
            try:
                # Read file content
                content = analyzer.read_file(file_path)
                
                # Get file language
                language = analyzer.get_file_language(file_path)
                
                # Generate summary with LLM
                progress.update(0)
                progress.set_postfix_str(f"Generating summary: {file_path.name}")
                summary = await llm_client.generate_summary(
                    file_path=str(file_path.relative_to(repo_path)),
                    content=content,
                    file_type=language
                )
                
                # Cache summary
                if analyzer.cache.enabled and not config.no_cache:
                    analyzer.cache.save_summary(file_path, summary)
                
                # Update progress
                progress.update(1)
                progress.set_postfix_str(f"Processed: {file_path.name}")
                cache_stats["from_llm"] += 1
                
                # Create FileInfo object
                return FileInfo(
                    path=str(file_path.relative_to(repo_path)),
                    language=language,
                    content=content,
                    summary=summary
                )
            except Exception as e:
                # Log error and continue
                logging.error(f"Error processing file {file_path}: {e}")
                progress.update(1, description=f"Error: {file_path.name}")
                return None
        
        # Process files in batches
        concurrency = config.get_concurrency()
        
        # Process files in batches
        batch_size = concurrency
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            
            # Process batch
            results = await asyncio.gather(*[process_file(file) for file in batch])
            
            # Add results to manifest
            for result in results:
                if result:
                    file_manifest[result.path] = result
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Display cache statistics
    display_cache_stats(cache_stats, elapsed_time, analyzer.cache.enabled)
    
    # Return file manifest
    return file_manifest

async def determine_processing_order(files: List[Path], llm_client: BaseLLMClient, progress: Optional[Any] = None) -> List[Path]:
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

def add_ai_attribution_to_files(files: List[Path]) -> None:
    """Add AI attribution to files.
    
    Args:
        files: List of files to add attribution to
    """
    for file in files:
        try:
            # Read file content
            content = file.read_text(encoding='utf-8')
            
            # Add attribution
            content = add_ai_attribution(content)
            
            # Write back to file
            file.write_text(content, encoding='utf-8')
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
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check for malformed headings like "# # Heading"
        if line.startswith('# #'):
            # Count the number of # characters
            count = 0
            for char in line:
                if char == '#':
                    count += 1
                elif char != ' ':
                    break
            
            # Replace with the correct number of # characters
            fixed_line = '#' * count + line[count:]
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

async def main():
    """Main entry point for the codebase-scribe tool."""
    parser = argparse.ArgumentParser(description='Generate documentation for a code repository')
    
    # Create a mutually exclusive group for repo source
    repo_source = parser.add_mutually_exclusive_group(required=True)
    repo_source.add_argument('--repo', help='Path to the local repository')
    repo_source.add_argument('--github', help='GitHub repository URL (e.g., https://github.com/username/repo)')
    
    # Add authentication options for GitHub
    parser.add_argument('--github-token', help='GitHub Personal Access Token for private repositories')
    parser.add_argument('--keep-clone', action='store_true',
                      help='Keep cloned repository after processing (default: remove)')
    
    # Add branch and PR options
    parser.add_argument('--create-pr', action='store_true',
                       help='Create a pull request with generated documentation (GitHub only)')
    parser.add_argument('--branch-name', default='docs/auto-generated-readme',
                       help='Branch name for PR creation (default: docs/auto-generated-readme)')
    parser.add_argument('--pr-title', default='Add AI-generated documentation',
                       help='Title for the pull request')
    parser.add_argument('--pr-body', default='This PR adds automatically generated README.md, ARCHITECTURE.md, and CONTRIBUTING.md files.',
                       help='Body text for the pull request')
    
    # Existing arguments
    parser.add_argument('--output', '-o', default='README.md', help='Output file name')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-file', action='store_true', help='Log debug output to file')
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Reduce console output verbosity (only warnings and errors)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Enable test mode (process only first 5 files)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching of file summaries')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear the cache for this repository before processing')
    parser.add_argument('--optimize-order', action='store_true',
                       help='Use LLM to determine optimal file processing order')
    parser.add_argument('--llm-provider', choices=['ollama', 'bedrock'], default=None,
                       help='LLM provider to use (overrides config file)')
    args = parser.parse_args()
    
    # Set up logging based on debug flag and quiet mode
    setup_logging(debug=args.debug, log_to_file=args.log_file, quiet=args.quiet)
    
    # Load config and update with command-line args
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    if config.debug:
        logging.debug("Debug mode enabled")
    
    # Get GitHub token from args or environment
    github_token = args.github_token
    if not github_token and 'GITHUB_TOKEN' in os.environ:
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            print("Using GitHub token from environment variables")
    
    # Temp directory for cloned repo
    temp_dir = None
    
    # Create output directory for potential copying later (only if needed)
    output_dir = None
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
        
        print(f"Generated files will be saved to: {output_dir}")
    
    try:
        # Check if the user wants to create a PR
        create_pr = args.create_pr
        
        if create_pr and args.repo:
            print("Warning: --create-pr can only be used with --github. Ignoring PR creation.")
            create_pr = False
        
        if create_pr and not github_token:
            print("Error: GitHub token is required for PR creation. Use --github-token or set GITHUB_TOKEN environment variable.")
            return
        
        # Process GitHub URL if provided
        if args.github:
            # Validate GitHub URL
            if not is_valid_github_url(args.github):
                print(f"Error: Invalid GitHub repository URL: {args.github}")
                print("Expected format: https://github.com/username/repository")
                return
            
            # Extract a stable repo identifier for caching
            repo_owner, repo_name = extract_repo_info(args.github)
            repo_id = f"{repo_owner}/{repo_name}"
            print(f"Using repository ID for caching: {repo_id}")
            
            # Clone the repository
            print(f"Cloning GitHub repository: {args.github}")
            try:
                temp_dir = await clone_github_repository(args.github, github_token)
                repo_path = Path(temp_dir).absolute()
                print(f"Repository cloned successfully to: {repo_path}")
                
                # Tell analyzer to use the stable repo ID for caching
                config.github_repo_id = repo_id
            except Exception as e:
                print(f"Error: {e}")
                return
        else:
            # Use provided local repository path
            repo_path = Path(args.repo).absolute()
        
        # Handle cache clearing first and exit
        if args.clear_cache:
            analyzer = CodebaseAnalyzer(repo_path, config)
            analyzer.cache.clear_repo_cache()
            print(f"Cleared cache for repository: {repo_path.name}")
            
            # Close the cache connections before clearing all caches
            analyzer.cache.close()
            
            # Also clear the global cache for this repository
            CacheManager.clear_all_caches(repo_path=repo_path, config=config)
            
            if temp_dir and not args.keep_clone:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return  # Exit after clearing cache
        
        # Initialize LLM client using factory
        print(f"Initializing {config.llm_provider} client...")
        
        try:
            # Create and initialize the appropriate LLM client
            llm_client = await LLMClientFactory.create_client(config)
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            sys.exit(1)
        
        # Now start repository analysis
        print("\nInitializing repository analysis...")
        analyzer = CodebaseAnalyzer(repo_path, config)
        
        # First analyze repository to get file manifest
        print("Analyzing repository structure...")
        manifest = analyzer.analyze_repository(show_progress=True)  # Enable progress bar
        
        if not manifest:
            print("No files found to analyze!")
            return
        
        print(f"\nFound {len(manifest)} files to analyze")
        
        # Set project structure in LLM client
        llm_client.set_project_structure_from_manifest(manifest)
        
        # Process files
        file_manifest = await process_files(repo_path, llm_client, config)
        
        if not file_manifest:
            print("No files were processed successfully!")
            return
        
        # Generate badges
        badges = generate_badges(file_manifest, repo_path)
        
        # Generate architecture documentation
        print("\nGenerating architecture documentation...")
        with create_documentation_progress_bar(repo_path) as progress:
            architecture_content = await generate_architecture(
                repo_path=repo_path,
                file_manifest=file_manifest,
                llm_client=llm_client,
                config=config
            )
            progress.update(1)
        
        # Create file summaries dictionary
        file_summaries = {}
        for path, info in file_manifest.items():
            if hasattr(info, 'summary'):
                file_summaries[path] = info.summary
            elif isinstance(info, dict) and 'summary' in info:
                file_summaries[path] = info['summary']
        
        # Create analyzer instance if not already created
        analyzer = CodebaseAnalyzer(repo_path, config)
        analyzer.file_manifest = file_manifest
        
        # Create output directory
        output_dir = Path("generated_docs")
        
        # Generate Contributing guide first (before README)
        print("\nGenerating Contributing guide...")
        with create_documentation_progress_bar(repo_path) as progress:
            contributing_content = await generate_contributing(
                repo_path=repo_path,
                file_manifest=file_manifest,
                llm_client=llm_client,
                config=config,
                analyzer=analyzer
            )
            progress.update(1)
        
        # Generate README after Contributing guide exists
        print("\nGenerating README...")
        with create_documentation_progress_bar(repo_path) as progress:
            readme_content = await generate_readme(
                repo_path=repo_path,
                file_manifest=file_manifest,
                llm_client=llm_client,
                file_summaries=file_summaries,
                config=config,
                analyzer=analyzer,
                output_dir=str(output_dir),
                architecture_file_exists=True  # Set to True since we're generating architecture file
            )
            progress.update(1)
        
        # Add badges to README
        if badges:
            # Add badges after the first heading
            lines = readme_content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('# '):
                    # Insert badges after the first heading
                    lines.insert(i + 1, '')
                    lines.insert(i + 2, badges)
                    break
            readme_content = '\n'.join(lines)
        
        # Write files
        readme_path = repo_path / args.output
        architecture_path = repo_path / "docs" / "ARCHITECTURE.md"
        contributing_path = repo_path / "CONTRIBUTING.md"  # In root directory for GitHub recognition
        
        # Create docs directory if it doesn't exist
        os.makedirs(repo_path / "docs", exist_ok=True)
        
        # Fix malformed headings in README content
        readme_content = fix_malformed_headings(readme_content)
        contributing_content = fix_malformed_headings(contributing_content)
        
        # Write README
        print(f"\nWriting README to {readme_path}")
        readme_path.write_text(readme_content, encoding='utf-8')
        
        # Write architecture documentation
        print(f"Writing architecture documentation to {architecture_path}")
        architecture_path.write_text(architecture_content, encoding='utf-8')
        
        # Write contributing guide
        print(f"Writing contributing guide to {contributing_path}")
        contributing_path.write_text(contributing_content, encoding='utf-8')
        
        # Add AI attribution to files
        add_ai_attribution_to_files([readme_path, architecture_path, contributing_path])
        
        # Copy files to output directory if needed
        if output_dir:
            shutil.copy2(readme_path, output_dir / args.output)
            shutil.copy2(architecture_path, output_dir / "ARCHITECTURE.md")
            shutil.copy2(contributing_path, output_dir / "CONTRIBUTING.md")  # Copy from root to output dir
            print(f"\nFiles copied to {output_dir}")
        
        # Create PR if requested
        if create_pr:
            print("\nCreating pull request...")
            try:
                # Prepare branch
                github_token = os.environ.get('GITHUB_TOKEN')
                if not github_token:
                    raise GitHubError("GitHub token not found in environment variables")
                
                # First clean up any existing branch/PR
                await prepare_github_branch(args.github, github_token, args.branch_name)
                
                # Create the branch
                create_git_branch(repo_path, args.branch_name)
                
                # Commit changes
                commit_documentation_changes(repo_path, [readme_path, architecture_path, contributing_path])
                
                # Push branch
                await push_branch_to_remote(repo_path, args.branch_name, github_token, args.github)
                
                # Create PR with automated and documentation labels
                pr_url = await create_pull_request(
                    args.github,
                    github_token,
                    args.branch_name,
                    args.pr_title,
                    args.pr_body,
                    labels=["automated", "documentation"]
                )
                
                print(f"Pull request created: {pr_url}")
            except Exception as e:
                print(f"Error creating pull request: {e}")
        
        # Display cache statistics for GitHub repositories
        display_github_cache_stats(config, analyzer)
        
        print("\nDone!")
    finally:
        # Clean up temporary directory if needed
        if temp_dir and not args.keep_clone:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Error removing temporary directory: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if os.environ.get('DEBUG') or '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)