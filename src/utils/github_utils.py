"""GitHub integration utilities for the readme generator.

This module provides utilities for interacting with GitHub repositories,
including cloning, branch management, and pull request creation.
"""

import os
import re
import tempfile
import logging
import urllib.parse
import shutil
import time
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, TypeVar, cast, List

# Constants for configuration
TEMP_DIR_PREFIX = "readme_gen_temp"
FILE_PERMISSIONS = 0o755  # More secure than 0o777
DIR_PERMISSIONS = 0o755
CLEANUP_WAIT_TIME = 1  # seconds to wait before cleanup
GIT_DEFAULT_BRANCH = "main"
GIT_REMOTE_NAME = "origin"

# Custom exceptions
class GitHubUtilsError(Exception):
    """Base exception for GitHub utilities."""
    pass

class GitHubAuthError(GitHubUtilsError):
    """Exception raised for authentication issues."""
    pass

class GitHubAPIError(GitHubUtilsError):
    """Exception raised for GitHub API issues."""
    pass

class GitOperationError(GitHubUtilsError):
    """Exception raised for Git operation issues."""
    pass

# Type variables for better typing
T = TypeVar('T')
GitType = TypeVar('GitType')
GithubType = TypeVar('GithubType')

# Optional imports - try to load them once
_git = None
_github_class = None

def _load_git() -> GitType:
    """Lazily load the git module when needed.
    
    Returns:
        The git module
        
    Raises:
        ImportError: If GitPython is not installed
    """
    global _git
    if _git is None:
        try:
            import git
            _git = git
        except ImportError:
            raise ImportError("GitPython not installed. Install with: pip install gitpython")
    return cast(GitType, _git)

def _load_github(token: Optional[str] = None) -> GithubType:
    """Lazily load the PyGithub module when needed.
    
    Args:
        token: GitHub token for authentication
        
    Returns:
        Github class if no token provided, or authenticated Github instance if token provided
        
    Raises:
        ImportError: If PyGithub is not installed
        GitHubAuthError: If authentication fails
    """
    global _github_class
    if _github_class is None:
        try:
            from github import Github
            _github_class = Github
        except ImportError:
            raise ImportError("PyGithub not installed. Install with: pip install PyGithub")
    
    # Return the class if no token, or an instance if token provided
    if token is None:
        return cast(GithubType, _github_class)
    else:
        try:
            return cast(GithubType, _github_class(token))
        except Exception as e:
            raise GitHubAuthError(f"Failed to authenticate with GitHub: {str(e)}") from e

def is_valid_github_url(url: str) -> bool:
    """Validate if a URL is a valid GitHub repository URL.
    
    Args:
        url: GitHub repository URL to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Examples:
        >>> is_valid_github_url("https://github.com/username/repo")
        True
        >>> is_valid_github_url("https://github.com/username/repo/")
        True
        >>> is_valid_github_url("https://github.com/username")
        False
    """
    # Basic pattern for GitHub repository URLs
    pattern = r'^https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+(/)?$'
    return re.match(pattern, url) is not None

def _create_auth_url(url: str, token: str) -> str:
    """Create an authenticated URL with token.
    
    Args:
        url: Base GitHub URL
        token: GitHub token for authentication
        
    Returns:
        Authenticated URL with token
    """
    parsed_url = urllib.parse.urlparse(url)
    auth_url = f"https://{token}@{parsed_url.netloc}{parsed_url.path}"
    if not auth_url.endswith('.git'):
        auth_url += '.git'
    return auth_url

async def clone_github_repository(url: str, github_token: Optional[str] = None) -> Path:
    """Clone a GitHub repository to a temporary directory and return the path.
    
    Args:
        url: GitHub repository URL
        github_token: Optional GitHub Personal Access Token for authentication
        
    Returns:
        Path to the cloned repository
    
    Raises:
        GitOperationError: If cloning fails
    """
    temp_dir = None
    
    # Try creating temp directory in user home first, then fallback to system temp
    try:
        # Create directory in user's home directory
        home_dir = Path.home()
        temp_base = home_dir / TEMP_DIR_PREFIX
        os.makedirs(temp_base, exist_ok=True)
        
        # Create a unique subdirectory
        timestamp = int(time.time())
        repo_name = url.split('/')[-1].replace('.git', '')
        temp_dir = temp_base / f"{repo_name}_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        logging.info(f"Created user home temporary directory: {temp_dir}")
    except Exception as e:
        logging.warning(f"Could not create directory in user home, falling back to system temp: {e}")
        # Fallback to system temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))
    
    try:
        # Set permissions (more secure than 0o777)
        os.chmod(temp_dir, DIR_PERMISSIONS)
        logging.info(f"Created temporary directory: {temp_dir}")
        
        # Modify URL to include token if provided
        clone_url = url
        if github_token:
            clone_url = _create_auth_url(url, github_token)
        
        # Try using direct subprocess call instead of GitPython
        logging.info(f"Cloning repository: {url}")
        
        # Use asyncio subprocess for better async behavior
        try:
            # First try with asyncio subprocess
            process = await asyncio.create_subprocess_exec(
                "git", "clone", clone_url, str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace') if stderr else "Unknown error"
                logging.warning(f"Subprocess git clone failed: {error_msg}")
                # Fall back to GitPython
                git = _load_git()
                logging.info("Falling back to GitPython")
                git.Repo.clone_from(clone_url, str(temp_dir))
        except Exception as e:
            logging.warning(f"Subprocess approach failed: {str(e)}, trying GitPython")
            git = _load_git()
            git.Repo.clone_from(clone_url, str(temp_dir))
            
        logging.info(f"Repository cloned successfully to {temp_dir}")
        return temp_dir
        
    except Exception as e:
        logging.error(f"Error cloning repository: {str(e)}")
        await cleanup_temp_dir(temp_dir)
        raise GitOperationError(f"Failed to clone repository: {str(e)}") from e

async def cleanup_temp_dir(temp_dir: Path) -> None:
    """Clean up a temporary directory, handling Git repositories properly.
    
    Args:
        temp_dir: Path to the temporary directory to clean up
        
    This function attempts multiple cleanup strategies to ensure the directory
    is properly removed, even if files are locked by Git processes.
    """
    if not temp_dir or not os.path.exists(temp_dir):
        return
        
    try:
        # Wait a moment for any processes to release locks
        await asyncio.sleep(CLEANUP_WAIT_TIME)
        
        # On Windows, make files writable before removal
        if os.name == 'nt':
            for root, dirs, files in os.walk(temp_dir):
                for dir_name in dirs:
                    try:
                        dir_path = os.path.join(root, dir_name)
                        os.chmod(dir_path, FILE_PERMISSIONS)
                    except Exception as e:
                        logging.debug(f"Could not change permissions on directory {dir_path}: {e}")
                for file_name in files:
                    try:
                        file_path = os.path.join(root, file_name)
                        os.chmod(file_path, FILE_PERMISSIONS)
                    except Exception as e:
                        logging.debug(f"Could not change permissions on file {file_path}: {e}")
        
        # Try to remove with standard approach first
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Standard cleanup failed, trying alternative approach: {e}")
            
            # Try with subprocess on Windows
            if os.name == 'nt':
                try:
                    # Use asyncio subprocess for better async behavior
                    process = await asyncio.create_subprocess_exec(
                        'cmd', '/c', f'rd /s /q "{temp_dir}"',
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    _, stderr = await process.communicate()
                    if process.returncode != 0 and stderr:
                        error_msg = stderr.decode('utf-8', errors='replace')
                        logging.warning(f"CMD cleanup failed: {error_msg}")
                except Exception as e2:
                    logging.warning(f"CMD cleanup failed: {e2}")
            else:
                # For non-Windows, try with force
                try:
                    # Use asyncio subprocess for better async behavior
                    process = await asyncio.create_subprocess_exec(
                        'rm', '-rf', str(temp_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await process.communicate()
                except Exception as e2:
                    logging.warning(f"rm -rf cleanup failed: {e2}")
                    
        if not os.path.exists(temp_dir):
            logging.info(f"Successfully cleaned up temporary directory: {temp_dir}")
        else:
            logging.warning(f"Could not completely remove directory: {temp_dir}")
    except Exception as e:
        logging.warning(f"Failed to clean up directory {temp_dir}: {str(e)}")

def _get_github_repo(github_url: str, token: str):
    """Helper function to get a GitHub repository instance.
    
    Args:
        github_url: GitHub repository URL
        token: GitHub token for authentication
        
    Returns:
        GitHub repository instance
        
    Raises:
        GitHubAuthError: If authentication fails
        GitHubAPIError: If the repository cannot be accessed
    """
    try:
        g = _load_github(token)
        owner, repo_name = extract_repo_info(github_url)
        return g.get_repo(f"{owner}/{repo_name}")
    except GitHubAuthError:
        raise
    except Exception as e:
        raise GitHubAPIError(f"Failed to access GitHub repository: {str(e)}") from e

def find_existing_pr(github_url: str, token: str, branch_name: str) -> Optional[Dict[str, Any]]:
    """Find an existing pull request for the specified branch.
    
    Args:
        github_url: GitHub repository URL
        token: GitHub token for authentication
        branch_name: Name of the branch to check
        
    Returns:
        Optional dict with PR info if found, None otherwise
        
    Raises:
        GitHubAuthError: If authentication fails
        GitHubAPIError: If the repository cannot be accessed
    """
    try:
        repo = _get_github_repo(github_url, token)
        
        # Look for open PRs from this branch
        for pr in repo.get_pulls(state='open'):
            if pr.head.ref == branch_name:
                logging.info(f"Found existing PR #{pr.number}: {pr.title}")
                return {
                    'number': pr.number,
                    'url': pr.html_url,
                    'title': pr.title,
                    'object': pr
                }
        
        logging.info(f"No existing PRs found for branch '{branch_name}'")
        return None
    except ImportError:
        logging.error("PyGithub package not installed. Install with: pip install PyGithub")
        return None
    except (GitHubAuthError, GitHubAPIError) as e:
        logging.error(f"GitHub API error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error finding existing PR: {str(e)}")
        return None

def close_pull_request(pr_object) -> bool:
    """Close an open pull request.
    
    Args:
        pr_object: Pull request object from PyGithub
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        pr_object.edit(state="closed")
        logging.info(f"Closed pull request #{pr_object.number}: {pr_object.title}")
        return True
    except Exception as e:
        logging.error(f"Error closing pull request: {str(e)}")
        return False

def delete_branch(github_url: str, token: str, branch_name: str) -> bool:
    """Delete a branch from the repository.
    
    Args:
        github_url: GitHub repository URL
        token: GitHub token for authentication
        branch_name: Name of the branch to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        repo = _get_github_repo(github_url, token)
        
        # Find the reference to the branch
        git_ref = f"heads/{branch_name}"
        try:
            ref = repo.get_git_ref(git_ref)
            ref.delete()
            logging.info(f"Deleted branch '{branch_name}' from remote repository")
            return True
        except Exception as e:
            if "Not Found" in str(e):
                logging.info(f"Branch '{branch_name}' doesn't exist in remote repository")
                return True  # Consider it a success if branch doesn't exist
            else:
                raise
    except ImportError:
        logging.error("PyGithub package not installed. Install with: pip install PyGithub")
        return False
    except (GitHubAuthError, GitHubAPIError) as e:
        logging.error(f"GitHub API error: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error deleting branch: {str(e)}")
        return False

def create_git_branch(repo_dir: Path, branch_name: str) -> bool:
    """Create a new git branch in the repository.
    
    Args:
        repo_dir: Path to the repository
        branch_name: Name of the branch to create
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        GitOperationError: If branch creation fails
    """
    try:
        git = _load_git()
        
        # Open the repository
        repo = git.Repo(repo_dir)
        
        # Check if we're already on the branch
        current_branch = repo.active_branch.name
        if current_branch == branch_name:
            logging.info(f"Already on branch {branch_name}")
            return True
        
        # Check if branch exists locally
        branch_exists = branch_name in [ref.name for ref in repo.references if isinstance(ref, git.refs.Reference)]
        
        if branch_exists:
            # Checkout existing branch
            repo.git.checkout(branch_name)
            logging.info(f"Checked out existing branch: {branch_name}")
        else:
            # Create and checkout new branch
            repo.git.checkout('-b', branch_name)
            logging.info(f"Created and checked out branch: {branch_name}")
            
        return True
        
    except ImportError:
        logging.error("GitPython not installed. Install with: pip install gitpython")
        return False
    except Exception as e:
        logging.error(f"Error creating branch: {str(e)}")
        raise GitOperationError(f"Failed to create or checkout branch '{branch_name}': {str(e)}") from e
def commit_documentation_changes(repo_dir: Path, files: list = None, message: str = "Add generated documentation") -> bool:
    """Commit documentation changes to the repository.
    
    Args:
        repo_dir: Path to the repository
        files: List of files to commit (optional)
        message: Commit message
        message: Commit message
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        GitOperationError: If commit operation fails
    """
    try:
        git = _load_git()
        
        # Open the repository
        repo = git.Repo(repo_dir)
        
        # Add files to the index
        if files:
            for file_path in files:
                # Convert to relative path if needed
                if isinstance(file_path, Path):
                    try:
                        # Get relative path from repo_dir to file_path
                        rel_path = file_path.relative_to(repo_dir)
                        repo.git.add(str(rel_path))
                        logging.debug(f"Added {rel_path} to git index")
                    except ValueError:
                        # If file_path is not relative to repo_dir, use it as is
                        repo.git.add(str(file_path))
                        logging.debug(f"Added {file_path} to git index")
                else:
                    # If it's already a string, use it directly
                    repo.git.add(file_path)
                    logging.debug(f"Added {file_path} to git index")
        else:
            # Default behavior: add README.md and docs/ARCHITECTURE.md
            repo.git.add("README.md")
            logging.debug("Added README.md to git index")
            
            # Ensure docs directory exists before adding
            docs_dir = repo_dir / "docs"
            arch_file = docs_dir / "ARCHITECTURE.md"
            
            if arch_file.exists():
                repo.git.add("docs/ARCHITECTURE.md")
                logging.debug("Added docs/ARCHITECTURE.md to git index")
            else:
                logging.debug("Architecture file not found, skipping")
        
        # Check if there are changes to be committed
        if not repo.index.diff("HEAD") and not repo.untracked_files:
            logging.info("No changes to commit")
            return True
            
        # Commit changes
        commit = repo.index.commit(message)
        logging.info(f"Committed changes with message: '{message}' (commit: {commit.hexsha[:8]})")
        return True
        
    except ImportError:
        logging.error("GitPython not installed. Install with: pip install gitpython")
        return False
    except Exception as e:
        logging.error(f"Error committing changes: {str(e)}")
        raise GitOperationError(f"Failed to commit documentation changes: {str(e)}") from e

async def push_branch_to_remote(repo_dir: Path, branch_name: str, token: str, github_url: str) -> bool:
    """Push a branch to the remote repository.
    
    Args:
        repo_dir: Path to the repository
        branch_name: Name of the branch to push
        token: GitHub token for authentication
        github_url: GitHub repository URL
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        GitOperationError: If push operation fails
    """
    repo = None
    try:
        git = _load_git()
        
        # Open the repository
        repo = git.Repo(repo_dir)
        
        # Ensure we're on the correct branch
        if repo.active_branch.name != branch_name:
            logging.info(f"Switching to branch {branch_name}")
            repo.git.checkout(branch_name)
        
        # Create authenticated URL with token
        auth_url = _create_auth_url(github_url, token)
            
        # Check if remote exists, set URL with token
        remote_name = GIT_REMOTE_NAME
        if remote_name in [remote.name for remote in repo.remotes]:
            remote = repo.remote(remote_name)
            # Update remote URL with authentication
            logging.debug(f"Updating remote URL for {remote_name}")
            remote.set_url(auth_url)
        else:
            # Add new remote
            logging.info(f"Creating new remote {remote_name}")
            remote = repo.create_remote(remote_name, auth_url)
        
        # Push to remote
        logging.info(f"Pushing branch {branch_name} to remote")
        push_info = remote.push(branch_name, force=True)  # Use force=True to ensure we can update existing branch
        
        # Check push results
        for info in push_info:
            if info.flags & info.ERROR:
                error_msg = f"Push failed: {info.summary}"
                logging.error(error_msg)
                raise GitOperationError(error_msg)
                
        logging.info(f"Successfully pushed branch {branch_name} to remote")
        
        # Reset remote URL to not contain token (security)
        remote.set_url(github_url)
        
        return True
        
    except ImportError:
        logging.error("GitPython not installed. Install with: pip install gitpython")
        return False
    except Exception as e:
        logging.error(f"Error pushing branch: {str(e)}")
        raise GitOperationError(f"Failed to push branch '{branch_name}' to remote: {str(e)}") from e
    finally:
        # Close the repository to prevent file locks
        if repo:
            try:
                repo.close()
                if hasattr(repo.git, 'clear_cache'):
                    repo.git.clear_cache()
                logging.debug("Repository closed successfully")
            except Exception as e:
                logging.debug(f"Error closing repository: {str(e)}")

def extract_repo_info(github_url: str) -> Tuple[str, str]:
    """Extract owner and repository name from GitHub URL.
    
    Args:
        github_url: GitHub repository URL
        
    Returns:
        Tuple of (owner, repo_name)
        
    Raises:
        ValueError: If the URL format is invalid
    """
    # Parse the URL
    parsed_url = urllib.parse.urlparse(github_url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    # Validate URL format
    if not parsed_url.netloc or parsed_url.netloc.lower() != 'github.com':
        raise ValueError(f"Not a GitHub URL: {github_url}")
    
    # Ensure we have both owner and repo parts
    if len(path_parts) < 2:
        raise ValueError(f"URL does not contain both owner and repository: {github_url}")
    
    # The owner is the first part of the path
    owner = path_parts[0]
    if not owner:
        raise ValueError(f"Could not extract owner from URL: {github_url}")
    
    # The repo is the second part of the path
    repo_name = path_parts[1]
    if not repo_name:
        raise ValueError(f"Could not extract repository name from URL: {github_url}")
    
    logging.debug(f"Extracted repo info from {github_url}: owner={owner}, repo={repo_name}")
    return owner, repo_name

async def prepare_github_branch(github_url: str, token: str, branch_name: str) -> bool:
    """Clean up any existing branch and PR for re-running.
    
    This function checks for existing PRs and branches with the same name,
    closes any open PRs, and deletes the branch to start fresh.
    
    Args:
        github_url: GitHub repository URL
        token: GitHub token for authentication
        branch_name: Name of the branch to prepare
        
    Returns:
        bool: True if cleanup successful, False otherwise
        
    Raises:
        GitHubAuthError: If authentication fails
        GitHubAPIError: If the repository cannot be accessed
    """
    logging.info(f"Preparing branch '{branch_name}' for GitHub repository")
    
    # Check for existing PR
    existing_pr = find_existing_pr(github_url, token, branch_name)
    
    if existing_pr:
        logging.info(f"Found existing PR #{existing_pr['number']}: {existing_pr['title']}")
        # Close the PR
        if not close_pull_request(existing_pr['object']):
            logging.warning("Failed to close existing PR. Continuing anyway...")
    
    # Delete the branch if it exists
    if not delete_branch(github_url, token, branch_name):
        logging.warning("Failed to delete existing branch. Continuing anyway...")
    
    logging.info(f"Branch preparation completed for '{branch_name}'")
    return True

async def create_pull_request(github_url: str, token: str, branch_name: str,
                             pr_title: str, pr_body: str,
                             labels: List[str] = None) -> Optional[str]:
    """Create a pull request on GitHub using PyGithub.
    
    Args:
        github_url: GitHub repository URL
        token: GitHub token for authentication
        branch_name: Name of the branch to create PR from
        pr_title: Title for the pull request
        pr_body: Body text for the pull request
        labels: List of labels to add to the pull request (default: None)
        
    Returns:
        Optional[str]: URL of the created pull request, or None if creation failed
        
    Raises:
        GitHubAuthError: If authentication fails
        GitHubAPIError: If the repository cannot be accessed
    """
    # Default labels for documentation PRs
    if labels is None:
        labels = ["automated", "documentation"]
        
    try:
        # Get repository using helper function
        repo = _get_github_repo(github_url, token)
        
        # Get default branch to use as base
        default_branch = repo.default_branch
        logging.info(f"Using default branch '{default_branch}' as PR base")
        
        # Create pull request
        pr = repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base=default_branch
        )
        
        # Add labels to the pull request
        if labels:
            pr.add_to_labels(*labels)
            logging.info(f"Added labels to PR: {', '.join(labels)}")
        
        logging.info(f"Created pull request #{pr.number}: {pr.title}")
        logging.info(f"Pull request URL: {pr.html_url}")
        return pr.html_url
    except ImportError:
        logging.error("PyGithub package not installed. Install with: pip install PyGithub")
        return None
    except (GitHubAuthError, GitHubAPIError) as e:
        logging.error(f"GitHub API error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error creating pull request: {str(e)} ({type(e).__name__})")
        return None