# GitHub Integration

This document provides detailed information about the GitHub integration utilities in the CodeBase Scribe AI project.

## Overview

The GitHub integration utilities (`src/utils/github_utils.py`) provide functionality for interacting with GitHub repositories, including:

- Cloning repositories
- Creating and managing branches
- Committing changes
- Creating pull requests
- Managing GitHub authentication

These utilities enable the CodeBase Scribe AI to work directly with GitHub repositories, generate documentation, and submit changes as pull requests.

## Key Components

### URL Validation and Parsing

```python
def is_valid_github_url(url: str) -> bool:
    """Validate if a URL is a valid GitHub repository URL."""
    
def extract_repo_info(github_url: str) -> Tuple[str, str]:
    """Extract owner and repository name from GitHub URL."""
```

These functions validate GitHub repository URLs and extract the owner and repository name, which are used throughout the GitHub integration process.

### Repository Cloning

```python
async def clone_github_repository(url: str, github_token: Optional[str] = None) -> Path:
    """Clone a GitHub repository to a temporary directory and return the path."""
```

This function clones a GitHub repository to a temporary directory, with support for:
- Authentication using GitHub tokens
- Fallback mechanisms if one cloning method fails
- Proper cleanup of temporary directories
- Cross-platform compatibility

### Branch Management

```python
def create_git_branch(repo_dir: Path, branch_name: str) -> bool:
    """Create a new git branch in the repository."""
    
async def push_branch_to_remote(repo_dir: Path, branch_name: str, token: str, github_url: str) -> bool:
    """Push a branch to the remote repository."""
    
def delete_branch(github_url: str, token: str, branch_name: str) -> bool:
    """Delete a branch from the repository."""
```

These functions handle Git branch operations, including creating branches locally, pushing them to GitHub, and deleting remote branches.

### Commit Management

```python
def commit_documentation_changes(repo_dir: Path, message: str = "Add generated documentation") -> bool:
    """Commit documentation changes to the repository."""
```

This function commits changes to documentation files (README.md and docs/ARCHITECTURE.md) to the repository.

### Pull Request Management

```python
async def prepare_github_branch(github_url: str, token: str, branch_name: str) -> bool:
    """Clean up any existing branch and PR for re-running."""
    
async def create_pull_request(github_url: str, token: str, branch_name: str, 
                             pr_title: str, pr_body: str) -> Optional[str]:
    """Create a pull request on GitHub using PyGithub."""
    
def find_existing_pr(github_url: str, token: str, branch_name: str) -> Optional[Dict[str, Any]]:
    """Find an existing pull request for the specified branch."""
    
def close_pull_request(pr_object) -> bool:
    """Close an open pull request."""
```

These functions manage pull requests, including creating new PRs, finding existing ones, and closing them if needed.

## Error Handling

The GitHub utilities use a custom exception hierarchy for better error handling:

- `GitHubUtilsError`: Base exception for all GitHub utility errors
- `GitHubAuthError`: Authentication-related errors
- `GitHubAPIError`: GitHub API-related errors
- `GitOperationError`: Git operation errors

This allows for more specific error handling and better error messages.

## Usage Examples

### Cloning a Repository

```python
import asyncio
from src.utils.github_utils import clone_github_repository

async def main():
    # Clone a public repository
    repo_path = await clone_github_repository("https://github.com/username/repo")
    print(f"Repository cloned to: {repo_path}")
    
    # Clone a private repository with authentication
    token = "your_github_token"
    repo_path = await clone_github_repository("https://github.com/username/private-repo", token)
    print(f"Private repository cloned to: {repo_path}")

asyncio.run(main())
```

### Creating a Pull Request

```python
import asyncio
from src.utils.github_utils import (
    clone_github_repository,
    create_git_branch,
    commit_documentation_changes,
    push_branch_to_remote,
    create_pull_request
)

async def create_documentation_pr():
    # Clone repository
    github_url = "https://github.com/username/repo"
    token = "your_github_token"
    repo_path = await clone_github_repository(github_url, token)
    
    # Create branch
    branch_name = "docs/update-readme"
    create_git_branch(repo_path, branch_name)
    
    # Make changes to documentation files
    # ... (your code to generate/update documentation)
    
    # Commit changes
    commit_documentation_changes(repo_path, "Update documentation")
    
    # Push branch
    await push_branch_to_remote(repo_path, branch_name, token, github_url)
    
    # Create PR
    pr_url = await create_pull_request(
        github_url,
        token,
        branch_name,
        "Documentation: Update README",
        "This PR updates the project documentation."
    )
    
    print(f"Pull request created: {pr_url}")

asyncio.run(create_documentation_pr())
```

## Configuration

The GitHub utilities don't require specific configuration files, but they do use environment variables:

- `GITHUB_TOKEN`: GitHub Personal Access Token for authentication

You can set this environment variable or provide the token directly to the functions that require it.

## Testing

The GitHub utilities have comprehensive test coverage using pytest and pytest-asyncio. The tests use mocking to avoid actual GitHub API calls during testing.

To run the tests:

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run the GitHub utilities tests
pytest tests/test_github_utils.py
```

## Best Practices

1. **Token Security**: Never hardcode GitHub tokens in your code. Use environment variables or secure credential storage.

2. **Error Handling**: Always handle exceptions from GitHub utilities, especially for network operations that might fail.

3. **Cleanup**: Always clean up temporary directories after use, especially when cloning repositories.

4. **Rate Limiting**: Be aware of GitHub API rate limits when making multiple API calls.

5. **Async Usage**: Use `await` with async functions and run them in an async context.

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:

1. Verify your GitHub token has the necessary permissions
2. Check that the token is valid and not expired
3. Ensure the token is correctly passed to the functions

### Network Issues

If you encounter network issues:

1. Check your internet connection
2. Verify that GitHub is accessible
3. Check if you're behind a proxy or firewall

### Permission Issues

If you encounter permission issues:

1. Verify that your GitHub token has the necessary permissions
2. Check that you have write access to the repository
3. Ensure you're not trying to push to a protected branch