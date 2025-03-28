"""Tests for the GitHub utilities module."""

import os
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Import pytest-asyncio for async tests
pytest_plugins = ['pytest_asyncio']

from src.utils.github_utils import (
    is_valid_github_url,
    extract_repo_info,
    _create_auth_url,
    create_pull_request,
    GitHubAuthError,
    GitHubAPIError,
    GitOperationError
)

class TestGitHubUtils:
    """Test suite for GitHub utilities."""

    def test_is_valid_github_url(self):
        """Test GitHub URL validation."""
        # Valid URLs
        assert is_valid_github_url("https://github.com/username/repo")
        assert is_valid_github_url("https://github.com/username/repo/")
        assert is_valid_github_url("http://github.com/username/repo")
        assert is_valid_github_url("https://github.com/username/repo-with-dash")
        assert is_valid_github_url("https://github.com/username/repo.with.dots")
        assert is_valid_github_url("https://github.com/username/repo_with_underscore")
        
        # Invalid URLs
        assert not is_valid_github_url("https://github.com/username")
        assert not is_valid_github_url("https://github.com/")
        assert not is_valid_github_url("https://gitlab.com/username/repo")
        assert not is_valid_github_url("https://github.com/username/repo/tree/main")
        assert not is_valid_github_url("github.com/username/repo")
        assert not is_valid_github_url("https://github.com/username/repo/issues")

    def test_extract_repo_info(self):
        """Test extraction of owner and repo name from GitHub URL."""
        # Test valid URLs
        owner, repo = extract_repo_info("https://github.com/username/repo")
        assert owner == "username"
        assert repo == "repo"
        
        owner, repo = extract_repo_info("https://github.com/org-name/repo-name/")
        assert owner == "org-name"
        assert repo == "repo-name"
        
        # Test with trailing slash
        owner, repo = extract_repo_info("https://github.com/username/repo/")
        assert owner == "username"
        assert repo == "repo"
        
        # Test invalid URLs
        with pytest.raises(ValueError):
            extract_repo_info("https://github.com/")
            
        with pytest.raises(ValueError):
            extract_repo_info("https://github.com/username")

    def test_create_auth_url(self):
        """Test creation of authenticated URL with token."""
        url = "https://github.com/username/repo"
        token = "abc123"
        
        auth_url = _create_auth_url(url, token)
        assert auth_url == "https://abc123@github.com/username/repo.git"
        
        # Test with URL already ending in .git
        url = "https://github.com/username/repo.git"
        auth_url = _create_auth_url(url, token)
        assert auth_url == "https://abc123@github.com/username/repo.git"

    @patch('src.utils.github_utils._load_github')
    def test_github_auth_error(self, mock_load_github):
        """Test GitHub authentication error handling."""
        from src.utils.github_utils import _get_github_repo
        
        # Setup mock to raise GitHubAuthError
        mock_load_github.side_effect = GitHubAuthError("Authentication failed")
        
        # Test that the error is propagated
        with pytest.raises(GitHubAuthError):
            _get_github_repo("https://github.com/username/repo", "invalid_token")

    @pytest.mark.asyncio
    @patch('src.utils.github_utils.asyncio.create_subprocess_exec')
    @patch('src.utils.github_utils._load_git')
    @patch('src.utils.github_utils.cleanup_temp_dir')
    @patch('src.utils.github_utils.os.makedirs')
    @patch('src.utils.github_utils.os.chmod')
    @patch('src.utils.github_utils.Path.home')
    @patch('src.utils.github_utils.tempfile.mkdtemp')
    async def test_clone_github_repository_error(
        self, mock_mkdtemp, mock_home, mock_chmod, mock_makedirs,
        mock_cleanup, mock_load_git, mock_subprocess
    ):
        """Test error handling in clone_github_repository."""
        from src.utils.github_utils import clone_github_repository
        
        # Setup mocks for directory creation
        mock_home.return_value = Path("/fake/home")
        mock_mkdtemp.return_value = "/fake/temp/dir"
        
        # Setup mock for subprocess (git clone)
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error cloning")
        mock_subprocess.return_value = mock_process
        
        # Setup mock git to raise an exception
        mock_git = MagicMock()
        mock_load_git.return_value = mock_git
        mock_git.Repo.clone_from.side_effect = Exception("Clone failed")
        
        # Make cleanup_temp_dir a coroutine that returns None
        mock_cleanup.return_value = None
        
        # Test that the error is propagated and cleanup is called
        with pytest.raises(GitOperationError):
            await clone_github_repository("https://github.com/username/repo", "token")
            
        # Verify cleanup was called
        assert mock_cleanup.called

    @pytest.mark.asyncio
    @patch('src.utils.github_utils._load_git')
    async def test_create_git_branch(self, mock_load_git):
        """Test creating a git branch."""
        from src.utils.github_utils import create_git_branch
        
        # Create a proper Reference class for isinstance check
        class Reference:
            pass
        
        # Setup mock git module
        mock_git = MagicMock()
        mock_git.refs.Reference = Reference
        mock_repo = MagicMock()
        mock_load_git.return_value = mock_git
        mock_git.Repo.return_value = mock_repo
        
        # Setup active branch
        mock_repo.active_branch.name = "main"
        
        # Setup references - make it an instance of our Reference class
        mock_ref = MagicMock(spec=Reference)
        mock_ref.name = "test-branch"
        mock_repo.references = [mock_ref]
        
        # Test creating an existing branch
        result = create_git_branch(Path("/fake/path"), "test-branch")
        assert result is True
        mock_repo.git.checkout.assert_called_with("test-branch")
        
        # Test creating a new branch
        mock_repo.references = []  # No references for new branch test
        result = create_git_branch(Path("/fake/path"), "new-branch")
        assert result is True
        mock_repo.git.checkout.assert_called_with("-b", "new-branch")

    @pytest.mark.asyncio
    @patch('src.utils.github_utils._get_github_repo')
    async def test_find_existing_pr(self, mock_get_repo):
        """Test finding an existing pull request."""
        from src.utils.github_utils import find_existing_pr
        
        # Setup mock GitHub repo
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_get_repo.return_value = mock_repo
        
        # Setup PR with matching branch
        mock_pr.head.ref = "test-branch"
        mock_pr.number = 123
        mock_pr.html_url = "https://github.com/username/repo/pull/123"
        mock_pr.title = "Test PR"
        mock_repo.get_pulls.return_value = [mock_pr]
        
        # Test finding an existing PR
        result = find_existing_pr("https://github.com/username/repo", "token", "test-branch")
        assert result is not None
        assert result["number"] == 123
        assert result["url"] == "https://github.com/username/repo/pull/123"
        assert result["title"] == "Test PR"
        assert result["object"] == mock_pr
        
        # Test not finding a PR
        mock_repo.get_pulls.return_value = []
        result = find_existing_pr("https://github.com/username/repo", "token", "non-existent-branch")
        assert result is None
        
    @pytest.mark.asyncio
    @patch('src.utils.github_utils._get_github_repo')
    async def test_create_pull_request(self, mock_get_repo):
        """Test creating a pull request with labels."""
        # Setup mock GitHub repo and PR
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_get_repo.return_value = mock_repo
        mock_repo.create_pull.return_value = mock_pr
        mock_repo.default_branch = "main"
        mock_pr.number = 123
        mock_pr.html_url = "https://github.com/username/repo/pull/123"
        mock_pr.title = "Test PR"
        
        # Test creating a PR with default labels
        result = await create_pull_request(
            "https://github.com/username/repo",
            "token",
            "test-branch",
            "Test PR",
            "Test PR body"
        )
        
        # Verify PR was created with correct parameters
        mock_repo.create_pull.assert_called_with(
            title="Test PR",
            body="Test PR body",
            head="test-branch",
            base="main"
        )
        
        # Verify labels were added
        mock_pr.add_to_labels.assert_called_with("automated", "documentation")
        assert result == "https://github.com/username/repo/pull/123"
        
        # Test creating a PR with custom labels
        mock_pr.add_to_labels.reset_mock()
        result = await create_pull_request(
            "https://github.com/username/repo",
            "token",
            "test-branch",
            "Test PR",
            "Test PR body",
            labels=["custom-label", "another-label"]
        )
        
        # Verify custom labels were added
        mock_pr.add_to_labels.assert_called_with("custom-label", "another-label")
        assert result == "https://github.com/username/repo/pull/123"
        
        # Test creating a PR with no labels
        mock_pr.add_to_labels.reset_mock()
        result = await create_pull_request(
            "https://github.com/username/repo",
            "token",
            "test-branch",
            "Test PR",
            "Test PR body",
            labels=[]
        )
        
        # Verify no labels were added
        assert not mock_pr.add_to_labels.called
        assert result == "https://github.com/username/repo/pull/123"