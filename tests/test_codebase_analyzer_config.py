#!/usr/bin/env python3

"""
Tests for CodebaseAnalyzer with ScribeConfig
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.analyzers.codebase import CodebaseAnalyzer
from src.utils.config_class import ScribeConfig


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration object."""
    from src.utils.config_class import ScribeConfig, BlacklistConfig
    
    config = ScribeConfig()
    config.debug = True
    config.test_mode = True
    config.no_cache = True
    config.blacklist = BlacklistConfig(
        extensions=['.log', '.tmp'],
        path_patterns=['/node_modules/', '/__pycache__/']
    )
    return config


@pytest.fixture
def sample_config():
    """Create a sample ScribeConfig instance."""
    from src.utils.config_class import ScribeConfig, BlacklistConfig
    
    config = ScribeConfig()
    config.debug = True
    config.test_mode = True
    config.no_cache = True
    config.blacklist = BlacklistConfig(
        extensions=['.log', '.tmp'],
        path_patterns=['/node_modules/', '/__pycache__/']
    )
    return config


@pytest.fixture
def temp_repo_path():
    """Create a temporary directory for the repository."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    
    # Create some test files
    (repo_path / 'file1.py').write_text('print("Hello")')
    (repo_path / 'file2.js').write_text('console.log("Hello")')
    (repo_path / 'file3.log').write_text('Log content')  # Should be blacklisted
    
    # Create a subdirectory
    subdir = repo_path / 'subdir'
    subdir.mkdir()
    (subdir / 'file4.py').write_text('print("World")')
    
    # Create a node_modules directory (should be blacklisted)
    node_modules = repo_path / 'node_modules'
    node_modules.mkdir()
    (node_modules / 'package.json').write_text('{}')
    
    yield repo_path
    
    # Clean up any potential open connections to SQLite databases
    import gc
    gc.collect()  # Force garbage collection to close any lingering connections
    
    # Try to remove the directory, but don't fail if it can't be fully removed
    import shutil
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass  # Ignore errors during cleanup


class TestCodebaseAnalyzer:
    """Test suite for CodebaseAnalyzer with ScribeConfig."""

    def test_init_with_scribe_config(self, sample_config_dict, temp_repo_path):
        """Test initializing CodebaseAnalyzer with a ScribeConfig object."""
        analyzer = CodebaseAnalyzer(temp_repo_path, sample_config_dict)
        
        assert analyzer.debug is True
        assert analyzer.repo_path == temp_repo_path
        assert '.log' in analyzer.blacklist_extensions
        assert '/node_modules/' in analyzer.blacklist_patterns

    def test_init_with_scribe_config(self, sample_config, temp_repo_path):
        """Test initializing CodebaseAnalyzer with a ScribeConfig instance."""
        analyzer = CodebaseAnalyzer(temp_repo_path, sample_config)
        
        assert analyzer.debug is True
        assert analyzer.repo_path == temp_repo_path
        assert '.log' in analyzer.blacklist_extensions
        assert '/node_modules/' in analyzer.blacklist_patterns

    @patch('src.analyzers.codebase.CacheManager')
    def test_cache_initialization(self, mock_cache_manager, sample_config, temp_repo_path):
        """Test that the cache is initialized correctly."""
        mock_cache_instance = MagicMock()
        mock_cache_manager.return_value = mock_cache_instance
        
        analyzer = CodebaseAnalyzer(temp_repo_path, sample_config)
        
        # Check that CacheManager was called with the correct parameters
        mock_cache_manager.assert_called_once()
        args, kwargs = mock_cache_manager.call_args
        assert kwargs['enabled'] is False  # no_cache is True
        assert kwargs['repo_path'] == temp_repo_path
        assert kwargs['config'] == sample_config

    def test_analyze_repository(self, sample_config, temp_repo_path):
        """Test analyzing a repository."""
        analyzer = CodebaseAnalyzer(temp_repo_path, sample_config)
        
        # Analyze the repository
        manifest = analyzer.analyze_repository()
        
        # Check that the manifest contains the expected files
        assert 'file1.py' in manifest
        assert 'file2.js' in manifest
        
        # Handle platform-specific path separators
        import os
        subdir_path = os.path.join('subdir', 'file4.py')
        assert subdir_path in manifest
        
        # Check that blacklisted files are not included
        assert 'file3.log' not in manifest
        assert 'node_modules/package.json' not in manifest

    def test_github_repo_id(self, sample_config, temp_repo_path):
        """Test using a GitHub repo ID for caching."""
        # Add github_repo_id to the config
        sample_config_dict = sample_config.to_dict()
        sample_config_dict['github_repo_id'] = 'test/repo'
        config = ScribeConfig.from_dict(sample_config_dict)
        
        analyzer = CodebaseAnalyzer(temp_repo_path, config)
        
        # Check that the cache was initialized with the correct repo_identifier
        assert analyzer.cache.repo_identifier == 'test/repo'