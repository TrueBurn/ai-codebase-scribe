#!/usr/bin/env python3

"""
Tests for CacheManager with ScribeConfig
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import CacheManager at the module level
from src.utils.cache import CacheManager
from src.utils.config_class import ScribeConfig


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration object."""
    from src.utils.config_class import ScribeConfig, CacheConfig
    
    config = ScribeConfig()
    config.debug = True
    config.cache = CacheConfig(
        enabled=True,
        directory='.test_cache',
        location='repo',
        hash_algorithm='sha256',
        global_directory='.test_global_cache'
    )
    return config


@pytest.fixture
def sample_config():
    """Create a sample ScribeConfig instance."""
    from src.utils.config_class import ScribeConfig, CacheConfig
    
    config = ScribeConfig()
    config.debug = True
    config.cache = CacheConfig(
        enabled=True,
        directory='.test_cache',
        location='repo',  # Same as sample_config_dict
        hash_algorithm='md5',  # Different from sample_config_dict
        global_directory='.test_global_cache'
    )
    return config


@pytest.fixture
def temp_repo_path():
    """Create a temporary directory for the repository."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    
    # Clean up any potential open connections to SQLite databases
    import gc
    gc.collect()  # Force garbage collection to close any lingering connections
    
    # Try to remove the directory, but don't fail if it can't be fully removed
    import shutil
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass  # Ignore errors during cleanup


class TestCacheManager:
    """Test suite for CacheManager with ScribeConfig."""

    def test_init_with_scribe_config(self, sample_config_dict, temp_repo_path):
        """Test initializing CacheManager with a ScribeConfig object."""
        cache_manager = CacheManager(
            enabled=True,
            repo_identifier='test-repo',
            repo_path=temp_repo_path,
            config=sample_config_dict
        )
        
        try:
            assert cache_manager.enabled is True
            assert cache_manager.repo_identifier == 'test-repo'
            assert cache_manager.hash_algorithm == 'sha256'
            assert cache_manager.debug is True
            assert cache_manager.cache_dir == temp_repo_path / '.test_cache'
        finally:
            # Ensure connections are closed
            cache_manager.close()
            CacheManager.close_all_connections()

    def test_init_with_sample_config(self, sample_config, temp_repo_path):
        """Test initializing CacheManager with a different ScribeConfig instance."""
        cache_manager = CacheManager(
            enabled=True,
            repo_identifier='test-repo',
            repo_path=temp_repo_path,
            config=sample_config
        )
        
        try:
            assert cache_manager.enabled is True
            assert cache_manager.repo_identifier == 'test-repo'
            assert cache_manager.hash_algorithm == 'md5'  # Different from test_init_with_scribe_config
            assert cache_manager.debug is True
            assert cache_manager.cache_dir == temp_repo_path / '.test_cache'
        finally:
            # Ensure connections are closed
            cache_manager.close()
            CacheManager.close_all_connections()

    def test_home_directory_cache(self, sample_config):
        """Test cache in home directory."""
        # Set location to 'home'
        sample_config.cache.location = 'home'
        
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path('/mock/home')
            
            cache_manager = CacheManager(
                enabled=True,
                repo_identifier='test-repo',
                config=sample_config
            )
            
            try:
                assert cache_manager.cache_dir == Path('/mock/home') / '.test_global_cache'
            finally:
                # Ensure connections are closed
                cache_manager.close()
                CacheManager.close_all_connections()

    def test_repo_cache_dir(self, sample_config, temp_repo_path):
        """Test getting repository cache directory."""
        cache_manager = CacheManager(
            enabled=True,
            repo_identifier='test-repo',
            repo_path=temp_repo_path,
            config=sample_config
        )
        
        try:
            repo_cache_dir = cache_manager.get_repo_cache_dir()
            assert repo_cache_dir.parent == cache_manager.cache_dir
            assert 'test-repo' in str(repo_cache_dir)
        finally:
            # Ensure connections are closed
            cache_manager.close()
            CacheManager.close_all_connections()

    def test_cache_operations(self, sample_config, temp_repo_path):
        """Test basic cache operations."""
        cache_manager = CacheManager(
            enabled=True,
            repo_identifier='test-repo',
            repo_path=temp_repo_path,
            config=sample_config
        )
        
        try:
            # Create a test file
            test_file = temp_repo_path / 'test.txt'
            with open(test_file, 'w') as f:
                f.write('Test content')
            
            # Test saving and retrieving from cache
            summary = 'This is a test summary'
            cache_manager.save_summary(test_file, summary)
            
            retrieved_summary = cache_manager.get_cached_summary(test_file)
            assert retrieved_summary == summary
        finally:
            # Ensure connections are closed
            cache_manager.close()
            CacheManager.close_all_connections()