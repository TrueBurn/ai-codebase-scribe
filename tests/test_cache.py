import pytest
import time
import os
import sqlite3
import pickle
import hashlib
from pathlib import Path
from src.utils.cache import CacheManager, CacheEntry, SQLiteCache, MemoryCache

@pytest.fixture
def test_file(tmp_path):
    """Create a test file in the temporary directory"""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir(parents=True, exist_ok=True)
    file_path = test_dir / "test.py"
    file_path.write_text("print('test')")
    return file_path

@pytest.fixture
def cache_config():
    """Standard cache configuration for tests"""
    from src.utils.config_class import ScribeConfig, CacheConfig
    
    config = ScribeConfig()
    config.cache = CacheConfig(
        directory='.cache',
        location='repo',
        hash_algorithm='md5'
    )
    return config

@pytest.fixture
def cache_manager(tmp_path, cache_config):
    """Create a cache manager with a temporary directory"""
    # Create a CacheManager with the correct parameters
    cm = CacheManager(
        repo_path=tmp_path, 
        config=cache_config
    )
    return cm

def test_save_and_get_summary(cache_manager, test_file):
    """Test saving and retrieving a summary"""
    test_summary = "This is a test summary"
    cache_manager.save_summary(test_file, test_summary)
    
    retrieved_summary = cache_manager.get_cached_summary(test_file)
    assert retrieved_summary == test_summary

def test_file_change_detection(cache_manager, test_file):
    """Test detection of file changes"""
    # Save initial summary
    cache_manager.save_summary(test_file, "initial summary")
    
    # File hasn't changed
    assert not cache_manager.is_file_changed(test_file)
    
    # Modify file
    time.sleep(0.1)  # Ensure file timestamp changes
    test_file.write_text("print('modified')")
    
    # File has changed
    assert cache_manager.is_file_changed(test_file)

def test_cache_invalid_file(cache_manager):
    """Test handling of non-existent files"""
    non_existent = Path("non_existent.py")
    assert cache_manager.get_cached_summary(non_existent) is None
    assert cache_manager.is_file_changed(non_existent) is True

def test_sqlite_cache_operations(tmp_path):
    """Test SQLite cache backend operations"""
    # Create a temporary database in the pytest temporary directory
    db_path = tmp_path / "test_cache.db"
    
    # Create a SQLite cache
    sqlite_cache = SQLiteCache(db_path)
    
    # Create a test entry
    entry = CacheEntry(
        key="test_key",
        value="test_value",
        hash="test_hash",
        timestamp=time.time(),
        metadata={"test": "metadata"}
    )
    
    # Set the entry
    sqlite_cache.set("test_key", entry)
    
    # Get the entry
    retrieved = sqlite_cache.get("test_key")
    assert retrieved is not None
    assert retrieved.key == "test_key"
    assert retrieved.value == "test_value"
    assert retrieved.hash == "test_hash"
    assert retrieved.metadata == {"test": "metadata"}
    
    # Clear the cache
    sqlite_cache.clear()
    assert sqlite_cache.get("test_key") is None
    
    # No need to clean up - pytest will handle the temporary directory

def test_memory_cache_operations():
    """Test memory cache backend operations"""
    # Create a memory cache
    memory_cache = MemoryCache()
    
    # Create a test entry
    entry = CacheEntry(
        key="test_key",
        value="test_value",
        hash="test_hash",
        timestamp=time.time(),
        metadata={"test": "metadata"}
    )
    
    # Set the entry
    memory_cache.set("test_key", entry)
    
    # Get the entry
    retrieved = memory_cache.get("test_key")
    assert retrieved is not None
    assert retrieved.key == "test_key"
    assert retrieved.value == "test_value"
    assert retrieved.hash == "test_hash"
    assert retrieved.metadata == {"test": "metadata"}
    
    # Clear the cache
    memory_cache.clear()
    assert memory_cache.get("test_key") is None
def test_calculate_file_hash(cache_manager, test_file):
    """Test file hash calculation with different algorithms"""
    # Set the hash algorithm to md5 explicitly
    cache_manager.hash_algorithm = 'md5'
    assert cache_manager.hash_algorithm == 'md5'
    
    # Calculate hashes directly with hashlib for comparison
    with open(test_file, 'rb') as f:
        content = f.read()
    expected_md5 = hashlib.md5(content).hexdigest()
    expected_sha1 = hashlib.sha1(content).hexdigest()
    expected_sha256 = hashlib.sha256(content).hexdigest()
    
    # Test with default algorithm (md5)
    hash1 = cache_manager._calculate_file_hash(test_file)
    print(f"MD5 hash: {hash1}, length: {len(hash1)}")
    print(f"Expected MD5: {expected_md5}, length: {len(expected_md5)}")
    
    # Test with sha1 algorithm
    cache_manager.hash_algorithm = 'sha1'
    hash2 = cache_manager._calculate_file_hash(test_file)
    print(f"SHA1 hash: {hash2}, length: {len(hash2)}")
    print(f"Expected SHA1: {expected_sha1}, length: {len(expected_sha1)}")
    
    # Test with sha256 algorithm
    cache_manager.hash_algorithm = 'sha256'
    hash3 = cache_manager._calculate_file_hash(test_file)
    print(f"SHA256 hash: {hash3}, length: {len(hash3)}")
    print(f"Expected SHA256: {expected_sha256}, length: {len(expected_sha256)}")
    
    # Compare with expected hashes
    assert hash1 == expected_md5
    assert hash2 == expected_sha1
    assert hash3 == expected_sha256
    
    # Verify correct hash lengths
    assert len(hash1) == 32  # MD5
    assert len(hash2) == 40  # SHA1
    assert len(hash3) == 64  # SHA256

def test_get_repo_cache_dir(cache_manager, tmp_path):
    """Test getting repository cache directory"""
    # Test with repo_identifier
    cache_manager.repo_identifier = "test-repo"
    cache_dir1 = cache_manager.get_repo_cache_dir()
    assert "test-repo" in str(cache_dir1)
    
    # Test with explicit repo_path
    cache_dir2 = cache_manager.get_repo_cache_dir(tmp_path)
    assert tmp_path.name in str(cache_dir2)

def test_clear_repo_cache(cache_manager, test_file):
    """Test clearing the repository cache"""
    # Save something to cache
    cache_manager.save_summary(test_file, "test summary")
    assert cache_manager.get_cached_summary(test_file) == "test summary"
    
    # Clear cache
    cache_manager.clear_repo_cache()
    
    # Verify cache is cleared
    assert cache_manager.get_cached_summary(test_file) is None

def test_clear_all_caches(tmp_path):
    """Test clearing all caches"""
    # Create cache files
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "test.db").touch()
    (cache_dir / "test.cache").touch()
    
    # Clear all caches
    CacheManager.clear_all_caches(repo_path=tmp_path)
    
    # Verify files are removed
    assert not (cache_dir / "test.db").exists()
    assert not (cache_dir / "test.cache").exists()