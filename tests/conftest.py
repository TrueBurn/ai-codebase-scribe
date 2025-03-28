import os
import sys
import pytest
from pathlib import Path

def pytest_configure(config):
    """Create cache directory with proper permissions before tests run."""
    cache_dir = Path(".pytest_cache/v/cache")
    
    try:
        # Create parent directories first
        cache_dir.parent.parent.mkdir(exist_ok=True)  # .pytest_cache
        cache_dir.parent.mkdir(exist_ok=True)         # .pytest_cache/v
        cache_dir.mkdir(exist_ok=True)                # .pytest_cache/v/cache

        if sys.platform == 'win32':
            import win32security
            import ntsecuritycon as con
            
            # Get the current user's SID
            username = os.getlogin()
            user_sid, _, _ = win32security.LookupAccountName(None, username)
            
            # Create a new DACL with full control for the current user
            dacl = win32security.ACL()
            dacl.AddAccessAllowedAce(
                win32security.ACL_REVISION,
                con.FILE_ALL_ACCESS,
                user_sid
            )
            
            # Apply the security settings to each directory
            for dir_path in [cache_dir, cache_dir.parent, cache_dir.parent.parent]:
                security_desc = win32security.SECURITY_DESCRIPTOR()
                security_desc.SetSecurityDescriptorDacl(1, dacl, 0)
                win32security.SetFileSecurity(
                    str(dir_path),
                    win32security.DACL_SECURITY_INFORMATION,
                    security_desc
                )
        else:
            # Unix-like systems
            os.chmod(str(cache_dir.parent.parent), 0o777)
            os.chmod(str(cache_dir.parent), 0o777)
            os.chmod(str(cache_dir), 0o777)
            
    except Exception as e:
        print(f"Warning: Could not set cache directory permissions: {e}")
        print("Tests will continue but caching may not work properly.")

@pytest.fixture
def test_repo():
    """Fixture providing path to test repository"""
    return Path(__file__).parent / 'fixtures' / 'test_repo'

@pytest.fixture
def config():
    """Fixture providing test configuration"""
    from src.utils.config_class import ScribeConfig, OllamaConfig, CacheConfig
    
    config = ScribeConfig()
    config.ollama = OllamaConfig(
        model='codellama',
        base_url='http://localhost:11434',
        max_tokens=4096,
        retries=3,
        retry_delay=1.0,
        timeout=30
    )
    config.cache = CacheConfig(
        ttl=3600,
        max_size=104857600
    )
    return config

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Setup test environment with proper permissions"""
    cache_dir = tmp_path / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure write permissions
    os.chmod(str(tmp_path), 0o777)
    os.chmod(str(cache_dir), 0o777)
    
    return cache_dir 