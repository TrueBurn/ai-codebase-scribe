import pytest
import sys
import os
import asyncio
from pathlib import Path
import httpx
from unittest.mock import patch, MagicMock
import importlib

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force reload the module to ensure we get the latest version
import src.utils.link_validator
importlib.reload(src.utils.link_validator)

from src.utils.link_validator import LinkValidator, LinkIssue

class TestLinkValidator:
    """Test suite for the LinkValidator class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.repo_path = Path(os.path.dirname(__file__)).parent
        self.test_dir = self.repo_path / "tests" / "fixtures" / "link_validator"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test markdown files
        self.test_md = self.test_dir / "test.md"
        self.test_md.write_text("""# Test Document
        
## Section 1
        
This is a [valid link](https://example.com) and an [invalid link](https://nonexistent.example.com).
        
This is a [relative link](../README.md) and a [nonexistent relative link](./nonexistent.md).
        
This is an [anchor link](#section-1) and an [invalid anchor link](#nonexistent-section).
        """)
        
        # Create a README.md file for testing relative links
        self.readme_md = self.repo_path / "README.md"
        if not self.readme_md.exists():
            self.readme_md.write_text("# Test README\n\nThis is a test README file.")
    
    def teardown_method(self):
        """Clean up after each test."""
        # Remove test files
        if self.test_md.exists():
            self.test_md.unlink()
        
        # Remove test directory if empty
        if self.test_dir.exists() and not any(self.test_dir.iterdir()):
            self.test_dir.rmdir()
    
    @pytest.mark.asyncio
    async def test_validate_document(self):
        """Test validation of a document with various links."""
        validator = LinkValidator(self.repo_path)
        content = self.test_md.read_text()
        issues = await validator.validate_document(content, self.test_dir)
        
        # We should have at least 1 issue (nonexistent file)
        assert len(issues) >= 1
        
        # Check for specific issues
        assert any(issue.message == "Referenced file does not exist" for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_internal_link_valid(self):
        """Test validation of a valid internal link."""
        validator = LinkValidator(self.repo_path)
        validator.issues = []
        
        # Test a valid relative link to the test file itself
        validator._validate_internal_link("test.md", 1, self.test_dir)
        assert len(validator.issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_internal_link_invalid(self):
        """Test validation of an invalid internal link."""
        validator = LinkValidator(self.repo_path)
        validator.issues = []
        
        # Test an invalid relative link
        validator._validate_internal_link("./nonexistent.md", 1, self.test_dir)
        assert len(validator.issues) == 1
        assert validator.issues[0].message == "Referenced file does not exist"
    
    @pytest.mark.asyncio
    async def test_validate_anchor_link_valid(self):
        """Test validation of a valid anchor link."""
        validator = LinkValidator(self.repo_path)
        validator.issues = []
        
        # Process the document to build the anchor map
        content = self.test_md.read_text()
        await validator.validate_document(content, self.test_dir)
        validator.issues = []  # Clear issues from validation
        
        # Test a valid anchor link
        validator._validate_internal_link("#section-1", 1, self.test_dir)
        assert len(validator.issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_anchor_link_invalid(self):
        """Test validation of an invalid anchor link."""
        validator = LinkValidator(self.repo_path)
        validator.issues = []
        
        # Process the document to build the anchor map
        content = self.test_md.read_text()
        await validator.validate_document(content, self.test_dir)
        validator.issues = []  # Clear issues from validation
        
        # Test an invalid anchor link
        validator._validate_internal_link("#nonexistent-section", 1, self.test_dir)
        assert len(validator.issues) == 1
        assert "not found in document" in validator.issues[0].message
    
    @pytest.mark.asyncio
    async def test_validate_external_link_valid(self):
        """Test validation of a valid external link with mocked response."""
        validator = LinkValidator(self.repo_path)
        validator.issues = []
        
        # Mock httpx.AsyncClient to avoid actual HTTP requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value.head.return_value = mock_response
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            await validator._validate_external_link("https://example.com", 1)
        
        assert len(validator.issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_external_link_broken(self):
        """Test validation of a broken external link with mocked response."""
        validator = LinkValidator(self.repo_path)
        validator.issues = []
        
        # Mock httpx.AsyncClient to avoid actual HTTP requests
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value.head.return_value = mock_response
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            await validator._validate_external_link("https://nonexistent.example.com", 1)
        
        assert len(validator.issues) == 1
        assert "Broken link" in validator.issues[0].message
        assert validator.issues[0].severity == "error"
    
    @pytest.mark.asyncio
    async def test_validate_external_link_timeout(self):
        """Test validation of an external link that times out."""
        validator = LinkValidator(self.repo_path)
        validator.max_retries = 1  # Set max_retries directly
        validator.issues = []
        
        # Mock httpx.AsyncClient to simulate a timeout
        mock_client = MagicMock()
        mock_client.__aenter__.return_value.head.side_effect = httpx.TimeoutException("Timeout")
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            with patch('asyncio.sleep', return_value=None):  # Mock sleep to speed up test
                await validator._validate_external_link("https://slow.example.com", 1)
        
        assert len(validator.issues) == 1
        assert "timeout" in validator.issues[0].message.lower()
        assert validator.issues[0].severity == "warning"
    
    @pytest.mark.asyncio
    async def test_validate_external_link_retry_success(self):
        """Test successful retry of an external link."""
        validator = LinkValidator(self.repo_path)
        validator.max_retries = 2  # Set max_retries directly
        validator.issues = []
        
        # Mock httpx.AsyncClient to simulate a timeout then success
        mock_client = MagicMock()
        
        # Create a mock response with status_code 200
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # First call raises a timeout exception, second call succeeds
        mock_client.__aenter__.return_value.head = MagicMock(side_effect=[
            httpx.TimeoutException("Timeout"),
            mock_response
        ])
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            with patch('asyncio.sleep', return_value=None):  # Mock sleep to speed up test
                await validator._validate_external_link("https://example.com", 1)
        
        # Verify that the head method was called multiple times (initial + retries)
        assert mock_client.__aenter__.return_value.head.call_count >= 2
        
        # Verify that no issues were added for a successful retry
        assert all("timeout" not in issue.message.lower() for issue in validator.issues)
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test that external link validation results are cached."""
        validator = LinkValidator(self.repo_path)
        
        # Mock httpx.AsyncClient to avoid actual HTTP requests
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value.head.return_value = mock_response
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            # First validation should make an HTTP request
            await validator._validate_link("https://example.com", 1, self.test_dir)
            
            # Reset the mock to verify it's not called again
            mock_client.reset_mock()
            
            # Second validation should use the cache
            await validator._validate_link("https://example.com", 2, self.test_dir)
            
            # Verify that head() was not called again
            assert not mock_client.__aenter__.return_value.head.called