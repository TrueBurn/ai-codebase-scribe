import pytest
import sys
import os
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force reload the module to ensure we get the latest version
import src.generators.contributing
importlib.reload(src.generators.contributing)

from src.generators.contributing import (
    generate_contributing, should_enhance_existing_contributing, enhance_existing_contributing,
    generate_new_contributing, generate_contributing_content, ensure_correct_title,
    validate_and_improve_content, check_readability, generate_fallback_contributing,
    CONTENT_THRESHOLDS
)

class TestContributingGenerator:
    """Test suite for the Contributing guide generator module."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = AsyncMock()
        mock_client.generate_contributing_guide = AsyncMock(return_value="This is a contributing guide.")
        mock_client.enhance_documentation = AsyncMock(return_value="# Enhanced Content\n\nThis is enhanced content.")
        return mock_client
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock CodebaseAnalyzer."""
        mock_analyzer = MagicMock()
        mock_analyzer.derive_project_name.return_value = "Test Project"
        return mock_analyzer
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock MarkdownValidator."""
        mock_validator = MagicMock()
        mock_validator.validate.return_value = []
        mock_validator.validate_with_link_checking = AsyncMock(return_value=[])
        mock_validator.fix_common_issues.return_value = "# Fixed Content\n\nThis is fixed content."
        return mock_validator
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        from src.utils.config_class import ScribeConfig
        
        config = ScribeConfig()
        config.debug = False
        config.preserve_existing = True
        return config
    
    @pytest.fixture
    def test_file_manifest(self):
        """Create a test file manifest."""
        return {
            "file1.py": MagicMock(),
            "file2.py": MagicMock(),
            "CONTRIBUTING.md": MagicMock()
        }
    
    def test_should_enhance_existing_contributing(self, tmp_path, test_config):
        """Test the should_enhance_existing_contributing function."""
        # Create a temporary repository path
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        contributing_path = repo_path / "CONTRIBUTING.md"
        
        # Test with no CONTRIBUTING file
        assert should_enhance_existing_contributing(repo_path, test_config) is False
        
        # Test with short CONTRIBUTING file (less than threshold)
        contributing_path.write_text("# Short CONTRIBUTING")
        assert should_enhance_existing_contributing(repo_path, test_config) is False
        
        # Test with meaningful CONTRIBUTING file
        meaningful_content = "\n".join([f"Line {i}" for i in range(10)])  # More than 5 lines
        contributing_path.write_text(meaningful_content)
        assert should_enhance_existing_contributing(repo_path, test_config) is True
        
        # Test with preserve_existing=False
        from src.utils.config_class import ScribeConfig
        config_no_preserve = ScribeConfig()
        config_no_preserve.preserve_existing = False
        assert should_enhance_existing_contributing(repo_path, config_no_preserve) is False
    
    @pytest.mark.asyncio
    async def test_enhance_existing_contributing(self, tmp_path, mock_llm_client):
        """Test the enhance_existing_contributing function."""
        # Create a temporary CONTRIBUTING file
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        contributing_path = repo_path / "CONTRIBUTING.md"
        contributing_content = "# Existing CONTRIBUTING\n\nThis is an existing CONTRIBUTING file."
        contributing_path.write_text(contributing_content)
        
        # Test enhancing existing CONTRIBUTING
        enhanced_content = "# Enhanced Content\n\nThis is enhanced content."
        mock_llm_client.enhance_documentation.return_value = enhanced_content
        
        result = await enhance_existing_contributing(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            project_name="Test Project"
        )
        
        # Verify the LLM client was called correctly
        mock_llm_client.enhance_documentation.assert_called_once()
        assert mock_llm_client.enhance_documentation.call_args[1]["existing_content"] == contributing_content
        
        # Verify the result
        assert result is not None
        # The ensure_correct_title function changes the title to "Contributing to Test Project"
        assert "Contributing to Test Project" in result
        assert "This is enhanced content." in result
    
    @pytest.mark.asyncio
    async def test_generate_contributing_content(self, mock_llm_client):
        """Test the generate_contributing_content function."""
        # Test successful generation
        result = await generate_contributing_content(
            mock_llm_client, {}, 
            10,  # Set min_length to 10 to ensure it passes
            "Fallback text"
        )
        
        # Verify result
        assert result == "This is a contributing guide."
        
        # Test with short content
        mock_llm_client.generate_contributing_guide.return_value = "Short"
        result = await generate_contributing_content(
            mock_llm_client, {}, 
            20,  # Set min_length to 20 to ensure it fails
            "Fallback text"
        )
        assert result == "Fallback text"
        
        # Test with LLM error
        mock_llm_client.generate_contributing_guide.side_effect = Exception("LLM error")
        result = await generate_contributing_content(
            mock_llm_client, {}, 
            10,
            "Fallback text"
        )
        assert result == "Fallback text"
    
    @patch('src.generators.contributing.re.search')
    def test_ensure_correct_title(self, mock_search):
        """Test the ensure_correct_title function."""
        # Test with correct title
        mock_search.return_value = MagicMock()
        mock_search.return_value.group.return_value = "Contributing to Test Project"
        
        content = "# Contributing to Test Project\n\nContent."
        result = ensure_correct_title(content, "Test Project")
        assert result == content
        
        # Test with generic title
        mock_search.return_value.group.return_value = "Contributing"
        content = "# Contributing\n\nContent."
        # The function only replaces the title if "contributing" is not in the title (case insensitive)
        # or if "project" is in the title
        result = ensure_correct_title(content, "Test Project")
        # Since "contributing" is in the title and "project" is not, the title should remain unchanged
        assert result == content
        
        # Test with title containing "project"
        mock_search.return_value.group.return_value = "Contributing to Project"
        content = "# Contributing to Project\n\nContent."
        result = ensure_correct_title(content, "Test Project")
        assert "# Contributing to Test Project" in result
        
        # Test with no title
        mock_search.return_value = None
        content = "Content without title."
        result = ensure_correct_title(content, "Test Project")
        assert result == "# Contributing to Test Project\n\nContent without title."
    
    @pytest.mark.asyncio
    @patch('src.generators.contributing.MarkdownValidator')
    @patch('src.generators.contributing.check_readability')
    async def test_validate_and_improve_content(self, mock_check_readability, MockValidator, tmp_path):
        """Test the validate_and_improve_content function."""
        # Setup mock validator
        mock_validator = MagicMock()
        mock_validator.validate_with_link_checking = AsyncMock(return_value=[])
        mock_validator.fix_common_issues.return_value = "# Improved Content\n\nThis is improved content."
        MockValidator.return_value = mock_validator
        
        # Test validation and improvement
        content = "# Test Content\n\nThis is test content."
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        result = await validate_and_improve_content(content, repo_path)
        
        # Verify validator was called correctly
        MockValidator.assert_called_once_with(content)
        mock_validator.validate_with_link_checking.assert_called_once_with(repo_path)
        mock_validator.fix_common_issues.assert_called_once()
        mock_check_readability.assert_called_once()
        
        # Verify result
        assert result == "# Improved Content\n\nThis is improved content."
    
    def test_generate_fallback_contributing(self):
        """Test the generate_fallback_contributing function."""
        result = generate_fallback_contributing(Path("test_repo"))
        assert "# Contributing to test_repo" in result
        assert "Thank you for your interest in contributing" in result
        assert "Code of Conduct" in result
        assert "Pull Request Process" in result
        assert "automatically generated" in result
    
    @pytest.mark.asyncio
    @patch('src.generators.contributing.validate_and_improve_content')
    @patch('src.generators.contributing.generate_contributing_content')
    async def test_generate_new_contributing(self, mock_generate_content, mock_validate, 
                                           mock_llm_client, mock_analyzer, test_config, tmp_path):
        """Test the generate_new_contributing function."""
        # Setup mocks
        mock_generate_content.return_value = "This is a contributing guide."
        mock_validate.return_value = "# Validated Content\n\nThis is validated content."
        
        # Test generating new CONTRIBUTING
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        result = await generate_new_contributing(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            project_name="Test Project",
            config=test_config
        )
        
        # Verify mocks were called correctly
        mock_generate_content.assert_called_once()
        mock_validate.assert_called_once()
        
        # Verify result
        assert result == "# Validated Content\n\nThis is validated content."
    
    @pytest.mark.asyncio
    @patch('src.generators.contributing.should_enhance_existing_contributing')
    @patch('src.generators.contributing.enhance_existing_contributing')
    @patch('src.generators.contributing.generate_new_contributing')
    async def test_generate_contributing(self, mock_generate_new, mock_enhance_existing, mock_should_enhance,
                                       mock_llm_client, mock_analyzer, test_config, tmp_path):
        """Test the main generate_contributing function."""
        # Setup mocks
        mock_should_enhance.return_value = False
        mock_generate_new.return_value = "# New CONTRIBUTING\n\nThis is a new CONTRIBUTING guide."
        mock_enhance_existing.return_value = "# Enhanced CONTRIBUTING\n\nThis is an enhanced CONTRIBUTING guide."
        
        # Test generating new CONTRIBUTING
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        result = await generate_contributing(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            config=test_config,
            analyzer=mock_analyzer
        )
        
        # Verify analyzer was called to get project name
        mock_analyzer.derive_project_name.assert_called_once()
        
        # Verify should_enhance_existing_contributing was called
        mock_should_enhance.assert_called_once()
        
        # Verify generate_new_contributing was called (not enhance_existing)
        mock_generate_new.assert_called_once()
        mock_enhance_existing.assert_not_called()
        
        # Verify result
        assert result == "# New CONTRIBUTING\n\nThis is a new CONTRIBUTING guide."
        
        # Test enhancing existing CONTRIBUTING
        mock_should_enhance.return_value = True
        mock_enhance_existing.return_value = "# Enhanced CONTRIBUTING\n\nThis is an enhanced CONTRIBUTING guide."
        
        result = await generate_contributing(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            config=test_config,
            analyzer=mock_analyzer
        )
        
        # Verify enhance_existing_contributing was called
        mock_enhance_existing.assert_called_once()
        
        # Verify result
        assert result == "# Enhanced CONTRIBUTING\n\nThis is an enhanced CONTRIBUTING guide."
        
        # Test with error
        mock_should_enhance.side_effect = Exception("Test error")
        
        result = await generate_contributing(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            config=test_config,
            analyzer=mock_analyzer
        )
        
        # Verify fallback CONTRIBUTING was generated
        assert "# Contributing to" in result
        assert "Thank you for your interest in contributing" in result