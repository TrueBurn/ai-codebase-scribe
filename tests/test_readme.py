import pytest
import sys
import os
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force reload the module to ensure we get the latest version
import src.generators.readme
importlib.reload(src.generators.readme)

from src.generators.readme import (
    generate_readme, should_enhance_existing_readme, enhance_existing_readme,
    generate_new_readme, generate_overview, generate_overview_with_fallbacks,
    extract_overview_from_architecture, generate_section, add_architecture_link_if_needed,
    ensure_correct_title, validate_and_improve_content, log_validation_issues,
    check_readability, generate_fallback_readme, _clean_section_headers,
    _format_anchor_link, CONTENT_THRESHOLDS, INSTRUCTION_PHRASES
)

class TestReadmeGenerator:
    """Test suite for the README generator module."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = AsyncMock()
        mock_client.generate_project_overview = AsyncMock(return_value="This is a test project overview.")
        mock_client.enhance_documentation = AsyncMock(return_value="# Enhanced Content\n\nThis is enhanced content.")
        mock_client.generate_usage_guide = AsyncMock(return_value="This is a usage guide.")
        mock_client.generate_contributing_guide = AsyncMock(return_value="This is a contributing guide.")
        mock_client.generate_license_info = AsyncMock(return_value="This is license information.")
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
    def mock_scorer(self):
        """Create a mock ReadabilityScorer."""
        mock_scorer = MagicMock()
        mock_scorer.analyze_text.return_value = {"overall": 30}
        return mock_scorer
    
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
            "README.md": MagicMock()
        }
    
    def test_clean_section_headers(self):
        """Test the _clean_section_headers function."""
        # Test with headers at the beginning
        content = "# Header\n## Subheader\nContent text"
        result = _clean_section_headers(content)
        assert result == "Content text"
        
        # Test with no headers
        content = "Content with no headers"
        result = _clean_section_headers(content)
        assert result == "Content with no headers"
        
        # Test with empty content
        result = _clean_section_headers("")
        assert result == ""
        
        # Test with None
        result = _clean_section_headers(None)
        assert result == ""
    
    def test_format_anchor_link(self):
        """Test the _format_anchor_link function."""
        # Test basic formatting
        assert _format_anchor_link("Section Title") == "section-title"
        
        # Test with special characters
        assert _format_anchor_link("Section.With/Special(Chars)") == "sectionwithspecialchars"
        
        # Test with mixed case
        assert _format_anchor_link("MixedCase") == "mixedcase"
    
    def test_should_enhance_existing_readme(self, tmp_path, test_config):
        """Test the should_enhance_existing_readme function."""
        # Create a temporary README file
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        readme_path = repo_path / "README.md"
        
        # Test with no README file
        assert should_enhance_existing_readme(repo_path, test_config) is False
        
        # Test with short README file (less than threshold)
        readme_path.write_text("# Short README")
        assert should_enhance_existing_readme(repo_path, test_config) is False
        
        # Test with meaningful README file
        meaningful_content = "\n".join([f"Line {i}" for i in range(CONTENT_THRESHOLDS['meaningful_readme_lines'] + 5)])
        readme_path.write_text(meaningful_content)
        assert should_enhance_existing_readme(repo_path, test_config) is True
        
        # Test with preserve_existing=False
        from src.utils.config_class import ScribeConfig
        config_no_preserve = ScribeConfig()
        config_no_preserve.preserve_existing = False
        assert should_enhance_existing_readme(repo_path, config_no_preserve) is False
    
    @pytest.mark.asyncio
    @patch('src.generators.readme.generate_overview')
    async def test_enhance_existing_readme(self, mock_generate_overview, tmp_path, mock_llm_client):
        """Test the enhance_existing_readme function."""
        # Setup mock for generate_overview
        mock_generate_overview.return_value = "This is a test project overview."
        
        # Create a temporary README file
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        readme_path = repo_path / "README.md"
        readme_content = "# Existing README\n\nThis is an existing README file."
        readme_path.write_text(readme_content)
        
        # Test enhancing existing README
        enhanced_content = "# Enhanced Content\n\nThis is enhanced content."
        mock_llm_client.enhance_documentation.return_value = enhanced_content
        
        result = await enhance_existing_readme(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            project_name="Test Project",
            architecture_file_exists=False
        )
        
        # Verify the LLM client was called correctly
        mock_llm_client.enhance_documentation.assert_called_once()
        assert mock_llm_client.enhance_documentation.call_args[1]["existing_content"] == readme_content
        
        # Verify the result
        assert result is not None
        assert "Enhanced Content" in result
        
        # Test with instruction phrases
        mock_llm_client.enhance_documentation.return_value = "# Enhanced Content\n\nYour task is to preserve this content."
        result = await enhance_existing_readme(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            project_name="Test Project",
            architecture_file_exists=False
        )
        
        # Verify instruction phrases were removed
        assert "Your task is to preserve" not in result
    
    @pytest.mark.asyncio
    async def test_generate_overview(self, mock_llm_client):
        """Test the generate_overview function."""
        # Test successful generation
        result = await generate_overview(mock_llm_client, {}, "Test Project")
        assert result == "This is a test project overview."
        
        # Test with LLM error
        mock_llm_client.generate_project_overview.side_effect = Exception("LLM error")
        result = await generate_overview(mock_llm_client, {}, "Test Project")
        assert "Test Project is a software project" in result
    
    @pytest.mark.asyncio
    async def test_generate_section(self, mock_llm_client):
        """Test the generate_section function."""
        # Setup mock for generate_usage_guide
        mock_llm_client.generate_usage_guide.return_value = "This is a usage guide."
        
        # Test successful generation
        result = await generate_section(
            mock_llm_client, {}, "usage_guide", 
            10,  # Set min_length to 10 to ensure it passes
            "Fallback text"
        )
        
        # Verify result
        assert result == "This is a usage guide."
        
        # Test with short content
        mock_llm_client.generate_usage_guide.return_value = "Short"
        result = await generate_section(
            mock_llm_client, {}, "usage_guide", 
            20,  # Set min_length to 20 to ensure it fails
            "Fallback text"
        )
        assert result == "Fallback text"
        
        # Test with LLM error
        mock_llm_client.generate_usage_guide.side_effect = Exception("LLM error")
        result = await generate_section(
            mock_llm_client, {}, "usage_guide", 
            10,
            "Fallback text"
        )
        assert result == "Fallback text"
    
    def test_add_architecture_link_if_needed(self):
        """Test the add_architecture_link_if_needed function."""
        # Test with no architecture file
        content = "# README\n\n## Usage\n\nUsage instructions."
        result = add_architecture_link_if_needed(content, False)
        assert result == content
        
        # Test with architecture file but already mentioned
        content = "# README\n\nSee [ARCHITECTURE.md](docs/ARCHITECTURE.md)\n\n## Usage\n\nUsage instructions."
        result = add_architecture_link_if_needed(content, True)
        assert result == content
        
        # Test with architecture file and not mentioned, with Usage section
        content = "# README\n\n## Usage\n\nUsage instructions."
        result = add_architecture_link_if_needed(content, True)
        assert "## Architecture" in result
        assert "ARCHITECTURE.md" in result
        assert result.index("## Architecture") < result.index("## Usage")
        
        # Test with architecture file and not mentioned, without Usage section
        content = "# README\n\nSome content."
        result = add_architecture_link_if_needed(content, True)
        assert "## Architecture" in result
        assert "ARCHITECTURE.md" in result
    
    @patch('src.generators.readme.re.search')
    def test_ensure_correct_title(self, mock_search):
        """Test the ensure_correct_title function."""
        # Test with correct title
        mock_search.return_value = MagicMock()
        mock_search.return_value.group.return_value = "Test Project"
        
        content = "# Test Project\n\nContent."
        result = ensure_correct_title(content, "Test Project")
        assert result == content
        
        # Test with generic title
        mock_search.return_value.group.return_value = "Project"
        content = "# Project\n\nContent."
        result = ensure_correct_title(content, "Test Project")
        assert "# Test Project" in result
        
        # Test with empty title
        mock_search.return_value.group.return_value = ""
        content = "# \n\nContent."
        result = ensure_correct_title(content, "Test Project")
        assert "# Test Project" in result
        
        # Test with no title
        mock_search.return_value = None
        content = "Content without title."
        result = ensure_correct_title(content, "Test Project")
        assert result == "# Test Project\n\nContent without title."
    
    @pytest.mark.asyncio
    @patch('src.generators.readme.MarkdownValidator')
    @patch('src.generators.readme.check_readability')
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
        # We don't check the exact number of calls because the implementation may create
        # multiple validators for different purposes
        MockValidator.assert_any_call(content)
        mock_validator.validate_with_link_checking.assert_called_once_with(repo_path)
        mock_validator.fix_common_issues.assert_called_once()
        mock_check_readability.assert_called_once()
        
        # Verify result
        assert result == "# Improved Content\n\nThis is improved content."
    
    def test_generate_fallback_readme(self):
        """Test the generate_fallback_readme function."""
        # Test with architecture file
        result = generate_fallback_readme(Path("test_repo"), True)
        assert "# test_repo" in result
        assert "README for the test_repo project" in result
        assert "ARCHITECTURE.md" in result
        assert "encountered errors" in result
        
        # Test without architecture file
        result = generate_fallback_readme(Path("test_repo"), False)
        assert "# test_repo" in result
        assert "README for the test_repo project" in result
        assert "ARCHITECTURE.md" in result  # Still includes the link
        assert "encountered errors" in result
    
    @pytest.mark.asyncio
    @patch('src.generators.readme.validate_and_improve_content')
    @patch('src.generators.readme.generate_overview_with_fallbacks')
    @patch('src.generators.readme.generate_section')
    async def test_generate_new_readme(self, mock_generate_section, mock_generate_overview, mock_validate, 
                                      mock_llm_client, mock_analyzer, test_config, tmp_path):
        """Test the generate_new_readme function."""
        # Setup mocks
        mock_generate_overview.return_value = "This is a test project overview."
        mock_generate_section.side_effect = [
            "This is a usage guide.",
            "This is a contributing guide.",
            "This is license information."
        ]
        mock_validate.return_value = "# Validated Content\n\nThis is validated content."
        
        # Test generating new README
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        result = await generate_new_readme(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            project_name="Test Project",
            architecture_file_exists=True,
            config=test_config
        )
        
        # Verify mocks were called correctly
        mock_generate_overview.assert_called_once()
        assert mock_generate_section.call_count == 3
        mock_validate.assert_called_once()
        
        # Verify result
        assert result == "# Validated Content\n\nThis is validated content."
    
    @pytest.mark.asyncio
    @patch('src.generators.readme.should_enhance_existing_readme')
    @patch('src.generators.readme.enhance_existing_readme')
    @patch('src.generators.readme.generate_new_readme')
    async def test_generate_readme(self, mock_generate_new, mock_enhance_existing, mock_should_enhance,
                                  mock_llm_client, mock_analyzer, test_config, tmp_path):
        """Test the main generate_readme function."""
        # Setup mocks
        mock_should_enhance.return_value = False
        mock_generate_new.return_value = "# New README\n\nThis is a new README."
        mock_enhance_existing.return_value = "# Enhanced README\n\nThis is an enhanced README."
        
        # Test generating new README
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        
        result = await generate_readme(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            file_summaries={},
            config=test_config,
            analyzer=mock_analyzer,
            output_dir="output",
            architecture_file_exists=True
        )
        
        # Verify analyzer was called to get project name
        mock_analyzer.derive_project_name.assert_called_once()
        
        # Verify should_enhance_existing_readme was called
        mock_should_enhance.assert_called_once()
        
        # Verify generate_new_readme was called (not enhance_existing)
        mock_generate_new.assert_called_once()
        mock_enhance_existing.assert_not_called()
        
        # Verify result
        assert result == "# New README\n\nThis is a new README."
        
        # Test enhancing existing README
        mock_should_enhance.return_value = True
        mock_enhance_existing.return_value = "# Enhanced README\n\nThis is an enhanced README."
        
        result = await generate_readme(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            file_summaries={},
            config=test_config,
            analyzer=mock_analyzer,
            output_dir="output",
            architecture_file_exists=True
        )
        
        # Verify enhance_existing_readme was called
        mock_enhance_existing.assert_called_once()
        
        # Verify result
        assert result == "# Enhanced README\n\nThis is an enhanced README."
        
        # Test with error
        mock_should_enhance.side_effect = Exception("Test error")
        
        result = await generate_readme(
            repo_path=repo_path,
            llm_client=mock_llm_client,
            file_manifest={},
            file_summaries={},
            config=test_config,
            analyzer=mock_analyzer,
            output_dir="output",
            architecture_file_exists=True
        )
        
        # Verify fallback README was generated
        assert "README for the" in result
        assert "encountered errors" in result