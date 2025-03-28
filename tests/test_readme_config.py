#!/usr/bin/env python3

"""
Tests for README generator with ScribeConfig
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.generators.readme import (
    generate_readme,
    should_enhance_existing_readme,
    generate_new_readme
)
from src.utils.config_class import ScribeConfig


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration object."""
    from src.utils.config_class import ScribeConfig, BlacklistConfig
    
    config = ScribeConfig()
    config.debug = True
    config.preserve_existing = True
    config.test_mode = True
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
    config.preserve_existing = True
    config.test_mode = True
    config.blacklist = BlacklistConfig(
        extensions=['.log', '.tmp'],
        path_patterns=['/node_modules/', '/__pycache__/']
    )
    return config
    return ScribeConfig.from_dict(sample_config_dict)


@pytest.fixture
def temp_repo_path():
    """Create a temporary directory for the repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Create some test files
        (repo_path / 'file1.py').write_text('print("Hello")')
        (repo_path / 'file2.js').write_text('console.log("Hello")')
        
        yield repo_path


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock_client = AsyncMock()
    mock_client.generate_project_overview = AsyncMock(return_value="Project overview")
    mock_client.generate_usage_guide = AsyncMock(return_value="Usage guide")
    mock_client.generate_contributing_guide = AsyncMock(return_value="Contributing guide")
    return mock_client


@pytest.fixture
def mock_analyzer():
    """Create a mock CodebaseAnalyzer."""
    mock = MagicMock()
    mock.derive_project_name = MagicMock(return_value="Test Project")
    mock.cache = MagicMock()
    return mock


@pytest.fixture
def file_manifest():
    """Create a sample file manifest."""
    return {
        'file1.py': MagicMock(summary="Python file"),
        'file2.js': MagicMock(summary="JavaScript file")
    }


class TestReadmeGenerator:
    """Test suite for README generator with ScribeConfig."""

    @pytest.mark.asyncio
    @patch('src.generators.readme.generate_new_readme')
    async def test_generate_readme_with_dict(self, mock_generate_new, temp_repo_path, mock_llm_client, mock_analyzer, file_manifest, sample_config_dict):
        """Test generating README with a dictionary config."""
        mock_generate_new.return_value = "Generated README"
        
        result = await generate_readme(
            repo_path=temp_repo_path,
            llm_client=mock_llm_client,
            file_manifest=file_manifest,
            file_summaries=file_manifest,
            config=sample_config_dict,
            analyzer=mock_analyzer,
            output_dir=str(temp_repo_path),
            architecture_file_exists=False
        )
        
        assert result == "Generated README"
        mock_analyzer.derive_project_name.assert_called_once_with(True)  # debug=True

    @pytest.mark.asyncio
    @patch('src.generators.readme.generate_new_readme')
    async def test_generate_readme_with_scribe_config(self, mock_generate_new, temp_repo_path, mock_llm_client, mock_analyzer, file_manifest, sample_config):
        """Test generating README with a ScribeConfig instance."""
        mock_generate_new.return_value = "Generated README"
        
        result = await generate_readme(
            repo_path=temp_repo_path,
            llm_client=mock_llm_client,
            file_manifest=file_manifest,
            file_summaries=file_manifest,
            config=sample_config,
            analyzer=mock_analyzer,
            output_dir=str(temp_repo_path),
            architecture_file_exists=False
        )
        
        assert result == "Generated README"
        mock_analyzer.derive_project_name.assert_called_once_with(True)  # debug=True

    def test_should_enhance_existing_readme_with_dict(self, temp_repo_path, sample_config_dict):
        """Test should_enhance_existing_readme with a dictionary config."""
        # Create a README file with more than 5 lines
        readme_path = temp_repo_path / 'README.md'
        readme_path.write_text('# Existing README\n\nThis is an existing README file.\n\nLine 4\nLine 5\nLine 6')
        
        result = should_enhance_existing_readme(temp_repo_path, sample_config_dict)
        
        assert result is True  # preserve_existing=True

    def test_should_enhance_existing_readme_with_scribe_config(self, temp_repo_path, sample_config):
        """Test should_enhance_existing_readme with a ScribeConfig instance."""
        # Create a README file with more than 5 lines
        readme_path = temp_repo_path / 'README.md'
        readme_path.write_text('# Existing README\n\nThis is an existing README file.\n\nLine 4\nLine 5\nLine 6')
        
        result = should_enhance_existing_readme(temp_repo_path, sample_config)
        
        assert result is True  # preserve_existing=True

    @pytest.mark.asyncio
    async def test_generate_new_readme_with_dict(self, temp_repo_path, mock_llm_client, file_manifest, sample_config_dict):
        """Test generate_new_readme with a dictionary config."""
        with patch('src.generators.readme.extract_license_info') as mock_extract_license:
            mock_extract_license.return_value = "MIT License"
            
            result = await generate_new_readme(
                repo_path=temp_repo_path,
                llm_client=mock_llm_client,
                file_manifest=file_manifest,
                project_name="Test Project",
                architecture_file_exists=False,
                config=sample_config_dict
            )
            
            assert "Test Project" in result
            assert "Project overview" in result
            assert "# # Usage" in result
            assert "Please refer to project documentation for contribution guidelines" in result

    @pytest.mark.asyncio
    async def test_generate_new_readme_with_scribe_config(self, temp_repo_path, mock_llm_client, file_manifest, sample_config):
        """Test generate_new_readme with a ScribeConfig instance."""
        with patch('src.generators.readme.extract_license_info') as mock_extract_license:
            mock_extract_license.return_value = "MIT License"
            
            result = await generate_new_readme(
                repo_path=temp_repo_path,
                llm_client=mock_llm_client,
                file_manifest=file_manifest,
                project_name="Test Project",
                architecture_file_exists=False,
                config=sample_config
            )
            
            assert "Test Project" in result
            assert "Project overview" in result
            assert "# # Usage" in result
            assert "Please refer to project documentation for contribution guidelines" in result