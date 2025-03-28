#!/usr/bin/env python3

"""
Tests for architecture generator with ScribeConfig
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.generators.architecture import generate_architecture
from src.utils.config_class import ScribeConfig


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration object."""
    from src.utils.config_class import ScribeConfig, BlacklistConfig
    
    config = ScribeConfig()
    config.debug = True
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
    config.test_mode = True
    config.blacklist = BlacklistConfig(
        extensions=['.log', '.tmp'],
        path_patterns=['/node_modules/', '/__pycache__/']
    )
    return config


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
    mock_client.generate_architecture_doc = AsyncMock(return_value="""
# Architecture Documentation

## Overview
This is a detailed overview of the architecture. This content is long enough to pass the MIN_CONTENT_LENGTH check.
The architecture follows a modular design with clear separation of concerns.

## Components
- Component A: Handles data processing
- Component B: Manages user interface
- Component C: Provides API integration

## Dependencies
This section describes the dependencies between components.
    """)
    mock_client.generate_component_diagram = AsyncMock(return_value="```mermaid\ngraph TD;\nA-->B;\n```")
    return mock_client


@pytest.fixture
def file_manifest():
    """Create a sample file manifest."""
    return {
        'file1.py': MagicMock(summary="Python file"),
        'file2.js': MagicMock(summary="JavaScript file")
    }


class TestArchitectureGenerator:
    """Test suite for architecture generator with ScribeConfig."""

    @pytest.mark.asyncio
    @patch('src.generators.architecture.CodebaseAnalyzer')
    async def test_generate_architecture_with_config(self, mock_analyzer_class, temp_repo_path, mock_llm_client, file_manifest, sample_config_dict):
        """Test generating architecture documentation with a ScribeConfig object."""
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.derive_project_name.return_value = "Test Project"
        mock_analyzer_class.return_value = mock_analyzer
        
        result = await generate_architecture(
            repo_path=temp_repo_path,
            file_manifest=file_manifest,
            llm_client=mock_llm_client,
            config=sample_config_dict
        )
        
        assert "Project Architecture Analysis: Test Project" in result
        assert "This is a detailed overview of the architecture" in result
        assert "Component A: Handles data processing" in result
        mock_analyzer_class.assert_called_once_with(temp_repo_path, sample_config_dict)
        mock_analyzer.derive_project_name.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.generators.architecture.CodebaseAnalyzer')
    async def test_generate_architecture_with_scribe_config(self, mock_analyzer_class, temp_repo_path, mock_llm_client, file_manifest, sample_config):
        """Test generating architecture documentation with a ScribeConfig instance."""
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.derive_project_name.return_value = "Test Project"
        mock_analyzer_class.return_value = mock_analyzer
        
        result = await generate_architecture(
            repo_path=temp_repo_path,
            file_manifest=file_manifest,
            llm_client=mock_llm_client,
            config=sample_config
        )
        
        assert "Project Architecture Analysis: Test Project" in result
        assert "This is a detailed overview of the architecture" in result
        assert "Component A: Handles data processing" in result
        mock_analyzer_class.assert_called_once_with(temp_repo_path, sample_config)
        mock_analyzer.derive_project_name.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.generators.architecture.CodebaseAnalyzer')
    @patch('src.generators.architecture.MermaidGenerator')
    async def test_mermaid_diagram_generation(self, mock_mermaid_class, mock_analyzer_class, temp_repo_path, mock_llm_client, file_manifest, sample_config):
        """Test that Mermaid diagrams are generated correctly."""
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.derive_project_name.return_value = "Test Project"
        mock_analyzer_class.return_value = mock_analyzer
        
        # Set up mock mermaid generator
        mock_mermaid = MagicMock()
        mock_mermaid.generate_dependency_flowchart.return_value = "```mermaid\ngraph TD;\nA-->B;\n```"
        mock_mermaid_class.return_value = mock_mermaid
        
        result = await generate_architecture(
            repo_path=temp_repo_path,
            file_manifest=file_manifest,
            llm_client=mock_llm_client,
            config=sample_config
        )
        
        assert "Project Architecture Analysis: Test Project" in result
        assert "This is a detailed overview of the architecture" in result
        assert "Component A: Handles data processing" in result
        assert "```mermaid" in result
        mock_mermaid_class.assert_called_once()
        mock_mermaid.generate_dependency_flowchart.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.generators.architecture.CodebaseAnalyzer')
    async def test_error_handling(self, mock_analyzer_class, temp_repo_path, file_manifest, sample_config):
        """Test error handling in generate_architecture."""
        # Set up mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.derive_project_name.return_value = "Test Project"
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create a mock LLM client that raises an exception
        mock_error_client = AsyncMock()
        mock_error_client.generate_architecture_doc.side_effect = Exception("Test error")
        
        result = await generate_architecture(
            repo_path=temp_repo_path,
            file_manifest=file_manifest,
            llm_client=mock_error_client,
            config=sample_config
        )
        
        # Should return a fallback architecture document
        assert "# Project Architecture Analysis: Test Project" in result
        assert "This document provides a basic analysis of the project architecture" in result