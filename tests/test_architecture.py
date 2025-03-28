import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generators.architecture import (
    generate_architecture,
    create_fallback_architecture,
    analyze_basic_structure,
    MIN_CONTENT_LENGTH,
    MAX_TREE_LINES
)
from src.analyzers.codebase import CodebaseAnalyzer
from src.clients.base_llm import BaseLLMClient


class TestArchitecture(unittest.TestCase):
    """Test cases for the architecture.py module."""

    def setUp(self):
        """Set up test fixtures."""
        from src.utils.config_class import ScribeConfig
        
        self.repo_path = Path('/test/repo')
        self.config = ScribeConfig()
        self.config.debug = False
        self.file_manifest = {
            'src/main.py': MagicMock(summary="Main entry point"),
            'src/utils.py': MagicMock(summary="Utility functions"),
            'tests/test_main.py': MagicMock(summary="Tests for main"),
            'README.md': MagicMock(summary="Project readme"),
            'docs/ARCHITECTURE.md': MagicMock(summary="Architecture docs"),
            '.github/workflows/ci.yml': MagicMock(summary="CI configuration"),
        }

    def test_analyze_basic_structure(self):
        """Test the analyze_basic_structure function."""
        result = analyze_basic_structure(self.file_manifest)
        
        # Check that we have the expected sections
        self.assertIn("Technology Stack", result)
        self.assertIn("Project Patterns", result)
        
        # Check that Python is detected in the technology stack
        self.assertIn("Python", result["Technology Stack"])
        
        # Check that project patterns are detected
        self.assertIn("Test suite is present", result["Project Patterns"])
        self.assertIn("Documentation is available", result["Project Patterns"])
        self.assertIn("CI/CD configuration is set up", result["Project Patterns"])

    def test_create_fallback_architecture(self):
        """Test the create_fallback_architecture function."""
        project_name = "Test Project"
        result = create_fallback_architecture(project_name, self.file_manifest)
        
        # Check that the result contains the expected sections
        self.assertIn(f"# Project Architecture Analysis: {project_name}", result)
        self.assertIn("## Table of Contents", result)
        self.assertIn("## Overview", result)
        self.assertIn("## Project Structure", result)
        self.assertIn("## Technology Stack", result)
        self.assertIn("## Project Patterns", result)
        
        # Check that the file tree is included
        self.assertIn("```", result)
        self.assertIn("src/", result)
        self.assertIn("tests/", result)
        self.assertIn("docs/", result)
        self.assertIn(".github/", result)

    @patch('src.generators.architecture.CodebaseAnalyzer')
    @patch('src.clients.base_llm.BaseLLMClient')
    def test_generate_architecture_success(self, mock_llm_client, mock_analyzer_class):
        """Test the generate_architecture function with successful LLM response."""
        # Set up mocks
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.derive_project_name.return_value = "Test Project"
        
        mock_llm = AsyncMock()
        mock_llm.generate_architecture_doc = AsyncMock(return_value="""
# Project Architecture

## Overview
This is an overview of the project.

## Components
These are the main components.

## Dependencies
These are the dependencies.
""")
        
        # Run the test
        result = asyncio.run(generate_architecture(
            repo_path=self.repo_path,
            file_manifest=self.file_manifest,
            llm_client=mock_llm,
            config=self.config
        ))
        
        # Check that the LLM was called
        mock_llm.generate_architecture_doc.assert_called_once_with(self.file_manifest)
        
        # Check that the result contains the expected content
        self.assertIn("# Project Architecture Analysis: Test Project", result)
        self.assertIn("## Table of Contents", result)
        self.assertIn("- [Overview](#overview)", result)
        self.assertIn("- [Components](#components)", result)
        self.assertIn("- [Dependencies](#dependencies)", result)
        self.assertIn("This is an overview of the project.", result)

    @patch('src.generators.architecture.CodebaseAnalyzer')
    @patch('src.clients.base_llm.BaseLLMClient')
    def test_generate_architecture_llm_failure(self, mock_llm_client, mock_analyzer_class):
        """Test the generate_architecture function when LLM fails."""
        # Set up mocks
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.derive_project_name.return_value = "Test Project"
        
        mock_llm = AsyncMock()
        mock_llm.generate_architecture_doc = AsyncMock(side_effect=Exception("LLM failed"))
        
        # Run the test
        result = asyncio.run(generate_architecture(
            repo_path=self.repo_path,
            file_manifest=self.file_manifest,
            llm_client=mock_llm,
            config=self.config
        ))
        
        # Check that the LLM was called
        mock_llm.generate_architecture_doc.assert_called_once_with(self.file_manifest)
        
        # Check that we got a fallback architecture
        self.assertIn("# Project Architecture Analysis: Test Project", result)
        self.assertIn("## Overview", result)
        self.assertIn("This document provides a basic analysis of the project architecture.", result)
        self.assertIn("## Project Structure", result)

    @patch('src.generators.architecture.CodebaseAnalyzer')
    @patch('src.clients.base_llm.BaseLLMClient')
    def test_generate_architecture_short_content(self, mock_llm_client, mock_analyzer_class):
        """Test the generate_architecture function when LLM returns too short content."""
        # Set up mocks
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.derive_project_name.return_value = "Test Project"
        
        mock_llm = AsyncMock()
        mock_llm.generate_architecture_doc = AsyncMock(return_value="Too short")
        
        # Run the test
        result = asyncio.run(generate_architecture(
            repo_path=self.repo_path,
            file_manifest=self.file_manifest,
            llm_client=mock_llm,
            config=self.config
        ))
        
        # Check that the LLM was called
        mock_llm.generate_architecture_doc.assert_called_once_with(self.file_manifest)
        
        # Check that we got a fallback architecture
        self.assertIn("# Project Architecture Analysis: Test Project", result)
        self.assertIn("## Overview", result)
        self.assertIn("This document provides a basic analysis of the project architecture.", result)
        self.assertIn("## Project Structure", result)

    @patch('src.generators.architecture.CodebaseAnalyzer')
    @patch('src.clients.base_llm.BaseLLMClient')
    def test_generate_architecture_with_debug(self, mock_llm_client, mock_analyzer_class):
        """Test the generate_architecture function with debug enabled."""
        # Set up mocks
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.derive_project_name.return_value = "Test Project"
        
        mock_llm = AsyncMock()
        mock_llm.generate_architecture_doc = AsyncMock(return_value="""
# Project Architecture

## Overview
This is an overview of the project. This content needs to be longer than MIN_CONTENT_LENGTH
to ensure that it passes the length check and triggers the success log message.
This is additional text to make the content longer than the minimum required length.
This should be sufficient to pass the length check in the generate_architecture function.
""")
        
        # Run the test with debug enabled
        with patch('src.generators.architecture.logging.info') as mock_log_info:
            from src.utils.config_class import ScribeConfig
            debug_config = ScribeConfig()
            debug_config.debug = True
            
            result = asyncio.run(generate_architecture(
                repo_path=self.repo_path,
                file_manifest=self.file_manifest,
                llm_client=mock_llm,
                config=debug_config
            ))
        
        # Check that debug logs were called
        mock_log_info.assert_any_call(f"Generating architecture documentation for Test Project")
        mock_log_info.assert_any_call("Calling LLM to generate architecture documentation...")
        mock_log_info.assert_any_call("Successfully received architecture content from LLM")


if __name__ == '__main__':
    unittest.main()