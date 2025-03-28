import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
from typing import Dict, Any, Optional, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clients.base_llm import BaseLLMClient


class TestBaseLLMClient(unittest.TestCase):
    """Test cases for the BaseLLMClient class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation of BaseLLMClient for testing
        class ConcreteLLMClient(BaseLLMClient):
            async def initialize(self) -> None:
                pass

            def init_token_counter(self) -> None:
                self.token_counter = MagicMock()

            async def generate_summary(self, prompt: str) -> Optional[str]:
                return "Summary of " + prompt[:10] + "..."

            async def generate_project_overview(self, file_manifest: Dict[str, Any]) -> str:
                return "Project overview"

            async def generate_usage_guide(self, file_manifest: Dict[str, Any]) -> str:
                return "Usage guide"

            async def generate_contributing_guide(self, file_manifest: Dict[str, Any]) -> str:
                return "Contributing guide"

            async def generate_license_info(self, file_manifest: Dict[str, Any]) -> str:
                return "License info"

            async def generate_architecture_content(self, file_manifest: Dict[str, Any], analyzer: Any) -> str:
                return "Architecture content"

            async def generate_component_relationships(self, file_manifest: Dict[str, Any]) -> str:
                return "Component relationships"

            async def enhance_documentation(self, existing_content: str, file_manifest: Dict[str, Any], doc_type: str) -> str:
                return f"Enhanced {doc_type}: {existing_content[:10]}..."

            def set_project_structure(self, structure: str) -> None:
                self.project_structure = structure

            async def get_file_order(self, project_files: Dict[str, Any]) -> List[str]:
                return list(project_files.keys())

        self.client = ConcreteLLMClient()

    def test_version(self):
        """Test that the VERSION constant is defined."""
        self.assertEqual(BaseLLMClient.VERSION, "1.0.0")

    def test_init(self):
        """Test initialization."""
        self.assertIsNone(self.client.token_counter)
        self.assertIsNone(self.client.project_structure)

    def test_validate_input(self):
        """Test the validate_input method."""
        # Valid input
        self.assertTrue(self.client.validate_input("Hello world"))
        
        # Invalid inputs
        self.assertFalse(self.client.validate_input(""))
        self.assertFalse(self.client.validate_input(None))
        self.assertFalse(self.client.validate_input(123))

    def test_validate_file_manifest(self):
        """Test the validate_file_manifest method."""
        # Valid manifest
        self.assertTrue(self.client.validate_file_manifest({"file.py": {}}))
        
        # Invalid manifests
        self.assertFalse(self.client.validate_file_manifest(None))
        self.assertFalse(self.client.validate_file_manifest("not a dict"))
        self.assertFalse(self.client.validate_file_manifest([]))

    def test_set_project_structure(self):
        """Test setting project structure."""
        structure = "src/\n  main.py\n  utils.py"
        self.client.set_project_structure(structure)
        self.assertEqual(self.client.project_structure, structure)


if __name__ == '__main__':
    unittest.main()