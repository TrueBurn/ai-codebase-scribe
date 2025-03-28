import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clients.llm_factory import (
    LLMClientFactory,
    ConfigValidationError,
    ClientInitializationError,
    LLMClientFactoryError
)
from src.clients.base_llm import BaseLLMClient
from src.utils.config_class import ScribeConfig, OllamaConfig, BedrockConfig


@pytest.fixture
def config():
    """Fixture to provide a test configuration."""
    config = ScribeConfig()
    config.llm_provider = 'ollama'
    config.debug = True
    
    # Configure Ollama
    config.ollama = OllamaConfig()
    config.ollama.base_url = 'http://localhost:11434'
    config.ollama.max_tokens = 4096
    
    # Configure Bedrock
    config.bedrock = BedrockConfig()
    config.bedrock.region = 'us-east-1'
    config.bedrock.model_id = 'test-model'
    
    return config


def test_validate_config_valid(config):
    """Test that validate_config accepts valid configurations."""
    # Valid Ollama config
    assert LLMClientFactory.validate_config(config) is True
    
    # Valid Bedrock config
    bedrock_config = ScribeConfig()
    bedrock_config.llm_provider = 'bedrock'
    bedrock_config.debug = True
    bedrock_config.bedrock = config.bedrock
    bedrock_config.ollama = config.ollama
    assert LLMClientFactory.validate_config(bedrock_config) is True


def test_validate_config_invalid(config):
    """Test that validate_config rejects invalid configurations."""
    # Invalid config type
    with pytest.raises(ConfigValidationError):
        LLMClientFactory.validate_config("not a dict")
    
    # Invalid provider
    invalid_provider = ScribeConfig()
    invalid_provider.llm_provider = 'invalid_provider'
    invalid_provider.ollama = config.ollama
    invalid_provider.bedrock = config.bedrock
    with pytest.raises(ConfigValidationError):
        LLMClientFactory.validate_config(invalid_provider)


def test_register_client_type():
    """Test registering a new client type."""
    # Create a mock client class
    class MockClient(BaseLLMClient):
        async def initialize(self):
            pass
        
        def init_token_counter(self):
            pass
        
        async def generate_summary(self, prompt):
            return "Mock summary"
        
        async def generate_project_overview(self, file_manifest):
            return "Mock overview"
        
        async def generate_usage_guide(self, file_manifest):
            return "Mock usage guide"
        
        async def generate_contributing_guide(self, file_manifest):
            return "Mock contributing guide"
        
        async def generate_license_info(self, file_manifest):
            return "Mock license info"
        
        async def generate_architecture_content(self, file_manifest, analyzer):
            return "Mock architecture content"
        
        async def generate_component_relationships(self, file_manifest):
            return "Mock component relationships"
        
        async def enhance_documentation(self, existing_content, file_manifest, doc_type):
            return "Mock enhanced documentation"
        
        def set_project_structure(self, structure):
            pass
        
        async def get_file_order(self, project_files):
            return list(project_files.keys())
    
    # Register the mock client
    LLMClientFactory.register_client_type('mock_provider', MockClient)
    
    # Verify it was registered
    assert 'mock_provider' in LLMClientFactory._client_registry
    assert LLMClientFactory._client_registry['mock_provider'] == MockClient


@pytest.mark.asyncio
async def test_create_client_ollama(config):
    """Test creating an Ollama client."""
    # Mock the OllamaClient class and its initialize method
    with patch('src.clients.llm_factory.OllamaClient') as mock_ollama_client:
        # Setup mock
        mock_instance = AsyncMock()
        mock_ollama_client.return_value = mock_instance
        
        # Call the method
        client = await LLMClientFactory.create_client(config)
        
        # Verify the result
        mock_ollama_client.assert_called_once_with(config)
        mock_instance.initialize.assert_called_once()
        assert client == mock_instance


@pytest.mark.asyncio
async def test_create_client_bedrock(config):
    """Test creating a Bedrock client."""
    # Modify config to use Bedrock
    bedrock_config = ScribeConfig()
    bedrock_config.llm_provider = 'bedrock'
    bedrock_config.debug = config.debug
    bedrock_config.bedrock = config.bedrock
    bedrock_config.ollama = config.ollama
    
    # Mock the BedrockClient class and its initialize method
    with patch('src.clients.llm_factory.BedrockClient') as mock_bedrock_client:
        # Setup mock
        mock_instance = AsyncMock()
        mock_bedrock_client.return_value = mock_instance
        
        # Call the method
        client = await LLMClientFactory.create_client(bedrock_config)
        
        # Verify the result
        mock_bedrock_client.assert_called_once_with(bedrock_config)
        mock_instance.initialize.assert_called_once()
        assert client == mock_instance


@pytest.mark.asyncio
async def test_bedrock_fallback_to_ollama(config):
    """Test fallback from Bedrock to Ollama when Bedrock initialization fails."""
    # Modify config to use Bedrock
    bedrock_config = ScribeConfig()
    bedrock_config.llm_provider = 'bedrock'
    bedrock_config.debug = config.debug
    bedrock_config.bedrock = config.bedrock
    bedrock_config.ollama = config.ollama
    
    # Mock both client classes
    with patch('src.clients.llm_factory.BedrockClient') as mock_bedrock_client, \
         patch('src.clients.llm_factory.OllamaClient') as mock_ollama_client:
        
        # Setup mocks
        mock_bedrock_instance = AsyncMock()
        mock_bedrock_instance.initialize.side_effect = Exception("Bedrock initialization failed")
        mock_bedrock_client.return_value = mock_bedrock_instance
        
        mock_ollama_instance = AsyncMock()
        mock_ollama_client.return_value = mock_ollama_instance
        
        # Call the method
        client = await LLMClientFactory.create_client(bedrock_config)
        
        # Verify the result
        mock_bedrock_client.assert_called_once_with(bedrock_config)
        mock_bedrock_instance.initialize.assert_called_once()
        mock_ollama_client.assert_called_once_with(bedrock_config)
        mock_ollama_instance.initialize.assert_called_once()
        assert client == mock_ollama_instance


@pytest.mark.asyncio
async def test_ollama_initialization_error(config):
    """Test error handling when Ollama initialization fails."""
    # Mock the OllamaClient class
    with patch('src.clients.llm_factory.OllamaClient') as mock_ollama_client:
        # Setup mock
        mock_instance = AsyncMock()
        mock_instance.initialize.side_effect = Exception("Ollama initialization failed")
        mock_ollama_client.return_value = mock_instance
        
        # Call the method and verify it raises the expected exception
        with pytest.raises(ClientInitializationError):
            await LLMClientFactory.create_client(config)
        
        # Verify the mock was called
        mock_ollama_client.assert_called_once_with(config)
        mock_instance.initialize.assert_called_once()