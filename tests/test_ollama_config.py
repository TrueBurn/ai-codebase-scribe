#!/usr/bin/env python3

"""
Tests for OllamaClient with ScribeConfig
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

from src.clients.ollama import OllamaClient
from src.utils.config_class import ScribeConfig


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration object."""
    from src.utils.config_class import ScribeConfig, OllamaConfig
    
    config = ScribeConfig()
    config.debug = True
    config.ollama = OllamaConfig(
        base_url='http://test-ollama:11434',
        max_tokens=2048,
        retries=5,
        retry_delay=2.0,
        timeout=60,
        temperature=0.5
    )
    return config

@pytest.fixture
def sample_config():
    """Create a sample ScribeConfig instance."""
    from src.utils.config_class import ScribeConfig, OllamaConfig
    
    config = ScribeConfig()
    config.debug = True
    config.ollama = OllamaConfig(
        base_url='http://test-ollama:11434',
        max_tokens=2048,
        retries=5,
        retry_delay=2.0,
        timeout=60,
        temperature=0.5
    )
    return config


class TestOllamaClient:
    """Test suite for OllamaClient with ScribeConfig."""

    def test_init_with_scribe_config(self, sample_config_dict):
        """Test initializing OllamaClient with a ScribeConfig instance."""
        client = OllamaClient(sample_config_dict)
        
        assert client.base_url == 'http://test-ollama:11434'
        assert client.max_tokens == 2048
        assert client.retries == 5
        assert client.retry_delay == 2.0
        assert client.timeout == 60
        assert client.temperature == 0.5
        assert client.debug is True
        assert client.debug is True

    def test_init_with_sample_config(self, sample_config):
        """Test initializing OllamaClient with a different ScribeConfig instance."""
        client = OllamaClient(sample_config)
        
        assert client.base_url == 'http://test-ollama:11434'
        assert client.max_tokens == 2048
        assert client.retries == 5
        assert client.retry_delay == 2.0
        assert client.timeout == 60
        assert client.temperature == 0.5
        assert client.debug is True

    @patch('src.clients.ollama.AsyncClient')
    def test_client_initialization(self, mock_async_client, sample_config):
        """Test that the AsyncClient is initialized with the correct host."""
        OllamaClient(sample_config)
        mock_async_client.assert_called_once_with(host='http://test-ollama:11434')

    @patch('src.clients.ollama.PromptTemplate')
    def test_prompt_template_initialization(self, mock_prompt_template, sample_config_dict):
        """Test that the PromptTemplate is initialized correctly."""
        sample_config_dict.template_path = 'test/path'
        OllamaClient(sample_config_dict)
        mock_prompt_template.assert_called_once_with('test/path')

    @pytest.mark.asyncio
    @patch('src.clients.ollama.AsyncClient')
    async def test_initialize_method(self, mock_async_client, sample_config):
        """Test the initialize method."""
        # Create a mock for the AsyncClient instance
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        # Mock the list method to return a list of models
        mock_client_instance.list.return_value = {
            'models': [
                {'name': 'model1'},
                {'name': 'model2'}
            ]
        }
        
        # Create the client and initialize it
        client = OllamaClient(sample_config)
        
        # Mock the _select_model_interactive method to avoid input() call
        with patch.object(client, '_select_model_interactive', new=AsyncMock(return_value='model1')):
            await client.initialize()
            
            # Check that the list method was called
            mock_client_instance.list.assert_called_once()
            
            # Check that the available_models list was populated
            assert len(client.available_models) == 2
            assert 'model1' in client.available_models
            assert 'model2' in client.available_models
            
            # Check that the selected model was set
            assert client.selected_model == 'model1'