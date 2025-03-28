#!/usr/bin/env python3

"""
Tests for config_utils.py
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.config_class import ScribeConfig
from src.utils.config_utils import (
    load_config,
    update_config_with_args,
    config_to_dict,
    dict_to_config,
    get_concurrency
)


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        'debug': True,
        'test_mode': False,
        'no_cache': False,
        'optimize_order': True,
        'llm_provider': 'ollama',
        'ollama': {
            'concurrency': 2,
            'model': 'llama2'
        },
        'bedrock': {
            'concurrency': 5,
            'model_id': 'anthropic.claude-v2'
        }
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """Create a sample ScribeConfig instance."""
    return ScribeConfig.from_dict(sample_config_dict)


class TestConfigUtils:
    """Test suite for config_utils.py."""

    @patch('src.utils.config_utils.load_config_dict')
    def test_load_config(self, mock_load_config_dict, sample_config_dict):
        """Test loading configuration from a file."""
        mock_load_config_dict.return_value = sample_config_dict
        
        config = load_config('config.yaml')
        
        assert isinstance(config, ScribeConfig)
        assert config.debug is True
        assert config.test_mode is False
        assert config.llm_provider == 'ollama'
        assert config.ollama.concurrency == 2
        assert config.bedrock.concurrency == 5

    def test_update_config_with_args(self, sample_config):
        """Test updating configuration with command-line arguments."""
        # Create mock args
        args = MagicMock()
        args.debug = True
        args.test_mode = True
        args.no_cache = True
        args.optimize_order = False
        args.llm_provider = 'bedrock'
        
        updated_config = update_config_with_args(sample_config, args)
        
        assert updated_config.debug is True
        assert updated_config.test_mode is True
        assert updated_config.no_cache is True
        assert updated_config.optimize_order is False
        assert updated_config.llm_provider == 'bedrock'

    def test_config_to_dict(self, sample_config, sample_config_dict):
        """Test converting ScribeConfig to dictionary."""
        result = config_to_dict(sample_config)
        assert isinstance(result, dict)
        assert result['debug'] == sample_config_dict['debug']
        assert result['llm_provider'] == sample_config_dict['llm_provider']

    def test_dict_to_config(self, sample_config_dict):
        """Test converting dictionary to ScribeConfig."""
        config = dict_to_config(sample_config_dict)
        
        assert isinstance(config, ScribeConfig)
        assert config.debug == sample_config_dict['debug']
        assert config.llm_provider == sample_config_dict['llm_provider']
        assert config.ollama.concurrency == sample_config_dict['ollama']['concurrency']

    def test_get_concurrency(self, sample_config):
        """Test getting concurrency setting from configuration."""
        # Test with ollama provider
        concurrency = get_concurrency(sample_config)
        assert concurrency == 2
        
        # Test with bedrock provider
        sample_config.llm_provider = 'bedrock'
        concurrency = get_concurrency(sample_config)
        assert concurrency == 5