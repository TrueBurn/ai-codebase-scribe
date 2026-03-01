#!/usr/bin/env python3

"""
Tests for config_class.py
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.utils.config_class import (
    BedrockConfig,
    BlacklistConfig,
    CacheConfig,
    ContributingConfig,
    DocTemplatesConfig,
    InstallationConfig,
    LargeRepoConfig,
    OllamaConfig,
    PersistenceConfig,
    PromptTemplatesConfig,
    ReadmeRefactorConfig,
    ScribeConfig,
    TemplatesConfig,
    TroubleshootingConfig,
    UsageConfig,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
        yield temp.name
    # Clean up the temp file after the test
    os.unlink(temp.name)


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "debug": True,
        "test_mode": False,
        "no_cache": False,
        "optimize_order": True,
        "preserve_existing": True,
        "llm_provider": "ollama",
        "ollama": {
            "base_url": "http://localhost:11434",
            "max_tokens": 4096,
            "retries": 3,
            "retry_delay": 1.0,
            "timeout": 30,
            "concurrency": 2,
            "temperature": 0.1,
        },
        "bedrock": {
            "region": "us-east-1",
            "model_id": "anthropic.claude-v2",
            "max_tokens": 8192,
            "timeout": 180,
            "retries": 5,
            "concurrency": 5,
            "verify_ssl": False,
            "temperature": 0.1,
        },
        "cache": {
            "enabled": True,
            "directory": ".cache",
            "location": "repo",
            "hash_algorithm": "md5",
            "global_directory": "readme_generator_cache",
        },
        "blacklist": {
            "extensions": [".txt", ".log"],
            "path_patterns": ["/temp/", "/cache/", "/node_modules/"],
        },
        "templates": {
            "prompts": {
                "file_summary": "Analyze the following code file: {file_path}",
                "project_overview": "Generate an overview for: {project_name}",
            },
            "docs": {"readme": "# {project_name}\n\n{project_overview}"},
        },
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """Create a sample ScribeConfig instance."""
    return ScribeConfig.from_dict(sample_config_dict)


class TestScribeConfig:
    """Test suite for ScribeConfig class."""

    def test_from_dict(self, sample_config_dict):
        """Test creating ScribeConfig from dictionary."""
        config = ScribeConfig.from_dict(sample_config_dict)

        assert isinstance(config, ScribeConfig)
        assert config.debug is True
        assert config.test_mode is False
        assert config.no_cache is False
        assert config.optimize_order is True
        assert config.preserve_existing is True
        assert config.llm_provider == "ollama"

        # Test nested configs
        assert isinstance(config.ollama, OllamaConfig)
        assert config.ollama.base_url == "http://localhost:11434"
        assert config.ollama.concurrency == 2

        assert isinstance(config.bedrock, BedrockConfig)
        assert config.bedrock.region == "us-east-1"
        assert config.bedrock.model_id == "anthropic.claude-v2"
        assert config.bedrock.concurrency == 5

        assert isinstance(config.cache, CacheConfig)
        assert config.cache.enabled is True
        assert config.cache.location == "repo"

        assert isinstance(config.blacklist, BlacklistConfig)
        assert ".txt" in config.blacklist.extensions
        assert "/temp/" in config.blacklist.path_patterns

        assert isinstance(config.templates, TemplatesConfig)
        assert isinstance(config.templates.prompts, PromptTemplatesConfig)
        assert isinstance(config.templates.docs, DocTemplatesConfig)
        assert "{file_path}" in config.templates.prompts.file_summary
        assert "{project_name}" in config.templates.docs.readme

    def test_to_dict(self, sample_config):
        """Test converting ScribeConfig to dictionary."""
        config_dict = sample_config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["debug"] is True
        assert config_dict["llm_provider"] == "ollama"
        assert config_dict["ollama"]["base_url"] == "http://localhost:11434"
        assert config_dict["bedrock"]["region"] == "us-east-1"
        assert config_dict["cache"]["enabled"] is True
        assert config_dict["blacklist"]["extensions"] == [".txt", ".log"]
        assert "{file_path}" in config_dict["templates"]["prompts"]["file_summary"]

    def test_get_concurrency(self, sample_config):
        """Test getting concurrency setting from configuration."""
        # Test with ollama provider
        concurrency = sample_config.get_concurrency()
        assert concurrency == 2

        # Test with bedrock provider
        sample_config.llm_provider = "bedrock"
        concurrency = sample_config.get_concurrency()
        assert concurrency == 5

    def test_write_to_file(self, sample_config, temp_config_file):
        """Test writing configuration to file."""
        sample_config.write_to_file(temp_config_file)

        # Read the file back and verify
        with open(temp_config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        assert config_dict["debug"] is True
        assert config_dict["llm_provider"] == "ollama"
        assert config_dict["ollama"]["base_url"] == "http://localhost:11434"

    def test_new_dataclasses_have_defaults(self):
        """Test that new dataclasses can be instantiated with default values."""
        config = ScribeConfig()

        assert isinstance(config.large_repo, LargeRepoConfig)
        assert config.large_repo.threshold == 450
        assert config.large_repo.max_files == 1000

        assert isinstance(config.persistence, PersistenceConfig)
        assert config.persistence.enabled is True
        assert config.persistence.generate_doc is True
        assert config.persistence.output_file == "docs/PERSISTENCE.md"

        assert isinstance(config.installation, InstallationConfig)
        assert config.installation.enabled is True
        assert config.installation.output_file == "docs/INSTALLATION.md"

        assert isinstance(config.usage, UsageConfig)
        assert config.usage.enabled is True
        assert config.usage.output_file == "docs/USAGE.md"

        assert isinstance(config.troubleshooting, TroubleshootingConfig)
        assert config.troubleshooting.enabled is True
        assert config.troubleshooting.output_file == "docs/TROUBLESHOOTING.md"

        assert isinstance(config.contributing, ContributingConfig)
        assert config.contributing.enabled is True
        assert config.contributing.output_file == "CONTRIBUTING.md"

        assert isinstance(config.readme_refactor, ReadmeRefactorConfig)
        assert config.readme_refactor.enabled is True
        assert config.readme_refactor.keep_brief_overview is True

    def test_bedrock_config_new_fields(self):
        """Test that BedrockConfig has the new fields added in the Capitec migration."""
        bedrock = BedrockConfig()

        # Fallback configuration
        assert hasattr(bedrock, "fallback_model_id")
        assert hasattr(bedrock, "enable_fallback")
        assert bedrock.enable_fallback is True
        assert hasattr(bedrock, "throttling_retry_delay")
        assert hasattr(bedrock, "throttling_max_retries")

        # Output token limits
        assert hasattr(bedrock, "max_output_tokens_architecture")
        assert bedrock.max_output_tokens_architecture == 32768
        assert hasattr(bedrock, "max_output_tokens_persistence")
        assert bedrock.max_output_tokens_persistence == 32768

        # Prompt caching
        assert hasattr(bedrock, "enable_prompt_caching")
        assert hasattr(bedrock, "cache_min_tokens")
        assert bedrock.cache_min_tokens == 1024
        assert hasattr(bedrock, "cache_ttl_minutes")
        assert hasattr(bedrock, "cache_strategy")
        assert bedrock.cache_strategy == "balanced"

        # Extended context
        assert hasattr(bedrock, "extended_context_enabled")
        assert bedrock.extended_context_enabled is True
        assert hasattr(bedrock, "extended_context_beta_header")
