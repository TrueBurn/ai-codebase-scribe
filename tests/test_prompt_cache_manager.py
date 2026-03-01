#!/usr/bin/env python3

"""
Tests for src/utils/prompt_cache_manager.py

Verifies that PromptCacheManager can be instantiated from a ScribeConfig,
should_cache_component() returns a bool, and get_performance_metrics()
returns a dict.
"""

import os
import sys

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_class import BedrockConfig, ScribeConfig
from src.utils.prompt_cache_manager import CacheableComponent, PromptCacheManager


@pytest.fixture
def scribe_config():
    """Provide a ScribeConfig with Bedrock prompt caching settings."""
    config = ScribeConfig()
    config.bedrock = BedrockConfig(
        region="us-east-1",
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        enable_prompt_caching=True,
        cache_min_tokens=1024,
        cache_strategy="balanced",
    )
    return config


@pytest.fixture
def disabled_cache_config():
    """Provide a ScribeConfig with prompt caching disabled."""
    config = ScribeConfig()
    config.bedrock = BedrockConfig(
        region="us-east-1",
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        enable_prompt_caching=False,
        cache_min_tokens=1024,
        cache_strategy="balanced",
    )
    return config


class TestPromptCacheManagerInstantiation:
    """Tests for PromptCacheManager construction."""

    def test_can_be_instantiated(self, scribe_config):
        """Test that PromptCacheManager can be created from a ScribeConfig."""
        manager = PromptCacheManager(scribe_config)
        assert manager is not None

    def test_cache_enabled_from_config(self, scribe_config):
        """Test that cache_enabled reflects the bedrock config."""
        manager = PromptCacheManager(scribe_config)
        assert manager.cache_enabled is True

    def test_cache_disabled_from_config(self, disabled_cache_config):
        """Test that cache_enabled is False when disabled in config."""
        manager = PromptCacheManager(disabled_cache_config)
        assert manager.cache_enabled is False

    def test_strategy_read_from_config(self, scribe_config):
        """Test that the cache strategy is read from the config."""
        manager = PromptCacheManager(scribe_config)
        assert manager.strategy == "balanced"

    def test_min_cache_tokens_read_from_config(self, scribe_config):
        """Test that min_cache_tokens is read from the config."""
        manager = PromptCacheManager(scribe_config)
        assert manager.min_cache_tokens == 1024

    def test_initial_metrics_are_zero(self, scribe_config):
        """Test that performance counters start at zero."""
        manager = PromptCacheManager(scribe_config)
        assert manager.cache_hits == 0
        assert manager.cache_misses == 0
        assert manager.total_input_tokens == 0
        assert manager.total_output_tokens == 0


class TestShouldCacheComponent:
    """Tests for PromptCacheManager.should_cache_component()."""

    def test_returns_bool(self, scribe_config):
        """Test that should_cache_component() returns a bool."""
        manager = PromptCacheManager(scribe_config)
        result = manager.should_cache_component("some content")
        assert isinstance(result, bool)

    def test_returns_false_when_disabled(self, disabled_cache_config):
        """Test that should_cache_component() returns False when caching is disabled."""
        manager = PromptCacheManager(disabled_cache_config)
        # Even a very large content should return False
        result = manager.should_cache_component("x" * 100000)
        assert result is False

    def test_large_content_is_cached(self, scribe_config):
        """Test that content exceeding the minimum token threshold is cached."""
        manager = PromptCacheManager(scribe_config)
        # 1024 tokens * 4 chars/token = 4096 chars minimum; use 8000 to be safe
        large_content = "a" * 8000
        result = manager.should_cache_component(large_content)
        assert result is True

    def test_small_content_is_not_cached(self, scribe_config):
        """Test that very small content below the threshold is not cached."""
        manager = PromptCacheManager(scribe_config)
        small_content = "short"
        result = manager.should_cache_component(small_content)
        assert result is False


class TestGetPerformanceMetrics:
    """Tests for PromptCacheManager.get_performance_metrics()."""

    def test_returns_dict(self, scribe_config):
        """Test that get_performance_metrics() returns a dict."""
        manager = PromptCacheManager(scribe_config)
        metrics = manager.get_performance_metrics()
        assert isinstance(metrics, dict)

    def test_dict_contains_required_keys(self, scribe_config):
        """Test that the metrics dict contains expected keys."""
        manager = PromptCacheManager(scribe_config)
        metrics = manager.get_performance_metrics()

        assert "prompt_cache_enabled" in metrics
        assert "prompt_cache_strategy" in metrics
        assert "prompt_cache_hit_rate" in metrics
        assert "total_prompt_cached_requests" in metrics
        assert "total_requests" in metrics
        assert "average_generation_time" in metrics

    def test_cache_enabled_flag_in_metrics(self, scribe_config, disabled_cache_config):
        """Test that the metrics reflect whether caching is enabled."""
        enabled_manager = PromptCacheManager(scribe_config)
        disabled_manager = PromptCacheManager(disabled_cache_config)

        assert enabled_manager.get_performance_metrics()["prompt_cache_enabled"] is True
        assert (
            disabled_manager.get_performance_metrics()["prompt_cache_enabled"] is False
        )

    def test_metrics_after_response_update(self, scribe_config):
        """Test that update_metrics_from_response() is reflected in performance metrics."""
        manager = PromptCacheManager(scribe_config)
        manager.update_metrics_from_response(
            usage_data={
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 0,
            },
            generation_time=1.5,
        )

        metrics = manager.get_performance_metrics()
        assert metrics["total_input_tokens"] == 1000
        assert metrics["total_output_tokens"] == 200


class TestEstimateTokens:
    """Tests for token estimation utility."""

    def test_empty_string_returns_zero(self, scribe_config):
        """Test that empty string returns 0 token estimate."""
        manager = PromptCacheManager(scribe_config)
        assert manager.estimate_tokens("") == 0

    def test_proportional_to_length(self, scribe_config):
        """Test that token estimate is proportional to text length."""
        manager = PromptCacheManager(scribe_config)
        short_estimate = manager.estimate_tokens("a" * 100)
        long_estimate = manager.estimate_tokens("a" * 400)
        assert long_estimate == short_estimate * 4


class TestCacheableComponent:
    """Tests for the CacheableComponent dataclass."""

    def test_can_be_instantiated(self):
        """Test that CacheableComponent can be created."""
        component = CacheableComponent(
            content="some content",
            cache_key="test_key",
            should_cache=True,
            token_estimate=100,
        )
        assert component.content == "some content"
        assert component.cache_key == "test_key"
        assert component.should_cache is True
        assert component.token_estimate == 100

    def test_default_values(self):
        """Test that CacheableComponent has correct default values."""
        component = CacheableComponent(content="content", cache_key="key")
        assert component.should_cache is False
        assert component.token_estimate == 0
