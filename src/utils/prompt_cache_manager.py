"""
AWS Bedrock Prompt Cache Management Utilities.

This module manages prompt caching strategy and optimization for AWS Bedrock,
separate from the existing file cache system. It handles cache-optimized
message structures and performance tracking for LLM calls.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass

# Try to import visual logger, fallback to regular logging
try:
    from .visual_logger import get_visual_logger

    _use_visual_logging = True
except ImportError:
    _use_visual_logging = False


def _get_logger():
    """Get appropriate logger (visual or standard)."""
    if _use_visual_logging:
        return get_visual_logger()
    return logging.getLogger(__name__)


@dataclass
class CacheableComponent:
    """Represents a component that can be prompt cached."""

    content: str
    cache_key: str
    should_cache: bool = False
    token_estimate: int = 0


class PromptCacheManager:
    """Manages AWS Bedrock prompt cache strategy and optimization.

    Note: This is separate from the existing file cache system.
    This handles AWS Bedrock's prompt caching feature for LLM calls.
    """

    def __init__(self, config):
        self.config = config
        # Safe configuration access with sensible defaults
        bedrock_config = getattr(config, "bedrock", None)
        self.cache_enabled = getattr(
            bedrock_config, "enable_prompt_caching", True
        )  # Default ON
        self.min_cache_tokens = getattr(bedrock_config, "cache_min_tokens", 1024)
        self.strategy = getattr(bedrock_config, "cache_strategy", "balanced")

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.generation_times = []

        # Token tracking (from Bedrock response metadata)
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Log prompt cache configuration on initialization
        logger = _get_logger()
        if self.cache_enabled:
            if _use_visual_logging:
                logger.info(
                    f"PROMPT_CACHE_CONFIG: Prompt caching ENABLED (strategy: {self.strategy}, min_tokens: {logger.format_number(self.min_cache_tokens)})",
                    emoji="prompt_cache",
                )
            else:
                logging.info(
                    f"PROMPT_CACHE_CONFIG: Prompt caching ENABLED (strategy: {self.strategy}, min_tokens: {self.min_cache_tokens})"
                )
        else:
            if _use_visual_logging:
                logger.info(
                    "PROMPT_CACHE_CONFIG: Prompt caching DISABLED", emoji="config"
                )
            else:
                logging.info("PROMPT_CACHE_CONFIG: Prompt caching DISABLED")

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4

    def should_cache_component(self, content: str) -> bool:
        """Determine if component should be prompt cached based on size and strategy."""
        # Early exit if prompt caching is disabled
        if not self.cache_enabled:
            logging.debug("PROMPT_CACHE_SKIP: Prompt caching disabled in configuration")
            return False

        token_count = self.estimate_tokens(content)

        # Log prompt caching decision for debugging
        cache_decision = self._evaluate_cache_strategy(token_count)
        if cache_decision:
            logging.debug(
                f"PROMPT_CACHE_INCLUDE: Component ({token_count} tokens, strategy: {self.strategy})"
            )
        else:
            logging.debug(
                f"PROMPT_CACHE_EXCLUDE: Component too small ({token_count} < {self._get_threshold()} tokens)"
            )

        return cache_decision

    def _evaluate_cache_strategy(self, token_count: int) -> bool:
        """Evaluate prompt caching strategy against token count."""
        threshold = self._get_threshold()
        return token_count >= threshold

    def _get_threshold(self) -> int:
        """Get token threshold based on strategy."""
        if self.strategy == "conservative":
            return self.min_cache_tokens * 2  # Cache only very large components
        elif self.strategy == "balanced":
            return self.min_cache_tokens  # Cache medium+ components
        else:  # aggressive
            return self.min_cache_tokens // 2  # Cache smaller components too

    def create_cacheable_components(
        self,
        project_structure: str = None,
        tech_report: str = None,
        key_components: str = None,
        system_constraints: str = None,
        task_content: str = "",
    ) -> List[CacheableComponent]:
        """Create optimized prompt cacheable components."""

        components = []

        # System constraints (anti-hallucination, Mermaid syntax)
        if system_constraints and self.should_cache_component(system_constraints):
            components.append(
                CacheableComponent(
                    content=system_constraints,
                    cache_key="system_constraints",
                    should_cache=True,
                    token_estimate=self.estimate_tokens(system_constraints),
                )
            )

        # Project structure (usually largest component)
        if project_structure and self.should_cache_component(project_structure):
            components.append(
                CacheableComponent(
                    content=f"## Project Structure\n{project_structure}",
                    cache_key="project_structure",
                    should_cache=True,
                    token_estimate=self.estimate_tokens(project_structure),
                )
            )

        # Technology report
        if tech_report and self.should_cache_component(tech_report):
            components.append(
                CacheableComponent(
                    content=f"## Detected Technologies\n{tech_report}",
                    cache_key="tech_report",
                    should_cache=True,
                    token_estimate=self.estimate_tokens(tech_report),
                )
            )

        # Key components analysis
        if key_components and self.should_cache_component(key_components):
            components.append(
                CacheableComponent(
                    content=f"## Key Components\n{key_components}",
                    cache_key="key_components",
                    should_cache=True,
                    token_estimate=self.estimate_tokens(key_components),
                )
            )

        # Task-specific content (never cached)
        if task_content:
            components.append(
                CacheableComponent(
                    content=task_content,
                    cache_key="task_specific",
                    should_cache=False,
                    token_estimate=self.estimate_tokens(task_content),
                )
            )

        return components

    def build_cached_message_content(
        self, components: List[CacheableComponent]
    ) -> List[Dict[str, Any]]:
        """Build message content with prompt cache controls."""
        content_blocks = []

        for component in components:
            # Create content block with InvokeModel format (like your JS example)
            block = {"type": "text", "text": component.content}

            # Add cache_control for cacheable components (InvokeModel format)
            if component.should_cache and self.cache_enabled:
                block["cache_control"] = {"type": "ephemeral"}
                logging.debug(
                    f"PROMPT_CACHE: Caching component '{component.cache_key}' ({component.token_estimate} tokens)"
                )

            content_blocks.append(block)

        return content_blocks

    def log_cache_usage(
        self, components: List[CacheableComponent], generation_time: float
    ):
        """Log prompt cache usage for performance monitoring."""
        cached_components = [c for c in components if c.should_cache]
        total_cached_tokens = sum(c.token_estimate for c in cached_components)

        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                f"PROMPT_CACHE_USAGE: {logger.format_number(len(cached_components))} components cached, {logger.format_number(total_cached_tokens)} tokens, {generation_time:.2f}s",
                emoji="stats",
            )
        else:
            logging.info(
                f"PROMPT_CACHE_USAGE: {len(cached_components)} components cached, {total_cached_tokens} tokens, {generation_time:.2f}s"
            )

        if cached_components:
            self.cache_hits += len(cached_components)
            self.generation_times.append(generation_time)

    def update_metrics_from_response(self, usage_data: dict, generation_time: float):
        """Update metrics from Bedrock response usage data."""
        # Update token counts (Claude API format)
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cache_read_tokens = usage_data.get("cache_read_input_tokens", 0)
        cache_write_tokens = usage_data.get("cache_creation_input_tokens", 0)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.total_cache_write_tokens += cache_write_tokens

        # Update hit/miss counters
        if cache_read_tokens > 0:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Track generation time
        self.generation_times.append(generation_time)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive prompt cache performance metrics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        avg_time = (
            sum(self.generation_times) / len(self.generation_times)
            if self.generation_times
            else 0
        )

        # Calculate token-based metrics
        total_tokens = self.total_input_tokens + self.total_output_tokens
        cache_efficiency = (
            (self.total_cache_read_tokens / self.total_input_tokens * 100)
            if self.total_input_tokens > 0
            else 0
        )
        cost_savings_tokens = (
            self.total_cache_read_tokens * 0.9
        )  # 90% savings on cached tokens

        metrics = {
            "prompt_cache_enabled": self.cache_enabled,
            "prompt_cache_strategy": self.strategy,
            "prompt_cache_hit_rate": f"{hit_rate:.1f}%",
            "total_prompt_cached_requests": self.cache_hits,
            "total_requests": total_requests,
            "average_generation_time": f"{avg_time:.2f}s",
        }

        # Add token metrics if available
        if total_tokens > 0:
            metrics.update(
                {
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_cached_read_tokens": self.total_cache_read_tokens,
                    "total_cached_write_tokens": self.total_cache_write_tokens,
                    "cache_efficiency_rate": f"{cache_efficiency:.1f}%",
                    "estimated_token_cost_savings": f"{cost_savings_tokens:.0f} token-equivalents",
                }
            )

        return metrics
