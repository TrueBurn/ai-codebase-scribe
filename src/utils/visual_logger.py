#!/usr/bin/env python3

"""
Visual logging module for ai-codebase-scribe.
Provides enhanced terminal output with colors, emojis, and rich formatting.
"""

import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from colorama import init as colorama_init

# Initialize colorama for Windows compatibility
colorama_init(autoreset=True)

# Create a rich console for beautiful output with same width as progress bars (150)
console = Console(width=150)

# Emoji mappings for different operations
EMOJIS = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🔍",
    "start": "🚀",
    "complete": "🎉",
    "found": "✨",
    "skip": "⏭️",
    "analyze": "🤖",
    "report": "📊",
    "config": "⚙️",
    "github": "🐙",
    "aws": "☁️",
    "cache": "💾",
    "repository": "📦",
    "branch": "🌿",
    "file": "📄",
    "processing": "⚡",
    "architecture": "🏗️",
    "readme": "📖",
    "contributing": "🤝",
    "persistence": "🗄️",
    "cicd": "🔄",
    "prompt_cache": "🚀",
    "token": "🪙",
    "timer": "⏱️",
    "link": "🔗",
    "badge": "🏷️",
    "package": "📦",  # Installation documentation
    "install": "📦",  # Alternative for installation
    "rocket": "🚀",  # Usage/Quick start documentation
    "wrench": "🔧",  # Troubleshooting documentation
    "refactor": "♻️",  # README refactoring
}

# Color themes for different types of content
COLORS = {
    "header": "bright_cyan",
    "success": "bright_green",
    "error": "bright_red",
    "warning": "yellow",
    "info": "bright_blue",
    "debug": "dim white",
    "highlight": "bright_white",
    "number": "bright_yellow",
    "percentage": "bright_magenta",
    "path": "cyan",
    "url": "blue",
    "filename": "green",
    "time": "bright_magenta",
    "size": "bright_yellow",
    "count": "bright_green",
}


class VisualLogger:
    """Enhanced logger with rich visual formatting."""

    def __init__(
        self,
        name: str = "ai-codebase-scribe",
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        enable_rich: bool = True,
    ):
        """
        Initialize the visual logger.

        Args:
            name: Logger name
            log_file: Optional log file path
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_rich: Whether to enable rich formatting
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.console = console
        self.enable_rich = enable_rich

        # Clear any existing handlers
        self.logger.handlers.clear()

        if enable_rich:
            # Add rich handler for console output
            rich_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                log_time_format="[%H:%M:%S]",
                keywords=[],
            )
            rich_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            self.logger.addHandler(rich_handler)
        else:
            # Fallback to standard console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            formatter = logging.Formatter("%(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Prevent propagation
        self.logger.propagate = False

    def header(self, title: str, subtitle: Optional[str] = None):
        """Display a beautiful header."""
        if not self.enable_rich:
            print(f"\n{'='*60}")
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print(f"{'='*60}\n")
            return

        header_text = Text(title, style=f"bold {COLORS['header']}")
        if subtitle:
            subtitle_text = Text(subtitle, style="dim white")
            content = Text.assemble(header_text, "\n", subtitle_text)
        else:
            content = header_text

        panel = Panel(
            content,
            style=COLORS["header"],
            border_style=COLORS["header"],
            padding=(1, 2),
            expand=False,
        )
        self.console.print(panel)

    def success(self, message: str, emoji: str = "success"):
        """Log a success message with rich formatting."""
        if self.enable_rich:
            self.logger.info(f"{EMOJIS[emoji]} [green]{message}[/green]")
        else:
            self.logger.info(f"✓ {message}")

    def error(self, message: str, emoji: str = "error"):
        """Log an error message with rich formatting."""
        if self.enable_rich:
            self.logger.error(f"{EMOJIS[emoji]} [red]{message}[/red]")
        else:
            self.logger.error(f"✗ {message}")

    def warning(self, message: str, emoji: str = "warning"):
        """Log a warning message with rich formatting."""
        if self.enable_rich:
            self.logger.warning(f"{EMOJIS[emoji]} [yellow]{message}[/yellow]")
        else:
            self.logger.warning(f"! {message}")

    def info(self, message: str, emoji: Optional[str] = None):
        """Log an info message with optional emoji and rich formatting."""
        if self.enable_rich:
            if emoji and emoji in EMOJIS:
                self.logger.info(f"{EMOJIS[emoji]} {message}")
            else:
                self.logger.info(message)
        else:
            self.logger.info(message)

    def debug(self, message: str):
        """Log a debug message with rich formatting."""
        if self.enable_rich:
            self.logger.debug(f"{EMOJIS['debug']} [dim]{message}[/dim]")
        else:
            self.logger.debug(f"DEBUG: {message}")

    def section(self, title: str, style: str = "bright_cyan"):
        """Create a visual section separator."""
        if not self.enable_rich:
            print(f"\n{'─'*60}")
            print(f"  {title}")
            print(f"{'─'*60}\n")
            return

        separator = "─" * 60
        self.console.print(f"\n[{style}]{separator}[/{style}]")
        self.console.print(f"[bold {style}] {title}[/bold {style}]")
        self.console.print(f"[{style}]{separator}[/{style}]\n")

    def format_number(self, number: Union[int, float], suffix: str = "") -> str:
        """Format a number with rich color styling."""
        if not self.enable_rich:
            return f"{number:,}{suffix}"

        if isinstance(number, float):
            return f"[{COLORS['number']}]{number:.1f}[/{COLORS['number']}]{suffix}"
        else:
            return f"[{COLORS['number']}]{number:,}[/{COLORS['number']}]{suffix}"

    def format_percentage(self, percentage: Union[int, float]) -> str:
        """Format a percentage with rich color styling."""
        if not self.enable_rich:
            return f"{percentage}%"
        return f"[{COLORS['percentage']}]{percentage}%[/{COLORS['percentage']}]"

    def format_path(self, path: Union[str, Path]) -> str:
        """Format a file path with rich color styling."""
        if not self.enable_rich:
            return str(path)
        return f"[{COLORS['path']}]{path}[/{COLORS['path']}]"

    def format_url(self, url: str) -> str:
        """Format a URL with rich color styling."""
        if not self.enable_rich:
            return url
        return f"[{COLORS['url']}]{url}[/{COLORS['url']}]"

    def format_filename(self, filename: str) -> str:
        """Format a filename with rich color styling."""
        if not self.enable_rich:
            return filename
        return f"[{COLORS['filename']}]{filename}[/{COLORS['filename']}]"

    def format_time(self, time_str: str) -> str:
        """Format time with rich color styling."""
        if not self.enable_rich:
            return time_str
        return f"[{COLORS['time']}]{time_str}[/{COLORS['time']}]"

    def cache_stats(self, stats: Dict[str, Any]):
        """Display cache statistics with rich formatting."""
        if not self.enable_rich:
            print("\nCache Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return

        self.section("Cache Performance", style="bright_green")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bright_cyan", no_wrap=True, width=25)
        table.add_column("Value", style="bright_white", justify="right", width=15)

        for key, value in stats.items():
            icon = self._get_cache_icon(key)
            formatted_key = f"{icon} {key.replace('_', ' ').title()}"

            if isinstance(value, (int, float)) and "percentage" in key.lower():
                formatted_value = self.format_percentage(value)
            elif isinstance(value, int):
                formatted_value = self.format_number(value)
            elif isinstance(value, float):
                formatted_value = self.format_number(value)
            else:
                formatted_value = str(value)

            table.add_row(formatted_key, formatted_value)

        self.console.print(table)

    def prompt_cache_metrics(self, metrics: Dict[str, Any]):
        """Display prompt cache performance metrics with rich formatting."""
        if not metrics.get("prompt_cache_enabled", False):
            return

        if not self.enable_rich:
            print("\nPrompt Cache Performance:")
            for key, value in metrics.items():
                if key != "prompt_cache_enabled":
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            return

        self.section("🚀 Prompt Cache Performance Report", style="bright_magenta")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bright_cyan", no_wrap=True, width=30)
        table.add_column("Value", style="bright_white", justify="right", width=15)

        for key, value in metrics.items():
            if key == "prompt_cache_enabled":
                continue

            icon = EMOJIS.get("prompt_cache", "🚀")
            formatted_key = f"{icon} {key.replace('_', ' ').title()}"

            # Format values based on type
            if "percentage" in key.lower() or key.endswith("_rate"):
                formatted_value = self.format_percentage(value)
            elif "tokens" in key.lower():
                formatted_value = self.format_number(value, " tokens")
            elif "time" in key.lower():
                formatted_value = self.format_time(str(value))
            elif isinstance(value, (int, float)):
                formatted_value = self.format_number(value)
            else:
                formatted_value = str(value)

            table.add_row(formatted_key, formatted_value)

        self.console.print(table)

    def repository_analysis(self, repo_name: str, file_count: int):
        """Log repository analysis start with rich formatting."""
        self.info(
            f"Analyzing repository {self.format_filename(repo_name)} "
            f"with {self.format_number(file_count)} files",
            emoji="analyze",
        )

    def document_generation(
        self,
        doc_type: str,
        status: str = "generated",
        elapsed_time: Optional[float] = None,
        token_metrics: Optional[Dict[str, int]] = None,
    ):
        """Log document generation with rich formatting and detailed token metrics."""
        emoji_map = {
            "architecture": "architecture",
            "readme": "readme",
            "contributing": "contributing",
            "persistence": "persistence",
            "config": "config",
            "cicd": "cicd",
        }

        emoji = emoji_map.get(doc_type.lower(), "complete")

        if status == "generated":
            # Build success message with optional timing
            message = f"{doc_type.title()} documentation generated successfully"
            if elapsed_time is not None:
                message += f" in {elapsed_time:.1f}s"

            self.success(message, emoji=emoji)

            # Show detailed token usage if available
            if token_metrics:
                input_tokens = token_metrics.get("input_tokens", 0)
                output_tokens = token_metrics.get("output_tokens", 0)
                cache_read_tokens = token_metrics.get("cache_read_tokens", 0)
                cache_write_tokens = token_metrics.get("cache_write_tokens", 0)

                # Always show input/output tokens
                if input_tokens > 0 or output_tokens > 0:
                    self.info(
                        f"Input tokens: {self.format_number(input_tokens)}, Output tokens: {self.format_number(output_tokens)}",
                        emoji="token",
                    )

                # Show cache metrics if cache was used
                if cache_read_tokens > 0:
                    self.info(
                        f"Cache read: {self.format_number(cache_read_tokens)} tokens (cache hit)",
                        emoji="prompt_cache",
                    )
                if cache_write_tokens > 0:
                    self.info(
                        f"Cache write: {self.format_number(cache_write_tokens)} tokens (cache miss)",
                        emoji="prompt_cache",
                    )

        elif status == "skipped":
            self.info(f"Skipping {doc_type.lower()} documentation", emoji="skip")
        elif status == "failed":
            self.error(
                f"Failed to generate {doc_type.lower()} documentation", emoji="error"
            )

    def github_operation(self, operation: str, details: str):
        """Log GitHub operations with rich formatting."""
        self.info(f"GitHub: {operation} - {details}", emoji="github")

    def _get_cache_icon(self, metric: str) -> str:
        """Get an appropriate icon for a cache metric."""
        metric_lower = metric.lower()
        if "hit" in metric_lower:
            return EMOJIS["success"]
        elif "miss" in metric_lower:
            return EMOJIS["warning"]
        elif "time" in metric_lower:
            return EMOJIS["timer"]
        elif "token" in metric_lower:
            return EMOJIS["token"]
        else:
            return EMOJIS["cache"]


# Singleton instance
_visual_logger: Optional[VisualLogger] = None


def get_visual_logger(
    name: str = "ai-codebase-scribe",
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    enable_rich: bool = True,
) -> VisualLogger:
    """Get or create the visual logger instance."""
    global _visual_logger
    if _visual_logger is None:
        _visual_logger = VisualLogger(name, log_file, log_level, enable_rich)
    return _visual_logger


def setup_visual_logging(
    debug: bool = False,
    log_to_file: bool = True,
    quiet: bool = False,
    enable_rich: bool = True,
) -> VisualLogger:
    """Set up visual logging with the same interface as the original setup_logging."""
    # Determine log levels
    "DEBUG" if debug else "INFO"
    console_log_level = "WARNING" if quiet else ("DEBUG" if debug else "INFO")

    # Create log file path if needed
    log_file = None
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        import time

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = str(log_dir / f"readme_generator_{timestamp}.log")

    # Create and return visual logger
    return get_visual_logger(
        name="ai-codebase-scribe",
        log_file=log_file,
        log_level=console_log_level,
        enable_rich=enable_rich,
    )
