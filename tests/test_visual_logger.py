#!/usr/bin/env python3

"""
Tests for src/utils/visual_logger.py

Verifies that VisualLogger can be imported and instantiated, that
get_visual_logger() returns a logger instance, and that
setup_visual_logging() completes without raising.
"""

import logging
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.visual_logger import VisualLogger, get_visual_logger, setup_visual_logging


class TestVisualLogger:
    """Tests for the VisualLogger class."""

    def test_import(self):
        """Test that VisualLogger can be imported."""
        assert VisualLogger is not None

    def test_instantiation_with_defaults(self):
        """Test that VisualLogger can be instantiated with default arguments."""
        logger = VisualLogger(name="test-visual-logger-defaults", enable_rich=False)
        assert logger is not None
        assert isinstance(logger.logger, logging.Logger)

    def test_instantiation_with_rich_disabled(self):
        """Test that VisualLogger works correctly when rich formatting is disabled."""
        logger = VisualLogger(name="test-visual-logger-no-rich", enable_rich=False)
        assert logger.enable_rich is False

    def test_instantiation_with_rich_enabled(self):
        """Test that VisualLogger can be instantiated with rich formatting enabled."""
        logger = VisualLogger(name="test-visual-logger-rich", enable_rich=True)
        assert logger.enable_rich is True

    def test_log_level_respected(self):
        """Test that the specified log level is applied."""
        logger = VisualLogger(
            name="test-visual-logger-level",
            log_level="WARNING",
            enable_rich=False,
        )
        assert logger.logger.level == logging.WARNING

    def test_success_method_does_not_raise(self):
        """Test that success() does not raise."""
        logger = VisualLogger(name="test-visual-logger-success", enable_rich=False)
        logger.success("Success message")

    def test_error_method_does_not_raise(self):
        """Test that error() does not raise."""
        logger = VisualLogger(name="test-visual-logger-error", enable_rich=False)
        logger.error("Error message")

    def test_warning_method_does_not_raise(self):
        """Test that warning() does not raise."""
        logger = VisualLogger(name="test-visual-logger-warning", enable_rich=False)
        logger.warning("Warning message")

    def test_info_method_does_not_raise(self):
        """Test that info() does not raise."""
        logger = VisualLogger(name="test-visual-logger-info", enable_rich=False)
        logger.info("Info message")

    def test_debug_method_does_not_raise(self):
        """Test that debug() does not raise."""
        logger = VisualLogger(name="test-visual-logger-debug", log_level="DEBUG", enable_rich=False)
        logger.debug("Debug message")

    def test_format_number_returns_string(self):
        """Test that format_number() returns a string."""
        logger = VisualLogger(name="test-visual-logger-fmt-num", enable_rich=False)
        result = logger.format_number(42)
        assert isinstance(result, str)
        assert "42" in result

    def test_format_percentage_returns_string(self):
        """Test that format_percentage() returns a string."""
        logger = VisualLogger(name="test-visual-logger-fmt-pct", enable_rich=False)
        result = logger.format_percentage(75)
        assert isinstance(result, str)
        assert "75" in result

    def test_format_path_returns_string(self):
        """Test that format_path() returns a string."""
        logger = VisualLogger(name="test-visual-logger-fmt-path", enable_rich=False)
        result = logger.format_path("/some/path/to/file.py")
        assert isinstance(result, str)
        assert "file.py" in result


class TestGetVisualLogger:
    """Tests for the get_visual_logger() factory function."""

    def test_returns_visual_logger_instance(self):
        """Test that get_visual_logger() returns a VisualLogger instance."""
        # Reset the module-level singleton so we get a fresh instance
        import src.utils.visual_logger as vl_module
        original = vl_module._visual_logger
        vl_module._visual_logger = None

        try:
            logger = get_visual_logger(name="test-get-logger", enable_rich=False)
            assert isinstance(logger, VisualLogger)
        finally:
            vl_module._visual_logger = original

    def test_returns_singleton(self):
        """Test that get_visual_logger() returns the same instance on repeated calls."""
        import src.utils.visual_logger as vl_module
        original = vl_module._visual_logger
        vl_module._visual_logger = None

        try:
            logger1 = get_visual_logger(name="test-singleton", enable_rich=False)
            logger2 = get_visual_logger(name="test-singleton", enable_rich=False)
            assert logger1 is logger2
        finally:
            vl_module._visual_logger = original


class TestSetupVisualLogging:
    """Tests for the setup_visual_logging() function."""

    def test_does_not_raise_with_defaults(self, tmp_path):
        """Test that setup_visual_logging() completes without raising."""
        # Reset singleton
        import src.utils.visual_logger as vl_module
        original = vl_module._visual_logger
        vl_module._visual_logger = None

        try:
            with patch('src.utils.visual_logger.Path.mkdir'):
                result = setup_visual_logging(
                    debug=False,
                    log_to_file=False,
                    quiet=False,
                    enable_rich=False,
                )
            assert isinstance(result, VisualLogger)
        finally:
            vl_module._visual_logger = original

    def test_does_not_raise_in_debug_mode(self):
        """Test that setup_visual_logging() works in debug mode."""
        import src.utils.visual_logger as vl_module
        original = vl_module._visual_logger
        vl_module._visual_logger = None

        try:
            result = setup_visual_logging(
                debug=True,
                log_to_file=False,
                quiet=False,
                enable_rich=False,
            )
            assert isinstance(result, VisualLogger)
        finally:
            vl_module._visual_logger = original
