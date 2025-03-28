#!/usr/bin/env python3

"""
Custom exceptions for codebase-scribe.
"""


class ScribeError(Exception):
    """Base exception for all codebase-scribe errors."""
    pass


class ConfigurationError(ScribeError):
    """Exception raised for configuration errors."""
    pass


class RepositoryError(ScribeError):
    """Exception raised for repository-related errors."""
    pass


class FileProcessingError(ScribeError):
    """Exception raised for file processing errors."""
    pass


class LLMError(ScribeError):
    """Exception raised for LLM-related errors."""
    pass


class CacheError(ScribeError):
    """Exception raised for cache-related errors."""
    pass


class GitHubError(ScribeError):
    """Exception raised for GitHub-related errors."""
    pass


class DocumentationError(ScribeError):
    """Exception raised for documentation generation errors."""
    pass


class NetworkError(ScribeError):
    """Exception raised for network-related errors."""
    pass


class TimeoutError(ScribeError):
    """Exception raised for timeout errors."""
    pass