#!/usr/bin/env python3

"""
Utilities for working with configuration in codebase-scribe.

This module provides functions to bridge between the old dictionary-based
configuration and the new ScribeConfig class-based configuration.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import warnings

from src.utils.config import load_config as load_config_dict
from src.utils.config_class import ScribeConfig


def load_config(config_path: Union[str, Path]) -> ScribeConfig:
    """
    Load configuration directly into ScribeConfig.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ScribeConfig instance
    """
    import logging
    
    try:
        # Debug logging
        logging.debug(f"Loading config from {config_path}")
        
        # Load config dictionary using the existing function
        config_dict = load_config_dict(config_path)
        
        # Create ScribeConfig directly from the loaded dictionary
        config = ScribeConfig.from_dict(config_dict)
        
        # Apply environment variable overrides
        config = apply_env_overrides(config)
        
        # Debug logging
        if hasattr(config, 'cache'):
            logging.debug(f"ScribeConfig cache location: {config.cache.location}")
        
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        return ScribeConfig()  # Return default config on error


def update_config_with_args(config: ScribeConfig, args: Any) -> ScribeConfig:
    """
    Update configuration with command-line arguments.
    
    Args:
        config: ScribeConfig instance
        args: Command-line arguments
        
    Returns:
        Updated ScribeConfig instance
    """
    import logging
    import copy
    
    # Debug logging
    logging.debug(f"Updating config with args")
    
    # Create a copy of the config
    new_config = copy.deepcopy(config)
    
    # Debug logging
    logging.debug(f"Before update - Cache location: {new_config.cache.location}")
    
    # Update with command-line arguments
    if hasattr(args, 'debug'):
        new_config.debug = args.debug
        logging.debug(f"Updated debug: {new_config.debug}")
    
    if hasattr(args, 'test_mode'):
        new_config.test_mode = args.test_mode
        logging.debug(f"Updated test_mode: {new_config.test_mode}")
    
    if hasattr(args, 'no_cache'):
        new_config.no_cache = args.no_cache
        if new_config.no_cache:
            new_config.cache.enabled = False
        logging.debug(f"Updated no_cache: {new_config.no_cache}")
    
    if hasattr(args, 'optimize_order'):
        new_config.optimize_order = args.optimize_order
        logging.debug(f"Updated optimize_order: {new_config.optimize_order}")
    
    if hasattr(args, 'llm_provider') and args.llm_provider:
        new_config.llm_provider = args.llm_provider
        logging.debug(f"Updated llm_provider: {new_config.llm_provider}")
    
    # Debug logging
    logging.debug(f"After update - Cache location: {new_config.cache.location}")
    
    return new_config


def config_to_dict(config: ScribeConfig) -> Dict[str, Any]:
    """
    Convert configuration to dictionary.
    
    DEPRECATED: This function is maintained for backward compatibility only.
    New code should use ScribeConfig.to_dict() directly.
    
    Args:
        config: ScribeConfig instance
        
    Returns:
        Configuration dictionary
    """
    warnings.warn(
        "config_to_dict is deprecated. Use ScribeConfig.to_dict() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return config.to_dict()


def dict_to_config(config_dict: Dict[str, Any]) -> ScribeConfig:
    """
    Convert dictionary to ScribeConfig.
    
    DEPRECATED: This function is maintained for backward compatibility only.
    New code should use ScribeConfig.from_dict() directly.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ScribeConfig instance
    """
    import logging
    warnings.warn(
        "dict_to_config is deprecated. Use ScribeConfig.from_dict() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    logging.debug("Converting dictionary to ScribeConfig")
    return ScribeConfig.from_dict(config_dict)


def apply_env_overrides(config: ScribeConfig) -> ScribeConfig:
    """
    Apply environment variable overrides to config.
    
    Args:
        config: ScribeConfig instance
        
    Returns:
        Updated ScribeConfig instance
    """
    import os
    import copy
    import logging
    
    # Create a copy of the config
    new_config = copy.deepcopy(config)
    
    # Environment variable constants
    ENV_LLM_PROVIDER = 'LLM_PROVIDER'
    ENV_DEBUG = 'DEBUG'
    ENV_AWS_REGION = 'AWS_REGION'
    ENV_AWS_BEDROCK_MODEL_ID = 'AWS_BEDROCK_MODEL_ID'
    ENV_AWS_VERIFY_SSL = 'AWS_VERIFY_SSL'
    ENV_CACHE_ENABLED = 'CACHE_ENABLED'
    ENV_CACHE_HASH_ALGORITHM = 'CACHE_HASH_ALGORITHM'
    ENV_CACHE_GLOBAL_DIRECTORY = 'CACHE_GLOBAL_DIRECTORY'
    
    # Apply environment variable overrides
    if os.getenv(ENV_LLM_PROVIDER):
        new_config.llm_provider = os.getenv(ENV_LLM_PROVIDER)
        logging.debug(f"Applied environment override for LLM provider: {new_config.llm_provider}")
    
    if os.getenv(ENV_DEBUG):
        new_config.debug = os.getenv(ENV_DEBUG).lower() in ('true', '1', 'yes')
        logging.debug(f"Applied environment override for debug: {new_config.debug}")
    
    if os.getenv(ENV_AWS_REGION):
        new_config.bedrock.region = os.getenv(ENV_AWS_REGION)
        logging.debug(f"Applied environment override for AWS region: {new_config.bedrock.region}")
    
    if os.getenv(ENV_AWS_BEDROCK_MODEL_ID):
        new_config.bedrock.model_id = os.getenv(ENV_AWS_BEDROCK_MODEL_ID)
        logging.debug(f"Applied environment override for Bedrock model ID: {new_config.bedrock.model_id}")
    
    if os.getenv(ENV_AWS_VERIFY_SSL):
        new_config.bedrock.verify_ssl = os.getenv(ENV_AWS_VERIFY_SSL).lower() in ('true', '1', 'yes')
        logging.debug(f"Applied environment override for verify SSL: {new_config.bedrock.verify_ssl}")
    
    if os.getenv(ENV_CACHE_ENABLED):
        new_config.cache.enabled = os.getenv(ENV_CACHE_ENABLED).lower() in ('true', '1', 'yes')
        logging.debug(f"Applied environment override for cache enabled: {new_config.cache.enabled}")
    
    if os.getenv(ENV_CACHE_HASH_ALGORITHM):
        hash_algo = os.getenv(ENV_CACHE_HASH_ALGORITHM)
        if hash_algo in ['md5', 'sha1', 'sha256']:
            new_config.cache.hash_algorithm = hash_algo
            logging.debug(f"Applied environment override for hash algorithm: {new_config.cache.hash_algorithm}")
    
    if os.getenv(ENV_CACHE_GLOBAL_DIRECTORY):
        new_config.cache.global_directory = os.getenv(ENV_CACHE_GLOBAL_DIRECTORY)
        logging.debug(f"Applied environment override for global directory: {new_config.cache.global_directory}")
    
    return new_config


def get_concurrency(config: ScribeConfig) -> int:
    """
    Get concurrency setting from configuration.
    
    Args:
        config: ScribeConfig instance
        
    Returns:
        Concurrency setting
    """
    return config.get_concurrency()