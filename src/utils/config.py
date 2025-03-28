"""
Configuration management module for the codebase-scribe application.

This module provides functionality to load, merge, and access configuration settings
from YAML files, with support for environment variable overrides and default values.

Configuration Schema:
    - llm_provider: The LLM provider to use ('ollama' or 'bedrock')
    - debug: Enable debug logging
    - ollama: Ollama-specific configuration
        - base_url: Ollama API base URL
        - max_tokens: Maximum tokens for generation
        - retries: Number of retries on failure
        - retry_delay: Delay between retries in seconds
        - timeout: Request timeout in seconds
        - concurrency: Number of concurrent requests
        - temperature: Temperature for generation (0.0 = deterministic)
    - bedrock: AWS Bedrock-specific configuration
        - region: AWS region
        - model_id: Bedrock model ID
        - max_tokens: Maximum tokens for generation
        - retries: Number of retries on failure
        - retry_delay: Delay between retries in seconds
        - timeout: Request timeout in seconds
        - verify_ssl: Whether to verify SSL certificates
        - concurrency: Number of concurrent requests
        - temperature: Temperature for generation (0.0 = deterministic)
    - cache: Cache configuration
        - enabled: Whether caching is enabled
        - directory: Cache directory name
        - location: Cache location ('repo' or 'home')
        - hash_algorithm: Hash algorithm for file content hashing ('md5', 'sha1', or 'sha256')
        - global_directory: Directory name for global cache when location is 'home'
    - optimize_order: Use LLM to determine optimal file processing order
    - preserve_existing: Preserve and enhance existing documentation
    - no_cache: Disable caching via command line
    - test_mode: Process only first 5 files for testing
    - blacklist: Files to exclude from processing
        - extensions: File extensions to exclude
        - path_patterns: Path patterns to exclude
    - templates: Template definitions
        - prompts: Prompt templates
        - docs: Documentation templates
"""

from pathlib import Path
import yaml
from typing import Dict, Any, Optional, TypedDict, List, Union, Literal
import os
import logging
import json

# Environment variable constants
ENV_LLM_PROVIDER = 'LLM_PROVIDER'
ENV_DEBUG = 'DEBUG'
ENV_AWS_REGION = 'AWS_REGION'
ENV_AWS_BEDROCK_MODEL_ID = 'AWS_BEDROCK_MODEL_ID'
ENV_AWS_VERIFY_SSL = 'AWS_VERIFY_SSL'
ENV_CACHE_ENABLED = 'CACHE_ENABLED'
ENV_CACHE_HASH_ALGORITHM = 'CACHE_HASH_ALGORITHM'
ENV_CACHE_GLOBAL_DIRECTORY = 'CACHE_GLOBAL_DIRECTORY'

# Type definitions for configuration
class OllamaConfigDict(TypedDict):
    base_url: str
    max_tokens: int
    retries: int
    retry_delay: float
    timeout: int
    concurrency: int
    temperature: float

class BedrockConfigDict(TypedDict):
    region: str
    model_id: str
    max_tokens: int
    retries: int
    retry_delay: float
    timeout: int
    verify_ssl: bool
    concurrency: int
    temperature: float

class CacheConfigDict(TypedDict):
    enabled: bool
    directory: str
    location: Literal['repo', 'home']
    hash_algorithm: str
    global_directory: str

class BlacklistConfigDict(TypedDict):
    extensions: List[str]
    path_patterns: List[str]

class TemplateConfigDict(TypedDict):
    prompts: Dict[str, str]
    docs: Dict[str, str]

class ConfigDict(TypedDict):
    llm_provider: str
    debug: bool
    ollama: OllamaConfigDict
    bedrock: BedrockConfigDict
    cache: CacheConfigDict
    optimize_order: bool
    preserve_existing: bool
    no_cache: bool
    test_mode: bool
    blacklist: BlacklistConfigDict
    templates: TemplateConfigDict

# Default configuration with type annotations
DEFAULT_CONFIG: ConfigDict = {
    # LLM provider configuration
    'llm_provider': 'ollama',  # Default to ollama
    'debug': False,  # Default debug setting
    
    'ollama': {
        'base_url': 'http://localhost:11434',
        'max_tokens': 4096,
        'retries': 3,
        'retry_delay': 1.0,
        'timeout': 30,
        'concurrency': 1,  # Default to sequential processing
        'temperature': 0.0  # Default to deterministic output
    },
    
    'bedrock': {
        'region': 'us-east-1',
        'model_id': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'max_tokens': 8192,
        'retries': 3,
        'retry_delay': 1.0,
        'timeout': 120,  # Default to 2 minutes
        'verify_ssl': True,
        'concurrency': 5,  # Default to moderate concurrency
        'temperature': 0.0  # Default to deterministic output
    },
    
    'cache': {
        'enabled': True,
        'directory': '.cache',
        'location': 'home',  # 'repo' (in target repository) or 'home' (in user's home directory)
        'hash_algorithm': 'md5',  # Hash algorithm to use for file content hashing (md5, sha1, or sha256)
        'global_directory': '.readme_generator_cache'  # Directory name for global cache when location is "home"
    },
    
    # Processing options
    'optimize_order': False,
    'preserve_existing': True,
    'no_cache': False,  # New option to disable cache via command line
    'test_mode': False,
    
    'blacklist': {
        'extensions': ['.txt', '.log'],
        'path_patterns': [
            '/temp/',
            '/cache/',
            '/node_modules/',
            '/__pycache__/',
            '/wwwroot/',
            '^aql/',
            'aql/',
            '/aql/'
        ]
    },
    
    'templates': {
        'prompts': {
            'file_summary': """
Analyze the following code file and provide a clear, concise summary:
File: {file_path}
Type: {file_type}
Context: {context}

Code:
{code}
""".strip(),
            'project_overview': """
Generate a comprehensive overview for:
Project: {project_name}
Files: {file_count}
Components: {key_components}
""".strip(),
            'enhance_documentation': """
You are enhancing an existing {doc_type} file.

EXISTING CONTENT:
{existing_content}

REPOSITORY ANALYSIS:
{analysis}

Your task is to create the best possible documentation by intelligently combining the existing content with new insights from the repository analysis.

Guidelines:
1. Preserve valuable information from the existing content, especially specific implementation details, configuration examples, and custom instructions.
2. Feel free to reorganize the document structure to improve clarity and flow.
3. Remove outdated, redundant, or incorrect information.
4. Add missing information and technical details based on the repository analysis.
5. Ensure proper markdown formatting with consistent header hierarchy.
6. Maintain code snippets and examples, updating them only if they're incorrect.
7. If the existing content has a specific tone or style, try to maintain it.

Return a completely restructured document that represents the best possible documentation for this codebase, combining the strengths of the existing content with new insights.
""".strip()
        },
        'docs': {
            'readme': """
# {project_name}

{project_overview}

## Documentation

{usage}

## Development

{contributing}

## License

{license}
""".strip()
        }
    }
}

class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass

class ConfigManager:
    """Manages configuration with defaults and custom overrides."""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
    
    def __getitem__(self, key):
        """Allow dictionary-like access to config."""
        return self.config[key]
    
    def get(self, key, default=None):
        """
        Get a configuration value with a default fallback.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        return self.config.get(key, default)
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load and merge custom configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Merged configuration dictionary
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the configuration file contains invalid YAML
        """
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            with open(config_path) as f:
                custom_config = yaml.safe_load(f)
            
            if custom_config is None:
                logging.warning(f"Empty or invalid config file: {config_path}")
                custom_config = {}
                
            # Merge with default configuration
            merged_config = self._deep_merge(DEFAULT_CONFIG, custom_config)
            
            # Validate the configuration
            self._validate_config(merged_config)
            
            # Apply environment variable overrides
            merged_config = self._apply_env_overrides(merged_config)
            
            return merged_config
            
        except FileNotFoundError as e:
            logging.error(f"Configuration file not found: {config_path}")
            logging.info("Using default configuration")
            # Apply environment variable overrides to default config
            return self._apply_env_overrides(DEFAULT_CONFIG.copy())
        except yaml.YAMLError as e:
            logging.error(f"Invalid YAML in config file {config_path}: {str(e)}")
            logging.info("Using default configuration")
            # Apply environment variable overrides to default config
            return self._apply_env_overrides(DEFAULT_CONFIG.copy())
        except ConfigValidationError as e:
            logging.error(f"Configuration validation error: {str(e)}")
            logging.info("Using default configuration")
            # Apply environment variable overrides to default config
            return self._apply_env_overrides(DEFAULT_CONFIG.copy())
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {str(e)}")
            logging.info("Using default configuration")
            # Apply environment variable overrides to default config
            return self._apply_env_overrides(DEFAULT_CONFIG.copy())
    
    def _deep_merge(self, default: dict, custom: dict) -> dict:
        """
        Deep merge two dictionaries, with custom values taking precedence.
        
        Args:
            default: Default dictionary
            custom: Custom dictionary with overrides
            
        Returns:
            Merged dictionary
        """
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override or add the custom value
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self, config: dict) -> dict:
        """
        Apply environment variable overrides to config.
        
        Args:
            config: Configuration dictionary to apply overrides to
            
        Returns:
            Updated configuration dictionary
        """
        # LLM provider override
        if os.getenv(ENV_LLM_PROVIDER):
            config['llm_provider'] = os.getenv(ENV_LLM_PROVIDER)
        
        # Debug mode override
        if os.getenv(ENV_DEBUG):
            config['debug'] = os.getenv(ENV_DEBUG).lower() in ('true', '1', 'yes')
        
        # AWS region override
        if os.getenv(ENV_AWS_REGION):
            config['bedrock']['region'] = os.getenv(ENV_AWS_REGION)
        
        # Bedrock model ID override
        if os.getenv(ENV_AWS_BEDROCK_MODEL_ID):
            config['bedrock']['model_id'] = os.getenv(ENV_AWS_BEDROCK_MODEL_ID)
        
        # SSL verification override
        if os.getenv(ENV_AWS_VERIFY_SSL):
            config['bedrock']['verify_ssl'] = os.getenv(ENV_AWS_VERIFY_SSL).lower() in ('true', '1', 'yes')
        
        # Cache enabled override
        if os.getenv(ENV_CACHE_ENABLED):
            config['cache']['enabled'] = os.getenv(ENV_CACHE_ENABLED).lower() in ('true', '1', 'yes')
            
        # Cache hash algorithm override
        if os.getenv(ENV_CACHE_HASH_ALGORITHM):
            hash_algo = os.getenv(ENV_CACHE_HASH_ALGORITHM)
            if hash_algo in ['md5', 'sha1', 'sha256']:
                config['cache']['hash_algorithm'] = hash_algo
                
        # Cache global directory override
        if os.getenv(ENV_CACHE_GLOBAL_DIRECTORY):
            config['cache']['global_directory'] = os.getenv(ENV_CACHE_GLOBAL_DIRECTORY)
        
        return config
    
    def _validate_config(self, config: dict) -> None:
        """
        Validate configuration values.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate LLM provider
        if config.get('llm_provider') not in ['ollama', 'bedrock']:
            raise ConfigValidationError(f"Invalid LLM provider: {config.get('llm_provider')}. Must be 'ollama' or 'bedrock'.")
        
        # Validate Ollama config
        ollama_config = config.get('ollama', {})
        if not isinstance(ollama_config, dict):
            raise ConfigValidationError("Ollama configuration must be a dictionary.")
        
        if not isinstance(ollama_config.get('base_url', ''), str):
            raise ConfigValidationError("Ollama base_url must be a string.")
        
        if not isinstance(ollama_config.get('max_tokens', 0), int) or ollama_config.get('max_tokens', 0) <= 0:
            raise ConfigValidationError("Ollama max_tokens must be a positive integer.")
        
        # Validate Bedrock config
        bedrock_config = config.get('bedrock', {})
        if not isinstance(bedrock_config, dict):
            raise ConfigValidationError("Bedrock configuration must be a dictionary.")
        
        if not isinstance(bedrock_config.get('region', ''), str):
            raise ConfigValidationError("Bedrock region must be a string.")
        
        if not isinstance(bedrock_config.get('model_id', ''), str):
            raise ConfigValidationError("Bedrock model_id must be a string.")
            
        if not isinstance(bedrock_config.get('max_tokens', 0), int) or bedrock_config.get('max_tokens', 0) <= 0:
            raise ConfigValidationError("Bedrock max_tokens must be a positive integer.")
        
        # Validate cache config
        cache_config = config.get('cache', {})
        if not isinstance(cache_config, dict):
            raise ConfigValidationError("Cache configuration must be a dictionary.")
        
        if not isinstance(cache_config.get('enabled', True), bool):
            raise ConfigValidationError("Cache enabled must be a boolean.")
        
        if cache_config.get('location') not in ['repo', 'home']:
            raise ConfigValidationError(f"Invalid cache location: {cache_config.get('location')}. Must be 'repo' or 'home'.")
            
        if cache_config.get('hash_algorithm') not in ['md5', 'sha1', 'sha256']:
            raise ConfigValidationError(f"Invalid hash algorithm: {cache_config.get('hash_algorithm')}. Must be 'md5', 'sha1', or 'sha256'.")
            
        if not isinstance(cache_config.get('global_directory', ''), str):
            raise ConfigValidationError("Cache global_directory must be a string.")
    
    def get_template(self, category: str, name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a specific template from config and optionally format it with context.
        
        Args:
            category: Template category ('prompts' or 'docs')
            name: Template name
            context: Optional context for formatting
            
        Returns:
            The template string, optionally formatted with context
        """
        if category not in self.config['templates'] or name not in self.config['templates'][category]:
            logging.warning(f"Template {category}/{name} not found in config")
            return f"Template {category}/{name} not found"
        
        template = self.config['templates'][category][name]
        
        if context:
            try:
                return template.format(**context)
            except KeyError as e:
                logging.warning(f"Missing key in template formatting: {e}")
                return template
        
        return template
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama-specific configuration."""
        return self.config['ollama']
    
    def get_bedrock_config(self) -> Dict[str, Any]:
        """Get Bedrock-specific configuration."""
        return self.config['bedrock']
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache-specific configuration."""
        return self.config['cache']
    
    def dump_config(self, format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        Dump the current configuration.
        
        Args:
            format: Output format ('dict', 'yaml', or 'json')
            
        Returns:
            Configuration as a dictionary, YAML string, or JSON string
        """
        if format == 'yaml':
            return yaml.dump(self.config, default_flow_style=False)
        elif format == 'json':
            return json.dumps(self.config, indent=2)
        else:
            return self.config

def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_manager = ConfigManager(config_path)
    return config_manager.config