#!/usr/bin/env python3

"""
Configuration class for codebase-scribe.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

@dataclass
class PromptTemplatesConfig:
    """Configuration for prompt templates."""
    file_summary: str = """Analyze the following code file and provide a clear, concise summary:
File: {file_path}
Type: {file_type}
Context: {context}

Code:
{code}"""
    project_overview: str = """Generate a comprehensive overview for:
Project: {project_name}
Files: {file_count}
Components: {key_components}"""
    enhance_existing: str = """You are enhancing an existing {doc_type} file.

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

Return a completely restructured document that represents the best possible documentation for this codebase, combining the strengths of the existing content with new insights."""


@dataclass
class DocTemplatesConfig:
    """Configuration for documentation templates."""
    readme: str = """# {project_name}

{project_overview}

## Documentation

{usage}

## Development

{contributing}

## License

{license}"""


@dataclass
class TemplatesConfig:
    """Configuration for templates."""
    prompts: PromptTemplatesConfig = field(default_factory=PromptTemplatesConfig)
    docs: DocTemplatesConfig = field(default_factory=DocTemplatesConfig)


@dataclass
class BlacklistConfig:
    """Configuration for file and directory blacklisting."""
    extensions: List[str] = field(default_factory=lambda: ['.pyc', '.pyo', '.pyd'])
    path_patterns: List[str] = field(default_factory=lambda: ['__pycache__', '\\.git'])


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1048576  # Max cache size in bytes
    location: str = "home"  # Cache location ('repo' or 'home')
    directory: str = ".cache"  # Cache directory name
    global_directory: str = "readme_generator_cache"  # Global cache directory name (no dot to make it visible)
    hash_algorithm: str = "md5"  # Hash algorithm for file content hashing


@dataclass
class LLMProviderConfig:
    """Base configuration for LLM providers."""
    concurrency: int = 1


@dataclass
class OllamaConfig(LLMProviderConfig):
    """Configuration for Ollama LLM provider."""
    model: str = "llama2"
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    max_tokens: int = 2048
    retries: int = 5
    retry_delay: float = 2.0
    temperature: float = 0.5


@dataclass
class BedrockConfig(LLMProviderConfig):
    """Configuration for Bedrock LLM provider."""
    model_id: str = "test-model-id"
    region: str = "us-west-2"
    concurrency: int = 3
    timeout: int = 60
    max_tokens: int = 2048
    retries: int = 5
    retry_delay: float = 2.0
    verify_ssl: bool = False
    temperature: float = 0.0


@dataclass
class ScribeConfig:
    """Main configuration class for codebase-scribe."""
    # General settings
    debug: bool = False
    test_mode: bool = False
    no_cache: bool = False
    optimize_order: bool = False
    preserve_existing: bool = True
    template_path: Optional[str] = None
    
    # Repository settings
    github_repo_id: Optional[str] = None
    
    # Blacklist settings
    blacklist: BlacklistConfig = field(default_factory=BlacklistConfig)
    
    # Cache settings
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # LLM provider settings
    llm_provider: str = "ollama"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    
    # Templates settings
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ScribeConfig':
        """Create a ScribeConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            A ScribeConfig instance
        """
        import logging
        
        # Debug logging
        logging.debug(f"Creating ScribeConfig from dictionary")
        
        config = cls()
        
        # Set general settings
        config.debug = config_dict.get('debug', False)
        config.test_mode = config_dict.get('test_mode', False)
        config.no_cache = config_dict.get('no_cache', False)
        config.optimize_order = config_dict.get('optimize_order', False)
        config.template_path = config_dict.get('template_path')
        config.github_repo_id = config_dict.get('github_repo_id')
        
        # Set blacklist settings
        if 'blacklist' in config_dict:
            blacklist_dict = config_dict['blacklist']
            config.blacklist = BlacklistConfig(
                extensions=blacklist_dict.get('extensions', ['.pyc', '.pyo', '.pyd']),
                path_patterns=blacklist_dict.get('path_patterns', ['__pycache__', '\\.git'])
            )
        
        # Set cache settings
        if 'cache' in config_dict:
            cache_dict = config_dict['cache']
            
            # Debug logging
            logging.debug(f"Cache dictionary: {cache_dict}")
            logging.debug(f"Cache location from dictionary: {cache_dict.get('location', 'default')}")
            
            config.cache = CacheConfig(
                enabled=not config.no_cache,
                ttl=cache_dict.get('ttl', 3600),
                max_size=cache_dict.get('max_size', 1048576),
                location=cache_dict.get('location', 'home'),
                directory=cache_dict.get('directory', '.cache'),
                global_directory=cache_dict.get('global_directory', 'readme_generator_cache'),
                hash_algorithm=cache_dict.get('hash_algorithm', 'md5')
            )
            
            # Debug logging
            logging.debug(f"Created CacheConfig with location: {config.cache.location}")
        
        # Set LLM provider settings
        config.llm_provider = config_dict.get('llm_provider', 'ollama')
        
        # Set Ollama settings
        if 'ollama' in config_dict:
            ollama_dict = config_dict['ollama']
            config.ollama = OllamaConfig(
                concurrency=ollama_dict.get('concurrency', 1),
                model=ollama_dict.get('model', 'llama2'),
                base_url=ollama_dict.get('base_url', 'http://localhost:11434'),
                timeout=ollama_dict.get('timeout', 60)
            )
        
        # Set Bedrock settings
        if 'bedrock' in config_dict:
            bedrock_dict = config_dict['bedrock']
            config.bedrock = BedrockConfig(
                concurrency=bedrock_dict.get('concurrency', 1),
                model_id=bedrock_dict.get('model_id', 'anthropic.claude-v2'),
                region=bedrock_dict.get('region', 'us-east-1'),
                timeout=bedrock_dict.get('timeout', 120)
            )
        
        # Set Templates settings
        if 'templates' in config_dict:
            templates_dict = config_dict['templates']
            
            # Create PromptTemplatesConfig
            prompts_config = PromptTemplatesConfig()
            if 'prompts' in templates_dict:
                prompts_dict = templates_dict['prompts']
                if 'file_summary' in prompts_dict:
                    prompts_config.file_summary = prompts_dict['file_summary']
                if 'project_overview' in prompts_dict:
                    prompts_config.project_overview = prompts_dict['project_overview']
                if 'enhance_existing' in prompts_dict:
                    prompts_config.enhance_existing = prompts_dict['enhance_existing']
            
            # Create DocTemplatesConfig
            docs_config = DocTemplatesConfig()
            if 'docs' in templates_dict:
                docs_dict = templates_dict['docs']
                if 'readme' in docs_dict:
                    docs_config.readme = docs_dict['readme']
            
            # Set the templates config
            config.templates = TemplatesConfig(
                prompts=prompts_config,
                docs=docs_config
            )
            
            # Debug logging
            logging.debug(f"Loaded templates configuration")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ScribeConfig instance to a dictionary.
        
        Returns:
            A dictionary representation of the configuration
        """
        return {
            'debug': self.debug,
            'test_mode': self.test_mode,
            'no_cache': self.no_cache,
            'optimize_order': self.optimize_order,
            'template_path': self.template_path,
            'github_repo_id': self.github_repo_id,
            'blacklist': {
                'extensions': self.blacklist.extensions,
                'path_patterns': self.blacklist.path_patterns
            },
            'cache': {
                'enabled': self.cache.enabled,
                'ttl': self.cache.ttl,
                'max_size': self.cache.max_size,
                'location': self.cache.location,
                'directory': self.cache.directory,
                'global_directory': self.cache.global_directory,
                'hash_algorithm': self.cache.hash_algorithm
            },
            'llm_provider': self.llm_provider,
            'ollama': {
                'concurrency': self.ollama.concurrency,
                'model': self.ollama.model,
                'base_url': self.ollama.base_url,
                'timeout': self.ollama.timeout
            },
            'bedrock': {
                'concurrency': self.bedrock.concurrency,
                'model_id': self.bedrock.model_id,
                'region': self.bedrock.region,
                'timeout': self.bedrock.timeout
            },
            'templates': {
                'prompts': {
                    'file_summary': self.templates.prompts.file_summary,
                    'project_overview': self.templates.prompts.project_overview,
                    'enhance_existing': self.templates.prompts.enhance_existing
                },
                'docs': {
                    'readme': self.templates.docs.readme
                }
            }
        }
    
    def get_concurrency(self) -> int:
        """Get the concurrency setting based on the LLM provider.
        
        Returns:
            The concurrency setting
        """
        if self.llm_provider.lower() == 'bedrock':
            return self.bedrock.concurrency
        else:
            return self.ollama.concurrency
            
    def write_to_file(self, file_path: str) -> None:
        """Write the configuration to a YAML file.
        
        Args:
            file_path: Path to the file to write to
        """
        import yaml
        
        # Convert to dictionary
        config_dict = self.to_dict()
        
        # Write to file
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)