#!/usr/bin/env python3

"""
Configuration class for codebase-scribe.
"""

import dataclasses
import logging
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
class LargeRepoConfig:
    """Configuration for large repository handling."""
    threshold: int = 450
    max_files: int = 1000
    collapsible_tree: bool = True
    enhanced_sampling: bool = True
    files_per_component: int = 10
    smart_prioritization: bool = True
    verbose_logging: bool = True
    batch_processing: bool = True
    time_limit_minutes: int = 45
    cache_only_mode: bool = True
    skip_docs_on_partial: bool = True
    create_pr_on_batch: bool = False  # IMPORTANT: False for open-source (requires GitHub token)
    batch_pr_branch: str = "batch-processing/cache-update"


@dataclass
class PersistenceSupportedTechnologiesConfig:
    """Configuration for supported persistence technologies."""
    flyway: bool = True
    efcore: bool = True
    prisma: bool = True
    hibernate: bool = True
    django: bool = True
    rails: bool = True
    sequelize: bool = True
    alembic: bool = True


@dataclass
class PersistenceAnalysisConfig:
    """Configuration for persistence analysis limits."""
    max_migration_files: int = 50
    max_migration_size_bytes: int = 100000
    analysis_timeout_seconds: int = 300
    max_concurrent_analyses: int = 5


@dataclass
class PersistenceDocumentationConfig:
    """Configuration for persistence documentation generation."""
    include_table_details: bool = True
    include_relationships: bool = True
    include_indexes: bool = True
    include_migration_history: bool = True
    max_tables_in_doc: int = 100


@dataclass
class PersistenceConfig:
    """Configuration for persistence layer detection and documentation."""
    enabled: bool = True
    generate_doc: bool = True
    output_file: str = "docs/PERSISTENCE.md"
    detection_threshold: float = 0.2
    supported_technologies: PersistenceSupportedTechnologiesConfig = field(
        default_factory=PersistenceSupportedTechnologiesConfig
    )
    analysis: PersistenceAnalysisConfig = field(
        default_factory=PersistenceAnalysisConfig
    )
    documentation: PersistenceDocumentationConfig = field(
        default_factory=PersistenceDocumentationConfig
    )


@dataclass
class InstallationConfig:
    """Configuration for installation guide generation."""
    enabled: bool = True
    output_file: str = "docs/INSTALLATION.md"


@dataclass
class UsageConfig:
    """Configuration for usage guide generation."""
    enabled: bool = True
    output_file: str = "docs/USAGE.md"


@dataclass
class TroubleshootingConfig:
    """Configuration for troubleshooting guide generation."""
    enabled: bool = True
    output_file: str = "docs/TROUBLESHOOTING.md"


@dataclass
class ContributingConfig:
    """Configuration for contributing guide generation."""
    enabled: bool = True
    output_file: str = "CONTRIBUTING.md"


@dataclass
class ReadmeRefactorConfig:
    """Configuration for README refactoring after documentation splitting."""
    enabled: bool = True
    keep_brief_overview: bool = True
    add_navigation_section: bool = True


@dataclass
class BedrockConfig(LLMProviderConfig):
    """Configuration for Bedrock LLM provider."""
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    region: str = "us-east-1"
    concurrency: int = 3
    timeout: int = 60
    max_tokens: int = 2048
    retries: int = 5
    retry_delay: float = 2.0
    verify_ssl: bool = False
    temperature: float = 0.0
    # Fallback configuration
    fallback_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    enable_fallback: bool = True
    throttling_retry_delay: float = 30.0
    throttling_max_retries: int = 5
    # Output token limits
    max_output_tokens_architecture: int = 32768
    max_output_tokens_persistence: int = 32768
    # Prompt caching configuration
    enable_prompt_caching: bool = True
    cache_min_tokens: int = 1024
    cache_ttl_minutes: int = 5
    cache_strategy: str = "balanced"
    # Extended context configuration (1M tokens)
    extended_context_enabled: bool = True
    extended_context_beta_header: str = "context-1m-2025-08-07"

    def __post_init__(self):
        """Validate Bedrock configuration."""
        if self.enable_prompt_caching:
            # Verify region supports prompt caching
            supported_regions = [
                "us-east-1", "us-west-2",
                "eu-west-1", "eu-west-3", "eu-central-1", "eu-north-1",
                "ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
                "ca-central-1", "sa-east-1"
            ]
            if self.region not in supported_regions:
                logging.warning(
                    f"PROMPT_CACHE_WARNING: Prompt caching may not be available in region {self.region}"
                )

            # Verify model supports caching
            supported_models = [
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "anthropic.claude-3-5-haiku-20241022-v2:0",
                "anthropic.claude-3-7-sonnet-20250109-v2:0",
                "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            ]
            if self.model_id not in supported_models:
                logging.warning(
                    f"PROMPT_CACHE_WARNING: Prompt caching may not be supported for model {self.model_id}"
                )

        # Validate cache strategy
        valid_strategies = ["conservative", "balanced", "aggressive"]
        if self.cache_strategy not in valid_strategies:
            logging.warning(
                f"PROMPT_CACHE_WARNING: Invalid cache strategy '{self.cache_strategy}', defaulting to 'balanced'"
            )
            self.cache_strategy = "balanced"


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

    # Early exit settings
    exit_on_docs_only_changes: bool = True

    # Repository settings
    github_repo_id: Optional[str] = None

    # Blacklist settings
    blacklist: BlacklistConfig = field(default_factory=BlacklistConfig)

    # Cache settings
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Large repository settings
    large_repo: LargeRepoConfig = field(default_factory=LargeRepoConfig)

    # Persistence layer settings
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    # Documentation generator settings
    installation: InstallationConfig = field(default_factory=InstallationConfig)
    usage: UsageConfig = field(default_factory=UsageConfig)
    troubleshooting: TroubleshootingConfig = field(default_factory=TroubleshootingConfig)
    contributing: ContributingConfig = field(default_factory=ContributingConfig)
    readme_refactor: ReadmeRefactorConfig = field(default_factory=ReadmeRefactorConfig)

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
        # Debug logging
        logging.debug("Creating ScribeConfig from dictionary")

        config = cls()

        # Set general settings
        config.debug = config_dict.get('debug', False)
        config.test_mode = config_dict.get('test_mode', False)
        config.no_cache = config_dict.get('no_cache', False)
        config.optimize_order = config_dict.get('optimize_order', False)
        config.template_path = config_dict.get('template_path')
        config.github_repo_id = config_dict.get('github_repo_id')

        # Set early exit settings
        if 'exit_on_docs_only_changes' in config_dict:
            config.exit_on_docs_only_changes = config_dict['exit_on_docs_only_changes']

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

            logging.debug(f"Created CacheConfig with location: {config.cache.location}")

        # Set large repository settings
        if 'large_repo' in config_dict:
            large_repo_data = config_dict['large_repo']
            config.large_repo = LargeRepoConfig(**{
                k: v for k, v in large_repo_data.items()
                if k in LargeRepoConfig.__dataclass_fields__
            })

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
                concurrency=bedrock_dict.get('concurrency', 3),
                model_id=bedrock_dict.get('model_id', 'us.anthropic.claude-sonnet-4-20250514-v1:0'),
                region=bedrock_dict.get('region', 'us-east-1'),
                timeout=bedrock_dict.get('timeout', 60),
                max_tokens=bedrock_dict.get('max_tokens', 2048),
                retries=bedrock_dict.get('retries', 5),
                retry_delay=bedrock_dict.get('retry_delay', 2.0),
                verify_ssl=bedrock_dict.get('verify_ssl', False),
                temperature=bedrock_dict.get('temperature', 0.0),
                fallback_model_id=bedrock_dict.get(
                    'fallback_model_id', 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
                ),
                enable_fallback=bedrock_dict.get('enable_fallback', True),
                throttling_retry_delay=bedrock_dict.get('throttling_retry_delay', 30.0),
                throttling_max_retries=bedrock_dict.get('throttling_max_retries', 5),
                max_output_tokens_architecture=bedrock_dict.get('max_output_tokens_architecture', 32768),
                max_output_tokens_persistence=bedrock_dict.get('max_output_tokens_persistence', 32768),
                enable_prompt_caching=bedrock_dict.get('enable_prompt_caching', True),
                cache_min_tokens=bedrock_dict.get('cache_min_tokens', 1024),
                cache_ttl_minutes=bedrock_dict.get('cache_ttl_minutes', 5),
                cache_strategy=bedrock_dict.get('cache_strategy', 'balanced'),
                extended_context_enabled=bedrock_dict.get('extended_context_enabled', True),
                extended_context_beta_header=bedrock_dict.get(
                    'extended_context_beta_header', 'context-1m-2025-08-07'
                ),
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

            logging.debug("Loaded templates configuration")

        # Set persistence settings
        if 'persistence' in config_dict:
            p_data = config_dict['persistence']
            p_config = PersistenceConfig()
            for k, v in p_data.items():
                if k == 'supported_technologies' and isinstance(v, dict):
                    p_config.supported_technologies = PersistenceSupportedTechnologiesConfig(**{
                        kk: vv for kk, vv in v.items()
                        if kk in PersistenceSupportedTechnologiesConfig.__dataclass_fields__
                    })
                elif k == 'analysis' and isinstance(v, dict):
                    p_config.analysis = PersistenceAnalysisConfig(**{
                        kk: vv for kk, vv in v.items()
                        if kk in PersistenceAnalysisConfig.__dataclass_fields__
                    })
                elif k == 'documentation' and isinstance(v, dict):
                    p_config.documentation = PersistenceDocumentationConfig(**{
                        kk: vv for kk, vv in v.items()
                        if kk in PersistenceDocumentationConfig.__dataclass_fields__
                    })
                elif k in PersistenceConfig.__dataclass_fields__:
                    setattr(p_config, k, v)
            config.persistence = p_config
            logging.debug(f"Loaded persistence configuration - enabled: {config.persistence.enabled}")

        # Set installation documentation settings
        if 'installation' in config_dict:
            config.installation = InstallationConfig(**{
                k: v for k, v in config_dict['installation'].items()
                if k in InstallationConfig.__dataclass_fields__
            })
            logging.debug(f"Loaded installation configuration - enabled: {config.installation.enabled}")

        # Set usage documentation settings
        if 'usage' in config_dict:
            config.usage = UsageConfig(**{
                k: v for k, v in config_dict['usage'].items()
                if k in UsageConfig.__dataclass_fields__
            })
            logging.debug(f"Loaded usage configuration - enabled: {config.usage.enabled}")

        # Set troubleshooting documentation settings
        if 'troubleshooting' in config_dict:
            config.troubleshooting = TroubleshootingConfig(**{
                k: v for k, v in config_dict['troubleshooting'].items()
                if k in TroubleshootingConfig.__dataclass_fields__
            })
            logging.debug(f"Loaded troubleshooting configuration - enabled: {config.troubleshooting.enabled}")

        # Set contributing documentation settings
        if 'contributing' in config_dict:
            config.contributing = ContributingConfig(**{
                k: v for k, v in config_dict['contributing'].items()
                if k in ContributingConfig.__dataclass_fields__
            })
            logging.debug(f"Loaded contributing configuration - enabled: {config.contributing.enabled}")

        # Set README refactor settings
        if 'readme_refactor' in config_dict:
            config.readme_refactor = ReadmeRefactorConfig(**{
                k: v for k, v in config_dict['readme_refactor'].items()
                if k in ReadmeRefactorConfig.__dataclass_fields__
            })
            logging.debug(f"Loaded readme_refactor configuration - enabled: {config.readme_refactor.enabled}")

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
            'exit_on_docs_only_changes': self.exit_on_docs_only_changes,
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
            'large_repo': dataclasses.asdict(self.large_repo),
            'persistence': dataclasses.asdict(self.persistence),
            'installation': dataclasses.asdict(self.installation),
            'usage': dataclasses.asdict(self.usage),
            'troubleshooting': dataclasses.asdict(self.troubleshooting),
            'contributing': dataclasses.asdict(self.contributing),
            'readme_refactor': dataclasses.asdict(self.readme_refactor),
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
                'timeout': self.bedrock.timeout,
                'max_tokens': self.bedrock.max_tokens,
                'retries': self.bedrock.retries,
                'retry_delay': self.bedrock.retry_delay,
                'verify_ssl': self.bedrock.verify_ssl,
                'temperature': self.bedrock.temperature,
                'fallback_model_id': self.bedrock.fallback_model_id,
                'enable_fallback': self.bedrock.enable_fallback,
                'throttling_retry_delay': self.bedrock.throttling_retry_delay,
                'throttling_max_retries': self.bedrock.throttling_max_retries,
                'max_output_tokens_architecture': self.bedrock.max_output_tokens_architecture,
                'max_output_tokens_persistence': self.bedrock.max_output_tokens_persistence,
                'enable_prompt_caching': self.bedrock.enable_prompt_caching,
                'cache_min_tokens': self.bedrock.cache_min_tokens,
                'cache_ttl_minutes': self.bedrock.cache_ttl_minutes,
                'cache_strategy': self.bedrock.cache_strategy,
                'extended_context_enabled': self.bedrock.extended_context_enabled,
                'extended_context_beta_header': self.bedrock.extended_context_beta_header,
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
