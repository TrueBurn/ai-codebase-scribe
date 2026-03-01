# Configuration System

This document provides detailed information about the configuration system used in the Codebase Scribe AI project.

## Overview

The configuration system is designed to be flexible and extensible, allowing users to customize the behavior of the application through a YAML configuration file. The system uses a class-based approach with `ScribeConfig` as the main configuration class.

The system supports:

- Default configuration values
- Custom overrides from YAML files
- Environment variable overrides
- Command-line argument overrides
- Type validation through Python type hints
- Configuration serialization and deserialization

## Configuration File

The default configuration file is `config.yaml` in the project root directory. You can specify a different configuration file using the `--config` command-line argument.

```bash
python codebase_scribe.py --repo ./my-project --config custom_config.yaml
```

## Configuration Schema

The configuration file uses YAML format. All sections are optional; unspecified keys fall back to their defaults.

### Top-Level Options

```yaml
llm_provider: "bedrock"  # "bedrock" or "ollama"
debug: false             # Enable verbose debug logging
test_mode: false         # Limit analysis to first 5 files (quick validation)
optimize_order: false    # Use LLM to determine optimal file processing order
preserve_existing: true  # Enhance existing docs instead of replacing them
exit_on_docs_only_changes: true  # Exit early when only documentation files changed
```

### Ollama

```yaml
ollama:
  base_url: "http://localhost:11434"
  max_tokens: 4096
  retries: 3
  retry_delay: 1.0
  timeout: 30
  concurrency: 1      # Number of concurrent requests (default 1 = sequential)
  temperature: 0.1    # Generation temperature (0.0 = deterministic)
```

### AWS Bedrock

```yaml
bedrock:
  region: "us-east-1"
  model_id: "us.anthropic.claude-sonnet-4-20250514-v1:0"
  max_tokens: 8192
  timeout: 180          # Request timeout in seconds
  retries: 5
  concurrency: 10
  verify_ssl: true      # Set false to bypass SSL for corporate proxies
  temperature: 0.1

  # Throttling fallback
  # When a ThrottlingException is received, the client automatically retries
  # using fallback_model_id (typically a smaller, less constrained model).
  fallback_model_id: "us.anthropic.claude-haiku-4-5-20251001-v1:0"
  enable_fallback: true
  throttling_retry_delay: 30.0   # Seconds to wait before fallback attempt
  throttling_max_retries: 5      # Maximum fallback retry attempts

  # Per-task output token limits
  # Override max_tokens for specific generation tasks that require longer output.
  max_output_tokens_architecture: 32768
  max_output_tokens_persistence: 32768

  # AWS Bedrock prompt caching
  # Caches stable portions of large prompts (project structure, tech report, etc.)
  # to reduce input token costs and latency on repeated calls.
  enable_prompt_caching: true
  cache_min_tokens: 1024         # Minimum token count for a component to be cached
  cache_ttl_minutes: 5           # Cache time-to-live in minutes
  cache_strategy: "balanced"     # "conservative" | "balanced" | "aggressive"
                                 # conservative: cache_min_tokens * 2
                                 # balanced:     cache_min_tokens
                                 # aggressive:   cache_min_tokens / 2

  # Extended context (1M token window via Bedrock beta header)
  extended_context_enabled: true
  extended_context_beta_header: "context-1m-2025-08-07"
```

### Cache (File Cache)

This is the local file-level cache used to avoid re-summarising unchanged source files. It is separate from the AWS Bedrock prompt cache.

```yaml
cache:
  enabled: true
  directory: ".cache"
  location: "home"       # "repo" (inside target repo) or "home" (user home dir)
  hash_algorithm: "md5"  # "md5", "sha1", or "sha256"
  global_directory: "readme_generator_cache"  # Directory name when location="home"
```

### Processing Options

```yaml
optimize_order: false    # Use LLM to determine optimal file processing order
preserve_existing: true  # Enhance existing docs instead of replacing them
test_mode: false         # Limit analysis to the first 5 non-ignored files
exit_on_docs_only_changes: true  # Skip generation when only docs files changed
```

### Large Repository Handling

When a repository's file count exceeds `threshold`, large-repo mode activates automatically. This enables batch processing and smart sampling to keep runs within time and memory limits.

```yaml
large_repo:
  threshold: 450              # File count that triggers large-repo mode
  max_files: 1000             # Hard cap on files processed per run
  collapsible_tree: true      # Use collapsible HTML tree in architecture doc
  enhanced_sampling: true     # Use smart sampling across the codebase
  files_per_component: 10     # Files sampled per detected component
  smart_prioritization: true  # Prioritize high-signal files
  verbose_logging: true       # Extra progress detail in large-repo mode
  batch_processing: true      # Process in batches and persist cache between batches
  time_limit_minutes: 45      # Abort and cache results after this many minutes
  cache_only_mode: true       # Only update cache; do not write documentation
  skip_docs_on_partial: true  # Skip doc generation on incomplete batch runs
  create_pr_on_batch: false   # Create a PR after each batch (requires GitHub token)
  batch_pr_branch: "batch-processing/cache-update"
```

### Persistence Layer Documentation

Detects and documents database migration and ORM frameworks in the analyzed repository.

```yaml
persistence:
  enabled: true
  generate_doc: true
  output_file: "docs/PERSISTENCE.md"
  detection_threshold: 0.2    # Confidence threshold (0.0–1.0) for technology detection
  supported_technologies:
    flyway: true
    efcore: true
    prisma: true
    hibernate: true
    django: true
    rails: true
    sequelize: true
    alembic: true
```

### Installation Guide

Generates a step-by-step installation guide based on the detected language, package manager, and dependency files.

```yaml
installation:
  enabled: true
  output_file: "docs/INSTALLATION.md"
```

### Usage Guide

Generates a CLI and API usage guide based on entry points and examples discovered in the codebase.

```yaml
usage:
  enabled: true
  output_file: "docs/USAGE.md"
```

### Troubleshooting Guide

Generates a common issues and resolutions guide based on error handling patterns and documentation found in the codebase.

```yaml
troubleshooting:
  enabled: true
  output_file: "docs/TROUBLESHOOTING.md"
```

### Contributing Guide

Generates contributor guidelines based on code style, test patterns, and existing CONTRIBUTING content.

```yaml
contributing:
  enabled: true
  output_file: "CONTRIBUTING.md"
```

### README Refactoring

After generating separate documentation files, the refactor step replaces the corresponding sections in `README.md` with brief overviews and links, keeping the README focused and navigable.

```yaml
readme_refactor:
  enabled: true
  keep_brief_overview: true    # Retain the first 2–3 lines of each migrated section
  add_navigation_section: true # Add or update a "Documentation" nav section
```

### File Filtering

```yaml
blacklist:
  extensions: [".txt", ".log"]   # File extensions to exclude from analysis
  path_patterns:                 # Regex patterns for paths to exclude
    - "/temp/"
    - "/cache/"
    - "/node_modules/"
    - "/__pycache__/"
    - "/wwwroot/"
```

### Templates

Prompt and documentation templates can be overridden in `config.yaml`. See the default `config.yaml` for the full set of available template keys.

```yaml
templates:
  prompts:
    file_summary: |
      Analyze the following code file and provide a clear, concise summary:
      File: {file_path}
      Type: {file_type}
      Context: {context}

      Code:
      {code}
  docs:
    readme: |
      # {project_name}

      {project_overview}

      ## Documentation

      {usage}

      ## Development

      {contributing}

      ## License

      {license}
```

## Environment Variables

The configuration system supports overriding settings using environment variables. The following environment variables are supported:

| Environment Variable | Configuration Setting | Description |
|---------------------|------------------------|-------------|
| `LLM_PROVIDER` | `llm_provider` | LLM provider to use ("ollama" or "bedrock") |
| `DEBUG` | `debug` | Enable debug logging (true/false) |
| `AWS_REGION` | `bedrock.region` | AWS region for Bedrock |
| `AWS_BEDROCK_MODEL_ID` | `bedrock.model_id` | Bedrock model ID |
| `AWS_VERIFY_SSL` | `bedrock.verify_ssl` | Whether to verify SSL certificates (true/false) |
| `CACHE_ENABLED` | `cache.enabled` | Enable caching (true/false) |

Example:

```bash
export LLM_PROVIDER=bedrock
export AWS_REGION=us-west-2
python codebase_scribe.py --repo ./my-project
```

## Configuration Validation

The configuration system validates the configuration values to ensure they meet the expected types and constraints. If validation fails, the system will log an error and fall back to the default configuration.

Validation checks include:

- LLM provider must be "ollama" or "bedrock"
- Ollama and Bedrock configurations must be dictionaries
- Numeric values must be of the correct type and within valid ranges
- Cache location must be "repo" or "home"

## Using the Configuration System in Code

### Loading Configuration

```python
from src.utils.config_utils import load_config

# Load configuration from default file (config.yaml)
config = load_config("config.yaml")

# Access configuration values
llm_provider = config.llm_provider
debug_mode = config.debug
```

### Using ScribeConfig

```python
from src.utils.config_class import ScribeConfig, OllamaConfig, BedrockConfig

# Create a ScribeConfig instance
config = ScribeConfig()
config.debug = True
config.llm_provider = "ollama"

# Configure Ollama
config.ollama = OllamaConfig(
    base_url="http://localhost:11434",
    max_tokens=4096,
    retries=3,
    timeout=30
)

# Access configuration values
llm_provider = config.llm_provider
debug_mode = config.debug

# Access provider-specific configuration
ollama_config = config.ollama
bedrock_config = config.bedrock
cache_config = config.cache

# Get templates
file_summary_template = config.templates.prompts.file_summary
readme_template = config.templates.docs.readme

# Write configuration to file
config.write_to_file("new_config.yaml")
```

### Backward Compatibility

For backward compatibility, you can convert between dictionary and class-based configurations:

```python
from src.utils.config_utils import config_to_dict, dict_to_config

# Convert ScribeConfig to dictionary
config_dict = config_to_dict(config)

# Convert dictionary to ScribeConfig
config = dict_to_config(config_dict)
```

## Extending the Configuration System

To add new configuration options:

1. Update the appropriate class in `src/utils/config_class.py`
2. Add default values in the class constructor
3. Update the `from_dict` and `to_dict` methods if needed
4. Add environment variable support in the `update_config_with_args` function if needed
5. Update the documentation in this file

## Best Practices

1. **Use the ScribeConfig class** for type safety and better organization
2. **Provide default values** in class constructors
3. **Use environment variables** for sensitive information or deployment-specific settings
4. **Document new configuration options** in this file
5. **Use type hints** for better IDE support and code quality

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   - Check that the configuration file exists at the specified path
   - The system will fall back to default configuration

2. **Invalid YAML syntax**
   - Check the YAML syntax in your configuration file
   - The system will fall back to default configuration

3. **Validation errors**
   - Check the error message in the logs
   - Ensure configuration values meet the expected types and constraints
   - The system will fall back to default configuration

4. **Environment variables not applied**
   - Check that environment variables are set correctly
   - Environment variables are case-sensitive