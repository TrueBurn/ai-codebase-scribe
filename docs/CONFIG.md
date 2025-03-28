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

The configuration file uses a YAML format with the following structure:

```yaml
# LLM provider configuration
llm_provider: "ollama"  # or "bedrock"
debug: false  # Enable debug logging

# Ollama-specific configuration
ollama:
  base_url: "http://localhost:11434"
  max_tokens: 4096
  retries: 3
  retry_delay: 1.0
  timeout: 30
  concurrency: 1  # Number of concurrent requests
  temperature: 0.0  # 0.0 = deterministic output

# AWS Bedrock-specific configuration
bedrock:
  region: "us-east-1"
  model_id: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
  max_tokens: 8192
  retries: 3
  retry_delay: 1.0
  timeout: 120  # Default to 2 minutes
  verify_ssl: true
  concurrency: 5  # Default to moderate concurrency
  temperature: 0.0  # Default to deterministic output

# Cache configuration
cache:
  enabled: true
  directory: ".cache"
  location: "home"  # "repo" or "home"
  hash_algorithm: "md5"  # "md5", "sha1", or "sha256"
  global_directory: ".readme_generator_cache"  # Used when location is "home"

# Processing options
optimize_order: false
preserve_existing: true
no_cache: false
test_mode: false

# File filtering
blacklist:
  extensions: [".txt", ".log"]
  path_patterns:
    - "/temp/"
    - "/cache/"
    - "/node_modules/"
    - "/__pycache__/"
    - "/wwwroot/"
    - "^aql/"
    - "aql/"
    - "/aql/"

# Templates for prompts and documentation
templates:
  prompts:
    file_summary: |
      Analyze the following code file and provide a clear, concise summary:
      File: {file_path}
      Type: {file_type}
      Context: {context}

      Code:
      {code}
    # Other prompt templates...
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
    # Other documentation templates...
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