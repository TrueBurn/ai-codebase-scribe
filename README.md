> **⚠️ WORK IN PROGRESS ⚠️**
> 
> This project is under active development and is **NOT** production-ready.
> Breaking changes are likely to occur without prior notice.
> Use at your own risk in non-production environments only.

# CodeBase Scribe AI

A Python tool that generates comprehensive project documentation using AI models. It analyzes your codebase, generates documentation, validates links, checks readability, and ensures high-quality output with flexible AI provider options.

## Documentation
For detailed technical documentation and architecture information, see:
- [Architecture Guide](docs/ARCHITECTURE.md) - System design and component interactions
- [Architecture Generator](docs/ARCHITECTURE_GENERATOR.md) - Documentation for the architecture generation system
- [Configuration Guide](docs/CONFIG.md) - Detailed configuration options and usage
- [GitHub Integration](docs/GITHUB_INTEGRATION.md) - GitHub utilities and integration features
- [Development Guide](docs/DEVELOPMENT.md) - Development setup and workflows
- [Contributing Guide](docs/CONTRIBUTING.md) - Guidelines for contributing to the project
- [Contributing Generator](docs/CONTRIBUTING_GENERATOR.md) - Documentation for the contributing guide generation system
- [Badges Guide](docs/BADGES.md) - Documentation for the badge generation system
- [README Generator](docs/README_GENERATOR.md) - Documentation for the README generation system
- [Mermaid Generator](docs/MERMAID_GENERATOR.md) - Documentation for the Mermaid diagram generation system
- [CodebaseAnalyzer](docs/CODEBASE_ANALYZER.md) - Documentation for the codebase analysis system
- [LLM Clients](docs/LLM_CLIENTS.md) - Documentation for the LLM client system
- [Message Manager](docs/MESSAGE_MANAGER.md) - Documentation for the message formatting system

## Features

- 🔍 **Intelligent Codebase Analysis**
  - AST parsing for code structure
  - Dependency graph generation
  - Import/export detection
  - Binary file detection
  
- 🤖 **Flexible AI Processing**
  - Local Ollama integration
    - Secure local processing
    - Interactive model selection
  - AWS Bedrock integration
    - Claude 3.7 Sonnet support
    - Enterprise-grade AI capabilities
  - Customizable prompt templates
  - Context-aware generation
  - Parallel processing support
  
- 📝 **Documentation Generation**
  - README.md generation
  - Architecture documentation
  - API documentation
  - Developer guides
  
- 🔄 **Smart Caching**
  - Multi-level cache (memory + SQLite)
  - Intelligent invalidation
  - TTL support
  - Size-based limits
  
- ✅ **Validation**
  - Link checking (internal + external)
  - Markdown validation
  - Badge verification
  - Reference checking
  
- 📊 **Quality Metrics**
  - Readability scoring
  - Complexity analysis
  - Documentation coverage
  - Improvement suggestions

- 🔄 **Repository Integration**
  - Local repository analysis
  - GitHub repository cloning
  - Automatic pull request creation
  - Branch management
  - Custom PR titles and descriptions

## Installation

### 1. Set Up Virtual Environment

#### Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show virtual environment path)
where python
```

#### macOS and Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show virtual environment path)
which python
```

### 2. Clone and Install

```bash
# Clone the repository
git clone https://github.com/TrueBurn/ai-codebase-scribe.git
cd ai-codebase-scribe

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running locally
# Visit https://ollama.ai for installation instructions
```

### 3. Deactivate Virtual Environment
When you're done, you can deactivate the virtual environment:
```bash
deactivate
```

## Model Selection

When you first run the tool, it will:
1. Connect to your Ollama instance
2. List all available models
3. Prompt you to select one interactively

Example selection dialog:
```bash
Available Ollama models:
1. llama3:latest
2. codellama:7b
3. mistral:instruct

Enter the number of the model to use: 2
Selected model: codellama:7b
```

## Requirements

- Python 3.8+
- Ollama running locally
- Git repository
- Required Python packages:
  - `ollama>=0.4.7`
  - `gitignore-parser>=0.1.11`
  - `networkx>=3.2.1`
  - `python-magic>=0.4.27`
  - `pyyaml>=6.0.1`
  - `tqdm>=4.66.1`
  - `textstat>=0.7.3`
  - `psutil>=5.9.0`

### Basic Usage

```bash
# Generate documentation for a local repository
python codebase_scribe.py --repo ./my-project

# Use a specific model
python codebase_scribe.py --repo ./my-project --model llama3

# Enable debug mode for verbose output
python codebase_scribe.py --repo ./my-project --debug
```

### GitHub Integration

```bash
# Clone and analyze a GitHub repository
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe

# Use a GitHub token for private repositories
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe --github-token YOUR_TOKEN

# Alternative: set token as environment variable
export GITHUB_TOKEN=your_github_token
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe
```

### Creating Pull Requests

```bash
# Create a PR with documentation changes
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe \
  --create-pr \
  --github-token YOUR_TOKEN \
  --branch-name docs/readme-update \
  --pr-title "Documentation: Add README and architecture docs" \
  --pr-body "This PR adds auto-generated documentation using the README generator tool."

# Keep the cloned repo after PR creation (for debugging)
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe \
  --create-pr --keep-clone
```

### Cache Management

```bash
# Disable caching (process all files)
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe --no-cache

# Clear cache before processing
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe --clear-cache

# Only clear cache (don't generate documentation)
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe --clear-cache --keep-clone
```

### LLM Providers

```bash
# Use Ollama (default)
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe --llm-provider ollama

# Use AWS Bedrock
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe --llm-provider bedrock
```

### Output Customization

```bash
# Generate additional API documentation
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe --api-docs

# Custom output files
python codebase_scribe.py --github https://github.com/TrueBurn/ai-codebase-scribe \
  --output-readme custom_readme.md \
  --output-arch custom_architecture.md
```

### Complete Workflow Example

```bash
# 1. Set your GitHub token as an environment variable
export GITHUB_TOKEN=ghp_your_personal_access_token

# 2. Generate documentation and create a PR
python codebase_scribe.py \
  --github https://github.com/TrueBurn/ai-codebase-scribe \
  --create-pr \
  --branch-name docs/update-documentation \
  --pr-title "Documentation: Update README and architecture docs" \
  --pr-body "This PR updates the project documentation with auto-generated content that reflects the current state of the codebase."
```

### Arguments

- `--repo`: Path to repository to analyze (required if not using --github)
- `--github`: GitHub repository URL to clone and analyze
- `--github-token`: GitHub Personal Access Token for private repositories
- `--keep-clone`: Keep cloned repository after processing (GitHub only)
- `--create-pr`: Create a pull request with generated documentation (GitHub only)
- `--branch-name`: Branch name for PR creation (default: docs/auto-generated-readme)
- `--pr-title`: Title for the pull request
- `--pr-body`: Body text for the pull request
- `--output`, `-o`: Output file name (default: README.md)
- `--config`, `-c`: Path to config file (default: config.yaml)
- `--debug`: Enable debug logging
- `--test-mode`: Enable test mode (process only first 5 files)
- `--no-cache`: Disable caching of file summaries
- `--clear-cache`: Clear the cache for this repository before processing
- `--optimize-order`: Use LLM to determine optimal file processing order
- `--llm-provider`: LLM provider to use (ollama or bedrock, overrides config file)

The generated documentation files will be created in the target repository directory (`--repo` path) by default. You can specify different output locations using the `--readme` and `--architecture` arguments.

Example with custom output paths:
```bash
python codebase_scribe.py \
  --repo /path/to/your/repo \
  --readme /path/to/output/README.md \
  --architecture /path/to/output/ARCHITECTURE.md
```

## Configuration

The `config.yaml` no longer specifies models. Instead, you'll choose from available models at runtime. The configuration only needs:

```yaml
ollama:
  base_url: "http://localhost:11434"  # Ollama API endpoint
  max_tokens: 4096  # Maximum tokens per request
  retries: 3  # Number of retry attempts
  retry_delay: 1.0  # Delay between retries
  timeout: 30  # Request timeout in seconds

cache:
  enabled: true  # Set to false to disable caching
  directory: ".cache"  # Directory to store cache files
  location: "repo"  # "repo" or "home" for cache location

templates:
  prompts:
    file_summary: |
      # Custom prompt for file summaries
      Analyze the following code file and provide a clear, concise summary:
      File: {file_path}
      Type: {file_type}
      Context: {context}
      
      Code:
      {code}
    
    project_overview: |
      # Custom prompt for project overview
      Generate a comprehensive overview for:
      Project: {project_name}
      Files: {file_count}
      Components: {key_components}
  
  docs:
    readme: |
      # {project_name}
      
      {project_overview}
      
      ## Usage
      {usage}
```

The `config.yaml` supports file filtering through a blacklist system:

```yaml
blacklist:
  extensions: [".md", ".txt", ".log"]  # File extensions to exclude
  path_patterns: 
    - "/temp/"                        # Path patterns to exclude
    - "/cache/"
    - "/node_modules/"
    - "/__pycache__/"
```

### Key Configuration Options
- **extensions**: List of file extensions to exclude from analysis
- **path_patterns**: List of regex patterns for paths to exclude

### Remote Ollama Setup

You can run Ollama on a different machine in your network:

1. **Local Machine** (default):
```yaml
ollama:
  base_url: "http://localhost:11434"
```

2. **Network Machine**:
```yaml
ollama:
  base_url: "http://192.168.1.100:11434"  # Replace with your machine's IP
```

3. **Custom Port**:
```yaml
ollama:
  base_url: "http://ollama.local:8000"  # Custom domain and port
```

Note: Ensure the Ollama server is accessible from your machine and any necessary firewall rules are configured.

## Project Structure

```
src/
├── analyzers/          # Code analysis tools
│   └── codebase.py     # Repository analysis
├── clients/            # External service clients
│   ├── ollama.py       # Ollama API integration
│   ├── bedrock.py      # AWS Bedrock integration
│   └── llm_utils.py    # Shared LLM utilities
├── generators/         # Content generation
│   ├── contributing.py # Contributing guide generation
│   └── readme.py       # README generation
├── models/            # Data models
│   └── file_info.py   # File information
└── utils/             # Utility functions
    ├── cache.py       # Caching system
    ├── config.py      # Configuration
    ├── link_validator.py  # Link validation
    ├── markdown_validator.py  # Markdown checks
    ├── progress.py    # Progress tracking
    ├── prompt_manager.py  # Prompt handling
    ├── readability.py # Readability scoring
    └── tree_formatter.py # Project structure visualization
```

## Development

### Testing

We use pytest for testing. To run tests:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest  # Note: On Windows, run terminal as Administrator

# Run with coverage report
pytest --cov=src tests/
```

#### Platform-Specific Notes
- **Windows**: Run terminal as Administrator for tests
- **Unix/Linux/macOS**: Regular user permissions are sufficient

See our [Development Guide](docs/DEVELOPMENT.md#testing) for detailed testing instructions.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/CONTRIBUTING.md) for details on:
- Code of Conduct
- Development process
- Pull request process
- Coding standards
- Documentation requirements

## Documentation

- [API Documentation](docs/API.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Architecture Generator](docs/ARCHITECTURE_GENERATOR.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Contributing Generator](docs/CONTRIBUTING_GENERATOR.md)
- [README Generator](docs/README_GENERATOR.md)
- [Mermaid Generator](docs/MERMAID_GENERATOR.md)
- [CodebaseAnalyzer](docs/CODEBASE_ANALYZER.md)
- [LLM Clients](docs/LLM_CLIENTS.md)
- [Message Manager](docs/MESSAGE_MANAGER.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Development Notes

### Configuration System

The project uses a class-based configuration system (`ScribeConfig`) that provides type safety and better organization of configuration options. The configuration is loaded from a YAML file and can be overridden by command-line arguments and environment variables.

The system provides:

- Type safety and better IDE support
- Structured organization of configuration options
- Validation of configuration values
- Environment variable overrides
- Command-line argument overrides

See the [Configuration Guide](docs/CONFIG.md) for detailed information on the configuration system.

### Path Compression

The project includes a path compression utility that reduces token usage when sending file paths to LLMs. This is particularly useful for Java projects with deep package structures, where file paths can consume a significant portion of the token budget.

The path compression system:

- Identifies common prefixes in file paths
- Replaces them with shorter keys (e.g., `@1`, `@2`)
- Adds an explanation of the compression scheme to the LLM prompt
- Significantly reduces token usage for large projects

For example, paths like `src/main/java/com/example/project/Controller.java` are compressed to `@1/Controller.java`, saving tokens while maintaining readability.
### Recent Fixes

- **Improved Project Structure Visualization**: Enhanced the project structure representation in architecture documentation with a new tree formatter that uses box-drawing characters to clearly show folder hierarchy and relationships.

- **Path Compression**: Added a path compression utility that reduces token usage when sending file paths to LLMs. This is particularly useful for Java projects with deep package structures.

- **Dependencies Error Fix**: Fixed an issue with the `dependencies` field in the project overview generation. The problem was that the original file manifest was being passed to the dependency analysis function instead of the converted manifest, causing errors when processing FileInfo objects.

- **Template Parameter Fix**: Fixed an issue in the `BedrockClient` class where the `dependencies` parameter was missing in the `format` method call for the project overview template, causing a KeyError.
- **Template Parameter Fix**: Fixed an issue in the `BedrockClient` class where the `dependencies` parameter was missing in the `format` method call for the project overview template, causing a KeyError.

### Bytecode Caching

Python bytecode caching is currently disabled for development purposes. To re-enable it:

1. Remove `sys.dont_write_bytecode = True` from `codebase_scribe.py`
2. Or unset the `PYTHONDONTWRITEBYTECODE` environment variable

This should be re-enabled before deploying to production for better performance.

## Cache Management

The tool provides several ways to manage caching:

### Cache Location Options

The cache can be stored in two locations, configurable via the `cache.location` setting in `config.yaml`:

1. **Repository-Based Cache** (`location: "repo"`, default)
   - Stored directly in the target repository's `.cache` directory
   - **Shared Cache**: Anyone running the script on the same repository benefits from previous analysis
   - **CI/CD Integration**: Makes it easier to integrate with GitHub Actions as the cache is part of the repository
   - **Portable**: Cache travels with the repository when cloned or forked

2. **Home Directory Cache** (`location: "home"`)
   - Stored in the user's home directory under `.readme_generator_cache`
   - **Privacy**: Cache files remain on the local machine
   - **Cleaner Repository**: Doesn't add cache files to the repository
   - **Legacy Behavior**: Matches the behavior of earlier versions

You can configure this in `config.yaml`:
```yaml
cache:
  enabled: true
  directory: ".cache"
  location: "repo"  # Change to "home" to use home directory
```

### Command Line Options
```bash
# Disable caching for current run
python codebase_scribe.py --repo /path/to/repo --no-cache

# Clear cache for specific repository
python codebase_scribe.py --repo /path/to/repo --clear-cache

# Note: --clear-cache will clear the cache and exit without processing
```

### Cache Optimization

The cache system includes several optimizations:

- **SQLite Vacuum**: Periodically compacts the database to minimize repository size
- **Content-Based Invalidation**: Only regenerates summaries when file content changes
- **Structured Storage**: Uses a database format that minimizes merge conflicts

### Cache Exclusions

While the cache is generally committed to the repository, certain directories are excluded:

- **Test Folders**: Cache files in `tests/` and its subdirectories are excluded from git
- This prevents test-generated cache files from being committed while still allowing the main project cache to be shared

## TODO

### Completed
- ✅ **Dependencies Error Fix**: Fixed an issue with the `dependencies` parameter missing in the `generate_project_overview` method in `BedrockClient` class.
- ✅ **Split Contributing Guide**: Moved the Contributing guide to a separate file with its own generator in the root directory, following GitHub conventions.

### Planned
- 📝 **Split Usage Guide**: Move the usage guide to a separate file with its own generator.
- 🔗 **Improve Documentation Links**: Ensure generated README properly links to all other documentation files.
- 🔄 **GitHub Workflows**: Add GitHub Actions workflows for CI/CD, automated testing, and code quality checks.
- 🧪 **Improve Test Coverage**: Add more unit tests for the core functionality.
- 🚀 **Performance Optimization**: Optimize the file processing pipeline for better performance.
- 🗑️ **Fix Lingering Folders**: Clean up lingering temporary folders in home directory when pulling from GitHub.