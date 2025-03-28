# Development Guide

This guide will help you set up your development environment and understand the project structure and workflows.

## Table of Contents
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Cache Management](#cache-management)

## Development Environment Setup

### 1. Prerequisites
- Python 3.8 or higher
- Git
- Ollama installed locally
- A code editor (VS Code recommended)

### 2. Initial Setup

#### Windows
```bash
# Clone repository
git clone https://github.com/yourusername/readme-generator.git
cd readme-generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install package in development mode with all dependencies
pip install -e .
```

#### macOS/Linux
```bash
# Clone repository
git clone https://github.com/yourusername/readme-generator.git
cd readme-generator

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install package in development mode with all dependencies
pip install -e .
```

### Dependencies

The project automatically handles platform-specific dependencies:
- Windows: Uses `python-magic-bin` for binary file detection
- Unix/Linux: Uses `python-magic` with system `libmagic`

Core dependencies include:
- networkx: For dependency graph generation
- gitignore_parser: For repository file filtering
- python-magic/python-magic-bin: For binary file detection

### 3. Editor Setup

#### VS Code
1. Install recommended extensions:
   - Python
   - Pylance
   - Black Formatter
   - isort
   - markdownlint

2. Configure settings.json:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

## Project Structure

The project uses a tree-like structure visualization that is:
- Generated automatically from the file manifest
- Used for context in AI model requests
- Included in generated documentation
- Maintained consistently across files

### Structure Generation

The project structure is built using the `OllamaClient._build_project_structure` method:
```python
def _build_project_structure(self, file_manifest: dict) -> str:
    """Build a tree-like project structure string."""
    # Convert paths to tree structure
    tree = {}
    for path in sorted(file_manifest.keys()):
        parts = Path(path).parts
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None
    
    # Convert tree to string representation
    def _build_tree(node, prefix="", is_last=True) -> str:
        # ... tree building logic ...
```

```
src/
├── analyzers/          # Code analysis tools
│   └── codebase.py     # Repository analysis
├── clients/            # External service clients
│   └── ollama.py       # Ollama API integration
├── generators/         # Content generation
│   ├── readme.py       # README generation
│   └── mermaid.py      # Mermaid diagram generation
├── models/            # Data models
│   └── file_info.py   # File information
└── utils/             # Utility functions
    ├── cache.py       # Caching system
    ├── config.py      # Configuration (see [CONFIG.md](CONFIG.md))
    ├── link_validator.py  # Link validation
    ├── markdown_validator.py  # Markdown checks
    ├── progress.py    # Progress tracking
    ├── prompt_manager.py  # Prompt handling
    └── readability.py # Readability scoring
```

## Core Components

### 1. CodebaseAnalyzer
- Analyzes repository structure
- Builds dependency graph
- Detects file types and relationships
- Manages file manifest

### 2. OllamaClient
- Handles AI model interactions
- Manages context windows
- Implements retry logic
- Handles response processing

### 3. Documentation Generator
- Generates README.md
- Creates architecture documentation
- Produces API documentation
- Manages templates

### 4. Utility Systems
- Caching (memory and disk)
- Memory optimization
- Link validation
- Readability scoring

### 5. Configuration System
- Manages application settings
- Provides defaults and overrides
- Supports environment variables
- Validates configuration values
- See [CONFIG.md](CONFIG.md) for detailed documentation

## Development Workflow

### 1. Feature Development
1. Create feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write tests first
3. Implement feature
4. Update documentation
5. Run tests
6. Submit PR

### 2. Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_specific.py
```

### 3. Code Quality Checks
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Type checking
mypy src

# Lint code
flake8 src tests
```

## Testing

### Running Tests

1. **Setup**
```bash
# Install the package in development mode (if not done already)
pip install -e .
```

2. **Platform-Specific Requirements**

#### Windows Users
> **Important**: On Windows, you need to run your terminal as Administrator when running tests. This is required for proper pytest cache handling.
```bash
# Run Command Prompt or PowerShell as Administrator, then:
pytest
```

#### Unix/Linux/macOS Users
```bash
# Regular user permissions are sufficient
pytest
```

3. **Basic Test Run**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analyzer.py

# Run specific test function
pytest tests/test_analyzer.py::test_analyze_repository
```

2. **Coverage Reports**
```bash
# Run tests with coverage
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
# Report will be in htmlcov/index.html
```

3. **Test Categories**
- **Unit Tests**: Test individual components
  ```bash
  pytest tests/test_analyzer.py tests/test_cache.py tests/test_mermaid.py
  ```
- **Integration Tests**: Test component interactions
  ```bash
  pytest tests/test_ollama.py
  ```

### Writing Tests

1. **Test Structure**
```python
def test_feature():
    # Arrange - Set up test data
    input_data = "test"
    
    # Act - Call the function being tested
    result = process_data(input_data)
    
    # Assert - Check the results
    assert result == expected_output
```

2. **Using Fixtures**
```python
@pytest.fixture
def test_repo():
    return Path(__file__).parent / 'fixtures' / 'test_repo'

def test_with_fixture(test_repo):
    analyzer = CodebaseAnalyzer(test_repo)
    result = analyzer.analyze()
    assert result is not None
```

3. **Async Tests**
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### Test Environment

1. **Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. **Install Test Dependencies**
```bash
pip install -r requirements-dev.txt
```

3. **Configure Test Environment**
```bash
export PYTHONPATH=src:$PYTHONPATH
export TEST_MODE=True
```

### Continuous Integration

Our CI pipeline runs these checks:
1. Code style (Black, Flake8)
2. Type checking (MyPy)
3. Unit tests
4. Integration tests
5. Coverage report

## Code Style

### General Guidelines
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused
- Maximum line length: 88 characters (Black default)

### Example
```python
from typing import Dict, Optional

def process_data(input_data: Dict[str, str]) -> Optional[str]:
    """Process input data and return formatted result.
    
    Args:
        input_data: Dictionary containing input values.
        
    Returns:
        Formatted string or None if processing fails.
    """
    # Implementation
```

## Performance Considerations

### Memory Management
- Use streaming for large files
- Implement proper cleanup
- Monitor memory usage
- Use chunking for large operations

### Caching
- Use appropriate cache levels
- Implement TTL
- Handle cache invalidation
- Monitor cache size

### Parallel Processing
- Use async where appropriate
- Control concurrency
- Handle errors properly
- Monitor resource usage

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Symptom: High memory usage
   - Solution: Check chunk sizes, enable memory monitoring

2. **Cache Problems**
   - Symptom: Slow performance
   - Solution: Verify cache configuration, check invalidation

3. **Ollama Connection**
   - Symptom: Connection failures
   - Solution: Check Ollama service, verify network settings

### Debug Tools
```bash
# Memory profiling
python -m memory_profiler your_script.py

# Performance profiling
python -m cProfile -o output.prof your_script.py
```

### Logging
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
``` 

### Using Test Mode
```bash
python codebase_scribe.py --repo ./test-repo --test-mode --debug
```
- Shows model selection dialog
- Limits processing to 5 files
- Enables verbose logging

## Cache Management

### Cache System Design
The caching system uses a multi-level approach:
- SQLite for persistent storage
- Content-based invalidation using file hashing
- Support for multiple hash algorithms (md5, sha1, sha256)
- Repository-aware caching
- Automatic initialization
- Graceful fallback

### Managing Cache During Development

1. **Disable Cache**
```bash
# Via command line
python codebase_scribe.py --repo ./your-repo --no-cache

# Via config.yaml
cache:
  enabled: false
```

2. **Clear Cache**
```bash
# Clear specific repository cache
python codebase_scribe.py --repo ./your-repo --clear-cache

# Programmatically clear all caches
from src.utils.cache import CacheManager
CacheManager.clear_all_caches()
```

3. **Cache Location**
- Default: `.cache` directory (created automatically)
- Configurable via `config.yaml`
- Separate database per repository
- Auto-creates missing directories/files

4. **Cache Initialization**
- Creates cache directory if missing
- Initializes SQLite database automatically
- Falls back to disabled cache on errors
- Logs initialization issues for debugging

5. **Cache Invalidation**
- Based on file content hash
- Different hash algorithms available (md5, sha1, sha256)
- Repository-aware cache keys
- Manual clearing
