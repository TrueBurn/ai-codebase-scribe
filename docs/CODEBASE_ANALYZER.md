# Codebase Analyzers

This document covers the analyzer components of the Codebase Scribe AI system. Two analyzers are available:

- **`CodebaseAnalyzer`** (`src/analyzers/codebase.py`) — General repository structure, file manifest, and dependency analysis.
- **`PersistenceAnalyzer`** (`src/analyzers/persistence.py`) — Database migration and ORM framework detection and schema extraction.

---

# CodebaseAnalyzer

The `CodebaseAnalyzer` class is a core component of the Codebase Scribe AI system, responsible for analyzing repository structure and content. This document provides detailed information about its functionality, usage, and implementation.

## Overview

The `CodebaseAnalyzer` scans a repository, analyzes its files, and builds a comprehensive file manifest with metadata. It handles gitignore rules, binary file detection, and can extract information about exports and dependencies from source code files.

## Key Features

- **Repository Traversal**: Efficiently scans repository directories
- **File Classification**: Determines file types and characteristics
- **Binary File Detection**: Identifies binary vs. text files
- **Dependency Tracking**: Builds a graph of file dependencies
- **Import/Export Detection**: Identifies imports and exports in source code
- **Project Name Derivation**: Intelligently determines project name
- **Markdown Header Validation**: Checks markdown header formatting

## Class Structure

```python
class CodebaseAnalyzer:
    """Analyzes repository structure and content."""
    
    # Constants
    BINARY_MIME_PREFIXES = ('text/', 'application/json', 'application/xml')
    BINARY_CHECK_BYTES = 1024
    SOURCE_CODE_EXTENSIONS = {'.py', '.js', '.ts', '.cs', '.java'}
    SPECIAL_FILES = {"README.md", "ARCHITECTURE.md", "CONTRIBUTING.md"}
    SPECIAL_DIRS = {".github"}
    
    def __init__(self, repo_path: Path, config: dict):
        # Initialize analyzer with repository path and configuration
```

## Key Methods

### Repository Analysis

```python
def analyze_repository(self, show_progress: bool = False) -> Dict[str, FileInfo]:
    """Analyze the full repository structure.
    
    This method scans the repository, analyzes each file, and builds a file manifest
    with metadata about each file. It can optionally show a progress bar during analysis.
    
    Args:
        show_progress: Whether to show a progress bar during analysis
        
    Returns:
        Dict[str, FileInfo]: Dictionary mapping file paths to FileInfo objects
        
    Raises:
        ValueError: If the repository path does not exist or is not a directory
        RuntimeError: If no files are found in the repository
        Exception: For other unexpected errors during analysis
    """
```

### File Inclusion Decision

```python
def should_include_file(self, file_path: Path) -> bool:
    """Determine if a file should be included in analysis.
    
    This unified method provides a single point of decision for file inclusion with clear rules.
    
    Args:
        file_path: Path to the file, relative to the repository root
        
    Returns:
        bool: True if the file should be included, False otherwise
    """
```

### Dependency Graph

```python
def build_dependency_graph(self) -> nx.DiGraph:
    """Build and return the dependency graph of files in the repository.
    
    Returns:
        nx.DiGraph: Directed graph of file dependencies
    """
```

### Project Name Detection

```python
def derive_project_name(self, debug: bool = False) -> str:
    """Derive project name from repository structure.
    
    This method attempts to determine the project name by examining various files
    in the repository, such as package.json, setup.py, pom.xml, etc. It also looks
    for patterns in directory structure and namespace declarations.
    
    Returns:
        str: The derived project name, or "Project" if no name could be determined
    """
```

### Python File Analysis

```python
def analyze_python_files(self) -> Dict[Path, FileInfo]:
    """Analyze Python files in the repository.
    
    Returns:
        Dict[Path, FileInfo]: Dictionary of Python files and their metadata
    """
```

## File Filtering Logic

The `should_include_file` method implements a unified approach to file inclusion with the following rules:

1. **Always Include**:
   - Special files (README.md, ARCHITECTURE.md, CONTRIBUTING.md)
   - Files in special directories (.github)

2. **Always Exclude**:
   - Hidden files/directories (except .gitignore)
   - Files with blacklisted extensions
   - Files matching blacklisted path patterns
   - Files matched by gitignore rules

## Binary File Detection

The analyzer uses a two-tier approach to detect binary files:

1. **Primary Method**: Uses the `python-magic` library to check MIME types
2. **Fallback Method**: Checks for null bytes in the file content

```python
def is_binary(self, file_path: Path) -> bool:
    """Check if a file is binary using MIME type detection."""
    
def _is_binary(self, file_path: Path) -> bool:
    """Simple binary file detection by checking for null bytes (fallback)."""
```

## Error Handling

The `CodebaseAnalyzer` implements robust error handling:

- Repository path validation
- File access error handling
- Content parsing error handling
- Graceful degradation when components fail

## Integration with Other Components

The `CodebaseAnalyzer` integrates with:

- **CacheManager**: For caching analysis results
- **ProgressTracker**: For displaying progress during analysis
- **FileInfo**: For storing file metadata

## Usage Example

```python
from pathlib import Path
from src.analyzers.codebase import CodebaseAnalyzer

# Configuration
config = {
    'debug': True,
    'blacklist': {
        'extensions': ['.pyc', '.pyo', '.pyd'],
        'path_patterns': [r'__pycache__', r'\.git']
    }
}

# Create analyzer
analyzer = CodebaseAnalyzer(Path('/path/to/repo'), config)

# Analyze repository
file_manifest = analyzer.analyze_repository(show_progress=True)

# Get dependency graph
graph = analyzer.build_dependency_graph()

# Get project name
project_name = analyzer.derive_project_name()

# Analyze Python files
python_files = analyzer.analyze_python_files()
```

## Testing

The `CodebaseAnalyzer` has comprehensive unit and integration tests:

- **Unit Tests**: Test individual methods and components
- **Integration Tests**: Test interaction with other components
- **Edge Case Tests**: Test handling of unusual inputs and error conditions

Run tests with:

```bash
python -m pytest tests/test_codebase.py tests/test_integration.py
```

Check code coverage with:

```bash
python -m pytest tests/test_codebase.py tests/test_integration.py --cov=src.analyzers.codebase
```

## Recent Improvements

1. **Unified File Inclusion Logic**: Merged `should_ignore` and `_should_include_file` into a single `should_include_file` method with clearer responsibility boundaries.

2. **Reduced Code Duplication**: Refactored `analyze_repository` to use `_get_repository_files` instead of duplicating file collection logic.

3. **Fixed C# Namespace Detection**: Improved the `derive_project_name` method to correctly handle C# namespaces.

4. **Enhanced Error Handling**: Added more specific error types and better error messages.

5. **Improved Test Coverage**: Added comprehensive unit and integration tests, increasing coverage from 64% to 73%.

## Future Enhancements

1. **Complete Error Validation**: Fully implement repository path validation in `analyze_repository`.

2. **Improve Cache Handling**: Address issues with file handles not being properly closed.

3. **Further Increase Test Coverage**: Add more tests for error handling branches and edge cases.

---

# PersistenceAnalyzer

The `PersistenceAnalyzer` class (`src/analyzers/persistence.py`) detects and analyzes database persistence patterns within a repository. It is designed to be used alongside `CodebaseAnalyzer`, operating on the same repository path but focusing exclusively on migration files, ORM model definitions, and schema artifacts.

## Overview

`PersistenceAnalyzer` scans a repository for well-known persistence layer patterns and extracts structured metadata including detected technologies, table definitions, relationships, views, and migration history — without requiring a live database connection.

## Supported Technologies

| Technology | Type | Detection Signals |
|---|---|---|
| Flyway | SQL migration runner | `db/migration/`, `V{n}__*.sql` naming convention |
| EF Core | .NET ORM | `Migrations/` directory, `DbContext` subclasses |
| Prisma | Node.js ORM | `schema.prisma`, `prisma/migrations/` |
| Hibernate | Java ORM | `@Entity`, `@Table` annotations, `*.hbm.xml` |
| Django | Python ORM | `migrations/`, `models.py` with `Model` subclasses |
| Rails / ActiveRecord | Ruby ORM | `db/migrate/`, ActiveRecord migration files |
| Sequelize | Node.js ORM | `migrations/`, `models/` with `define()` calls |
| Alembic | Python migration tool | `alembic/versions/`, `alembic.ini` |

## Key Data Structures

```python
class PersistenceType(Enum):
    """Enumeration of supported persistence layer types."""
    FLYWAY = "flyway"
    EFCORE = "efcore"
    PRISMA = "prisma"
    HIBERNATE = "hibernate"
    DJANGO = "django"
    RAILS = "rails"
    SEQUELIZE = "sequelize"
    ALEMBIC = "alembic"
    UNKNOWN = "unknown"

@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[Dict[str, Any]]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    migration_file: Optional[str]

@dataclass
class RelationshipInfo:
    """Information about table relationships."""
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    relationship_type: str  # "one-to-one", "one-to-many", "many-to-many"
    constraint_name: Optional[str]

@dataclass
class PersistenceLayerInfo:
    """Complete information about the persistence layer."""
    # Aggregates PersistenceType, tables, views, relationships, migrations
```

## Detection Threshold

The `detection_threshold` configuration value (default `0.2`) sets the minimum confidence score for a technology to be included in the analysis. Raising this value makes detection more conservative.

```yaml
persistence:
  detection_threshold: 0.2
```

## Integration with Persistence Generator

After analysis, `PersistenceLayerInfo` is passed to `src/generators/persistence.py`, which uses an LLM to synthesize the extracted schema data into a readable `docs/PERSISTENCE.md` file covering:

- Detected technologies and their versions where available
- Database schema (tables, columns, keys, indexes)
- Table relationships and ERD summary
- Migration history and patterns

## Usage Example

```python
from pathlib import Path
from src.analyzers.persistence import PersistenceAnalyzer
from src.utils.config_class import ScribeConfig

config = ScribeConfig()
analyzer = PersistenceAnalyzer(Path('/path/to/repo'), config)
persistence_info = analyzer.analyze()

print(f"Detected: {persistence_info.technology}")
print(f"Tables: {[t.name for t in persistence_info.tables]}")
```

## Testing

```bash
python -m pytest tests/test_architecture.py -k persistence
```