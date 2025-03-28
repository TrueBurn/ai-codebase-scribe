# API Documentation

This document details the key classes and functions in the AI README Generator.

## Table of Contents
- [Core Components](#core-components)
  - [CodebaseAnalyzer](#codebaseanalyzer)
  - [OllamaClient](#ollamaclient)
  - [DocumentationGenerator](#documentationgenerator)
- [Utility Components](#utility-components)
  - [CacheManager](#cachemanager)
  - [LinkValidator](#linkvalidator)
  - [MarkdownValidator](#markdownvalidator)
  - [ReadabilityScorer](#readabilityscorer)
  - [BadgeGenerator](#badgegenerator)

## Core Components

### CodebaseAnalyzer

```python
class CodebaseAnalyzer:
    """Analyzes repository structure and code relationships.
    
    Uses configuration from config.yaml to determine:
    - File extensions to blacklist
    - Path patterns to exclude
    - Special directories to include
    """
    
    def __init__(self, repo_path: Path, config: Dict[str, Any]):
        """Initialize analyzer with repository path and configuration.
        
        Args:
            repo_path: Path to repository root
            config: Configuration dictionary
        """
        
    async def analyze_repository(self) -> Dict[str, FileInfo]:
        """Analyze repository while respecting blacklist rules.
        
        Returns:
            Filtered dictionary of file paths to FileInfo objects
        """
        
    def build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph of repository.
        
        Returns:
            NetworkX directed graph of file dependencies
        """
```

### OllamaClient

```python
class OllamaClient:
    """Handles all interactions with local Ollama instance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama client with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - model: Model name
                - base_url: API endpoint
                - max_tokens: Maximum tokens per request
                - retries: Number of retry attempts
        """
    
    def set_project_structure(self, file_manifest: dict) -> None:
        """Set the project structure for context in all requests.
        
        Args:
            file_manifest: Dictionary mapping file paths to FileInfo objects
        """
    
    def _build_project_structure(self, file_manifest: dict) -> str:
        """Build a tree-like project structure string.
        
        Args:
            file_manifest: Dictionary mapping file paths to FileInfo objects
            
        Returns:
            Formatted string representing project structure
        """
    
    @async_retry(retries=3, delay=1.0)
    async def generate_summary(self, file_content: str, context: Dict) -> str:
        """Generate a summary for a single file.
        
        Args:
            file_content: Content of file to summarize
            context: Additional context (file path, type, etc.)
            
        Returns:
            Generated summary text
        """
    
    async def generate_component_relationships(self, file_manifest: dict) -> str:
        """Generate description of how components interact.
        
        Args:
            file_manifest: Dictionary mapping file paths to FileInfo objects
            
        Returns:
            Generated component relationship description
        """
```

### DocumentationGenerator

```python
class DocumentationGenerator:
    """Generates various documentation files."""
    
    def __init__(self, file_manifest: Dict[str, FileInfo], config: Dict[str, Any]):
        """Initialize generator with file manifest and configuration.
        
        Args:
            file_manifest: Dictionary of file information
            config: Configuration dictionary
        """
    
    def generate_readme(self) -> str:
        """Generate README.md content.
        
        Returns:
            Generated README content
        """
    
    def generate_architecture_doc(self) -> str:
        """Generate architecture documentation.
        
        Note: This method is deprecated. Use the standalone generate_architecture function
        from src.generators.architecture instead.
        
        Returns:
            Generated architecture documentation
        """
```

## Utility Components

### CacheManager

```python
class CacheManager:
    """Manages caching of file summaries.
    
    This class provides functionality to cache file summaries and other data
    to avoid redundant processing. It supports repository-aware caching with
    content-based invalidation using file hashing.
    """
    
    # Default cache directory name in user's home directory
    DEFAULT_GLOBAL_CACHE_DIR = '.readme_generator_cache'
    
    # Default cache directory name in repository
    DEFAULT_REPO_CACHE_DIR = '.cache'
    
    # Default hash algorithm
    DEFAULT_HASH_ALGORITHM = 'md5'
    
    def __init__(self, enabled: bool = True, repo_identifier: str = None, repo_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the cache manager.
        
        Args:
            enabled: Whether caching is enabled
            repo_identifier: Unique identifier for the repository (e.g., GitHub repo name)
            repo_path: Path to the repository
            config: Configuration dictionary with cache settings including:
                - directory: str (default: '.cache')
                - location: str (default: 'repo')
                - hash_algorithm: str (default: 'md5')
                - global_directory: str (default: '.readme_generator_cache')
                
        Notes:
            - Creates cache directory if it doesn't exist
            - Initializes SQLite database automatically
            - Falls back to disabled cache if initialization fails
        """
    
    def _init_db(self):
        """Initialize SQLite database for file caching.
        
        Creates the database and required tables if they don't exist.
        """
    
    def get_repo_cache_dir(self, repo_path: Optional[Path] = None) -> Path:
        """Get the cache directory for a repository.
        
        Args:
            repo_path: Path to the repository (optional, uses stored path if not provided)
            
        Returns:
            Path to the cache directory for the repository
        """
    
    def clear_repo_cache(self) -> None:
        """Clear the cache for the current repository.
        
        Removes all entries from the SQLite database and vacuums it to reclaim space.
        """
    
    @classmethod
    def clear_all_caches(cls, cache_dir: Optional[Path] = None, repo_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """Clear all caches for all repositories.
        
        Args:
            cache_dir: The cache directory to clear (default: None, will use config)
            repo_path: The repository path to clear cache for (default: None)
            config: Configuration dictionary (default: None)
        """
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache.
        
        Args:
            key: Cache key (file path as string)
            
        Returns:
            Cached value or None if not found
        """
    
    def set(self, key: str, value: str) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key (file path as string)
            value: Value to cache
        """
    
    def get_cached_summary(self, file_path: Path) -> Optional[str]:
        """Get cached summary for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Cached summary or None if not found or outdated
        """
    
    def save_summary(self, file_path: Path, summary: str) -> None:
        """Save a file summary to the cache.
        
        Args:
            file_path: Path to file
            summary: Summary to cache
        """
    
    def is_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last cache.
        
        This method compares the current file hash with the cached hash
        to determine if the file has been modified.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file has changed or has no cache entry, False otherwise
        """
    
    def _create_cache_key(self, file_path: Path) -> str:
        """Create a consistent cache key for a file path.
        
        Always uses the path relative to the repository root, ensuring
        cache hits even when the repository is cloned to different locations.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cache key string
        """
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate a hash of the file contents using the configured algorithm.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string of the file contents
        """
```

### LinkValidator

```python
class LinkValidator:
    """Validates internal and external links in documentation."""
    
    def __init__(self, repo_path: Path, debug: bool = False, timeout: float = 5.0, max_retries: int = 2):
        """Initialize link validator.
        
        Args:
            repo_path: Repository root path
            debug: Enable debug output
            timeout: Timeout for external link validation in seconds
            max_retries: Maximum number of retry attempts for external links
        """
    
    async def validate_document(self, content: str, base_path: Path) -> List[LinkIssue]:
        """Validate all links in a markdown document.
        
        Args:
            content: Document content
            base_path: Base path for relative links
            
        Returns:
            List of link validation issues
        """
        
    def _text_to_anchor(self, text: str) -> str:
        """Convert header text to GitHub-style anchor.
        
        Args:
            text: Header text
            
        Returns:
            Anchor ID
        """
        
    async def _validate_link(self, url: str, line_number: int, base_path: Path) -> None:
        """Validate a single link.
        
        Args:
            url: URL to validate
            line_number: Line number where the link appears
            base_path: Base path for relative links
        """
        
    def _validate_internal_link(self, url: str, line_number: int, base_path: Path) -> None:
        """Validate internal file or anchor links.
        
        Args:
            url: URL to validate
            line_number: Line number where the link appears
            base_path: Base path for relative links
        """
        
    async def _validate_external_link(self, url: str, line_number: int) -> None:
        """Validate external URLs with retry mechanism.
        
        Args:
            url: URL to validate
            line_number: Line number where the link appears
        """
```

### MarkdownValidator

```python
class MarkdownValidator:
    """Validates and fixes common markdown issues."""
    
    def __init__(self, content: str, max_line_length: int = 10000):
        """Initialize the validator with markdown content.
        
        Args:
            content: The markdown content to validate
            max_line_length: Maximum line length to process (for performance)
        """
    
    def validate(self) -> List[ValidationIssue]:
        """Run all validation checks and return found issues.
        
        Returns:
            A list of ValidationIssue objects representing all found issues
        """
    
    async def validate_with_link_checking(self, repo_path: Path, base_path: Path = None) -> List[ValidationIssue]:
        """Run all validation checks including comprehensive link validation.
        
        This method extends the standard validation with additional checks for link
        validity, including checking if internal links point to existing files and
        if external links are accessible.
        
        Args:
            repo_path: The root path of the repository for resolving relative links
            base_path: The base path for resolving relative links (defaults to repo_path)
            
        Returns:
            A list of ValidationIssue objects representing all found issues
        """
    
    def fix_common_issues(self) -> str:
        """Attempt to fix common markdown issues automatically.
        
        Returns:
            A string containing the fixed markdown content
        """
```

### ReadabilityScorer

```python
class ReadabilityScorer:
    """Analyzes and scores documentation readability."""
    
    def analyze_text(self, text: str, section_name: str) -> Dict:
        """Analyze text and return readability metrics.
        
        Args:
            text: Text to analyze
            section_name: Name of section being analyzed
            
        Returns:
            Dictionary of readability metrics
        """
    
    def get_recommendations(self, section_name: str) -> List[str]:
        """Get improvement recommendations based on scores.
        
        Args:
            section_name: Name of analyzed section
            
        Returns:
            List of improvement recommendations
        """
```

### BadgeGenerator

```python
def generate_badges(file_manifest: Dict[str, FileInfo], repo_path: Path, badge_style: str = "for-the-badge") -> str:
    """Generate appropriate badges based on repository content.
    
    This function analyzes the repository content to generate markdown badges for:
    - License type
    - CI/CD tools
    - Testing frameworks
    - Documentation
    - Docker usage
    - Programming languages and frameworks
    - Databases
    
    Args:
        file_manifest: Dictionary of files in the repository
        repo_path: Path to the repository
        badge_style: Style of badges to generate (default: for-the-badge)
        
    Returns:
        str: Space-separated string of markdown badges
        
    Raises:
        ValueError: If file_manifest is None or empty
    """
```

The badge generation system uses various detection strategies to identify project characteristics:

1. **License Detection**: Analyzes license files to determine the license type
2. **CI/CD Detection**: Detects CI/CD configuration files
3. **Testing Framework Detection**: Detects testing framework files and directories
4. **Documentation Detection**: Checks for documentation files and directories
5. **Docker Detection**: Checks for Docker-related files
6. **Language and Framework Detection**: Detects programming languages and frameworks
7. **Database Detection**: Detects database technologies

The badges are generated using the shields.io service with the following format:
```
![Label](https://img.shields.io/badge/Label-Message-Color?style=Style&logo=LogoName&logoColor=LogoColor)
```

For more detailed information, see the [Badges Guide](BADGES.md).

## Architecture Generator

```python
async def generate_architecture(
    repo_path: Path,
    file_manifest: dict,
    llm_client: BaseLLMClient,
    config: dict
) -> str:
    """
    Generate architecture documentation for the repository.
    
    This function uses an LLM to generate comprehensive architecture documentation
    with proper formatting, table of contents, and sections. If the LLM fails or
    returns invalid content, it falls back to a basic architecture document.
    
    Args:
        repo_path: Path to the repository
        file_manifest: Dictionary of files in the repository
        llm_client: LLM client for generating content
        config: Configuration dictionary
        
    Returns:
        Formatted architecture documentation as a string
    """
```

The architecture generator also provides supporting functions:

```python
def create_fallback_architecture(project_name: str, file_manifest: dict) -> str:
    """
    Create a fallback architecture document with basic project structure.
    
    This function is used when the LLM fails to generate architecture documentation
    or returns invalid content. It creates a basic document with project structure,
    technology stack, and other information that can be derived from the file manifest.
    """

def analyze_basic_structure(file_manifest: dict) -> Dict[str, str]:
    """
    Perform basic structure analysis without LLM.
    
    This function analyzes the file manifest to determine the technology stack
    and project patterns based on file extensions and directory names.
    """
```

For more details, see the [Architecture Generator Documentation](ARCHITECTURE_GENERATOR.md).

## Usage Examples

### Basic Usage

```python
# Initialize components
analyzer = CodebaseAnalyzer(repo_path, config)
ollama = OllamaClient(config)
cache = CacheManager(repo_path, config)

# Analyze repository
file_manifest = await analyzer.analyze_repository()

# Generate README documentation
from src.generators.readme import generate_readme
readme_content = await generate_readme(
    repo_path=repo_path,
    llm_client=ollama,
    file_manifest=file_manifest,
    config=config
)

# Generate architecture documentation
from src.generators.architecture import generate_architecture
arch_content = await generate_architecture(
    repo_path=repo_path,
    file_manifest=file_manifest,
    llm_client=ollama,
    config=config
)

# Validate documentation
markdown_validator = MarkdownValidator(readme_content)
validation_issues = await markdown_validator.validate_with_link_checking(repo_path)

# Or use LinkValidator directly for just link validation
link_validator = LinkValidator(repo_path)
link_issues = await link_validator.validate_document(readme_content, repo_path)
```

### Advanced Usage

```python
# Readability analysis
scorer = ReadabilityScorer()
metrics = scorer.analyze_text(content, "overview")
recommendations = scorer.get_recommendations("overview")
```

## Error Handling

All components use custom exceptions for specific error cases:

```python
class OllamaClientError(Exception):
    """Raised for Ollama API related errors."""

class CacheError(Exception):
    """Raised for caching related errors."""

class ValidationError(Exception):
    """Raised for validation related errors."""
```

## Configuration

See [README.md](../README.md#configuration) for detailed configuration options. 

### Error Handling

The cache system is designed to fail gracefully:

1. **Missing Directory**: Creates `.cache` directory if it doesn't exist
2. **Database Initialization**: Creates database and tables if missing
3. **Access Errors**: Disables caching instead of failing
4. **Concurrent Access**: Handles multiple processes safely