# Standard library imports
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Callable

# Third-party imports
import magic
import networkx as nx
from gitignore_parser import parse_gitignore
from tqdm import tqdm

# Local imports
from ..models.file_info import FileInfo
from ..utils.cache import CacheManager  # Use the standard cache module
from ..utils.progress import ProgressTracker
from ..utils.config_class import ScribeConfig

class CodebaseAnalyzer:
    """Analyzes repository structure and content.
    
    This class is responsible for scanning a repository, analyzing its files,
    and building a comprehensive file manifest with metadata. It handles gitignore
    rules, binary file detection, and can extract information about exports and
    dependencies from source code files.
    
    Attributes:
        repo_path: Path to the repository root
        config: Configuration dictionary
        debug: Whether debug mode is enabled
        file_manifest: Dictionary mapping file paths to FileInfo objects
        graph: Dependency graph of files in the repository
    """
    
    # Binary file detection constants
    BINARY_MIME_PREFIXES = ('text/', 'application/json', 'application/xml')
    BINARY_CHECK_BYTES = 1024
    
    # File extension constants
    SOURCE_CODE_EXTENSIONS = {'.py', '.js', '.ts', '.cs', '.java'}
    
    # Special files that are always included
    SPECIAL_FILES = {"README.md", "ARCHITECTURE.md", "CONTRIBUTING.md"}
    
    # Special directories that are always included
    SPECIAL_DIRS = {".github"}
    
    def __init__(self, repo_path: Path, config: ScribeConfig):
        # Windows-specific normalization
        if os.name == 'nt':
            self.repo_path = Path(os.path.normpath(str(repo_path))).absolute()
        else:
            self.repo_path = Path(repo_path).absolute()
            
        # Store config
        self.config_obj = config
            
        # Initialize debug flag first before using it in other methods
        self.debug = self.config_obj.debug
        
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger('codebase_analyzer')
            self.logger.debug(f"Initialized analyzer for repo: {self.repo_path}")
        
        # Validate repository path early to prevent issues with cache initialization
        if not self.repo_path.exists():
            error_msg = f"Repository path does not exist: {self.repo_path}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        if not self.repo_path.is_dir():
            error_msg = f"Repository path is not a directory: {self.repo_path}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Now load gitignore after debug is initialized
        self.gitignore = self._load_gitignore()
        
        self.graph = nx.DiGraph()
        self.file_manifest: Dict[str, FileInfo] = {}
        
        # Set up cache using github_repo_id if available
        if hasattr(self.config_obj, 'github_repo_id') and self.config_obj.github_repo_id:
            # Use stable GitHub repo ID from ScribeConfig
            repo_identifier = self.config_obj.github_repo_id
            if self.debug:
                logging.debug(f"Using GitHub repo ID for caching: {repo_identifier}")
        else:
            # Use repository path as identifier
            repo_identifier = str(self.repo_path)
        
        # Initialize cache with ScribeConfig
        self.cache = CacheManager(
            enabled=not self.config_obj.no_cache,
            repo_identifier=repo_identifier,
            repo_path=self.repo_path,
            config=self.config_obj
        )
        
        # Set debug mode on cache if needed
        self.cache.debug = self.debug
        
        # Initialize blacklist from config
        self.blacklist_extensions = set(self.config_obj.blacklist.extensions)
        self.blacklist_patterns = self.config_obj.blacklist.path_patterns
        
        if self.debug:
            logging.debug(f"Blacklist extensions: {self.blacklist_extensions}")
            logging.debug(f"Blacklist patterns: {self.blacklist_patterns}")
        
    def _load_gitignore(self):
        """Load all .gitignore files from the repository"""
        try:
            ignore_funcs = []
            # Find all .gitignore files in the repository
            for gitignore_path in self.repo_path.rglob('.gitignore'):
                if gitignore_path.is_file():
                    if self.debug:
                        logging.debug(f"Found .gitignore file: {gitignore_path}")
                    # Get relative path to repo root
                    rel_path = gitignore_path.parent.relative_to(self.repo_path)
                    ignore_func = parse_gitignore(gitignore_path, base_dir=str(self.repo_path))
                    ignore_funcs.append(ignore_func)
                    
                    # Print the first few patterns from this gitignore for debugging
                    if self.debug:
                        try:
                            with open(gitignore_path, 'r') as f:
                                patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')][:5]
                                logging.debug(f"Sample patterns from {gitignore_path}: {patterns}")
                        except Exception as e:
                            logging.error(f"Error reading gitignore patterns: {e}")
            
            # Combine all gitignore rules
            def combined_ignore(path):
                # Convert to absolute path if it's not already
                if not isinstance(path, Path):
                    path = Path(path)
                    
                if not path.is_absolute():
                    abs_path = (self.repo_path / path).absolute()
                else:
                    abs_path = path.absolute()
                    
                # Get path relative to repo root for gitignore matching
                try:
                    rel_path = abs_path.relative_to(self.repo_path)
                    path_str = str(rel_path)
                    
                    # Check if any ignore function matches
                    for fn in ignore_funcs:
                        if fn(path_str):
                            if self.debug:
                                logging.debug(f"Ignoring {path_str} due to gitignore rule")
                            return True
                    return False
                except ValueError:
                    # Path is not relative to repo_path
                    return False
            
            return combined_ignore
            
        except Exception as e:
            if self.debug:
                logging.error(f"Error loading .gitignore: {e}")
            return lambda _: False

    def is_binary(self, file_path: Path) -> bool:
        """Check if a file is binary.
        
        Uses the python-magic library to determine if a file is binary based on its MIME type.
        Falls back to a simple binary check if python-magic fails.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if the file is binary, False otherwise
        """
        try:
            # Check if file is readable and exists
            if not os.access(str(file_path), os.R_OK) or not file_path.exists():
                if self.debug:
                    logging.debug(f"File not accessible or doesn't exist: {file_path}")
                return False
                
            # Use python-magic to determine MIME type
            mime = magic.from_file(str(file_path), mime=True)
            is_binary = not mime.startswith(self.BINARY_MIME_PREFIXES)
            
            if self.debug and is_binary:
                logging.debug(f"Detected binary file: {file_path} (MIME: {mime})")
                
            return is_binary
            
        except (OSError, PermissionError) as e:
            if self.debug:
                logging.error(f"Error checking if file is binary: {e}")
            # Fall back to simple binary check
            return self._is_binary(file_path)
        except Exception as e:
            # Handle regex memory errors or other unexpected errors
            error_str = str(e)
            if "regex error" in error_str and "failed to get memory" in error_str:
                logging.warning(f"Regex memory error when checking file type for {file_path}. Falling back to extension-based check.")
                # For C++ files that might trigger the regex memory error
                ext = file_path.suffix.lower()
                if ext in ['.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx']:
                    return False  # Treat C++ files as text
                # Fall back to simple binary check
                return self._is_binary(file_path)
            else:
                # For other unexpected errors, log and fall back
                logging.error(f"Unexpected error checking if file is binary: {e}")
                return self._is_binary(file_path)

    def should_include_file(self, file_path: Path) -> bool:
        """Determine if a file should be included in analysis.
        
        This unified method replaces both should_ignore and _should_include_file,
        providing a single point of decision for file inclusion with clear rules.
        
        Args:
            file_path: Path to the file, relative to the repository root
            
        Returns:
            bool: True if the file should be included, False otherwise
        """
        # Step 1: Always include special files and directories
        if file_path.name in self.SPECIAL_FILES or any(file_path.name.endswith(f) for f in self.SPECIAL_FILES):
            if self.debug:
                logging.debug(f"Including special file: {file_path}")
            return True
            
        if any(dir_name in file_path.parts for dir_name in self.SPECIAL_DIRS):
            if self.debug:
                logging.debug(f"Including file in special directory: {file_path}")
            return True
            
        # Step 2: Check for files that should always be excluded
        
        # Skip hidden files/directories except .gitignore
        if any(part.startswith('.') and part != '.gitignore' for part in file_path.parts):
            if self.debug:
                logging.debug(f"Excluding hidden file/directory: {file_path}")
            return False
            
        # Skip files with blacklisted extensions
        if file_path.suffix.lower() in self.blacklist_extensions:
            if self.debug:
                logging.debug(f"Excluding file with blacklisted extension: {file_path}")
            return False
            
        # Skip files matching blacklisted path patterns
        for pattern in self.blacklist_patterns:
            if re.search(pattern, str(file_path)):
                if self.debug:
                    logging.debug(f"Excluding file matching blacklist pattern '{pattern}': {file_path}")
                return False
                
        # Step 3: Check gitignore rules
        path_str = str(file_path)
        if self.gitignore(path_str):
            if self.debug:
                logging.debug(f"Excluding file due to gitignore rules: {file_path}")
            return False
            
        # If we've passed all exclusion checks, include the file
        return True

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
        # Tell the cache manager about our repository path
        if hasattr(self.cache, 'repo_path'):
            self.cache.repo_path = self.repo_path
        
        # Repository path is already validated in the constructor
        
        try:
            # Get all files in repository using the _get_repository_files method
            all_files = self._get_repository_files()
            
            # Debug output for file detection
            print(f"Found {len(all_files)} files to analyze")
            if len(all_files) == 0:
                error_msg = "No files found in repository"
                logging.warning(error_msg)
                logging.debug(f"Repository path: {self.repo_path}")
                logging.debug(f"Repository contents: {list(self.repo_path.iterdir())}")
                return {}
            
            # Test mode - limit to first 5 files
            test_mode = self.config_obj.test_mode
            
            if test_mode:
                all_files = all_files[:5]
                if self.debug:
                    logging.debug(f"Test mode enabled, limiting to {len(all_files)} files")
            
            # Process files with or without progress bar
            if show_progress:
                try:
                    # Get or create the progress tracker instance
                    progress_tracker = ProgressTracker.get_instance(self.repo_path)
                    with progress_tracker.progress_bar(
                        desc="Analyzing repository",
                        total=len(all_files),
                        unit="files"
                    ) as pbar:
                        # Process each file with progress bar
                        for file_path in all_files:
                            try:
                                rel_path = file_path.relative_to(self.repo_path)
                                
                                # Skip files that should not be included
                                if not self.should_include_file(rel_path):
                                    continue
                                    
                                # Analyze file
                                file_info = self._analyze_file(file_path)
                                
                                # Add to manifest
                                self.file_manifest[str(rel_path)] = file_info
                                
                                # Update progress bar
                                pbar.update(1)
                            except Exception as e:
                                # Log error but continue processing other files
                                logging.error(f"Error processing file {file_path}: {e}")
                                if self.debug:
                                    import traceback
                                    logging.debug(traceback.format_exc())
                                # Still update progress bar
                                pbar.update(1)
                except Exception as e:
                    # Fall back to regular iteration if progress bar fails
                    logging.warning(f"Failed to create progress bar: {e}")
                    # Process each file without progress bar
                    for file_path in all_files:
                        try:
                            rel_path = file_path.relative_to(self.repo_path)
                            
                            # Skip files that should not be included
                            if not self.should_include_file(rel_path):
                                continue
                                
                            # Analyze file
                            file_info = self._analyze_file(file_path)
                            
                            # Add to manifest
                            self.file_manifest[str(rel_path)] = file_info
                        except Exception as e:
                            # Log error but continue processing other files
                            logging.error(f"Error processing file {file_path}: {e}")
                            if self.debug:
                                import traceback
                                logging.debug(traceback.format_exc())
            else:
                # Process each file without progress bar
                for file_path in all_files:
                    try:
                        rel_path = file_path.relative_to(self.repo_path)
                        
                        # Skip files that should not be included
                        if not self.should_include_file(rel_path):
                            continue
                            
                        # Analyze file
                        file_info = self._analyze_file(file_path)
                        
                        # Add to manifest
                        self.file_manifest[str(rel_path)] = file_info
                    except Exception as e:
                        # Log error but continue processing other files
                        logging.error(f"Error processing file {file_path}: {e}")
                        if self.debug:
                            import traceback
                            logging.debug(traceback.format_exc())
                
            if self.debug:
                logging.debug(f"Analyzed {len(self.file_manifest)} files")
                
            # Print a summary of the analysis
            print(f"Successfully analyzed {len(self.file_manifest)} files")
            
            return self.file_manifest
            
        except Exception as e:
            error_msg = f"Error analyzing repository: {e}"
            logging.error(error_msg)
            if self.debug:
                import traceback
                logging.debug(traceback.format_exc())
            print(error_msg)
            return {}

    def _is_binary(self, file_path: Path) -> bool:
        """Simple binary file detection by checking for null bytes.
        
        This is a fallback method used when python-magic fails.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if the file contains null bytes (likely binary), False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                return b'\0' in f.read(self.BINARY_CHECK_BYTES)
        except Exception as e:
            if self.debug:
                self.logger.error(f"Error reading {file_path}: {e}")
            # If we can't read the file, assume it's binary to be safe
            return True

    # Method removed: _should_include_file has been merged with should_ignore into should_include_file

    def _get_repository_files(self) -> list[Path]:
        """Get all files in repository that should be analyzed."""
        files = []
        try:
            for file_path in self.repo_path.rglob('*'):
                if not file_path.is_file():  # Skip directories
                    continue
                    
                # Get path relative to repo root for filtering
                rel_path = file_path.relative_to(self.repo_path)
                
                # Check if file should be included
                if not self.should_include_file(rel_path):
                    if self.debug:
                        logging.debug(f"Excluding file due to inclusion rules: {rel_path}")
                    continue
                    
                logging.debug(f"Including file for analysis: {rel_path}")
                files.append(file_path)
                
            if self.debug:
                logging.debug(f"Found {len(files)} files to analyze")
                
            return sorted(files)  # Sort for consistent ordering
            
        except Exception as e:
            if self.debug:
                logging.error(f"Error getting repository files: {e}")
            return [] 

    def _analyze_file(self, file_path: Path) -> FileInfo:
        """Analyze a single file and return FileInfo object.
        
        This method creates a FileInfo object for the given file, determines if it's binary,
        reads its content if it's a text file, and extracts exports and dependencies
        for source code files.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileInfo: Object containing metadata about the file
        """
        try:
            # Create basic FileInfo object
            file_info = FileInfo(
                path=file_path.relative_to(self.repo_path),
                is_binary=self.is_binary(file_path)
            )
            
            # Skip detailed analysis for binary files
            if file_info.is_binary:
                return file_info
            
            # Try to read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_info.content = content
            except UnicodeDecodeError:
                # If we can't read as text, mark as binary
                file_info.is_binary = True
                return file_info
            
            # Extract exports (functions, classes, etc.) for source code files
            if file_path.suffix in self.SOURCE_CODE_EXTENSIONS:
                file_info.exports = self._extract_exports(content)
            
            # Extract dependencies (imports, requires, etc.)
            if hasattr(file_info, 'exports'):
                file_info.dependencies = self._extract_dependencies(content)
            
            # Update dependency graph
            for dep in getattr(file_info, 'dependencies', []):
                self.graph.add_edge(str(file_info.path), dep)
            
            return file_info
            
        except Exception as e:
            if self.debug:
                logging.error(f"Error analyzing file {file_path}: {e}")
            # Return basic FileInfo on error
            return FileInfo(
                path=file_path.relative_to(self.repo_path),
                is_binary=False
            )

    def _extract_exports(self, content: str) -> set[str]:
        """Extract exported symbols from file content."""
        exports = set()
        
        # Simple regex patterns for common exports
        patterns = [
            r'(?:^|\s)class\s+(\w+)',  # Classes
            r'(?:^|\s)def\s+(\w+)',    # Python functions
            r'function\s+(\w+)',       # JavaScript functions
            r'export\s+(?:const|let|var|function|class)\s+(\w+)',  # JS/TS exports
            r'public\s+(?:class|interface|enum)\s+(\w+)',  # C#/Java
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            exports.update(match.group(1) for match in matches)
        
        return exports

    def _extract_dependencies(self, content: str) -> set[str]:
        """Extract dependencies from file content."""
        deps = set()
        
        # Simple regex patterns for common imports
        patterns = [
            r'import\s+(\w+)',         # Python/Java imports
            r'from\s+(\S+)\s+import',  # Python from imports
            r'require\([\'"](.+?)[\'"]\)',  # Node.js requires
            r'using\s+(.+?);',         # C# using statements
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            deps.update(match.group(1) for match in matches)
        
        return deps
        
    def read_file(self, file_path: Path) -> str:
        """Read file content with proper encoding handling.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            str: Content of the file
            
        Raises:
            UnicodeDecodeError: If the file cannot be decoded as UTF-8
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read due to permissions
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                if self.debug:
                    logging.error(f"Error reading file {file_path} with latin-1 encoding: {e}")
                raise
        except Exception as e:
            if self.debug:
                logging.error(f"Error reading file {file_path}: {e}")
            raise
            
    def get_file_language(self, file_path: Path) -> str:
        """Determine the programming language of a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Language name (lowercase) or 'text' if unknown
        """
        # Extension to language mapping
        extension_map = {
            # Python
            '.py': 'python',
            '.pyw': 'python',
            '.pyx': 'python',
            
            # JavaScript
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.mjs': 'javascript',
            
            # TypeScript
            '.ts': 'typescript',
            '.tsx': 'typescript',
            
            # Web
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            
            # Java
            '.java': 'java',
            '.class': 'java',
            '.jar': 'java',
            
            # C#
            '.cs': 'csharp',
            '.csx': 'csharp',
            
            # C/C++
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            
            # Ruby
            '.rb': 'ruby',
            '.erb': 'ruby',
            
            # PHP
            '.php': 'php',
            
            # Go
            '.go': 'go',
            
            # Rust
            '.rs': 'rust',
            
            # Swift
            '.swift': 'swift',
            
            # Shell
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            
            # Markdown
            '.md': 'markdown',
            '.markdown': 'markdown',
            
            # JSON
            '.json': 'json',
            
            # YAML
            '.yml': 'yaml',
            '.yaml': 'yaml',
            
            # XML
            '.xml': 'xml',
            '.pom': 'xml',
            
            # SQL
            '.sql': 'sql',
            
            # Docker
            'Dockerfile': 'dockerfile',
            '.dockerfile': 'dockerfile',
            
            # Configuration
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.properties': 'properties',
            '.toml': 'toml',
        }
        
        # Check if the file is a special case like Dockerfile
        if file_path.name in extension_map:
            return extension_map[file_path.name]
            
        # Get the file extension (lowercase)
        ext = file_path.suffix.lower()
        
        # Return the language based on extension, or 'text' if unknown
        return extension_map.get(ext, 'text')

    def check_markdown_headers(self, content: str) -> list[str]:
        """Check markdown header formatting and hierarchy.
        
        This method analyzes markdown content to identify potential issues with headers,
        such as excessive nesting, missing spaces, or improper capitalization.
        
        Args:
            content: The markdown content to analyze
            
        Returns:
            list[str]: A list of issues found in the headers
        """
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Skip lines that aren't headers
            match = re.match(r'^#+', line)
            if not match:
                continue
            
            level = len(match.group())
            
            # Check maximum header level (usually shouldn't go beyond h4 or h5)
            if level > 5:
                issues.append(f"Header on line {i+1} has too many # symbols (level {level})")
            
            # Check for space after #
            if not re.match(r'^#+\s', line):
                issues.append(f"Header on line {i+1} is missing a space after # symbols")
            
            # Check for capitalization
            header_text = line.lstrip('#').strip()
            if header_text and not header_text[0].isupper():
                issues.append(f"Header on line {i+1} should start with a capital letter")
            
        return issues

    def analyze_python_files(self) -> Dict[Path, FileInfo]:
        """Analyze Python files in the repository.
        
        This method filters the file manifest to include only Python files
        and returns a dictionary mapping file paths to their FileInfo objects.
        
        Returns:
            Dict[Path, FileInfo]: Dictionary of Python files and their metadata
        """
        python_files = {}
        
        for path_str, file_info in self.file_manifest.items():
            path = Path(path_str)
            if path.suffix == '.py':
                python_files[path] = file_info
                
        if self.debug:
            logging.debug(f"Found {len(python_files)} Python files")
            
        return python_files
        
    def build_dependency_graph(self) -> nx.DiGraph:
        """Build and return the dependency graph of files in the repository.
        
        This method returns the graph that was built during repository analysis,
        representing dependencies between files based on imports and requires.
        
        Returns:
            nx.DiGraph: Directed graph of file dependencies
        """
        if self.debug:
            logging.debug(f"Dependency graph has {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            
        return self.graph
    
    def derive_project_name(self, debug: bool = False) -> str:
        """Derive project name from repository structure.
        
        This method attempts to determine the project name by examining various files
        in the repository, such as package.json, setup.py, pom.xml, etc. It also looks
        for patterns in directory structure and namespace declarations.
        
        Args:
            debug: Whether to print debug information
            
        Returns:
            str: The derived project name, or "Project" if no name could be determined
        """
        try:
            # Dictionary of file patterns and their corresponding regex patterns
            name_patterns = {
                # JavaScript/Node.js
                'package.json': r'"name"\s*:\s*"([^"]+)"',
                
                # Python
                'setup.py': r'name=[\'"]([^\'"]+)[\'"]',
                'pyproject.toml': r'name\s*=\s*[\'"]([^\'"]+)[\'"]',
                
                # Java/Kotlin
                'pom.xml': r'<artifactId>([^<]+)</artifactId>|<name>([^<]+)</name>',
                'build.gradle': r'rootProject\.name\s*=\s*[\'"]([^\'"]+)[\'"]',
                'settings.gradle': r'rootProject\.name\s*=\s*[\'"]([^\'"]+)[\'"]',
                
                # C#/.NET
                '.csproj': r'<AssemblyName>([^<]+)</AssemblyName>|<RootNamespace>([^<]+)</RootNamespace>',
                
                # Ruby
                'Gemfile': r'^\s*#\s*([^\s]+)\s*$|gemspec.*name\s*=\s*[\'"]([^\'"]+)[\'"]',
                '.gemspec': r'\.name\s*=\s*[\'"]([^\'"]+)[\'"]',
                
                # Go
                'go.mod': r'module\s+([^\s]+)',
                
                # Rust
                'Cargo.toml': r'name\s*=\s*[\'"]([^\'"]+)[\'"]',
                
                # PHP
                'composer.json': r'"name"\s*:\s*"([^"]+)"',
            }
            
            # Check for each file pattern
            for file_pattern, regex_pattern in name_patterns.items():
                for path, info in self.file_manifest.items():
                    # Handle both dictionary and object access
                    is_binary = info.get('is_binary', False) if hasattr(info, 'get') else getattr(info, 'is_binary', False)
                    
                    if str(path).endswith(file_pattern) and not is_binary:
                        try:
                            # Handle both dictionary and object access
                            content = info.get('content', '') if hasattr(info, 'get') else getattr(info, 'content', '')
                            
                            if content:
                                # Special handling for pom.xml to prioritize the name tag over artifactId
                                if file_pattern == 'pom.xml':
                                    # First try to find the name tag
                                    name_match = re.search(r'<name>([^<]+)</name>', content)
                                    if name_match and name_match.group(1).strip():
                                        return name_match.group(1).strip()
                                    
                                    # If no name tag, look for artifactId at the project level (not in dependencies)
                                    # Find the main artifactId (not in a dependency section)
                                    artifact_matches = list(re.finditer(r'<artifactId>([^<]+)</artifactId>', content))
                                    for match in artifact_matches:
                                        # Check if this artifactId is not inside a dependency
                                        pos = match.start()
                                        section_before = content[:pos]
                                        # If the last opening tag before this is not a dependency, it's likely the project artifactId
                                        if '<dependency>' not in section_before[-500:]:
                                            return match.group(1)
                                else:
                                    # Extract name using regex for other file types
                                    match = re.search(regex_pattern, content)
                                    if match:
                                        # Get the first non-empty group
                                        for group in match.groups():
                                            if group:
                                                # Clean up the name (remove organization prefixes for some formats)
                                                name = group.split('/')[-1] if '/' in group else group
                                                return name
                        except Exception as e:
                            if debug:
                                print(f"Error parsing {path} for name: {e}")
                                logging.error(f"Error parsing {path} for name: {e}")
            
            # Look for namespace declarations in C# files - FIXED: Access content directly
            for path, info in self.file_manifest.items():
                if str(path).endswith('.cs'):
                    try:
                        # Access content directly from the FileInfo object
                        content = info.content if hasattr(info, 'content') else ""
                        if content:
                            namespace_match = re.search(r'namespace\s+([^\s.;{]+)', content)
                            if namespace_match:
                                return namespace_match.group(1)
                    except Exception as e:
                        if debug:
                            print(f"Error parsing C# file for namespace: {e}")
                            logging.error(f"Error parsing C# file for namespace: {e}")
            
            # If no name found in config files, try to derive from directory structure
            # Look for src/main/java/com/company/project pattern (common in Java)
            java_pattern = re.compile(r'src/main/java/([^/]+)/([^/]+)/([^/]+)')
            for path in self.file_manifest.keys():
                path_str = str(path)
                match = java_pattern.search(path_str)
                if match:
                    # Use the last component as the project name
                    return match.group(3)
            
            # Try to derive from directory name (use the repository root name)
            # This is a fallback if no other method works
            for path in self.file_manifest.keys():
                path_str = str(path)
                parts = path_str.split('/')
                if len(parts) > 1:
                    # The first part is usually the repository name
                    return parts[0]
            
            # Default fallback
            return "Project"
        except Exception as e:
            if debug:
                print(f"Error deriving project name: {e}")
            logging.error(f"Error deriving project name: {e}")
            return "Project"