from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Set, Optional, List, Union, Dict, Any

@dataclass
class FileInfo:
    """Information about a source file in the repository.
    
    This class serves as a core data structure for representing source files
    and their metadata throughout the application.
    
    Attributes:
        path: Path to the file, either as string or Path object
        is_binary: Whether the file is binary or text
        size: Size of the file in bytes
        language: Programming language of the file
        content: Text content of the file
        summary: Brief summary of the file's purpose
        file_type: Type of the file (e.g., "python", "markdown")
        last_modified: Timestamp of last modification
        imports: Collection of imports found in the file
        exports: Collection of exports (functions, classes) found in the file
        from_cache: Whether the file info was loaded from cache
    """
    path: Union[str, Path]  # Support both string and Path objects
    is_binary: bool = False
    size: int = 0
    language: str = ""  # Fixed: Changed from None to empty string for consistency
    content: str = ""  # Fixed: Changed from None to empty string for consistency
    summary: Optional[str] = None
    file_type: str = ""
    last_modified: float = 0.0
    imports: Union[List[str], Set[str]] = None
    exports: Union[List[str], Set[str]] = None
    from_cache: bool = False  # Track if summary came from cache
    
    def __post_init__(self):
        """Initialize the FileInfo object after creation.
        
        This method:
        1. Initializes empty collections for imports and exports if they are None
        2. Converts string paths to Path objects
        3. Validates that the path field is not empty
        """
        # Validate required fields
        if not self.path:
            raise ValueError("File path cannot be empty")
            
        # Initialize empty collections if None
        if self.imports is None:
            self.imports: Set[str] = set()
        if self.exports is None:
            self.exports: Set[str] = set()
        
        # Convert path to Path object if it's a string
        if isinstance(self.path, str):
            self.path = Path(self.path)
    
    def __repr__(self) -> str:
        """Return a string representation of the FileInfo object.
        
        Returns:
            A string with key information about the file
        """
        return f"FileInfo(path='{self.path}', type='{self.file_type}', binary={self.is_binary})"
    
    def is_language(self, lang: str) -> bool:
        """Check if the file is of a specific language.
        
        Args:
            lang: Language to check against
            
        Returns:
            True if the file is of the specified language, False otherwise
        """
        return self.language and self.language.lower() == lang.lower()
    
    def has_extension(self, ext: str) -> bool:
        """Check if the file has a specific extension.
        
        Args:
            ext: Extension to check (with or without leading dot)
            
        Returns:
            True if the file has the specified extension, False otherwise
        """
        if not ext.startswith('.'):
            ext = f'.{ext}'
        return self.path.suffix.lower() == ext.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the FileInfo object to a dictionary.
        
        Returns:
            Dictionary representation of the FileInfo object
        """
        result = asdict(self)
        # Convert Path to string for serialization
        result['path'] = str(self.path)
        # Convert sets to lists for serialization
        if isinstance(self.imports, set):
            result['imports'] = list(self.imports)
        if isinstance(self.exports, set):
            result['exports'] = list(self.exports)
        return result