from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union
from ..utils.tokens import TokenCounter
from ..utils.config_class import ScribeConfig

class BaseLLMClient(ABC):
    """
    Base abstract class for LLM clients.
    
    This class defines the interface that all LLM client implementations must follow.
    It provides abstract methods for interacting with language models to generate
    documentation and analyze code.
    
    Version: 1.0.0
    
    Example:
        ```python
        class MyLLMClient(BaseLLMClient):
            async def initialize(self):
                # Implementation here
                pass
                
            # Implement other required methods
        
        # Usage
        # With dictionary config
        client = MyLLMClient(config_dict)
        await client.initialize()
        overview = await client.generate_project_overview(file_manifest)
        
        # Or with ScribeConfig
        from src.utils.config_class import ScribeConfig
        config = ScribeConfig.from_dict(config_dict)
        client = MyLLMClient(config)
        await client.initialize()
        overview = await client.generate_project_overview(file_manifest)
        ```
    """
    
    # Class constants
    VERSION = "1.0.0"
    
    def __init__(self):
        """Initialize the base client."""
        self.token_counter = None
        self.project_structure = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the client.
        
        This method should handle any setup required for the LLM client,
        such as establishing connections, loading models, or setting up
        authentication.
        
        Example:
            ```python
            async def initialize(self):
                self.client = SomeLLMLibrary(api_key=self.api_key)
                self.init_token_counter()
            ```
        """
        pass
    
    @abstractmethod
    def init_token_counter(self) -> None:
        """
        Initialize the token counter for this client.
        
        This method should set up the TokenCounter instance with the
        appropriate model name and configuration.
        
        Example:
            ```python
            def init_token_counter(self):
                self.token_counter = TokenCounter(model_name=self.model_name)
            ```
        """
        pass
    
    def validate_input(self, text: str) -> bool:
        """
        Validate input text before sending to the LLM.
        
        Args:
            text: The input text to validate
            
        Returns:
            bool: True if the input is valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        return True
    
    def validate_file_manifest(self, file_manifest: Dict[str, Any]) -> bool:
        """
        Validate the file manifest structure.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            bool: True if the manifest is valid, False otherwise
        """
        if not isinstance(file_manifest, dict):
            return False
        return True
    
    @abstractmethod
    async def generate_summary(self, content: str, file_type: str = "text", file_path: str = None) -> Optional[str]:
        """
        Generate a summary for a file's content.
        
        This method processes the content of a file and produces a concise
        summary describing its purpose and functionality.
        
        Args:
            content: The content of the file to summarize
            file_type: The type/language of the file (default: "text")
            file_path: The path to the file (default: None)
            
        Returns:
            Optional[str]: Generated summary or None if generation fails
            
        Example:
            ```python
            summary = await client.generate_summary(
                "def hello(): print('Hello world')",
                file_type="python",
                file_path="src/hello.py"
            )
            # Returns: "A function that prints 'Hello world'"
            ```
        """
        pass
    
    @abstractmethod
    async def generate_project_overview(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate project overview based on file manifest.
        
        This method analyzes the project structure and files to create a
        comprehensive overview of the project's purpose and components.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated project overview
            
        Example:
            ```python
            overview = await client.generate_project_overview({
                "src/main.py": {"content": "...", "summary": "Main entry point"},
                "src/utils.py": {"content": "...", "summary": "Utility functions"}
            })
            ```
        """
        pass
    
    @abstractmethod
    async def generate_usage_guide(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate usage guide based on project structure.
        
        This method creates documentation explaining how to use the project,
        including installation, configuration, and common operations.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated usage guide
            
        Example:
            ```python
            guide = await client.generate_usage_guide(file_manifest)
            ```
        """
        pass
    
    @abstractmethod
    async def generate_contributing_guide(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate contributing guide based on project structure.
        
        This method creates documentation explaining how to contribute to the project,
        including coding standards, pull request process, and development setup.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated contributing guide
            
        Example:
            ```python
            guide = await client.generate_contributing_guide(file_manifest)
            ```
        """
        pass
    
    @abstractmethod
    async def generate_license_info(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate license information based on project structure.
        
        This method analyzes the project to determine its license and creates
        appropriate license information documentation.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated license information
            
        Example:
            ```python
            license_info = await client.generate_license_info(file_manifest)
            ```
        """
        pass
    
    @abstractmethod
    async def generate_architecture_content(self, file_manifest: Dict[str, Any], analyzer: Any) -> str:
        """
        Generate architecture documentation content.
        
        This method creates comprehensive documentation about the project's
        architecture, including component diagrams and design patterns.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            analyzer: CodebaseAnalyzer instance for additional analysis
            
        Returns:
            str: Generated architecture documentation
            
        Example:
            ```python
            architecture = await client.generate_architecture_content(
                file_manifest, codebase_analyzer
            )
            ```
        """
        pass
    
    @abstractmethod
    async def generate_component_relationships(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate description of how components interact.
        
        This method analyzes the relationships between different components
        in the project and describes their interactions.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated component relationship description
            
        Example:
            ```python
            relationships = await client.generate_component_relationships(file_manifest)
            ```
        """
        pass
    
    @abstractmethod
    async def enhance_documentation(self, existing_content: str, file_manifest: Dict[str, Any], doc_type: str) -> str:
        """
        Enhance existing documentation with new insights.
        
        This method takes existing documentation and improves it based on
        analysis of the codebase.
        
        Args:
            existing_content: The existing documentation content
            file_manifest: Dictionary mapping file paths to file information
            doc_type: Type of documentation being enhanced (e.g., "README", "ARCHITECTURE")
            
        Returns:
            str: Enhanced documentation content
            
        Example:
            ```python
            enhanced = await client.enhance_documentation(
                "# Project\nThis is a project.", file_manifest, "README"
            )
            ```
        """
        pass
    
    @abstractmethod
    def set_project_structure(self, structure: str) -> None:
        """
        Set the project structure for use in prompts.
        
        This method stores a string representation of the project structure
        to provide context for LLM prompts.
        
        Args:
            structure: String representation of the project structure
            
        Example:
            ```python
            client.set_project_structure("src/\n  main.py\n  utils.py\ntests/\n  test_main.py")
            ```
        """
        pass
    
    @abstractmethod
    async def get_file_order(self, project_files: Dict[str, Any]) -> List[str]:
        """
        Determine optimal file processing order.
        
        This method analyzes dependencies between files to determine the
        most efficient order for processing them.
        
        Args:
            project_files: Dictionary mapping file paths to file information
            
        Returns:
            List[str]: List of file paths in optimal processing order
            
        Example:
            ```python
            order = await client.get_file_order({
                "src/utils.py": {...},
                "src/main.py": {...}
            })
            # Returns: ["src/utils.py", "src/main.py"]
            ```
        """
        pass