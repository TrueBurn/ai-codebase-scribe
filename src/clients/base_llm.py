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
        self.related_repo_context = None

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

    def _validate_file_summary(
        self, summary: Optional[str], file_path: Optional[str] = None
    ) -> tuple:
        """
        Validate that a file summary contains actual analysis, not system prompts.

        This method performs multi-layer validation to ensure summary quality:
        1. Basic checks (not None, minimum length)
        2. System prompt detection (negative patterns)
        3. Structure validation (positive patterns)

        Args:
            summary: The summary text to validate
            file_path: Optional file path for logging context

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, failure_reason)
            - (True, None) if valid
            - (False, "reason") if invalid

        Example:
            ```python
            is_valid, reason = self._validate_file_summary(summary, file_path)
            if not is_valid:
                logger.warning(f"Invalid summary for {file_path}: {reason}")
                return None
            ```
        """
        import re

        # Check 1: Not None or too short
        if not summary or not isinstance(summary, str):
            return False, "empty_or_none"

        if len(summary.strip()) < 50:
            return False, "too_short"

        # Check 2: System prompt markers (CRITICAL - prevents caching system prompts)
        system_markers = [
            r"I'm ready to conduct",
            r"Please paste the code",
            r"TASK:\s*Conduct systematic",
            r"You are an expert",
            r"waiting for your code",
            r"I'll analyze the code",
            r"I'll conduct",
            r"I will conduct",
        ]

        for pattern in system_markers:
            if re.search(pattern, summary, re.IGNORECASE):
                return False, f"system_prompt:{pattern}"

        # Check 3: Has expected analysis structure (lenient check)
        # Look for common documentation headers or technical content
        required_markers = ["##", "PRIMARY FUNCTION", "COMPONENT"]
        has_structure = any(marker in summary.upper() for marker in required_markers)

        if not has_structure:
            return False, "missing_structure"

        return True, None

    def set_related_repo_context(
        self, related_repo_data: Optional[Dict[str, Any]]
    ) -> None:
        """
        Set related repository context (config and GitOps data).

        Args:
            related_repo_data: Dictionary containing config and GitOps repository data
        """
        self.related_repo_context = related_repo_data

    @abstractmethod
    async def generate_summary(
        self, content: str, file_type: str = "text", file_path: str = None
    ) -> Optional[str]:
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

    @abstractmethod
    async def generate_installation_guide(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate installation guide based on project structure.

        This method creates documentation explaining how to install and set up the project,
        including prerequisites, installation steps, and verification procedures.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated installation guide

        Example:
            ```python
            guide = await client.generate_installation_guide(file_manifest)
            ```
        """

    @abstractmethod
    async def generate_troubleshooting_guide(
        self, file_manifest: Dict[str, Any]
    ) -> str:
        """
        Generate troubleshooting guide based on project structure.

        This method creates documentation explaining common issues, debugging procedures,
        and solutions for problems users might encounter.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated troubleshooting guide

        Example:
            ```python
            guide = await client.generate_troubleshooting_guide(file_manifest)
            ```
        """

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

    @abstractmethod
    async def generate_architecture_content(
        self, file_manifest: Dict[str, Any], analyzer: Any
    ) -> str:
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

    @abstractmethod
    async def generate_component_relationships(
        self, file_manifest: Dict[str, Any]
    ) -> str:
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

    @abstractmethod
    async def analyze_migration_contents(
        self, migration_contents: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze migration file contents to extract detailed schema information.

        This method uses LLM intelligence to parse migration files and extract
        comprehensive database schema details including tables, columns, relationships,
        indexes, views, stored procedures, and constraints.

        Args:
            migration_contents: List of migration file info dictionaries containing:
                - file_path: Path to the migration file
                - file_name: Name of the migration file
                - content: Full content of the migration file
                - type: Type of migration (sql, java, etc.)

        Returns:
            Dict containing extracted schema information:
                - tables: List of table definitions with columns, constraints
                - relationships: List of foreign key relationships
                - indexes: List of index definitions
                - views: List of view definitions
                - procedures: List of stored procedures/functions
                - triggers: List of trigger definitions
        """

    @abstractmethod
    async def generate_persistence_doc(
        self, file_manifest: Dict[str, Any], persistence_info: Any
    ) -> str:
        """
        Generate persistence layer documentation.

        This method creates comprehensive documentation about the project's
        persistence layer, including database schema, tables, relationships,
        and migration patterns.

        Args:
            file_manifest: Dictionary mapping file paths to file information
            persistence_info: PersistenceLayerInfo object with analyzed persistence data

        Returns:
            str: Generated persistence documentation

        Example:
            ```python
            persistence_doc = await client.generate_persistence_doc(
                file_manifest, persistence_info
            )
            ```
        """

    @abstractmethod
    async def enhance_documentation(
        self, existing_content: str, file_manifest: Dict[str, Any], doc_type: str
    ) -> str:
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

    @abstractmethod
    async def generate_structured_json_response(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a structured JSON response from the LLM with robust parsing.

        This method should be used whenever expecting a JSON response from the LLM.
        It includes enhanced prompting to ensure clean JSON output and robust
        extraction logic to handle various response formats.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum tokens for the response

        Returns:
            Clean JSON string that can be parsed with json.loads()

        Raises:
            ValueError: If JSON extraction fails after all fallback strategies

        Example:
            ```python
            messages = [{"role": "user", "content": "Return user data as JSON"}]
            json_str = await client.generate_structured_json_response(messages)
            data = json.loads(json_str)
            ```
        """

    @abstractmethod
    async def analyze_single_migration(
        self, migration_info: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze a single migration file to extract schema information.

        This method analyzes an individual migration file and extracts
        database schema details including tables, columns, relationships,
        indexes, views, stored procedures, and constraints.

        Args:
            migration_info: Dictionary containing migration file information:
                - file_path: Path to the migration file
                - file_name: Name of the migration file
                - content: Content of the migration file
                - type: Type of migration (sql, java, csharp, etc.)

        Returns:
            Dict[str, Any]: Extracted schema information containing:
                - tables: List of table definitions
                - indexes: List of index definitions
                - views: List of view definitions
                - relationships: List of relationship definitions
                - procedures: List of stored procedure definitions
                - triggers: List of trigger definitions

        Example:
            ```python
            migration = {
                "file_path": "/path/to/V1__Create_Users.sql",
                "file_name": "V1__Create_Users.sql",
                "content": "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));",
                "type": "sql"
            }
            schema = await client.analyze_single_migration(migration)
            # Returns: {"tables": [{"name": "users", "columns": [...]}], ...}
            ```
        """

    @abstractmethod
    async def aggregate_migration_analyses(
        self, individual_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate individual migration analyses into a comprehensive schema.

        This method takes multiple individual migration analyses and combines
        them into a single comprehensive database schema, detecting relationships
        and resolving conflicts between migrations.

        Args:
            individual_analyses: List of individual migration analysis results
                Each analysis contains schema information from a single migration file

        Returns:
            Dict[str, Any]: Comprehensive aggregated schema information containing:
                - tables: Consolidated list of all tables
                - indexes: Consolidated list of all indexes
                - views: Consolidated list of all views
                - relationships: Detected relationships between tables
                - procedures: Consolidated list of stored procedures
                - triggers: Consolidated list of triggers

        Example:
            ```python
            analyses = [
                {"tables": [{"name": "users", ...}], "indexes": [...]},
                {"tables": [{"name": "orders", ...}], "indexes": [...]}
            ]
            schema = await client.aggregate_migration_analyses(analyses)
            # Returns consolidated schema with detected relationships
            ```
        """
