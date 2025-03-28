# Standard library imports
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

# Third-party imports
import httpx
from ollama import AsyncClient
from pydantic import BaseModel

# Local application imports
from .base_llm import BaseLLMClient
from .message_manager import MessageManager
from src.utils.config_class import ScribeConfig
from .llm_utils import (
    format_project_structure,
    find_common_dependencies,
    identify_key_components,
    fix_markdown_issues,
    prepare_file_order_data,
    process_file_order_response
)
from ..analyzers.codebase import CodebaseAnalyzer
from ..utils.prompt_manager import PromptTemplate
from ..utils.progress import ProgressTracker
from ..utils.retry import async_retry
from ..utils.tokens import TokenCounter

# Constants for default configuration values
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_TIMEOUT = 30
DEFAULT_TEMPERATURE = 0

class OllamaClientError(Exception):
    """Custom exception for Ollama client errors."""
    pass

class OllamaClient(BaseLLMClient):
    """
    Client for interacting with local Ollama LLM instances.
    
    This class handles all interactions with a local Ollama instance, including
    model selection, token management, and generating various types of documentation.
    
    Attributes:
        base_url (str): URL of the Ollama API endpoint
        max_tokens (int): Maximum number of tokens for responses
        retries (int): Number of retries for API calls
        retry_delay (float): Delay between retries in seconds
        timeout (int): Timeout for API calls in seconds
        temperature (float): Temperature for LLM generation (0-1)
        client (AsyncClient): Ollama API client
        prompt_template (PromptTemplate): Template manager for prompts
        debug (bool): Whether to print debug information
        available_models (list): List of available models
        selected_model (str): Currently selected model
        token_counter (TokenCounter): Counter for token usage
        project_structure (str): String representation of project structure
    """
    
    def __init__(self, config: ScribeConfig):
        """
        Initialize the Ollama client.
        
        Args:
            config: Configuration with Ollama-specific settings
        """
        # Call parent class constructor
        super().__init__()
        
        # Get Ollama config from ScribeConfig
        self.base_url = config.ollama.base_url
        self.max_tokens = config.ollama.max_tokens
        self.retries = config.ollama.retries
        self.retry_delay = config.ollama.retry_delay
        self.timeout = config.ollama.timeout
        self.temperature = config.ollama.temperature
        self.debug = config.debug
        
        # Initialize Ollama client
        self.client = AsyncClient(host=self.base_url)
        # Initialize prompt template
        if hasattr(config, 'template_path') and config.template_path:
            self.prompt_template = PromptTemplate(config.template_path)
        else:
            self.prompt_template = PromptTemplate()
        
        # Initialize model-related attributes
        self.available_models = []
        self.selected_model = None
        self.selected_model = None
    
    async def initialize(self) -> None:
        """
        Initialize the client asynchronously.
        
        This method fetches available models from the Ollama instance,
        prompts the user to select a model, and initializes the token counter.
        
        Raises:
            OllamaClientError: If no models are available or initialization fails
        """
        try:
            self.available_models = await self._get_available_models()
            if not self.available_models:
                raise OllamaClientError("No models available in Ollama instance. Please ensure Ollama is running and has models installed.")
            
            self.selected_model = await self._select_model_interactive()
            
            # Initialize token counter after model selection
            self.init_token_counter()
            
            print(f"\nInitialized with model: {self.selected_model}")
            print("Starting analysis...\n")
            
            if self.debug:
                print(f"Selected model: {self.selected_model}")
            
        except httpx.HTTPError as e:
            if self.debug:
                print(f"HTTP error during initialization: {traceback.format_exc()}")
            raise OllamaClientError(f"Failed to connect to Ollama API: {str(e)}")
        except Exception as e:
            if self.debug:
                print(f"Initialization error: {traceback.format_exc()}")
            raise OllamaClientError(f"Failed to initialize client: {str(e)}")

    def init_token_counter(self) -> None:
        """
        Initialize the token counter for this client.
        
        This method sets up the TokenCounter instance with the
        selected model name and debug configuration.
        """
        self.token_counter = TokenCounter(model_name=self.selected_model, debug=self.debug)

    async def _get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of model names available in the Ollama instance
            
        Raises:
            No exceptions are raised, but returns an empty list on error
        """
        try:
            response = await self.client.list()
            
            # Handle both object-style response and dictionary-style response
            if hasattr(response, 'models'):
                # Object-style response (actual API)
                models = response.models
                return [model.model for model in models]
            elif isinstance(response, dict) and 'models' in response:
                # Dictionary-style response (for testing)
                models = response['models']
                return [model['name'] for model in models]
            else:
                if self.debug:
                    print(f"Unexpected response format: {response}")
                return []
                
        except httpx.HTTPError as e:
            if self.debug:
                print(f"HTTP error fetching models: {str(e)}")
            logging.error(f"HTTP error fetching models from Ollama: {str(e)}")
            return []
        except Exception as e:
            if self.debug:
                print(f"Error fetching models: {str(e)}")
            logging.error(f"Error fetching models from Ollama: {str(e)}")
            return []

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_summary(self, content: str, file_type: str = "text", file_path: str = None) -> Optional[str]:
        """
        Generate a summary for a file's content.
        
        This method processes the content of a file and produces a concise
        summary describing its purpose and functionality. It handles token
        limit checking and truncation if necessary.
        
        Args:
            content: The content of the file to summarize
            file_type: The type/language of the file (default: "text")
            file_path: The path to the file (default: None)
            
        Returns:
            Optional[str]: Generated summary or None if generation fails
        """
        try:
            # Create a prompt that includes file information
            file_info = f"File: {file_path}\nType: {file_type}\n\n" if file_path else ""
            prompt = f"{file_info}{content}"
            
            # Check token count
            will_exceed, token_count = self.token_counter.will_exceed_limit(prompt, self.selected_model)
            
            if will_exceed:
                if self.debug:
                    print(f"Content exceeds token limit ({token_count} tokens). Truncating...")
                prompt = self.token_counter.truncate_text(prompt)
                
                # Re-check after truncation
                _, new_token_count = self.token_counter.will_exceed_limit(prompt, self.selected_model)
                if self.debug:
                    print(f"Truncated to {new_token_count} tokens")
            
            response = await self.client.chat(
                model=self.selected_model,
                messages=MessageManager.get_file_summary_messages(prompt),
                options={"temperature": self.temperature}
            )
            
            if response and 'message' in response:
                content = response['message'].get('content', '')
                return self._fix_markdown_issues(content)
            
            logging.warning("Empty or invalid response from Ollama API")
            return None
            
        except httpx.HTTPError as e:
            logging.error(f"HTTP error generating summary: {e}")
            return None
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return None

    # _update_progress method removed - now using ProgressTracker.update_progress_async

    def _fix_markdown_issues(self, content: str) -> str:
        """
        Fix common markdown formatting issues before returning content.
        
        This method delegates to the imported fix_markdown_issues utility function
        to ensure consistent markdown formatting across all generated content.
        
        Args:
            content: The markdown content to fix
            
        Returns:
            The fixed markdown content with proper formatting
        """
        return fix_markdown_issues(content)

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_project_overview(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate project overview based strictly on observed evidence.
        
        This method analyzes the project structure and files to create a
        comprehensive overview of the project's purpose and components.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated project overview in markdown format
            
        Raises:
            Exception: If an error occurs during generation
        """
        # Get progress tracker instance
        progress_tracker = ProgressTracker.get_instance(Path("."))
        with progress_tracker.progress_bar(
            desc="Generating project overview",
            bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
            ncols=150
        ) as pbar:
            try:
                update_task = asyncio.create_task(progress_tracker.update_progress_async(pbar))
                
                # Get detected technologies
                tech_report = self._find_common_dependencies(file_manifest)
                
                # Get key components with the improved method
                key_components = self._identify_key_components(file_manifest)
                
                # Create template content with project information
                template_content = self.prompt_template.get_template('project_overview', {
                    'project_name': self._derive_project_name(file_manifest),
                    'file_count': len(file_manifest),
                    'key_components': key_components,
                    'dependencies': tech_report,
                    'project_structure': self.project_structure
                })
                
                response = await self.client.chat(
                    model=self.selected_model,
                    messages=MessageManager.get_project_overview_messages(
                        self.project_structure, 
                        tech_report, 
                        template_content
                    ),
                    options={"temperature": self.temperature}
                )
                
                update_task.cancel()
                content = response['message']['content']
                
                # Fix any remaining markdown issues
                fixed_content = self._fix_markdown_issues(content)
                return fixed_content
                
            except Exception as e:
                if self.debug:
                    print(f"\nError generating overview: {str(e)}")
                raise

    def _format_project_structure(self, file_manifest: Dict[str, Any]) -> str:
        """
        Build a tree-like project structure string from file manifest.
        
        This method delegates to the imported format_project_structure utility function
        to create a formatted string representation of the project structure.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            A formatted string representing the project structure
        """
        return format_project_structure(file_manifest, self.debug)

    def set_project_structure(self, structure: str) -> None:
        """
        Set the project structure for use in prompts.
        
        Args:
            structure: String representation of the project structure
        """
        self.project_structure = structure
        
    def set_project_structure_from_manifest(self, file_manifest: Dict[str, Any]) -> None:
        """
        Set the project structure from a file manifest.
        
        This is a convenience method that formats the file manifest into a
        string representation and then sets it as the project structure.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
        """
        self.project_structure = self._format_project_structure(file_manifest)

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_component_relationships(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate description of how components interact.
        
        This method analyzes the relationships between different components
        in the project and describes their interactions based on the file manifest.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated component relationship description in markdown format
        """
        
        # Get detected technologies first
        tech_report = self._find_common_dependencies(file_manifest)
        
        response = await self.client.chat(
            model=self.selected_model,
            messages=MessageManager.get_component_relationship_messages(
                self.project_structure, 
                tech_report
            ),
            options={"temperature": self.temperature}
        )
        content = response['message']['content']
        
        # Fix any markdown issues
        return self._fix_markdown_issues(content)

    def _identify_key_components(self, file_manifest: Dict[str, Any]) -> str:
        """
        Identify key components from file manifest.
        
        This method analyzes the file manifest to identify the main components
        of the project based on directory structure and file patterns.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: A formatted string listing key components
        """
        return identify_key_components(file_manifest, self.debug)

    def _find_common_dependencies(self, file_manifest: Dict[str, Any]) -> str:
        """
        Extract common dependencies from file manifest.
        
        This method analyzes the file manifest to identify common dependencies
        used in the project, such as libraries, frameworks, and packages.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: A formatted string listing detected dependencies
        """
        return find_common_dependencies(file_manifest, self.debug)

    async def _select_model_interactive(self) -> str:
        """
        Interactive model selection.
        
        This method prompts the user to select a model from the list of
        available models in the Ollama instance.
        
        Returns:
            str: The name of the selected model
            
        Raises:
            No exceptions are raised, but will retry until a valid selection is made
        """
        while True:
            print("Available Ollama models:")
            for i, model in enumerate(self.available_models, 1):
                print(f"{i}. {model}")
            
            try:
                selection = int(input("Enter the number of the model to use: "))
                if 1 <= selection <= len(self.available_models):
                    return self.available_models[selection - 1]
                else:
                    print(f"\nError: '{selection}' is not a valid option.")
                    print(f"Please choose a number between 1 and {len(self.available_models)}")
                    print("\n" + "-" * 50 + "\n")
            except ValueError:
                print("\nError: Please enter a valid number, not text.")
                print("\n" + "-" * 50 + "\n")            

    class FileOrderResponse(BaseModel):
        """Schema for file ordering response"""
        file_order: List[str]
        reasoning: Optional[str] = None

    async def get_file_order(self, project_files: Dict[str, Any]) -> List[str]:
        """
        Ask LLM to determine optimal file processing order.
        
        This method analyzes dependencies between files to determine the
        most efficient order for processing them, using the LLM to identify
        relationships and dependencies.
        
        Args:
            project_files: Dictionary mapping file paths to file information
            
        Returns:
            List[str]: List of file paths in optimal processing order
        """
        try:
            print("\nStarting file order optimization...")
            logging.info("Preparing file order optimization request")
            
            # Use common utility to prepare data
            core_files, resource_files, files_info = prepare_file_order_data(project_files, self.debug)
            
            print(f"Sending request to LLM with {len(files_info)} files...")
            logging.info(f"Sending file order request to LLM with {len(files_info)} files")
            
            # Get messages from MessageManager
            messages = MessageManager.get_file_order_messages(files_info)
            
            # Send request to Ollama
            response = await self.client.chat(
                model=self.selected_model,
                messages=messages,
                options={"temperature": self.temperature}
            )
            
            content = response['message']['content']
            
            # Use common utility to process response
            return process_file_order_response(content, core_files, resource_files, self.debug)
            
        except Exception as e:
            print(f"Error in file order optimization: {str(e)}")
            logging.error(f"Error getting file order: {str(e)}", exc_info=True)
            return list(project_files.keys())

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_architecture_content(self, file_manifest: Dict[str, Any], analyzer: Any) -> str:
        """
        Generate architecture documentation content with flow diagrams.
        
        This method creates comprehensive documentation about the project's
        architecture, including component diagrams and design patterns.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            analyzer: CodebaseAnalyzer instance for additional analysis
            
        Returns:
            str: Generated architecture documentation in markdown format
        """
        # Get progress tracker instance
        progress_tracker = ProgressTracker.get_instance(Path("."))
        with progress_tracker.progress_bar(
            desc="Generating architecture documentation",
            bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
            ncols=150
        ) as pbar:
            try:
                update_task = asyncio.create_task(progress_tracker.update_progress_async(pbar))
                
                # Ensure project structure is set
                if not self.project_structure or len(self.project_structure) < 10:
                    self.set_project_structure_from_manifest(file_manifest)
                    if self.debug:
                        print(f"Project structure generated ({len(self.project_structure)} chars)")
                
                # Get detected technologies
                tech_report = self._find_common_dependencies(file_manifest)
                
                # Get key components
                key_components = self._identify_key_components(file_manifest)
                
                # Create a summary of file contents for context
                file_summaries = []
                
                # First, categorize files by directory/component
                file_by_component = {}
                for path, info in file_manifest.items():
                    if info.get('summary') and not info.get('is_binary', False):
                        directory = str(Path(path).parent)
                        if directory not in file_by_component:
                            file_by_component[directory] = []
                        file_by_component[directory].append((path, info.get('summary', 'No summary available')))
                
                # For each component, include a representative sample of files
                for directory, files in file_by_component.items():
                    # Add component header
                    file_summaries.append(f"## Component: {directory}")
                    
                    # Sort files by potential importance (e.g., longer summaries might be more important)
                    files.sort(key=lambda x: len(x[1]), reverse=True)
                    
                    # Take up to 3 files per component to ensure broad coverage
                    for path, summary in files[:3]:
                        file_summaries.append(f"File: {path}\nSummary: {summary}")
                
                file_summaries_text = "\n\n".join(file_summaries)
                
                # Get messages from MessageManager with enhanced content
                messages = MessageManager.get_architecture_content_messages(
                    self.project_structure, 
                    key_components,
                    tech_report
                )
                
                # Add file summaries to the user message
                for i, msg in enumerate(messages):
                    if msg["role"] == "user":
                        messages[i]["content"] += f"\n\nFile Summaries:\n{file_summaries_text}"
                        break
                
                response = await self.client.chat(
                    model=self.selected_model,
                    messages=messages,
                    options={"temperature": self.temperature}
                )
                
                update_task.cancel()
                content = response['message']['content']
                
                # Ensure the project structure is included in the output
                if "```" not in content[:500]:
                    content = f"# Architecture Documentation\n\n## Project Structure\n```\n{self.project_structure}\n```\n\n{content}"
                
                # Fix any markdown issues
                return self._fix_markdown_issues(content)
                
            except Exception as e:
                if self.debug:
                    print(f"\nError generating architecture content: {str(e)}")
                return "Error generating architecture documentation."

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_usage_guide(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate usage guide based on project structure.
        
        This method creates documentation explaining how to use the project,
        including installation, configuration, and common operations.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated usage guide in markdown format
        """
        try:
            response = await self.client.chat(
                model=self.selected_model,
                messages=MessageManager.get_usage_guide_messages(
                    self.project_structure,
                    self._find_common_dependencies(file_manifest)
                ),
                options={"temperature": self.temperature}
            )
            
            content = response['message']['content']
            
            # Fix any markdown issues
            return self._fix_markdown_issues(content)
            
        except Exception as e:
            if self.debug:
                print(f"\nError generating usage guide: {str(e)}")
            return "### Usage\n\nUsage instructions could not be generated."

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_contributing_guide(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate contributing guide based on project structure.
        
        This method creates documentation explaining how to contribute to the project,
        including coding standards, pull request process, and development setup.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated contributing guide in markdown format
        """
        try:
            response = await self.client.chat(
                model=self.selected_model,
                messages=MessageManager.get_contributing_guide_messages(
                    self.project_structure
                ),
                options={"temperature": self.temperature}
            )
            
            content = response['message']['content']
            
            # Fix any markdown issues
            return self._fix_markdown_issues(content)
            
        except Exception as e:
            if self.debug:
                print(f"\nError generating contributing guide: {str(e)}")
            return "### Contributing\n\nContribution guidelines could not be generated."

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_license_info(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate license information based on project structure.
        
        This method analyzes the project to determine its license and creates
        appropriate license information documentation.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            str: Generated license information in markdown format
        """
        try:
            response = await self.client.chat(
                model=self.selected_model,
                messages=MessageManager.get_license_info_messages(
                    self.project_structure
                ),
                options={"temperature": self.temperature}
            )
            
            content = response['message']['content']
            
            # Fix any markdown issues
            return self._fix_markdown_issues(content)
            
        except Exception as e:
            if self.debug:
                print(f"\nError generating license info: {str(e)}")
            return "This project's license information could not be determined."

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def enhance_documentation(self, existing_content: str, file_manifest: Dict[str, Any], doc_type: str) -> str:
        """
        Enhance existing documentation with new insights.
        
        This method takes existing documentation and improves it based on
        analysis of the codebase and file manifest.
        
        Args:
            existing_content: The existing documentation content
            file_manifest: Dictionary mapping file paths to file information
            doc_type: Type of documentation being enhanced (e.g., "README", "ARCHITECTURE")
            
        Returns:
            str: Enhanced documentation content in markdown format
        """
        try:
            # Get detected technologies
            tech_report = self._find_common_dependencies(file_manifest)
            
            # Get key components
            key_components = self._identify_key_components(file_manifest)
            
            response = await self.client.chat(
                model=self.selected_model,
                messages=MessageManager.get_enhance_documentation_messages(
                    existing_content,
                    self.project_structure,
                    key_components,
                    tech_report,
                    doc_type
                ),
                options={"temperature": self.temperature}
            )
            
            content = response['message']['content']
            
            # Fix any markdown issues
            return self._fix_markdown_issues(content)
            
        except Exception as e:
            if self.debug:
                print(f"\nError enhancing documentation: {str(e)}")
            return existing_content  # Return original content on error 

    def _derive_project_name(self, file_manifest: Dict[str, Any]) -> str:
        """
        Derive project name from repository structure.
        
        This method attempts to determine a project name based on the repository
        structure, looking at common files like package.json, setup.py, etc.
        
        Args:
            file_manifest: Dictionary mapping file paths to file information
            
        Returns:
            A string representing the derived project name
        """
        # Use the existing CodebaseAnalyzer import to avoid circular imports
        temp_analyzer = CodebaseAnalyzer(Path("."), {"debug": self.debug})
        temp_analyzer.file_manifest = file_manifest
        return temp_analyzer.derive_project_name(self.debug)