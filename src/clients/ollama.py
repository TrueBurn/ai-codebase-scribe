# Standard library imports
import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

# Third-party imports
import httpx
from ollama import AsyncClient
from pydantic import BaseModel

# Try to import visual logger, fallback to regular logging
try:
    from src.utils.visual_logger import get_visual_logger

    _use_visual_logging = True
except ImportError:
    _use_visual_logging = False


def _get_logger():
    """Get appropriate logger (visual or standard)."""
    if _use_visual_logging:
        return get_visual_logger()
    return logging.getLogger(__name__)


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
                fixed_content = self._fix_markdown_issues(content)

                # VALIDATION: Check if summary is valid
                is_valid, reason = self._validate_file_summary(fixed_content, file_path)

                if not is_valid:
                    logging.warning(
                        f"Invalid summary for {file_path}: {reason}. Retrying once..."
                    )

                    # Retry with more explicit prompt
                    retry_prompt = f"{prompt}\n\nIMPORTANT: Provide detailed code analysis following the template structure. Do NOT provide a confirmation message."
                    retry_response = await self.client.chat(
                        model=self.selected_model,
                        messages=MessageManager.get_file_summary_messages(retry_prompt),
                        options={"temperature": self.temperature},
                    )

                    if retry_response and "message" in retry_response:
                        retry_content = retry_response["message"].get("content", "")
                        retry_fixed = self._fix_markdown_issues(retry_content)

                        # Validate retry
                        is_valid, reason = self._validate_file_summary(
                            retry_fixed, file_path
                        )
                        if not is_valid:
                            logging.error(
                                f"Retry failed for {file_path}: {reason}. Returning None."
                            )
                            return None
                        return retry_fixed
                    else:
                        logging.error(
                            f"Retry failed for {file_path}: Empty response. Returning None."
                        )
                        return None

                return fixed_content

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

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_installation_guide(self, file_manifest: Dict[str, Any]) -> str:
        """
        Generate installation guide based on project structure.

        This method creates documentation explaining how to install and set up the project,
        including prerequisites, installation steps, and verification procedures.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated installation guide in markdown format
        """
        try:
            # Build tech report from manifest
            tech_report = self._find_common_dependencies(file_manifest)

            # Build installation guide messages inline since MessageManager
            # may not have this method yet
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a technical writer creating installation documentation. "
                        "Generate a comprehensive installation guide based on the project structure and technologies used. "
                        "Include prerequisites, step-by-step installation instructions, configuration steps, "
                        "and verification procedures. Use markdown formatting."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Project Structure:\n{self.project_structure}\n\n"
                        f"Technologies:\n{tech_report}\n\n"
                        "Generate a detailed installation guide for this project."
                    ),
                },
            ]

            # Use MessageManager method if available, otherwise use inline messages
            if hasattr(MessageManager, 'get_installation_guide_messages'):
                messages = MessageManager.get_installation_guide_messages(
                    self.project_structure, tech_report, self.related_repo_context
                )

            response = await self.client.chat(
                model=self.selected_model,
                messages=messages,
                options={"temperature": self.temperature},
            )

            content = response["message"]["content"]

            # Fix any markdown issues
            return self._fix_markdown_issues(content)

        except Exception as e:
            if self.debug:
                print(f"\nError generating installation guide: {str(e)}")
            return (
                "### Installation\n\nInstallation instructions could not be generated."
            )

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
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
            str: Generated troubleshooting guide in markdown format
        """
        try:
            # Build tech report from manifest
            tech_report = self._find_common_dependencies(file_manifest)

            # Build troubleshooting guide messages inline since MessageManager
            # may not have this method yet
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a technical writer creating troubleshooting documentation. "
                        "Generate a comprehensive troubleshooting guide based on the project structure and technologies used. "
                        "Include common issues, debugging procedures, error message explanations, "
                        "and solutions. Use markdown formatting."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Project Structure:\n{self.project_structure}\n\n"
                        f"Technologies:\n{tech_report}\n\n"
                        "Generate a detailed troubleshooting guide for this project."
                    ),
                },
            ]

            # Use MessageManager method if available, otherwise use inline messages
            if hasattr(MessageManager, 'get_troubleshooting_guide_messages'):
                messages = MessageManager.get_troubleshooting_guide_messages(
                    self.project_structure, tech_report, self.related_repo_context
                )

            response = await self.client.chat(
                model=self.selected_model,
                messages=messages,
                options={"temperature": self.temperature},
            )

            content = response["message"]["content"]

            # Fix any markdown issues
            return self._fix_markdown_issues(content)

        except Exception as e:
            if self.debug:
                print(f"\nError generating troubleshooting guide: {str(e)}")
            return (
                "### Troubleshooting\n\nTroubleshooting guide could not be generated."
            )

    @async_retry(
        retries=DEFAULT_RETRIES,
        delay=DEFAULT_RETRY_DELAY,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(httpx.HTTPError, ConnectionError, TimeoutError),
    )
    async def generate_persistence_doc(
        self, file_manifest: Dict[str, Any], persistence_info: Any
    ) -> str:
        """
        Generate persistence layer documentation using multi-part strategy.

        Splits documentation into manageable parts to avoid token limits:
        1. Overview + schema statistics
        2. Detailed table documentation (batched)
        3. Relationships + ER diagram

        Note: Ollama doesn't support prompt caching, but batching still helps
        with token limits and generation quality.

        Args:
            file_manifest: Dictionary mapping file paths to file information
            persistence_info: PersistenceLayerInfo object with analyzed persistence data

        Returns:
            str: Generated persistence documentation in markdown format
        """
        try:
            # Extract schema data
            schema_data = getattr(persistence_info, "schema_data", {})
            tables = schema_data.get("tables", [])
            relationships = schema_data.get("relationships", [])

            # If no tables, return early
            if not tables:
                return "# Persistence Layer Documentation\n\nNo database schema information available."

            # === PART 1: Overview ===
            overview_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a database documentation expert. Generate an overview section "
                        "for the persistence layer documentation based on the project structure "
                        "and schema statistics provided."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Project Structure:\n{self.project_structure}\n\n"
                        f"Number of tables: {len(tables)}\n"
                        f"Number of relationships: {len(relationships)}\n"
                        f"Table names: {', '.join(t.get('name', 'unknown') for t in tables[:20])}\n\n"
                        "Generate a persistence layer overview section."
                    ),
                },
            ]

            # Use MessageManager method if available
            if hasattr(MessageManager, 'get_persistence_overview_messages'):
                overview_messages = MessageManager.get_persistence_overview_messages(
                    self.project_structure, persistence_info
                )

            overview_response = await self.client.chat(
                model=self.selected_model,
                messages=overview_messages,
                options={"temperature": self.temperature},
            )
            overview_content = overview_response["message"]["content"]

            # === PART 2: Tables (Batched) ===
            batch_size = 12
            tables_sorted = sorted(
                tables, key=lambda t: len(t.get("foreign_keys", [])), reverse=True
            )
            table_batches = [
                tables_sorted[i : i + batch_size]
                for i in range(0, len(tables_sorted), batch_size)
            ]

            tables_content = ""
            for batch_num, table_batch in enumerate(table_batches, 1):
                batch_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a database documentation expert. Document the following "
                            "database tables with their columns, constraints, and relationships."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Project Structure:\n{self.project_structure}\n\n"
                            f"Tables batch {batch_num} of {len(table_batches)}:\n"
                            f"{json.dumps(table_batch, indent=2)}\n\n"
                            "Document these tables in detail."
                        ),
                    },
                ]

                # Use MessageManager method if available
                if hasattr(MessageManager, 'get_persistence_tables_batch_messages'):
                    batch_messages = MessageManager.get_persistence_tables_batch_messages(
                        self.project_structure,
                        table_batch,
                        batch_num,
                        len(table_batches),
                    )

                batch_response = await self.client.chat(
                    model=self.selected_model,
                    messages=batch_messages,
                    options={"temperature": self.temperature},
                )
                tables_content += batch_response["message"]["content"] + "\n\n"

            # === PART 3: Relationships ===
            relationships_content = ""
            if relationships:
                tables_summary = ", ".join(
                    [t.get("name", "unknown") for t in tables[:20]]
                )
                if len(tables) > 20:
                    tables_summary += f" (and {len(tables) - 20} more)"

                rel_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a database documentation expert. Document the relationships "
                            "between database tables and generate an ER diagram in Mermaid format."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Project Structure:\n{self.project_structure}\n\n"
                            f"Tables: {tables_summary}\n\n"
                            f"Relationships:\n{json.dumps(relationships, indent=2)}\n\n"
                            "Document the relationships and generate a Mermaid ER diagram."
                        ),
                    },
                ]

                # Use MessageManager method if available
                if hasattr(MessageManager, 'get_persistence_relationships_messages'):
                    rel_messages = MessageManager.get_persistence_relationships_messages(
                        self.project_structure, relationships, tables_summary
                    )

                rel_response = await self.client.chat(
                    model=self.selected_model,
                    messages=rel_messages,
                    options={"temperature": self.temperature},
                )
                relationships_content = rel_response["message"]["content"]

            # === Combine all parts ===
            final_content = f"""# Persistence Layer Documentation

{overview_content}

## Tables

{tables_content}"""

            if relationships_content:
                final_content += f"\n\n{relationships_content}"

            # Fix any markdown issues
            return self._fix_markdown_issues(final_content)

        except Exception as e:
            if self.debug:
                print(f"\nError generating persistence documentation: {str(e)}")
            return "Persistence layer documentation could not be generated."

    async def generate_structured_json_response(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a structured JSON response from the LLM with robust parsing.

        This method implements the base class interface for structured JSON responses.
        It enhances the prompt to ensure raw JSON output and applies multiple
        fallback strategies for JSON extraction.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Optional maximum tokens for the response (not used by Ollama
                       but kept for interface compatibility)

        Returns:
            str: Valid JSON string extracted from the LLM response

        Raises:
            ValueError: If no valid JSON can be extracted from the response
        """
        # Extract system and user content from messages
        system_content = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), ""
        )
        user_content = next(
            (msg["content"] for msg in messages if msg["role"] == "user"), ""
        )

        # Combine messages into a single prompt
        prompt = f"{system_content}\n\n{user_content}"

        # Enhance the prompt to ensure raw JSON response
        enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS FOR JSON RESPONSE:
1. Return ONLY valid JSON - no explanatory text, no markdown formatting
2. Do NOT wrap the JSON in code blocks (```)
3. Do NOT include any text before or after the JSON
4. Start your response immediately with {{ and end with }}
5. Ensure the JSON is properly formatted and valid
6. Use double quotes for all strings
7. Do not include any comments in the JSON

Your response must be parseable by json.loads() function directly."""

        try:
            # Use Ollama to generate JSON response
            response = await self.client.chat(
                model=self.selected_model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                options={"temperature": 0.1},  # Lower temperature for consistent JSON
            )

            # Extract content from response
            if response and "message" in response:
                raw_content = response["message"].get("content", "")
            else:
                raise ValueError("Invalid response format from Ollama")

            if not raw_content:
                raise ValueError("Empty response from Ollama")

            # Extract JSON using robust parsing
            return self._extract_json_from_response(raw_content)

        except Exception as e:
            logging.error(f"Error generating JSON response from Ollama: {e}")
            raise

    def _extract_json_from_response(self, response_content: str) -> str:
        """
        Extract valid JSON from response content with multiple fallback strategies.

        Strategies applied in order:
        1. Direct parsing (response is already valid JSON)
        2. Remove markdown code blocks
        3. Brace-counting extraction
        4. Cleanup and trim
        5. JSON repair for common syntax errors

        Args:
            response_content: Raw response text from the LLM

        Returns:
            str: Valid JSON string

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        import re

        if not response_content:
            raise ValueError("Empty response content")

        # Log the raw response for debugging
        logging.debug(f"Raw Ollama response length: {len(response_content)}")
        logging.debug(
            f"Raw Ollama response starts with: {repr(response_content[:100])}"
        )

        # Strategy 1: Try parsing as-is (for properly formatted responses)
        try:
            json.loads(response_content.strip())
            logging.debug("Ollama response is already valid JSON")
            return response_content.strip()
        except json.JSONDecodeError:
            logging.debug(
                "Ollama response is not valid JSON as-is, trying extraction strategies"
            )

        # Strategy 2: Remove markdown code blocks
        if "```json" in response_content or "```" in response_content:
            logging.debug(
                "Found markdown code blocks in Ollama response, removing them"
            )
            patterns = [
                r"```json\s*(.*?)\s*```",
                r"```\s*(.*?)\s*```",
                r"```json\s*(.*)",
                r"```\s*(.*)",
            ]

            for pattern in patterns:
                match = re.search(pattern, response_content, re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    try:
                        json.loads(extracted)
                        logging.debug(
                            f"Successfully extracted JSON from Ollama using pattern: {pattern}"
                        )
                        return extracted
                    except json.JSONDecodeError:
                        continue

        # Strategy 3: Find JSON object boundaries using brace counting
        start_idx = response_content.find("{")
        if start_idx != -1:
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(response_content)):
                char = response_content[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            extracted = response_content[start_idx : i + 1]
                            try:
                                json.loads(extracted)
                                logging.debug(
                                    "Successfully extracted JSON from Ollama using brace counting"
                                )
                                return extracted
                            except json.JSONDecodeError:
                                logging.debug(
                                    "Extracted content from Ollama is not valid JSON, continuing search"
                                )
                                break

        # Strategy 4: Clean up common issues
        cleaned = response_content.strip()

        start_brace = cleaned.find("{")
        if start_brace > 0:
            cleaned = cleaned[start_brace:]

        end_brace = cleaned.rfind("}")
        if end_brace != -1 and end_brace < len(cleaned) - 1:
            cleaned = cleaned[: end_brace + 1]

        try:
            json.loads(cleaned)
            logging.debug("Successfully cleaned and extracted JSON from Ollama")
            return cleaned
        except json.JSONDecodeError as e:
            logging.debug(f"Cleaned JSON from Ollama still has errors: {e}")

            # Strategy 5: JSON repair for large responses with syntax errors
            logging.debug("Attempting JSON repair for malformed Ollama response")
            try:
                repaired_json = self._repair_malformed_json(cleaned, e)
                if repaired_json:
                    json.loads(repaired_json)  # Validate the repair
                    logging.info("Successfully repaired malformed JSON from Ollama")
                    return repaired_json
            except (json.JSONDecodeError, Exception) as repair_error:
                logging.debug(f"Ollama JSON repair failed: {repair_error}")

            logging.warning(
                f"All JSON extraction strategies failed for Ollama. Last error: {e}"
            )
            logging.warning(
                f"Ollama response content preview: {response_content[:500]}"
            )
            raise ValueError(f"Could not extract valid JSON from Ollama response: {e}")

    def _repair_malformed_json(
        self, json_content: str, error: json.JSONDecodeError
    ) -> Optional[str]:
        """
        Attempt to repair common JSON syntax errors in large responses.

        Handles:
        - Missing comma delimiters
        - Trailing commas before closing brackets/braces
        - Unclosed objects/arrays near the end of large responses
        - Truncated responses from large LLM outputs

        Args:
            json_content: The malformed JSON string
            error: The JSONDecodeError that was raised

        Returns:
            Optional[str]: Repaired JSON string, or None if repair fails
        """
        import re

        try:
            error_pos = getattr(error, "pos", 0)
            error_msg = str(error)

            logging.debug(
                f"Attempting to repair Ollama JSON error: {error_msg} at position {error_pos}"
            )

            # Strategy 1: Fix missing comma errors
            if "Expecting ',' delimiter" in error_msg:
                repaired = json_content[:error_pos] + "," + json_content[error_pos:]
                logging.debug("Applied comma fix to Ollama response")
                return repaired

            # Strategy 2: Fix trailing comma errors
            elif (
                "Expecting ':' delimiter" in error_msg or "Expecting value" in error_msg
            ):
                repaired = re.sub(r",(\s*[}\]])", r"\1", json_content)
                if repaired != json_content:
                    logging.debug("Applied trailing comma fix to Ollama response")
                    return repaired

            # Strategy 3: Fix unclosed objects/arrays in very large responses
            elif "Expecting" in error_msg and error_pos > len(json_content) * 0.8:
                open_braces = json_content.count("{") - json_content.count("}")
                open_brackets = json_content.count("[") - json_content.count("]")

                if open_braces > 0 or open_brackets > 0:
                    repair_suffix = "}" * open_braces + "]" * open_brackets
                    repaired = json_content + repair_suffix
                    logging.debug(
                        f"Applied structure closing fix to Ollama response: added {repair_suffix}"
                    )
                    return repaired

            # Strategy 4: Truncate at last valid object for very large responses
            elif error_pos > 50000:
                truncate_pos = json_content.rfind("}", 0, error_pos)
                if truncate_pos > 0:
                    before_truncate = json_content[: truncate_pos + 1]
                    open_braces = before_truncate.count("{") - before_truncate.count(
                        "}"
                    )
                    open_brackets = before_truncate.count("[") - before_truncate.count(
                        "]"
                    )

                    if open_braces == 0:
                        repair_suffix = "]" * open_brackets
                        repaired = before_truncate + repair_suffix
                        logging.debug(
                            f"Applied truncation fix to Ollama response at position {truncate_pos}"
                        )
                        return repaired

            return None

        except Exception as e:
            logging.debug(f"Exception during Ollama JSON repair: {e}")
            return None

    async def analyze_migration_contents(
        self, migration_contents: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze migration file contents to extract detailed schema information using LLM intelligence.

        Args:
            migration_contents: List of migration file info dictionaries with keys:
                - file_name: Name of the migration file
                - file_path: Path to the migration file
                - type: Type of migration (sql, java, csharp, etc.)
                - content: The migration file content

        Returns:
            Dict containing extracted schema information with keys:
                tables, relationships, indexes, views, procedures, triggers
        """
        if not migration_contents:
            return self._empty_schema_structure()

        # Create prompt for migration analysis
        migration_files_info = []
        for migration in migration_contents:
            migration_files_info.append(
                f"""
## File: {migration['file_name']}
**Path:** {migration['file_path']}
**Type:** {migration['type']}

**Content:**
```{migration['type']}
{migration['content']}
```
"""
            )

        migrations_text = "\n".join(migration_files_info)

        prompt = f"""You are a database schema analyst. Analyze the following migration files and extract detailed schema information.

{migrations_text}

Please analyze these migration files and provide a comprehensive JSON response with the following structure:

{{
    "tables": [
        {{
            "name": "table_name",
            "columns": [
                {{
                    "name": "column_name",
                    "type": "data_type",
                    "nullable": true,
                    "primary_key": false,
                    "unique": false,
                    "default_value": null,
                    "constraints": []
                }}
            ],
            "primary_keys": ["column1"],
            "foreign_keys": [
                {{
                    "column": "local_column",
                    "references_table": "referenced_table",
                    "references_column": "referenced_column",
                    "on_delete": "CASCADE",
                    "on_update": "CASCADE"
                }}
            ],
            "unique_constraints": [],
            "check_constraints": [],
            "migration_file": "migration_file_name"
        }}
    ],
    "indexes": [
        {{
            "name": "index_name",
            "table": "table_name",
            "columns": ["column1"],
            "unique": false,
            "type": "BTREE",
            "migration_file": "migration_file_name"
        }}
    ],
    "views": [
        {{
            "name": "view_name",
            "definition": "SQL definition",
            "depends_on_tables": ["table1"],
            "migration_file": "migration_file_name"
        }}
    ],
    "procedures": [
        {{
            "name": "procedure_name",
            "type": "PROCEDURE",
            "parameters": [],
            "definition": "SQL definition",
            "migration_file": "migration_file_name"
        }}
    ],
    "triggers": [
        {{
            "name": "trigger_name",
            "table": "table_name",
            "event": "INSERT",
            "timing": "BEFORE",
            "definition": "SQL definition",
            "migration_file": "migration_file_name"
        }}
    ],
    "relationships": [
        {{
            "from_table": "table1",
            "from_column": "column1",
            "to_table": "table2",
            "to_column": "column2",
            "relationship_type": "one-to-many",
            "constraint_name": "fk_constraint_name"
        }}
    ]
}}

IMPORTANT INSTRUCTIONS:
- Analyze ALL migration files provided
- Extract complete table schemas with all columns, data types, and constraints
- Identify all relationships (foreign keys) between tables
- Find all indexes, views, stored procedures, functions, and triggers
- For SQL files, parse all DDL statements
- IMPORTANT: Keep the response concise - limit to 5 most important tables and their key columns
- Return valid JSON only - no additional text or explanations"""

        try:
            # Convert prompt to messages format for the JSON method
            messages = [{"role": "user", "content": prompt}]

            # Use the dedicated JSON method for structured responses
            response = await self.generate_structured_json_response(
                messages, max_tokens=16384
            )

            if not response:
                logging.warning("Empty response from Ollama for migration analysis")
                return self._empty_schema_structure()

            try:
                schema_data = json.loads(response)
                logging.info(
                    f"Successfully analyzed {len(migration_contents)} migration files with Ollama"
                )
                return schema_data
            except json.JSONDecodeError as e:
                logging.error(
                    f"Unexpected JSON parsing error after extraction from Ollama: {e}"
                )
                logging.error(f"Response content: {response[:500]}")
                return self._empty_schema_structure()

        except Exception as e:
            logging.error(f"Error during Ollama migration content analysis: {e}")
            return self._empty_schema_structure()

    async def analyze_single_migration(
        self, migration_info: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze a single migration file to extract schema information.

        Args:
            migration_info: Dictionary with keys:
                - file_name: Name of the migration file
                - type: Type of migration (sql, java, csharp, etc.)
                - content: The migration file content

        Returns:
            Dict containing extracted schema information for this single migration
        """
        try:
            file_name = migration_info.get("file_name", "unknown")
            migration_type = migration_info.get("type", "unknown")
            content = migration_info.get("content", "")

            logging.debug(
                f"Analyzing individual migration with Ollama: {file_name} ({len(content)} chars)"
            )

            if not content.strip():
                logging.warning(f"Empty migration content for {file_name}")
                return self._empty_schema_structure()

            # Build the analysis prompt for a single migration
            prompt = f"""Analyze this single database migration file and extract comprehensive schema information.

Migration File: {file_name}
Migration Type: {migration_type}

Content:
{content}

You must analyze the migration file and return a JSON response with this exact structure:
{{
    "tables": [
        {{
            "name": "table_name",
            "columns": [
                {{
                    "name": "column_name",
                    "type": "data_type",
                    "nullable": true,
                    "primary_key": false,
                    "foreign_key": null,
                    "default_value": null,
                    "constraints": []
                }}
            ],
            "primary_keys": [],
            "foreign_keys": [
                {{
                    "column": "local_column",
                    "references": "foreign_table.foreign_column",
                    "constraint_name": "fk_name"
                }}
            ],
            "indexes": [
                {{
                    "name": "index_name",
                    "columns": ["col1"],
                    "unique": false,
                    "type": "btree"
                }}
            ],
            "migration_file": "{file_name}"
        }}
    ],
    "views": [],
    "indexes": [],
    "relationships": [],
    "procedures": [],
    "triggers": []
}}

Focus on extracting accurate information from the migration file. For SQL migrations, look for CREATE TABLE, ALTER TABLE, CREATE INDEX, CREATE VIEW statements. For Java/C# migrations, look for migration builder calls. Return only valid JSON."""

            # Use structured JSON response for consistency
            messages = [{"role": "user", "content": prompt}]
            response = await self.generate_structured_json_response(
                messages, max_tokens=8192
            )

            try:
                schema_data = json.loads(response)

                # Log analysis results
                table_count = len(schema_data.get("tables", []))
                index_count = len(schema_data.get("indexes", []))
                logging.debug(
                    f"Single migration analysis completed with Ollama: {table_count} tables, {index_count} indexes"
                )

                return schema_data

            except json.JSONDecodeError as e:
                logging.error(
                    f"JSON parsing error for single migration {file_name} with Ollama: {e}"
                )
                return self._empty_schema_structure()

        except Exception as e:
            logging.error(
                f"Error analyzing single migration {migration_info.get('file_name', 'unknown')} with Ollama: {e}"
            )
            return self._empty_schema_structure()

    async def aggregate_migration_analyses(
        self, individual_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate individual migration analyses into a comprehensive schema.

        This method takes the results from multiple analyze_single_migration calls
        and consolidates them into a single unified schema, resolving duplicates
        and detecting cross-table relationships.

        Args:
            individual_analyses: List of schema dictionaries from individual migration analyses

        Returns:
            Dict containing the consolidated schema information
        """
        try:
            if not individual_analyses:
                logging.warning("No individual analyses to aggregate")
                return self._empty_schema_structure()

            logging.info(
                f"Aggregating {len(individual_analyses)} individual migration analyses with Ollama"
            )

            # Count total elements across all analyses
            total_tables = sum(
                len(analysis.get("tables", [])) for analysis in individual_analyses
            )
            total_indexes = sum(
                len(analysis.get("indexes", [])) for analysis in individual_analyses
            )
            total_views = sum(
                len(analysis.get("views", [])) for analysis in individual_analyses
            )

            logging.info(
                f"Individual analyses contain: {total_tables} tables, {total_indexes} indexes, {total_views} views"
            )

            # Build aggregation prompt with summaries of individual analyses
            analysis_summaries = []
            for i, analysis in enumerate(individual_analyses):
                summary = {
                    "analysis_id": i + 1,
                    "tables": analysis.get("tables", []),
                    "indexes": analysis.get("indexes", []),
                    "views": analysis.get("views", []),
                    "relationships": analysis.get("relationships", []),
                    "procedures": analysis.get("procedures", []),
                    "triggers": analysis.get("triggers", []),
                }
                analysis_summaries.append(summary)

            # Convert to JSON for the prompt
            analyses_json = json.dumps(analysis_summaries, indent=2)

            prompt = f"""You are analyzing database schema information extracted from individual migration files.
Your task is to aggregate these individual analyses into a comprehensive, consolidated database schema.

Individual Migration Analyses:
{analyses_json}

Instructions:
1. Consolidate all tables, removing duplicates (same table name) and merging information
2. Consolidate all indexes, removing duplicates and merging information
3. Consolidate all views, procedures, and triggers
4. Most importantly: DETECT RELATIONSHIPS between tables by analyzing:
   - Foreign key constraints mentioned in any migration
   - Column names that suggest relationships (e.g., user_id references users.id)
   - Naming patterns that indicate relationships
5. Resolve any conflicts between migrations (later migrations override earlier ones)
6. Maintain migration_file references to show which migration created each element

Return a JSON response with this exact structure:
{{
    "tables": [
        {{
            "name": "table_name",
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": [],
            "migration_file": "source_migration.sql"
        }}
    ],
    "views": [],
    "indexes": [],
    "relationships": [
        {{
            "from_table": "table1",
            "from_column": "column1",
            "to_table": "table2",
            "to_column": "column2",
            "relationship_type": "one_to_many",
            "constraint_name": "fk_name",
            "migration_file": "source_migration.sql"
        }}
    ],
    "procedures": [],
    "triggers": []
}}

Focus on detecting relationships between tables. Return only valid JSON."""

            # Use structured JSON response
            messages = [{"role": "user", "content": prompt}]
            response = await self.generate_structured_json_response(
                messages, max_tokens=16384
            )

            try:
                aggregated_schema = json.loads(response)

                # Log aggregation results
                final_tables = len(aggregated_schema.get("tables", []))
                final_relationships = len(aggregated_schema.get("relationships", []))
                final_indexes = len(aggregated_schema.get("indexes", []))

                logging.info(
                    f"Ollama aggregation completed: {final_tables} tables, {final_relationships} relationships, {final_indexes} indexes"
                )

                return aggregated_schema

            except json.JSONDecodeError as e:
                logging.error(
                    f"JSON parsing error during Ollama aggregation: {e}"
                )
                return self._simple_aggregation_fallback(individual_analyses)

        except Exception as e:
            logging.error(
                f"Error during Ollama migration analysis aggregation: {e}"
            )
            return self._simple_aggregation_fallback(individual_analyses)

    def _simple_aggregation_fallback(
        self, individual_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simple fallback aggregation without LLM when aggregation fails.

        Combines all elements from individual analyses and removes duplicate
        tables by name (keeping the last occurrence).

        Args:
            individual_analyses: List of schema dictionaries

        Returns:
            Dict containing the combined schema information
        """
        logging.info("Using simple aggregation fallback with Ollama")

        all_tables = []
        all_indexes = []
        all_views = []
        all_relationships = []
        all_procedures = []
        all_triggers = []

        for analysis in individual_analyses:
            all_tables.extend(analysis.get("tables", []))
            all_indexes.extend(analysis.get("indexes", []))
            all_views.extend(analysis.get("views", []))
            all_relationships.extend(analysis.get("relationships", []))
            all_procedures.extend(analysis.get("procedures", []))
            all_triggers.extend(analysis.get("triggers", []))

        # Remove duplicate tables by name (keep last occurrence)
        unique_tables = {}
        for table in all_tables:
            unique_tables[table.get("name", "unknown")] = table

        return {
            "tables": list(unique_tables.values()),
            "indexes": all_indexes,
            "views": all_views,
            "relationships": all_relationships,
            "procedures": all_procedures,
            "triggers": all_triggers,
        }

    def _empty_schema_structure(self) -> Dict[str, Any]:
        """Return empty schema structure."""
        return {
            "tables": [],
            "relationships": [],
            "indexes": [],
            "views": [],
            "procedures": [],
            "triggers": [],
        }

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