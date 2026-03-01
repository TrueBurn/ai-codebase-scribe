# Standard library imports
import asyncio
import json
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import boto3
import botocore
from botocore.config import Config as BotocoreConfig
from dotenv import load_dotenv

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


from src.utils.config_class import ScribeConfig

from ..analyzers.codebase import CodebaseAnalyzer
from ..utils.progress import ProgressTracker
from ..utils.prompt_cache_manager import PromptCacheManager
from ..utils.prompt_manager import PromptTemplate
from ..utils.retry import async_retry
from ..utils.tokens import TokenCounter

# Local imports
from .base_llm import BaseLLMClient
from .llm_utils import (
    find_common_dependencies,
    fix_markdown_issues,
    format_project_structure,
    get_default_order,
    identify_key_components,
    prepare_file_order_data,
    process_file_order_response,
)
from .message_manager import MessageManager

# Constants
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_TEMPERATURE = 0
BEDROCK_API_VERSION = "bedrock-2023-05-31"


class BedrockClientError(Exception):
    """Custom exception for Bedrock client errors."""


class BedrockClient(BaseLLMClient):
    """Handles all interactions with AWS Bedrock."""

    def __init__(self, config: ScribeConfig):
        """
        Initialize the BedrockClient with the provided configuration.

        Args:
            config: Configuration containing parameters
                - bedrock: Configuration with Bedrock-specific settings
                - debug: Boolean to enable debug output
                - template_path: Path to prompt templates
        """
        # Call parent class constructor
        super().__init__()

        # Initialize logger
        self.logger = _get_logger()

        # Load environment variables from .env file
        load_dotenv()

        # Use environment variables if available, otherwise use config
        self.region = os.getenv("AWS_REGION") or config.bedrock.region
        self.model_id = os.getenv("AWS_BEDROCK_MODEL_ID") or config.bedrock.model_id

        # Print model ID for debugging
        if config.debug:
            print(f"Using Bedrock model ID: {self.model_id}")

        # Set configuration properties
        self.max_tokens = config.bedrock.max_tokens
        self.max_output_tokens_architecture = (
            config.bedrock.max_output_tokens_architecture
        )
        self.max_output_tokens_persistence = (
            config.bedrock.max_output_tokens_persistence
        )
        self.retries = config.bedrock.retries
        self.retry_delay = config.bedrock.retry_delay
        self.timeout = config.bedrock.timeout
        self.debug = config.debug
        self.concurrency = config.bedrock.concurrency

        # Fallback model configuration
        self.fallback_model_id = (
            os.getenv("AWS_BEDROCK_FALLBACK_MODEL_ID")
            or config.bedrock.fallback_model_id
        )
        self.enable_fallback = config.bedrock.enable_fallback
        self.throttling_retry_delay = config.bedrock.throttling_retry_delay
        self.throttling_max_retries = config.bedrock.throttling_max_retries
        self.current_model_id = (
            self.model_id
        )  # Track which model is currently being used
        self.throttling_retry_count = 0  # Track throttling retry attempts

        # Store large repository configuration
        self.large_repo_threshold = config.large_repo.threshold
        self.large_repo_files_per_component = config.large_repo.files_per_component
        self.large_repo_smart_prioritization = config.large_repo.smart_prioritization
        self.large_repo_verbose_logging = config.large_repo.verbose_logging

        # Add concurrency support
        self.semaphore = asyncio.Semaphore(self.concurrency)

        # Track last operation's detailed token metrics
        self.last_operation_metrics = {}

        # Get SSL verification setting from config or environment
        # Environment variable takes precedence over config
        env_verify_ssl = os.getenv("AWS_VERIFY_SSL")
        if env_verify_ssl is not None:
            self.verify_ssl = env_verify_ssl.lower() != "false"
        else:
            self.verify_ssl = config.bedrock.verify_ssl

        # Set environment variable for tiktoken SSL verification to match our setting
        if not self.verify_ssl:
            os.environ["TIKTOKEN_VERIFY_SSL"] = "false"
            if self.debug:
                print("SSL verification disabled for tiktoken")
        # Initialize Bedrock client
        self.client = self._initialize_bedrock_client()

        # Initialize prompt template
        # Note: template_path is not in ScribeConfig yet, will need to be added
        self.prompt_template = PromptTemplate()

        # Add temperature setting
        self.temperature = config.bedrock.temperature

        # Initialize prompt cache manager
        self.cache_manager = PromptCacheManager(config)

        # Cache reusable components for prompt caching optimization
        self._cached_project_structure: Optional[str] = None
        self._cached_tech_report: Optional[str] = None

        # Initialize extended context configuration
        self.extended_context_enabled = config.bedrock.extended_context_enabled
        self.extended_context_beta_header = config.bedrock.extended_context_beta_header

    def _initialize_bedrock_client(self) -> boto3.client:
        """
        Initialize the AWS Bedrock client with proper configuration.

        Returns:
            boto3.client: Configured Bedrock client
        """
        # AWS SDK will automatically use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from env
        return boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            verify=self.verify_ssl,
            config=BotocoreConfig(
                connect_timeout=self.timeout,
                read_timeout=self.timeout,
                retries={"max_attempts": self.retries},
                max_pool_connections=max(self.concurrency, 10),
                tcp_keepalive=True,
            ),
        )

    async def validate_aws_credentials(self) -> bool:
        """
        Validate that AWS credentials are properly configured.

        Returns:
            bool: True if credentials are valid, False otherwise
        """
        try:
            # Try a simple operation to validate credentials
            await asyncio.to_thread(self.client.list_foundation_models)
            logger = _get_logger()
            logger.info("AWS credentials validated successfully", emoji="aws")
            return True
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("UnrecognizedClientException", "AccessDeniedException"):
                logger = _get_logger()
                logger.error(
                    f"AWS credential validation failed: {error_code}", emoji="aws"
                )
                if self.debug:
                    print(f"AWS credential error: {str(e)}")
                return False
            # For other errors, credentials might be valid but other issues exist
            return True
        except Exception as e:
            self.logger.warning(
                f"AWS credential validation error: {str(e)}", emoji="aws"
            )
            return False

    async def initialize(self) -> None:
        """
        Perform async initialization tasks.

        Initializes token counter, validates credentials, and sets up project structure.

        Raises:
            BedrockClientError: If initialization fails
        """
        try:
            # Initialize token counter
            self.init_token_counter()

            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    f"Initialized with model: {logger.format_filename(self.model_id)}",
                    emoji="aws",
                )
                logger.info(
                    f"Using AWS region: {logger.format_filename(self.region)}",
                    emoji="aws",
                )
                logger.info("Starting analysis...", emoji="analyze")
            else:
                self.logger.info(
                    f"Initialized with model: {self.model_id}", emoji="aws"
                )
                self.logger.info(f"Using AWS region: {self.region}", emoji="aws")
                self.logger.info("Starting analysis...", emoji="start")

            if self.debug:
                print(f"Selected model: {self.model_id}")
                print(
                    f"AWS credentials: {'Found' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not found'} in environment"
                )

                # Validate credentials
                is_valid = await self.validate_aws_credentials()
                logger = _get_logger()
                if is_valid:
                    logger.info(f"AWS credentials valid: {is_valid}", emoji="aws")
                else:
                    logger.warning(f"AWS credentials valid: {is_valid}", emoji="aws")

            # Initialize project structure if not already done
            if self.project_structure is None:
                self.project_structure = "Project structure not initialized. Please analyze repository first."

        except Exception as e:
            if self.debug:
                print(f"Initialization error: {str(e)}")
            raise BedrockClientError(f"Failed to initialize client: {str(e)}")

    def init_token_counter(self) -> None:
        """Initialize the token counter for this client."""
        self.token_counter = TokenCounter(model_name=self.model_id, debug=self.debug)

    def _supports_extended_context(self) -> bool:
        """
        Check if current model supports extended context (1M tokens).

        According to Anthropic official documentation:
        - ONLY Claude Sonnet 4 and Sonnet 4.5 support 1M context via anthropic_beta header
        - NOT supported: Opus 4, Haiku models, older Claude versions

        Returns:
            bool: True if model supports extended context, False otherwise
        """
        # ONLY Sonnet 4 and 4.5 support extended context
        extended_context_models = [
            "sonnet-4",  # Matches both claude-sonnet-4 and claude-sonnet-4.5
            "sonnet-4-5",  # Explicit match for 4.5
        ]

        # Check if current model ID contains any of these patterns
        model_id_lower = self.model_id.lower()

        # Must be Sonnet 4 or 4.5, NOT Haiku, NOT Opus
        is_sonnet_4 = any(
            pattern in model_id_lower for pattern in extended_context_models
        )
        is_haiku = "haiku" in model_id_lower
        is_opus = "opus" in model_id_lower

        return is_sonnet_4 and not is_haiku and not is_opus

    async def close(self) -> None:
        """
        Clean up resources when the client is no longer needed.

        This method should be called when you're done using the client to ensure
        proper cleanup of resources.
        """
        # Cancel any pending tasks
        tasks = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]

        for task in tasks:
            task.cancel()

        # Wait for tasks to be cancelled
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("BedrockClient resources cleaned up", emoji="success")

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_summary(
        self, content: str, file_type: str = "text", file_path: str = None
    ) -> Optional[str]:
        """Generate a summary for a file's content.

        Args:
            content: The content of the file to summarize
            file_type: The type/language of the file (default: "text")
            file_path: The path to the file (default: None)

        Returns:
            Optional[str]: The generated summary or None if an error occurred
        """
        async with self.semaphore:  # Use semaphore to control concurrency
            try:
                # Create a prompt that includes file information
                file_info = (
                    f"File: {file_path}\nType: {file_type}\n\n" if file_path else ""
                )
                prompt = f"{file_info}{content}"

                messages = MessageManager.get_file_summary_messages(prompt)

                # Use the new token-aware invocation method (disable cache logging during file processing)
                context = {"file_path": file_path} if file_path else None
                summary = await self._invoke_model_with_token_management(
                    messages, log_cache_metrics=False, context=context
                )

                # VALIDATION: Check if summary is valid
                is_valid, reason = self._validate_file_summary(summary, file_path)

                if not is_valid:
                    self.logger.warning(
                        f"Invalid summary for {file_path}: {reason}. Retrying once...",
                        emoji="warning",
                    )

                    # Retry with more explicit prompt
                    retry_prompt = f"{prompt}\n\nIMPORTANT: Provide detailed code analysis following the template structure. Do NOT provide a confirmation message."
                    retry_messages = MessageManager.get_file_summary_messages(
                        retry_prompt
                    )
                    summary = await self._invoke_model_with_token_management(
                        retry_messages, log_cache_metrics=False, context=context
                    )

                    # Validate retry
                    is_valid, reason = self._validate_file_summary(summary, file_path)
                    if not is_valid:
                        self.logger.error(
                            f"Retry failed for {file_path}: {reason}. Returning None.",
                            emoji="error",
                        )
                        return None

                return summary

            except Exception as e:
                self.logger.error(f"Error generating summary: {e}", emoji="error")
                return None

    # _update_progress method removed - now using ProgressTracker.update_progress_async

    def _fix_markdown_issues(self, content: str) -> str:
        """Fix common markdown formatting issues before returning content."""
        return fix_markdown_issues(content)

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(botocore.exceptions.ClientError, ConnectionError, TimeoutError),
    )
    async def generate_project_overview(self, file_manifest: dict) -> str:
        """Generate project overview based on file manifest."""
        try:
            # Get progress tracker instance
            progress_tracker = ProgressTracker.get_instance(Path("."))
            with progress_tracker.progress_bar(
                total=100,
                desc="Generating project overview",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            ) as pbar:
                try:
                    # Create progress update task
                    update_task = asyncio.create_task(
                        progress_tracker.update_progress_async(pbar)
                    )

                    # Get project name
                    logging.debug("Deriving project name...")
                    project_name = self._derive_project_name(file_manifest)
                    logging.debug(f"Project name derived: {project_name}")

                    # Update progress
                    pbar.update(10)

                    # Get detected technologies
                    logging.debug("Finding common dependencies...")
                    try:
                        tech_report = self._find_common_dependencies(file_manifest)
                        logging.debug(
                            f"Dependencies found, report length: {len(tech_report)}"
                        )
                    except Exception as e:
                        logging.error(f"Error in _find_common_dependencies: {str(e)}")
                        logging.error(f"Exception type: {type(e)}")
                        logging.error(f"Exception traceback: {traceback.format_exc()}")
                        tech_report = "No dependencies detected."

                    # Update progress
                    pbar.update(20)

                    # Get key components
                    logging.debug("Identifying key components...")
                    try:
                        key_components = self._identify_key_components(file_manifest)
                        logging.debug(
                            f"Key components identified, report length: {len(key_components)}"
                        )
                    except Exception as e:
                        logging.error(f"Error in _identify_key_components: {str(e)}")
                        logging.error(f"Exception type: {type(e)}")
                        logging.error(f"Exception traceback: {traceback.format_exc()}")
                        key_components = "No key components identified."

                    # Update progress
                    pbar.update(30)

                    # Get template content
                    template_content = self.prompt_template.get_template(
                        "project_overview"
                    ).format(
                        project_name=project_name,
                        file_count=len(file_manifest),
                        key_components=key_components,
                        dependencies=tech_report,
                    )

                    # Update progress
                    pbar.update(40)

                    # Get messages
                    messages = MessageManager.get_project_overview_messages(
                        self.project_structure, tech_report, template_content
                    )

                    # Update progress
                    pbar.update(50)

                    # Check token limits and truncate if needed
                    if self.token_counter:
                        messages = MessageManager.check_and_truncate_messages(
                            messages, self.token_counter, self.model_id
                        )

                    # Update progress
                    pbar.update(60)

                    # Extract system and user content
                    system_content = next(
                        (msg["content"] for msg in messages if msg["role"] == "system"),
                        "",
                    )
                    user_content = next(
                        (msg["content"] for msg in messages if msg["role"] == "user"),
                        "",
                    )

                    # Use the helper method to create and invoke the request
                    content = await self._create_and_invoke_bedrock_request(
                        system_content, user_content
                    )

                    # Update progress
                    pbar.update(70)

                    # Fix any markdown issues
                    fixed_content = self._fix_markdown_issues(content)

                    # Update progress
                    pbar.update(80)

                    return fixed_content
                except Exception as e:
                    if self.debug:
                        print(f"\nError generating overview: {str(e)}")
                    logging.error(f"Error in generate_project_overview: {e}")
                    logging.error(f"Exception type: {type(e)}")
                    logging.error(f"Exception traceback: {traceback.format_exc()}")
                    return f"This is a software project containing {len(file_manifest)} files."
                finally:
                    # Update progress
                    pbar.update(90)
                    # Cancel progress update task
                    if "update_task" in locals():
                        update_task.cancel()
        except Exception as e:
            if self.debug:
                print(f"\nError setting up progress tracker: {str(e)}")
            logging.error(f"Error in generate_project_overview setup: {e}")
            return f"This is a software project containing {len(file_manifest)} files."
            raise

    def _format_project_structure(
        self, file_manifest: dict, force_compression: Optional[bool] = None
    ) -> str:
        """Build a tree-like project structure string."""
        try:
            result = format_project_structure(
                file_manifest, self.debug, force_compression
            )
            # Ensure result is a string
            if not isinstance(result, str):
                logging.warning(
                    f"format_project_structure returned non-string: {type(result)}"
                )
                return "Project structure not available."
            return result
        except Exception as e:
            logging.error(f"Error in _format_project_structure: {e}")
            return "Project structure not available."

    def _find_common_dependencies(self, file_manifest: dict) -> str:
        """Extract common dependencies from file manifest."""
        try:
            # Log detailed information about file_manifest
            logging.debug(
                f"_find_common_dependencies called with file_manifest type: {type(file_manifest)}"
            )
            if file_manifest:
                sample_key = next(iter(file_manifest))
                sample_value = file_manifest[sample_key]
                logging.debug(
                    f"Sample key type: {type(sample_key)}, value: {sample_key}"
                )
                logging.debug(f"Sample value type: {type(sample_value)}")
                if hasattr(sample_value, "__dict__"):
                    logging.debug(f"Sample value attributes: {dir(sample_value)}")
                elif isinstance(sample_value, dict):
                    logging.debug(f"Sample value keys: {sample_value.keys()}")
            else:
                logging.debug("file_manifest is empty")

            # Convert file_manifest if needed
            # If file_manifest contains FileInfo objects but find_common_dependencies expects dicts
            converted_manifest = {}
            for path, info in file_manifest.items():
                if hasattr(info, "to_dict"):
                    # Convert FileInfo to dict if needed
                    converted_manifest[path] = info.to_dict()
                else:
                    # Keep as is
                    converted_manifest[path] = info

            result = find_common_dependencies(converted_manifest, self.debug)
            # Ensure result is a string
            if not isinstance(result, str):
                logging.warning(
                    f"find_common_dependencies returned non-string: {type(result)}"
                )
                return "No dependencies detected."
            return result
        except Exception as e:
            logging.error(f"Error in _find_common_dependencies: {e}")
            logging.error(f"Exception type: {type(e)}")
            logging.error(f"Exception traceback: {traceback.format_exc()}")
            return "No dependencies detected."

    def _get_cached_components(self, file_manifest: dict) -> tuple:
        """Get cached reusable components for prompt caching.

        This method ensures that the same project_structure and tech_report
        are used across all generators to enable cache reuse.
        """
        # Only compute once per session to ensure identical content
        if self._cached_project_structure is None:
            self._cached_project_structure = self._format_project_structure(
                file_manifest, force_compression=False
            )

        if self._cached_tech_report is None:
            self._cached_tech_report = self._find_common_dependencies(file_manifest)

        return self._cached_project_structure, self._cached_tech_report

    def _identify_key_components(self, file_manifest: dict) -> str:
        """Identify key components from file manifest."""
        try:
            # Convert file_manifest if needed
            # If file_manifest contains FileInfo objects but identify_key_components expects dicts
            converted_manifest = {}
            for path, info in file_manifest.items():
                if hasattr(info, "to_dict"):
                    # Convert FileInfo to dict if needed
                    converted_manifest[path] = info.to_dict()
                else:
                    # Keep as is
                    converted_manifest[path] = info

            result = identify_key_components(converted_manifest, self.debug)
            # Ensure result is a string
            if not isinstance(result, str):
                logging.warning(
                    f"identify_key_components returned non-string: {type(result)}"
                )
                return "No key components identified."
            return result
        except Exception as e:
            logging.error(f"Error in _identify_key_components: {e}")
            return "No key components identified."

    def _derive_project_name(self, file_manifest: dict) -> str:
        """Derive project name from repository structure."""
        try:
            # Create a temporary analyzer instance to use its method
            from src.utils.config_class import ScribeConfig

            config = ScribeConfig()
            config.debug = self.debug
            temp_analyzer = CodebaseAnalyzer(Path("."), config)
            temp_analyzer.file_manifest = file_manifest

            result = temp_analyzer.derive_project_name(self.debug)

            # Ensure result is a string
            if not isinstance(result, str) or not result.strip():
                logging.warning(
                    f"derive_project_name returned invalid result: {result}"
                )
                return "Unknown Project"

            return result
        except Exception as e:
            logging.error(f"Error in _derive_project_name: {e}")
            return "Unknown Project"

    def set_project_structure(self, structure: str) -> None:
        """
        Set the project structure for use in prompts.

        Args:
            structure: String representation of the project structure
        """
        self.project_structure = structure
        if self.debug:
            print(f"Project structure set ({len(structure)} chars)")

    def set_project_structure_from_manifest(
        self, file_manifest: Dict[str, Any]
    ) -> None:
        """
        Set the project structure from a file manifest.

        This is a convenience method that formats the file manifest into a
        string representation and then sets it as the project structure.

        Args:
            file_manifest: Dictionary mapping file paths to file information
        """
        self.project_structure = self._format_project_structure(file_manifest)

    async def _create_and_invoke_bedrock_request(
        self, system_content: str, user_content: str, max_tokens: Optional[int] = None
    ) -> str:
        """
        Helper method to create and invoke a Bedrock request with the given content.

        Args:
            system_content: System message content
            user_content: User message content
            max_tokens: Maximum tokens to generate (uses default if None)

        Returns:
            str: The generated content with markdown issues fixed

        Raises:
            Various exceptions from the underlying API call
        """
        # Combine for Claude
        combined_content = f"{system_content}\n\n{user_content}"

        # Create proper Bedrock format
        bedrock_messages = [
            {"role": "user", "content": [{"type": "text", "text": combined_content}]}
        ]

        # Use provided max_tokens or default
        tokens_to_generate = max_tokens or self.max_tokens

        # Invoke model
        response = await asyncio.to_thread(
            self.client.invoke_model,
            body=json.dumps(
                {
                    "anthropic_version": BEDROCK_API_VERSION,
                    "max_tokens": tokens_to_generate,
                    "messages": bedrock_messages,
                    "temperature": self.temperature,
                }
            ),
            modelId=self.model_id,
        )

        # Process response
        response_body = json.loads(response.get("body").read())
        content = response_body["content"][0]["text"]

        # Fix any markdown issues
        return self._fix_markdown_issues(content)

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_architecture_content(self, file_manifest: dict, analyzer) -> str:
        """Generate architecture documentation content with flow diagrams."""
        async with self.semaphore:
            # Get progress tracker instance
            progress_tracker = ProgressTracker.get_instance(Path("."))
            with progress_tracker.progress_bar(
                desc="Generating architecture documentation",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                ncols=150,
            ) as pbar:
                try:
                    update_task = asyncio.create_task(
                        progress_tracker.update_progress_async(pbar)
                    )

                    # Ensure project structure is set
                    if not self.project_structure or len(self.project_structure) < 10:
                        self.set_project_structure_from_manifest(file_manifest)
                        if self.debug:
                            print(
                                f"Project structure generated ({len(self.project_structure)} chars)"
                            )

                    # Get detected technologies
                    tech_report = self._find_common_dependencies(file_manifest)

                    # Get key components
                    key_components = self._identify_key_components(file_manifest)

                    # Create a summary of file contents for context
                    file_summaries = []

                    # First, categorize files by directory/component
                    file_by_component = {}
                    for path, info in file_manifest.items():
                        if info.get("summary") and not info.get("is_binary", False):
                            directory = str(Path(path).parent)
                            if directory not in file_by_component:
                                file_by_component[directory] = []
                            file_by_component[directory].append(
                                (path, info.get("summary", "No summary available"))
                            )

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

                    # Get messages from MessageManager
                    messages = MessageManager.get_architecture_content_messages(
                        self.project_structure, key_components, tech_report
                    )

                    # Add file summaries to the user message
                    for i, msg in enumerate(messages):
                        if msg["role"] == "user":
                            messages[i][
                                "content"
                            ] += f"\n\nFile Summaries:\n{file_summaries_text}"
                            break

                    # Extract system and user content
                    system_content = next(
                        (msg["content"] for msg in messages if msg["role"] == "system"),
                        "",
                    )
                    user_content = next(
                        (msg["content"] for msg in messages if msg["role"] == "user"),
                        "",
                    )

                    # Use the helper method to create and invoke the request
                    content = await self._create_and_invoke_bedrock_request(
                        system_content, user_content
                    )

                    update_task.cancel()

                    # Ensure the project structure is included in the output
                    if "```" not in content[:500]:
                        content = f"# Architecture Documentation\n\n## Project Structure\n```\n{self.project_structure}\n```\n\n{content}"

                    # Fix any remaining markdown issues
                    fixed_content = self._fix_markdown_issues(content)
                    return fixed_content

                except Exception as e:
                    if self.debug:
                        print(f"\nError generating architecture content: {str(e)}")
                    return "Error generating architecture documentation."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(botocore.exceptions.ClientError, ConnectionError, TimeoutError),
    )
    async def generate_architecture_doc(self, file_manifest: dict) -> str:
        """Generate architecture documentation based on file manifest."""
        try:
            # Get progress tracker instance
            progress_tracker = ProgressTracker.get_instance(Path("."))
            with progress_tracker.progress_bar(
                total=100,
                desc="Generating architecture documentation",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            ) as pbar:
                # Create progress update task
                update_task = asyncio.create_task(
                    progress_tracker.update_progress_async(pbar)
                )

                # Get project name
                _ = self._derive_project_name(file_manifest)

                # Get detected technologies
                tech_report = self._find_common_dependencies(file_manifest)

                # Get key components
                key_components = self._identify_key_components(file_manifest)

                # Format project structure without compression for architecture documentation
                self.project_structure = self._format_project_structure(
                    file_manifest, force_compression=False
                )

                # Get messages
                messages = MessageManager.get_architecture_content_messages(
                    self.project_structure, key_components, tech_report
                )

                # Update progress
                pbar.update(20)

                try:
                    content = await self._invoke_model_with_token_management(messages)

                    # Update progress
                    pbar.update(70)

                    # Cancel progress update task
                    update_task.cancel()

                    logging.info("Successfully received architecture content from LLM")
                    return content

                except botocore.exceptions.ClientError as e:
                    # Cancel progress update task
                    update_task.cancel()

                    error_code = e.response.get("Error", {}).get("Code", "Unknown")
                    error_message = e.response.get("Error", {}).get("Message", str(e))

                    if error_code in [
                        "UnrecognizedClientException",
                        "InvalidSignatureException",
                        "SignatureDoesNotMatch",
                        "ExpiredToken",
                    ]:
                        logging.error(
                            f"AWS authentication error: {error_code} - {error_message}"
                        )
                        logging.error(
                            "Please check your AWS credentials and ensure they are valid and not expired."
                        )

                        # Return a more helpful error message
                        return (
                            "# Architecture Documentation\n\n"
                            "## Error: AWS Authentication Failed\n\n"
                            f"Unable to generate architecture documentation due to an AWS authentication error: {error_code}.\n\n"
                            "### Possible Solutions:\n\n"
                            "1. Check that your AWS credentials are valid and not expired\n"
                            "2. Verify that your IAM user has permissions to access Bedrock\n"
                            "3. Try using a different LLM provider with `--llm-provider ollama`\n"
                        )
                    else:
                        logging.error(
                            f"Error in LLM architecture generation: {error_code} - {error_message}"
                        )
                        logging.error(f"Exception details: {traceback.format_exc()}")

                        # Return a fallback message
                        return "# Architecture Documentation\n\nUnable to generate architecture documentation due to an error."

                except Exception as e:
                    # Cancel progress update task
                    update_task.cancel()

                    logging.error(f"Error in LLM architecture generation: {str(e)}")
                    logging.error(f"Exception details: {traceback.format_exc()}")

                    # Return a fallback message
                    return "# Architecture Documentation\n\nUnable to generate architecture documentation due to an error."

        except Exception as e:
            if self.debug:
                print(f"\nError generating architecture documentation: {str(e)}")
            logging.error(f"Error in LLM architecture generation: {str(e)}")
            logging.error(f"Exception details: {traceback.format_exc()}")
            return "# Architecture Documentation\n\nUnable to generate architecture documentation due to an error."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_component_relationships(self, file_manifest: dict) -> str:
        """Generate description of how components interact."""
        async with self.semaphore:
            try:
                # Get detected technologies
                tech_report = self._find_common_dependencies(file_manifest)

                # Get messages from MessageManager
                messages = MessageManager.get_component_relationship_messages(
                    self.project_structure, tech_report
                )

                # Use the new token-aware invocation method
                content = await self._invoke_model_with_token_management(messages)
                return content

            except Exception as e:
                if self.debug:
                    print(f"\nError generating component relationships: {str(e)}")
                return "# Component Relationships\n\nUnable to generate component relationships due to an error."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(botocore.exceptions.ClientError, ConnectionError, TimeoutError),
    )
    async def enhance_documentation(
        self, existing_content: str, file_manifest: dict, doc_type: str
    ) -> str:
        """Enhance existing documentation with new insights."""
        try:
            # Get detected technologies
            tech_report = self._find_common_dependencies(file_manifest)

            # Get key components
            key_components = self._identify_key_components(file_manifest)

            # Ensure project structure is set
            if not self.project_structure or len(self.project_structure) < 10:
                self.set_project_structure_from_manifest(file_manifest)
                if self.debug:
                    print(
                        f"Project structure generated ({len(self.project_structure)} chars)"
                    )

            # Create context for template
            context = {"doc_type": doc_type, "existing_content": existing_content}

            # Get template content with context
            _ = self.prompt_template.get_template("enhance_documentation", context)

            # Get messages
            messages = MessageManager.get_enhance_documentation_messages(
                existing_content,
                self.project_structure,
                key_components,
                tech_report,
                doc_type,
            )

            # Use the token-aware invocation method
            content = await self._invoke_model_with_token_management(messages)

            # Fix any markdown issues
            return self._fix_markdown_issues(content)

        except Exception as e:
            if self.debug:
                print(f"\nError enhancing documentation: {str(e)}")
            logging.error(f"Error enhancing documentation: {str(e)}")
            return existing_content  # Return original content on error

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_usage_guide(self, file_manifest: dict) -> str:
        """
        Generate usage guide based on project structure.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated usage guide content in markdown format
        """
        async with self.semaphore:
            try:
                # Get messages from MessageManager
                messages = MessageManager.get_usage_guide_messages(
                    self.project_structure,
                    self._find_common_dependencies(file_manifest),
                )

                # Extract system and user content
                system_content = next(
                    (msg["content"] for msg in messages if msg["role"] == "system"), ""
                )
                user_content = next(
                    (msg["content"] for msg in messages if msg["role"] == "user"), ""
                )

                # Use the helper method to create and invoke the request
                return await self._create_and_invoke_bedrock_request(
                    system_content, user_content
                )

            except Exception as e:
                if self.debug:
                    print(f"\nError generating usage guide: {str(e)}")
                logging.error(f"Error generating usage guide: {str(e)}")
                return "### Usage\n\nUsage instructions could not be generated."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_contributing_guide(self, file_manifest: dict) -> str:
        """
        Generate contributing guide based on project structure.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated contributing guide content in markdown format
        """
        async with self.semaphore:
            try:
                # Get messages from MessageManager
                messages = MessageManager.get_contributing_guide_messages(
                    self.project_structure
                )

                # Extract system and user content
                system_content = next(
                    (msg["content"] for msg in messages if msg["role"] == "system"), ""
                )
                user_content = next(
                    (msg["content"] for msg in messages if msg["role"] == "user"), ""
                )

                # Use the helper method to create and invoke the request
                return await self._create_and_invoke_bedrock_request(
                    system_content, user_content
                )

            except Exception as e:
                if self.debug:
                    print(f"\nError generating contributing guide: {str(e)}")
                logging.error(f"Error generating contributing guide: {str(e)}")
                return "### Contributing\n\nContributing guidelines could not be generated."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_license_info(self, file_manifest: dict) -> str:
        """
        Generate license information based on project structure.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated license information content in markdown format
        """
        async with self.semaphore:
            try:
                # Get messages from MessageManager
                messages = MessageManager.get_license_info_messages(
                    self.project_structure
                )

                # Extract system and user content
                system_content = next(
                    (msg["content"] for msg in messages if msg["role"] == "system"), ""
                )
                user_content = next(
                    (msg["content"] for msg in messages if msg["role"] == "user"), ""
                )

                # Use the helper method to create and invoke the request
                return await self._create_and_invoke_bedrock_request(
                    system_content, user_content
                )

            except Exception as e:
                if self.debug:
                    print(f"\nError generating license info: {str(e)}")
                logging.error(f"Error generating license info: {str(e)}")
                return "This project's license information could not be determined."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_installation_guide(self, file_manifest: dict) -> str:
        """
        Generate installation guide based on project structure.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated installation guide content in markdown format
        """
        async with self.semaphore:
            try:
                # Get cached reusable components for prompt caching
                project_structure_cached, tech_report = self._get_cached_components(
                    file_manifest
                )

                # Get messages from MessageManager
                # Use get_installation_guide_messages if available, otherwise construct manually
                if hasattr(MessageManager, "get_installation_guide_messages"):
                    messages = MessageManager.get_installation_guide_messages(
                        project_structure_cached, tech_report
                    )
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a technical documentation expert. Generate comprehensive installation documentation.",
                        },
                        {
                            "role": "user",
                            "content": f"""Generate a detailed installation guide for this project.

<project_structure>
{project_structure_cached}
</project_structure>

<tech_report>
{tech_report}
</tech_report>

Include:
1. Prerequisites and system requirements
2. Step-by-step installation instructions
3. Configuration steps
4. Verification steps
5. Common installation issues and solutions

Use proper markdown formatting.""",
                        },
                    ]

                # Use the token management method that supports caching
                return await self._invoke_model_with_token_management(
                    messages, max_tokens=self.max_output_tokens_architecture
                )

            except Exception as e:
                if self.debug:
                    print(f"\nError generating installation guide: {str(e)}")
                self.logger.error(
                    f"Error generating installation guide: {str(e)}", emoji="error"
                )
                return "### Installation\n\nInstallation instructions could not be generated."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_troubleshooting_guide(self, file_manifest: dict) -> str:
        """
        Generate troubleshooting guide based on project structure.

        Args:
            file_manifest: Dictionary mapping file paths to file information

        Returns:
            str: Generated troubleshooting guide content in markdown format
        """
        async with self.semaphore:
            try:
                # Get cached reusable components for prompt caching
                project_structure_cached, tech_report = self._get_cached_components(
                    file_manifest
                )

                # Get messages from MessageManager
                if hasattr(MessageManager, "get_troubleshooting_guide_messages"):
                    messages = MessageManager.get_troubleshooting_guide_messages(
                        project_structure_cached, tech_report
                    )
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a technical documentation expert. Generate comprehensive troubleshooting documentation.",
                        },
                        {
                            "role": "user",
                            "content": f"""Generate a detailed troubleshooting guide for this project.

<project_structure>
{project_structure_cached}
</project_structure>

<tech_report>
{tech_report}
</tech_report>

Include:
1. Common issues and their solutions
2. Debugging tips and techniques
3. Log analysis guidance
4. Performance troubleshooting
5. Environment-specific issues
6. FAQ section

Use proper markdown formatting.""",
                        },
                    ]

                # Use the token management method that supports caching
                return await self._invoke_model_with_token_management(
                    messages, max_tokens=self.max_output_tokens_architecture
                )

            except Exception as e:
                if self.debug:
                    print(f"\nError generating troubleshooting guide: {str(e)}")
                self.logger.error(
                    f"Error generating troubleshooting guide: {str(e)}", emoji="error"
                )
                return "### Troubleshooting\n\nTroubleshooting guide could not be generated."

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def generate_persistence_doc(
        self, file_manifest: dict, persistence_info: Any
    ) -> str:
        """
        Generate persistence layer documentation using multi-part strategy.

        Splits documentation generation into 3 parts to leverage prompt caching:
        1. Overview + schema statistics (cached base)
        2. Detailed table documentation (batched, reuses cache)
        3. Relationships + ER diagram (reuses cache)

        Args:
            file_manifest: Dictionary mapping file paths to file information
            persistence_info: PersistenceLayerInfo object with analyzed persistence data

        Returns:
            str: Generated persistence documentation content in markdown format
        """
        async with self.semaphore:
            try:
                # Extract schema data
                schema_data = getattr(persistence_info, "schema_data", {})
                tables = schema_data.get("tables", [])
                relationships = schema_data.get("relationships", [])

                # If no tables, return early
                if not tables:
                    self.logger.info(
                        "No tables found in schema data, skipping detailed documentation",
                        emoji="database",
                    )
                    return "# Persistence Layer Documentation\n\nNo database schema information available."

                self.logger.info(
                    f"Generating persistence documentation for {len(tables)} tables using multi-part strategy",
                    emoji="database",
                )

                # === PART 1: Overview + Schema Statistics ===
                self.logger.info(
                    "Part 1/3: Generating overview and schema statistics",
                    emoji="document",
                )
                if hasattr(MessageManager, "get_persistence_overview_messages"):
                    overview_messages = (
                        MessageManager.get_persistence_overview_messages(
                            self.project_structure, persistence_info
                        )
                    )
                else:
                    overview_messages = [
                        {
                            "role": "system",
                            "content": "You are a database documentation expert.",
                        },
                        {
                            "role": "user",
                            "content": f"""Generate a persistence layer overview for a project with {len(tables)} tables and {len(relationships)} relationships.

Project Structure:
{self.project_structure}

Provide an overview including:
1. Database technology used
2. Schema statistics
3. Key design patterns observed
4. Data model overview""",
                        },
                    ]
                overview_content = await self._invoke_model_with_token_management(
                    overview_messages, max_tokens=4096
                )

                # === PART 2: Table Documentation (Batched) ===
                batch_size = 12
                tables_sorted = sorted(
                    tables, key=lambda t: len(t.get("foreign_keys", [])), reverse=True
                )
                table_batches = [
                    tables_sorted[i : i + batch_size]
                    for i in range(0, len(tables_sorted), batch_size)
                ]

                self.logger.info(
                    f"Part 2/3: Generating detailed documentation for {len(tables)} tables in {len(table_batches)} batch(es)",
                    emoji="document",
                )

                tables_content = ""
                for batch_num, table_batch in enumerate(table_batches, 1):
                    self.logger.info(
                        f"Processing table batch {batch_num}/{len(table_batches)} ({len(table_batch)} tables)",
                        emoji="processing",
                    )
                    if hasattr(MessageManager, "get_persistence_tables_batch_messages"):
                        batch_messages = (
                            MessageManager.get_persistence_tables_batch_messages(
                                self.project_structure,
                                table_batch,
                                batch_num,
                                len(table_batches),
                            )
                        )
                    else:
                        table_info = json.dumps(table_batch, indent=2, default=str)
                        batch_messages = [
                            {
                                "role": "system",
                                "content": "You are a database documentation expert.",
                            },
                            {
                                "role": "user",
                                "content": f"""Document the following database tables (batch {batch_num}/{len(table_batches)}):

{table_info}

For each table provide:
1. Table purpose and description
2. Column documentation
3. Constraints and indexes
4. Usage patterns""",
                            },
                        ]
                    batch_content = await self._invoke_model_with_token_management(
                        batch_messages, max_tokens=self.max_output_tokens_persistence
                    )
                    tables_content += batch_content + "\n\n"

                # === PART 3: Relationships + ER Diagram ===
                if relationships:
                    self.logger.info(
                        f"Part 3/3: Generating relationship documentation and ER diagram ({len(relationships)} relationships)",
                        emoji="document",
                    )

                    tables_summary = ", ".join(
                        [t.get("name", "unknown") for t in tables[:20]]
                    )
                    if len(tables) > 20:
                        tables_summary += f" (and {len(tables) - 20} more)"

                    if hasattr(
                        MessageManager, "get_persistence_relationships_messages"
                    ):
                        rel_messages = (
                            MessageManager.get_persistence_relationships_messages(
                                self.project_structure, relationships, tables_summary
                            )
                        )
                    else:
                        rel_info = json.dumps(relationships, indent=2, default=str)
                        rel_messages = [
                            {
                                "role": "system",
                                "content": "You are a database documentation expert.",
                            },
                            {
                                "role": "user",
                                "content": f"""Document the following database relationships and generate an ER diagram in Mermaid format.

Tables: {tables_summary}

Relationships:
{rel_info}

Include:
1. Relationship descriptions
2. Mermaid ER diagram
3. Data flow patterns""",
                            },
                        ]
                    relationships_content = (
                        await self._invoke_model_with_token_management(
                            rel_messages, max_tokens=8192
                        )
                    )
                else:
                    self.logger.info(
                        "Part 3/3: Skipping relationships (none found)",
                        emoji="document",
                    )
                    relationships_content = ""

                # === Combine all parts ===
                final_content = f"""# Persistence Layer Documentation

{overview_content}

## Tables

{tables_content}"""

                if relationships_content:
                    final_content += f"\n\n{relationships_content}"

                self.logger.info(
                    f"Multi-part persistence documentation completed: {len(tables)} tables, {len(relationships)} relationships",
                    emoji="complete",
                )

                return final_content

            except Exception as e:
                if self.debug:
                    print(f"\nError generating persistence documentation: {str(e)}")
                self.logger.error(
                    f"Error generating persistence documentation: {str(e)}",
                    emoji="error",
                )
                return "Persistence layer documentation could not be generated."

    def _get_default_order(self, core_files: dict, resource_files: dict) -> list[str]:
        """Get a sensible default order when LLM ordering fails."""
        return get_default_order(core_files, resource_files)

    async def get_file_order(self, project_files: dict) -> list[str]:
        """
        Ask LLM to determine optimal file processing order.

        Args:
            project_files: Dictionary mapping file paths to file information

        Returns:
            list[str]: Ordered list of file paths
        """
        try:
            print("\nStarting file order optimization...")
            logging.info("Preparing file order optimization request")

            # Use common utility to prepare data
            core_files, resource_files, files_info = prepare_file_order_data(
                project_files, self.debug
            )

            print(f"Sending request to LLM with {len(files_info)} files...")
            logging.info(
                f"Sending file order request to LLM with {len(files_info)} files"
            )

            # Get messages from MessageManager
            messages = MessageManager.get_file_order_messages(files_info)

            # Extract system and user content
            system_content = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), ""
            )
            user_content = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), ""
            )

            # Use the helper method to create and invoke the request
            content = await self._create_and_invoke_bedrock_request(
                system_content, user_content
            )

            # Use common utility to process response
            return process_file_order_response(
                content, core_files, resource_files, self.debug
            )

        except Exception as e:
            print(f"Error in file order optimization: {str(e)}")
            logging.error(f"Error getting file order: {str(e)}", exc_info=True)
            return list(project_files.keys())

    async def _invoke_with_prompt_caching(
        self,
        messages,
        max_tokens=None,
        retry_on_token_error=True,
        log_cache_metrics=True,
        context=None,
    ):
        """Invoke model using prompt caching for optimal performance."""
        import time

        start_time = time.time()

        try:
            # Convert standard messages to InvokeModel format with caching
            cached_content_blocks = self._convert_messages_to_cached_format(messages)

            # Extract system content
            system_content = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), ""
            )

            # Use provided max_tokens or default
            tokens_to_generate = max_tokens or self.max_tokens

            # Create InvokeModel request body with cache support
            bedrock_messages = [{"role": "user", "content": cached_content_blocks}]

            # Combine system content if present
            if system_content:
                bedrock_messages[0]["content"] = [
                    {"type": "text", "text": system_content}
                ] + cached_content_blocks

            request_body = json.dumps(
                {
                    "anthropic_version": BEDROCK_API_VERSION,
                    "max_tokens": tokens_to_generate,
                    "messages": bedrock_messages,
                    "temperature": self.temperature,
                }
            )

            # Use InvokeModel API (supports cache_control)
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.invoke_model,
                    body=request_body,
                    modelId=self.current_model_id,
                ),
                timeout=self.timeout,
            )

            # Extract content from InvokeModel response
            response_body = json.loads(response.get("body").read())
            content = response_body["content"][0]["text"]

            # Log detailed token and cache usage from response metadata
            generation_time = time.time() - start_time
            self._log_bedrock_cache_metrics(
                response_body, generation_time, log_cache_metrics
            )

            # Fix any markdown issues
            return self._fix_markdown_issues(content)

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if "Connection was closed" in error_message or error_code in (
                "RequestTimeout",
                "ServiceUnavailable",
            ):
                self.logger.warning(
                    f"Prompt cache connection issue: {error_message}",
                    emoji="prompt_cache",
                )
            else:
                self.logger.error(
                    f"Prompt cache operation failed: {e}", emoji="prompt_cache"
                )
            raise  # Re-raise to trigger fallback

        except ConnectionError as e:
            self.logger.warning(
                f"Prompt cache connection error: {str(e)}", emoji="prompt_cache"
            )
            raise

        except Exception as e:
            self.logger.error(
                f"Prompt cache operation failed: {e}", emoji="prompt_cache"
            )
            raise

    def _log_bedrock_cache_metrics(
        self, response, generation_time: float, log_metrics: bool = True
    ):
        """Log detailed token and cache usage metrics from Bedrock response."""
        try:
            usage = response.get("usage", {})

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            cache_write_tokens = usage.get("cache_creation_input_tokens", 0)

            # Always store detailed metrics for the last operation
            self.last_operation_metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "generation_time": generation_time,
            }

            if not log_metrics:
                return

            if cache_read_tokens > 0 or cache_write_tokens > 0:
                self.logger.info(
                    f"Cache usage: Read {cache_read_tokens} tokens, Wrote {cache_write_tokens} tokens",
                    emoji="cache",
                )

                if input_tokens > 0:
                    cache_hit_rate = (cache_read_tokens / input_tokens) * 100
                    self.logger.info(
                        f"Cache efficiency: {cache_hit_rate:.1f}% ({cache_read_tokens}/{input_tokens} tokens)",
                        emoji="cache",
                    )

            # Update cache manager with detailed metrics from response
            if hasattr(self, "cache_manager"):
                self.cache_manager.update_metrics_from_response(usage, generation_time)

        except Exception as e:
            self.logger.warning(f"Error logging cache metrics: {e}", emoji="warning")

    def _convert_messages_to_cached_format(self, messages):
        """Convert standard messages to prompt cache format with smart component detection."""
        content_blocks = []

        for message in messages:
            if message["role"] == "user":
                content = message["content"]

                cacheable_parts, non_cacheable_parts = self._split_content_for_caching(
                    content
                )

                for part in cacheable_parts:
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": part,
                            "cache_control": {"type": "ephemeral"},
                        }
                    )

                for part in non_cacheable_parts:
                    if part.strip():
                        content_blocks.append({"type": "text", "text": part})

        return content_blocks

    def _split_content_for_caching(self, content: str):
        """Split content into cacheable and non-cacheable parts.

        Returns:
            tuple: (cacheable_parts, non_cacheable_parts)
        """
        cacheable_parts = []
        non_cacheable_parts = []

        cacheable_tags = [
            "project_structure",
            "tech_report",
            "key_components",
            "repository_context",
        ]

        remaining_content = content

        for tag in cacheable_tags:
            pattern = f"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, remaining_content, re.DOTALL)

            for match in matches:
                tagged_content = f"<{tag}>\n{match.strip()}\n</{tag}>"
                cacheable_parts.append(tagged_content)
                remaining_content = remaining_content.replace(
                    f"<{tag}>{match}</{tag}>", "", 1
                )

        if not cacheable_parts and self._is_large_content(content):
            cacheable_parts.append(content)
            remaining_content = ""

        if remaining_content.strip():
            non_cacheable_parts.append(remaining_content.strip())

        return cacheable_parts, non_cacheable_parts

    def _is_large_content(self, content: str) -> bool:
        """Determine if content is large enough to benefit from caching."""
        if not self.cache_manager.cache_enabled:
            return False

        estimated_tokens = len(content) // 4
        should_cache = estimated_tokens >= self.cache_manager.min_cache_tokens

        return should_cache

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(botocore.exceptions.ClientError, ConnectionError, TimeoutError),
    )
    async def _invoke_model_with_token_management(
        self,
        messages,
        max_tokens=None,
        retry_on_token_error=True,
        log_cache_metrics=True,
        context=None,
    ):
        """Invoke model with automatic token management and optional prompt caching.

        Args:
            messages: List of messages to send to the model
            max_tokens: Maximum tokens for output (optional)
            retry_on_token_error: Whether to retry on token errors (default: True)
            log_cache_metrics: Whether to log cache metrics (default: True)
            context: Optional context information (e.g., {'file_path': 'path/to/file'}) for enhanced logging
        """

        # Check if prompt caching is enabled and message structure supports it
        if (
            self.cache_manager.cache_enabled
            and len(messages) >= 1
            and isinstance(messages, list)
        ):
            try:
                return await self._invoke_with_prompt_caching(
                    messages,
                    max_tokens,
                    retry_on_token_error,
                    log_cache_metrics,
                    context,
                )
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))

                if (
                    "Connection was closed" in error_message
                    or "ConnectionError" in error_message
                    or "too many connections" in error_message.lower()
                    or error_code in ("RequestTimeout", "ServiceUnavailable")
                ):
                    self.logger.warning(
                        f"Prompt cache connection error, attempting fallback model: {error_message}",
                        emoji="prompt_cache",
                    )
                    try:
                        return await self._handle_throttling_error(
                            e, messages, max_tokens
                        )
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Fallback handling failed: {str(fallback_error)}",
                            emoji="error",
                        )
                else:
                    self.logger.warning(
                        f"Prompt cache unavailable, using standard invocation: {e}",
                        emoji="prompt_cache",
                    )
            except Exception as e:
                self.logger.warning(
                    f"Prompt cache unavailable, using standard invocation: {e}",
                    emoji="prompt_cache",
                )

        # Standard invocation (existing logic)
        try:
            # Extract system and user content
            system_content = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), ""
            )
            user_content = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), ""
            )

            # Combine for Claude
            combined_content = f"{system_content}\n\n{user_content}"

            # Initialize extended context flag
            use_extended_context = False

            # Check token count before sending
            if self.token_counter:
                total_tokens = self.token_counter.count_tokens(combined_content)
                model_limit = self.token_counter.get_token_limit(self.model_id)

                tokens_to_generate = max_tokens or self.max_tokens

                # Safety buffer for token counting discrepancies
                safety_factor = 1.3
                adjusted_total_tokens = int(total_tokens * safety_factor)

                # Check if we should use extended context (1M tokens)
                if (
                    adjusted_total_tokens + tokens_to_generate > 200000
                    and self._supports_extended_context()
                ):
                    extended_context_enabled = getattr(
                        self, "extended_context_enabled", True
                    )

                    if extended_context_enabled:
                        use_extended_context = True
                        model_limit = 1000000

                        self.logger.warning(
                            f"Input exceeds 200k limit ({adjusted_total_tokens} tokens), "
                            f"automatically using extended context (1M)",
                            emoji="token",
                        )

                # Calculate effective input limit
                effective_input_limit = model_limit - tokens_to_generate

                if adjusted_total_tokens + tokens_to_generate > model_limit:
                    self.logger.warning(
                        f"Token limit exceeded: input {adjusted_total_tokens} + output {tokens_to_generate} > limit {model_limit}",
                        emoji="token",
                    )

                    target_tokens = (effective_input_limit / safety_factor) * 0.95
                    self.logger.info(
                        f"Reducing input to {int(target_tokens)} tokens to accommodate output",
                        emoji="processing",
                    )

                    # Use smart content prioritization
                    combined_content = self._prioritize_architectural_content(
                        combined_content, target_tokens=int(target_tokens)
                    )
                    new_tokens = self.token_counter.count_tokens(combined_content)

                    if new_tokens > effective_input_limit:
                        file_context = (
                            f" for {context.get('file_path', 'unknown file')}"
                            if context and context.get("file_path")
                            else ""
                        )
                        self.logger.warning(
                            f"Prioritization insufficient: {new_tokens} > {effective_input_limit}, applying emergency truncation{file_context}",
                            emoji="warning",
                        )
                        combined_content = self.token_counter.truncate_text(
                            combined_content,
                            int(effective_input_limit * 0.95),
                        )

            # Create proper Bedrock format
            bedrock_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": combined_content}],
                }
            ]

            # Use provided max_tokens or default
            tokens_to_generate = max_tokens or self.max_tokens

            # Create request body
            request_body_dict = {
                "anthropic_version": BEDROCK_API_VERSION,
                "max_tokens": tokens_to_generate,
                "messages": bedrock_messages,
                "temperature": self.temperature,
            }

            # Add anthropic_beta header if using extended context
            if use_extended_context:
                extended_context_beta_header = getattr(
                    self, "extended_context_beta_header", "context-1m-2025-08-07"
                )
                request_body_dict["anthropic_beta"] = [extended_context_beta_header]

                self.logger.info(
                    f"Using extended context with beta header: {extended_context_beta_header}",
                    emoji="token",
                )

            request_body = json.dumps(request_body_dict)

            logging.debug(f"Request body size: {len(request_body)} bytes")

            # Invoke model with timeout using current model ID (supports fallback)
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.invoke_model,
                        body=request_body,
                        modelId=self.current_model_id,
                    ),
                    timeout=self.timeout,
                )

                # Process response
                response_body = json.loads(response.get("body").read())
                content = response_body["content"][0]["text"]

                # Fix any markdown issues
                fixed_content = self._fix_markdown_issues(content)

                return fixed_content

            except asyncio.TimeoutError:
                self.logger.error(
                    f"Request timed out after {self.timeout} seconds", emoji="timer"
                )
                raise TimeoutError(
                    f"Bedrock API call timed out after {self.timeout} seconds"
                )

            except botocore.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))
                self.logger.error(
                    f"Bedrock API error: {error_code} - {error_message}", emoji="aws"
                )

                # Handle ThrottlingException with fallback model
                if (
                    error_code == "ThrottlingException"
                    or "throttling" in error_message.lower()
                    or "too many tokens" in error_message.lower()
                ):
                    self.logger.warning(
                        f"AWS throttling detected: {error_message}", emoji="aws"
                    )

                    try:
                        return await self._handle_throttling_error(
                            e, messages, max_tokens
                        )
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Fallback handling failed: {str(fallback_error)}",
                            emoji="error",
                        )

                # Handle connection errors with fallback model
                if (
                    "Connection was closed" in error_message
                    or "ConnectionError" in error_message
                    or "too many connections" in error_message.lower()
                    or error_code in ("RequestTimeout", "ServiceUnavailable")
                ):
                    self.logger.warning(
                        f"AWS connection issue detected: {error_message}", emoji="aws"
                    )

                    try:
                        return await self._handle_throttling_error(
                            e, messages, max_tokens
                        )
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Fallback handling failed: {str(fallback_error)}",
                            emoji="error",
                        )

                # Handle "Input is too long" error with emergency truncation
                if (
                    retry_on_token_error
                    and error_code == "ValidationException"
                    and "Input is too long" in error_message
                ):
                    self.logger.warning(
                        "Input too long error detected, attempting emergency truncation",
                        emoji="warning",
                    )

                    if self.token_counter:
                        max_emergency_tokens = int(model_limit * 0.5)
                        emergency_content = self.token_counter.truncate_text(
                            combined_content, max_emergency_tokens
                        )
                        emergency_tokens = self.token_counter.count_tokens(
                            emergency_content
                        )
                        self.logger.info(
                            f"Emergency truncation to {emergency_tokens} tokens (50% of limit)",
                            emoji="processing",
                        )

                        emergency_messages = [
                            {
                                "role": "system",
                                "content": "Provide a concise response due to input length constraints.",
                            },
                            {"role": "user", "content": emergency_content},
                        ]

                        return await self._invoke_model_with_token_management(
                            emergency_messages,
                            max_tokens=tokens_to_generate,
                            retry_on_token_error=False,
                        )

                # Re-raise the exception if we can't handle it
                raise

            except ConnectionError as e:
                self.logger.warning(f"Connection error detected: {str(e)}", emoji="aws")

                try:
                    return await self._handle_throttling_error(e, messages, max_tokens)
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback handling failed: {str(fallback_error)}",
                        emoji="error",
                    )
                    raise e

        except Exception as e:
            if self.debug:
                print(f"Error in model invocation: {str(e)}")
                print(f"Exception details: {traceback.format_exc()}")
            self.logger.error(f"Model invocation failed: {str(e)}", emoji="aws")
            if self.debug:
                self.logger.error(
                    f"Exception details: {traceback.format_exc()}", emoji="debug"
                )
            raise

    async def generate_structured_json_response(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a structured JSON response from the LLM with robust parsing.

        This method implements the base class interface for structured JSON responses.
        """
        try:
            # Extract system and user content
            system_content = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), ""
            )
            user_content = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), ""
            )

            # Enhance the prompt to ensure raw JSON response
            enhanced_user_content = f"""{user_content}

CRITICAL INSTRUCTIONS FOR JSON RESPONSE:
1. Return ONLY valid JSON - no explanatory text, no markdown formatting
2. Do NOT wrap the JSON in code blocks (```)
3. Do NOT include any text before or after the JSON
4. Start your response immediately with {{ and end with }}
5. Ensure the JSON is properly formatted and valid
6. Use double quotes for all strings
7. Do not include any comments in the JSON

Your response must be parseable by json.loads() function directly."""

            # Combine for Claude
            combined_content = f"{system_content}\n\n{enhanced_user_content}"

            # Use token management
            if self.token_counter:
                total_tokens = self.token_counter.count_tokens(combined_content)
                model_limit = self.token_counter.get_token_limit(self.model_id)

                tokens_to_generate = max_tokens or self.max_tokens
                safety_factor = 1.3
                adjusted_total_tokens = int(total_tokens * safety_factor)
                effective_input_limit = model_limit - tokens_to_generate

                if adjusted_total_tokens + tokens_to_generate > model_limit:
                    self.logger.warning(
                        "JSON invocation token limit exceeded: reducing input tokens",
                        emoji="token",
                    )
                    target_tokens = (effective_input_limit / safety_factor) * 0.95
                    combined_content = self._prioritize_architectural_content(
                        combined_content, target_tokens=int(target_tokens)
                    )

            # Prepare Bedrock request body
            bedrock_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": combined_content}],
                }
            ]

            request_body = json.dumps(
                {
                    "anthropic_version": BEDROCK_API_VERSION,
                    "max_tokens": max_tokens or self.max_tokens,
                    "messages": bedrock_messages,
                    "temperature": 0.1,  # Lower temperature for more consistent JSON structure
                }
            )

            # Invoke model with throttling detection and fallback
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.invoke_model,
                        body=request_body,
                        modelId=self.current_model_id,
                    ),
                    timeout=self.timeout,
                )

                # Process response - get raw content
                response_body = json.loads(response.get("body").read())
                raw_content = response_body["content"][0]["text"]

                # Robust JSON extraction
                return self._extract_json_from_response(raw_content)

            except asyncio.TimeoutError:
                self.logger.error(
                    f"JSON request timed out after {self.timeout} seconds",
                    emoji="timer",
                )
                raise TimeoutError(
                    f"Bedrock API call timed out after {self.timeout} seconds"
                )

            except botocore.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))
                self.logger.error(
                    f"Error in JSON Bedrock API call: {error_code} - {error_message}",
                    emoji="aws",
                )

                # Handle ThrottlingException with fallback model
                if (
                    error_code == "ThrottlingException"
                    or "throttling" in error_message.lower()
                    or "too many tokens" in error_message.lower()
                ):
                    self.logger.warning(
                        f"ThrottlingException detected in JSON generation: {error_message}",
                        emoji="aws",
                    )

                    try:
                        fallback_response = (
                            await self._handle_throttling_error_for_json(
                                e, messages, max_tokens
                            )
                        )
                        return fallback_response
                    except Exception as fallback_error:
                        self.logger.error(
                            f"JSON fallback handling failed: {str(fallback_error)}",
                            emoji="error",
                        )
                        raise e

                raise e

        except Exception as e:
            self.logger.error(
                f"Error in JSON model invocation: {str(e)}", emoji="error"
            )
            self.logger.error(
                f"Exception details: {traceback.format_exc()}", emoji="debug"
            )
            raise

    def _extract_json_from_response(self, response_content: str) -> str:
        """
        Extract valid JSON from response content with multiple fallback strategies.
        """
        if not response_content:
            raise ValueError("Empty response content")

        # Strategy 1: Try parsing as-is
        try:
            json.loads(response_content.strip())
            return response_content.strip()
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown code blocks
        if "```json" in response_content or "```" in response_content:
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
                                return extracted
                            except json.JSONDecodeError:
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
            return cleaned
        except json.JSONDecodeError as e:
            # Strategy 5: JSON repair for large responses
            try:
                repaired_json = self._repair_malformed_json(cleaned, e)
                if repaired_json:
                    json.loads(repaired_json)
                    return repaired_json
            except (json.JSONDecodeError, Exception):
                pass

            self.logger.warning(
                f"All JSON extraction strategies failed. Last error: {e}",
                emoji="warning",
            )
            raise ValueError(f"Could not extract valid JSON from response: {e}")

    def _repair_malformed_json(
        self, json_content: str, error: json.JSONDecodeError
    ) -> Optional[str]:
        """
        Attempt to repair common JSON syntax errors in large responses.
        """
        try:
            error_pos = getattr(error, "pos", 0)
            error_msg = str(error)

            # Strategy 1: Fix invalid escape sequences
            if "Invalid \\escape" in error_msg or "Invalid escape" in error_msg:
                repaired = json_content
                repaired = re.sub(r"\\x", r"\\\\x", repaired)
                repaired = re.sub(r'\\([^"\\/bfnrtu])', r"\\\\\\1", repaired)

                if repaired != json_content:
                    return repaired

            # Strategy 2: Fix missing comma errors
            if "Expecting ',' delimiter" in error_msg:
                before_error = json_content[:error_pos]
                at_error = (
                    json_content[error_pos : error_pos + 10]
                    if error_pos < len(json_content)
                    else ""
                )

                if (
                    before_error.rstrip().endswith("}")
                    or before_error.rstrip().endswith("]")
                ) and (
                    at_error.lstrip().startswith('"')
                    or at_error.lstrip().startswith("{")
                    or at_error.lstrip().startswith("[")
                ):
                    insertion_pos = len(before_error.rstrip())
                    repaired = (
                        json_content[:insertion_pos]
                        + ","
                        + json_content[insertion_pos:]
                    )
                    return repaired

                repaired = json_content[:error_pos] + "," + json_content[error_pos:]
                return repaired

            # Strategy 3: Fix trailing comma errors
            elif (
                "Expecting ':' delimiter" in error_msg or "Expecting value" in error_msg
            ):
                repaired = re.sub(r",(\s*[}\]])", r"\1", json_content)
                if repaired != json_content:
                    return repaired

            # Strategy 4: Fix unclosed objects/arrays in very large responses
            elif "Expecting" in error_msg and error_pos > len(json_content) * 0.8:
                open_braces = json_content.count("{") - json_content.count("}")
                open_brackets = json_content.count("[") - json_content.count("]")

                if open_braces > 0 or open_brackets > 0:
                    repair_suffix = "}" * open_braces + "]" * open_brackets
                    repaired = json_content + repair_suffix
                    return repaired

            return None

        except Exception:
            return None

    def _prioritize_architectural_content(
        self, content: str, target_tokens: int
    ) -> str:
        """
        Intelligently prioritize architectural content to preserve key information.
        """
        if not self.token_counter:
            return content[: int(len(content) * 0.8)]

        current_tokens = self.token_counter.count_tokens(content)
        if current_tokens <= target_tokens:
            return content

        # Split content into sections for analysis
        lines = content.split("\n")

        critical_sections = []
        file_summaries = []
        component_sections = []
        other_content = []

        current_section = []
        section_type = "other"

        for line in lines:
            if line.startswith("## Component:"):
                if current_section:
                    if section_type == "critical":
                        critical_sections.extend(current_section)
                    elif section_type == "component":
                        component_sections.extend(current_section)
                    elif section_type == "file":
                        file_summaries.extend(current_section)
                    else:
                        other_content.extend(current_section)
                current_section = [line]
                section_type = "component"
            elif line.startswith("File:") and any(
                keyword in line.lower()
                for keyword in [
                    "controller",
                    "service",
                    "config",
                    "main",
                    "app",
                    "index",
                ]
            ):
                if current_section:
                    if section_type == "critical":
                        critical_sections.extend(current_section)
                    elif section_type == "component":
                        component_sections.extend(current_section)
                    elif section_type == "file":
                        file_summaries.extend(current_section)
                    else:
                        other_content.extend(current_section)
                current_section = [line]
                section_type = "critical"
            elif line.startswith("File:"):
                if current_section:
                    if section_type == "critical":
                        critical_sections.extend(current_section)
                    elif section_type == "component":
                        component_sections.extend(current_section)
                    elif section_type == "file":
                        file_summaries.extend(current_section)
                    else:
                        other_content.extend(current_section)
                current_section = [line]
                section_type = "file"
            elif any(
                keyword in line.lower()
                for keyword in [
                    "project structure",
                    "dependencies",
                    "technology",
                    "architecture",
                ]
            ):
                if current_section:
                    if section_type == "critical":
                        critical_sections.extend(current_section)
                    elif section_type == "component":
                        component_sections.extend(current_section)
                    elif section_type == "file":
                        file_summaries.extend(current_section)
                    else:
                        other_content.extend(current_section)
                current_section = [line]
                section_type = "critical"
            else:
                current_section.append(line)

        if current_section:
            if section_type == "critical":
                critical_sections.extend(current_section)
            elif section_type == "component":
                component_sections.extend(current_section)
            elif section_type == "file":
                file_summaries.extend(current_section)
            else:
                other_content.extend(current_section)

        # Build prioritized content
        prioritized_content = []

        prioritized_content.extend(critical_sections)
        current_tokens = self.token_counter.count_tokens("\n".join(prioritized_content))

        if current_tokens < target_tokens:
            remaining_tokens = target_tokens - current_tokens
            component_content = "\n".join(component_sections)
            component_tokens = self.token_counter.count_tokens(component_content)

            if component_tokens <= remaining_tokens:
                prioritized_content.extend(component_sections)
                current_tokens += component_tokens
            else:
                truncated_components = self._truncate_to_token_limit(
                    component_content, remaining_tokens
                )
                prioritized_content.extend(truncated_components.split("\n"))
                current_tokens = target_tokens

        if current_tokens < target_tokens:
            remaining_tokens = target_tokens - current_tokens
            for file_content_line in file_summaries:
                line_tokens = self.token_counter.count_tokens(file_content_line)
                if current_tokens + line_tokens <= target_tokens:
                    prioritized_content.append(file_content_line)
                    current_tokens += line_tokens
                else:
                    break

        if current_tokens < target_tokens and other_content:
            remaining_tokens = target_tokens - current_tokens
            other_content_str = "\n".join(other_content)
            other_tokens = self.token_counter.count_tokens(other_content_str)

            if other_tokens <= remaining_tokens:
                prioritized_content.extend(other_content)
            else:
                truncated_other = self._truncate_to_token_limit(
                    other_content_str, remaining_tokens
                )
                prioritized_content.extend(truncated_other.split("\n"))

        return "\n".join(prioritized_content)

    def _truncate_to_token_limit(self, content: str, token_limit: int) -> str:
        """
        Truncate content to fit within token limit while preserving structure.
        """
        if not self.token_counter:
            ratio = token_limit / (len(content) / 4)
            return content[: int(len(content) * ratio)]

        current_tokens = self.token_counter.count_tokens(content)
        if current_tokens <= token_limit:
            return content

        # Binary search for optimal truncation point
        lines = content.split("\n")
        left, right = 0, len(lines)

        while left < right:
            mid = (left + right + 1) // 2
            test_content = "\n".join(lines[:mid])
            test_tokens = self.token_counter.count_tokens(test_content)

            if test_tokens <= token_limit:
                left = mid
            else:
                right = mid - 1

        return "\n".join(lines[:left])

    async def _handle_throttling_error(
        self, error: Exception, messages, max_tokens=None
    ) -> str:
        """
        Handle ThrottlingException by switching to fallback model if available.
        """
        if not self.enable_fallback:
            self.logger.error(
                "Resource constraint error occurred but fallback is disabled",
                emoji="error",
            )
            raise error

        if self.current_model_id == self.fallback_model_id:
            self.logger.error(
                "Resource constraint on fallback model - no further fallback available",
                emoji="error",
            )
            raise error

        if self.throttling_retry_count >= self.throttling_max_retries:
            self.logger.error(
                f"Exceeded maximum fallback retries ({self.throttling_max_retries})",
                emoji="error",
            )
            raise error

        self.throttling_retry_count += 1

        self.logger.warning(
            f"Resource constraint detected on primary model: {self.current_model_id}",
            emoji="aws",
        )
        self.logger.info(
            f"Switching to fallback model: {self.fallback_model_id}", emoji="aws"
        )
        self.logger.info(
            f"Retry attempt {self.throttling_retry_count}/{self.throttling_max_retries}",
            emoji="sync",
        )

        original_model = self.current_model_id
        self.current_model_id = self.fallback_model_id

        try:
            self.logger.info(
                f"Waiting {self.throttling_retry_delay}s before retrying with fallback model",
                emoji="timer",
            )
            await asyncio.sleep(self.throttling_retry_delay)

            self.logger.info(
                f"Attempting request with fallback model: {self.fallback_model_id}",
                emoji="aws",
            )
            response = await self._invoke_model_with_token_management(
                messages, max_tokens=max_tokens, retry_on_token_error=True
            )

            self.throttling_retry_count = 0

            self.logger.info(
                f"Successfully completed request with fallback model: {self.fallback_model_id}",
                emoji="complete",
            )
            return response

        except Exception as fallback_error:
            self.current_model_id = original_model

            self.logger.error(
                f"Fallback model {self.fallback_model_id} also failed: {str(fallback_error)}",
                emoji="error",
            )

            if (
                "ThrottlingException" in str(fallback_error)
                or "throttling" in str(fallback_error).lower()
            ):
                self.logger.error(
                    "Both primary and fallback models are experiencing resource constraints",
                    emoji="error",
                )
                raise error
            else:
                raise fallback_error

    async def _handle_throttling_error_for_json(
        self, error: Exception, messages, max_tokens=None
    ) -> str:
        """
        Handle ThrottlingException specifically for JSON generation by switching to fallback model.
        """
        if not self.enable_fallback:
            self.logger.error(
                "Throttling error occurred but fallback is disabled", emoji="error"
            )
            raise error

        if self.current_model_id == self.fallback_model_id:
            self.logger.error(
                "Throttling error occurred on fallback model - no further fallback available",
                emoji="error",
            )
            raise error

        if self.throttling_retry_count >= self.throttling_max_retries:
            self.logger.error(
                f"Exceeded maximum throttling retries ({self.throttling_max_retries})",
                emoji="error",
            )
            raise error

        self.throttling_retry_count += 1

        self.logger.warning(
            f"ThrottlingException detected in JSON generation on model {self.current_model_id}",
            emoji="aws",
        )
        self.logger.info(
            f"Switching to fallback model for JSON generation: {self.fallback_model_id}",
            emoji="aws",
        )

        original_model = self.current_model_id
        self.current_model_id = self.fallback_model_id

        try:
            self.logger.info(
                f"Waiting {self.throttling_retry_delay}s before retrying JSON generation with fallback model",
                emoji="timer",
            )
            await asyncio.sleep(self.throttling_retry_delay)

            # Extract system and user content
            system_content = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), ""
            )
            user_content = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), ""
            )

            enhanced_user_content = f"""{user_content}

CRITICAL INSTRUCTIONS FOR JSON RESPONSE:
1. Return ONLY valid JSON - no explanatory text, no markdown formatting
2. Do NOT wrap the JSON in code blocks (```)
3. Do NOT include any text before or after the JSON
4. Start your response immediately with {{ and end with }}
5. Ensure the JSON is properly formatted and valid
6. Use double quotes for all strings
7. Do not include any comments in the JSON

Your response must be parseable by json.loads() function directly."""

            combined_content = f"{system_content}\n\n{enhanced_user_content}"

            if self.token_counter:
                total_tokens = self.token_counter.count_tokens(combined_content)
                model_limit = self.token_counter.get_token_limit(self.model_id)

                tokens_to_generate = max_tokens or self.max_tokens
                safety_factor = 1.3
                adjusted_total_tokens = int(total_tokens * safety_factor)
                effective_input_limit = model_limit - tokens_to_generate

                if adjusted_total_tokens + tokens_to_generate > model_limit:
                    target_tokens = (effective_input_limit / safety_factor) * 0.95
                    combined_content = self._prioritize_architectural_content(
                        combined_content, target_tokens=int(target_tokens)
                    )

            bedrock_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": combined_content}],
                }
            ]

            request_body = json.dumps(
                {
                    "anthropic_version": BEDROCK_API_VERSION,
                    "max_tokens": max_tokens or self.max_tokens,
                    "messages": bedrock_messages,
                    "temperature": 0.1,
                }
            )

            api_response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.invoke_model,
                    body=request_body,
                    modelId=self.current_model_id,
                ),
                timeout=self.timeout,
            )

            response_body = json.loads(api_response.get("body").read())
            raw_content = response_body["content"][0]["text"]

            response = self._extract_json_from_response(raw_content)

            self.throttling_retry_count = 0

            self.logger.info(
                f"Successfully completed JSON request with fallback model: {self.fallback_model_id}",
                emoji="complete",
            )
            return response

        except Exception as fallback_error:
            self.current_model_id = original_model

            self.logger.error(
                f"Fallback model {self.fallback_model_id} also failed for JSON generation: {str(fallback_error)}",
                emoji="error",
            )

            if (
                "ThrottlingException" in str(fallback_error)
                or "throttling" in str(fallback_error).lower()
            ):
                raise error
            else:
                raise fallback_error

    def _is_response_truncated(self, content: str) -> bool:
        """Detect if the LLM response appears to be truncated."""
        if not content or len(content) < 100:
            return True

        truncation_indicators = [
            content.rstrip().endswith(("...", "..", ". .", ".")),
            not content.rstrip().endswith(
                (".", "!", "?", "```", ">", ")", "]", "}", '"', "'")
            ),
            content.count("##") < 3 and "architecture" in content.lower(),
            len(content) < 1000 and "architecture" in content.lower(),
        ]

        return any(truncation_indicators)

    def _validate_architecture_response(self, content: str) -> bool:
        """Validate that architecture documentation contains expected sections."""
        if not content:
            return False

        expected_sections = [
            "project structure",
            "architecture",
            "component",
            "overview",
        ]

        content_lower = content.lower()
        sections_found = sum(
            1 for section in expected_sections if section in content_lower
        )

        return sections_found >= 2 and len(content) > 500

    @async_retry(
        retries=3,
        delay=1.0,
        backoff=2.0,
        max_delay=30.0,
        jitter=True,
        exceptions=(ConnectionError, TimeoutError),
    )
    async def analyze_migration_contents(
        self, migration_contents: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze migration file contents to extract detailed schema information using LLM intelligence.

        Args:
            migration_contents: List of migration file info dictionaries

        Returns:
            Dict containing extracted schema information
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

        messages = [
            {
                "role": "user",
                "content": f"""You are a database schema analyst. Analyze the following migration files and extract detailed schema information.

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
    "indexes": [],
    "views": [],
    "procedures": [],
    "triggers": [],
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
- Keep the response concise - limit to 5 most important tables and their key columns
- Return valid JSON only - no additional text or explanations""",
            }
        ]

        try:
            response = await self.generate_structured_json_response(
                messages, max_tokens=16384
            )

            if not response:
                self.logger.warning(
                    "Empty response from LLM for migration analysis", emoji="warning"
                )
                return self._empty_schema_structure()

            try:
                schema_data = json.loads(response)
                self.logger.info(
                    f"Successfully analyzed {len(migration_contents)} migration files",
                    emoji="success",
                )
                return schema_data
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Unexpected JSON parsing error after extraction: {e}",
                    emoji="error",
                )
                return self._empty_schema_structure()

        except Exception as e:
            self.logger.error(
                f"Error during migration content analysis: {e}", emoji="error"
            )
            return self._empty_schema_structure()

    async def analyze_single_migration(
        self, migration_info: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze a single migration file to extract schema information.
        """
        try:
            file_name = migration_info.get("file_name", "unknown")
            migration_type = migration_info.get("type", "unknown")
            content = migration_info.get("content", "")

            logging.debug(
                f"Analyzing individual migration: {file_name} ({len(content)} chars)"
            )

            if not content.strip():
                self.logger.warning(
                    f"Empty migration content for {file_name}", emoji="warning"
                )
                return self._empty_schema_structure()

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
            "foreign_keys": [],
            "indexes": [],
            "migration_file": "{file_name}"
        }}
    ],
    "views": [],
    "indexes": [],
    "relationships": [],
    "procedures": [],
    "triggers": []
}}

Focus on extracting accurate information from the migration file. Return only valid JSON."""

            messages = [{"role": "user", "content": prompt}]
            response = await self.generate_structured_json_response(
                messages, max_tokens=8192
            )

            try:
                schema_data = json.loads(response)

                table_count = len(schema_data.get("tables", []))
                index_count = len(schema_data.get("indexes", []))
                logging.debug(
                    f"Single migration analysis completed: {table_count} tables, {index_count} indexes"
                )

                return schema_data

            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"JSON parsing error for migration {file_name}: {e}",
                    emoji="warning",
                )

                try:
                    fallback_result = self._extract_partial_schema_from_text(
                        response, file_name
                    )
                    if fallback_result and fallback_result.get("tables"):
                        return fallback_result
                except Exception:
                    pass

                return self._empty_schema_structure()

        except Exception as e:
            self.logger.error(
                f"Error analyzing single migration {migration_info.get('file_name', 'unknown')}: {e}",
                emoji="error",
            )
            return self._empty_schema_structure()

    async def aggregate_migration_analyses(
        self, individual_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate individual migration analyses into a comprehensive schema.
        """
        try:
            if not individual_analyses:
                self.logger.warning(
                    "No individual analyses to aggregate", emoji="warning"
                )
                return self._empty_schema_structure()

            self.logger.info(
                f"Aggregating {len(individual_analyses)} individual migration analyses",
                emoji="processing",
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
    "tables": [...],
    "views": [...],
    "indexes": [...],
    "relationships": [...],
    "procedures": [...],
    "triggers": [...]
}}

Return only valid JSON."""

            messages = [{"role": "user", "content": prompt}]
            response = await self.generate_structured_json_response(
                messages, max_tokens=self.max_output_tokens_persistence
            )

            try:
                aggregated_schema = json.loads(response)

                final_tables = len(aggregated_schema.get("tables", []))
                final_relationships = len(aggregated_schema.get("relationships", []))
                final_indexes = len(aggregated_schema.get("indexes", []))

                self.logger.info(
                    f"Aggregation completed: {final_tables} tables, {final_relationships} relationships, {final_indexes} indexes",
                    emoji="complete",
                )

                return aggregated_schema

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"JSON parsing error during aggregation: {e}", emoji="error"
                )
                return self._simple_aggregation_fallback(individual_analyses)

        except Exception as e:
            self.logger.error(
                f"Error during migration analysis aggregation: {e}", emoji="error"
            )
            return self._simple_aggregation_fallback(individual_analyses)

    def _simple_aggregation_fallback(
        self, individual_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simple fallback aggregation without LLM when aggregation fails."""
        self.logger.info("Using simple aggregation fallback", emoji="processing")

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

        # Remove duplicate tables by name
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

    def _extract_partial_schema_from_text(
        self, response_text: str, file_name: str
    ) -> Dict[str, Any]:
        """
        Extract partial schema information from malformed JSON using regex fallback.
        """
        schema_info = self._empty_schema_structure()

        try:
            tables = []
            table_matches = re.findall(
                r'"tables":\s*\[(.*?)\]', response_text, re.DOTALL
            )
            if table_matches:
                for table_section in table_matches:
                    name_matches = re.findall(r'"name":\s*"([^"]+)"', table_section)
                    for table_name in name_matches:
                        tables.append(
                            {
                                "name": table_name,
                                "columns": [],
                                "source_file": file_name,
                            }
                        )

            indexes = []
            index_matches = re.findall(
                r'"indexes":\s*\[(.*?)\]', response_text, re.DOTALL
            )
            if index_matches:
                for index_section in index_matches:
                    name_matches = re.findall(r'"name":\s*"([^"]+)"', index_section)
                    for index_name in name_matches:
                        indexes.append(
                            {
                                "name": index_name,
                                "table": "",
                                "columns": [],
                                "source_file": file_name,
                            }
                        )

            schema_info["tables"] = tables
            schema_info["indexes"] = indexes

        except Exception as e:
            logging.debug(f"Regex fallback extraction failed: {e}")

        return schema_info
