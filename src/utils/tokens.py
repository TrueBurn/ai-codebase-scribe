import tiktoken
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import os
import ssl
import requests
from urllib3.exceptions import SSLError
import json
from pathlib import Path

class TokenCounter:
    """Utility for counting tokens in LLM requests and responses."""
    
    # Default buffer percentages for different operations
    DEFAULT_BUFFER_PERCENTAGE = 0.1  # Default 10% buffer for will_exceed_limit
    DEFAULT_TRUNCATE_BUFFER = 0.9    # Default 90% of limit for truncate_text
    DEFAULT_CHUNK_BUFFER = 0.8       # Default 80% of limit for chunk_text
    DEFAULT_OVERSIZED_TARGET = 0.8   # Default 80% target for handle_oversized_input
    
    # Default token limits for different models
    MODEL_LIMITS = {
        # Anthropic models
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-7-sonnet-20250219-v1:0": 200000,
        
        # OpenAI models
        "gpt-3.5-turbo": 16385,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        
        # Ollama models - conservative defaults
        "llama2": 4096,
        "llama3": 8192,
        "mistral": 8192,
        "codellama": 16384,
        "phi3": 4096,
        "gemma": 8192,
        "mixtral": 32768,
    }
    
    # Mapping of model IDs to encoding names
    ENCODING_MAP = {
        # Default to cl100k for most models
        "default": "cl100k_base",
        
        # Specific encodings for certain model families
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "claude": "cl100k_base",
    }
    
    def __init__(self, model_name: str = "default", debug: bool = False,
                 buffer_percentage: float = None,
                 truncate_buffer: float = None,
                 chunk_buffer: float = None,
                 oversized_target: float = None,
                 config_path: Optional[str] = None):
        """Initialize the token counter with a specific model.
        
        Args:
            model_name: Name of the model to use for token counting
            debug: Whether to enable debug logging
            buffer_percentage: Buffer percentage for will_exceed_limit (default: 10%)
            truncate_buffer: Buffer percentage for truncate_text (default: 90%)
            chunk_buffer: Buffer percentage for chunk_text (default: 80%)
            oversized_target: Target percentage for handle_oversized_input (default: 80%)
            config_path: Optional path to a JSON config file with model limits
        """
        self.model_name = model_name
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Set buffer percentages (use provided values or defaults)
        self.buffer_percentage = buffer_percentage if buffer_percentage is not None else self.DEFAULT_BUFFER_PERCENTAGE
        self.truncate_buffer = truncate_buffer if truncate_buffer is not None else self.DEFAULT_TRUNCATE_BUFFER
        self.chunk_buffer = chunk_buffer if chunk_buffer is not None else self.DEFAULT_CHUNK_BUFFER
        self.oversized_target = oversized_target if oversized_target is not None else self.DEFAULT_OVERSIZED_TARGET
        
        # Load model limits from config file if provided
        if config_path:
            self._load_model_limits_from_config(config_path)
        
        # Check for SSL verification environment variable
        self.verify_ssl = True
        env_verify_ssl = os.getenv('TIKTOKEN_VERIFY_SSL')
        if env_verify_ssl is not None:
            self.verify_ssl = env_verify_ssl.lower() != 'false'
        
        # Determine the encoding to use
        encoding_name = self._get_encoding_name(model_name)
        try:
            # Handle SSL verification for tiktoken
            if not self.verify_ssl:
                self._configure_ssl_verification()
            
            self.encoding = tiktoken.get_encoding(encoding_name)
            if debug:
                self.logger.debug(f"Using encoding {encoding_name} for model {model_name}")
                
        except (Exception, ImportError) as e:
            self.logger.warning(f"Failed to load encoding {encoding_name}: {str(e)}. Falling back to cl100k_base.")
            try:
                # Try to load a simpler encoding as fallback
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception as e2:
                self.logger.error(f"Failed to load fallback encoding: {str(e2)}. Token counting will be approximate.")
                # Create a very simple fallback tokenizer that just counts words
                self.encoding = self._create_fallback_tokenizer()
    
    def _configure_ssl_verification(self) -> None:
        """Configure SSL verification settings for tiktoken."""
        # Create a custom SSL context that doesn't verify certificates
        ssl_context = ssl._create_unverified_context()
        # Patch the requests session to use our custom SSL context
        old_merge_environment_settings = requests.Session.merge_environment_settings
        
        def merge_environment_settings(self, url, proxies, stream, verify, cert):
            settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
            settings['verify'] = False
            return settings
        
        # Apply the patch
        requests.Session.merge_environment_settings = merge_environment_settings
    
    def _load_model_limits_from_config(self, config_path: str) -> None:
        """Load model limits from a JSON configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                if 'model_limits' in config_data and isinstance(config_data['model_limits'], dict):
                    # Update the MODEL_LIMITS dictionary with values from config
                    for model, limit in config_data['model_limits'].items():
                        self.MODEL_LIMITS[model] = int(limit)
                    
                    if self.debug:
                        self.logger.debug(f"Loaded model limits from {config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load model limits from {config_path}: {str(e)}")
    
    def _create_fallback_tokenizer(self) -> Any:
        """Create a simple fallback tokenizer that approximates tokens."""
        class FallbackTokenizer:
            def encode(self, text):
                # Approximate tokens: 1 token ≈ 4 chars in English
                return [0] * (len(text) // 4 + 1)
                
            def decode(self, tokens):
                # Return the original text if we're trying to decode
                return "".join([" "] * len(tokens))
        
        return FallbackTokenizer()
    
    def _get_encoding_name(self, model_name: str) -> str:
        """Get the appropriate encoding name for a model."""
        # Check for exact match
        if model_name in self.ENCODING_MAP:
            return self.ENCODING_MAP[model_name]
        
        # Check for partial matches (e.g., if model_name contains "gpt-4")
        for prefix, encoding in self.ENCODING_MAP.items():
            if prefix in model_name:
                return encoding
        
        # Default to cl100k_base for most models
        return self.ENCODING_MAP["default"]
    
    def get_token_limit(self, model_name: Optional[str] = None) -> int:
        """Get the token limit for a specific model."""
        model = model_name or self.model_name
        
        # Check for exact matches
        if model in self.MODEL_LIMITS:
            return self.MODEL_LIMITS[model]
        
        # Check for partial matches
        for prefix, limit in self.MODEL_LIMITS.items():
            if model.startswith(prefix):
                return limit
        
        # Default to a conservative limit
        return 4096
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if not text:
            return 0
        
        tokens = self.encoding.encode(text)
        return len(tokens)
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of chat messages."""
        total_tokens = 0
        
        for message in messages:
            # Count tokens in the message content
            content = message.get("content", "")
            role = message.get("role", "user")
            
            # Add tokens for the content
            content_tokens = self.count_tokens(content)
            
            # Add tokens for message metadata (role, etc.)
            # This is an approximation and may vary by model
            metadata_tokens = 4  # ~4 tokens for message formatting
            
            total_tokens += content_tokens + metadata_tokens
        
        # Add tokens for overall formatting
        total_tokens += 3  # ~3 tokens for overall chat formatting
        
        return total_tokens
    
    def will_exceed_limit(self, text_or_messages: Union[str, List[Dict[str, str]]],
                           model_name: Optional[str] = None,
                           buffer_percentage: Optional[float] = None) -> Tuple[bool, int]:
        """Check if text or messages will exceed the token limit for the model.
        
        Args:
            text_or_messages: Text string or list of message dictionaries
            model_name: Optional model name to check against (defaults to instance model)
            buffer_percentage: Percentage buffer to leave (defaults to self.buffer_percentage)
            
        Returns:
            Tuple of (will_exceed, token_count)
        """
        model = model_name or self.model_name
        limit = self.get_token_limit(model)
        
        # Use provided buffer_percentage or instance default
        buffer = buffer_percentage if buffer_percentage is not None else self.buffer_percentage
        
        # Apply buffer
        effective_limit = int(limit * (1 - buffer))
        
        # Count tokens
        if isinstance(text_or_messages, str):
            token_count = self.count_tokens(text_or_messages)
        else:
            token_count = self.count_message_tokens(text_or_messages)
        
        will_exceed = token_count > effective_limit
        
        if self.debug and will_exceed:
            self.logger.warning(
                f"Token count {token_count} exceeds effective limit {effective_limit} "
                f"({limit} with {buffer_percentage:.0%} buffer) for model {model}"
            )
        
        return will_exceed, token_count
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None,
                   overlap: int = 100) -> List[str]:
        """Split text into chunks that won't exceed token limits.
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum tokens per chunk (defaults to model limit with buffer)
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if chunk_size is None:
            # Default to model limit with configured buffer
            limit = self.get_token_limit(self.model_name)
            chunk_size = int(limit * self.chunk_buffer)
        
        # Encode the full text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= chunk_size:
            return [text]
        
        # Split into chunks
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Decode this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move to next chunk with overlap
            start_idx = end_idx - overlap
            if start_idx >= len(tokens):
                break
        
        if self.debug:
            self.logger.debug(f"Split text of {len(tokens)} tokens into {len(chunks)} chunks")
        
        return chunks
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (defaults to model limit with buffer)
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        if max_tokens is None:
            # Default to model limit with configured buffer
            limit = self.get_token_limit(self.model_name)
            max_tokens = int(limit * self.truncate_buffer)
        
        # Encode the text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)
        
        if self.debug:
            self.logger.debug(f"Truncated text from {len(tokens)} to {max_tokens} tokens")
        
        return truncated_text
    
    def handle_oversized_input(self, text: str, target_percentage: Optional[float] = None) -> str:
        """Handle oversized input by intelligently reducing it to fit within token limits.
        
        This is more sophisticated than simple truncation - it tries to preserve
        important parts of the input while reducing overall size.
        
        Args:
            text: The oversized text input
            target_percentage: Target percentage of model's token limit (defaults to self.oversized_target)
            
        Returns:
            Reduced text that should fit within limits
        """
        if not text:
            return ""
        
        # Use provided target_percentage or instance default
        target = target_percentage if target_percentage is not None else self.oversized_target
        
        # Get model limit and calculate target token count
        model_limit = self.get_token_limit(self.model_name)
        target_tokens = int(model_limit * target)
        
        # Count current tokens
        current_tokens = self.count_tokens(text)
        
        # If already under target, return as is
        if current_tokens <= target_tokens:
            return text
        
        # Log the reduction needed
        if self.debug:
            self.logger.debug(f"Reducing input from {current_tokens} to {target_tokens} tokens")
        
        # Split text into lines
        lines = text.split('\n')
        
        # Strategy 1: Remove duplicate empty lines
        text = '\n'.join([lines[i] for i in range(len(lines)) if i == 0 or not (lines[i] == '' and lines[i-1] == '')])
        
        # Check if that was enough
        current_tokens = self.count_tokens(text)
        if current_tokens <= target_tokens:
            if self.debug:
                self.logger.debug(f"Reduced to {current_tokens} tokens by removing duplicate empty lines")
            return text
        
        # Strategy 2: For project structure, keep only first level of directories and files
        if "Project Structure:" in text:
            sections = text.split("Project Structure:")
            before = sections[0]
            structure_and_after = sections[1]
            
            # Split the structure section from what comes after it
            structure_parts = structure_and_after.split("\n\n", 1)
            structure = structure_parts[0]
            after = structure_parts[1] if len(structure_parts) > 1 else ""
            
            # Simplify the structure - keep only first level
            structure_lines = structure.split('\n')
            simplified_structure = []
            for line in structure_lines:
                # Keep lines with 0 or 1 level of indentation
                if not line.startswith('    ') or line.startswith('  ') and not line.startswith('    '):
                    simplified_structure.append(line)
            
            # Reassemble the text
            text = before + "Project Structure:\n" + '\n'.join(simplified_structure) + "\n\n" + after
            
            # Check if that was enough
            current_tokens = self.count_tokens(text)
            if current_tokens <= target_tokens:
                if self.debug:
                    self.logger.debug(f"Reduced to {current_tokens} tokens by simplifying project structure")
                return text
        
        # Strategy 3: If still too large, use smart sampling
        # Take the beginning, important middle sections, and the end
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= target_tokens:
            return text
        
        # Calculate how to distribute tokens
        start_tokens = target_tokens // 2
        end_tokens = target_tokens - start_tokens
        
        # Get start and end tokens
        start_text = self.encoding.decode(tokens[:start_tokens])
        end_text = self.encoding.decode(tokens[-end_tokens:])
        
        # Combine with a note about truncation
        result = start_text + "\n\n... [content reduced for length] ...\n\n" + end_text
        
        if self.debug:
            final_tokens = self.count_tokens(result)
            self.logger.debug(f"Reduced to {final_tokens} tokens using smart sampling")
        
        return result
