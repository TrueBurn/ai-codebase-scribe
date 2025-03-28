from typing import Dict, Any, Type, Optional, ClassVar, Mapping, Union
from .base_llm import BaseLLMClient
from .ollama import OllamaClient
from .bedrock import BedrockClient
import logging
from types import TracebackType

from src.utils.config_class import ScribeConfig
# No need to import config_to_dict and dict_to_config anymore

class LLMClientFactoryError(Exception):
    """Base exception for LLM client factory errors."""
    pass

class ConfigValidationError(LLMClientFactoryError):
    """Exception raised when configuration validation fails."""
    pass

class ClientInitializationError(LLMClientFactoryError):
    """Exception raised when client initialization fails."""
    pass

class LLMClientFactory:
    """
    Factory for creating LLM clients.
    
    This factory creates and initializes LLM clients based on configuration.
    
    Supported providers:
    - 'ollama': Uses the local Ollama instance (default)
    - 'bedrock': Uses AWS Bedrock service
    
    Example:
        ```python
        from src.utils.config_class import ScribeConfig, OllamaConfig
        
        config = ScribeConfig()
        config.llm_provider = 'ollama'
        config.ollama = OllamaConfig(
            base_url='http://localhost:11434',
            max_tokens=4096
        )
        
        # Create the client using the factory
        llm_client = await LLMClientFactory.create_client(config)
        ```
    """
    
    # Registry of available client types
    _client_registry: ClassVar[Dict[str, Type[BaseLLMClient]]] = {
        'ollama': OllamaClient,
        'bedrock': BedrockClient
    }
    
    @classmethod
    def register_client_type(cls, provider_name: str, client_class: Type[BaseLLMClient]) -> None:
        """
        Register a new client type with the factory.
        
        Args:
            provider_name: The name of the provider (used in configuration)
            client_class: The client class to instantiate for this provider
            
        Example:
            ```python
            class NewProviderClient(BaseLLMClient):
                # Implementation here
                pass
                
            LLMClientFactory.register_client_type('new_provider', NewProviderClient)
            ```
        """
        cls._client_registry[provider_name.lower()] = client_class
        logging.info(f"Registered new LLM client type: {provider_name}")
    
    @classmethod
    def validate_config(cls, config: ScribeConfig) -> bool:
        """
        Validate the configuration.
        
        Args:
            config: The ScribeConfig instance
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Ensure config is a ScribeConfig instance
        if not isinstance(config, ScribeConfig):
            raise ConfigValidationError(f"Configuration must be a ScribeConfig instance, got {type(config)}")
        
        # Check if provider is valid
        provider = config.llm_provider.lower()
        if provider not in cls._client_registry:
            raise ConfigValidationError(
                f"Invalid provider: {provider}. "
                f"Supported providers: {', '.join(cls._client_registry.keys())}"
            )
                
        return True
    
    @classmethod
    async def create_client(cls, config: ScribeConfig) -> BaseLLMClient:
        """
        Create and initialize an LLM client based on configuration.
        
        Args:
            config: ScribeConfig instance with provider and client-specific settings
            
        Returns:
            BaseLLMClient: Initialized LLM client
            
        Raises:
            ConfigValidationError: If configuration is invalid
            ClientInitializationError: If client initialization fails
        """
        # Validate configuration
        try:
            cls.validate_config(config)
        except ConfigValidationError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise
            
        provider = config.llm_provider.lower()
        
        # Add debug logging to see what provider is being selected
        if config.debug:
            logging.info(f"Creating LLM client with provider: {provider}")
        
        if provider == 'bedrock':
            try:
                if config.debug:
                    logging.info("Initializing Bedrock client...")
                client = BedrockClient(config)
                await client.initialize()
                if config.debug:
                    logging.info("Bedrock client initialized successfully")
                return client
            except Exception as e:
                error_msg = f"Error initializing Bedrock client: {e}"
                logging.error(error_msg)
                print(f"\n{error_msg}")
                print("Falling back to Ollama client...")
                provider = 'ollama'
        
        # Default to Ollama
        if config.debug:
            logging.info("Initializing Ollama client...")
            
        try:
            client = OllamaClient(config)
            await client.initialize()
            return client
        except Exception as e:
            error_msg = f"Error initializing Ollama client: {e}"
            logging.error(error_msg)
            raise ClientInitializationError(error_msg) from e