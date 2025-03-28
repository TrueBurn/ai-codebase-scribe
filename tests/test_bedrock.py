import unittest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import sys
import os
import json
import pytest
import asyncio
import botocore.exceptions
from typing import Dict, Any, Optional, List

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the module
from src.clients.bedrock import BedrockClient, BedrockClientError, BEDROCK_API_VERSION


@pytest.fixture
def bedrock_config():
    """Fixture to provide a test configuration for BedrockClient."""
    from src.utils.config_class import ScribeConfig, BedrockConfig
    
    config = ScribeConfig()
    config.bedrock = BedrockConfig(
        region='us-east-1',
        model_id='test-model-id',
        max_tokens=4096,
        timeout=120,
        retries=3,
        retry_delay=1.0,
        temperature=0
    )
    config.debug = True
    return config


@pytest.mark.asyncio
async def test_initialization(bedrock_config):
    """Test initialization of BedrockClient."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch('src.clients.bedrock.TokenCounter') as mock_token_counter, \
         patch('src.clients.bedrock.PromptTemplate') as mock_prompt_template, \
         patch.dict('os.environ', {'AWS_BEDROCK_MODEL_ID': ''}):  # Clear environment variable
        
        # Setup mocks
        mock_boto3_client.return_value = MagicMock()
        mock_token_counter.return_value = MagicMock()
        mock_prompt_template.return_value = MagicMock()
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Verify initialization
        assert client.region == 'us-east-1'
        # The model_id comes from the config now that env var is cleared
        assert client.model_id == 'test-model-id'
        assert client.max_tokens == 4096
        assert client.timeout == 120
        assert client.retries == 3
        assert client.retry_delay == 1.0
        assert client.temperature == 0
        assert client.debug == True
        
        # Verify boto3 client was created
        mock_boto3_client.assert_called_once_with(
            'bedrock-runtime',
            region_name='us-east-1',
            verify=False,  # The actual implementation sets verify=False
            config=unittest.mock.ANY  # BotocoreConfig is complex to verify exactly
        )


@pytest.mark.asyncio
async def test_validate_aws_credentials_success(bedrock_config):
    """Test successful validation of AWS credentials."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch('src.clients.bedrock.asyncio.to_thread') as mock_to_thread:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_to_thread.return_value = MagicMock()
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Test validation
        result = await client.validate_aws_credentials()
        
        # Verify result
        assert result is True
        mock_to_thread.assert_called_once_with(mock_client.list_foundation_models)


@pytest.mark.asyncio
async def test_validate_aws_credentials_failure(bedrock_config):
    """Test failed validation of AWS credentials."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch('src.clients.bedrock.asyncio.to_thread') as mock_to_thread:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Create error response
        error_response = {
            'Error': {
                'Code': 'UnrecognizedClientException',
                'Message': 'Invalid credentials'
            }
        }
        mock_exception = botocore.exceptions.ClientError(error_response, 'operation')
        mock_to_thread.side_effect = mock_exception
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Test validation
        result = await client.validate_aws_credentials()
        
        # Verify result
        assert result is False


@pytest.mark.asyncio
async def test_create_and_invoke_bedrock_request(bedrock_config):
    """Test the _create_and_invoke_bedrock_request method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch('src.clients.bedrock.asyncio.to_thread') as mock_to_thread, \
         patch.dict('os.environ', {'AWS_BEDROCK_MODEL_ID': ''}):  # Clear environment variable
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            'content': [{'text': 'Test response'}]
        })
        mock_response.get.return_value = mock_body
        
        # Set up the to_thread mock to return our mock_response
        mock_to_thread.return_value = mock_response
        
        # Create client
        client = BedrockClient(bedrock_config)
        client._fix_markdown_issues = MagicMock(return_value="Fixed test response")
        
        # Test method
        result = await client._create_and_invoke_bedrock_request(
            "System content",
            "User content"
        )
        
        # Verify result
        assert result == "Fixed test response"
        
        # Verify API call was made
        assert mock_to_thread.called
        
        # We can't easily verify the exact request body in this test
        # because asyncio.to_thread is mocked and doesn't preserve the arguments
        # in the same way as a regular function call


@pytest.mark.asyncio
async def test_generate_usage_guide(bedrock_config):
    """Test the generate_usage_guide method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch.object(BedrockClient, '_create_and_invoke_bedrock_request') as mock_invoke:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_invoke.return_value = "Test usage guide"
        
        # Create client
        client = BedrockClient(bedrock_config)
        client._find_common_dependencies = MagicMock(return_value="Test dependencies")
        
        # Mock MessageManager
        with patch('src.clients.bedrock.MessageManager') as mock_message_manager:
            mock_message_manager.get_usage_guide_messages.return_value = [
                {"role": "system", "content": "System content"},
                {"role": "user", "content": "User content"}
            ]
            
            # Test method
            file_manifest = {"file1.py": {}, "file2.py": {}}
            result = await client.generate_usage_guide(file_manifest)
            
            # Verify result
            assert result == "Test usage guide"
            
            # Verify MessageManager call
            mock_message_manager.get_usage_guide_messages.assert_called_once_with(
                client.project_structure,
                "Test dependencies"
            )
            
            # Verify invoke call
            mock_invoke.assert_called_once_with("System content", "User content")


@pytest.mark.asyncio
async def test_generate_contributing_guide(bedrock_config):
    """Test the generate_contributing_guide method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch.object(BedrockClient, '_create_and_invoke_bedrock_request') as mock_invoke:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_invoke.return_value = "Test contributing guide"
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Mock MessageManager
        with patch('src.clients.bedrock.MessageManager') as mock_message_manager:
            mock_message_manager.get_contributing_guide_messages.return_value = [
                {"role": "system", "content": "System content"},
                {"role": "user", "content": "User content"}
            ]
            
            # Test method
            file_manifest = {"file1.py": {}, "file2.py": {}}
            result = await client.generate_contributing_guide(file_manifest)
            
            # Verify result
            assert result == "Test contributing guide"
            
            # Verify MessageManager call
            mock_message_manager.get_contributing_guide_messages.assert_called_once_with(
                client.project_structure
            )
            
            # Verify invoke call
            mock_invoke.assert_called_once_with("System content", "User content")


@pytest.mark.asyncio
async def test_generate_license_info(bedrock_config):
    """Test the generate_license_info method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch.object(BedrockClient, '_create_and_invoke_bedrock_request') as mock_invoke:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_invoke.return_value = "Test license info"
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Mock MessageManager
        with patch('src.clients.bedrock.MessageManager') as mock_message_manager:
            mock_message_manager.get_license_info_messages.return_value = [
                {"role": "system", "content": "System content"},
                {"role": "user", "content": "User content"}
            ]
            
            # Test method
            file_manifest = {"file1.py": {}, "file2.py": {}}
            result = await client.generate_license_info(file_manifest)
            
            # Verify result
            assert result == "Test license info"
            
            # Verify MessageManager call
            mock_message_manager.get_license_info_messages.assert_called_once_with(
                client.project_structure
            )
            
            # Verify invoke call
            mock_invoke.assert_called_once_with("System content", "User content")


@pytest.mark.asyncio
async def test_get_file_order(bedrock_config):
    """Test the get_file_order method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch.object(BedrockClient, '_create_and_invoke_bedrock_request') as mock_invoke, \
         patch('src.clients.bedrock.prepare_file_order_data') as mock_prepare_data, \
         patch('src.clients.bedrock.process_file_order_response') as mock_process_response:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_invoke.return_value = "Test file order response"
        
        # Mock prepare_file_order_data
        mock_prepare_data.return_value = (
            {"core_file1.py": {}},  # core_files
            {"resource_file1.py": {}},  # resource_files
            "Files info"  # files_info
        )
        
        # Mock process_file_order_response
        mock_process_response.return_value = ["file1.py", "file2.py"]
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Mock MessageManager
        with patch('src.clients.bedrock.MessageManager') as mock_message_manager:
            mock_message_manager.get_file_order_messages.return_value = [
                {"role": "system", "content": "System content"},
                {"role": "user", "content": "User content"}
            ]
            
            # Test method
            project_files = {"file1.py": {}, "file2.py": {}}
            result = await client.get_file_order(project_files)
            
            # Verify result
            assert result == ["file1.py", "file2.py"]
            
            # Verify prepare_file_order_data call
            mock_prepare_data.assert_called_once_with(project_files, client.debug)
            
            # Verify MessageManager call
            mock_message_manager.get_file_order_messages.assert_called_once_with("Files info")
            
            # Verify invoke call
            mock_invoke.assert_called_once_with("System content", "User content")
            
            # Verify process_file_order_response call
            mock_process_response.assert_called_once_with(
                "Test file order response",
                {"core_file1.py": {}},
                {"resource_file1.py": {}},
                client.debug
            )


@pytest.mark.asyncio
async def test_close(bedrock_config):
    """Test the close method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch('src.clients.bedrock.asyncio.all_tasks') as mock_all_tasks, \
         patch('src.clients.bedrock.asyncio.current_task') as mock_current_task, \
         patch('src.clients.bedrock.asyncio.gather') as mock_gather, \
         patch.dict('os.environ', {'AWS_BEDROCK_MODEL_ID': ''}):  # Clear environment variable
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock tasks
        current_task = MagicMock()
        mock_current_task.return_value = current_task
        
        task1 = MagicMock()
        task1.done.return_value = False
        task2 = MagicMock()
        task2.done.return_value = True
        task3 = MagicMock()
        task3.done.return_value = False
        
        mock_all_tasks.return_value = [current_task, task1, task2, task3]
        
        # Make gather return a coroutine
        async def mock_gather_coro(*args, **kwargs):
            return None
        mock_gather.side_effect = mock_gather_coro
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Test method
        await client.close()
        
        # Verify task cancellation
        task1.cancel.assert_called_once()
        task2.cancel.assert_not_called()  # Already done
        task3.cancel.assert_called_once()
        
        # Verify gather was called with the right tasks
        mock_gather.assert_called_once_with(task1, task3, return_exceptions=True)


@pytest.mark.asyncio
async def test_initialize(bedrock_config):
    """Test the initialize method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch('src.clients.bedrock.TokenCounter') as mock_token_counter, \
         patch.object(BedrockClient, 'validate_aws_credentials') as mock_validate_credentials, \
         patch.dict('os.environ', {'AWS_BEDROCK_MODEL_ID': '', 'AWS_ACCESS_KEY_ID': 'test_key'}):
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_token_counter_instance = MagicMock()
        mock_token_counter.return_value = mock_token_counter_instance
        mock_validate_credentials.return_value = True
        
        # Create client
        client = BedrockClient(bedrock_config)
        
        # Test method
        await client.initialize()
        
        # Verify token counter was initialized
        assert client.token_counter == mock_token_counter_instance
        
        # Test with debug=True
        bedrock_config.debug = True
        client = BedrockClient(bedrock_config)
        mock_validate_credentials.reset_mock()
        
        # Test method with debug=True
        await client.initialize()
        
        # Verify validate_aws_credentials was called
        mock_validate_credentials.assert_called_once()


@pytest.mark.asyncio
async def test_generate_component_relationships(bedrock_config):
    """Test the generate_component_relationships method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch.object(BedrockClient, '_invoke_model_with_token_management') as mock_invoke_model, \
         patch.dict('os.environ', {'AWS_BEDROCK_MODEL_ID': ''}):
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock _invoke_model_with_token_management
        mock_invoke_model.return_value = "Test component relationships"
        
        # Create client
        client = BedrockClient(bedrock_config)
        client._find_common_dependencies = MagicMock(return_value="Test dependencies")
        client.project_structure = "Test project structure"
        
        # Create file manifest
        file_manifest = {
            "file1.py": {"summary": "Test summary 1"},
            "file2.py": {"summary": "Test summary 2"}
        }
        
        # Mock MessageManager
        with patch('src.clients.bedrock.MessageManager') as mock_message_manager:
            mock_message_manager.get_component_relationship_messages.return_value = [
                {"role": "system", "content": "System content"},
                {"role": "user", "content": "User content"}
            ]
            
            # Test method
            result = await client.generate_component_relationships(file_manifest)
            
            # Verify result
            assert result == "Test component relationships"
            
            # Verify MessageManager call
            mock_message_manager.get_component_relationship_messages.assert_called_once_with(
                client.project_structure,
                "Test dependencies"
            )
            
            # Verify _invoke_model_with_token_management was called
            mock_invoke_model.assert_called_once_with([
                {"role": "system", "content": "System content"},
                {"role": "user", "content": "User content"}
            ])


@pytest.mark.asyncio
async def test_enhance_documentation(bedrock_config):
    """Test the enhance_documentation method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch.object(BedrockClient, '_invoke_model_with_token_management') as mock_invoke_model, \
         patch.dict('os.environ', {'AWS_BEDROCK_MODEL_ID': ''}):
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock _invoke_model_with_token_management
        mock_invoke_model.return_value = "Enhanced documentation"
        
        # Create client
        client = BedrockClient(bedrock_config)
        client._find_common_dependencies = MagicMock(return_value="Test dependencies")
        client._identify_key_components = MagicMock(return_value="Test components")
        client._fix_markdown_issues = MagicMock(return_value="Fixed enhanced documentation")
        client.prompt_template = MagicMock()
        client.prompt_template.get_template.return_value = "Template content"
        
        # Create file manifest
        file_manifest = {
            "file1.py": {"summary": "Test summary 1"},
            "file2.py": {"summary": "Test summary 2"}
        }
        
        # Test method
        result = await client.enhance_documentation("Existing content", file_manifest, "README")
        
        # Verify result
        assert result == "Fixed enhanced documentation"
        
        # Verify _invoke_model_with_token_management was called
        mock_invoke_model.assert_called_once()
        
        # Verify prompt_template.get_template was called
        client.prompt_template.get_template.assert_called_once_with("enhance_documentation", {
            "doc_type": "README",
            "existing_content": "Existing content"
        })


@pytest.mark.asyncio
async def test_invoke_model_with_token_management(bedrock_config):
    """Test the _invoke_model_with_token_management method."""
    with patch('src.clients.bedrock.boto3.client') as mock_boto3_client, \
         patch('src.clients.bedrock.asyncio.to_thread') as mock_to_thread, \
         patch('src.clients.bedrock.asyncio.wait_for') as mock_wait_for, \
         patch.dict('os.environ', {'AWS_BEDROCK_MODEL_ID': ''}):
        
        # Setup mocks
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            'content': [{'text': 'Test response'}]
        })
        mock_response.get.return_value = mock_body
        
        # Mock wait_for to return the response
        mock_wait_for.return_value = mock_response
        
        # Create client
        client = BedrockClient(bedrock_config)
        client.token_counter = MagicMock()
        client.token_counter.count_tokens.return_value = 100
        client.token_counter.get_token_limit.return_value = 1000
        client._fix_markdown_issues = MagicMock(return_value="Fixed test response")
        
        # Create messages
        messages = [
            {"role": "system", "content": "System content"},
            {"role": "user", "content": "User content"}
        ]
        
        # Test method
        result = await client._invoke_model_with_token_management(messages)
        
        # Verify result
        assert result == "Fixed test response"
        
        # Verify wait_for was called
        mock_wait_for.assert_called_once()
        
        # Test with token limit exceeded
        client.token_counter.count_tokens.return_value = 2000
        client.token_counter.handle_oversized_input = MagicMock(return_value="Truncated content")
        client.token_counter.truncate_text = MagicMock(return_value="Further truncated content")
        
        # Reset mocks
        mock_wait_for.reset_mock()
        
        # Test method again
        result = await client._invoke_model_with_token_management(messages)
        
        # Verify token handling methods were called
        client.token_counter.handle_oversized_input.assert_called_once()
        
        # Verify result
        assert result == "Fixed test response"


if __name__ == '__main__':
    pytest.main()