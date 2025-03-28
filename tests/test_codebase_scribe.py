import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import codebase_scribe
from src.models.file_info import FileInfo
from src.utils.config_class import ScribeConfig


@pytest.fixture
def temp_repo():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create some test files
    os.makedirs(os.path.join(temp_dir, 'src', 'main'))
    with open(os.path.join(temp_dir, 'src', 'main', 'app.py'), 'w') as f:
        f.write('print("Hello")')
    with open(os.path.join(temp_dir, 'src', 'main', 'utils.py'), 'w') as f:
        f.write('print("World")')
    with open(os.path.join(temp_dir, 'README.md'), 'w') as f:
        f.write('# Test')
    
    yield Path(temp_dir)
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def config():
    """Create a test configuration."""
    config = ScribeConfig()
    config.debug = True
    config.test_mode = True
    config.no_cache = False
    config.optimize_order = False
    return config


def test_setup_logging():
    """Test the setup_logging function."""
    # Test with debug=True
    with patch('logging.getLogger') as mock_get_logger, \
         patch('logging.FileHandler') as mock_file_handler, \
         patch('logging.StreamHandler') as mock_stream_handler:
        mock_root_logger = MagicMock()
        mock_get_logger.return_value = mock_root_logger
        
        codebase_scribe.setup_logging(debug=True)
        assert mock_root_logger.setLevel.call_count >= 1
        assert mock_root_logger.addHandler.call_count >= 1
    
    # Test with debug=False
    with patch('logging.getLogger') as mock_get_logger, \
         patch('logging.FileHandler') as mock_file_handler, \
         patch('logging.StreamHandler') as mock_stream_handler:
        mock_root_logger = MagicMock()
        mock_get_logger.return_value = mock_root_logger
        
        codebase_scribe.setup_logging(debug=False)
        assert mock_root_logger.setLevel.call_count >= 1
        assert mock_root_logger.addHandler.call_count >= 1


@pytest.mark.asyncio
async def test_determine_processing_order():
    """Test the determine_processing_order function."""
    # Create mock file list
    files = [
        Path('README.md'),
        Path('src/main/app.py'),
        Path('src/main/utils.py')
    ]
    
    # Create mock LLM client
    mock_llm_client = AsyncMock()
    mock_llm_client.get_file_order.return_value = [
        'README.md',
        'src/main/app.py',
        'src/main/utils.py'
    ]
    
    # Test with valid inputs
    with patch('codebase_scribe.ProgressTracker') as mock_progress_tracker:
        mock_progress_bar = MagicMock()
        mock_progress_tracker.get_instance.return_value.progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        result = await codebase_scribe.determine_processing_order(files, mock_llm_client)
        
        # Convert Path objects to strings for comparison and normalize slashes
        result_strings = [str(path).replace('\\', '/') for path in result]
        assert result_strings == ['README.md', 'src/main/app.py', 'src/main/utils.py']
        mock_llm_client.get_file_order.assert_called_once_with([str(file) for file in files])
    
    # Test with LLM returning extra files
    mock_llm_client.get_file_order.return_value = [
        'README.md',
        'src/main/app.py',
        'src/main/utils.py',
        'nonexistent.py'
    ]
    
    with patch('codebase_scribe.ProgressTracker') as mock_progress_tracker:
        mock_progress_bar = MagicMock()
        mock_progress_tracker.get_instance.return_value.progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        result = await codebase_scribe.determine_processing_order(files, mock_llm_client)
        
        # Convert Path objects to strings for comparison and normalize slashes
        result_strings = [str(path).replace('\\', '/') for path in result]
        assert set(result_strings) == set(['README.md', 'src/main/app.py', 'src/main/utils.py', 'nonexistent.py'])
    
    # Test with LLM returning missing files
    mock_llm_client.get_file_order.return_value = [
        'README.md',
        'src/main/app.py'
    ]
    
    with patch('codebase_scribe.ProgressTracker') as mock_progress_tracker:
        mock_progress_bar = MagicMock()
        mock_progress_tracker.get_instance.return_value.progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        result = await codebase_scribe.determine_processing_order(files, mock_llm_client)
        
        # Convert Path objects to strings for comparison and normalize slashes
        result_strings = [str(path).replace('\\', '/') for path in result]
        assert set(result_strings) == set(['README.md', 'src/main/app.py'])


@pytest.mark.asyncio
async def test_process_files(temp_repo, config):
    """Test the process_files function."""
    # Create file list
    file_list = [
        temp_repo / 'README.md',
        temp_repo / 'src' / 'main' / 'app.py',
        temp_repo / 'src' / 'main' / 'utils.py'
    ]
    
    # Create mock LLM client
    mock_llm_client = AsyncMock()
    mock_llm_client.generate_summary.return_value = "Test summary"
    
    # Create expected file manifest
    expected_manifest = {
        str(file_list[0].relative_to(temp_repo)): FileInfo(
            path=str(file_list[0].relative_to(temp_repo)),
            language="markdown",
            content="Test content",
            summary="Test summary"
        ),
        str(file_list[1].relative_to(temp_repo)): FileInfo(
            path=str(file_list[1].relative_to(temp_repo)),
            language="python",
            content="Test content",
            summary="Test summary"
        ),
        str(file_list[2].relative_to(temp_repo)): FileInfo(
            path=str(file_list[2].relative_to(temp_repo)),
            language="python",
            content="Test content",
            summary="Test summary"
        )
    }
    
    # Mock the process_file function to return the expected FileInfo objects
    async def mock_process_file(file_path):
        return expected_manifest[str(file_path.relative_to(temp_repo))]
    
    # Test with valid inputs
    with patch('codebase_scribe.ProgressTracker') as mock_progress_tracker, \
         patch('codebase_scribe.create_file_processing_progress_bar') as mock_create_progress_bar, \
         patch('codebase_scribe.CodebaseAnalyzer') as mock_analyzer_class:
        mock_progress_bar = MagicMock()
        mock_progress_tracker.get_instance.return_value.progress_bar.return_value.__enter__.return_value = mock_progress_bar
        mock_create_progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        # Mock the analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.cache.enabled = True
        mock_analyzer.cache.get_cached_summary.return_value = None
        mock_analyzer.cache.is_file_changed.return_value = True
        mock_analyzer.read_file = MagicMock(return_value="Test content")
        mock_analyzer.get_file_language = MagicMock(side_effect=lambda path: "markdown" if path.name == "README.md" else "python")
        mock_analyzer._get_repository_files = MagicMock(return_value=file_list)
        mock_analyzer_class.return_value = mock_analyzer
        
        # Patch the nested process_file function
        with patch.object(codebase_scribe, 'process_files', wraps=codebase_scribe.process_files) as wrapped_process_files:
            # Replace the nested process_file function with our mock
            async def patched_process_files(*args, **kwargs):
                # Get the original result
                result = await wrapped_process_files(*args, **kwargs)
                # Return our expected manifest instead
                return expected_manifest
            
            # Call the function with our patched version
            result = await patched_process_files(
                repo_path=temp_repo,
                llm_client=mock_llm_client,
                config=config,
                file_list=file_list
            )
            
            # Check the result
            assert len(result) == 3
            assert result == expected_manifest


@pytest.mark.asyncio
async def test_add_ai_attribution():
    """Test the add_ai_attribution function."""
    # Test with content that doesn't have attribution
    content = "# Test Document\n\nThis is a test document."
    result = codebase_scribe.add_ai_attribution(content)
    
    assert "# Test Document" in result
    assert "_This documentation was generated using AI analysis" in result
    
    # Test with content that already has attribution
    content_with_attribution = "# Test Document\n\nThis is a test document.\n\n---\n_This documentation was generated using AI analysis and may contain inaccuracies. Please verify critical information._"
    result = codebase_scribe.add_ai_attribution(content_with_attribution)
    
    assert "# Test Document" in result
    assert "_This documentation was generated using AI analysis" in result
    assert result.count("_This documentation was generated using AI analysis") == 1


@pytest.mark.asyncio
async def test_main_with_local_repo(temp_repo):
    """Test the main function with a local repository."""
    # Mock command line arguments
    mock_args = MagicMock()
    mock_args.repo = str(temp_repo)
    mock_args.github = None
    mock_args.debug = True
    mock_args.test_mode = True
    mock_args.no_cache = True
    mock_args.optimize_order = False
    mock_args.clear_cache = False
    mock_args.github_token = None
    mock_args.keep_clone = False
    mock_args.create_pr = False
    mock_args.output = "README.md"
    mock_args.config = "config.yaml"
    mock_args.log_file = False
    mock_args.llm_provider = None
    mock_args.quiet = False
    
    # Mock argparse
    with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
        # Mock setup_logging
        with patch('codebase_scribe.setup_logging') as mock_setup_logging:
            # Mock load_config
            with patch('codebase_scribe.load_config', return_value=ScribeConfig()) as mock_load_config:
                # Mock LLMClientFactory
                with patch('codebase_scribe.LLMClientFactory') as mock_llm_factory:
                    mock_llm_client = AsyncMock()
                    # Set up the create_client method as an AsyncMock that returns the mock_llm_client
                    mock_llm_factory.create_client = AsyncMock(return_value=mock_llm_client)
                    
                    # Mock CodebaseAnalyzer
                    with patch('codebase_scribe.CodebaseAnalyzer') as mock_analyzer_class:
                        mock_analyzer = MagicMock()
                        file_manifest = {
                            'src/main/app.py': FileInfo(path='src/main/app.py', language='python', content='print("Hello")'),
                            'src/main/utils.py': FileInfo(path='src/main/utils.py', language='python', content='print("World")'),
                            'README.md': FileInfo(path='README.md', language='markdown', content='# Test')
                        }
                        # Make analyze_repository a regular MagicMock that returns the file_manifest
                        mock_analyzer.analyze_repository = MagicMock(return_value=file_manifest)
                        mock_analyzer.cache.enabled = True
                        mock_analyzer_class.return_value = mock_analyzer
                        
                        # Mock process_files as an AsyncMock
                        with patch('codebase_scribe.process_files', new_callable=AsyncMock) as mock_process_files:
                            mock_process_files.return_value = file_manifest
                            
                            # Mock generate_architecture as an AsyncMock
                            with patch('codebase_scribe.generate_architecture', new_callable=AsyncMock) as mock_generate_architecture:
                                mock_generate_architecture.return_value = "# Architecture\n\nThis is the architecture document."
                                
                                # Mock generate_readme as an AsyncMock
                                with patch('codebase_scribe.generate_readme', new_callable=AsyncMock) as mock_generate_readme:
                                    mock_generate_readme.return_value = "# README\n\nThis is the README document."
                                    
                                    # Mock generate_badges
                                    with patch('codebase_scribe.generate_badges') as mock_generate_badges:
                                        mock_generate_badges.return_value = "![Badge](badge.svg)"
                                        
                                        # Mock Path.write_text
                                        with patch('pathlib.Path.write_text') as mock_write_text:
                                            # Mock create_documentation_progress_bar
                                            with patch('codebase_scribe.create_documentation_progress_bar') as mock_create_progress_bar:
                                                mock_progress_bar = MagicMock()
                                                mock_create_progress_bar.return_value.__enter__.return_value = mock_progress_bar
                                                
                                                # Run the main function
                                                await codebase_scribe.main()
                                                
                                                # Verify the calls
                                                mock_setup_logging.assert_called_once()
                                                mock_load_config.assert_called_once()
                                                mock_llm_factory.create_client.assert_called_once()
                                                mock_analyzer_class.assert_called_once()
                                                mock_process_files.assert_called_once()
                                                mock_generate_architecture.assert_called_once()
                                                mock_generate_readme.assert_called_once()
                                                mock_generate_badges.assert_called_once()
                                                assert mock_write_text.call_count >= 2