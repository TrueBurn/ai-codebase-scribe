import pytest
from pathlib import Path
import tempfile
import yaml
import os
from src.utils.prompt_manager import PromptTemplate

@pytest.fixture
def temp_config_file():
    """Create a temporary config file with custom templates."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'templates': {
                'custom_template': 'This is a {custom_value} template',
                'versioned_template': {
                    'version': '1.2.3',
                    'content': 'This is a versioned {template}'
                }
            }
        }, f)
        config_path = f.name
    
    yield Path(config_path)
    
    # Clean up
    os.unlink(config_path)

def test_default_templates():
    """Test that default templates are loaded correctly."""
    template_manager = PromptTemplate()
    
    # Check that default templates exist
    assert 'file_summary' in template_manager.templates
    assert 'project_overview' in template_manager.templates
    assert 'enhance_documentation' in template_manager.templates
    
    # Check template versions
    assert template_manager.template_versions['file_summary'] == PromptTemplate.VERSION

def test_custom_templates(temp_config_file):
    """Test loading custom templates from config file."""
    template_manager = PromptTemplate(temp_config_file)
    
    # Check that custom templates were loaded
    assert 'custom_template' in template_manager.templates
    assert template_manager.templates['custom_template'] == 'This is a {custom_value} template'
    
    # Check versioned template
    assert 'versioned_template' in template_manager.templates
    assert template_manager.templates['versioned_template'] == 'This is a versioned {template}'
    assert template_manager.template_versions['versioned_template'] == '1.2.3'

def test_get_template():
    """Test retrieving templates with and without context."""
    template_manager = PromptTemplate()
    
    # Get template without context
    template = template_manager.get_template('file_summary')
    assert '{file_path}' in template
    assert '{code}' in template
    
    # Get template with context
    context = {
        'file_path': 'test.py',
        'file_type': 'Python',
        'code': 'print("Hello, world!")',
        'imports': ['os', 'sys'],
        'exports': ['main'],
        'dependencies': ['pytest']
    }
    
    formatted = template_manager.get_template('file_summary', context)
    assert 'test.py' in formatted
    assert 'Python' in formatted
    assert 'print("Hello, world!")' in formatted
    assert '- os' in formatted  # List formatting

def test_missing_template():
    """Test behavior when requesting a non-existent template."""
    template_manager = PromptTemplate()
    
    # Request non-existent template
    template = template_manager.get_template('non_existent_template')
    assert 'Please analyze the following content:' in template

def test_template_validation():
    """Test template validation functionality."""
    template_manager = PromptTemplate()
    
    # Test validation of existing template
    missing = template_manager._validate_template(
        "Template with {placeholder1} and {placeholder2}",
        ["placeholder1", "placeholder3"]
    )
    assert "placeholder3" in missing
    assert "placeholder1" not in missing
    
    # Add invalid template and check validation
    template_manager.add_template(
        'file_summary_invalid', 
        'Invalid template missing required placeholders',
        '1.0.0'
    )
    
    # Get template info to see validation results
    info = template_manager.get_template_info()
    assert 'file_summary_invalid' in info

def test_prepare_context():
    """Test context preparation with different data types."""
    template_manager = PromptTemplate()
    
    # Test with various data types
    context = {
        'list_value': ['item1', 'item2', 'item3'],
        'dict_value': {'key1': 'value1', 'key2': 'value2'},
        'string_value': 'simple string',
        'number_value': 42
    }
    
    prepared = template_manager._prepare_context(context)
    
    # Check list formatting
    assert prepared['list_value'].startswith('- item1')
    assert '- item2' in prepared['list_value']
    assert '- item3' in prepared['list_value']
    
    # Check dict formatting
    assert '- key1: value1' in prepared['dict_value']
    assert '- key2: value2' in prepared['dict_value']
    
    # Check primitive values
    assert prepared['string_value'] == 'simple string'
    assert prepared['number_value'] == 42
    
    # Check defaults were added
    assert 'file_path' in prepared
    assert prepared['file_path'] == ''

def test_add_template():
    """Test adding new templates."""
    template_manager = PromptTemplate()
    
    # Add a new template
    result = template_manager.add_template(
        'new_template',
        'This is a {new} template with {placeholders}',
        '1.0.0'
    )
    
    assert result is True
    assert 'new_template' in template_manager.templates
    assert template_manager.template_versions['new_template'] == '1.0.0'
    
    # Test the new template
    formatted = template_manager.get_template(
        'new_template', 
        {'new': 'brand new', 'placeholders': 'multiple values'}
    )
    
    assert formatted == 'This is a brand new template with multiple values'

def test_get_template_info():
    """Test retrieving template information."""
    template_manager = PromptTemplate()
    
    info = template_manager.get_template_info()
    
    # Check that all default templates are included
    assert 'file_summary' in info
    assert 'project_overview' in info
    assert 'enhance_documentation' in info
    
    # Check info structure
    assert 'version' in info['file_summary']
    assert 'description' in info['file_summary']
    assert 'required_placeholders' in info['file_summary']
    assert 'all_placeholders' in info['file_summary']
    assert 'length' in info['file_summary']