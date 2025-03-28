import json
import pytest
from pathlib import Path
from src.clients.llm_utils import (
    format_project_structure,
    find_common_dependencies,
    identify_key_components,
    get_default_order,
    fix_markdown_issues,
    prepare_file_order_data,
    process_file_order_response,
    DEFAULT_MAX_COMPONENTS,
    DEFAULT_VENDOR_PATTERNS
)

@pytest.fixture
def sample_file_manifest():
    """Create a sample file manifest for testing."""
    return {
        "src/main.py": {"content": "print('Hello world')", "file_type": "python", "size": 20, "is_binary": False},
        "src/utils.py": {"content": "def util(): pass", "file_type": "python", "size": 18, "is_binary": False},
        "src/config.json": {"content": '{"key": "value"}', "file_type": "json", "size": 15, "is_binary": False},
        "requirements.txt": {"content": "pytest==7.0.0\nrequests==2.28.1", "file_type": "text", "size": 30, "is_binary": False},
        "package.json": {"content": '{"dependencies": {"react": "^17.0.2", "lodash": "^4.17.21"}}', "file_type": "json", "size": 60, "is_binary": False},
        "dist/bundle.min.js": {"content": "minified js", "file_type": "js", "size": 1000, "is_binary": False},
        "images/logo.png": {"content": "binary data", "file_type": "png", "size": 2000, "is_binary": True}
    }

def test_format_project_structure(sample_file_manifest):
    """Test the format_project_structure function."""
    result = format_project_structure(sample_file_manifest)
    assert "src/" in result
    assert "- main.py" in result
    assert "- utils.py" in result
    assert "- config.json" in result
    assert "- requirements.txt" in result
    assert "- package.json" in result
    assert "dist/" in result
    assert "images/" in result

def test_find_common_dependencies(sample_file_manifest):
    """Test the find_common_dependencies function."""
    result = find_common_dependencies(sample_file_manifest)
    assert "Detected dependencies:" in result
    assert "pytest==7.0.0" in result
    assert "requests==2.28.1" in result
    assert "react@^17.0.2" in result
    assert "lodash@^4.17.21" in result

def test_identify_key_components(sample_file_manifest):
    """Test the identify_key_components function."""
    result = identify_key_components(sample_file_manifest)
    assert "Key components:" in result
    assert "src" in result
    assert "root" in result
    assert "dist" in result
    assert "images" in result

def test_identify_key_components_with_custom_max(sample_file_manifest):
    """Test the identify_key_components function with custom max_components."""
    result = identify_key_components(sample_file_manifest, max_components=2)
    assert "Key components:" in result
    # Count the number of lines (excluding the header)
    component_count = len([line for line in result.split('\n') if line.startswith('- ')])
    assert component_count == 2

def test_get_default_order():
    """Test the get_default_order function."""
    core_files = {
        "src/main.py": {},
        "src/utils.py": {},
        "src/config.json": {},
        "README.md": {}
    }
    resource_files = {
        "dist/bundle.min.js": {},
        "images/logo.png": {}
    }
    
    result = get_default_order(core_files, resource_files)
    
    # Config files should come first
    assert result.index("src/config.json") < result.index("src/main.py")
    assert result.index("src/config.json") < result.index("src/utils.py")
    
    # Resource files should come last
    assert result.index("src/main.py") < result.index("dist/bundle.min.js")
    assert result.index("src/utils.py") < result.index("images/logo.png")

def test_fix_markdown_issues():
    """Test the fix_markdown_issues function."""
    markdown = """#Header without space
##Another header

- List item
  - Subitem with wrong indentation
   - Another subitem with wrong indentation

###Header with wrong level after ##

#####Header with too many levels after ###"""

    fixed = fix_markdown_issues(markdown)
    
    # Check space after # in headers
    assert "# Header without space" in fixed
    assert "## Another header" in fixed
    
    # Check header levels
    assert "### Header with wrong level after ##" in fixed  # This is fine
    assert "#### Header with too many levels after ###" in fixed  # Should be adjusted to level 4
    
    # Check list indentation
    assert "  - Subitem with wrong indentation" in fixed
    assert "  - Another subitem with wrong indentation" in fixed  # Should be adjusted to 2 spaces

def test_prepare_file_order_data(sample_file_manifest):
    """Test the prepare_file_order_data function."""
    core_files, resource_files, files_info = prepare_file_order_data(sample_file_manifest)
    
    # Check core files (non-vendor files)
    assert "src/main.py" in core_files
    assert "src/utils.py" in core_files
    assert "src/config.json" in core_files
    assert "requirements.txt" in core_files
    assert "package.json" in core_files
    
    # Check resource files (vendor files)
    assert "dist/bundle.min.js" in resource_files
    assert "images/logo.png" in resource_files
    
    # Check files_info
    assert files_info["src/main.py"]["type"] == "python"
    assert files_info["src/utils.py"]["size"] == 18
    assert not files_info["src/config.json"]["is_binary"]

def test_process_file_order_response_json():
    """Test the process_file_order_response function with JSON response."""
    content = json.dumps({
        "file_order": ["file1.py", "file2.py", "file3.py"],
        "reasoning": "Ordered by dependency"
    })
    core_files = {"file1.py": {}, "file2.py": {}, "file3.py": {}, "file4.py": {}}
    resource_files = {"vendor.js": {}}
    
    result = process_file_order_response(content, core_files, resource_files)
    
    assert result == ["file1.py", "file2.py", "file3.py", "vendor.js"]

def test_process_file_order_response_text():
    """Test the process_file_order_response function with text response."""
    content = """
    First, we should include file2.py because it contains base classes.
    Then, file1.py which depends on file2.py.
    Finally, file3.py which uses both.
    """
    core_files = {"file1.py": {}, "file2.py": {}, "file3.py": {}, "file4.py": {}}
    resource_files = {"vendor.js": {}}
    
    result = process_file_order_response(content, core_files, resource_files)
    
    # The exact order might vary depending on how the text is processed,
    # but all files mentioned in the text should be included
    assert "file1.py" in result
    assert "file2.py" in result
    assert "file3.py" in result
    assert "vendor.js" in result

def test_process_file_order_response_invalid():
    """Test the process_file_order_response function with invalid response."""
    content = "This response doesn't contain any file paths or valid JSON."
    core_files = {"file1.py": {}, "file2.py": {}, "file3.py": {}}
    resource_files = {"vendor.js": {}}
    
    result = process_file_order_response(content, core_files, resource_files)
    
    # Should fall back to default order
    assert set(result) == {"file1.py", "file2.py", "file3.py", "vendor.js"}
    # Config files should come first if any
    for file in result:
        if file.endswith(('.json', '.config', '.settings')):
            assert result.index(file) < result.index("file1.py")