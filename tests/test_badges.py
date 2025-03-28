import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from src.utils.badges import generate_badges
from src.models.file_info import FileInfo

# Sample file manifest for testing
@pytest.fixture
def sample_file_manifest():
    return {
        "src/main.py": FileInfo(path="src/main.py", language="python"),
        "src/utils/helpers.py": FileInfo(path="src/utils/helpers.py", language="python"),
        "tests/test_main.py": FileInfo(path="tests/test_main.py", language="python"),
        "README.md": FileInfo(path="README.md", language="markdown"),
        "LICENSE": FileInfo(path="LICENSE", language="text"),
        "requirements.txt": FileInfo(path="requirements.txt", language="text"),
    }

@pytest.fixture
def sample_repo_path():
    return Path("/fake/repo/path")

def test_generate_badges_empty_manifest():
    """Test that an empty file manifest returns an empty string."""
    result = generate_badges({}, Path("/fake/path"))
    assert result == ""

def test_generate_badges_with_license(sample_file_manifest, sample_repo_path):
    """Test license badge generation with a mock license file."""
    license_content = "MIT License\n\nCopyright (c) 2023 Test User"
    
    with patch("pathlib.Path.read_text", return_value=license_content):
        with patch("pathlib.Path.exists", return_value=True):
            result = generate_badges(sample_file_manifest, sample_repo_path)
            assert "License: MIT" in result
            assert "badge/License-MIT-yellow" in result

def test_generate_badges_with_missing_license_file(sample_file_manifest, sample_repo_path):
    """Test license badge generation when license file exists in manifest but not on disk."""
    with patch("pathlib.Path.exists", return_value=False):
        result = generate_badges(sample_file_manifest, sample_repo_path)
        assert "License-Custom-blue" in result

def test_generate_badges_with_license_read_error(sample_file_manifest, sample_repo_path):
    """Test license badge generation when there's an error reading the license file."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.read_text", side_effect=Exception("Read error")):
            result = generate_badges(sample_file_manifest, sample_repo_path)
            assert "License-Custom-blue" in result

def test_generate_badges_with_python(sample_file_manifest, sample_repo_path):
    """Test Python badge generation."""
    with patch("pathlib.Path.exists", return_value=False):
        result = generate_badges(sample_file_manifest, sample_repo_path)
        assert "Python" in result
        assert "badge/Python-3776AB" in result

def test_generate_badges_with_pytest(sample_repo_path):
    """Test pytest badge generation."""
    file_manifest = {
        "src/main.py": FileInfo(path="src/main.py", language="python"),
        "tests/test_main.py": FileInfo(path="tests/test_main.py", language="python"),
        "pytest.ini": FileInfo(path="pytest.ini", language="text"),
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        result = generate_badges(file_manifest, sample_repo_path)
        assert "Tests-Pytest" in result
        assert "badge/Tests-Pytest" in result

def test_generate_badges_with_docs(sample_repo_path):
    """Test documentation badge generation."""
    file_manifest = {
        "src/main.py": FileInfo(path="src/main.py", language="python"),
        "docs/index.md": FileInfo(path="docs/index.md", language="markdown"),
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        result = generate_badges(file_manifest, sample_repo_path)
        assert "Documentation" in result
        assert "badge/Documentation-Yes" in result

def test_generate_badges_with_docker(sample_repo_path):
    """Test Docker badge generation."""
    file_manifest = {
        "src/main.py": FileInfo(path="src/main.py", language="python"),
        "Dockerfile": FileInfo(path="Dockerfile", language="dockerfile"),
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        result = generate_badges(file_manifest, sample_repo_path)
        assert "Docker" in result
        assert "badge/Docker-2496ED" in result

def test_generate_badges_with_github_actions(sample_repo_path):
    """Test GitHub Actions badge generation."""
    file_manifest = {
        "src/main.py": FileInfo(path="src/main.py", language="python"),
        ".github/workflows/ci.yml": FileInfo(path=".github/workflows/ci.yml", language="yaml"),
    }
    
    with patch("pathlib.Path.exists", return_value=False):
        result = generate_badges(file_manifest, sample_repo_path)
        assert "CI%2FCD" in result
        assert "GitHub_Actions" in result

def test_generate_badges_custom_style(sample_file_manifest, sample_repo_path):
    """Test custom badge style."""
    with patch("pathlib.Path.exists", return_value=False):
        result = generate_badges(sample_file_manifest, sample_repo_path, badge_style="flat")
        assert "style=flat" in result
        assert "style=for-the-badge" not in result