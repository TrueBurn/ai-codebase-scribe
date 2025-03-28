# Badges System

This document provides detailed information about the badge generation system in the Codebase Scribe AI project.

## Overview

The badges system automatically generates appropriate markdown badges for a repository based on its content. These badges provide visual indicators of various aspects of the project, such as:

- License type
- CI/CD tools
- Testing frameworks
- Documentation
- Docker usage
- Programming languages and frameworks
- Databases

## Badge Generation

The badge generation is handled by the `generate_badges` function in `src/utils/badges.py`. This function analyzes the repository content and generates markdown badges that can be included in README.md and other documentation files.

### Function Signature

```python
def generate_badges(file_manifest: Dict[str, FileInfo], repo_path: Path, badge_style: str = "for-the-badge") -> str:
    """Generate appropriate badges based on repository content.
    
    This function analyzes the repository content to generate markdown badges for:
    - License type
    - CI/CD tools
    - Testing frameworks
    - Documentation
    - Docker usage
    - Programming languages and frameworks
    - Databases
    
    Args:
        file_manifest: Dictionary of files in the repository
        repo_path: Path to the repository
        badge_style: Style of badges to generate (default: for-the-badge)
        
    Returns:
        str: Space-separated string of markdown badges
        
    Raises:
        ValueError: If file_manifest is None or empty
    """
```

### Badge Styles

The badge generation system supports various badge styles from shields.io:

- `flat`: Flat badges with no shadows
- `flat-square`: Flat square badges with no rounded corners
- `plastic`: Plastic badges with shadows
- `for-the-badge`: Larger badges with all caps text (default)
- `social`: Social style badges

### Badge Format

Badges are generated using the shields.io service with the following format:

```
![Label](https://img.shields.io/badge/Label-Message-Color?style=Style&logo=LogoName&logoColor=LogoColor)
```

For example:
```
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
```

## Detection Logic

The badge generation system uses various detection strategies to identify project characteristics:

### License Detection

The system looks for license files in the repository and analyzes their content to determine the license type. Supported license types include:

- MIT
- Apache 2.0
- GPL v3
- BSD
- Mozilla Public License
- ISC
- Unlicense

If no license file is found or the license type cannot be determined, a generic "Custom" license badge is generated.

### CI/CD Detection

The system detects CI/CD configuration files to determine which CI/CD system is used. Supported CI/CD systems include:

- GitHub Actions
- Azure Pipelines
- Travis CI
- Jenkins
- GitLab CI
- CircleCI
- Bitbucket Pipelines

### Testing Framework Detection

The system detects testing framework files and directories to determine which testing frameworks are used. Supported testing frameworks include:

- JUnit
- NUnit
- Jest
- Mocha
- Pytest
- RSpec
- Cypress
- Playwright
- TestNG
- PHPUnit
- Jasmine
- Selenium
- Vitest

### Documentation Detection

The system checks for documentation files and directories, including:

- docs/
- documentation/
- wiki/
- README.md
- CONTRIBUTING.md
- ARCHITECTURE.md

### Docker Detection

The system checks for Docker-related files, including:

- Dockerfile
- docker-compose.yml
- .dockerignore
- docker/

### Language and Framework Detection

The system detects programming languages and frameworks based on file extensions and specific files. Supported languages and frameworks include:

- Java (Spring, Spring Boot)
- C# (.NET, ASP.NET)
- JavaScript (React, Angular, Vue.js)
- TypeScript
- Python (Django, Flask, FastAPI)
- Ruby (Rails)
- PHP (Laravel)
- Go
- Rust
- Swift
- Kotlin
- Dart

### Database Detection

The system detects database technologies based on configuration files and imports. Supported databases include:

- MongoDB
- MySQL
- PostgreSQL
- Redis

## Usage

The badge generation is automatically integrated into the documentation generation process. When generating README.md and ARCHITECTURE.md files, the system will:

1. Analyze the repository content
2. Generate appropriate badges
3. Insert the badges after the title in the generated documentation

### Manual Usage

You can also use the badge generation system manually in your code:

```python
from src.utils.badges import generate_badges
from src.models.file_info import FileInfo
from pathlib import Path

# Create a file manifest
file_manifest = {
    "src/main.py": FileInfo(path="src/main.py", language="python"),
    "LICENSE": FileInfo(path="LICENSE", language="text"),
    "README.md": FileInfo(path="README.md", language="markdown"),
    "Dockerfile": FileInfo(path="Dockerfile", language="dockerfile"),
    ".github/workflows/ci.yml": FileInfo(path=".github/workflows/ci.yml", language="yaml"),
}

# Generate badges
repo_path = Path("/path/to/repo")
badges = generate_badges(file_manifest, repo_path)

# Use custom badge style
badges = generate_badges(file_manifest, repo_path, badge_style="flat")

# Add badges to README
readme_content = "# My Project\n\n"
readme_content = readme_content[:readme_content.find("\n")+1] + "\n" + badges + "\n" + readme_content[readme_content.find("\n")+1:]
```

## Testing

The badge generation system includes comprehensive tests in `tests/test_badges.py`. These tests cover:

- Empty file manifest handling
- License detection with various scenarios
- Programming language detection
- Testing framework detection
- Documentation detection
- Docker detection
- CI/CD system detection
- Custom badge style support

## Customization

The badge generation system is designed to be easily extensible. To add support for new languages, frameworks, or tools:

1. Update the appropriate detection dictionaries in `src/utils/badges.py`
2. Add appropriate test cases in `tests/test_badges.py`
3. Update this documentation to reflect the new capabilities

## Future Enhancements

Planned enhancements for the badge generation system include:

- Support for more languages and frameworks
- Support for more CI/CD systems
- Support for more testing frameworks
- Support for more databases
- Support for custom badge templates
- Support for custom badge colors
- Support for custom badge logos