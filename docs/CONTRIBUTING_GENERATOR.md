# Contributing Generator

This document describes the Contributing guide generator component of the Codebase Scribe AI tool.

## Overview

The Contributing generator is responsible for creating a standalone `CONTRIBUTING.md` file in the root directory of the repository. This file provides guidelines for contributing to the project, including code of conduct, development process, pull request process, coding standards, and more.

The generator follows GitHub's convention where `CONTRIBUTING.md` files in the root directory are automatically recognized and linked when users create new issues or pull requests.

## Key Features

- Generates a comprehensive Contributing guide as a standalone file
- Places the file in the root directory for GitHub recognition
- Includes sections for code of conduct, development process, PR process, etc.
- Validates and improves content for readability and markdown compliance
- Handles existing Contributing guides with enhancement capabilities
- Provides fallback mechanisms for error handling

## Implementation

The Contributing generator is implemented in `src/generators/contributing.py` and follows a similar pattern to the README generator:

```python
async def generate_contributing(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    config: ScribeConfig,
    analyzer: CodebaseAnalyzer
) -> str:
    """
    Generate CONTRIBUTING.md content.
    
    Args:
        repo_path: Path to the repository root
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        config: Configuration
        analyzer: CodebaseAnalyzer instance
        
    Returns:
        str: Generated CONTRIBUTING.md content
    """
    # Implementation details...
```

## Integration with README

The README generator has been updated to link to the Contributing guide instead of including its content directly. This is done by replacing the Contributing section in the README with a link to the `CONTRIBUTING.md` file:

```markdown
## Contributing

For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
```

## Generation Process

The Contributing guide is generated before the README to ensure the file exists before the README links to it. The generation process follows these steps:

1. Check if an existing Contributing guide should be enhanced
2. If yes, enhance the existing guide
3. If no, generate a new Contributing guide
4. Validate and improve the content
5. Write the Contributing guide to the root directory

## Configuration

The Contributing generator uses the same configuration system as the other generators. It respects the following configuration options:

- `preserve_existing`: Whether to enhance existing Contributing guides or replace them
- `debug`: Whether to enable debug logging

## Error Handling

The Contributing generator includes robust error handling and fallback mechanisms:

- If the LLM fails to generate content, a fallback Contributing guide is used
- If the content is too short or invalid, a fallback is used
- If enhancement fails, a new Contributing guide is generated

## Testing

The Contributing generator includes comprehensive unit tests in `tests/test_contributing.py` that cover:

- Checking if an existing Contributing guide should be enhanced
- Enhancing an existing Contributing guide
- Generating Contributing guide content
- Ensuring correct title formatting
- Validating and improving content
- Generating fallback Contributing guide
- Generating a new Contributing guide
- Main Contributing guide generation function

## Future Improvements

Potential future improvements for the Contributing generator include:

- Customizable templates for different project types
- More detailed sections based on project needs
- Integration with code of conduct generation
- Support for multiple languages