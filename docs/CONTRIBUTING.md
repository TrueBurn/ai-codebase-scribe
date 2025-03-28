# Contributing Guide

Thank you for your interest in contributing to the AI README Generator! This document provides guidelines and workflows for contributing to the project.

> **Note**: This file serves as documentation for contributing to the project. When using the Codebase Scribe AI tool, a separate `CONTRIBUTING.md` file is generated in the root directory of the target repository. This follows GitHub's convention where `CONTRIBUTING.md` files in the root directory are automatically recognized and linked when users create new issues or pull requests. For details on how the Contributing guide is generated, see [Contributing Generator](CONTRIBUTING_GENERATOR.md).

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge
We are committed to making participation in our project a harassment-free experience for everyone, regardless of level of experience, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Our Standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/readme-generator.git
cd readme-generator
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-fix-name
```

## Development Process

### 1. Before Starting
- Check existing issues and PRs
- Discuss major changes in an issue first
- Read the [Development Guide](DEVELOPMENT.md)

### 2. During Development
- Write tests for new features
- Keep changes focused and small
- Follow coding standards
- Update documentation

### 3. Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run style checks
black src tests
flake8 src tests
mypy src
```

## Pull Request Process

### 1. Preparation
- Update documentation
- Add tests for new features
- Update CHANGELOG.md
- Ensure all tests pass
- Check code style compliance

### 2. Submitting
- Fill in the PR template completely
- Link related issues
- Provide clear description
- Include screenshots if relevant

### 3. Review Process
- Address review comments
- Keep discussions focused
- Be responsive to feedback
- Update PR as needed

### 4. After Merge
- Delete your branch
- Update your fork
- Close related issues

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters
- Use descriptive variable names

### Example
```python
from typing import List, Optional

def process_items(items: List[str], limit: Optional[int] = None) -> List[str]:
    """Process a list of items and return filtered results.
    
    Args:
        items: List of items to process.
        limit: Optional maximum number of items to return.
        
    Returns:
        Filtered and processed items.
    """
    processed = [item.strip().lower() for item in items]
    return processed[:limit] if limit else processed
```

### Documentation
- Use Google-style docstrings
- Document all public functions/classes
- Include usage examples
- Keep comments current

## Commit Guidelines

### Commit Messages
Follow conventional commits specification:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Example:
```
feat(cache): add TTL support to memory cache

- Implement time-based cache invalidation
- Add configuration options for TTL
- Update documentation with TTL examples

Closes #123
```

## Documentation

### Required Documentation
- Update README.md for user-facing changes
- Update API documentation for interface changes
- Add/update examples for new features
- Include docstrings for new code

### Documentation Style
- Use clear, concise language
- Include code examples
- Add screenshots when helpful
- Keep formatting consistent

## Community

### Getting Help
- Check existing documentation
- Search closed issues
- Ask in discussions
- Be clear and specific

### Reporting Issues
- Use issue templates
- Include reproduction steps
- Provide system information
- Add relevant logs

### Feature Requests
- Check existing issues/PRs
- Provide clear use cases
- Explain the benefit
- Consider implementation

### Communication
- Be respectful
- Stay on topic
- Help others when possible
- Share knowledge

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to AI README Generator! Your efforts help make this project better for everyone. 