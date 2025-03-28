# Architecture Generator Documentation

This document provides detailed information about the architecture documentation generation system in the codebase-scribe-ai project.

## Overview

The Architecture Generator is responsible for creating comprehensive architecture documentation for a codebase. It analyzes the repository structure, uses an LLM to generate content, and formats the output with proper markdown structure, including Mermaid diagrams for visual representation.

## Implementation

The Architecture Generator is implemented in `src/generators/architecture.py` and follows a modular design:

```
src/generators/architecture.py
```

## Key Components

### Main Function

```python
async def generate_architecture(
    repo_path: Path,
    file_manifest: dict,
    llm_client: BaseLLMClient,
    config: dict
) -> str:
    """
    Generate architecture documentation for the repository.
    
    This function uses an LLM to generate comprehensive architecture documentation
    with proper formatting, table of contents, and sections. If the LLM fails or
    returns invalid content, it falls back to a basic architecture document.
    """
```

This is the main function for architecture documentation generation. It takes a repository path, file manifest, LLM client, and configuration parameters.

### Supporting Functions

The module provides several supporting functions:

1. **create_fallback_architecture**: Creates a basic architecture document when LLM generation fails
2. **analyze_basic_structure**: Analyzes the file manifest to determine technology stack and project patterns
3. **_format_section_anchor**: Helper function to format section names into valid markdown anchors

## Configuration Options

The Architecture Generator supports several configuration options through constants:

```python
# Constants for configuration
DEFAULT_DIAGRAM_DIRECTION = "TB"  # Top-to-bottom for better package visualization
DEPENDENCY_DIAGRAM_DIRECTION = "LR"  # Left-to-right for dependency flowcharts
MAX_TREE_LINES = 100  # Maximum number of tree lines to display
MAX_COMPONENT_NAMES = 5  # Maximum number of component names to extract
MIN_CONTENT_LENGTH = 100  # Minimum length for valid architecture content
```

## Integration with MermaidGenerator

The Architecture Generator now uses a hybrid approach that combines LLM-generated content with programmatically generated Mermaid diagrams:

```python
# Build a dependency graph from the file manifest
dependency_graph = build_dependency_graph_from_manifest(file_manifest)

# Create MermaidGenerator with the dependency graph
mermaid_gen = MermaidGenerator(dependency_graph)

# Generate a dependency flowchart
diagram = mermaid_gen.generate_dependency_flowchart()

# Insert the diagram into the LLM-generated content
if diagram and "```mermaid" in diagram:
    # Add the diagram before the first section or at the end if no sections
    if "## " in architecture_content:
        parts = architecture_content.split("## ", 1)
        architecture_content = parts[0] + "\n" + diagram + "\n\n## " + parts[1]
    else:
        architecture_content += "\n\n" + diagram
```

This approach ensures that even if the LLM doesn't generate diagrams, the architecture documentation will still include visual representations of the codebase structure.

### Path Compression

The Architecture Generator now includes a path compression system that reduces token usage when sending file paths to the LLM:

```python
# Apply path compression to reduce token usage
compressed_paths, decompression_map = compress_paths(file_paths)

# Generate the compressed project structure
project_structure = "\n".join([f"- {path}" for path in compressed_paths])

# Add explanation of compression scheme
compression_explanation = get_compression_explanation(decompression_map)
project_structure = compression_explanation + "\n\n" + project_structure
```

This is particularly useful for Java projects with deep package structures, where file paths can consume a significant portion of the token budget.

### Dependency Graph Generation

The Architecture Generator now includes a sophisticated dependency graph generation system that:

1. Analyzes the file manifest to identify components
2. Creates relationships between components based on:
   - File imports and exports
   - Package structure (for Java projects)
   - Common architectural patterns

For Java projects, it specifically:
- Identifies components based on package naming conventions
- Creates relationships based on common Java architectural patterns (e.g., controllers depend on services)
- Handles deep package structures common in Java projects

This results in more meaningful and accurate component diagrams, especially for Java projects.

### Tree View Generation

The Architecture Generator now includes enhanced tree view generation for project structures using box-drawing characters that clearly show folder hierarchy:

```python
# Import the tree formatter
from ..utils.tree_formatter import format_tree_structure

# Generate a proper tree structure with clear hierarchy
tree_content = format_tree_structure(file_manifest)
```

This provides a more visually clear representation of the project structure in the architecture documentation:

```
├── .gitignore
├── Dockerfile
├── README.md
├── pom.xml
├── .github/
│   └── workflows/
│       ├── ai-code-review.yml
│       └── build-deploy.yaml
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── example/
│   │               ├── Application.java
│   │               ├── controller/
│   │               │   └── ApiController.java
│   │               └── service/
│   │                   └── ApiService.java
│   └── test/
│       └── java/
│           └── com/
│               └── example/
│                   └── ApplicationTest.java
```

The tree formatter is implemented in `src/utils/tree_formatter.py` and provides functions to format project structures with proper indentation and connecting lines, making it much easier to understand the folder depth and relationships between files and directories.

## Error Handling

The Architecture Generator implements comprehensive error handling:

- Try/except blocks around LLM calls
- Fallback mechanism for when LLM generation fails
- Validation of LLM-generated content
- Logging of errors and warnings

## Testing

The Architecture Generator has comprehensive tests in `tests/test_architecture.py` that cover:

- Successful LLM response handling
- LLM failure handling
- Short content handling
- Debug mode behavior
- Fallback architecture generation
- Basic structure analysis

## Usage

The Architecture Generator is typically used in the main application flow:

```python
# Generate architecture documentation
arch_content = await generate_architecture(
    repo_path=repo_path,
    file_manifest=processed_files,
    llm_client=llm_client,
    config=config
)

# Write to file
arch_path = repo_path / "docs" / "ARCHITECTURE.md"
arch_path.write_text(arch_content, encoding='utf-8')
```

## Extending

To extend the Architecture Generator:

1. Add new analysis functions for different aspects of the codebase
2. Enhance the fallback mechanism with more detailed analysis
3. Improve the LLM prompting for better architecture content
4. Add new configuration options for customization