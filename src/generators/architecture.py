# Standard library imports
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx

# Local imports
from ..analyzers.codebase import CodebaseAnalyzer
from ..clients.base_llm import BaseLLMClient
from ..utils.markdown_validator import MarkdownValidator
from ..utils.config_class import ScribeConfig
from ..utils.path_compression import compress_paths, get_compression_explanation
from ..utils.tree_formatter import format_tree_structure, _format_tree_node
from .mermaid import MermaidGenerator
from .readme import _format_anchor_link

# Constants for configuration
DEFAULT_DIAGRAM_DIRECTION = "TB"  # Top-to-bottom for better package visualization
DEPENDENCY_DIAGRAM_DIRECTION = "LR"  # Left-to-right for dependency flowcharts
MAX_TREE_LINES = 100  # Maximum number of tree lines to display
MAX_COMPONENT_NAMES = 5  # Maximum number of component names to extract
MIN_CONTENT_LENGTH = 100  # Minimum length for valid architecture content

async def generate_architecture(
    repo_path: Path,
    file_manifest: dict,
    llm_client: BaseLLMClient,
    config: ScribeConfig
) -> str:
    """
    Generate architecture documentation for the repository.
    
    This function uses an LLM to generate comprehensive architecture documentation
    with proper formatting, table of contents, and sections. If the LLM fails or
    returns invalid content, it falls back to a basic architecture document.
    
    Args:
        repo_path: Path to the repository
        file_manifest: Dictionary of files in the repository
        llm_client: LLM client for generating content
        config: Configuration
        
    Returns:
        Formatted architecture documentation as a string
    """
    try:
        # Create a temporary analyzer to use its method
        temp_analyzer = CodebaseAnalyzer(repo_path, config)
        temp_analyzer.file_manifest = file_manifest
        
        # Get debug mode
        debug_mode = config.debug
        project_name = temp_analyzer.derive_project_name(debug_mode)
        
        # Set up logging
        if debug_mode:
            logging.info(f"Generating architecture documentation for {project_name}")
        
        # Generate architecture content using LLM
        try:
            logging.info("Calling LLM to generate architecture documentation...")
            architecture_content = await llm_client.generate_architecture_doc(file_manifest)
            
            # Log the response for debugging
            if debug_mode:
                content_preview = architecture_content[:200] if architecture_content else "None"
                logging.info(f"LLM response preview: {content_preview}...")
                logging.info(f"Generated architecture content length: {len(architecture_content) if architecture_content else 0}")
        except Exception as e:
            logging.error(f"Error in LLM architecture generation: {str(e)}")
            logging.error(f"Exception details: {traceback.format_exc()}")
            # Create a more detailed fallback with project structure
            return create_fallback_architecture(project_name, file_manifest)
        
        # If we got a valid response, format it properly
        if architecture_content and len(architecture_content) > MIN_CONTENT_LENGTH:
            # Log successful generation
            logging.info("Successfully received architecture content from LLM")
            
            # Add mermaid diagram if MermaidGenerator is available
            try:
                # Build a dependency graph from the file manifest
                dependency_graph = build_dependency_graph_from_manifest(file_manifest)
                mermaid_gen = MermaidGenerator(dependency_graph)
                diagram = mermaid_gen.generate_dependency_flowchart()
                if diagram and "```mermaid" in diagram:
                    # Add the diagram before the first section or at the end if no sections
                    if "## " in architecture_content:
                        parts = architecture_content.split("## ", 1)
                        architecture_content = parts[0] + "\n" + diagram + "\n\n## " + parts[1]
                    else:
                        architecture_content += "\n\n" + diagram
                    
                    # Replace flat project structure with tree view if present
                    if ("## Project Structure" in architecture_content or "## 2. Project Structure" in architecture_content) and "```" in architecture_content:
                        # Replace the flat structure with improved tree view
                        pattern = r'(## (?:\d+\. )?Project Structure\s*\n+```[^\n]*\n)(.*?)(```)'
                        
                        # Generate a proper tree structure with clear hierarchy
                        tree_content = format_tree_structure(file_manifest)
                        
                        # Truncate if too long
                        tree_lines = tree_content.split('\n')
                        if len(tree_lines) > MAX_TREE_LINES:
                            tree_content = '\n'.join(tree_lines[:MAX_TREE_LINES])
                            tree_content += "\n... (truncated)"
                        
                        replacement = r'\1' + tree_content + r'\n```'
                        architecture_content = re.sub(pattern, replacement, architecture_content, flags=re.DOTALL)
            except Exception as e:
                logging.warning(f"Failed to generate mermaid diagram: {e}")
            
            # Extract sections to build table of contents
            sections = []
            for line in architecture_content.split('\n'):
                if line.startswith('## '):
                    section_name = line[3:].strip()
                    section_anchor = _format_anchor_link(section_name)
                    sections.append(f"- [{section_name}](#{section_anchor})")
            
            # If no sections were found, add a default one
            if not sections:
                sections = ["- [Overview](#overview)"]
                # Add an overview section if it doesn't exist
                if "## Overview" not in architecture_content:
                    architecture_content += "\n\n## Overview\n\nThis section provides an overview of the project architecture."
            
            # Create table of contents
            toc = "## Table of Contents\n" + "\n".join(sections) + "\n\n"
            
            # Ensure the document has a proper title
            if not architecture_content.strip().startswith('# '):
                architecture_content = f"# Project Architecture Analysis: {project_name}\n\n{toc}{architecture_content}"
            else:
                # Replace any existing title with our properly formatted one
                architecture_content = re.sub(
                    r'^#\s+.*$',
                    f"# Project Architecture Analysis: {project_name}",
                    architecture_content,
                    count=1,
                    flags=re.MULTILINE
                )
                
                # Check if there's already a table of contents
                if "## Table of Contents" not in architecture_content:
                    # Find the position after the title to insert TOC
                    lines = architecture_content.split('\n')
                    title_index = next((i for i, line in enumerate(lines) if line.startswith('# ')), 0)
                    
                    # Insert TOC after the title and a blank line
                    if title_index + 1 < len(lines) and lines[title_index + 1].strip() == '':
                        lines.insert(title_index + 2, toc)
                    else:
                        lines.insert(title_index + 1, '\n' + toc)
                    
                    architecture_content = '\n'.join(lines)
            
            if debug_mode:
                logging.info("Successfully formatted architecture content with TOC")
            return architecture_content
        else:
            # Fallback for invalid or empty content
            logging.warning(f"Architecture content too short or invalid: {architecture_content[:100]}...")
            return create_fallback_architecture(project_name, file_manifest)
    except Exception as e:
        logging.error(f"Error generating architecture documentation: {e}")
        return create_fallback_architecture("Project", file_manifest)

# Removed unused functions - these are now handled by the tree_formatter module
def create_fallback_architecture(project_name: str, file_manifest: dict) -> str:
    """
    Create a fallback architecture document with basic project structure.
    
    This function is used when the LLM fails to generate architecture documentation
    or returns invalid content. It creates a basic document with project structure,
    technology stack, and other information that can be derived from the file manifest.
    
    Args:
        project_name: Name of the project
        file_manifest: Dictionary of files in the repository
        
    Returns:
        Basic architecture documentation as a string
    """
    # Format project structure with improved tree visualization
    project_structure = format_tree_structure(file_manifest)
    
    # Create a basic structure analysis
    structure_sections = analyze_basic_structure(file_manifest)
    
    # Add the project structure to the sections
    structure_sections["Project Structure"] = f"```\n{project_structure}\n```"
    
    # Build the document
    content = f"# Project Architecture Analysis: {project_name}\n\n"
    content += "## Table of Contents\n"
    content += "- [Overview](#overview)\n"
    content += "- [Project Structure](#project-structure)\n"
    
    # Add more sections if available
    if structure_sections:
        for section in structure_sections.keys():
            if section != "Overview" and section != "Project Structure":
                # Use the _format_anchor_link function from readme.py
                anchor = _format_anchor_link(section)
                content += f"- [{section}](#{anchor})\n"
    
    content += "\n## Overview\n\n"
    content += "This document provides a basic analysis of the project architecture. "
    content += "A more detailed analysis could not be generated automatically.\n\n"
    
    content += "## Project Structure\n\n"
    # Use the project structure we already generated with uncompressed paths
    content += structure_sections["Project Structure"] + "\n\n"
    
    # Add any additional structure sections
    if structure_sections:
        for section, text in structure_sections.items():
            if section != "Overview" and section != "Project Structure":
                content += f"## {section}\n\n{text}\n\n"
    
    return content

def analyze_basic_structure(file_manifest: dict) -> Dict[str, str]:
    """
    Perform basic structure analysis without LLM.
    
    This function analyzes the file manifest to determine the technology stack
    and project patterns based on file extensions and directory names.
    
    Args:
        file_manifest: Dictionary of files in the repository
        
    Returns:
        Dictionary of section names to section content
    """
    sections = {}
    
    # Count file types
    file_types = {}
    for path in file_manifest.keys():
        path_str = str(path)
        ext = path_str.split('.')[-1] if '.' in path_str else 'unknown'
        file_types[ext] = file_types.get(ext, 0) + 1
    
    # Create technology stack section based on file extensions
    tech_stack = "Based on file extensions, this project appears to use:\n\n"
    if file_types.get('py', 0) > 0:
        tech_stack += "- Python\n"
    if file_types.get('js', 0) > 0:
        tech_stack += "- JavaScript\n"
    if file_types.get('jsx', 0) > 0 or file_types.get('tsx', 0) > 0:
        tech_stack += "- React\n"
    if file_types.get('ts', 0) > 0:
        tech_stack += "- TypeScript\n"
    if file_types.get('html', 0) > 0:
        tech_stack += "- HTML\n"
    if file_types.get('css', 0) > 0:
        tech_stack += "- CSS\n"
    if file_types.get('java', 0) > 0:
        tech_stack += "- Java\n"
    if file_types.get('cs', 0) > 0:
        tech_stack += "- C#\n"
    
    sections["Technology Stack"] = tech_stack
    
    # Look for common project patterns
    has_tests = any('test' in str(path).lower() for path in file_manifest.keys())
    has_docs = any('doc' in str(path).lower() for path in file_manifest.keys())
    has_ci = any('.github/workflows' in str(path) for path in file_manifest.keys())
    
    project_patterns = "The project structure indicates:\n\n"
    if has_tests:
        project_patterns += "- Test suite is present\n"
    if has_docs:
        project_patterns += "- Documentation is available\n"
    if has_ci:
        project_patterns += "- CI/CD configuration is set up\n"
    
    sections["Project Patterns"] = project_patterns
    
    return sections

def build_dependency_graph_from_manifest(file_manifest: dict) -> nx.DiGraph:
    """
    Build a dependency graph from the file manifest.
    
    This function analyzes the file manifest to create a directed graph
    representing dependencies between components.
    
    Args:
        file_manifest: Dictionary of files in the repository
        
    Returns:
        nx.DiGraph: A directed graph representing component dependencies
    """
    graph = nx.DiGraph()
    
    # Detect if this is a Java project
    is_java_project = any(str(path).endswith('.java') for path in file_manifest.keys())
    
    if is_java_project:
        # For Java projects, focus on package structure
        return build_java_dependency_graph(file_manifest)
    else:
        # For other projects, use the generic approach
        return build_generic_dependency_graph(file_manifest)

def build_java_dependency_graph(file_manifest: dict) -> nx.DiGraph:
    """
    Build a dependency graph specifically for Java projects.
    
    Args:
        file_manifest: Dictionary of files in the repository
        
    Returns:
        nx.DiGraph: A directed graph representing Java component dependencies
    """
    graph = nx.DiGraph()
    
    # Map of component name to files
    components = {}
    
    # Extract Java package structure
    for path_obj, info in file_manifest.items():
        path = str(path_obj).replace('\\', '/')
        
        # Only process Java files
        if not path.endswith('.java'):
            continue
            
        # Extract component from path
        if 'src/main/java' in path or 'src\\main\\java' in path:
            parts = path.split('/')
            if '/' not in path:
                parts = path.split('\\')
                
            # Find the index of 'java' in the path
            try:
                idx = parts.index('java')
                if idx + 1 < len(parts):
                    # In a typical Java project, the component is often the first package after the base package
                    # e.g., com.example.component
                    if len(parts) > idx + 4:
                        component_name = parts[idx + 4]  # The component name
                        
                        # Add component to our tracking
                        if component_name not in components:
                            components[component_name] = []
                        components[component_name].append(path)
            except ValueError:
                # 'java' not in path
                pass
    
    # If no components found, try a simpler approach
    if not components:
        for path_obj in file_manifest.keys():
            path = str(path_obj).replace('\\', '/')
            if path.endswith('.java'):
                parts = path.split('/')
                if len(parts) > 3:  # Assuming some depth in the package structure
                    component_name = parts[3] if len(parts) > 3 else parts[-2]
                    if component_name not in components:
                        components[component_name] = []
                    components[component_name].append(path)
    
    # Add components as nodes
    for component, files in components.items():
        # Skip components with too few files
        if len(files) < 2:
            continue
            
        # Add node with file count as an attribute
        graph.add_node(component, files=len(files))
    
    # For Java Spring projects, create relationships based on common patterns
    common_patterns = {
        'controller': ['service', 'dto', 'model'],
        'service': ['repository', 'model', 'dto', 'domain'],
        'repository': ['entity', 'model'],
        'config': ['service'],
        'util': [],  # Utilities are usually depended upon, not depending on others
        'exception': ['controller', 'service'],
        'domain': ['model', 'entity'],
        'dto': ['model', 'entity'],
        'model': ['entity'],
        'kafka': ['service', 'model'],
        'client': ['service', 'dto'],
        'proxy': ['service', 'client', 'model']
    }
    
    # Add edges based on common patterns
    for component in list(graph.nodes()):
        lower_component = component.lower()
        
        # Find matching pattern
        for pattern, dependencies in common_patterns.items():
            if pattern in lower_component:
                for dep_pattern in dependencies:
                    # Find components matching the dependency pattern
                    for potential_dep in list(graph.nodes()):
                        if dep_pattern in potential_dep.lower() and potential_dep != component:
                            graph.add_edge(component, potential_dep)
    
    return graph

def build_generic_dependency_graph(file_manifest: dict) -> nx.DiGraph:
    """
    Build a dependency graph for non-Java projects.
    
    Args:
        file_manifest: Dictionary of files in the repository
        
    Returns:
        nx.DiGraph: A directed graph representing component dependencies
    """
    graph = nx.DiGraph()
    
    # Extract main components (directories)
    components = set()
    for path in file_manifest.keys():
        parts = str(path).replace('\\', '/').split('/')
        if len(parts) > 1:
            # Add first-level directory as a component
            components.add(parts[0])
            
            # If it's a deeper structure, add second-level directory too
            if len(parts) > 2:
                components.add(f"{parts[0]}/{parts[1]}")
    
    # Add components as nodes
    for component in components:
        graph.add_node(component)
    
    # Infer dependencies based on imports (simplified approach)
    for path, info in file_manifest.items():
        parts = str(path).replace('\\', '/').split('/')
        if len(parts) < 2:
            continue
            
        # Get the component for this file
        component = parts[0]
        
        # Check if the file has imports
        imports = getattr(info, 'imports', [])
        if imports:
            for imported in imports:
                # Try to map the import to a component
                for potential_component in components:
                    if potential_component in imported and component != potential_component:
                        graph.add_edge(component, potential_component)
                        break
    
    # If the graph is still empty, create some basic relationships
    # based on common architectural patterns
    if len(graph.edges()) == 0:
        common_components = {
            'src': ['lib', 'utils', 'helpers'],
            'app': ['src', 'utils', 'components'],
            'controllers': ['models', 'services'],
            'services': ['models', 'repositories'],
            'views': ['controllers', 'components'],
            'components': ['utils', 'helpers'],
            'api': ['services', 'controllers'],
            'test': ['src', 'app', 'lib']
        }
        
        for component in components:
            base_component = component.split('/')[0]
            if base_component in common_components:
                for dependency in common_components[base_component]:
                    for potential_component in components:
                        if dependency in potential_component and component != potential_component:
                            graph.add_edge(component, potential_component)
    
    return graph

# End of file