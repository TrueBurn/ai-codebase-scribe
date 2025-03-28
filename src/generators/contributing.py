# Standard library imports
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, Optional

# Local imports
from ..clients.base_llm import BaseLLMClient
from ..utils.markdown_validator import MarkdownValidator
from ..utils.readability import ReadabilityScorer
from ..analyzers.codebase import CodebaseAnalyzer
from ..utils.config_class import ScribeConfig

# Constants for configuration
CONTENT_THRESHOLDS = {
    'contributing_guide_length': 50,  # Minimum length for a valid contributing guide
    'readability_score_threshold': 40  # Threshold for readability warnings
}

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
    try:
        # Use the analyzer's method to get a consistent project name
        debug_mode = config.debug
        project_name = analyzer.derive_project_name(debug_mode)
        logging.info(f"Generating contributing guide for: {project_name}")
        
        # Check if we should enhance existing CONTRIBUTING.md or create a new one
        if should_enhance_existing_contributing(repo_path, config):
            return await enhance_existing_contributing(
                repo_path, llm_client, file_manifest, project_name
            )
        
        # Generate new CONTRIBUTING.md from scratch
        return await generate_new_contributing(
            repo_path, llm_client, file_manifest, project_name, config
        )
    except Exception as e:
        logging.error(f"Error generating CONTRIBUTING.md: {e}")
        return generate_fallback_contributing(repo_path)

def should_enhance_existing_contributing(repo_path: Path, config: ScribeConfig) -> bool:
    """
    Determine if we should enhance an existing CONTRIBUTING.md.
    
    Args:
        repo_path: Path to the repository
        config: Configuration
        
    Returns:
        bool: True if we should enhance existing CONTRIBUTING.md
    """
    # Check if we should preserve existing content
    preserve_existing = config.preserve_existing
    contributing_path = repo_path / 'CONTRIBUTING.md'  # In root directory for GitHub recognition
    # Check if CONTRIBUTING.md exists and preserve_existing is True
    if not (preserve_existing and contributing_path.exists()):
        return False
        
    try:
        existing_content = contributing_path.read_text(encoding='utf-8')
        # Check if it's not just a placeholder or default CONTRIBUTING
        return len(existing_content.strip().split('\n')) > 5
    except Exception as e:
        logging.error(f"Error reading existing CONTRIBUTING.md: {e}")
        return False

async def enhance_existing_contributing(
    repo_path: Path, 
    llm_client: BaseLLMClient, 
    file_manifest: dict,
    project_name: str
) -> str:
    """
    Enhance an existing CONTRIBUTING.md file.
    
    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        
    Returns:
        str: Enhanced CONTRIBUTING.md content
    """
    logging.info("Found existing CONTRIBUTING.md with meaningful content. Will enhance rather than replace.")
    print("Enhancing existing CONTRIBUTING.md rather than replacing it.")
    
    try:
        # Read existing content
        contributing_path = repo_path / 'CONTRIBUTING.md'  # In root directory for GitHub recognition
        existing_content = contributing_path.read_text(encoding='utf-8')
        
        # Enhance existing content
        enhanced_content = await llm_client.enhance_documentation(
            existing_content=existing_content,
            file_manifest=file_manifest,
            doc_type="CONTRIBUTING.md"
        )
        
        # Ensure correct project name in title
        enhanced_content = ensure_correct_title(enhanced_content, project_name)
        
        return enhanced_content
    except Exception as e:
        logging.error(f"Error enhancing existing CONTRIBUTING.md: {e}")
        logging.info("Falling back to generating new CONTRIBUTING.md")
        return None

async def generate_new_contributing(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    config: ScribeConfig
) -> str:
    """
    Generate a new CONTRIBUTING.md from scratch.
    
    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        config: Configuration
        
    Returns:
        str: Generated CONTRIBUTING.md content
    """
    # Generate contributing guide content
    contributing = await generate_contributing_content(
        llm_client, file_manifest, 
        CONTENT_THRESHOLDS['contributing_guide_length'],
        f"# Contributing to {project_name}\n\nThank you for your interest in contributing to {project_name}. Please refer to the project documentation for contribution guidelines."
    )
    
    # Ensure the document has a proper title
    if not contributing.strip().startswith('# '):
        contributing = f"# Contributing to {project_name}\n\n{contributing}"
    
    # Validate and improve the content
    return await validate_and_improve_content(contributing, repo_path)

async def generate_contributing_content(
    llm_client: BaseLLMClient,
    file_manifest: dict,
    min_length: int,
    fallback_text: str
) -> str:
    """
    Generate contributing guide content with error handling.
    
    Args:
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        min_length: Minimum acceptable length for the content
        fallback_text: Text to use if generation fails
        
    Returns:
        str: Generated contributing guide content
    """
    try:
        content = await llm_client.generate_contributing_guide(file_manifest)
        
        if not content or len(content) < min_length:
            return fallback_text
        return content
    except Exception as e:
        logging.error(f"Error generating contributing guide: {e}")
        logging.error(f"Exception type: {type(e)}")
        logging.error(f"Exception traceback: {traceback.format_exc()}")
        return fallback_text

def ensure_correct_title(content: str, project_name: str) -> str:
    """
    Ensure the CONTRIBUTING.md has the correct project name in the title.
    
    Args:
        content: CONTRIBUTING.md content
        project_name: Name of the project
        
    Returns:
        str: Content with corrected title
    """
    import re
    
    title_match = re.search(r'^# (.+?)(?:\n|$)', content)
    if title_match:
        old_title = title_match.group(1)
        if "contributing" not in old_title.lower() or "project" in old_title.lower():
            return content.replace(f"# {old_title}", f"# Contributing to {project_name}")
    else:
        # No title found, add one
        return f"# Contributing to {project_name}\n\n{content}"
    
    return content

async def validate_and_improve_content(content: str, repo_path: Path) -> str:
    """
    Validate markdown structure and links, check readability, and apply fixes.
    
    Args:
        content: CONTRIBUTING.md content to validate
        repo_path: Path to the repository
        
    Returns:
        str: Improved content
    """
    # Validate markdown structure and links
    md_validator = MarkdownValidator(content)
    validation_issues = await md_validator.validate_with_link_checking(repo_path)
    
    # Log markdown and link issues
    if validation_issues:
        for issue in validation_issues:
            logging.warning(f"Validation issue: {issue}")
    
    # Check readability
    check_readability(content)
    
    # Apply automatic fixes
    improved_content = md_validator.fix_common_issues()
    
    return improved_content

def check_readability(content: str) -> None:
    """
    Check readability of the content and log warnings if too complex.
    
    Args:
        content: Content to check
    """
    scorer = ReadabilityScorer()
    score = scorer.analyze_text(content, "CONTRIBUTING")
    
    threshold = CONTENT_THRESHOLDS['readability_score_threshold']
    
    if isinstance(score, dict):
        # Extract the overall score from the dictionary
        overall_score = score.get('overall', 0)
        if overall_score > threshold:  # Higher score means more complex text
            print("\nWarning: CONTRIBUTING.md content may be too complex.")
            print("Recommendations:")
            print("- Consider simplifying language for better readability")
            print("- Use simpler words where possible")
    elif score > threshold:  # For backward compatibility if score is a number
        print("\nWarning: CONTRIBUTING.md content may be too complex.")
        print("Recommendations:")
        print("- Consider simplifying language for better readability")
        print("- Use simpler words where possible")

def generate_fallback_contributing(repo_path: Path) -> str:
    """
    Generate a minimal valid CONTRIBUTING.md in case of errors.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        str: Minimal CONTRIBUTING.md content
    """
    return f"""# Contributing to {repo_path.name}

Thank you for your interest in contributing to {repo_path.name}!

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## Pull Request Process

1. Ensure your code follows the project's coding standards
2. Update the documentation as needed
3. The pull request will be reviewed by maintainers

## Development Setup

Please refer to the [README.md](../README.md) for development setup instructions.

---
*This CONTRIBUTING.md was automatically generated.*
"""