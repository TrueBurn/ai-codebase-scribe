# Standard library imports
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Local imports
from ..utils.markdown_validator import MarkdownValidator
from ..utils.readability import ReadabilityScorer
from ..clients.base_llm import BaseLLMClient
from ..analyzers.codebase import CodebaseAnalyzer
from ..utils.config_class import ScribeConfig

# Constants for configuration
# Thresholds for content length and quality checks
CONTENT_THRESHOLDS = {
    'meaningful_readme_lines': 5,  # Minimum lines for a README to be considered meaningful
    'usage_text_length': 100,      # Minimum length for usage text to be used as overview
    'overview_paragraph_length': 50,  # Minimum length for a paragraph to be used as overview
    'usage_guide_length': 50,      # Minimum length for a valid usage guide
    'contributing_guide_length': 50,  # Minimum length for a valid contributing guide
    'license_info_length': 20,     # Minimum length for valid license info
    'readability_score_threshold': 40  # Threshold for readability warnings
}

def extract_license_info(repo_path: Path) -> str:
    """
    Extract license information from the repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        str: License information or default text
    """
    # Check for LICENSE file
    license_file = repo_path / "LICENSE"
    if license_file.exists():
        # Read first 10 lines to get license type
        with open(license_file, 'r', encoding='utf-8', errors='ignore') as f:
            license_text = ''.join(f.readlines()[:10])
            
        # Try to identify common licenses
        if "MIT" in license_text:
            return "MIT License"
        elif "Apache" in license_text:
            return "Apache License"
        elif "GPL" in license_text or "GNU GENERAL PUBLIC LICENSE" in license_text:
            return "GPL License"
        else:
            return "Custom License (see LICENSE file for details)"
    
    # Check for license information in package.json
    package_json = repo_path / "package.json"
    if package_json.exists():
        try:
            import json
            with open(package_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "license" in data:
                    return f"{data['license']} License"
        except:
            pass
    
    # Default license text
    return "No license information found. Consider adding a LICENSE file."

# LLM instruction phrases that should be removed from generated content
INSTRUCTION_PHRASES = [
    "Your task is to preserve",
    "Do not remove specific implementation",
    "Focus on adding missing information",
    "Maintain the original structure",
    "Return the enhanced document"
]

async def generate_readme(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    file_summaries: dict,
    config: ScribeConfig,
    analyzer: CodebaseAnalyzer,
    output_dir: str,
    existing_readme: Optional[str] = None,
    architecture_file_exists: bool = False
) -> str:
    """
    Generate README content using file summaries.
    
    Args:
        repo_path: Path to the repository root
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        file_summaries: Dictionary of file summaries
        config: Configuration dictionary
        analyzer: CodebaseAnalyzer instance
        output_dir: Output directory for generated files
        existing_readme: Optional existing README content
        architecture_file_exists: Whether ARCHITECTURE.md exists
        
    Returns:
        str: Generated README content
    """
    try:
        # Use the analyzer's method to get a consistent project name
        debug_mode = config.debug
        project_name = analyzer.derive_project_name(debug_mode)
        logging.info(f"Using project name: {project_name}")
        
        # Check if we should enhance existing README or create a new one
        if should_enhance_existing_readme(repo_path, config):
            return await enhance_existing_readme(
                repo_path, llm_client, file_manifest, project_name, architecture_file_exists
            )
        
        # Generate new README from scratch
        return await generate_new_readme(
            repo_path, llm_client, file_manifest, project_name, 
            architecture_file_exists, config
        )
    except Exception as e:
        logging.error(f"Error generating README: {e}")
        return generate_fallback_readme(repo_path, architecture_file_exists)

def should_enhance_existing_readme(repo_path: Path, config: ScribeConfig) -> bool:
    """
    Determine if we should enhance an existing README.
    
    Args:
        repo_path: Path to the repository
        config: Configuration
        
    Returns:
        bool: True if we should enhance existing README
    """
    # Check if we should preserve existing content
    preserve_existing = config.preserve_existing
    readme_path = repo_path / 'README.md'
    # Check if README exists and preserve_existing is True
    if not (preserve_existing and readme_path.exists()):
        return False
        
    try:
        existing_content = readme_path.read_text(encoding='utf-8')
        # Check if it's not just a placeholder or default README
        return len(existing_content.strip().split('\n')) > CONTENT_THRESHOLDS['meaningful_readme_lines']
    except Exception as e:
        logging.error(f"Error reading existing README: {e}")
        return False

async def enhance_existing_readme(
    repo_path: Path, 
    llm_client: BaseLLMClient, 
    file_manifest: dict,
    project_name: str,
    architecture_file_exists: bool
) -> str:
    """
    Enhance an existing README file.
    
    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        architecture_file_exists: Whether ARCHITECTURE.md exists
        
    Returns:
        str: Enhanced README content
    """
    logging.info("Found existing README.md with meaningful content. Will enhance rather than replace.")
    print("Enhancing existing README.md rather than replacing it.")
    
    try:
        # Read existing content
        readme_path = repo_path / 'README.md'
        existing_content = readme_path.read_text(encoding='utf-8')
        
        # Generate project overview for context
        overview = await generate_overview(llm_client, file_manifest, project_name)
        
        # Enhance existing content
        enhanced_content = await llm_client.enhance_documentation(
            existing_content=existing_content,
            file_manifest=file_manifest,
            doc_type="README.md"
        )
        
        # Remove any leaked instructions
        for phrase in INSTRUCTION_PHRASES:
            enhanced_content = enhanced_content.replace(phrase, "")
        
        # Add architecture link if needed
        enhanced_content = add_architecture_link_if_needed(
            enhanced_content, architecture_file_exists
        )
        
        # Ensure correct project name in title
        enhanced_content = ensure_correct_title(enhanced_content, project_name)
        
        return enhanced_content
    except Exception as e:
        logging.error(f"Error enhancing existing README: {e}")
        logging.info("Falling back to generating new README")
        return None

async def generate_new_readme(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    architecture_file_exists: bool,
    config: ScribeConfig
) -> str:
    """
    Generate a new README from scratch.
    
    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        architecture_file_exists: Whether ARCHITECTURE.md exists
        config: Configuration
        
    Returns:
        str: Generated README content
    """
    # Generate all the sections
    overview = await generate_overview_with_fallbacks(
        repo_path, llm_client, file_manifest, project_name, architecture_file_exists
    )
    
    usage = await generate_section(
        llm_client, file_manifest, 'usage_guide', 
        CONTENT_THRESHOLDS['usage_guide_length'],
        "Please refer to project documentation for usage instructions."
    )
    
    # Contributing section is now in a separate file
    contributing = "For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)."
    
    license_info = await generate_section(
        llm_client, file_manifest, 'license_info',
        CONTENT_THRESHOLDS['license_info_length'],
        "Please refer to the LICENSE file for license information."
    )
    
    # Add architecture link if ARCHITECTURE.md exists
    architecture_section = ""
    if architecture_file_exists:
        architecture_section = "\n## Architecture\n\nFor detailed architecture documentation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).\n"
    
    # Remove any existing markdown headers from the sections
    overview = _clean_section_headers(overview)
    usage = _clean_section_headers(usage)
    contributing = _clean_section_headers(contributing)
    license_info = _clean_section_headers(license_info)
    
    # Assemble the README content
    readme_content = f"""# {project_name}

## Table of Contents
- [Overview](#overview)
{architecture_section and f"- [Architecture](#{_format_anchor_link('Architecture')})" or ""}
- [Usage](#{_format_anchor_link('Usage')})
- [Contributing](#{_format_anchor_link('Contributing')})
- [License](#{_format_anchor_link('License')})

## Overview

{overview}

{architecture_section}

## Usage

{usage}

## Contributing

{contributing}

## License

{license_info}
"""
    
    # Validate and improve the content
    return await validate_and_improve_content(readme_content, repo_path)

async def generate_overview(
    llm_client: BaseLLMClient, 
    file_manifest: dict, 
    project_name: str
) -> str:
    """
    Generate a project overview using the LLM.
    
    Args:
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        
    Returns:
        str: Generated overview or fallback text
    """
    try:
        overview = await llm_client.generate_project_overview(file_manifest)
        logging.info(f"Generated overview length: {len(overview) if overview else 0}")
        return overview
    except Exception as e:
        logging.error(f"Error generating overview: {e}")
        logging.error(f"Exception type: {type(e)}")
        logging.error(f"Exception traceback: {traceback.format_exc()}")
        return f"{project_name} is a software project."

async def generate_overview_with_fallbacks(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    architecture_file_exists: bool
) -> str:
    """
    Generate a project overview with multiple fallback strategies.
    
    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        architecture_file_exists: Whether ARCHITECTURE.md exists
        
    Returns:
        str: Generated overview text
    """
    logging.info("Generating project overview...")
    
    # Strategy 1: Generate from LLM
    try:
        overview = await llm_client.generate_project_overview(file_manifest)
        if overview:
            logging.info(f"Generated overview length: {len(overview)}")
            return overview
    except Exception as e:
        logging.error(f"Error in LLM call for project overview: {e}")
        logging.error(f"Exception type: {type(e)}")
        logging.error(f"Exception traceback: {traceback.format_exc()}")
    
    # Strategy 2: Extract from architecture document
    if architecture_file_exists:
        overview = extract_overview_from_architecture(repo_path)
        if overview:
            return overview
    
    # Strategy 3: Extract from usage guide
    try:
        usage_text = await llm_client.generate_usage_guide(file_manifest)
        if usage_text and len(usage_text) > CONTENT_THRESHOLDS['usage_text_length']:
            # Extract first paragraph as overview
            first_para = usage_text.split('\n\n')[0]
            if len(first_para) > CONTENT_THRESHOLDS['overview_paragraph_length']:
                logging.info(f"Using first paragraph of usage as overview, length: {len(first_para)}")
                return first_para
    except Exception as e:
        logging.error(f"Error generating usage for overview: {e}")
    
    # Fallback
    logging.info("Using fallback overview text")
    return f"{project_name} is a software project containing {len(file_manifest)} files. Please refer to the documentation for more details."

def extract_overview_from_architecture(repo_path: Path) -> Optional[str]:
    """
    Extract overview section from ARCHITECTURE.md if it exists.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Optional[str]: Extracted overview or None
    """
    logging.info("Attempting to extract overview from ARCHITECTURE.md...")
    arch_path = repo_path / "docs" / "ARCHITECTURE.md"
    if arch_path.exists():
        try:
            arch_content = arch_path.read_text(encoding='utf-8')
            overview_match = re.search(r'## Overview\s+(.+?)(?=##|\Z)', arch_content, re.DOTALL)
            if overview_match:
                overview = overview_match.group(1).strip()
                logging.info(f"Using overview from ARCHITECTURE.md, length: {len(overview)}")
                return overview
        except Exception as e:
            logging.error(f"Error reading ARCHITECTURE.md: {e}")
    return None

async def generate_section(
    llm_client: BaseLLMClient,
    file_manifest: dict,
    section_type: str,
    min_length: int,
    fallback_text: str
) -> str:
    """
    Generate a section of the README with error handling.
    
    Args:
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        section_type: Type of section to generate (e.g., 'usage_guide')
        min_length: Minimum acceptable length for the section
        fallback_text: Text to use if generation fails
        
    Returns:
        str: Generated section content
    """
    try:
        # Use getattr to call the appropriate method on llm_client
        method = getattr(llm_client, f"generate_{section_type}")
        content = await method(file_manifest)
        
        if not content or len(content) < min_length:
            return fallback_text
        return content
    except Exception as e:
        logging.error(f"Error generating {section_type}: {e}")
        logging.error(f"Exception type: {type(e)}")
        logging.error(f"Exception traceback: {traceback.format_exc()}")
        return fallback_text

def add_architecture_link_if_needed(content: str, architecture_file_exists: bool) -> str:
    """
    Add link to ARCHITECTURE.md if it exists and not already mentioned.
    
    Args:
        content: README content
        architecture_file_exists: Whether ARCHITECTURE.md exists
        
    Returns:
        str: Updated content with architecture link if needed
    """
    if not architecture_file_exists or "ARCHITECTURE.md" in content:
        return content
        
    architecture_link = "\n\n## Architecture\n\nFor detailed architecture documentation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).\n"
    
    # Find a good place to insert the link
    if "## Usage" in content:
        return content.replace("## Usage", f"{architecture_link}\n## Usage")
    else:
        # Append to the end if no good insertion point
        return content + architecture_link

def ensure_correct_title(content: str, project_name: str) -> str:
    """
    Ensure the README has the correct project name in the title.
    
    Args:
        content: README content
        project_name: Name of the project
        
    Returns:
        str: Content with corrected title
    """
    title_match = re.search(r'^# (.+?)(?:\n|$)', content)
    if title_match:
        old_title = title_match.group(1)
        if old_title.lower() == "project" or old_title.strip() == "":
            return content.replace(f"# {old_title}", f"# {project_name}")
    else:
        # No title found, add one
        return f"# {project_name}\n\n{content}"
    
    return content

async def validate_and_improve_content(content: str, repo_path: Path) -> str:
    """
    Validate markdown structure and links, check readability, and apply fixes.
    
    Args:
        content: README content to validate
        repo_path: Path to the repository
        
    Returns:
        str: Improved content
    """
    # Validate markdown structure and links
    md_validator = MarkdownValidator(content)
    validation_issues = await md_validator.validate_with_link_checking(repo_path)
    
    # Log markdown and link issues
    if validation_issues:
        log_validation_issues(validation_issues)
    
    # Check readability
    check_readability(content)
    
    # Apply automatic fixes
    improved_content = md_validator.fix_common_issues()
    
    # Validate again to see if issues were resolved
    fixed_validator = MarkdownValidator(improved_content)
    remaining_issues = fixed_validator.validate()
    if remaining_issues:
        logging.info(f"Fixed some issues. {len(remaining_issues)} issues remain.")
    else:
        logging.info("All markdown issues were automatically fixed!")
    
    return improved_content

def log_validation_issues(validation_issues: List[Any]) -> None:
    """
    Log markdown validation issues.
    
    Args:
        validation_issues: List of validation issues
    """
    link_issues = [issue for issue in validation_issues if "Link issue" in issue.message]
    format_issues = [issue for issue in validation_issues if "Link issue" not in issue.message]
    
    if format_issues:
        logging.warning("Found markdown formatting issues in new README:")
        for issue in format_issues:
            logging.warning(f"  - {issue}")
    
    if link_issues:
        logging.warning(f"Found {len(link_issues)} invalid links in new README")
        for issue in link_issues:
            logging.warning(f"  - {issue}")

def check_readability(content: str) -> None:
    """
    Check readability of the content and log warnings if too complex.
    
    Args:
        content: Content to check
    """
    scorer = ReadabilityScorer()
    score = scorer.analyze_text(content, "README")
    
    threshold = CONTENT_THRESHOLDS['readability_score_threshold']
    
    if isinstance(score, dict):
        # Extract the overall score from the dictionary
        overall_score = score.get('overall', 0)
        if overall_score > threshold:  # Higher score means more complex text
            print("\nWarning: README.md content may be too complex.")
            print("Recommendations:")
            print("- Consider simplifying language for better readability")
            print("- Use simpler words where possible")
    elif score > threshold:  # For backward compatibility if score is a number
        print("\nWarning: README.md content may be too complex.")
        print("Recommendations:")
        print("- Consider simplifying language for better readability")
        print("- Use simpler words where possible")

def generate_fallback_readme(repo_path: Path, architecture_file_exists: bool) -> str:
    """
    Generate a minimal valid README in case of errors.
    
    Args:
        repo_path: Path to the repository
        architecture_file_exists: Whether ARCHITECTURE.md exists
        
    Returns:
        str: Minimal README content
    """
    return f"""# {repo_path.name}

## Overview

This is the README for the {repo_path.name} project.

## Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---
*This README was automatically generated but encountered errors during processing.*
"""

def _clean_section_headers(content: str) -> str:
    """
    Remove existing markdown headers from a section to prevent duplicates.
    
    Args:
        content: Section content
        
    Returns:
        str: Content without headers
    """
    if not content:
        return ""
    
    # Remove any lines that start with # at the beginning of the content
    lines = content.split('\n')
    while lines and lines[0].strip().startswith('#'):
        lines.pop(0)
    
    # Join the remaining lines
    return '\n'.join(lines).strip()

def _format_anchor_link(section_name: str) -> str:
    """
    Format a section name into a proper anchor link by removing special characters.
    
    Args:
        section_name: Name of the section
        
    Returns:
        str: Formatted anchor link
    """
    return section_name.lower().replace(' ', '-').replace('.', '').replace('/', '').replace('(', '').replace(')', '')