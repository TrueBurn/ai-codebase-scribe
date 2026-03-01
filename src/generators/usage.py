# Standard library imports
import asyncio
import hashlib
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import visual logger, fallback to regular logging
try:
    from ..utils.visual_logger import get_visual_logger

    _use_visual_logging = True
except ImportError:
    _use_visual_logging = False


def _get_logger():
    """Get appropriate logger (visual or standard)."""
    if _use_visual_logging:
        return get_visual_logger()
    return logging.getLogger(__name__)


from ..analyzers.codebase import CodebaseAnalyzer

# Local imports
from ..clients.base_llm import BaseLLMClient
from ..utils.config_class import ScribeConfig
from ..utils.doc_utils import extract_section_by_name
from ..utils.markdown_validator import MarkdownValidator
from ..utils.readability import ReadabilityScorer

# Constants for configuration
CONTENT_THRESHOLDS = {
    "usage_guide_length": 50,  # Minimum length for a valid usage guide
    "readability_score_threshold": 40,  # Threshold for readability warnings
}


async def generate_usage(
    repo_path: Path,
    file_manifest: dict,
    llm_client: BaseLLMClient,
    config: ScribeConfig,
    analyzer: CodebaseAnalyzer,
    related_repo_data: Optional[Dict[str, Any]] = None,
    persistence_info: Optional[Any] = None,
) -> str:
    """
    Generate USAGE.md content.

    Args:
        repo_path: Path to the repository root
        file_manifest: Dictionary of files in the repository
        llm_client: LLM client for generating content
        config: Configuration
        analyzer: CodebaseAnalyzer instance
        related_repo_data: Optional related repository data for context
        persistence_info: Optional persistence layer information

    Returns:
        str: Generated USAGE.md content
    """
    try:
        # Use the analyzer's method to get a consistent project name
        debug_mode = config.debug
        project_name = analyzer.derive_project_name(debug_mode)
        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                f"Generating usage guide for: {logger.format_filename(project_name)}",
                emoji="rocket",
            )
        else:
            logging.info(f"Generating usage guide for: {project_name}")

        # Check if we should enhance existing USAGE.md or create a new one
        if should_enhance_existing_usage(repo_path, config):
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "Found existing USAGE.md with meaningful content. Will enhance rather than replace.",
                    emoji="rocket",
                )
            else:
                logging.info(
                    "Found existing USAGE.md with meaningful content. Will enhance rather than replace."
                )

            try:
                enhanced_content = await enhance_existing_usage(
                    repo_path,
                    llm_client,
                    file_manifest,
                    project_name,
                    related_repo_data,
                )
                # If enhancement succeeded, return it
                if enhanced_content:
                    return enhanced_content
            except (asyncio.TimeoutError, TimeoutError) as e:
                # Debug log to see if we're catching the timeout at generator level
                logger = _get_logger()
                if _use_visual_logging:
                    logger.debug(
                        f"Caught timeout in generate_usage: {type(e).__name__}: {e}"
                    )
                else:
                    logging.debug(
                        f"Caught timeout in generate_usage: {type(e).__name__}: {e}"
                    )
                # If timeout occurs at this level, return original content
                logger = _get_logger()
                if _use_visual_logging:
                    logger.warning(
                        "USAGE enhancement timed out. Using original content",
                        emoji="timer",
                    )
                else:
                    logging.warning(
                        "USAGE enhancement timed out. Using original content"
                    )

                # Read and return the original USAGE content
                usage_path = repo_path / "docs" / "USAGE.md"
                if usage_path.exists():
                    return usage_path.read_text(encoding="utf-8")

            except Exception as e:
                logger = _get_logger()
                if _use_visual_logging:
                    logger.warning(
                        f"USAGE enhancement failed: {e}. Using original content",
                        emoji="warning",
                    )
                else:
                    logging.warning(
                        f"USAGE enhancement failed: {e}. Using original content"
                    )

                # Read and return the original USAGE content
                usage_path = repo_path / "docs" / "USAGE.md"
                if usage_path.exists():
                    return usage_path.read_text(encoding="utf-8")

            # If we reach here, enhancement failed, so proceed to generate new content
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "Enhancement failed, proceeding to generate new USAGE.md",
                    emoji="rocket",
                )
            else:
                logging.info("Enhancement failed, proceeding to generate new USAGE.md")

        # Generate new USAGE.md from scratch
        return await generate_new_usage(
            repo_path,
            llm_client,
            file_manifest,
            project_name,
            config,
            related_repo_data,
        )
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Error generating USAGE.md: {e}", emoji="error")
            logger.error(f"Exception type: {type(e).__name__}", emoji="error")
            logger.error(f"Exception details: {str(e)}", emoji="error")
            if hasattr(e, "__traceback__"):
                logger.error(f"Traceback: {traceback.format_exc()}", emoji="error")
            logger.info("Falling back to minimal USAGE.md template", emoji="template")
        else:
            logging.error(f"Error generating USAGE.md: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            logging.error(f"Exception details: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
            logging.info("Falling back to minimal USAGE.md template")
        return generate_fallback_usage(repo_path)


def extract_existing_content_from_sources(
    repo_path: Path, section_names: List[str]
) -> str:
    """
    Extract existing usage-related content from README.md or CONTRIBUTING.md.

    Args:
        repo_path: Path to the repository root
        section_names: List of section names to extract (e.g., ["Usage", "Using"])

    Returns:
        str: Extracted content from existing files
    """
    extracted_content = []

    # Check README.md first
    readme_path = repo_path / "README.md"
    if readme_path.exists():
        try:
            readme_content = readme_path.read_text(encoding="utf-8")
            for section_name in section_names:
                section_content = extract_section_by_name(readme_content, section_name)
                if section_content:
                    extracted_content.append(
                        f"## {section_name}\n\n{section_content}\n"
                    )
        except Exception as e:
            logger = _get_logger()
            if _use_visual_logging:
                logger.debug(f"Could not extract from README.md: {e}")
            else:
                logging.debug(f"Could not extract from README.md: {e}")

    # Check CONTRIBUTING.md
    contributing_path = repo_path / "CONTRIBUTING.md"
    if contributing_path.exists():
        try:
            contributing_content = contributing_path.read_text(encoding="utf-8")
            for section_name in section_names:
                section_content = extract_section_by_name(
                    contributing_content, section_name
                )
                if section_content:
                    extracted_content.append(
                        f"## {section_name}\n\n{section_content}\n"
                    )
        except Exception as e:
            logger = _get_logger()
            if _use_visual_logging:
                logger.debug(f"Could not extract from CONTRIBUTING.md: {e}")
            else:
                logging.debug(f"Could not extract from CONTRIBUTING.md: {e}")

    return "\n".join(extracted_content) if extracted_content else ""


def extract_enhancement_log(content: str) -> Tuple[str, Optional[str]]:
    """
    Extract enhancement log from content and return cleaned content.

    Args:
        content: Content that may contain enhancement log

    Returns:
        Tuple of (cleaned_content, enhancement_log)
    """
    # Pattern to match enhancement log comments
    log_pattern = r"<!--\s*ENHANCEMENT_LOG:\s*([^>]+)\s*-->"

    match = re.search(log_pattern, content, re.IGNORECASE | re.DOTALL)
    if match:
        enhancement_log = match.group(1).strip()
        cleaned_content = re.sub(
            log_pattern, "", content, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        return cleaned_content, enhancement_log

    return content, None


def detect_content_changes(original: str, enhanced: str) -> Dict[str, Any]:
    """
    Detect if meaningful changes were made to content.

    Args:
        original: Original content
        enhanced: Enhanced content

    Returns:
        Dictionary with change detection results
    """
    original_hash = hashlib.md5(original.encode()).hexdigest()
    enhanced_hash = hashlib.md5(enhanced.encode()).hexdigest()

    return {
        "content_changed": original_hash != enhanced_hash,
        "size_change_bytes": len(enhanced) - len(original),
        "original_size": len(original),
        "enhanced_size": len(enhanced),
    }


def should_enhance_existing_usage(repo_path: Path, config: ScribeConfig) -> bool:
    """
    Determine if we should enhance an existing USAGE.md.

    Args:
        repo_path: Path to the repository
        config: Configuration

    Returns:
        bool: True if we should enhance existing USAGE.md
    """
    # Check if we should preserve existing content
    preserve_existing = config.preserve_existing
    usage_path = repo_path / "docs" / "USAGE.md"
    # Check if USAGE.md exists and preserve_existing is True
    if not (preserve_existing and usage_path.exists()):
        return False

    try:
        existing_content = usage_path.read_text(encoding="utf-8")
        # Check if it's not just a placeholder or default USAGE
        return len(existing_content.strip().split("\n")) > 5
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Error reading existing USAGE.md: {e}", emoji="error")
        else:
            logging.error(f"Error reading existing USAGE.md: {e}")
        return False


async def enhance_existing_usage(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enhance an existing USAGE.md file.

    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        related_repo_data: Optional related repository data for context

    Returns:
        str: Enhanced USAGE.md content
    """
    logger = _get_logger()
    if _use_visual_logging:
        logger.info(
            "Found existing USAGE.md with meaningful content. Will enhance rather than replace.",
            emoji="rocket",
        )
    else:
        logging.info(
            "Found existing USAGE.md with meaningful content. Will enhance rather than replace."
        )

    try:
        # Read existing content
        usage_path = repo_path / "docs" / "USAGE.md"
        existing_content = usage_path.read_text(encoding="utf-8")

        try:
            # Attempt standard enhancement
            enhanced_content = await llm_client.enhance_documentation(
                existing_content=existing_content,
                file_manifest=file_manifest,
                doc_type="USAGE.md",
            )
        except Exception as e:
            # Catch any exception (including TimeoutError) and check the type
            logger = _get_logger()
            if "timeout" in str(e).lower() or "TimeoutError" in str(type(e)):
                # This is a timeout - log and use original content
                file_size = len(existing_content)
                file_size_kb = file_size / 1024
                if _use_visual_logging:
                    logger.warning(
                        f"USAGE enhancement timed out for large file ({file_size_kb:.1f}KB). Using original content.",
                        emoji="timer",
                    )
                else:
                    logging.warning(
                        f"USAGE enhancement timed out for large file ({file_size_kb:.1f}KB). Using original content."
                    )
                enhanced_content = existing_content
            else:
                # This is some other error
                if _use_visual_logging:
                    logger.warning(
                        f"USAGE enhancement failed: {e}. Using original content.",
                        emoji="warning",
                    )
                else:
                    logging.warning(
                        f"USAGE enhancement failed: {e}. Using original content."
                    )
                enhanced_content = existing_content

        # Extract enhancement log and clean content
        enhanced_content, enhancement_log = extract_enhancement_log(enhanced_content)

        # Detect changes
        changes = detect_content_changes(existing_content, enhanced_content)

        # Log enhancement results
        if changes["content_changed"]:
            if enhancement_log:
                logger = _get_logger()
                if _use_visual_logging:
                    logger.success(
                        f"USAGE.md enhanced with changes: {enhancement_log}",
                        emoji="rocket",
                    )
                else:
                    logging.info(f"USAGE.md enhanced with changes: {enhancement_log}")
            else:
                logger = _get_logger()
                if _use_visual_logging:
                    logger.success(
                        f"USAGE.md enhanced (size change: {changes['size_change_bytes']} bytes)",
                        emoji="rocket",
                    )
                else:
                    logging.info(
                        f"USAGE.md enhanced (size change: {changes['size_change_bytes']} bytes)"
                    )
        else:
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "USAGE.md: No meaningful changes made by enhancement",
                    emoji="info",
                )
            else:
                logging.info("USAGE.md: No meaningful changes made by enhancement")

        # Ensure correct project name in title
        enhanced_content = ensure_correct_title(enhanced_content, project_name)

        return enhanced_content
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Error enhancing existing USAGE.md: {e}", emoji="error")
            logger.error(f"Exception type: {type(e).__name__}", emoji="error")
            if hasattr(e, "__traceback__"):
                logger.error(f"Traceback: {traceback.format_exc()}", emoji="error")
            logger.info("Falling back to generating new USAGE.md", emoji="rocket")
        else:
            logging.error(f"Error enhancing existing USAGE.md: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
            logging.info("Falling back to generating new USAGE.md")
        # Return None to trigger fallback to new generation
        return None


async def progressive_enhance_usage(
    existing_content: str,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enhance USAGE by processing sections individually to avoid timeouts.

    Args:
        existing_content: Current USAGE content
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        related_repo_data: Optional related repository data for context

    Returns:
        str: Progressively enhanced USAGE content
    """
    logger = _get_logger()
    if _use_visual_logging:
        logger.info(
            "Using progressive enhancement strategy for large USAGE",
            emoji="rocket",
        )
    else:
        logging.info("Using progressive enhancement strategy for large USAGE")

    try:
        # Split content into logical sections
        sections = split_usage_sections(existing_content)
        enhanced_sections = []

        # Process each section individually
        for i, section in enumerate(sections):
            section_name = section.get("name", f"Section {i + 1}")
            section_content = section.get("content", "")

            if len(section_content.strip()) < 50:  # Skip very small sections
                enhanced_sections.append(section_content)
                continue

            try:
                if _use_visual_logging:
                    logger.info(
                        f"Enhancing section: {section_name}", emoji="processing"
                    )
                else:
                    logging.info(f"Enhancing section: {section_name}")

                # Enhance this section with reduced context
                enhanced_section = await llm_client.enhance_documentation(
                    existing_content=section_content,
                    file_manifest=file_manifest,
                    doc_type=f"USAGE.md section: {section_name}",
                )
                enhanced_sections.append(enhanced_section)

            except (asyncio.TimeoutError, TimeoutError):
                if _use_visual_logging:
                    logger.warning(
                        f"Section '{section_name}' timed out, keeping original",
                        emoji="warning",
                    )
                else:
                    logging.warning(
                        f"Section '{section_name}' timed out, keeping original"
                    )
                enhanced_sections.append(section_content)

            except Exception as e:
                if _use_visual_logging:
                    logger.warning(
                        f"Error enhancing section '{section_name}': {e}. Keeping original.",
                        emoji="warning",
                    )
                else:
                    logging.warning(
                        f"Error enhancing section '{section_name}': {e}. Keeping original."
                    )
                enhanced_sections.append(section_content)

        # Recombine sections
        enhanced_content = "\n\n".join(enhanced_sections)

        # Apply structural improvements (these are fast operations)
        enhanced_content = ensure_correct_title(enhanced_content, project_name)

        if _use_visual_logging:
            logger.success("Progressive USAGE enhancement completed", emoji="rocket")
        else:
            logging.info("Progressive USAGE enhancement completed")

        return enhanced_content

    except Exception as e:
        if _use_visual_logging:
            logger.error(
                f"Progressive enhancement failed: {e}. Returning original content.",
                emoji="error",
            )
        else:
            logging.error(
                f"Progressive enhancement failed: {e}. Returning original content."
            )
        return existing_content


def split_usage_sections(content: str) -> List[Dict[str, str]]:
    """
    Split USAGE content into logical sections based on headers.

    Args:
        content: USAGE content to split

    Returns:
        List of dictionaries with 'name' and 'content' keys
    """
    sections = []
    lines = content.split("\n")
    current_section = {"name": "Header", "content": ""}

    for line in lines:
        # Check if this line is a header (markdown)
        if line.strip().startswith("#"):
            # Save previous section if it has content
            if current_section["content"].strip():
                sections.append(current_section)

            # Start new section
            header_text = line.strip().lstrip("#").strip()
            current_section = {
                "name": header_text or "Unnamed Section",
                "content": line,
            }
        else:
            # Add line to current section
            current_section["content"] += "\n" + line

    # Add the final section
    if current_section["content"].strip():
        sections.append(current_section)

    return sections


async def generate_new_usage(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    config: ScribeConfig,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a new USAGE.md from scratch.

    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        config: Configuration
        related_repo_data: Optional related repository data for context

    Returns:
        str: Generated USAGE.md content
    """
    # Generate usage guide content
    usage = await generate_usage_content(
        llm_client,
        file_manifest,
        CONTENT_THRESHOLDS["usage_guide_length"],
        f"# {project_name} Usage\n\nPlease refer to the project documentation for usage instructions.",
        related_repo_data,
    )

    # Ensure the document has a proper title
    if not usage.strip().startswith("# "):
        usage = f"# {project_name} Usage\n\n{usage}"

    # Add remote configuration section if related repositories exist
    if related_repo_data:
        remote_config_section = "\n\n## Configuration\n\n"
        if related_repo_data.get("config"):
            remote_config_section += "Configuration for this service is managed in a separate repository. See [CONFIG.md](CONFIG.md) for details on configuration management and environment-specific settings.\n\n"
        if related_repo_data.get("gitops"):
            remote_config_section += "Deployment configurations and GitOps workflows are managed in a separate repository. See [CICD.md](CICD.md) for details on deployment processes and CI/CD pipelines.\n\n"
        if related_repo_data.get("config") or related_repo_data.get("gitops"):
            remote_config_section += "Ensure you have access to these repositories for complete operational context."
            usage += remote_config_section

    # Validate and improve the content
    return await validate_and_improve_content(usage, repo_path)


async def generate_usage_content(
    llm_client: BaseLLMClient,
    file_manifest: dict,
    min_length: int,
    fallback_text: str,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate usage guide content with error handling.

    Args:
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        min_length: Minimum acceptable length for the content
        fallback_text: Text to use if generation fails
        related_repo_data: Optional related repository data for context

    Returns:
        str: Generated usage guide content
    """
    try:
        # Call the LLM client's generate_usage_guide method
        content = await llm_client.generate_usage_guide(file_manifest)

        if not content or len(content) < min_length:
            return fallback_text
        return content
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Error generating usage guide: {e}", emoji="error")
            logger.error(f"Exception type: {type(e)}", emoji="error")
            logger.error(
                f"Exception traceback: {traceback.format_exc()}", emoji="error"
            )
        else:
            logging.error(f"Error generating usage guide: {e}")
            logging.error(f"Exception type: {type(e)}")
            logging.error(f"Exception traceback: {traceback.format_exc()}")
        return fallback_text


def ensure_correct_title(content: str, project_name: str) -> str:
    """
    Ensure the USAGE.md has the correct project name in the title.

    Args:
        content: USAGE.md content
        project_name: Name of the project

    Returns:
        str: Content with corrected title
    """
    import re

    title_match = re.search(r"^# (.+?)(?:\n|$)", content)
    if title_match:
        old_title = title_match.group(1)
        if "usage" not in old_title.lower() or "project" in old_title.lower():
            return content.replace(f"# {old_title}", f"# {project_name} Usage")
    else:
        # No title found, add one
        return f"# {project_name} Usage\n\n{content}"

    return content


async def validate_and_improve_content(content: str, repo_path: Path) -> str:
    """
    Validate markdown structure and links, check readability, and apply fixes.

    Args:
        content: USAGE.md content to validate
        repo_path: Path to the repository

    Returns:
        str: Improved content
    """
    # Validate markdown structure and links
    md_validator = MarkdownValidator(content)
    validation_issues = await md_validator.validate_with_link_checking(repo_path)

    # Log markdown and link issues
    if validation_issues:
        logger = _get_logger()
        for issue in validation_issues:
            if _use_visual_logging:
                logger.warning(f"Validation issue: {issue}", emoji="warning")
            else:
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
    score = scorer.analyze_text(content, "USAGE")

    threshold = CONTENT_THRESHOLDS["readability_score_threshold"]

    # Get logger once at the start of the function
    logger = _get_logger()

    if isinstance(score, dict):
        # Extract the overall score from the dictionary
        overall_score = score.get("overall", 0)
        if overall_score > threshold:  # Higher score means more complex text
            if hasattr(logger, "warning"):
                logger.warning("USAGE.md content may be too complex.", emoji="warning")
                logger.info(
                    "Consider simplifying language for better readability", emoji="info"
                )
                logger.info("Use simpler words where possible", emoji="info")
            else:
                logging.warning("USAGE.md content may be too complex.")
    elif score > threshold:  # For backward compatibility if score is a number
        if hasattr(logger, "warning"):
            logger.warning("USAGE.md content may be too complex.", emoji="warning")
            logger.info(
                "Consider simplifying language for better readability", emoji="info"
            )
            logger.info("Use simpler words where possible", emoji="info")
        else:
            logging.warning("USAGE.md content may be too complex.")


def generate_fallback_usage(repo_path: Path) -> str:
    """
    Generate a minimal valid USAGE.md in case of errors.

    Args:
        repo_path: Path to the repository

    Returns:
        str: Minimal USAGE.md content
    """
    return f"""# {repo_path.name} Usage

This guide provides instructions for using {repo_path.name}.

## Overview

{repo_path.name} provides tools and functionality to help you accomplish your goals.
This document covers basic usage, common operations, and configuration options.

## Basic Usage

### Getting Started

After installation, you can start using {repo_path.name} with the following basic commands:

```bash
# Basic command structure
{repo_path.name.lower()} [command] [options]

# Display help information
{repo_path.name.lower()} --help

# Display version information
{repo_path.name.lower()} --version
```

### Common Commands

Here are the most commonly used commands:

```bash
# Command 1: [Description]
{repo_path.name.lower()} command1 [options]

# Command 2: [Description]
{repo_path.name.lower()} command2 [options]

# Command 3: [Description]
{repo_path.name.lower()} command3 [options]
```

## Common Operations

### Operation 1

Description of common operation 1.

```bash
# Example command
{repo_path.name.lower()} operation1 --param value
```

### Operation 2

Description of common operation 2.

```bash
# Example command
{repo_path.name.lower()} operation2 --param value
```

## Configuration

{repo_path.name} can be configured using configuration files or environment variables.

### Configuration File

Create a configuration file at the appropriate location:

```bash
# Create configuration file
cp config.example config.yaml

# Edit configuration
vim config.yaml
```

### Environment Variables

Common environment variables:

- `{repo_path.name.upper()}_CONFIG` - Path to configuration file
- `{repo_path.name.upper()}_DEBUG` - Enable debug mode
- `{repo_path.name.upper()}_LOG_LEVEL` - Set logging level

## Examples

### Example 1: Basic Usage

```bash
# Basic usage example
{repo_path.name.lower()} command --option value
```

### Example 2: Advanced Configuration

```bash
# Advanced usage with configuration
{repo_path.name.lower()} command --config custom.yaml --verbose
```

### Example 3: Common Workflow

```bash
# Step 1: Initialize
{repo_path.name.lower()} init

# Step 2: Configure
{repo_path.name.lower()} configure --preset production

# Step 3: Execute
{repo_path.name.lower()} run
```

## Next Steps

After familiarizing yourself with basic usage:

- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions
- Check [README.md](../README.md) for project overview and additional resources
- See [CONTRIBUTING.md](../CONTRIBUTING.md) if you'd like to contribute

---
*This USAGE.md was automatically generated.*
"""
