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
    "troubleshooting_guide_length": 50,  # Minimum length for a valid troubleshooting guide
    "readability_score_threshold": 40,  # Threshold for readability warnings
}


async def generate_troubleshooting(
    repo_path: Path,
    file_manifest: dict,
    llm_client: BaseLLMClient,
    config: ScribeConfig,
    analyzer: CodebaseAnalyzer,
    related_repo_data: Optional[Dict[str, Any]] = None,
    persistence_info: Optional[Any] = None,
) -> str:
    """
    Generate TROUBLESHOOTING.md content.

    Args:
        repo_path: Path to the repository root
        file_manifest: Dictionary of files in the repository
        llm_client: LLM client for generating content
        config: Configuration
        analyzer: CodebaseAnalyzer instance
        related_repo_data: Optional related repository data for context
        persistence_info: Optional persistence layer information

    Returns:
        str: Generated TROUBLESHOOTING.md content
    """
    try:
        # Use the analyzer's method to get a consistent project name
        debug_mode = config.debug
        project_name = analyzer.derive_project_name(debug_mode)
        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                f"Generating troubleshooting guide for: {logger.format_filename(project_name)}",
                emoji="wrench",
            )
        else:
            logging.info(f"Generating troubleshooting guide for: {project_name}")

        # Check if we should enhance existing TROUBLESHOOTING.md or create a new one
        if should_enhance_existing_troubleshooting(repo_path, config):
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "Found existing TROUBLESHOOTING.md with meaningful content. Will enhance rather than replace.",
                    emoji="wrench",
                )
            else:
                logging.info(
                    "Found existing TROUBLESHOOTING.md with meaningful content. Will enhance rather than replace."
                )

            try:
                enhanced_content = await enhance_existing_troubleshooting(
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
                        f"Caught timeout in generate_troubleshooting: {type(e).__name__}: {e}"
                    )
                else:
                    logging.debug(
                        f"Caught timeout in generate_troubleshooting: {type(e).__name__}: {e}"
                    )
                # If timeout occurs at this level, return original content
                logger = _get_logger()
                if _use_visual_logging:
                    logger.warning(
                        "TROUBLESHOOTING enhancement timed out. Using original content",
                        emoji="timer",
                    )
                else:
                    logging.warning(
                        "TROUBLESHOOTING enhancement timed out. Using original content"
                    )

                # Read and return the original TROUBLESHOOTING content
                troubleshooting_path = repo_path / "docs" / "TROUBLESHOOTING.md"
                if troubleshooting_path.exists():
                    return troubleshooting_path.read_text(encoding="utf-8")

            except Exception as e:
                logger = _get_logger()
                if _use_visual_logging:
                    logger.warning(
                        f"TROUBLESHOOTING enhancement failed: {e}. Using original content",
                        emoji="warning",
                    )
                else:
                    logging.warning(
                        f"TROUBLESHOOTING enhancement failed: {e}. Using original content"
                    )

                # Read and return the original TROUBLESHOOTING content
                troubleshooting_path = repo_path / "docs" / "TROUBLESHOOTING.md"
                if troubleshooting_path.exists():
                    return troubleshooting_path.read_text(encoding="utf-8")

            # If we reach here, enhancement failed, so proceed to generate new content
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "Enhancement failed, proceeding to generate new TROUBLESHOOTING.md",
                    emoji="wrench",
                )
            else:
                logging.info(
                    "Enhancement failed, proceeding to generate new TROUBLESHOOTING.md"
                )

        # Generate new TROUBLESHOOTING.md from scratch
        return await generate_new_troubleshooting(
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
            logger.error(f"Error generating TROUBLESHOOTING.md: {e}", emoji="error")
            logger.error(f"Exception type: {type(e).__name__}", emoji="error")
            logger.error(f"Exception details: {str(e)}", emoji="error")
            if hasattr(e, "__traceback__"):
                logger.error(f"Traceback: {traceback.format_exc()}", emoji="error")
            logger.info(
                "Falling back to minimal TROUBLESHOOTING.md template", emoji="template"
            )
        else:
            logging.error(f"Error generating TROUBLESHOOTING.md: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            logging.error(f"Exception details: {str(e)}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
            logging.info("Falling back to minimal TROUBLESHOOTING.md template")
        return generate_fallback_troubleshooting(repo_path)


def extract_existing_content_from_sources(
    repo_path: Path, section_names: List[str]
) -> str:
    """
    Extract existing troubleshooting-related content from README.md or CONTRIBUTING.md.

    Args:
        repo_path: Path to the repository root
        section_names: List of section names to extract (e.g., ["Troubleshooting", "Common Issues"])

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


def should_enhance_existing_troubleshooting(
    repo_path: Path, config: ScribeConfig
) -> bool:
    """
    Determine if we should enhance an existing TROUBLESHOOTING.md.

    Args:
        repo_path: Path to the repository
        config: Configuration

    Returns:
        bool: True if we should enhance existing TROUBLESHOOTING.md
    """
    # Check if we should preserve existing content
    preserve_existing = config.preserve_existing
    troubleshooting_path = repo_path / "docs" / "TROUBLESHOOTING.md"
    # Check if TROUBLESHOOTING.md exists and preserve_existing is True
    if not (preserve_existing and troubleshooting_path.exists()):
        return False

    try:
        existing_content = troubleshooting_path.read_text(encoding="utf-8")
        # Check if it's not just a placeholder or default TROUBLESHOOTING
        return len(existing_content.strip().split("\n")) > 5
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(
                f"Error reading existing TROUBLESHOOTING.md: {e}", emoji="error"
            )
        else:
            logging.error(f"Error reading existing TROUBLESHOOTING.md: {e}")
        return False


async def enhance_existing_troubleshooting(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enhance an existing TROUBLESHOOTING.md file.

    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        related_repo_data: Optional related repository data for context

    Returns:
        str: Enhanced TROUBLESHOOTING.md content
    """
    logger = _get_logger()
    if _use_visual_logging:
        logger.info(
            "Found existing TROUBLESHOOTING.md with meaningful content. Will enhance rather than replace.",
            emoji="wrench",
        )
    else:
        logging.info(
            "Found existing TROUBLESHOOTING.md with meaningful content. Will enhance rather than replace."
        )

    try:
        # Read existing content
        troubleshooting_path = repo_path / "docs" / "TROUBLESHOOTING.md"
        existing_content = troubleshooting_path.read_text(encoding="utf-8")

        try:
            # Attempt standard enhancement
            enhanced_content = await llm_client.enhance_documentation(
                existing_content=existing_content,
                file_manifest=file_manifest,
                doc_type="TROUBLESHOOTING.md",
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
                        f"TROUBLESHOOTING enhancement timed out for large file ({file_size_kb:.1f}KB). Using original content.",
                        emoji="timer",
                    )
                else:
                    logging.warning(
                        f"TROUBLESHOOTING enhancement timed out for large file ({file_size_kb:.1f}KB). Using original content."
                    )
                enhanced_content = existing_content
            else:
                # This is some other error
                if _use_visual_logging:
                    logger.warning(
                        f"TROUBLESHOOTING enhancement failed: {e}. Using original content.",
                        emoji="warning",
                    )
                else:
                    logging.warning(
                        f"TROUBLESHOOTING enhancement failed: {e}. Using original content."
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
                        f"TROUBLESHOOTING.md enhanced with changes: {enhancement_log}",
                        emoji="wrench",
                    )
                else:
                    logging.info(
                        f"TROUBLESHOOTING.md enhanced with changes: {enhancement_log}"
                    )
            else:
                logger = _get_logger()
                if _use_visual_logging:
                    logger.success(
                        f"TROUBLESHOOTING.md enhanced (size change: {changes['size_change_bytes']} bytes)",
                        emoji="wrench",
                    )
                else:
                    logging.info(
                        f"TROUBLESHOOTING.md enhanced (size change: {changes['size_change_bytes']} bytes)"
                    )
        else:
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "TROUBLESHOOTING.md: No meaningful changes made by enhancement",
                    emoji="info",
                )
            else:
                logging.info(
                    "TROUBLESHOOTING.md: No meaningful changes made by enhancement"
                )

        # Ensure correct project name in title
        enhanced_content = ensure_correct_title(enhanced_content, project_name)

        return enhanced_content
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(
                f"Error enhancing existing TROUBLESHOOTING.md: {e}", emoji="error"
            )
            logger.error(f"Exception type: {type(e).__name__}", emoji="error")
            if hasattr(e, "__traceback__"):
                logger.error(f"Traceback: {traceback.format_exc()}", emoji="error")
            logger.info(
                "Falling back to generating new TROUBLESHOOTING.md", emoji="wrench"
            )
        else:
            logging.error(f"Error enhancing existing TROUBLESHOOTING.md: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            if hasattr(e, "__traceback__"):
                logging.error(f"Traceback: {traceback.format_exc()}")
            logging.info("Falling back to generating new TROUBLESHOOTING.md")
        # Return None to trigger fallback to new generation
        return None


async def progressive_enhance_troubleshooting(
    existing_content: str,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enhance TROUBLESHOOTING by processing sections individually to avoid timeouts.

    Args:
        existing_content: Current TROUBLESHOOTING content
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        related_repo_data: Optional related repository data for context

    Returns:
        str: Progressively enhanced TROUBLESHOOTING content
    """
    logger = _get_logger()
    if _use_visual_logging:
        logger.info(
            "Using progressive enhancement strategy for large TROUBLESHOOTING",
            emoji="wrench",
        )
    else:
        logging.info("Using progressive enhancement strategy for large TROUBLESHOOTING")

    try:
        # Split content into logical sections
        sections = split_troubleshooting_sections(existing_content)
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
                    doc_type=f"TROUBLESHOOTING.md section: {section_name}",
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
            logger.success(
                "Progressive TROUBLESHOOTING enhancement completed", emoji="wrench"
            )
        else:
            logging.info("Progressive TROUBLESHOOTING enhancement completed")

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


def split_troubleshooting_sections(content: str) -> List[Dict[str, str]]:
    """
    Split TROUBLESHOOTING content into logical sections based on headers.

    Args:
        content: TROUBLESHOOTING content to split

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


async def generate_new_troubleshooting(
    repo_path: Path,
    llm_client: BaseLLMClient,
    file_manifest: dict,
    project_name: str,
    config: ScribeConfig,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a new TROUBLESHOOTING.md from scratch.

    Args:
        repo_path: Path to the repository
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        project_name: Name of the project
        config: Configuration
        related_repo_data: Optional related repository data for context

    Returns:
        str: Generated TROUBLESHOOTING.md content
    """
    # Generate troubleshooting guide content
    troubleshooting = await generate_troubleshooting_content(
        llm_client,
        file_manifest,
        CONTENT_THRESHOLDS["troubleshooting_guide_length"],
        f"# {project_name} Troubleshooting\n\nPlease refer to the project documentation for troubleshooting guidance.",
        related_repo_data,
    )

    # Ensure the document has a proper title
    if not troubleshooting.strip().startswith("# "):
        troubleshooting = f"# {project_name} Troubleshooting\n\n{troubleshooting}"

    # Add remote configuration section if related repositories exist
    if related_repo_data:
        remote_config_section = "\n\n## Configuration Issues\n\n"
        if related_repo_data.get("config"):
            remote_config_section += "If experiencing configuration-related problems, check [CONFIG.md](CONFIG.md) for details on configuration management and environment-specific settings.\n\n"
        if related_repo_data.get("gitops"):
            remote_config_section += "For deployment and CI/CD pipeline issues, refer to [CICD.md](CICD.md) for troubleshooting deployment processes.\n\n"
        if related_repo_data.get("config") or related_repo_data.get("gitops"):
            remote_config_section += "Ensure you have access to these repositories and correct configuration values."
            troubleshooting += remote_config_section

    # Validate and improve the content
    return await validate_and_improve_content(troubleshooting, repo_path)


async def generate_troubleshooting_content(
    llm_client: BaseLLMClient,
    file_manifest: dict,
    min_length: int,
    fallback_text: str,
    related_repo_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate troubleshooting guide content with error handling.

    Args:
        llm_client: LLM client for generating content
        file_manifest: Dictionary of files in the repository
        min_length: Minimum acceptable length for the content
        fallback_text: Text to use if generation fails
        related_repo_data: Optional related repository data for context

    Returns:
        str: Generated troubleshooting guide content
    """
    try:
        # Call the LLM client's generate_troubleshooting_guide method
        content = await llm_client.generate_troubleshooting_guide(file_manifest)

        if not content or len(content) < min_length:
            return fallback_text
        return content
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Error generating troubleshooting guide: {e}", emoji="error")
            logger.error(f"Exception type: {type(e)}", emoji="error")
            logger.error(
                f"Exception traceback: {traceback.format_exc()}", emoji="error"
            )
        else:
            logging.error(f"Error generating troubleshooting guide: {e}")
            logging.error(f"Exception type: {type(e)}")
            logging.error(f"Exception traceback: {traceback.format_exc()}")
        return fallback_text


def ensure_correct_title(content: str, project_name: str) -> str:
    """
    Ensure the TROUBLESHOOTING.md has the correct project name in the title.

    Args:
        content: TROUBLESHOOTING.md content
        project_name: Name of the project

    Returns:
        str: Content with corrected title
    """
    import re

    title_match = re.search(r"^# (.+?)(?:\n|$)", content)
    if title_match:
        old_title = title_match.group(1)
        if "troubleshooting" not in old_title.lower() or "project" in old_title.lower():
            return content.replace(
                f"# {old_title}", f"# {project_name} Troubleshooting"
            )
    else:
        # No title found, add one
        return f"# {project_name} Troubleshooting\n\n{content}"

    return content


async def validate_and_improve_content(content: str, repo_path: Path) -> str:
    """
    Validate markdown structure and links, check readability, and apply fixes.

    Args:
        content: TROUBLESHOOTING.md content to validate
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
    score = scorer.analyze_text(content, "TROUBLESHOOTING")

    threshold = CONTENT_THRESHOLDS["readability_score_threshold"]

    # Get logger once at the start of the function
    logger = _get_logger()

    if isinstance(score, dict):
        # Extract the overall score from the dictionary
        overall_score = score.get("overall", 0)
        if overall_score > threshold:  # Higher score means more complex text
            if hasattr(logger, "warning"):
                logger.warning(
                    "TROUBLESHOOTING.md content may be too complex.", emoji="warning"
                )
                logger.info(
                    "Consider simplifying language for better readability", emoji="info"
                )
                logger.info("Use simpler words where possible", emoji="info")
            else:
                logging.warning("TROUBLESHOOTING.md content may be too complex.")
    elif score > threshold:  # For backward compatibility if score is a number
        if hasattr(logger, "warning"):
            logger.warning(
                "TROUBLESHOOTING.md content may be too complex.", emoji="warning"
            )
            logger.info(
                "Consider simplifying language for better readability", emoji="info"
            )
            logger.info("Use simpler words where possible", emoji="info")
        else:
            logging.warning("TROUBLESHOOTING.md content may be too complex.")


def generate_fallback_troubleshooting(repo_path: Path) -> str:
    """
    Generate a minimal valid TROUBLESHOOTING.md in case of errors.

    Args:
        repo_path: Path to the repository

    Returns:
        str: Minimal TROUBLESHOOTING.md content
    """
    return f"""# {repo_path.name} Troubleshooting

This guide provides solutions to common problems and debugging techniques for {repo_path.name}.

## Common Issues

### Issue: Application Won't Start

**Symptoms:**
- Application fails to launch
- Error messages on startup
- Process crashes immediately

**Solutions:**
1. Check that all dependencies are installed correctly
2. Verify configuration files are present and valid
3. Review logs for specific error messages
4. Ensure required services are running

### Issue: Connection Errors

**Symptoms:**
- Cannot connect to database
- API requests failing
- Network timeouts

**Solutions:**
1. Verify network connectivity
2. Check service endpoints and ports
3. Review firewall and security group settings
4. Validate connection credentials

### Issue: Performance Problems

**Symptoms:**
- Slow response times
- High memory usage
- CPU utilization spikes

**Solutions:**
1. Check system resources (CPU, memory, disk)
2. Review application logs for bottlenecks
3. Analyze database query performance
4. Consider scaling resources if needed

## Debugging

### Enable Debug Mode

To enable detailed logging and debugging information:

```bash
# Set environment variable
export DEBUG=true

# Or configure in your settings file
debug: true
```

### View Logs

Access application logs to diagnose issues:

```bash
# View recent logs
tail -f logs/application.log

# Search for errors
grep -i error logs/application.log
```

### Check System Status

Verify system components are functioning:

```bash
# Check service status
# Add relevant status commands for your application

# Verify connectivity
# Add relevant connectivity checks
```

## Configuration Problems

### Invalid Configuration

If you encounter configuration errors:

1. Verify all required configuration values are set
2. Check for typos in configuration keys
3. Ensure values match expected formats (e.g., URLs, numbers)
4. Review configuration file syntax (YAML, JSON, etc.)

### Environment Variables

Common environment variable issues:

- Missing required variables
- Incorrect variable names
- Wrong variable values
- Variables not exported properly

### Configuration Files

Check configuration file issues:

- File not found or wrong location
- Invalid syntax or formatting
- Missing required sections
- Incorrect file permissions

## Performance Issues

### High Memory Usage

If experiencing memory problems:

1. Monitor memory consumption over time
2. Check for memory leaks
3. Review caching configurations
4. Consider adjusting memory limits

### Slow Performance

For performance degradation:

1. Profile application execution
2. Identify slow operations or queries
3. Check for network latency
4. Review resource allocation

### Database Performance

For database-related issues:

1. Analyze slow queries
2. Check index usage
3. Review connection pool settings
4. Monitor database load

## Error Messages

### Common Error Messages

**"Configuration file not found"**
- Verify configuration file exists in expected location
- Check file name and path
- Ensure file has correct permissions

**"Connection refused"**
- Verify service is running
- Check port numbers
- Review firewall settings
- Validate network connectivity

**"Authentication failed"**
- Verify credentials are correct
- Check token expiration
- Review permission settings
- Ensure user account is active

**"Resource not found"**
- Verify resource exists
- Check path or identifier
- Review access permissions
- Confirm correct environment

## Getting Help

If you continue experiencing issues:

1. **Check Documentation**: Review the full documentation in [README.md](../README.md)
2. **Search Issues**: Look for similar problems in the project issue tracker
3. **Create an Issue**: Report new bugs or problems with detailed information
4. **Contact Support**: Reach out to the development team or community
5. **Provide Details**: Include error messages, logs, and steps to reproduce

### Information to Include

When reporting issues, provide:

- Operating system and version
- Application version
- Configuration details (without sensitive data)
- Error messages and stack traces
- Steps to reproduce the problem
- Expected vs actual behavior

## Additional Resources

- [INSTALLATION.md](INSTALLATION.md) - Installation and setup instructions
- [USAGE.md](USAGE.md) - Usage documentation and examples
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [README.md](../README.md) - Project overview

---
*This TROUBLESHOOTING.md was automatically generated.*
"""
