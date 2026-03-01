#!/usr/bin/env python3

"""
README refactoring utilities for documentation splitting.

This module provides functions to refactor README.md files when content is
migrated to separate documentation files (INSTALLATION.md, USAGE.md, etc.).
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

# Try to import visual logger, fallback to regular logging
try:
    from .visual_logger import get_visual_logger

    _use_visual_logging = True
except ImportError:
    _use_visual_logging = False

# Import section manipulation utilities
from .doc_utils import insert_link_at_section, extract_section_by_name


def _get_logger():
    """Get appropriate logger (visual or standard)."""
    if _use_visual_logging:
        return get_visual_logger()
    return logging.getLogger(__name__)


def refactor_readme_for_split_docs(
    repo_path: Path,
    sections_to_migrate: List[Dict[str, Any]],
    config: Any,
) -> None:
    """Refactor README.md to replace migrated sections with links.

    This function modifies README.md by:
    1. For each migrated section, replacing full content with brief overview + link
    2. Preserving the heading and first 2-3 lines (if keep_brief_overview=True)
    3. Adding a link to the new separate documentation file

    Args:
        repo_path: Path to repository root
        sections_to_migrate: List of dicts with keys:
            - section_names: List[str] - possible section names to find
            - doc_path: str - path to new doc (e.g., "docs/INSTALLATION.md")
            - doc_title: str - title for the link (e.g., "INSTALLATION.md")
        config: ScribeConfig object with readme_refactor settings

    Example section_to_migrate:
        {
            "section_names": ["Installation", "Setup", "Getting Started"],
            "doc_path": "docs/INSTALLATION.md",
            "doc_title": "INSTALLATION.md"
        }
    """
    logger = _get_logger()

    readme_path = repo_path / "README.md"

    if not readme_path.exists():
        logger.warning("README.md not found, cannot refactor")
        return

    # Read README content
    try:
        readme_content = readme_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading README.md: {e}")
        return

    # Get config settings
    keep_brief_overview = config.readme_refactor.keep_brief_overview

    # Process each section to migrate
    modified = False
    for section_info in sections_to_migrate:
        section_names = section_info["section_names"]
        doc_path = section_info["doc_path"]
        doc_title = section_info.get("doc_title", doc_path.split("/")[-1])

        # Try to find which section name exists in README
        found_section = None
        for section_name in section_names:
            # Check if section exists (try both level 2 and level 3 headings)
            for heading_level in [2, 3]:
                section_content = extract_section_by_name(
                    readme_content, section_name, heading_level
                )
                if section_content is not None:
                    found_section = (section_name, heading_level)
                    break
            if found_section:
                break

        if not found_section:
            logger.debug(
                f"None of the section names {section_names} found in README, "
                "skipping refactor for this section"
            )
            continue

        section_name, heading_level = found_section
        logger.info(
            f"Refactoring section '{section_name}' (level {heading_level}) "
            f"to link to {doc_title}"
        )

        # Replace section with brief overview + link
        link_text = f"detailed {doc_title.replace('.md', '').lower()}"
        readme_content = insert_link_at_section(
            content=readme_content,
            section_name=section_name,
            link_text=link_text,
            link_url=doc_path,
            heading_level=heading_level,
            keep_brief_overview=keep_brief_overview,
        )
        modified = True

    # Write modified README if changes were made
    if modified:
        try:
            readme_path.write_text(readme_content, encoding="utf-8")
            logger.success(
                f"README.md refactored successfully "
                f"({len(sections_to_migrate)} sections processed)"
            )
        except Exception as e:
            logger.error(f"Error writing refactored README.md: {e}")
    else:
        logger.info("No sections found in README to refactor")


def add_navigation_section_to_readme(repo_path: Path, doc_links: List[str]) -> None:
    """Add a Documentation navigation section to README.md.

    This function adds a "Documentation" section near the top of README.md
    (after title and badges) with links to all generated documentation files.

    Args:
        repo_path: Path to repository root
        doc_links: List of documentation file paths (e.g., ["docs/INSTALLATION.md"])

    Example output:
        ## Documentation

        - [Installation Guide](docs/INSTALLATION.md)
        - [Usage Guide](docs/USAGE.md)
        - [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
        - [Architecture](docs/ARCHITECTURE.md)
        - [Contributing](CONTRIBUTING.md)
    """
    logger = _get_logger()

    readme_path = repo_path / "README.md"

    if not readme_path.exists():
        logger.warning("README.md not found, cannot add navigation section")
        return

    # Read README content
    try:
        readme_content = readme_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading README.md: {e}")
        return

    # Check if Documentation section already exists
    if "## Documentation" in readme_content:
        logger.debug("Documentation section already exists, skipping")
        return

    # Build navigation section
    nav_section = "## Documentation\n\n"

    # Create friendly titles for each doc
    for doc_path in doc_links:
        filename = doc_path.split("/")[-1]
        doc_name = filename.replace(".md", "").replace("_", " ").title()

        # Special case: Standardize common doc names
        if doc_name.upper() == "INSTALLATION":
            doc_name = "Installation Guide"
        elif doc_name.upper() == "USAGE":
            doc_name = "Usage Guide"
        elif doc_name.upper() == "TROUBLESHOOTING":
            doc_name = "Troubleshooting Guide"
        elif doc_name.upper() == "ARCHITECTURE":
            doc_name = "Architecture"
        elif doc_name.upper() == "CONTRIBUTING":
            doc_name = "Contributing Guidelines"
        elif doc_name.upper() == "PERSISTENCE":
            doc_name = "Database & Persistence"
        elif doc_name.upper() == "CONFIG":
            doc_name = "Configuration"
        elif doc_name.upper() == "CICD":
            doc_name = "CI/CD Pipeline"

        nav_section += f"- [{doc_name}]({doc_path})\n"

    # Find insertion point (after title and any badges, before first ## section)
    lines = readme_content.split("\n")

    # Find title line
    title_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("# "):
            title_idx = i
            break

    if title_idx == -1:
        logger.warning("No title found in README, adding navigation at top")
        readme_content = nav_section + "\n\n" + readme_content
    else:
        # Find first ## section after title
        first_section_idx = -1
        for i in range(title_idx + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith("##") and not stripped.startswith("###"):
                first_section_idx = i
                break

        if first_section_idx == -1:
            # No sections found, add at end
            readme_content += "\n\n" + nav_section
        else:
            # Insert before first section
            lines.insert(first_section_idx, "")
            lines.insert(first_section_idx + 1, nav_section.strip())
            lines.insert(first_section_idx + 2, "")
            readme_content = "\n".join(lines)

    # Write modified README
    try:
        readme_path.write_text(readme_content, encoding="utf-8")
        logger.success(
            f"Added Documentation navigation section to README.md "
            f"({len(doc_links)} links)"
        )
    except Exception as e:
        logger.error(f"Error writing README.md with navigation: {e}")


def add_documentation_links_to_contributing(
    repo_path: Path, doc_links: List[str]
) -> None:
    """Add documentation links to CONTRIBUTING.md (optional helper).

    This function can be used to add a "Related Documentation" section to
    CONTRIBUTING.md with links to other generated docs.

    Args:
        repo_path: Path to repository root
        doc_links: List of documentation file paths

    Note: This is optional and not called by default. Can be enabled via config.
    """
    logger = _get_logger()

    contributing_path = repo_path / "CONTRIBUTING.md"

    if not contributing_path.exists():
        logger.debug("CONTRIBUTING.md not found, skipping doc links")
        return

    # Read content
    try:
        content = contributing_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading CONTRIBUTING.md: {e}")
        return

    # Check if section already exists
    if "## Related Documentation" in content:
        logger.debug("Related Documentation section already exists in CONTRIBUTING.md")
        return

    # Build section
    related_docs = "\n\n## Related Documentation\n\n"
    related_docs += "For additional information, see:\n\n"

    for doc_path in doc_links:
        filename = doc_path.split("/")[-1]
        doc_name = filename.replace(".md", "").replace("_", " ").title()
        related_docs += f"- [{doc_name}]({doc_path})\n"

    # Append to end
    content += related_docs

    # Write back
    try:
        contributing_path.write_text(content, encoding="utf-8")
        logger.success("Added Related Documentation to CONTRIBUTING.md")
    except Exception as e:
        logger.error(f"Error writing CONTRIBUTING.md: {e}")
