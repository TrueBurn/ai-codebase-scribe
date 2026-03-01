#!/usr/bin/env python3

"""
Utilities for documentation generation and manipulation.
"""

import logging
import re
from typing import List, Optional, Tuple

# Try to import visual logger, fallback to regular logging
try:
    from .visual_logger import get_visual_logger

    _use_visual_logging = True
except ImportError:
    _use_visual_logging = False


def _get_logger():
    """Get appropriate logger (visual or standard)."""
    if _use_visual_logging:
        return get_visual_logger()
    return logging.getLogger(__name__)


def detect_existing_badges(content: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """Detect existing badge sections in markdown content using comprehensive regex patterns.

    Args:
        content: The markdown content to search

    Returns:
        Tuple of (has_badges, (start_pos, end_pos)) where positions indicate badge section location
    """
    # Pattern to match badge lines - looks for lines containing multiple ![...](...)
    # This handles both single-line and multi-line badge sections
    badge_pattern = r"^[ \t]*(?:(?:!\[[^\]]*\]\([^)]+\)[ \t]*){1,}[ \t]*(?:\n|$))+"

    # Search for badge patterns in the content
    matches = list(re.finditer(badge_pattern, content, re.MULTILINE))

    if matches:
        # Find the first badge section after the title
        lines = content.split("\n")
        title_line = -1

        # Find the first heading
        for i, line in enumerate(lines):
            if line.strip().startswith("# "):
                title_line = i
                break

        if title_line >= 0:
            # Look for badges after the title
            for match in matches:
                match_start_line = content[: match.start()].count("\n")
                if match_start_line > title_line:
                    end_line = content[: match.end()].count("\n")
                    logging.debug(
                        f"Found existing badges at lines {match_start_line}-{end_line}"
                    )
                    return True, (match.start(), match.end())

    # Also check for common badge patterns that might not be caught by the main regex
    simple_badge_pattern = r"!\[[^\]]*\]\([^)]+\)"
    badge_matches = re.findall(simple_badge_pattern, content)

    if len(badge_matches) >= 2:  # Multiple badges likely indicate a badge section
        logging.debug(f"Found {len(badge_matches)} badge patterns in content")
        return True, None

    return False, None


def remove_existing_badge_section(content: str) -> str:
    """Remove existing badge sections from markdown content.

    Args:
        content: The markdown content to clean

    Returns:
        Content with badge sections removed
    """
    has_badges, badge_location = detect_existing_badges(content)

    if not has_badges or not badge_location:
        return content

    start_pos, end_pos = badge_location

    # Remove the badge section and any extra whitespace
    before_badges = content[:start_pos].rstrip()
    after_badges = content[end_pos:].lstrip()

    # Ensure proper spacing
    if before_badges and after_badges:
        cleaned_content = before_badges + "\n\n" + after_badges
    else:
        cleaned_content = before_badges + after_badges

    logger = _get_logger()
    if _use_visual_logging:
        logger.info(
            "Removed existing badge section to prevent duplication", emoji="badge"
        )
    else:
        logging.info("Removed existing badge section to prevent duplication")
    return cleaned_content


def insert_badges_after_title(content: str, badges: str) -> str:
    """Insert badges after the first heading in markdown content.

    Args:
        content: The markdown content
        badges: The badge string to insert

    Returns:
        Content with badges inserted after the title
    """
    if not badges or not badges.strip():
        return content

    # First, remove any existing badges to prevent duplication
    content = remove_existing_badge_section(content)

    lines = content.split("\n")

    # Find the first heading
    for i, line in enumerate(lines):
        if line.strip().startswith("# "):
            # Insert badges after the first heading with proper spacing
            lines.insert(i + 1, "")
            lines.insert(i + 2, badges.strip())
            lines.insert(i + 3, "")

            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    f"Inserted badges after title at line {i + 1}", emoji="badge"
                )
            else:
                logging.info(f"Inserted badges after title at line {i + 1}")
            return "\n".join(lines)

    # If no heading found, add badges at the beginning
    logger = _get_logger()
    if _use_visual_logging:
        logger.warning(
            "No title heading found, adding badges at the beginning", emoji="warning"
        )
    else:
        logging.warning("No title heading found, adding badges at the beginning")
    return badges.strip() + "\n\n" + content


def add_ai_attribution(
    content: str, doc_type: str = "documentation", badges: str = ""
) -> str:
    """Add AI attribution footer and badges to generated content if not already present.

    Args:
        content: The content to add attribution to
        doc_type: The type of document (e.g., "README", "ARCHITECTURE.md")
        badges: Badges to add to the document

    Returns:
        The content with attribution and badges added
    """
    attribution_text = f"\n\n---\n_This {doc_type} was generated using AI analysis and may contain inaccuracies. Please verify critical information._"

    # Check if content already has an attribution footer
    has_attribution = (
        "_This " in content
        and ("generated" in content.lower() or "enhanced" in content.lower())
        and "AI" in content
    )

    if has_attribution:
        # Already has attribution, just add badges if needed using comprehensive detection
        if badges:
            has_existing_badges, _ = detect_existing_badges(content)
            if not has_existing_badges:
                content = insert_badges_after_title(content, badges)
                logger = _get_logger()
                if _use_visual_logging:
                    logger.info(
                        "Added badges to content with existing attribution",
                        emoji="badge",
                    )
                else:
                    logging.info("Added badges to content with existing attribution")
        return content

    # Add badges after the title if provided (this handles deduplication internally)
    if badges:
        content = insert_badges_after_title(content, badges)

    # Add the attribution
    return content + attribution_text


def format_anchor_link(section_name: str) -> str:
    """
    Format a section name into a proper anchor link by removing special characters.
    This follows GitHub's anchor generation rules for maximum compatibility.

    This function is centralized to ensure consistent anchor link generation
    across all documentation generators (README, Architecture, Contributing, etc.).

    Args:
        section_name: Name of the section

    Returns:
        str: Formatted anchor link compatible with GitHub's anchor generation

    Example:
        >>> format_anchor_link("SECTION 2: PROJECT STRUCTURE ANALYSIS")
        'section-2-project-structure-analysis'
        >>> format_anchor_link("Usage & Installation")
        'usage--installation'
    """
    return (
        section_name.lower()
        .replace(" ", "-")
        .replace(".", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace("&", "")
        .replace("#", "")
    )


# ============================================================================
# Section Manipulation Utilities (NEW - for docs splitting)
# ============================================================================


def extract_section_by_name(
    content: str, section_name: str, heading_level: int = 2
) -> Optional[str]:
    """Extract a specific section's content by name.

    This function finds a section heading and extracts all content until the
    next heading of the same or higher level (or end of document).

    Args:
        content: Markdown content to search
        section_name: Name of the section to extract (case-insensitive)
        heading_level: Heading level to match (2 for ##, 3 for ###, etc.)

    Returns:
        Optional[str]: Section content (without heading), or None if not found

    Example:
        >>> content = "# Title\\n\\n## Installation\\nRun npm install\\n\\n## Usage\\nRun npm start"
        >>> extract_section_by_name(content, "Installation", heading_level=2)
        'Run npm install'
    """
    logger = _get_logger()

    # Build regex pattern for the heading
    heading_marker = "#" * heading_level
    # Match heading with section name (case-insensitive)
    # Capture content until next heading of same/higher level or EOF
    # Build the pattern without f-string to avoid brace conflicts
    pattern = (
        r"^"
        + heading_marker
        + r"\s+"
        + re.escape(section_name)
        + r"\s*\n(.*?)(?=^#{1,"
        + str(heading_level)
        + r"}\s|\Z)"
    )

    match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)

    if match:
        extracted_content = match.group(1).strip()
        logger.debug(
            f"Extracted section '{section_name}' (level {heading_level}): "
            f"{len(extracted_content)} characters"
        )
        return extracted_content

    logger.debug(f"Section '{section_name}' not found (level {heading_level})")
    return None


def remove_section_by_name(
    content: str, section_name: str, heading_level: int = 2
) -> str:
    """Remove a specific section by name.

    This function finds and removes a section heading and all its content
    until the next heading of the same or higher level.

    Args:
        content: Markdown content
        section_name: Name of the section to remove (case-insensitive)
        heading_level: Heading level to match (2 for ##, 3 for ###, etc.)

    Returns:
        str: Content with section removed

    Example:
        >>> content = "## Installation\\nRun npm install\\n\\n## Usage\\nRun npm start"
        >>> remove_section_by_name(content, "Installation", heading_level=2)
        '## Usage\\nRun npm start'
    """
    logger = _get_logger()

    # Build regex pattern to match entire section (heading + content)
    heading_marker = "#" * heading_level
    # Build the pattern without f-string to avoid brace conflicts
    pattern = (
        r"^"
        + heading_marker
        + r"\s+"
        + re.escape(section_name)
        + r"\s*\n.*?(?=^#{1,"
        + str(heading_level)
        + r"}\s|\Z)"
    )

    # Check if section exists
    if not re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE):
        logger.debug(f"Section '{section_name}' not found, nothing to remove")
        return content

    # Remove the section
    modified_content = re.sub(
        pattern, "", content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    # Clean up excessive whitespace
    # Replace multiple consecutive newlines with max 2
    modified_content = re.sub(r"\n{3,}", "\n\n", modified_content)

    # Clean up whitespace at beginning and end
    modified_content = modified_content.strip()

    logger.debug(f"Removed section '{section_name}' (level {heading_level})")
    return modified_content


def insert_link_at_section(
    content: str,
    section_name: str,
    link_text: str,
    link_url: str,
    heading_level: int = 2,
    keep_brief_overview: bool = True,
) -> str:
    """Replace a section with brief overview and link to separate doc.

    This function finds a section and either:
    1. Keeps first 2-3 lines + adds link (if keep_brief_overview=True)
    2. Replaces entire section with heading + link (if keep_brief_overview=False)

    Args:
        content: Markdown content
        section_name: Name of the section to replace
        link_text: Text for the link (e.g., "detailed installation instructions")
        link_url: URL for the link (e.g., "docs/INSTALLATION.md")
        heading_level: Heading level to match (2 for ##, 3 for ###, etc.)
        keep_brief_overview: If True, keep first 2-3 lines of section

    Returns:
        str: Content with section replaced

    Example:
        >>> content = "## Installation\\nPrereqs: Node 18+\\nRun npm install\\nVerify with npm --version"
        >>> insert_link_at_section(content, "Installation", "detailed guide", "docs/INSTALL.md")
        '## Installation\\nPrereqs: Node 18+\\n\\nFor detailed guide, see [INSTALL.md](docs/INSTALL.md).'
    """
    logger = _get_logger()

    # Extract the section content
    section_content = extract_section_by_name(content, section_name, heading_level)

    if section_content is None:
        logger.warning(f"Section '{section_name}' not found, cannot insert link")
        return content

    # Build replacement content
    heading_marker = "#" * heading_level
    new_section = f"{heading_marker} {section_name}\n"

    if keep_brief_overview and section_content:
        # Extract first 2-3 lines (or first paragraph)
        lines = section_content.split("\n")
        brief_lines = []
        line_count = 0

        for line in lines:
            stripped = line.strip()
            if stripped:  # Non-empty line
                brief_lines.append(line)
                line_count += 1
                if line_count >= 2:  # Keep first 2 non-empty lines
                    break

        if brief_lines:
            brief_overview = "\n".join(brief_lines)
            new_section += f"{brief_overview}\n\n"

    # Add link
    new_section += f"For {link_text}, see [{link_url.split('/')[-1]}]({link_url}).\n"

    # Replace the original section with new version
    heading_marker_regex = "#" * heading_level
    # Build the pattern without f-string to avoid brace conflicts
    pattern = (
        r"^"
        + heading_marker_regex
        + r"\s+"
        + re.escape(section_name)
        + r"\s*\n.*?(?=^#{1,"
        + str(heading_level)
        + r"}\s|\Z)"
    )

    modified_content = re.sub(
        pattern, new_section, content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    logger.debug(
        f"Inserted link at section '{section_name}' "
        f"(keep_overview={keep_brief_overview})"
    )
    return modified_content


def detect_sections(content: str) -> List[Tuple[str, int]]:
    """Detect all sections in markdown content.

    This function parses all markdown headings and returns their names
    and levels.

    Args:
        content: Markdown content

    Returns:
        List[Tuple[str, int]]: List of (section_name, heading_level) tuples

    Example:
        >>> content = "# Title\\n## Installation\\n### Prerequisites\\n## Usage"
        >>> detect_sections(content)
        [('Title', 1), ('Installation', 2), ('Prerequisites', 3), ('Usage', 2)]
    """
    sections = []

    # Pattern to match any markdown heading
    heading_pattern = r"^(#{1,6})\s+(.+?)$"

    for match in re.finditer(heading_pattern, content, re.MULTILINE):
        heading_marker = match.group(1)
        section_name = match.group(2).strip()
        heading_level = len(heading_marker)

        sections.append((section_name, heading_level))

    return sections
