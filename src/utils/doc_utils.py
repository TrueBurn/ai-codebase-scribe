#!/usr/bin/env python3

"""
Utilities for documentation generation and manipulation.
"""

import re
from typing import Optional


def add_ai_attribution(content: str, doc_type: str = "documentation", badges: str = "") -> str:
    """Add AI attribution footer and badges to generated content if not already present.
    
    Args:
        content: The content to add attribution to
        doc_type: The type of document (e.g., "README", "ARCHITECTURE.md")
        badges: Badges to add to the document
        
    Returns:
        The content with attribution and badges added
    """
    attribution_text = f"\n\n---\n_This {doc_type} was generated using AI analysis and may contain inaccuracies. Please verify critical information._"
    
    def add_badges_after_title(content: str, badges: str) -> str:
        """Helper function to add badges after the title.
        
        Args:
            content: The content to add badges to
            badges: The badges to add
            
        Returns:
            The content with badges added after the title
        """
        if not badges:
            return content
            
        title_match = re.search(r"^# (.+?)(?:\n|$)", content)
        if not title_match:
            return content
            
        title_end = title_match.end()
        return content[:title_end] + "\n\n" + badges + "\n" + content[title_end:]
    
    # Check if content already has an attribution footer
    has_attribution = "_This " in content and ("generated" in content.lower() or "enhanced" in content.lower()) and "AI" in content
    
    if has_attribution:
        # Already has attribution, just add badges if needed
        if badges and "![" not in content[:500]:  # Only add badges if they don't exist in the first 500 chars
            content = add_badges_after_title(content, badges)
        return content
    
    # Add badges after the title if provided
    content = add_badges_after_title(content, badges)
    
    # Add the attribution
    return content + attribution_text