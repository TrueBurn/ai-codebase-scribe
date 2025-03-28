from pathlib import Path
from typing import List, Set, Dict, Optional
import re
import urllib.parse
import httpx
from dataclasses import dataclass
import time
import asyncio

@dataclass
class LinkIssue:
    """Represents an issue with a link in the documentation."""
    link: str
    line_number: int
    message: str
    severity: str  # 'error' or 'warning'

class LinkValidator:
    """Validates internal and external links in documentation."""
    
    def __init__(self, repo_path: Path, debug: bool = False, timeout: float = 5.0, max_retries: int = 2):
        self.repo_path = repo_path
        self.checked_urls: Dict[str, Dict] = {}  # Cache for URL validation results
        self.issues: List[LinkIssue] = []
        self.debug = debug
        self.timeout = timeout
        self.max_retries = max_retries
        self.anchor_map: Dict[str, Set[str]] = {}  # Map of document paths to sets of anchors
    
    async def validate_document(self, content: str, base_path: Path) -> List[LinkIssue]:
        """Validate all links in a markdown document."""
        self.issues = []
        lines = content.split('\n')
        
        # Build anchor map for this document
        document_path = str(base_path)
        self.anchor_map[document_path] = set()
        
        # Find all headers to build anchor map
        header_pattern = r'^(#{1,6})\s+(.+)$'
        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                header_text = header_match.group(2).strip()
                anchor = self._text_to_anchor(header_text)
                self.anchor_map[document_path].add(anchor)
        
        # Find all markdown links [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        for i, line in enumerate(lines, 1):
            for match in re.finditer(link_pattern, line):
                link_text, url = match.groups()
                await self._validate_link(url, i, base_path)
        
        # Find all badge links
        badge_pattern = r'!\[([^\]]+)\]\(([^)]+)\)'
        for i, line in enumerate(lines, 1):
            for match in re.finditer(badge_pattern, line):
                badge_text, url = match.groups()
                await self._validate_link(url, i, base_path)
        
        return self.issues
        
    def _text_to_anchor(self, text: str) -> str:
        """Convert header text to GitHub-style anchor."""
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        return re.sub(r'[^\w\- ]', '', text.lower()).replace(' ', '-')
    
    async def _validate_link(self, url: str, line_number: int, base_path: Path) -> None:
        """Validate a single link."""
        parsed = urllib.parse.urlparse(url)
        
        # Handle internal links
        if not parsed.scheme and not parsed.netloc:
            self._validate_internal_link(url, line_number, base_path)
            return
        
        # Handle external links
        cache_key = url
        if cache_key not in self.checked_urls:
            await self._validate_external_link(url, line_number)
            # Cache the result
            self.checked_urls[cache_key] = {
                "timestamp": time.time(),
                "issues": [issue for issue in self.issues if issue.link == url]
            }
        else:
            # Use cached result if it's less than 1 hour old
            cache_entry = self.checked_urls[cache_key]
            if time.time() - cache_entry["timestamp"] < 3600:  # 1 hour cache
                # Add cached issues to current issues list
                for issue in cache_entry["issues"]:
                    # Update line number to current context
                    self.issues.append(LinkIssue(
                        issue.link,
                        line_number,
                        issue.message,
                        issue.severity
                    ))
            else:
                # Cache expired, validate again
                await self._validate_external_link(url, line_number)
                # Update cache
                self.checked_urls[cache_key] = {
                    "timestamp": time.time(),
                    "issues": [issue for issue in self.issues if issue.link == url]
                }
    
    def _validate_internal_link(self, url: str, line_number: int, base_path: Path) -> None:
        """Validate internal file or anchor links."""
        # Split URL into file part and anchor part
        parts = url.split('#', 1)
        file_part = parts[0]
        anchor_part = parts[1] if len(parts) > 1 else None
        
        # Determine target document path
        if not file_part:
            # Anchor-only link, validate against current document
            target_doc_path = str(base_path)
            if anchor_part and target_doc_path in self.anchor_map:
                if anchor_part not in self.anchor_map[target_doc_path]:
                    self.issues.append(LinkIssue(
                        url,
                        line_number,
                        f"Anchor '{anchor_part}' not found in document",
                        "warning"
                    ))
            return
        
        # Handle relative paths for file part
        target_path = (base_path / file_part).resolve()
        target_doc_path = str(target_path)
        
        try:
            relative_to_repo = target_path.relative_to(self.repo_path)
        except ValueError:
            self.issues.append(LinkIssue(
                url,
                line_number,
                "Link points outside repository",
                "error"
            ))
            return
        
        if not target_path.exists():
            self.issues.append(LinkIssue(
                url,
                line_number,
                "Referenced file does not exist",
                "error"
            ))
            return
        
        # Validate anchor if present
        if anchor_part:
            # If we haven't processed this document yet, we can't validate its anchors
            if target_doc_path not in self.anchor_map:
                # Try to read and process the target document to build its anchor map
                try:
                    target_content = target_path.read_text(encoding='utf-8')
                    self.anchor_map[target_doc_path] = set()
                    
                    # Find all headers to build anchor map
                    header_pattern = r'^(#{1,6})\s+(.+)$'
                    for line in target_content.split('\n'):
                        header_match = re.match(header_pattern, line)
                        if header_match:
                            header_text = header_match.group(2).strip()
                            anchor = self._text_to_anchor(header_text)
                            self.anchor_map[target_doc_path].add(anchor)
                except Exception:
                    # If we can't read the file, we can't validate anchors
                    return
            
            # Now check if the anchor exists in the target document
            if target_doc_path in self.anchor_map and anchor_part not in self.anchor_map[target_doc_path]:
                self.issues.append(LinkIssue(
                    url,
                    line_number,
                    f"Anchor '{anchor_part}' not found in target document",
                    "warning"
                ))
    
    async def _validate_external_link(self, url: str, line_number: int) -> None:
        """Validate external URLs."""
        # Implement retry logic
        retries = 0
        while retries <= self.max_retries:
            try:
                # Create a client with SSL verification disabled (as requested by user)
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.head(
                        url,
                        follow_redirects=True,
                        timeout=self.timeout
                    )
                    
                    if response.status_code >= 400:
                        self.issues.append(LinkIssue(
                            url,
                            line_number,
                            f"Broken link (HTTP {response.status_code})",
                            "error"
                        ))
                    elif response.status_code >= 300:
                        self.issues.append(LinkIssue(
                            url,
                            line_number,
                            f"Redirected link (HTTP {response.status_code})",
                            "warning"
                        ))
                    
                    # If we got here, the request was successful
                    break
                        
            except httpx.TimeoutException:
                if retries == self.max_retries:
                    self.issues.append(LinkIssue(
                        url,
                        line_number,
                        f"Link timeout after {retries + 1} attempts",
                        "warning"
                    ))
                retries += 1
                if retries <= self.max_retries:
                    # Wait a bit before retrying (exponential backoff)
                    await asyncio.sleep(1 * (2 ** retries))
                    continue
                break
                
            except Exception as e:
                # Log the error but don't treat SSL errors as link validation failures
                if "SSL" in str(e) or "certificate" in str(e).lower():
                    if self.debug:
                        print(f"SSL verification error for {url}: {e}")
                    break  # Don't retry SSL errors
                else:
                    if retries == self.max_retries:
                        self.issues.append(LinkIssue(
                            url,
                            line_number,
                            f"Link validation failed: {str(e)}",
                            "error"
                        ))
                    retries += 1
                    if retries <= self.max_retries:
                        # Wait a bit before retrying
                        await asyncio.sleep(1 * (2 ** retries))
                        continue
                    break