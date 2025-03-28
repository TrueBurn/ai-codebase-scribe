from pathlib import Path
from typing import Dict, Optional, Any, List, Set, Tuple
import yaml
import os
import logging
import re
from datetime import datetime

class PromptTemplate:
    """
    Manages customizable prompt templates with context injection.
    
    This class provides functionality to load default templates, override them with
    custom templates from a configuration file, and format templates with context variables.
    It includes template validation and versioning support.
    """
    
    # Current version of the template system
    VERSION = "1.0.0"
    
    DEFAULT_TEMPLATES = {
        'file_summary': """
Analyze the following code file and provide a clear, concise summary:

File: {file_path}
Type: {file_type}
Context:
- Imports: {imports}
- Exports: {exports}
- Dependencies: {dependencies}

Code:
{code}

Generate a summary that includes:
1. Main purpose of the file
2. Key components and their roles
3. How it interacts with other modules
4. Any notable patterns or design choices

Keep the summary clear and concise.
""".strip(),
        
        'project_overview': """
Analyze this project's structure and generate a comprehensive overview:

Project: {project_name}
Files: {file_count}
Key Components:
{key_components}

Dependencies:
{dependencies}

Generate a project overview that includes:
1. Main purpose and goals
2. Key features and capabilities
3. Architecture overview
4. Technology stack and design patterns

Focus on high-level understanding while highlighting unique aspects.
""".strip(),

        'enhance_documentation': """
You are enhancing an existing {doc_type} file.

EXISTING CONTENT:
{existing_content}

REPOSITORY ANALYSIS:
{analysis}

Your task is to create the best possible documentation by intelligently combining the existing content with new insights from the repository analysis.

Guidelines:
1. Preserve valuable information from the existing content, especially specific implementation details, configuration examples, and custom instructions.
2. Feel free to reorganize the document structure to improve clarity and flow.
3. Remove outdated, redundant, or incorrect information.
4. Add missing information and technical details based on the repository analysis.
5. Ensure proper markdown formatting with consistent header hierarchy.
6. Maintain code snippets and examples, updating them only if they're incorrect.
7. If the existing content has a specific tone or style, try to maintain it.

Return a completely restructured document that represents the best possible documentation for this codebase, combining the strengths of the existing content with new insights.
""".strip()
    }
    
    # Define required placeholders for each template
    TEMPLATE_REQUIREMENTS = {
        'file_summary': ['file_path', 'file_type', 'code'],
        'project_overview': ['project_name', 'file_count', 'key_components', 'dependencies'],
        'enhance_documentation': ['doc_type', 'existing_content', 'analysis']
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the PromptTemplate with default templates and optionally load custom templates.
        
        Args:
            config_path: Optional path to a YAML config file containing custom templates
        """
        self.templates = self.DEFAULT_TEMPLATES.copy()
        self.template_versions = {name: self.VERSION for name in self.DEFAULT_TEMPLATES}
        self.last_loaded = datetime.now().isoformat()
        
        if config_path and config_path.exists():
            self._load_custom_templates(config_path)
        
        # Validate all templates after loading
        self._validate_all_templates()
    
    def _load_custom_templates(self, config_path: Path):
        """
        Load custom templates from config file.
        
        Args:
            config_path: Path to a YAML config file containing custom templates
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            if config is None:
                logging.warning(f"Empty or invalid config file: {config_path}")
                return
            
            # Load templates with version tracking
            if 'templates' in config:
                for name, content in config['templates'].items():
                    self.templates[name] = content
                    # Track template version if provided, otherwise use current timestamp
                    if isinstance(content, dict) and 'version' in content and 'content' in content:
                        self.template_versions[name] = content['version']
                        self.templates[name] = content['content']
                    else:
                        self.template_versions[name] = f"custom-{datetime.now().strftime('%Y%m%d')}"
            
            self.last_loaded = datetime.now().isoformat()
            logging.info(f"Loaded {len(config.get('templates', {}))} custom templates from {config_path}")
            
        except FileNotFoundError:
            logging.error(f"Template config file not found: {config_path}")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML in template config: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading template config: {e}")
    
    def _validate_all_templates(self):
        """
        Validate all loaded templates to ensure they contain required placeholders.
        """
        for name, template in self.templates.items():
            if name in self.TEMPLATE_REQUIREMENTS:
                missing = self._validate_template(template, self.TEMPLATE_REQUIREMENTS[name])
                if missing:
                    logging.warning(f"Template '{name}' is missing required placeholders: {', '.join(missing)}")
    
    def _validate_template(self, template: str, required_placeholders: List[str]) -> List[str]:
        """
        Validate that a template contains all required placeholders.
        
        Args:
            template: The template string to validate
            required_placeholders: List of placeholder names that must be present
            
        Returns:
            List of missing placeholders, empty if all required placeholders are present
        """
        # Find all placeholders in the template using regex
        found_placeholders = set(re.findall(r'\{([^{}]+)\}', template))
        
        # Check for missing required placeholders
        missing = [p for p in required_placeholders if p not in found_placeholders]
        return missing
    
    def get_template(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a template by name and optionally format it with context.
        
        Args:
            template_name: Name of the template to retrieve
            context: Optional dictionary of values to format the template with
            
        Returns:
            The template string, optionally formatted with context
        """
        if template_name not in self.templates:
            logging.warning(f"Template '{template_name}' not found, using default")
            # Return a simple default template if the requested one doesn't exist
            return f"Please analyze the following content: {{content}}"
        
        template = self.templates[template_name]
        
        # If context is provided, prepare and format the template
        if context:
            # Validate context against required placeholders
            if template_name in self.TEMPLATE_REQUIREMENTS:
                missing = [p for p in self.TEMPLATE_REQUIREMENTS[template_name]
                          if p not in context and p not in self._get_default_keys()]
                if missing:
                    logging.warning(f"Context for template '{template_name}' is missing required keys: {', '.join(missing)}")
            
            prepared_context = self._prepare_context(context)
            try:
                return template.format(**prepared_context)
            except KeyError as e:
                logging.warning(f"Missing key in template formatting: {e}")
                return template
        
        return template
    
    def _get_default_keys(self) -> Set[str]:
        """
        Get the set of all default keys available in the context preparation.
        
        Returns:
            Set of default key names
        """
        return set({
            'file_path', 'file_type', 'imports', 'exports', 'dependencies',
            'code', 'project_name', 'file_count', 'key_components',
            'existing_content', 'doc_type', 'analysis',
        })
    
    def _prepare_context(self, context: Dict) -> Dict:
        """
        Prepare context variables for template injection.
        
        This method ensures all potential template variables have defaults and
        formats complex data types (lists, dicts) for better display in templates.
        
        Args:
            context: Dictionary of context variables to prepare
            
        Returns:
            Prepared context dictionary with defaults and formatted values
        """
        # Ensure all potential template variables have defaults
        defaults = {
            'file_path': '',
            'file_type': '',
            'imports': '(none)',
            'exports': '(none)',
            'dependencies': '(none)',
            'code': '',
            'project_name': 'Unknown Project',
            'file_count': 0,
            'key_components': '(none)',
            'existing_content': '',
            'doc_type': 'documentation',
            'analysis': '(none)',
        }
        
        # Update defaults with provided context
        defaults.update(context)
        
        # Format lists and dicts for better display
        for key, value in defaults.items():
            if isinstance(value, (list, set)):
                defaults[key] = '\n'.join(f'- {item}' for item in sorted(value))
            elif isinstance(value, dict):
                defaults[key] = '\n'.join(f'- {k}: {v}' for k, v in sorted(value.items()))
        
        return defaults
    
    def get_template_info(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all available templates.
        
        Returns:
            Dictionary with template names as keys and info dictionaries as values
        """
        result = {}
        for name, template in self.templates.items():
            # Extract first line as description
            description = template.split('\n')[0].strip() if template else "No description"
            
            # Get required placeholders
            required = self.TEMPLATE_REQUIREMENTS.get(name, [])
            
            # Get all placeholders
            all_placeholders = set(re.findall(r'\{([^{}]+)\}', template))
            
            result[name] = {
                'version': self.template_versions.get(name, 'unknown'),
                'description': description,
                'required_placeholders': ', '.join(required),
                'all_placeholders': ', '.join(sorted(all_placeholders)),
                'length': len(template)
            }
        
        return result
    
    def add_template(self, name: str, content: str, version: Optional[str] = None) -> bool:
        """
        Add a new template or update an existing one.
        
        Args:
            name: Template name
            content: Template content
            version: Optional version string
            
        Returns:
            True if successful, False otherwise
        """
        if not name or not content:
            logging.error("Template name and content are required")
            return False
        
        self.templates[name] = content
        self.template_versions[name] = version or f"custom-{datetime.now().strftime('%Y%m%d')}"
        
        # Validate the new template if we have requirements for it
        if name in self.TEMPLATE_REQUIREMENTS:
            missing = self._validate_template(content, self.TEMPLATE_REQUIREMENTS[name])
            if missing:
                logging.warning(f"New template '{name}' is missing required placeholders: {', '.join(missing)}")
        
        return True