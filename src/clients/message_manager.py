from typing import Dict, List, final, TypeVar
import json
import logging

# Type for token counter objects
TokenCounter = TypeVar('TokenCounter')

@final
class MessageManager:
    """Manages message formatting for different LLM providers.
    
    This class provides a centralized place to define system and user prompts
    for different LLM interactions, ensuring consistency across the application.
    All methods are static and should not be overridden.
    """
    
    # Version tracking for API compatibility
    VERSION = "1.0.0"
    
    @staticmethod
    def create_system_user_messages(system_content: str, user_content: str) -> List[Dict[str, str]]:
        """Create standard system and user messages.
        
        This is the core method used by all other methods to create properly
        formatted message pairs for LLM requests.
        
        Args:
            system_content: Content for the system message that sets context and instructions
            user_content: Content for the user message that contains the specific request
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.create_system_user_messages(
                "You are a helpful assistant.",
                "Explain Python decorators."
            )
            # Result:
            # [
            #     {"role": "system", "content": "You are a helpful assistant."},
            #     {"role": "user", "content": "Explain Python decorators."}
            # ]
            ```
        """
        # Validate inputs
        if not isinstance(system_content, str) or not system_content.strip():
            raise ValueError("System content must be a non-empty string")
        if not isinstance(user_content, str) or not user_content.strip():
            raise ValueError("User content must be a non-empty string")
            
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    @staticmethod
    def get_project_overview_messages(project_structure: str, tech_report: str, template_content: str) -> List[Dict[str, str]]:
        """Get standardized messages for project overview generation.
        
        Creates a message pair for generating a comprehensive project overview
        based on the project structure and detected technologies.
        
        Args:
            project_structure: String representation of the project's file structure
            tech_report: String containing detected technologies and dependencies
            template_content: Template content for the user message
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_project_overview_messages(
                project_structure="src/\n  main.py\n  utils.py",
                tech_report="Python 3.9\nRequirements: requests, numpy",
                template_content="Generate a README for this project"
            )
            ```
        
        Raises:
            ValueError: If any of the input parameters are empty or not strings
        """
        # Log input parameters
        logging.debug(f"get_project_overview_messages called with:")
        logging.debug(f"project_structure type: {type(project_structure)}, length: {len(project_structure) if isinstance(project_structure, str) else 'N/A'}")
        logging.debug(f"tech_report type: {type(tech_report)}, value: {tech_report}")
        logging.debug(f"template_content type: {type(template_content)}, length: {len(template_content) if isinstance(template_content, str) else 'N/A'}")
        
        # Validate inputs
        if not isinstance(project_structure, str) or not project_structure.strip():
            logging.error("Project structure validation failed")
            raise ValueError("Project structure must be a non-empty string")
        
        # Ensure tech_report is a string
        if not isinstance(tech_report, str):
            logging.warning(f"tech_report is not a string, it's a {type(tech_report)}")
            tech_report = "No dependencies detected."
        
        if not isinstance(template_content, str) or not template_content.strip():
            logging.error("Template content validation failed")
            raise ValueError("Template content must be a non-empty string")
        system_content = f"""Project Structure:
{project_structure}

Detected Technologies:
{tech_report}

Analyze in English and ensure all responses are in English.
Use proper markdown formatting:
- Ensure headers follow proper hierarchy (H1 → H2 → H3)
- Use consistent list indentation (multiples of 2 spaces)
- Add blank lines before headers"""
        
        return MessageManager.create_system_user_messages(system_content, template_content)
    
    @staticmethod
    def get_component_relationship_messages(project_structure: str, tech_report: str) -> List[Dict[str, str]]:
        """Get standardized messages for component relationship analysis.
        
        Creates a message pair for analyzing how components in a project interact
        with each other, focusing on data flow, dependencies, and design patterns.
        
        Args:
            project_structure: String representation of the project's file structure
            tech_report: String containing detected technologies and dependencies
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_component_relationship_messages(
                project_structure="src/\n  main.py\n  utils.py",
                tech_report="Python 3.9\nRequirements: requests, numpy"
            )
            ```
            
        Raises:
            ValueError: If any of the input parameters are empty or not strings
        """
        # Validate inputs
        if not isinstance(project_structure, str) or not project_structure.strip():
            raise ValueError("Project structure must be a non-empty string")
        if not isinstance(tech_report, str) or not tech_report.strip():
            raise ValueError("Tech report must be a non-empty string")
        system_content = f"""Project Structure:
{project_structure}

Detected Technologies:
{tech_report}

IMPORTANT INSTRUCTION: ONLY discuss technologies and patterns that are explicitly 
evidenced in the Detected Technologies section above. DO NOT assume or infer the
presence of any framework, library, or architecture that is not directly observable
in the codebase. If the technology stack is unclear, acknowledge the limitations
rather than making assumptions.

Analyze in English and ensure all responses are in English.
Use proper markdown formatting:
- Ensure headers follow proper hierarchy (H1 → H2 → H3)
- Use consistent list indentation (multiples of 2 spaces)
- Add blank lines before headers"""
        
        user_content = """Analyze how the major components in this project interact with each other.

Include:
- Main data/control flow
- Key dependencies
- Important interfaces
- Notable design patterns

Focus on high-level architectural relationships that are ACTUALLY present in the code,
not what you think should be there based on common patterns."""
        
        return MessageManager.create_system_user_messages(system_content, user_content)
    
    @staticmethod
    def get_file_summary_messages(prompt: str) -> List[Dict[str, str]]:
        """Get standardized messages for file summary generation.
        
        Creates a message pair for generating a summary of a code file.
        The prompt should contain the file content to be summarized.
        
        Args:
            prompt: String containing the file content to summarize
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            file_content = "def hello():\n    print('Hello world')"
            messages = MessageManager.get_file_summary_messages(file_content)
            ```
            
        Raises:
            ValueError: If the prompt is empty or not a string
        """
        # Validate input
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        system_content = """You are a code documentation expert. Analyze the provided code file and create a clear, 
concise summary that explains:
1. The purpose and functionality of the file
2. Key components, classes, or functions
3. Important algorithms or patterns used
4. How this file relates to the overall project
5. Any notable dependencies or imports
6. Use proper markdown formatting with consistent indentation"""
        
        return MessageManager.create_system_user_messages(system_content, prompt)
    
    @staticmethod
    def get_architecture_content_messages(project_structure: str, key_components: str, tech_report: str) -> List[Dict[str, str]]:
        """Get standardized messages for architecture documentation generation.
        
        Creates a message pair for generating comprehensive architecture documentation
        including component diagrams, data flow, and design patterns.
        
        Args:
            project_structure: String representation of the project's file structure
            key_components: String describing the key components in the project
            tech_report: String containing detected technologies and dependencies
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_architecture_content_messages(
                project_structure="src/\n  main.py\n  utils.py",
                key_components="Main module, Utils module",
                tech_report="Python 3.9\nRequirements: requests, numpy"
            )
            ```
            
        Raises:
            ValueError: If any of the input parameters are empty or not strings
        """
        # Validate inputs
        if not isinstance(project_structure, str) or not project_structure.strip():
            raise ValueError("Project structure must be a non-empty string")
        if not isinstance(key_components, str) or not key_components.strip():
            raise ValueError("Key components must be a non-empty string")
        if not isinstance(tech_report, str) or not tech_report.strip():
            raise ValueError("Tech report must be a non-empty string")
        system_content = """You are a technical documentation expert. Create comprehensive architecture documentation for this project.
        
Include the following sections:
1. Overview - A high-level description of the project architecture
2. Project Structure - Include the provided structure as a code block
3. Component Diagram - Create a mermaid flowchart diagram showing the main components and their relationships
4. Data Flow - Describe how data flows through the system
5. Key Technologies - List and explain the technologies used
6. Design Patterns - Identify any design patterns used in the project

Use proper markdown formatting with appropriate headers and code blocks.
For the Component Diagram, use mermaid.js flowchart syntax like this:
```mermaid
flowchart TD
    A[Component A] --> B[Component B]
    A --> C[Component C]
    B --> D[Component D]
    C --> D
```

Analyze the project structure to identify the main components and their relationships."""

        user_content = f"""Based on the following project information, create detailed architecture documentation.

Project Structure:
```
{project_structure}
```

Key Components:
{key_components}

Technologies:
{tech_report}

Please include a mermaid flowchart diagram showing the relationships between key components."""

        return MessageManager.create_system_user_messages(system_content, user_content)
    
    @staticmethod
    def get_enhance_documentation_messages(existing_content: str, project_structure: str,
                                          key_components: str, tech_report: str, doc_type: str) -> List[Dict[str, str]]:
        """Get standardized messages for enhancing existing documentation.
        
        Creates a message pair for improving existing documentation by preserving
        valuable information, reorganizing structure, and adding missing details.
        
        Args:
            existing_content: Current content of the documentation file
            project_structure: String representation of the project's file structure
            key_components: String describing the key components in the project
            tech_report: String containing detected technologies and dependencies
            doc_type: Type of documentation being enhanced (e.g., "README", "ARCHITECTURE")
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_enhance_documentation_messages(
                existing_content="# Project\nThis is a sample project.",
                project_structure="src/\n  main.py\n  utils.py",
                key_components="Main module, Utils module",
                tech_report="Python 3.9\nRequirements: requests, numpy",
                doc_type="README"
            )
            ```
            
        Raises:
            ValueError: If any of the input parameters are empty or not strings
        """
        # Validate inputs
        if not isinstance(existing_content, str) or not existing_content.strip():
            raise ValueError("Existing content must be a non-empty string")
        if not isinstance(project_structure, str) or not project_structure.strip():
            raise ValueError("Project structure must be a non-empty string")
        if not isinstance(key_components, str) or not key_components.strip():
            raise ValueError("Key components must be a non-empty string")
        if not isinstance(tech_report, str) or not tech_report.strip():
            raise ValueError("Tech report must be a non-empty string")
        if not isinstance(doc_type, str) or not doc_type.strip():
            raise ValueError("Doc type must be a non-empty string")
        system_content = f"""You are a technical documentation expert specializing in {doc_type} files.
        
Project Structure:
{project_structure}

Key Components:
{key_components}

Detected Technologies:
{tech_report}

Your task is to enhance the existing {doc_type} file by:
1. Preserving valuable information from the existing content
2. Reorganizing the document structure for better clarity and flow
3. Removing outdated, redundant, or incorrect information
4. Adding missing information based on the repository analysis
5. Ensuring proper markdown formatting with consistent header hierarchy
6. Maintaining code snippets and examples, updating them only if incorrect
7. Maintaining the original tone and style where appropriate

Return a completely restructured document that represents the best possible documentation for this codebase.
"""
        
        user_content = f"""Here is the existing {doc_type} content:

{existing_content}

Please enhance this document by intelligently combining the existing content with new insights from the repository analysis.
Create the best possible documentation that accurately represents the codebase.
"""
        
        return MessageManager.create_system_user_messages(system_content, user_content)
    
    @staticmethod
    def get_usage_guide_messages(project_structure: str, tech_report: str) -> List[Dict[str, str]]:
        """Get standardized messages for usage guide generation.
        
        Creates a message pair for generating a comprehensive usage guide
        with examples, configuration options, and troubleshooting tips.
        
        Args:
            project_structure: String representation of the project's file structure
            tech_report: String containing detected technologies and dependencies
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_usage_guide_messages(
                project_structure="src/\n  main.py\n  utils.py",
                tech_report="Python 3.9\nRequirements: requests, numpy"
            )
            ```
            
        Raises:
            ValueError: If any of the input parameters are empty or not strings
        """
        # Validate inputs
        if not isinstance(project_structure, str) or not project_structure.strip():
            raise ValueError("Project structure must be a non-empty string")
        if not isinstance(tech_report, str) or not tech_report.strip():
            raise ValueError("Tech report must be a non-empty string")
        system_content = """You are a technical documentation expert. Create clear usage instructions based on the project structure."""
        
        user_content = f"""Based on the project structure and dependencies, generate a usage guide.
        
Project Structure:
{project_structure}

Dependencies:
{tech_report}

Please provide clear instructions for:
1. Basic usage examples
2. Common operations
3. Configuration options
4. Troubleshooting tips

Use proper markdown formatting with clear section headers."""
        
        return MessageManager.create_system_user_messages(system_content, user_content)
    
    @staticmethod
    def get_contributing_guide_messages(project_structure: str) -> List[Dict[str, str]]:
        """Get standardized messages for contributing guide generation.
        
        Creates a message pair for generating a comprehensive contributing guide
        with development setup, code standards, and PR process information.
        
        Args:
            project_structure: String representation of the project's file structure
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_contributing_guide_messages(
                project_structure="src/\n  main.py\n  utils.py"
            )
            ```
            
        Raises:
            ValueError: If the project_structure is empty or not a string
        """
        # Validate input
        if not isinstance(project_structure, str) or not project_structure.strip():
            raise ValueError("Project structure must be a non-empty string")
        system_content = """You are a technical documentation expert. Create clear contributing guidelines based on the project structure."""
        
        user_content = f"""Based on the project structure, generate a contributing guide.
        
Project Structure:
{project_structure}

Please provide clear guidelines for:
1. Setting up the development environment
2. Code style and standards
3. Testing procedures
4. Pull request process
5. Issue reporting

Use proper markdown formatting with clear section headers."""
        
        return MessageManager.create_system_user_messages(system_content, user_content)
    
    @staticmethod
    def get_license_info_messages(project_structure: str) -> List[Dict[str, str]]:
        """Get standardized messages for license information generation.
        
        Creates a message pair for generating license information based on
        the project structure, including license type and copyright notices.
        
        Args:
            project_structure: String representation of the project's file structure
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_license_info_messages(
                project_structure="src/\n  main.py\n  utils.py\nLICENSE"
            )
            ```
            
        Raises:
            ValueError: If the project_structure is empty or not a string
        """
        # Validate input
        if not isinstance(project_structure, str) or not project_structure.strip():
            raise ValueError("Project structure must be a non-empty string")
        system_content = """You are a technical documentation expert. Create clear license information based on the project structure."""
        
        user_content = f"""Based on the project structure, generate license information.
        
Project Structure:
{project_structure}

Please provide:
1. License type (if detectable)
2. Brief explanation of the license
3. Any copyright notices found

If no license information is found, provide a generic statement about licensing.
Use proper markdown formatting."""
        
        return MessageManager.create_system_user_messages(system_content, user_content)
    
    @staticmethod
    def get_file_order_messages(files_info: dict) -> List[Dict[str, str]]:
        """Get standardized messages for file order optimization.
        
        Creates a message pair for determining the optimal order to process
        files in a codebase, based on dependencies and complexity.
        
        Args:
            files_info: Dictionary containing information about the files to order
            
        Returns:
            A list of message dictionaries formatted for LLM API requests
            
        Example:
            ```python
            messages = MessageManager.get_file_order_messages({
                "file1.py": {"imports": ["utils.py"]},
                "utils.py": {"imports": []}
            })
            ```
            
        Raises:
            ValueError: If files_info is not a dictionary
        """
        # Validate input
        if not isinstance(files_info, dict):
            raise ValueError("Files info must be a dictionary")
        system_content = """Determine the optimal order for analyzing files in a codebase.
        Follow these rules:
        1. Configuration files first (e.g., appsettings.json)
        2. Core infrastructure and utilities next
        3. Base classes before implementations
        4. Simpler files before complex ones
        5. Respect dependencies (if A depends on B, process B first)
        6. Always respond in English
        
        IMPORTANT: Return your response as a JSON object with a "file_order" array containing the file paths in the optimal order.
        Example: {"file_order": ["config.json", "utils.py", "main.py"], "reasoning": "Config files first, then utilities, then main application code."}"""
        
        user_content = json.dumps({
            "files": files_info,
            "task": "Analyze these files and return them in optimal processing order"
        }, indent=2)
        
        return MessageManager.create_system_user_messages(system_content, user_content)
    
    @staticmethod
    def check_and_truncate_messages(messages: List[Dict[str, str]], token_counter: TokenCounter, model_name: str) -> List[Dict[str, str]]:
        """Check if messages exceed token limit and truncate if needed.
        
        This method implements a multi-stage token reduction strategy:
        1. First attempts intelligent content reduction that preserves important information
        2. If still over limit, falls back to hard truncation of user messages
        
        The system message is preserved as much as possible since it contains critical
        context and instructions, while user messages are prioritized for truncation.
        
        Args:
            messages: List of message dictionaries to check and potentially truncate
            token_counter: TokenCounter instance used to count tokens and perform truncation
            model_name: Name of the model to check token limits against
            
        Returns:
            Original messages if under limit, or truncated messages if over limit
            
        Example:
            ```python
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Very long content..."}
            ]
            truncated = MessageManager.check_and_truncate_messages(
                messages, token_counter, "gpt-4"
            )
            ```
        
        Raises:
            ValueError: If messages is not a valid list of message dictionaries
            TypeError: If token_counter doesn't have required methods
        """
        # Validate inputs
        if not isinstance(messages, list) or not all(
            isinstance(m, dict) and "role" in m and "content" in m
            for m in messages
        ):
            raise ValueError("Messages must be a list of dictionaries with 'role' and 'content' keys")
            
        if not hasattr(token_counter, "will_exceed_limit") or not callable(getattr(token_counter, "will_exceed_limit")):
            raise TypeError("token_counter must have a 'will_exceed_limit' method")
            
        # Check if we're already under the limit
        will_exceed, token_count = token_counter.will_exceed_limit(messages, model_name)
        
        if not will_exceed:
            return messages
        
        # If we exceed the limit, we need to truncate the user message content
        # System message is usually shorter and more important for context
        truncated_messages = messages.copy()
        
        # STRATEGY 1: Intelligent content reduction
        # This preserves important parts of the content while reducing overall size
        for i, message in enumerate(truncated_messages):
            if message["role"] == "user":
                # Use the intelligent reduction method that tries to preserve meaning
                truncated_content = token_counter.handle_oversized_input(
                    message["content"],
                    target_percentage=0.8  # Target 80% of model's limit
                )
                truncated_messages[i]["content"] = truncated_content
                
                # Check if we're now under the limit
                still_exceeds, new_count = token_counter.will_exceed_limit(truncated_messages, model_name)
                if not still_exceeds:
                    logging.info(f"Intelligently reduced message from {token_count} to {new_count} tokens")
                    return truncated_messages
        
        # STRATEGY 2: Hard truncation as last resort
        # If intelligent reduction wasn't enough, fall back to simple truncation
        for i, message in enumerate(truncated_messages):
            if message["role"] == "user":
                # Calculate the effective token limit for this message
                # We'll use 90% of the model's limit minus the tokens from other messages
                other_messages = [m for j, m in enumerate(messages) if j != i]
                other_tokens = token_counter.count_message_tokens(other_messages)
                model_limit = token_counter.get_token_limit(model_name)
                effective_limit = int(model_limit * 0.9) - other_tokens
                
                # Hard truncate the content to fit within the limit
                truncated_content = token_counter.truncate_text(message["content"], effective_limit)
                truncated_messages[i]["content"] = truncated_content
                
                logging.warning(f"Forced to truncate message to {effective_limit} tokens")
                break  # Only truncate one message (the user message)
        
        return truncated_messages