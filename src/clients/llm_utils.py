"""Shared utilities for LLM clients."""
import json
import logging
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional
# Default configuration values that could be moved to a config file
DEFAULT_MAX_COMPONENTS = 10  # Maximum number of key components to display

# Default vendor patterns for file filtering
DEFAULT_VENDOR_PATTERNS = [
    r'[\\/]bootstrap[\\/]',           # Bootstrap files
    r'[\\/]vendor[\\/]',              # Vendor directories
    r'[\\/]wwwroot[\\/]lib[\\/]',     # Library resources
    r'\.min\.(js|css|map)$',          # Minified files
    r'\.css\.map$',                   # Source maps
    r'\.ico$|\.png$|\.jpg$|\.gif$',   # Images
    r'[\\/]node_modules[\\/]',        # Node modules
    r'[\\/]dist[\\/]',                # Distribution files
    r'[\\/]packages[\\/]',            # Package files
    r'[\\/]PublishProfiles[\\/]',     # Publish profiles
    r'\.pubxml(\.user)?$',            # Publish XML files
    r'\.csproj(\.user)?$',            # Project files
    r'\.sln$'                         # Solution files
]

def format_project_structure(file_manifest: Dict[str, Dict], debug: bool = False, force_compression: Optional[bool] = None) -> str:
    """
    Build a tree-like project structure string from file manifest.
    
    Args:
        file_manifest: Dictionary mapping file paths to file information
        debug: Whether to print debug information
        force_compression: If provided, overrides the automatic decision on whether to use compression
        
    Returns:
        A formatted string representing the project structure
    """
    logging.debug("Formatting project structure")
    try:
        # Import path compression utilities
        from ..utils.path_compression import compress_paths, get_compression_explanation
        
        # Get file paths from manifest
        file_paths = [str(path).replace('\\', '/') for path in file_manifest.keys()]
        
        # Check if we should use path compression
        # If force_compression is provided, use that value
        # Otherwise, use compression for large projects (more than 50 files)
        use_compression = force_compression if force_compression is not None else len(file_paths) > 50
        
        if use_compression:
            # Apply path compression to reduce token usage
            compressed_paths, decompression_map = compress_paths(file_paths)
            
            # Create a nested dictionary to represent the directory structure
            directories: Dict[str, Dict] = {}
            
            for path in compressed_paths:
                parts = path.split('/')
                current_dir = directories
                
                # Navigate through the directory structure
                for i, part in enumerate(parts[:-1]):  # All parts except the last one (filename)
                    if part not in current_dir:
                        current_dir[part] = {}
                    current_dir = current_dir[part]
                
                # Add the file to the current directory
                if '_files' not in current_dir:
                    current_dir['_files'] = []
                current_dir['_files'].append(parts[-1])  # The last part is the filename
            
            # Format the directory structure as a string
            def format_dir(dir_dict: Dict, indent: int = 0) -> List[str]:
                result = []
                # First list all files in the current directory
                if '_files' in dir_dict:
                    for file in sorted(dir_dict['_files']):
                        result.append(' ' * indent + '- ' + file)
                
                # Then list all subdirectories
                for name, contents in sorted(dir_dict.items()):
                    if name != '_files':
                        result.append(' ' * indent + '+ ' + name + '/')
                        result.extend(format_dir(contents, indent + 2))
                
                return result
            
            structure_lines = format_dir(directories)
            formatted_structure = '\n'.join(structure_lines)
            
            # Add explanation of compression scheme
            compression_explanation = get_compression_explanation(decompression_map)
            formatted_structure = compression_explanation + "\n\n" + formatted_structure
        else:
            # Use the original approach for smaller projects
            directories: Dict[str, Dict] = {}
            
            for path in file_paths:
                parts = path.split('/')
                current_dir = directories
                
                # Navigate through the directory structure
                for i, part in enumerate(parts[:-1]):  # All parts except the last one (filename)
                    if part not in current_dir:
                        current_dir[part] = {}
                    current_dir = current_dir[part]
                
                # Add the file to the current directory
                if '_files' not in current_dir:
                    current_dir['_files'] = []
                current_dir['_files'].append(parts[-1])  # The last part is the filename
            
            # Format the directory structure as a string
            def format_dir(dir_dict: Dict, indent: int = 0) -> List[str]:
                result = []
                # First list all files in the current directory
                if '_files' in dir_dict:
                    for file in sorted(dir_dict['_files']):
                        result.append(' ' * indent + '- ' + file)
                
                # Then list all subdirectories
                for name, contents in sorted(dir_dict.items()):
                    if name != '_files':
                        result.append(' ' * indent + '+ ' + name + '/')
                        result.extend(format_dir(contents, indent + 2))
                
                return result
            
            structure_lines = format_dir(directories)
            formatted_structure = '\n'.join(structure_lines)
        
        logging.debug(f"Project structure formatted successfully with {len(structure_lines)} lines")
        return formatted_structure
    except Exception as e:
        logging.error(f"Error formatting project structure: {e}")
        if debug:
            print(f"Error formatting project structure: {e}")
            print(traceback.format_exc())
        return "Error formatting project structure"
def find_common_dependencies(file_manifest: Dict[str, Dict], debug: bool = False) -> str:
    """
    Extract common dependencies from file manifest.
    
    Args:
        file_manifest: Dictionary mapping file paths to file information
        debug: Whether to print debug information
        
    Returns:
        A formatted string listing detected dependencies
    """
    logging.debug("Finding common dependencies")
    try:
        # Log file_manifest structure
        logging.debug(f"File manifest type: {type(file_manifest)}")
        if file_manifest:
            sample_key = next(iter(file_manifest))
            sample_value = file_manifest[sample_key]
            logging.debug(f"Sample key type: {type(sample_key)}, value: {sample_key}")
            logging.debug(f"Sample value type: {type(sample_value)}")
            if hasattr(sample_value, '__dict__'):
                logging.debug(f"Sample value attributes: {dir(sample_value)}")
        else:
            logging.debug("File manifest is empty")
            
        # Collect all dependencies
        dependencies = defaultdict(int)
        package_json_count = 0
        requirements_txt_count = 0
        csproj_count = 0
        packages_config_count = 0
        pom_xml_count = 0
        gradle_count = 0
        gradle_count = 0
        
        # Look for package.json files (JavaScript/Node.js)
        for path, info in file_manifest.items():
            path_str = str(path)
            # Check if info is a dictionary or an object with is_binary attribute
            is_binary = info.get('is_binary', False) if isinstance(info, dict) else getattr(info, 'is_binary', False)
            if path_str.endswith('package.json') and not is_binary:
                package_json_count += 1
                try:
                    # Get content from either a dictionary or an object
                    content = info.get('content', '') if isinstance(info, dict) else getattr(info, 'content', '')
                    if content:
                        # Check if content is a string
                        if not isinstance(content, str):
                            logging.warning(f"Content for {path} is not a string, but {type(content)}")
                            continue
                            
                        # Check for dependencies section with more flexible pattern matching
                        if '"dependencies"' in content or '"devDependencies"' in content:
                            # Try to parse as JSON first (most reliable)
                            try:
                                package_data = json.loads(content)
                                # Process dependencies
                                if "dependencies" in package_data and isinstance(package_data["dependencies"], dict):
                                    for dep_name, dep_version in package_data["dependencies"].items():
                                        if isinstance(dep_version, str):
                                            dependencies[f"npm:{dep_name}@{dep_version}"] += 1
                                # Process devDependencies
                                if "devDependencies" in package_data and isinstance(package_data["devDependencies"], dict):
                                    for dep_name, dep_version in package_data["devDependencies"].items():
                                        if isinstance(dep_version, str):
                                            dependencies[f"npm-dev:{dep_name}@{dep_version}"] += 1
                            except json.JSONDecodeError:
                                # Fallback to regex if JSON parsing fails
                                # Extract dependencies section
                                deps_match = re.search(r'"dependencies"\s*:\s*{([^}]+)}', content)
                                if deps_match:
                                    deps_str = deps_match.group(1)
                                    # Extract each dependency
                                    for dep_match in re.finditer(r'"([^"]+)"\s*:\s*"([^"]+)"', deps_str):
                                        dep_name = dep_match.group(1)
                                        dep_version = dep_match.group(2)
                                        dependencies[f"npm:{dep_name}@{dep_version}"] += 1
                except Exception as e:
                    logging.warning(f"Error parsing package.json at {path}: {e}")
                    logging.warning(f"Exception type: {type(e)}")
                    logging.warning(f"Exception traceback: {traceback.format_exc()}")
                    if debug:
                        print(f"Error parsing package.json at {path}: {e}")
        
        # Look for requirements.txt files (Python)
        for path, info in file_manifest.items():
            path_str = str(path)
            # Check if info is a dictionary or an object with is_binary attribute
            is_binary = info.get('is_binary', False) if isinstance(info, dict) else getattr(info, 'is_binary', False)
            if path_str.endswith('requirements.txt') and not is_binary:
                requirements_txt_count += 1
                try:
                    # Get content from either a dictionary or an object
                    content = info.get('content', '') if isinstance(info, dict) else getattr(info, 'content', '')
                    if content:
                        # Check if content is a string
                        if not isinstance(content, str):
                            logging.warning(f"Content for {path} is not a string, but {type(content)}")
                            continue
                            
                        # Extract each dependency
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                dependencies[f"python:{line}"] += 1
                except Exception as e:
                    logging.warning(f"Error parsing requirements.txt at {path}: {e}")
                    logging.warning(f"Exception type: {type(e)}")
                    logging.warning(f"Exception traceback: {traceback.format_exc()}")
                    if debug:
                        print(f"Error parsing requirements.txt at {path}: {e}")
        
        # Look for .csproj files (C#)
        for path, info in file_manifest.items():
            path_str = str(path)
            # Check if info is a dictionary or an object with is_binary attribute
            is_binary = info.get('is_binary', False) if isinstance(info, dict) else getattr(info, 'is_binary', False)
            if path_str.endswith('.csproj') and not is_binary:
                csproj_count += 1
                try:
                    # Get content from either a dictionary or an object
                    content = info.get('content', '') if isinstance(info, dict) else getattr(info, 'content', '')
                    if content:
                        # Check if content is a string
                        if not isinstance(content, str):
                            logging.warning(f"Content for {path} is not a string, but {type(content)}")
                            continue
                            
                        # Extract PackageReference elements
                        for match in re.finditer(r'<PackageReference\s+Include="([^"]+)"\s+Version="([^"]+)"', content):
                            package_name = match.group(1)
                            version = match.group(2)
                            dependencies[f"nuget:{package_name}@{version}"] += 1
                except Exception as e:
                    logging.warning(f"Error parsing .csproj at {path}: {e}")
                    logging.warning(f"Exception type: {type(e)}")
                    logging.warning(f"Exception traceback: {traceback.format_exc()}")
                    if debug:
                        print(f"Error parsing .csproj at {path}: {e}")
        
        # Look for packages.config files (C#)
        for path, info in file_manifest.items():
            path_str = str(path)
            # Check if info is a dictionary or an object with is_binary attribute
            is_binary = info.get('is_binary', False) if isinstance(info, dict) else getattr(info, 'is_binary', False)
            if path_str.endswith('packages.config') and not is_binary:
                packages_config_count += 1
                try:
                    # Get content from either a dictionary or an object
                    content = info.get('content', '') if isinstance(info, dict) else getattr(info, 'content', '')
                    if content:
                        # Check if content is a string
                        if not isinstance(content, str):
                            logging.warning(f"Content for {path} is not a string, but {type(content)}")
                            continue
                            
                        # Extract package elements
                        for match in re.finditer(r'<package\s+id="([^"]+)"\s+version="([^"]+)"', content):
                            package_name = match.group(1)
                            version = match.group(2)
                            dependencies[f"nuget:{package_name}@{version}"] += 1
                except Exception as e:
                    logging.warning(f"Error parsing packages.config at {path}: {e}")
                    logging.warning(f"Exception type: {type(e)}")
                    logging.warning(f"Exception traceback: {traceback.format_exc()}")
                    if debug:
                        print(f"Error parsing packages.config at {path}: {e}")
        
        # Look for pom.xml files (Java/Maven)
        for path, info in file_manifest.items():
            path_str = str(path)
            # Check if info is a dictionary or an object with is_binary attribute
            is_binary = info.get('is_binary', False) if isinstance(info, dict) else getattr(info, 'is_binary', False)
            if path_str.endswith('pom.xml') and not is_binary:
                pom_xml_count += 1
                try:
                    # Get content from either a dictionary or an object
                    content = info.get('content', '') if isinstance(info, dict) else getattr(info, 'content', '')
                    if content:
                        # Check if content is a string
                        if not isinstance(content, str):
                            logging.warning(f"Content for {path} is not a string, but {type(content)}")
                            continue
                            
                        # Extract dependency elements
                        for match in re.finditer(r'<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>\s*<version>([^<]+)</version>', content, re.DOTALL):
                            group_id = match.group(1).strip()
                            artifact_id = match.group(2).strip()
                            version = match.group(3).strip()
                            dependencies[f"maven:{group_id}:{artifact_id}@{version}"] += 1
                except Exception as e:
                    logging.warning(f"Error parsing pom.xml at {path}: {e}")
                    logging.warning(f"Exception type: {type(e)}")
                    logging.warning(f"Exception traceback: {traceback.format_exc()}")
                    if debug:
                        print(f"Error parsing pom.xml at {path}: {e}")
        
        # Look for build.gradle files (Java/Gradle)
        for path, info in file_manifest.items():
            path_str = str(path)
            # Check if info is a dictionary or an object with is_binary attribute
            is_binary = info.get('is_binary', False) if isinstance(info, dict) else getattr(info, 'is_binary', False)
            if path_str.endswith('build.gradle') and not is_binary:
                gradle_count += 1
                try:
                    # Get content from either a dictionary or an object
                    content = info.get('content', '') if isinstance(info, dict) else getattr(info, 'content', '')
                    if content:
                        # Check if content is a string
                        if not isinstance(content, str):
                            logging.warning(f"Content for {path} is not a string, but {type(content)}")
                            continue
                            
                        # Extract implementation/compile dependencies
                        for match in re.finditer(r'(implementation|compile)\s+[\'"]([^:\'"]*)(?::([^:\'"]*))?(?::([^\'"]*))?(:[^\'"]*)?[\'"]', content):
                            dep_type = match.group(1)
                            group = match.group(2) if match.group(2) else ""
                            artifact = match.group(3) if match.group(3) else ""
                            version = match.group(4) if match.group(4) else ""
                            if artifact:
                                dependencies[f"gradle:{group}:{artifact}@{version}"] += 1
                except Exception as e:
                    logging.warning(f"Error parsing build.gradle at {path}: {e}")
                    logging.warning(f"Exception type: {type(e)}")
                    logging.warning(f"Exception traceback: {traceback.format_exc()}")
                    if debug:
                        print(f"Error parsing build.gradle at {path}: {e}")
        
        # Look for build.gradle.kts files (Kotlin DSL)
        for path, info in file_manifest.items():
            path_str = str(path)
            # Check if info is a dictionary or an object with is_binary attribute
            is_binary = info.get('is_binary', False) if isinstance(info, dict) else getattr(info, 'is_binary', False)
            if path_str.endswith('build.gradle.kts') and not is_binary:
                gradle_count += 1
                try:
                    # Get content from either a dictionary or an object
                    content = info.get('content', '') if isinstance(info, dict) else getattr(info, 'content', '')
                    if content:
                        # Check if content is a string
                        if not isinstance(content, str):
                            logging.warning(f"Content for {path} is not a string, but {type(content)}")
                            continue
                            
                        # Extract implementation/compile dependencies
                        for match in re.finditer(r'(implementation|compile)\([\'"](.*?)[\'"]', content):
                            dep_type = match.group(1)
                            dep_string = match.group(2)
                            dependencies[f"gradle-kts:{dep_string}"] += 1
                except Exception as e:
                    logging.warning(f"Error parsing build.gradle.kts at {path}: {e}")
                    logging.warning(f"Exception type: {type(e)}")
                    logging.warning(f"Exception traceback: {traceback.format_exc()}")
                    if debug:
                        print(f"Error parsing build.gradle.kts at {path}: {e}")
        
        logging.debug(f"Processed {package_json_count} package.json, {requirements_txt_count} requirements.txt, "
                     f"{csproj_count} .csproj, {packages_config_count} packages.config, "
                     f"{pom_xml_count} pom.xml, and {gradle_count} build.gradle files")
        
        # Format the dependencies as a string
        if dependencies:
            result = "Detected dependencies:\n"
            for dep, count in sorted(dependencies.items(), key=lambda x: (-x[1], x[0])):
                result += f"- {dep}\n"
            logging.debug(f"Found {len(dependencies)} unique dependencies")
            return result
        else:
            logging.debug("No dependencies detected")
            return "No dependencies detected."
    except Exception as e:
        logging.error(f"Error finding common dependencies: {e}")
        if debug:
            print(f"Error finding common dependencies: {e}")
        return "Error finding common dependencies"

def identify_key_components(file_manifest: Dict[str, Dict], debug: bool = False, max_components: int = DEFAULT_MAX_COMPONENTS) -> str:
    """
    Identify key components from file manifest.
    
    Args:
        file_manifest: Dictionary mapping file paths to file information
        debug: Whether to print debug information
        max_components: Maximum number of key components to display
        
    Returns:
        A formatted string listing key components
    """
    logging.debug("Identifying key components")
    try:
        # Group files by directory
        directories = defaultdict(list)
        for path in file_manifest.keys():
            directory = str(Path(path).parent)
            if directory == '.':
                directory = 'root'
            directories[directory].append(path)
        
        # Sort directories by number of files
        sorted_dirs = sorted(directories.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Format the key components as a string
        result = "Key components:\n"
        for directory, files in sorted_dirs[:max_components]:
            result += f"- {directory} ({len(files)} files)\n"
        
        logging.debug(f"Identified {min(len(sorted_dirs), max_components)} key components out of {len(sorted_dirs)} directories")
        return result
    except Exception as e:
        logging.error(f"Error identifying key components: {e}")
        if debug:
            print(f"Error identifying key components: {e}")
        return "Error identifying key components"

def get_default_order(core_files: Dict[str, Dict], resource_files: Dict[str, Dict]) -> List[str]:
    """
    Get a sensible default order when LLM ordering fails.
    
    Args:
        core_files: Dictionary of core files to order
        resource_files: Dictionary of resource files to append at the end
        
    Returns:
        A list of file paths in a sensible order
    """
    logging.debug("Generating default file order")
    # Start with configuration files
    config_files = [p for p in core_files if p.endswith(('.json', '.config', '.settings'))]
    # Then other core files
    other_files = [p for p in core_files if p not in config_files]
    # End with resource files
    resource_list = list(resource_files.keys())
    
    logging.debug(f"Default order: {len(config_files)} config files, {len(other_files)} other files, {len(resource_list)} resource files")
    return config_files + other_files + resource_list

def fix_markdown_issues(content: str) -> str:
    """
    Fix common markdown formatting issues before returning content.
    
    Args:
        content: The markdown content to fix
        
    Returns:
        The fixed markdown content
    """
    logging.debug("Fixing markdown formatting issues")
    if not content:
        return content
        
    lines = content.split('\n')
    fixed_lines = []
    
    # Track header levels to ensure proper hierarchy
    header_levels: List[int] = []
    in_list = False
    list_indent_level = 0
    fixes_applied = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Fix headers without space after #
        if re.match(r'^#+[^#\s]', line):
            line = re.sub(r'^(#+)([^#\s])', r'\1 \2', line)
            if line != original_line:
                fixes_applied += 1
        
        # Track header levels for hierarchy
        header_match = re.match(r'^(#+)\s', line)
        if header_match:
            level = len(header_match.group(1))
            
            # Ensure there's a blank line before headers (except at the start)
            if i > 0 and fixed_lines and fixed_lines[-1].strip():
                fixed_lines.append('')
                fixes_applied += 1
            
            # Ensure there's a blank line after headers
            if i < len(lines) - 1 and lines[i+1].strip():
                line = line + '\n'
                fixes_applied += 1
            
            # Check header hierarchy
            if not header_levels or level <= header_levels[-1]:
                header_levels.append(level)
            elif level > header_levels[-1] + 1:
                # Header level jumped too much, adjust it
                old_level = level
                level = header_levels[-1] + 1
                line = '#' * level + line[header_match.end(1):]
                header_levels.append(level)
                logging.debug(f"Fixed header hierarchy: H{old_level} -> H{level}")
                fixes_applied += 1
            else:
                header_levels.append(level)
        
        # Fix list indentation
        list_match = re.match(r'^(\s*)[-*+]\s', line)
        if list_match:
            indent = len(list_match.group(1))
            if not in_list:
                in_list = True
                list_indent_level = indent
            elif indent > list_indent_level and indent - list_indent_level != 2:
                # Adjust indentation to be multiple of 2
                old_indent = indent
                new_indent = list_indent_level + 2
                line = ' ' * new_indent + line.lstrip()
                logging.debug(f"Fixed list indentation: {old_indent} -> {new_indent}")
                fixes_applied += 1
        elif line.strip() and in_list:
            in_list = False
        
        fixed_lines.append(line)
    
    result = '\n'.join(fixed_lines)
    logging.debug(f"Applied {fixes_applied} markdown fixes")
    return result

def prepare_file_order_data(project_files: Dict[str, Dict], debug: bool = False,
                           vendor_patterns: List[str] = DEFAULT_VENDOR_PATTERNS) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Prepare data for file order optimization.
    
    Args:
        project_files: Dictionary mapping file paths to file information
        debug: Whether to print debug information
        vendor_patterns: List of regex patterns to identify vendor/resource files
        
    Returns:
        A tuple containing (core_files, resource_files, files_info)
    """
    logging.debug("Preparing file order data")
    
    # Filter out vendor/resource files
    core_files: Dict[str, Dict] = {}
    resource_files: Dict[str, Dict] = {}
    
    if debug:
        print("Filtering files...")
    
    for path, info in project_files.items():
        is_vendor = any(re.search(pattern, path, re.IGNORECASE) for pattern in vendor_patterns)
        if is_vendor:
            resource_files[path] = info
        else:
            core_files[path] = info
    
    if debug:
        print(f"Filtered {len(core_files)} core files and {len(resource_files)} resource files")
    logging.info(f"File filtering: {len(core_files)} core files and {len(resource_files)} resource files")
    
    # Build simplified file info for core files only
    if debug:
        print("Building file information...")
    files_info: Dict[str, Dict] = {}
    
    for path, info in core_files.items():
        # Get attributes with safe defaults
        deps = getattr(info, 'dependencies', None) or []
        exports = getattr(info, 'exports', None) or []
        
        files_info[path] = {
            "type": info.file_type if hasattr(info, 'file_type') and info.file_type else Path(path).suffix,
            "size": info.size if hasattr(info, 'size') else 0,
            "is_binary": info.is_binary if hasattr(info, 'is_binary') else False,
            "dependencies": list(deps),
            "exports": list(exports)
        }
    
    logging.debug(f"Prepared file order data with {len(files_info)} files")
    return core_files, resource_files, files_info

def process_file_order_response(content: str, core_files: Dict[str, Dict], resource_files: Dict[str, Dict], debug: bool = False) -> List[str]:
    """
    Process LLM response to extract file order.
    
    Args:
        content: The LLM response content to process
        core_files: Dictionary of core files to order
        resource_files: Dictionary of resource files to append at the end
        debug: Whether to print debug information
        
    Returns:
        A list of file paths in the extracted order
    """
    logging.debug("Processing file order response")
    try:
        # Try to parse as JSON first
        result = json.loads(content)
        if isinstance(result, dict) and "file_order" in result:
            file_order = result["file_order"]
            # Validate paths exist in original files
            valid_paths = [path for path in file_order if path in core_files]
            
            if valid_paths:
                if debug and "reasoning" in result:
                    reasoning = result["reasoning"] if "reasoning" in result else ""
                    print(f"Ordering reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else reasoning)
                
                # Append resource files at the end
                full_order = valid_paths + list(resource_files.keys())
                logging.debug(f"Successfully extracted file order from JSON response with {len(valid_paths)} valid paths")
                return full_order
            else:
                logging.warning("JSON response contained file_order but no valid paths")
    except json.JSONDecodeError as e:
        logging.debug(f"JSON parsing failed: {e}")
    except Exception as e:
        logging.error(f"Error processing file order response: {e}")
    
    # If JSON parsing fails, try to extract file paths from text
    try:
        lines = content.split('\n')
        file_paths = []
        for line in lines:
            # Look for file paths in the response
            for path in core_files.keys():
                if path in line:
                    file_paths.append(path)
                    break
        
        if file_paths:
            # Append resource files at the end
            full_order = file_paths + list(resource_files.keys())
            logging.debug(f"Extracted {len(file_paths)} file paths from text response")
            return full_order
        else:
            logging.warning("Could not extract any file paths from text response")
    except Exception as e:
        logging.error(f"Error extracting file paths from text: {e}")
    
    # If all else fails, use default order
    logging.info("Using default file order")
    return get_default_order(core_files, resource_files)