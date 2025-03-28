from pathlib import Path
from typing import Dict, List, Optional, Union
from src.models.file_info import FileInfo
import logging
from collections import defaultdict

# Badge style constants
BADGE_STYLE = "for-the-badge"  # Options: flat, flat-square, plastic, for-the-badge, social

def generate_badges(file_manifest: Dict[str, FileInfo], repo_path: Path,
                   badge_style: str = BADGE_STYLE) -> str:
    """Generate appropriate badges based on repository content.
    
    This function analyzes the repository content to generate markdown badges for:
    - License type
    - CI/CD tools
    - Testing frameworks
    - Documentation
    - Docker usage
    - Programming languages and frameworks
    - Databases
    
    Args:
        file_manifest: Dictionary of files in the repository
        repo_path: Path to the repository
        badge_style: Style of badges to generate (default: for-the-badge)
        
    Returns:
        str: Space-separated string of markdown badges
        
    Raises:
        ValueError: If file_manifest is None or empty
    """
    # Validate input parameters
    if not file_manifest:
        logging.warning("Empty file manifest provided to generate_badges")
        return ""
        
    badges = []
    
    # Badge format: ![Label](https://img.shields.io/badge/Label-Message-Color?style=Style&logo=LogoName&logoColor=LogoColor)
    
    # Check for license file to add license badge
    license_files = [f for f in file_manifest.keys() if 'license' in str(f).lower()]
    if license_files:
        try:
            license_file = license_files[0]
            license_path = repo_path / license_file
            
            if not license_path.exists():
                logging.warning(f"License file path does not exist: {license_path}")
                badges.append(f"![License](https://img.shields.io/badge/License-Custom-blue.svg?style={badge_style})")
            else:
                license_content = license_path.read_text(encoding='utf-8', errors='ignore')
                
                # Detect license type with more specific patterns
                license_type = "Custom"
                license_color = "blue"
                
                # Dictionary mapping license identifiers to their display names and colors
                license_types = {
                    "MIT": {"name": "MIT", "color": "yellow"},
                    "Apache License": {"name": "Apache_2.0", "color": "blue"},
                    "GNU GENERAL PUBLIC LICENSE": {"name": "GPLv3", "color": "blue"},
                    "BSD": {"name": "BSD", "color": "blue"},
                    "Mozilla Public License": {"name": "MPL_2.0", "color": "orange"},
                    "ISC": {"name": "ISC", "color": "green"},
                    "Unlicense": {"name": "Unlicense", "color": "gray"}
                }
                
                # Check for each license type
                for identifier, info in license_types.items():
                    if identifier in license_content:
                        license_type = info["name"]
                        license_color = info["color"]
                        break
                
                badges.append(f"![License: {license_type}](https://img.shields.io/badge/License-{license_type}-{license_color}.svg?style={badge_style})")
        except Exception as e:
            logging.warning(f"Could not read license file: {e}")
            # If we can't read the license file, just add a generic license badge
            badges.append(f"![License](https://img.shields.io/badge/License-Custom-blue.svg?style={badge_style})")
    else:
        # Add a custom license badge even when no license file is found
        badges.append(f"![License](https://img.shields.io/badge/License-Custom-blue.svg?style={badge_style})")
    
    # Check for CI/CD configuration - more efficient with a single pass through file_manifest
    ci_cd_systems = {
        '.github/workflows': {"name": "GitHub_Actions", "color": "2088FF", "logo": "github-actions"},
        'azure-pipelines': {"name": "Azure_Pipelines", "color": "2560E0", "logo": "azure-pipelines"},
        '.travis.yml': {"name": "Travis", "color": "B22222", "logo": "travis"},
        'Jenkinsfile': {"name": "Jenkins", "color": "D24939", "logo": "jenkins"},
        'gitlab-ci.yml': {"name": "GitLab", "color": "FC6D26", "logo": "gitlab"},
        'circleci': {"name": "CircleCI", "color": "343434", "logo": "circleci"},
        '.github/actions': {"name": "GitHub_Actions", "color": "2088FF", "logo": "github-actions"},
        'bitbucket-pipelines.yml': {"name": "Bitbucket_Pipelines", "color": "0052CC", "logo": "bitbucket"}
    }
    
    # Find the first matching CI/CD system
    ci_cd_found = False
    for file_path in file_manifest.keys():
        file_path_str = str(file_path)
        for ci_pattern, ci_info in ci_cd_systems.items():
            if ci_pattern.lower() in file_path_str.lower():
                badges.append(
                    f"![CI/CD](https://img.shields.io/badge/CI%2FCD-{ci_info['name']}-{ci_info['color']}?style={badge_style}&logo={ci_info['logo']}&logoColor=white)"
                )
                ci_cd_found = True
                break
        if ci_cd_found:
            break
    
    # Check for testing frameworks - more efficient with a dictionary approach
    test_frameworks = {
        'junit': {"name": "JUnit", "color": "25A162", "logo": "junit5"},
        'nunit': {"name": "NUnit", "color": "008C45", "logo": "dotnet"},
        'jest': {"name": "Jest", "color": "C21325", "logo": "jest"},
        'mocha': {"name": "Mocha", "color": "8D6748", "logo": "mocha"},
        'pytest': {"name": "Pytest", "color": "0A9EDC", "logo": "pytest"},
        'rspec': {"name": "RSpec", "color": "CC342D", "logo": "ruby"},
        'cypress': {"name": "Cypress", "color": "17202C", "logo": "cypress"},
        'playwright': {"name": "Playwright", "color": "2EAD33", "logo": "playwright"},
        'testng': {"name": "TestNG", "color": "007396", "logo": "java"},
        'phpunit': {"name": "PHPUnit", "color": "777BB4", "logo": "php"},
        'jasmine': {"name": "Jasmine", "color": "8A4182", "logo": "jasmine"},
        'selenium': {"name": "Selenium", "color": "43B02A", "logo": "selenium"},
        'vitest': {"name": "Vitest", "color": "6E9F18", "logo": "vitest"}
    }
    
    # Special case detection for language-specific test directories
    language_test_patterns = [
        ('.java', 'JUnit', '25A162', 'junit5'),
        ('.cs', 'NUnit', '008C45', 'dotnet'),
        ('.py', 'Pytest', '0A9EDC', 'pytest'),
        ('.js', 'Jest', 'C21325', 'jest'),
        ('.rb', 'RSpec', 'CC342D', 'ruby'),
        ('.php', 'PHPUnit', '777BB4', 'php')
    ]
    
    # Find testing frameworks
    test_framework_found = False
    
    # First check for explicit framework files
    for file_path in file_manifest.keys():
        file_lower = str(file_path).lower()
        for framework, info in test_frameworks.items():
            if framework in file_lower:
                badges.append(
                    f"![Tests](https://img.shields.io/badge/Tests-{info['name']}-{info['color']}?style={badge_style}&logo={info['logo']}&logoColor=white)"
                )
                test_framework_found = True
                break
        if test_framework_found:
            break
    
    # If no explicit framework found, check for test directories with language extensions
    if not test_framework_found:
        for file_path in file_manifest.keys():
            file_lower = str(file_path).lower()
            if 'test' in file_lower:
                for ext, name, color, logo in language_test_patterns:
                    if ext in file_lower:
                        badges.append(
                            f"![Tests](https://img.shields.io/badge/Tests-{name}-{color}?style={badge_style}&logo={logo}&logoColor=white)"
                        )
                        test_framework_found = True
                        break
            if test_framework_found:
                break
    
    # Check for documentation - more comprehensive detection
    doc_patterns = ['docs/', 'documentation/', 'wiki/', 'readme.md', 'contributing.md', 'architecture.md']
    has_docs = False
    
    for file_path in file_manifest.keys():
        file_lower = str(file_path).lower()
        if any(pattern in file_lower for pattern in doc_patterns):
            has_docs = True
            break
    
    if has_docs:
        badges.append(f"![Documentation](https://img.shields.io/badge/Documentation-Yes-brightgreen?style={badge_style})")
    
    # Check for Docker - more comprehensive detection
    docker_patterns = ['dockerfile', 'docker-compose', '.dockerignore', 'docker/']
    has_docker = False
    
    for file_path in file_manifest.keys():
        file_lower = str(file_path).lower()
        if any(pattern in file_lower for pattern in docker_patterns):
            has_docker = True
            break
    
    if has_docker:
        badges.append(f"![Docker](https://img.shields.io/badge/Docker-2496ED?style={badge_style}&logo=docker&logoColor=white)")
    
    # Check for common languages/frameworks - more efficient with a dictionary approach
    languages = set()
    
    # Define language detection patterns and their badge info
    language_patterns = {
        # Languages
        '.java': {"name": "Java", "color": "ED8B00", "logo": "openjdk"},
        '.cs': {"name": "C%23", "color": "239120", "logo": "c-sharp"},
        '.js': {"name": "JavaScript", "color": "F7DF1E", "logo": "javascript", "logoColor": "black"},
        '.ts': {"name": "TypeScript", "color": "3178C6", "logo": "typescript"},
        '.py': {"name": "Python", "color": "3776AB", "logo": "python"},
        '.rb': {"name": "Ruby", "color": "CC342D", "logo": "ruby"},
        '.php': {"name": "PHP", "color": "777BB4", "logo": "php"},
        '.go': {"name": "Go", "color": "00ADD8", "logo": "go"},
        '.rs': {"name": "Rust", "color": "000000", "logo": "rust"},
        '.swift': {"name": "Swift", "color": "F05138", "logo": "swift"},
        '.kt': {"name": "Kotlin", "color": "7F52FF", "logo": "kotlin"},
        '.dart': {"name": "Dart", "color": "0175C2", "logo": "dart"}
    }
    
    # Define framework detection patterns and their badge info
    framework_patterns = {
        # Java frameworks
        'spring-boot': {"name": "Spring_Boot", "color": "6DB33F", "logo": "spring-boot", "requires": ".java"},
        'spring': {"name": "Spring", "color": "6DB33F", "logo": "spring", "requires": ".java"},
        
        # .NET frameworks
        'asp.net': {"name": "ASP.NET", "color": "5C2D91", "logo": ".net", "requires": ".cs"},
        'aspnet': {"name": "ASP.NET", "color": "5C2D91", "logo": ".net", "requires": ".cs"},
        '.net': {"name": ".NET", "color": "5C2D91", "logo": ".net", "requires": ".cs"},
        
        # JavaScript frameworks
        'react': {"name": "React", "color": "20232A", "logo": "react", "logoColor": "61DAFB"},
        'next.js': {"name": "Next.js", "color": "000000", "logo": "next.js", "requires": "react"},
        'nextjs': {"name": "Next.js", "color": "000000", "logo": "next.js", "requires": "react"},
        'gatsby': {"name": "Gatsby", "color": "663399", "logo": "gatsby", "requires": "react"},
        'angular': {"name": "Angular", "color": "DD0031", "logo": "angular"},
        'vue': {"name": "Vue.js", "color": "4FC08D", "logo": "vue.js"},
        
        # Python frameworks
        'django': {"name": "Django", "color": "092E20", "logo": "django", "requires": ".py"},
        'flask': {"name": "Flask", "color": "000000", "logo": "flask", "requires": ".py"},
        'fastapi': {"name": "FastAPI", "color": "009688", "logo": "fastapi", "requires": ".py"},
        
        # Ruby frameworks
        'rails': {"name": "Rails", "color": "CC0000", "logo": "ruby-on-rails", "requires": ".rb"},
        
        # PHP frameworks
        'laravel': {"name": "Laravel", "color": "FF2D20", "logo": "laravel", "requires": ".php"},
        
        # Databases
        'mongodb': {"name": "MongoDB", "color": "4EA94B", "logo": "mongodb"},
        'mysql': {"name": "MySQL", "color": "4479A1", "logo": "mysql"},
        'postgresql': {"name": "PostgreSQL", "color": "316192", "logo": "postgresql"},
        'postgres': {"name": "PostgreSQL", "color": "316192", "logo": "postgresql"},
        'redis': {"name": "Redis", "color": "DC382D", "logo": "redis"}
    }
    
    # First pass: detect languages and create a set of found languages
    found_languages = set()
    for file_path in file_manifest.keys():
        file_lower = str(file_path).lower()
        for pattern, info in language_patterns.items():
            if pattern in file_lower:
                found_languages.add(pattern)
                logo_color = f"&logoColor={info.get('logoColor', 'white')}"
                languages.add(
                    f"![{info['name']}](https://img.shields.io/badge/{info['name']}-{info['color']}?style={badge_style}&logo={info['logo']}{logo_color})"
                )
    
    # Second pass: detect frameworks, but only if their required language is found
    for file_path in file_manifest.keys():
        file_lower = str(file_path).lower()
        for pattern, info in framework_patterns.items():
            # Skip if this framework requires a language that wasn't found
            if "requires" in info and info["requires"] not in found_languages:
                continue
                
            if pattern in file_lower:
                logo_color = f"&logoColor={info.get('logoColor', 'white')}"
                languages.add(
                    f"![{info['name']}](https://img.shields.io/badge/{info['name']}-{info['color']}?style={badge_style}&logo={info['logo']}{logo_color})"
                )
    
    # Add all detected languages to badges
    badges.extend(languages)
    
    # Return space-separated badges
    return " ".join(badges) 