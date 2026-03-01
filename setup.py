from setuptools import setup, find_packages

setup(
    name="codebase-scribe-ai",
    packages=find_packages(),
    version="0.2.0",
    install_requires=[
        "python-magic-bin; platform_system == 'Windows'",
        "python-magic; platform_system != 'Windows'",
        "networkx",
        "gitignore_parser",
        "rich>=13.0.0",
        "colorama>=0.4.6",
    ]
) 