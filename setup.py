from setuptools import setup, find_packages

setup(
    name="codebase-scribe-ai",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "python-magic-bin; platform_system == 'Windows'",
        "python-magic; platform_system != 'Windows'",
        "networkx",
        "gitignore_parser",
    ]
) 