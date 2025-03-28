import os
import pytest
import tempfile
import shutil
from pathlib import Path
import networkx as nx

from src.analyzers.codebase import CodebaseAnalyzer
from src.models.file_info import FileInfo
from src.utils.cache import CacheManager

class TestIntegration:
    """Integration tests for the CodebaseAnalyzer with other components."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository with test files."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a simple project structure
            # Python files
            os.makedirs(os.path.join(temp_dir, "src", "main"), exist_ok=True)
            with open(os.path.join(temp_dir, "src", "main", "app.py"), "w") as f:
                f.write("import os\nimport sys\n\nclass App:\n    def run(self):\n        print('Running app')\n")
            
            with open(os.path.join(temp_dir, "src", "main", "utils.py"), "w") as f:
                f.write("def helper():\n    return 'Helper function'\n")
            
            # C# file with namespace
            os.makedirs(os.path.join(temp_dir, "src", "csharp"), exist_ok=True)
            with open(os.path.join(temp_dir, "src", "csharp", "Program.cs"), "w") as f:
                f.write("using System;\n\nnamespace TestProject\n{\n    class Program\n    {\n        static void Main()\n        {\n            Console.WriteLine(\"Hello World\");\n        }\n    }\n}\n")
            
            # Project configuration files
            with open(os.path.join(temp_dir, "setup.py"), "w") as f:
                f.write("from setuptools import setup\n\nsetup(name='test-project', version='0.1.0')\n")
            
            with open(os.path.join(temp_dir, "README.md"), "w") as f:
                f.write("# Test Project\n\n## Overview\n\nThis is a test project.\n")
            
            # Create a .gitignore file
            with open(os.path.join(temp_dir, ".gitignore"), "w") as f:
                f.write("__pycache__/\n*.py[cod]\n*.so\n.env\n")
            yield Path(temp_dir)
        finally:
            # Close any open file handles before removing the directory
            try:
                import gc
                gc.collect()  # Force garbage collection to close any open file handles
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
                
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {e}")
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        from src.utils.config_class import ScribeConfig, BlacklistConfig, CacheConfig
        
        config = ScribeConfig()
        config.debug = True
        config.blacklist = BlacklistConfig(
            extensions=['.pyc', '.pyo', '.pyd'],
            path_patterns=[r'__pycache__', r'\.git']
        )
        config.cache = CacheConfig(
            ttl=3600,
            max_size=1048576
        )
        return config
    
    def test_analyzer_with_cache(self, temp_repo, config):
        """Test that the analyzer works with the cache manager."""
        # First run - should populate cache
        analyzer = CodebaseAnalyzer(temp_repo, config)
        manifest = analyzer.analyze_repository()
        
        # Verify basic results
        assert isinstance(manifest, dict)
        assert len(manifest) > 0
        assert any(str(path).endswith('app.py') for path in manifest.keys())
        assert any(str(path).endswith('README.md') for path in manifest.keys())
        
        # Check that cache was used
        assert analyzer.cache.enabled
        
        # Second run - should use cache
        analyzer2 = CodebaseAnalyzer(temp_repo, config)
        manifest2 = analyzer2.analyze_repository()
        
        # Results should be the same
        assert len(manifest) == len(manifest2)
        assert set(manifest.keys()) == set(manifest2.keys())
    
    def test_analyzer_with_dependency_graph(self, temp_repo, config):
        """Test that the analyzer correctly builds a dependency graph."""
        analyzer = CodebaseAnalyzer(temp_repo, config)
        analyzer.analyze_repository()
        
        # Get the dependency graph
        graph = analyzer.build_dependency_graph()
        
        # Verify it's a valid graph
        assert isinstance(graph, nx.DiGraph)
        
        # Check for expected dependencies
        app_py_path = str(Path("src") / "main" / "app.py")
        if app_py_path in graph.nodes():
            # app.py imports os and sys
            edges = list(graph.edges(app_py_path))
            assert any('os' in edge[1] for edge in edges)
            assert any('sys' in edge[1] for edge in edges)
    
    def test_project_name_detection(self, temp_repo, config):
        """Test that the analyzer correctly detects the project name."""
        analyzer = CodebaseAnalyzer(temp_repo, config)
        analyzer.analyze_repository()
        
        # Should detect from setup.py
        project_name = analyzer.derive_project_name()
        assert project_name == "test-project"
        
        # If we remove setup.py, it should detect from C# namespace
        setup_py = temp_repo / "setup.py"
        if setup_py.exists():
            os.remove(setup_py)
            
        analyzer = CodebaseAnalyzer(temp_repo, config)
        analyzer.analyze_repository()
        project_name = analyzer.derive_project_name()
        assert project_name == "TestProject"
    
    def test_error_handling(self, temp_repo, config):
        """Test error handling in the analyzer."""
        # Create a file to test with
        file_path = temp_repo / "README.md"
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write("# Test README")
        
        # Test with a file path instead of a directory
        # Should raise ValueError for file path during initialization
        with pytest.raises(ValueError, match="Repository path is not a directory"):
            analyzer = CodebaseAnalyzer(file_path, config)
            
        # Test with a non-existent path by creating a temporary path and then deleting it
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        non_existent_path = Path(temp_dir)
        shutil.rmtree(temp_dir)  # Delete the directory to make it non-existent
        
        # Should raise ValueError for non-existent path during initialization
        with pytest.raises(ValueError, match="Repository path does not exist"):
            analyzer = CodebaseAnalyzer(non_existent_path, config)