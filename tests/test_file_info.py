import unittest
from pathlib import Path
from src.models.file_info import FileInfo

class TestFileInfo(unittest.TestCase):
    """Test cases for the FileInfo class."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Basic initialization with just path
        file_info = FileInfo(path="test.py")
        self.assertEqual(str(file_info.path), "test.py")
        self.assertFalse(file_info.is_binary)
        self.assertEqual(file_info.size, 0)
        self.assertEqual(file_info.language, "")
        self.assertEqual(file_info.content, "")
        self.assertIsNone(file_info.summary)
        self.assertEqual(file_info.file_type, "")
        self.assertEqual(file_info.last_modified, 0.0)
        self.assertEqual(file_info.imports, set())
        self.assertEqual(file_info.exports, set())
        self.assertFalse(file_info.from_cache)
        
        # Initialization with Path object
        path_obj = Path("test.py")
        file_info = FileInfo(path=path_obj)
        self.assertEqual(file_info.path, path_obj)
        
        # Initialization with additional parameters
        file_info = FileInfo(
            path="test.py",
            is_binary=True,
            size=100,
            language="python",
            content="print('hello')",
            summary="A test file",
            file_type="python",
            last_modified=123456.0,
            imports=["os", "sys"],
            exports=["main"],
            from_cache=True
        )
        self.assertTrue(file_info.is_binary)
        self.assertEqual(file_info.size, 100)
        self.assertEqual(file_info.language, "python")
        self.assertEqual(file_info.content, "print('hello')")
        self.assertEqual(file_info.summary, "A test file")
        self.assertEqual(file_info.file_type, "python")
        self.assertEqual(file_info.last_modified, 123456.0)
        self.assertEqual(file_info.imports, ["os", "sys"])
        self.assertEqual(file_info.exports, ["main"])
        self.assertTrue(file_info.from_cache)
    
    def test_post_init(self):
        """Test the __post_init__ method."""
        # Test path conversion from string to Path
        file_info = FileInfo(path="test.py")
        self.assertIsInstance(file_info.path, Path)
        
        # Test empty collections initialization
        file_info = FileInfo(path="test.py")
        self.assertEqual(file_info.imports, set())
        self.assertEqual(file_info.exports, set())
        
        # Test validation of required fields
        with self.assertRaises(ValueError):
            FileInfo(path="")
    
    def test_repr(self):
        """Test the __repr__ method."""
        file_info = FileInfo(path="test.py", file_type="python", is_binary=False)
        repr_str = repr(file_info)
        self.assertIn("test.py", repr_str)
        self.assertIn("python", repr_str)
        self.assertIn("False", repr_str)
    
    def test_is_language(self):
        """Test the is_language method."""
        # Test with matching language
        file_info = FileInfo(path="test.py", language="python")
        self.assertTrue(file_info.is_language("python"))
        self.assertTrue(file_info.is_language("PYTHON"))  # Case insensitive
        
        # Test with non-matching language
        self.assertFalse(file_info.is_language("java"))
        
        # Test with empty language
        file_info = FileInfo(path="test.py")
        self.assertFalse(file_info.is_language("python"))
    
    def test_has_extension(self):
        """Test the has_extension method."""
        # Test with matching extension
        file_info = FileInfo(path="test.py")
        self.assertTrue(file_info.has_extension("py"))
        self.assertTrue(file_info.has_extension(".py"))  # With leading dot
        self.assertTrue(file_info.has_extension(".PY"))  # Case insensitive
        
        # Test with non-matching extension
        self.assertFalse(file_info.has_extension("js"))
        self.assertFalse(file_info.has_extension(".js"))
    
    def test_to_dict(self):
        """Test the to_dict method."""
        # Create a FileInfo object with various types of data
        file_info = FileInfo(
            path="test.py",
            language="python",
            imports=set(["os", "sys"]),
            exports=set(["main"])
        )
        
        # Convert to dictionary
        result = file_info.to_dict()
        
        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that path is converted to string
        self.assertEqual(result["path"], "test.py")
        
        # Check that sets are converted to lists
        self.assertIsInstance(result["imports"], list)
        self.assertIsInstance(result["exports"], list)
        self.assertIn("os", result["imports"])
        self.assertIn("sys", result["imports"])
        self.assertIn("main", result["exports"])

if __name__ == "__main__":
    unittest.main()