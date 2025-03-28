"""Tests for path compression utilities."""

import unittest
from src.utils.path_compression import (
    compress_paths,
    decompress_paths,
    _identify_common_prefixes,
    get_compression_explanation
)


class TestPathCompression(unittest.TestCase):
    """Test cases for path compression utilities."""

    def test_identify_common_prefixes(self):
        """Test identification of common prefixes."""
        file_paths = [
            "src/main/java/com/example/project/Controller.java",
            "src/main/java/com/example/project/Service.java",
            "src/main/java/com/example/project/Repository.java",
            "src/test/java/com/example/project/ControllerTest.java",
            "src/main/resources/application.properties",
            "README.md"
        ]
        
        prefixes = _identify_common_prefixes(file_paths)
        # Only prefixes that appear multiple times are included
        self.assertIn("src/main/java/com/example/project/", prefixes)
        
        # These won't be included because they only appear once
        self.assertNotIn("src/test/java/com/example/project/", prefixes)
        self.assertNotIn("src/main/resources/", prefixes)
        
    def test_compress_paths(self):
        """Test path compression."""
        file_paths = [
            "src/main/java/com/example/project/Controller.java",
            "src/main/java/com/example/project/Service.java",
            "src/main/java/com/example/project/Repository.java",
            "src/test/java/com/example/project/ControllerTest.java",
            "src/main/resources/application.properties",
            "README.md"
        ]
        
        compressed_paths, decompression_map = compress_paths(file_paths)
        
        # Check that we have compression keys
        self.assertTrue(any(path.startswith("@") for path in compressed_paths))
        
        # Check that README.md is unchanged
        self.assertIn("README.md", compressed_paths)
        
        # Check decompression map
        self.assertTrue(len(decompression_map) > 0)
        
    def test_decompress_paths(self):
        """Test path decompression."""
        original_paths = [
            "src/main/java/com/example/project/Controller.java",
            "src/main/java/com/example/project/Service.java",
            "src/test/java/com/example/project/ControllerTest.java",
            "README.md"
        ]
        
        compressed_paths, decompression_map = compress_paths(original_paths)
        decompressed_paths = decompress_paths(compressed_paths, decompression_map)
        
        # Check that decompression restores original paths
        self.assertEqual(set(original_paths), set(decompressed_paths))
        
    def test_get_compression_explanation(self):
        """Test generation of compression explanation."""
        decompression_map = {
            "src/main/java/com/example/project/": "@1",
            "src/test/java/com/example/project/": "@2"
        }
        
        explanation = get_compression_explanation(decompression_map)
        
        # Check that explanation contains all keys
        for key, value in decompression_map.items():
            self.assertIn(key, explanation)
            self.assertIn(value, explanation)


if __name__ == "__main__":
    unittest.main()