import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module
from src.clients.message_manager import MessageManager


class TestMessageManager(unittest.TestCase):
    """Test cases for the MessageManager class."""
    
    def test_version_constant(self):
        """Test that the VERSION constant exists and is a string."""
        self.assertTrue(hasattr(MessageManager, 'VERSION'))
        self.assertIsInstance(MessageManager.VERSION, str)
        self.assertTrue(len(MessageManager.VERSION) > 0)
    
    def test_create_system_user_messages(self):
        """Test the create_system_user_messages method."""
        # Test with valid inputs
        system_content = "You are a helpful assistant."
        user_content = "Explain Python decorators."
        
        messages = MessageManager.create_system_user_messages(system_content, user_content)
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], system_content)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], user_content)
        
        # Test with empty system content
        with self.assertRaises(ValueError):
            MessageManager.create_system_user_messages("", user_content)
        
        # Test with empty user content
        with self.assertRaises(ValueError):
            MessageManager.create_system_user_messages(system_content, "")
        
        # Test with non-string inputs
        with self.assertRaises(ValueError):
            MessageManager.create_system_user_messages(123, user_content)
        
        with self.assertRaises(ValueError):
            MessageManager.create_system_user_messages(system_content, 123)
    
    def test_get_project_overview_messages(self):
        """Test the get_project_overview_messages method."""
        # Test with valid inputs
        project_structure = "src/\n  main.py\n  utils.py"
        tech_report = "Python 3.9\nRequirements: requests, numpy"
        template_content = "Generate a README for this project"
        
        messages = MessageManager.get_project_overview_messages(
            project_structure, tech_report, template_content
        )
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(project_structure, messages[0]["content"])
        self.assertIn(tech_report, messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], template_content)
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            MessageManager.get_project_overview_messages("", tech_report, template_content)
        
        with self.assertRaises(ValueError):
            MessageManager.get_project_overview_messages(project_structure, "", template_content)
        
        with self.assertRaises(ValueError):
            MessageManager.get_project_overview_messages(project_structure, tech_report, "")
    
    def test_get_file_summary_messages(self):
        """Test the get_file_summary_messages method."""
        # Test with valid input
        file_content = "def hello():\n    print('Hello world')"
        
        messages = MessageManager.get_file_summary_messages(file_content)
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("code documentation expert", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], file_content)
        
        # Test with invalid input
        with self.assertRaises(ValueError):
            MessageManager.get_file_summary_messages("")
    
    def test_check_and_truncate_messages_under_limit(self):
        """Test check_and_truncate_messages when messages are under the token limit."""
        # Create mock token counter
        mock_token_counter = MagicMock()
        mock_token_counter.will_exceed_limit.return_value = (False, 100)
        
        # Create test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"}
        ]
        
        # Call the method
        result = MessageManager.check_and_truncate_messages(messages, mock_token_counter, "test-model")
        
        # Verify the result
        self.assertEqual(result, messages)
        mock_token_counter.will_exceed_limit.assert_called_once_with(messages, "test-model")
    
    def test_check_and_truncate_messages_intelligent_reduction(self):
        """Test check_and_truncate_messages with intelligent reduction."""
        # Create mock token counter
        mock_token_counter = MagicMock()
        # First check shows we're over the limit
        mock_token_counter.will_exceed_limit.side_effect = [
            (True, 1000),  # First call returns over limit
            (False, 800)   # Second call (after reduction) returns under limit
        ]
        mock_token_counter.handle_oversized_input.return_value = "Reduced content"
        
        # Create test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Very long content that exceeds the token limit."}
        ]
        
        # Call the method
        result = MessageManager.check_and_truncate_messages(messages, mock_token_counter, "test-model")
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], messages[0])  # System message unchanged
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], "Reduced content")
        
        # Verify the token counter methods were called correctly
        mock_token_counter.handle_oversized_input.assert_called_once_with(
            "Very long content that exceeds the token limit.", target_percentage=0.8
        )
    
    def test_check_and_truncate_messages_hard_truncation(self):
        """Test check_and_truncate_messages with hard truncation."""
        # Create mock token counter
        mock_token_counter = MagicMock()
        # Both checks show we're over the limit
        mock_token_counter.will_exceed_limit.side_effect = [
            (True, 1000),  # First call returns over limit
            (True, 900)    # Second call (after intelligent reduction) still over limit
        ]
        mock_token_counter.handle_oversized_input.return_value = "Intelligently reduced but still too long"
        mock_token_counter.count_message_tokens.return_value = 50  # System message tokens
        mock_token_counter.get_token_limit.return_value = 500  # Model token limit
        mock_token_counter.truncate_text.return_value = "Hard truncated content"
        
        # Create test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Very long content that exceeds the token limit."}
        ]
        
        # Call the method
        result = MessageManager.check_and_truncate_messages(messages, mock_token_counter, "test-model")
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], messages[0])  # System message unchanged
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], "Hard truncated content")
        
        # Verify the token counter methods were called correctly
        mock_token_counter.handle_oversized_input.assert_called_once_with(
            "Very long content that exceeds the token limit.", target_percentage=0.8
        )
        mock_token_counter.count_message_tokens.assert_called_once()
        mock_token_counter.get_token_limit.assert_called_once_with("test-model")
        
        # Calculate the expected effective limit (90% of model limit minus system tokens)
        expected_effective_limit = int(500 * 0.9) - 50
        mock_token_counter.truncate_text.assert_called_once_with(
            "Intelligently reduced but still too long", expected_effective_limit
        )
    
    def test_check_and_truncate_messages_invalid_inputs(self):
        """Test check_and_truncate_messages with invalid inputs."""
        mock_token_counter = MagicMock()
        
        # Test with non-list messages
        with self.assertRaises(ValueError):
            MessageManager.check_and_truncate_messages("not a list", mock_token_counter, "test-model")
        
        # Test with list containing invalid message format
        with self.assertRaises(ValueError):
            MessageManager.check_and_truncate_messages(
                [{"not_role": "system", "content": "content"}], 
                mock_token_counter, 
                "test-model"
            )
        
        # Test with invalid token counter
        with self.assertRaises(TypeError):
            MessageManager.check_and_truncate_messages(
                [{"role": "system", "content": "content"}],
                "not a token counter",
                "test-model"
            )


if __name__ == '__main__':
    unittest.main()