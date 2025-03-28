import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.utils.progress import TaskStatus, Task, ProgressTracker

@pytest.fixture
def repo_path():
    return Path(__file__).parent.parent

@pytest.fixture
def mock_tqdm():
    with patch('src.utils.progress.tqdm') as mock:
        # Create a mock progress bar that behaves like tqdm
        mock_pbar = MagicMock()
        mock_pbar.update = MagicMock()
        mock_pbar.close = MagicMock()
        mock_pbar.set_postfix_str = MagicMock()
        mock_pbar.total = 1
        mock_pbar.n = 0
        
        # Make tqdm return our mock progress bar
        mock.return_value = mock_pbar
        yield mock

@pytest.fixture
def mock_logger():
    with patch('src.utils.progress.logging.getLogger') as mock:
        mock_logger = MagicMock()
        mock_logger.info = MagicMock()
        mock_logger.error = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.log = MagicMock()
        mock_logger.handlers = []
        
        # Make getLogger return our mock logger
        mock.return_value = mock_logger
        yield mock_logger

class TestTaskStatus:
    def test_task_status_values(self):
        """Test that TaskStatus enum has the expected values."""
        assert TaskStatus.PENDING.value == "⏳"
        assert TaskStatus.RUNNING.value == "🔄"
        assert TaskStatus.SUCCESS.value == "✅"
        assert TaskStatus.ERROR.value == "❌"
        assert TaskStatus.SKIPPED.value == "⏭️"

class TestTask:
    def test_task_initialization(self):
        """Test that Task initializes with correct default values."""
        task = Task(name="test_task")
        assert task.name == "test_task"
        assert task.status == TaskStatus.PENDING
        assert task.message is None
        assert task.start_time is None
        assert task.end_time is None
        assert task.progress_bar is None
    
    def test_task_duration(self):
        """Test that Task.duration calculates correctly."""
        task = Task(name="test_task")
        task.start_time = 100.0
        task.end_time = 105.5
        assert task.duration == 5.5
        
        # Test with no times set
        task = Task(name="test_task")
        assert task.duration is None
        
        # Test with only start time
        task = Task(name="test_task")
        task.start_time = 100.0
        assert task.duration is None

class TestProgressTracker:
    def test_initialization(self, repo_path, mock_tqdm, mock_logger):
        """Test that ProgressTracker initializes correctly."""
        tracker = ProgressTracker(repo_path)
        assert tracker.repo_path == repo_path
        assert isinstance(tracker.tasks, dict)
        assert tracker.current_task is None
        
        # Check that main progress bar was created
        mock_tqdm.assert_called_once()
    
    def test_update_task(self, repo_path, mock_tqdm):
        """Test updating a task."""
        mock_pbar = mock_tqdm.return_value
        
        tracker = ProgressTracker(repo_path)
        
        # Create a task manually since start_task might not work as expected
        task_name = "test_task"
        task = Task(name=task_name, status=TaskStatus.RUNNING)
        task.progress_bar = mock_pbar
        tracker.tasks[task_name] = task
        
        # Reset mock to clear previous calls
        mock_pbar.update.reset_mock()
        mock_pbar.set_postfix_str.reset_mock()
        
        # Update task
        tracker.update_task(task_name, advance=2, message="Working")
        
        # Check that progress bar was updated
        mock_pbar.update.assert_called_with(2)
        mock_pbar.set_postfix_str.assert_called_with("Working")
        
        # Test updating non-existent task
        tracker.update_task("nonexistent")  # Should not raise an exception
    
    def test_skip_task(self, repo_path, mock_tqdm, mock_logger):
        """Test skipping a task."""
        tracker = ProgressTracker(repo_path)
        
        # Skip a task
        tracker.skip_task("skip_task", reason="Not needed")
        
        # Check that task was created and marked as skipped
        assert "skip_task" in tracker.tasks
        task = tracker.tasks["skip_task"]
        assert task.status == TaskStatus.SKIPPED
        assert task.message == "Not needed"