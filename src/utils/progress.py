import sys
import time
import atexit
from typing import Optional, Dict, Generator, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
import asyncio

class TaskStatus(Enum):
    """Represents the status of a task in the generation process."""
    PENDING = "⏳"
    RUNNING = "🔄"
    SUCCESS = "✅"
    ERROR = "❌"
    SKIPPED = "⏭️"

@dataclass
class Task:
    """Represents a task in the generation process.
    
    Attributes:
        name: The name of the task
        status: Current status of the task
        message: Optional message associated with the task
        start_time: When the task started (timestamp)
        end_time: When the task ended (timestamp)
        progress_bar: tqdm progress bar instance for this task
    """
    name: str
    status: TaskStatus = TaskStatus.PENDING
    message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress_bar: Optional[tqdm] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def close_progress_bar(self) -> None:
        """Safely close the progress bar if it exists."""
        if self.progress_bar is not None:
            try:
                self.progress_bar.close()
            except:
                pass  # Ignore errors when closing progress bar
            self.progress_bar = None

class ProgressTracker:
    """Tracks and displays progress of the generation process.
    
    This class provides a centralized way to track and display progress of various
    operations in the codebase-scribe-ai project. It manages tasks and their progress,
    with methods for starting, updating, completing, and skipping tasks.
    
    Example usage:
        ```python
        # Initialize the tracker
        tracker = ProgressTracker(repo_path)
        
        # Start a task
        tracker.start_task("Processing files", total=10)
        
        # Update task progress
        tracker.update_task("Processing files", advance=1, message="File 1/10")
        
        # Complete a task
        tracker.complete_task("Processing files", success=True, message="All files processed")
        
        # Use as a context manager
        with tracker.task("Analyzing code") as task:
            # Do work here
            # You can update the task directly
            task.message = "Processing item 1"
        
        # Create a simple progress bar
        with tracker.progress_bar(total=100, desc="Processing") as pbar:
            for i in range(100):
                # Do work
                pbar.update(1)
        
        # Display summary
        tracker.summary()
        
        # Clean up resources when done
        tracker.cleanup()
        ```
    """
    
    # Singleton instance for global access
    _instance = None
    
    @classmethod
    def get_instance(cls, repo_path: Optional[Path] = None) -> 'ProgressTracker':
        """Get or create the singleton instance of ProgressTracker.
        
        Args:
            repo_path: Path to the repository (required only for first call)
            
        Returns:
            The singleton ProgressTracker instance
        """
        if cls._instance is None:
            if repo_path is None:
                raise ValueError("repo_path is required when creating the first instance")
            cls._instance = cls(repo_path)
        return cls._instance
    
    def __init__(self, repo_path: Path):
        """Initialize the progress tracker.
        
        Args:
            repo_path: Path to the repository being processed
        """
        self.repo_path = repo_path
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[Task] = None
        self._active_progress_bars: List[tqdm] = []
        
        # Set up logging
        self.log_file = repo_path / '.readme_cache' / 'generation.log'
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Create a logger with a unique name to avoid conflicts
        logger_name = f'readme_generator_{id(self)}'
        self.logger = logging.getLogger(logger_name)
        
        # Only set up handlers if they don't exist
        if not self.logger.handlers:
            # Create file handler with INFO level
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Create console handler that respects the root logger's level
            # This ensures it follows the --quiet flag setting
            console_handler = logging.StreamHandler(sys.stdout)
            # Get the root logger's level for console output
            root_console_level = logging.WARNING
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.StreamHandler):
                    root_console_level = handler.level
                    break
            
            console_handler.setLevel(root_console_level)
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # Set the logger's level to DEBUG to allow all messages to be processed
            # The handlers will filter based on their individual levels
            self.logger.setLevel(logging.DEBUG)
        
        # Main progress bar for overall progress
        self.main_pbar = tqdm(
            desc="Overall Progress",
            unit="task",
            position=0,
            leave=True
        )
        self._active_progress_bars.append(self.main_pbar)
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def start_task(self, name: str, total: Optional[int] = None) -> Task:
        """Start a new task with optional subtasks.
        
        Args:
            name: Name of the task
            total: Total number of steps in the task (default: 1)
            
        Returns:
            The created Task object
        """
        progress_bar = None
        try:
            task = Task(name=name, status=TaskStatus.RUNNING)
            task.start_time = time.time()
            
            # Create progress bar for this task
            progress_bar = tqdm(
                desc=name,
                total=total or 1,
                unit="step",
                position=len(self.tasks) + 1,
                leave=False
            )
            self._active_progress_bars.append(progress_bar)
            task.progress_bar = progress_bar
            
            self.tasks[name] = task
            self.current_task = task
            self.logger.info(f"Started task: {name}")
            return task
        except Exception as e:
            # Clean up progress bar if creation failed
            if progress_bar is not None:
                try:
                    progress_bar.close()
                except:
                    pass
                if progress_bar in self._active_progress_bars:
                    self._active_progress_bars.remove(progress_bar)
            
            self.logger.error(f"Error starting task '{name}': {e}")
            raise
    
    def update_task(self, name: str, advance: int = 1, message: Optional[str] = None) -> None:
        """Update task progress.
        
        Args:
            name: Name of the task to update
            advance: Number of steps to advance (default: 1)
            message: Optional message to display in the progress bar
        """
        try:
            if name in self.tasks:
                task = self.tasks[name]
                if task.progress_bar:
                    task.progress_bar.update(advance)
                    if message:
                        task.progress_bar.set_postfix_str(message)
            else:
                self.logger.warning(f"Attempted to update non-existent task: {name}")
        except Exception as e:
            self.logger.error(f"Error updating task '{name}': {e}")
    
    def complete_task(self, name: str, success: bool = True, message: Optional[str] = None) -> None:
        """Complete a task with success or error.
        
        Args:
            name: Name of the task to complete
            success: Whether the task completed successfully (default: True)
            message: Optional message about task completion
        """
        try:
            if name in self.tasks:
                task = self.tasks[name]
                task.end_time = time.time()
                task.status = TaskStatus.SUCCESS if success else TaskStatus.ERROR
                task.message = message
                
                # Update and close task progress bar
                if task.progress_bar:
                    try:
                        task.progress_bar.update(task.progress_bar.total - task.progress_bar.n)
                        if message:
                            task.progress_bar.set_postfix_str(message)
                        task.progress_bar.close()
                        if task.progress_bar in self._active_progress_bars:
                            self._active_progress_bars.remove(task.progress_bar)
                        task.progress_bar = None
                    except Exception as pb_error:
                        self.logger.warning(f"Error closing progress bar for task '{name}': {pb_error}")
                
                # Update main progress bar
                try:
                    self.main_pbar.update(1)
                except Exception as mpb_error:
                    self.logger.warning(f"Error updating main progress bar: {mpb_error}")
                
                log_level = logging.INFO if success else logging.ERROR
                duration = task.duration
                duration_str = f" ({duration:.2f}s)" if duration else ""
                self.logger.log(log_level, f"Completed task: {name}{duration_str} - {message if message else 'Done'}")
            else:
                self.logger.warning(f"Attempted to complete non-existent task: {name}")
        except Exception as e:
            self.logger.error(f"Error completing task '{name}': {e}")
    
    def skip_task(self, name: str, reason: str) -> None:
        """Mark a task as skipped.
        
        Args:
            name: Name of the task to skip
            reason: Reason for skipping the task
        """
        progress_bar = None
        try:
            task = Task(name=name, status=TaskStatus.SKIPPED, message=reason)
            
            # Create and immediately update/close progress bar
            try:
                progress_bar = tqdm(
                    desc=name,
                    total=1,
                    unit="step",
                    position=len(self.tasks) + 1,
                    leave=False
                )
                self._active_progress_bars.append(progress_bar)
                task.progress_bar = progress_bar
                
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Skipped: {reason}")
                progress_bar.close()
                self._active_progress_bars.remove(progress_bar)
                task.progress_bar = None
            except Exception as pb_error:
                self.logger.warning(f"Error with progress bar for skipped task '{name}': {pb_error}")
                if progress_bar in self._active_progress_bars:
                    self._active_progress_bars.remove(progress_bar)
            
            self.tasks[name] = task
            
            # Update main progress bar
            try:
                self.main_pbar.update(1)
            except Exception as mpb_error:
                self.logger.warning(f"Error updating main progress bar: {mpb_error}")
                
            self.logger.info(f"Skipped task: {name} - {reason}")
        except Exception as e:
            self.logger.error(f"Error skipping task '{name}': {e}")
    
    def summary(self) -> None:
        """Display final summary of all tasks."""
        try:
            # Close main progress bar if it's still active
            if hasattr(self, 'main_pbar') and self.main_pbar in self._active_progress_bars:
                try:
                    self.main_pbar.close()
                    self._active_progress_bars.remove(self.main_pbar)
                except:
                    pass
            
            total_time = 0
            successful = 0
            failed = 0
            skipped = 0
            
            for task in self.tasks.values():
                if task.duration:
                    total_time += task.duration
                if task.status == TaskStatus.SUCCESS:
                    successful += 1
                elif task.status == TaskStatus.ERROR:
                    failed += 1
                elif task.status == TaskStatus.SKIPPED:
                    skipped += 1
            
            self.logger.info("\nGeneration Summary:")
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Tasks: {len(self.tasks)} total, {successful} successful, {failed} failed, {skipped} skipped")
            
            if failed > 0:
                self.logger.error("\nFailed tasks:")
                for task in self.tasks.values():
                    if task.status == TaskStatus.ERROR:
                        self.logger.error(f"❌ {task.name}: {task.message}")
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources used by the progress tracker.
        
        This method ensures all progress bars are properly closed and resources
        are released. It should be called when the tracker is no longer needed.
        """
        try:
            # Close all active progress bars
            for pbar in self._active_progress_bars[:]:
                try:
                    pbar.close()
                except:
                    pass
            self._active_progress_bars.clear()
            
            # Close progress bars in tasks
            for task in self.tasks.values():
                task.close_progress_bar()
                
            # Remove the atexit handler
            try:
                atexit.unregister(self.cleanup)
            except:
                pass
                
            self.logger.info("Progress tracker resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    @contextmanager
    def task(self, description: str) -> Generator[Task, None, None]:
        """Context manager for tracking task progress.
        
        This provides a convenient way to track a task using a context manager.
        The task is automatically started when entering the context and completed
        when exiting.
        
        Args:
            description: Name/description of the task
            
        Yields:
            The Task object for the current task
            
        Example:
            ```python
            with tracker.task("Processing data") as task:
                # Do work here
                # You can update task properties directly
                task.message = "Processing item 1"
            ```
        """
        task_name = description
        task = None
        try:
            # Start the task
            task = self.start_task(task_name)
            yield task
            # Complete the task successfully if no exception
            self.complete_task(task_name, success=True)
        except Exception as e:
            # Complete the task with error if exception occurred
            if task_name in self.tasks:
                self.complete_task(task_name, success=False, message=str(e))
            raise
    
    @contextmanager
    def progress_bar(self, **kwargs) -> Generator[tqdm, None, None]:
        """Create a simple progress bar without task tracking.
        
        This is a lightweight alternative to full task tracking when you just need
        a progress bar. The progress bar is automatically closed when the context exits.
        
        Args:
            **kwargs: Arguments to pass to tqdm constructor
            
        Yields:
            A tqdm progress bar instance
            
        Example:
            ```python
            with tracker.progress_bar(total=100, desc="Processing") as pbar:
                for i in range(100):
                    # Do work
                    pbar.update(1)
            ```
        """
        pbar = None
        try:
            pbar = tqdm(**kwargs)
            self._active_progress_bars.append(pbar)
            yield pbar
        finally:
            if pbar is not None:
                try:
                    pbar.close()
                    if pbar in self._active_progress_bars:
                        self._active_progress_bars.remove(pbar)
                except:
                    pass
    
    async def update_progress_async(self, pbar: tqdm, interval: float = 0.1) -> None:
        """Updates progress bar periodically while an async operation is running.
        
        This is useful for showing progress during long-running async operations
        where you can't update the progress bar directly.
        
        Args:
            pbar: The progress bar to update
            interval: How often to update the progress bar in seconds
            
        Example:
            ```python
            with tracker.progress_bar(total=100, desc="Processing") as pbar:
                update_task = asyncio.create_task(tracker.update_progress_async(pbar))
                try:
                    # Do async work
                    await some_long_running_task()
                    # Update progress at the end
                    pbar.update(100)
                finally:
                    # Cancel the update task
                    update_task.cancel()
            ```
        """
        try:
            while True:
                await asyncio.sleep(interval)
                pbar.update(0)  # Update display without incrementing counter
        except asyncio.CancelledError:
            pass