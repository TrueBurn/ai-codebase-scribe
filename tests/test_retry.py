import pytest
import asyncio
import logging
import time
from unittest.mock import MagicMock, patch
from src.utils.retry import async_retry, sync_retry, RetryException


class TestRetry:
    """Test suite for retry decorators."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock logger
        self.logger = MagicMock(spec=logging.Logger)
        
        # Reset counters for test functions
        self.async_call_count = 0
        self.sync_call_count = 0

    @pytest.mark.asyncio
    async def test_async_retry_success_first_attempt(self):
        """Test async_retry when function succeeds on first attempt."""
        
        @async_retry(retries=3, logger=self.logger)
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
        self.logger.warning.assert_not_called()
        self.logger.error.assert_not_called()
        self.logger.debug.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_retry_success_after_retries(self):
        """Test async_retry when function succeeds after a few retries."""
        
        @async_retry(retries=3, delay=0.01, logger=self.logger)
        async def test_func():
            self.async_call_count += 1
            if self.async_call_count < 3:
                raise ValueError("Temporary error")
            return "success after retry"
        
        result = await test_func()
        assert result == "success after retry"
        assert self.async_call_count == 3
        assert self.logger.warning.call_count == 2
        assert self.logger.error.call_count == 0
        # Debug logging is optional, so we don't assert on it

    @pytest.mark.asyncio
    async def test_async_retry_max_retries_exceeded(self):
        """Test async_retry when max retries are exceeded."""
        
        @async_retry(retries=2, delay=0.01, logger=self.logger)
        async def test_func():
            self.async_call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(RetryException) as excinfo:
            await test_func()
        
        assert "Max retries (2) exceeded" in str(excinfo.value)
        assert self.async_call_count == 3  # Initial + 2 retries
        assert self.logger.warning.call_count == 2
        assert self.logger.error.call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_with_specific_exceptions(self):
        """Test async_retry with specific exception types."""
        
        @async_retry(retries=2, delay=0.01, exceptions=(ValueError,), logger=self.logger)
        async def test_func(error_type):
            self.async_call_count += 1
            if error_type == "value":
                raise ValueError("Value error")
            else:
                raise TypeError("Type error")
        
        # Should retry on ValueError
        self.async_call_count = 0
        with pytest.raises(RetryException):
            await test_func("value")
        assert self.async_call_count == 3  # Initial + 2 retries
        
        # Should not retry on TypeError
        self.async_call_count = 0
        with pytest.raises(TypeError):
            await test_func("type")
        assert self.async_call_count == 1  # No retries

    def test_sync_retry_success_first_attempt(self):
        """Test sync_retry when function succeeds on first attempt."""
        
        @sync_retry(retries=3, logger=self.logger)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
        self.logger.warning.assert_not_called()
        self.logger.error.assert_not_called()
        self.logger.debug.assert_not_called()

    def test_sync_retry_success_after_retries(self):
        """Test sync_retry when function succeeds after a few retries."""
        
        @sync_retry(retries=3, delay=0.01, logger=self.logger)
        def test_func():
            self.sync_call_count += 1
            if self.sync_call_count < 3:
                raise ValueError("Temporary error")
            return "success after retry"
        
        result = test_func()
        assert result == "success after retry"
        assert self.sync_call_count == 3
        assert self.logger.warning.call_count == 2
        assert self.logger.error.call_count == 0
        # Debug logging is optional, so we don't assert on it

    def test_sync_retry_max_retries_exceeded(self):
        """Test sync_retry when max retries are exceeded."""
        
        @sync_retry(retries=2, delay=0.01, logger=self.logger)
        def test_func():
            self.sync_call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(RetryException) as excinfo:
            test_func()
        
        assert "Max retries (2) exceeded" in str(excinfo.value)
        assert self.sync_call_count == 3  # Initial + 2 retries
        assert self.logger.warning.call_count == 2
        assert self.logger.error.call_count == 1

    def test_sync_retry_with_specific_exceptions(self):
        """Test sync_retry with specific exception types."""
        
        @sync_retry(retries=2, delay=0.01, exceptions=(ValueError,), logger=self.logger)
        def test_func(error_type):
            self.sync_call_count += 1
            if error_type == "value":
                raise ValueError("Value error")
            else:
                raise TypeError("Type error")
        
        # Should retry on ValueError
        self.sync_call_count = 0
        with pytest.raises(RetryException):
            test_func("value")
        assert self.sync_call_count == 3  # Initial + 2 retries
        
        # Should not retry on TypeError
        self.sync_call_count = 0
        with pytest.raises(TypeError):
            test_func("type")
        assert self.sync_call_count == 1  # No retries