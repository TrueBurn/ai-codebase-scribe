import asyncio
from functools import wraps
import logging
import random
from typing import TypeVar, Callable, Optional, Any
import time

T = TypeVar('T')

class RetryException(Exception):
    """Custom exception for retry-related errors."""
    pass

def async_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async functions to implement retry logic with exponential backoff.
    
    Args:
        retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add randomized jitter to the delay
        exceptions: Tuple of exceptions to catch and retry
        logger: Optional logger instance for logging retry attempts
    
    Returns:
        A decorator function that wraps the original function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Log successful retry if it wasn't the first attempt
                    if attempt > 0 and logger:
                        logger.debug(
                            f"Successfully executed {func.__name__} after {attempt} retries"
                        )
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == retries:
                        if logger:
                            logger.error(f"Final retry attempt failed for {func.__name__}: {str(e)}")
                        raise RetryException(f"Max retries ({retries}) exceeded") from e
                    
                    if logger:
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{retries} for {func.__name__} "
                            f"failed: {str(e)}. Retrying in {current_delay:.1f}s..."
                        )
                    
                    # Apply jitter if enabled (adds or subtracts up to 20% of the delay)
                    actual_delay = current_delay
                    if jitter:
                        actual_delay = current_delay * (1 + random.uniform(-0.2, 0.2))
                    
                    await asyncio.sleep(actual_delay)
                    
                    # Calculate next delay with backoff, but cap at max_delay
                    current_delay = min(current_delay * backoff, max_delay)
            
            raise last_exception
        return wrapper
    return decorator

def sync_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Synchronous version of the retry decorator with exponential backoff.
    
    This decorator can be used for synchronous functions that need retry capability,
    such as file operations, database queries, or synchronous API calls.
    
    Args:
        retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add randomized jitter to the delay
        exceptions: Tuple of exceptions to catch and retry
        logger: Optional logger instance for logging retry attempts
    
    Returns:
        A decorator function that wraps the original function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful retry if it wasn't the first attempt
                    if attempt > 0 and logger:
                        logger.debug(
                            f"Successfully executed {func.__name__} after {attempt} retries"
                        )
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == retries:
                        if logger:
                            logger.error(f"Final retry attempt failed for {func.__name__}: {str(e)}")
                        raise RetryException(f"Max retries ({retries}) exceeded") from e
                    
                    if logger:
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{retries} for {func.__name__} "
                            f"failed: {str(e)}. Retrying in {current_delay:.1f}s..."
                        )
                    
                    # Apply jitter if enabled (adds or subtracts up to 20% of the delay)
                    actual_delay = current_delay
                    if jitter:
                        actual_delay = current_delay * (1 + random.uniform(-0.2, 0.2))
                    
                    time.sleep(actual_delay)
                    
                    # Calculate next delay with backoff, but cap at max_delay
                    current_delay = min(current_delay * backoff, max_delay)
            
            raise last_exception
        return wrapper
    return decorator