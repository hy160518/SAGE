import time
import functools
from typing import Callable, Any, Type, Tuple
import logging

class APIRetryHandler:
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)):

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on_exceptions = retry_on_exceptions

        self.logger = logging.getLogger(__name__)

    def calculate_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    def retry(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(self.max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    if attempt > 0:
                        self.logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")

                    return result

                except self.retry_on_exceptions as e:
                    last_exception = e

                    if attempt < self.max_retries:
                        delay = self.calculate_delay(attempt)

                        self.logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}: {str(e)}, "
                            f"retrying in {delay:.2f}s"
                        )

                        time.sleep(delay)
                    else:
                        self.logger.error(
                            f"{func.__name__} failed after {self.max_retries + 1} attempts"
                        )

            raise last_exception

        return wrapper

    def __call__(self, func: Callable) -> Callable:
        return self.retry(func)

def retry_on_api_error(max_retries: int = 3,
                       base_delay: float = 1.0,
                       max_delay: float = 60.0) -> Callable:

    handler = APIRetryHandler(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay
    )

    return handler.retry

class SmartRetryHandler(APIRetryHandler):
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def should_retry(self, exception: Exception) -> bool:
        exception_str = str(exception).lower()

        network_keywords = ['timeout', 'connection', 'network', 'socket']
        if any(keyword in exception_str for keyword in network_keywords):
            return True

        if 'rate limit' in exception_str or '429' in exception_str:
            return True

        if any(code in exception_str for code in ['500', '502', '503', '504']):
            return True

        return False

    def retry_smart(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(self.max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    if attempt > 0:
                        self.logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")

                    return result

                except Exception as e:
                    last_exception = e

                    if not self.should_retry(e):
                        self.logger.error(
                            f"{func.__name__} encountered non-retryable error: {str(e)}"
                        )
                        raise e

                    if attempt < self.max_retries:
                        delay = self.calculate_delay(attempt)

                        self.logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}: {str(e)}, "
                            f"retrying in {delay:.2f}s"
                        )

                        time.sleep(delay)
                    else:
                        self.logger.error(
                            f"{func.__name__} failed after {self.max_retries + 1} attempts"
                        )

            raise last_exception

        return wrapper

default_retry_handler = APIRetryHandler(max_retries=3, base_delay=2.0)
smart_retry_handler = SmartRetryHandler(max_retries=3, base_delay=2.0)

retry_on_error = default_retry_handler.retry
smart_retry = smart_retry_handler.retry_smart
