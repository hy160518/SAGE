"""
API重试机制
用于处理API调用失败的情况，实现指数退避重试
"""
import time
import functools
from typing import Callable, Any, Type, Tuple
import logging


class APIRetryHandler:
    """
    API重试处理器
    
    实现功能：
    1. 指数退避重试（Exponential Backoff）
    2. 可配置的最大重试次数
    3. 特定异常类型的重试
    4. 重试日志记录
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)):
        """
        初始化重试处理器
        
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟（秒）
            max_delay: 最大延迟（秒）
            exponential_base: 指数基数
            retry_on_exceptions: 需要重试的异常类型元组
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on_exceptions = retry_on_exceptions
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        计算重试延迟时间（指数退避）
        
        Args:
            attempt: 当前重试次数（从0开始）
        
        Returns:
            延迟时间（秒）
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def retry(self, func: Callable) -> Callable:
        """
        装饰器：为函数添加重试逻辑
        
        Args:
            func: 要包装的函数
        
        Returns:
            包装后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    # 尝试执行函数
                    result = func(*args, **kwargs)
                    
                    # 成功，返回结果
                    if attempt > 0:
                        self.logger.info(f"{func.__name__} 在第 {attempt + 1} 次尝试成功")
                    
                    return result
                
                except self.retry_on_exceptions as e:
                    last_exception = e
                    
                    # 如果还有重试机会
                    if attempt < self.max_retries:
                        delay = self.calculate_delay(attempt)
                        
                        self.logger.warning(
                            f"{func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}, "
                            f"将在 {delay:.2f}秒 后重试"
                        )
                        
                        time.sleep(delay)
                    else:
                        # 已达到最大重试次数
                        self.logger.error(
                            f"{func.__name__} 在 {self.max_retries + 1} 次尝试后仍然失败"
                        )
            
            # 所有重试都失败，抛出最后一个异常
            raise last_exception
        
        return wrapper
    
    def __call__(self, func: Callable) -> Callable:
        """
        使实例可以作为装饰器使用
        """
        return self.retry(func)


def retry_on_api_error(max_retries: int = 3, 
                       base_delay: float = 1.0,
                       max_delay: float = 60.0) -> Callable:
    """
    便捷的装饰器函数：为API调用添加重试
    
    使用示例：
    @retry_on_api_error(max_retries=3, base_delay=2.0)
    def call_api():
        response = api.call()
        return response
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟
        max_delay: 最大延迟
    
    Returns:
        装饰器函数
    """
    handler = APIRetryHandler(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay
    )
    
    return handler.retry


class SmartRetryHandler(APIRetryHandler):
    """
    智能重试处理器
    根据错误类型和API响应码智能决定是否重试
    """
    
    # HTTP状态码是否应该重试的映射
    RETRYABLE_STATUS_CODES = {
        429,  # Too Many Requests (速率限制)
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }
    
    def should_retry(self, exception: Exception) -> bool:
        """
        判断异常是否应该重试
        
        Args:
            exception: 捕获的异常
        
        Returns:
            是否应该重试
        """
        # 检查是否是网络错误
        exception_str = str(exception).lower()
        
        # 网络相关错误应该重试
        network_keywords = ['timeout', 'connection', 'network', 'socket']
        if any(keyword in exception_str for keyword in network_keywords):
            return True
        
        # 速率限制错误应该重试
        if 'rate limit' in exception_str or '429' in exception_str:
            return True
        
        # 服务器错误应该重试
        if any(code in exception_str for code in ['500', '502', '503', '504']):
            return True
        
        # 其他错误不重试（如参数错误、权限错误等）
        return False
    
    def retry_smart(self, func: Callable) -> Callable:
        """
        智能重试装饰器
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        self.logger.info(f"{func.__name__} 在第 {attempt + 1} 次尝试成功")
                    
                    return result
                
                except Exception as e:
                    last_exception = e
                    
                    # 判断是否应该重试
                    if not self.should_retry(e):
                        self.logger.error(
                            f"{func.__name__} 发生不可重试的错误: {str(e)}"
                        )
                        raise e
                    
                    # 如果还有重试机会
                    if attempt < self.max_retries:
                        delay = self.calculate_delay(attempt)
                        
                        self.logger.warning(
                            f"{func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}, "
                            f"将在 {delay:.2f}秒 后重试"
                        )
                        
                        time.sleep(delay)
                    else:
                        self.logger.error(
                            f"{func.__name__} 在 {self.max_retries + 1} 次尝试后仍然失败"
                        )
            
            raise last_exception
        
        return wrapper


# 全局重试处理器实例
default_retry_handler = APIRetryHandler(max_retries=3, base_delay=2.0)
smart_retry_handler = SmartRetryHandler(max_retries=3, base_delay=2.0)

# 导出便捷装饰器
retry_on_error = default_retry_handler.retry
smart_retry = smart_retry_handler.retry_smart
